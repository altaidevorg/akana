//! Morphological generator for Turkish words given root/lemma and inflectional tags.
//!
//! Generation runs on the analyzer's own morphotactic graph, as Zemberek's WordGenerator
//! does: from the item's stems it walks the same transitions with the same templates and
//! conditions, and keeps a path that renders the tags asked for. The word it returns is one the
//! analyzer reads back with those tags.

use super::analyzer::{render, tag_of, Step, TurkishMorphology, ENGINE, MAX_STATE_REPEAT, PRIOR};
use super::dictionary::DictionaryItem;
use super::morphotactics::{Context, Morpheme as M, PathView};
use super::phonetics::{attributes_of, PhoneticAttrs as P};
use super::pos::{PrimaryPos, RootAttr, SecondaryPos};
use super::stems::{self, StemTransition};

/// Tags a request may spell out that the analyzer leaves unshown after a suffix.
const DEFAULT_TAGS: [&str; 3] = ["A3sg", "Pnon", "Nom"];
/// Search steps one generation may take before it gives up.
const MAX_WORK: usize = 200_000;
/// Candidate words collected before they are checked against the analyzer.
const MAX_CANDIDATES: usize = 8;

/// Generates surface forms; a thin handle on [`TurkishMorphology::generate`] that shares the
/// built-in lexicon, so creating one is cheap.
pub struct TurkishGenerator {
    morphology: TurkishMorphology,
}

impl Default for TurkishGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl TurkishGenerator {
    pub fn new() -> Self {
        Self {
            morphology: TurkishMorphology::new(),
        }
    }

    /// A generator over an analyzer's lexicon, with the items it has added.
    pub fn with_morphology(morphology: TurkishMorphology) -> Self {
        Self { morphology }
    }

    /// Generates the surface form for a lemma and a sequence of morpheme tags.
    /// Example: `generate("kitap", &["Noun", "A3sg", "P1sg", "Dat"]) -> "kitabıma"`
    pub fn generate(&self, lemma: &str, tags: &[&str]) -> Option<String> {
        self.morphology.generate(lemma, tags)
    }
}

impl TurkishMorphology {
    /// Generates the word with akana's tags (`["Noun", "P1sg", "Dat"]`; the first tag is the
    /// POS, and the defaults A3sg, Pnon and Nom may be spelled out) from a root or a lemma
    /// (kitap; gel or gelmek). Among homographs of that POS it takes a common word before a
    /// name, then the item the corpus prior finds most frequent (hak the noun: hakkım), then
    /// the shortest word (hal "state": halinde, not hal "solution": hallinde). `None` for an
    /// unknown word, POS or tag, or when no path of the graph renders the tags.
    pub fn generate(&self, word: &str, tags: &[&str]) -> Option<String> {
        let lower = crate::phonology::to_turkish_lower(word.trim());
        let wanted: Vec<&str> = tags
            .iter()
            .copied()
            .filter(|t| !DEFAULT_TAGS.contains(t))
            .collect();
        let pos = wanted.first().and_then(|t| PrimaryPos::from_name(t))?;
        let infinitive_root = lower
            .strip_suffix("mek")
            .or_else(|| lower.strip_suffix("mak"));
        let items = self
            .root_items(&lower)
            .iter()
            .chain(
                infinitive_root
                    .map_or(&[][..], |root| self.root_items(root))
                    .iter()
                    .filter(|i| i.lemma == lower),
            )
            .filter(|i| i.primary_pos == pos);
        let prior = &*PRIOR;
        let mut best: Option<((bool, i32, usize), String)> = None;
        for item in items {
            let mut search = Generation {
                wanted: &wanted,
                surface: String::new(),
                steps: Vec::new(),
                work: 0,
                special: false,
                candidates: Vec::new(),
            };
            for (stem, owner) in stems::item_stems(item, self.lexicon(), &ENGINE.roots) {
                let root_morpheme = ENGINE.graph.states[stem.to.0 as usize].morpheme;
                search.surface = stem.surface.to_string();
                search.special = stem.surface.as_ref() != owner.root;
                let ctx = Context::at_root(stem.to, root_morpheme);
                search.walk(&owner, &stem, ctx, stem.attrs, 1);
            }
            // The analyzer must read the word back with the same item and tags. Of several
            // spellings of the reading, one on the root as spelt comes before one on a special
            // stem (yukarıda, not yukarda), then the shortest is the unmarked one (okuyor, not
            // okumakta; gidin, not gitsenize).
            let mut candidates = search.candidates;
            candidates.sort_by_key(|(s, special)| (*special, s.chars().count()));
            let found = candidates.into_iter().map(|(s, _)| s).find(|surface| {
                self.analyze_unranked(surface).iter().any(|p| {
                    p.lemma == item.lemma
                        && p.root == item.root
                        && p.primary_pos == item.primary_pos
                        && p.morpheme_tags
                            .iter()
                            .map(String::as_str)
                            .filter(|t| !DEFAULT_TAGS.contains(t))
                            .eq(wanted.iter().copied())
                })
            });
            if let Some(surface) = found {
                let key = (
                    item.secondary_pos == SecondaryPos::ProperNoun,
                    -prior.item_log_p(item),
                    surface.chars().count(),
                );
                if best.as_ref().is_none_or(|(b, _)| key < *b) {
                    best = Some((key, surface));
                }
            }
        }
        best.map(|(_, surface)| surface)
    }
}

struct Generation<'a> {
    wanted: &'a [&'a str],
    surface: String,
    steps: Vec<Step>,
    work: usize,
    /// The current stem is not the root as spelt (yukar- of yukarı, kitab- of kitap).
    special: bool,
    /// Words found, with whether they come from a special stem.
    candidates: Vec<(String, bool)>,
}

impl Generation<'_> {
    /// Walks on from the current state; `next` is the index of the first requested tag not yet
    /// produced by a suffix that is always shown.
    fn walk(
        &mut self,
        item: &DictionaryItem,
        stem: &StemTransition,
        ctx: Context,
        attrs: P,
        next: usize,
    ) {
        self.work += 1;
        if self.work > MAX_WORK || self.candidates.len() >= MAX_CANDIDATES {
            return;
        }
        let engine = &*ENGINE;
        let graph = &engine.graph;
        let state = &graph.states[ctx.current.0 as usize];
        if state.terminal && !attrs.contains(P::CANNOT_TERMINATE) {
            let parse = render(
                engine,
                item,
                stem,
                &self.steps,
                &self.surface,
                &self.surface,
            );
            let rendered = parse
                .morpheme_tags
                .iter()
                .map(String::as_str)
                .filter(|t| !DEFAULT_TAGS.contains(t));
            if rendered.eq(self.wanted.iter().copied())
                && !self.candidates.iter().any(|(s, _)| *s == self.surface)
            {
                self.candidates.push((self.surface.clone(), self.special));
            }
        }
        // Silent steps (zero derivations, defaults) come at most a few per requested tag.
        if self.steps.len() >= 3 * self.wanted.len() + 6 {
            return;
        }
        for (id, t) in graph.outgoing(ctx.current) {
            if t.has_surface() && item.attributes.contains(RootAttr::NO_SUFFIX) {
                continue;
            }
            let Some(surface) = t.surface(attrs) else {
                continue;
            };
            let view = PathView {
                ctx: &ctx,
                item,
                stem_surface: &stem.surface,
                attrs,
                tail_empty: false,
            };
            if !t.condition.eval(graph, &view) {
                continue;
            }
            let m = graph.states[t.to.0 as usize].morpheme;
            // A suffix that is always shown must be the next requested tag still to come, and
            // one shown only with a surface must be requested at all.
            let next = match tag_of(m) {
                Some(tag) => match self.wanted[next..].iter().position(|w| *w == tag) {
                    Some(k) => next + k + 1,
                    None => continue,
                },
                None => next,
            };
            if !surface.is_empty()
                && item.lemma != "değil"
                && !shown_as(m).is_empty()
                && !shown_as(m).iter().any(|tag| self.wanted.contains(tag))
            {
                continue;
            }
            let repeats = self
                .steps
                .iter()
                .filter(|s| graph.transitions[s.transition as usize].to == t.to)
                .count();
            if repeats >= MAX_STATE_REPEAT {
                continue;
            }
            let next_attrs = if t.has_surface() {
                let mut a = attributes_of(surface, attrs);
                a -= P::CANNOT_TERMINATE;
                match t.ends_voiced() {
                    Some(false) => a |= P::EXPECTS_CONSONANT,
                    Some(true) => a |= P::EXPECTS_VOWEL | P::CANNOT_TERMINATE,
                    None => {}
                }
                a
            } else {
                attrs
            };
            let next_ctx = ctx.step(graph, t.to, !surface.is_empty());
            let start = self.surface.len();
            self.surface.push_str(surface);
            self.steps.push(Step {
                transition: id,
                start,
                end: self.surface.len(),
            });
            self.walk(item, stem, next_ctx, next_attrs, next);
            self.steps.pop();
            self.surface.truncate(start);
        }
    }
}

/// The tags a person, possessive or copular morpheme with a surface can be shown as (render
/// decides which), for pruning; empty for the others.
fn shown_as(m: M) -> &'static [&'static str] {
    match m {
        M::P1sg => &["P1sg"],
        M::P2sg => &["P2sg"],
        M::P3sg => &["P3sg"],
        M::P1pl => &["P1pl"],
        M::P2pl => &["P2pl"],
        M::P3pl => &["P3pl"],
        M::A1sg => &["A1sg", "CopPres1sg"],
        M::A2sg => &["A2sg", "CopPres2sg"],
        M::A1pl => &["A1pl", "CopPres1pl"],
        M::A2pl => &["A2pl", "CopPres2pl"],
        M::A3pl => &["A3pl", "CopPres3pl"],
        M::Past => &["Past", "CopPast"],
        M::Narr => &["Evid", "CopEvid"],
        M::Cond => &["CopCond"],
        M::Cop => &["Cop", "CopPres3sg"],
        M::Neg => &["Neg"],
        M::Recip => &["Recip"],
        M::Reflex => &["Reflex"],
        _ => &[],
    }
}
