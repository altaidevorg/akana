//! Morphological generator for Turkish words given root/lemma and inflectional tags.
//!
//! Generation runs on the analyzer's own morphotactic graph, as Zemberek's WordGenerator
//! does: from the item's stems it walks the same transitions with the same templates and
//! conditions, and keeps a path that renders the tags asked for. The word it returns is one the
//! analyzer reads back with those tags.

use super::analyzer::{
    render, suffixes_without_apostrophe, tag_of, Step, TurkishMorphology, ENGINE, MAX_STATE_REPEAT,
    PRIOR,
};
use super::dictionary::{normalize_word, DictionaryItem};
use super::morphotactics::{Context, Morpheme as M, PathView};
use super::phonetics::{attributes_of, PhoneticAttrs as P};
use super::pos::{PrimaryPos, RootAttr, SecondaryPos};
use super::stems::{self, StemTransition};
use std::sync::LazyLock;

/// Tags a request may spell out that the analyzer leaves unshown after a suffix.
const DEFAULT_TAGS: [&str; 3] = ["A3sg", "Pnon", "Nom"];
/// The possessives, which a CompoundP3sg noun's plural always shows (cumhurbaşkan-lar-ı).
const POSSESSIVES: [&str; 6] = ["P1sg", "P2sg", "P3sg", "P1pl", "P2pl", "P3pl"];

/// Every tag the analyzer shows, and Prog1 and Prog2, which a request may use to tell the
/// two progressives apart (both are shown as Prog).
static TAGS: LazyLock<Vec<&'static str>> = LazyLock::new(|| {
    let mut tags: Vec<&'static str> = DEFAULT_TAGS.to_vec();
    tags.extend(["Prog1", "Prog2"]);
    for &m in M::ALL {
        tags.extend(tag_of(m));
        tags.extend(shown_as(m));
    }
    tags.sort_unstable();
    tags.dedup();
    tags
});

/// A requested tag as the analyzer spells it, whatever its case (p1sg, DAT); `None` for a tag
/// it never shows.
fn canonical_tag(tag: &str) -> Option<&'static str> {
    TAGS.iter().copied().find(|t| t.eq_ignore_ascii_case(tag))
}
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
    /// Generates the word with akana's tags (`["Noun", "P1sg", "Dat"]`, in any case; the first
    /// tag is the POS, and the defaults A3sg, Pnon and Nom may be spelled out) from a root or a
    /// lemma (kitap; gel or gelmek), spelt as `analyze` reports it or folded as it folds words
    /// (kâğıt or kağıt, e-posta or eposta, Fin-Ugor). The word comes back in the folded
    /// spelling (kağıtta), a name's root as the dictionary spells it (Hakkâri'de).
    ///
    /// - Prog is the -Iyor progressive (gidiyorum, diyor), except before the copula -DIr
    ///   (CopPres3sg), where written Turkish has -mAktA (gelmektedir, not geliyordur); Prog2
    ///   asks for the -mAktA one (bulunmaktadır), and Prog1 for -Iyor only.
    /// - The copula after a verb is shown as CopPres3sg (gelmektedir), CopPast (geliyordu),
    ///   CopEvid and CopCond, and after a nominal as Cop or CopPres3sg, as `analyze` shows it.
    /// - A name or an abbreviation is written as the lexicon spells it, with an apostrophe
    ///   before its case, possessive and copula (Türkiye'de, Ali'nin, ABD'nin, TBMM'de), and
    ///   its suffixes follow its pronunciation.
    /// - A CompoundP3sg noun's plural keeps its own possessive (cumhurbaşkanı with A3pl and Dat
    ///   is cumhurbaşkanlarına).
    ///
    /// Among homographs of that POS it takes a common word before a name, or a name before a
    /// common word when the request is written with a capital (Ali, ABD); then the item the
    /// corpus prior finds most frequent (hak the noun: hakkım), then the shortest word (hal
    /// "state": halinde, not hal "solution": hallinde). `None` for an unknown word, POS or tag,
    /// or when no path of the graph renders the tags.
    pub fn generate(&self, word: &str, tags: &[&str]) -> Option<String> {
        let written = word.trim();
        let lower = normalize_word(written);
        let capitalised = crate::phonology::to_turkish_lower(written) != written;
        let (first, rest) = tags.split_first()?;
        let pos = PrimaryPos::from_name(first)?;
        let mut wanted: Vec<&str> = vec![pos.as_str()];
        let mut progressive = None;
        for tag in rest {
            match canonical_tag(tag)? {
                "Prog1" => {
                    progressive = Some(M::Prog1);
                    wanted.push("Prog");
                }
                "Prog2" => {
                    progressive = Some(M::Prog2);
                    wanted.push("Prog");
                }
                t if DEFAULT_TAGS.contains(&t) => {}
                t => wanted.push(t),
            }
        }
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
        let mut best: Option<((bool, bool, i32, usize), String)> = None;
        for item in items {
            let name = matches!(
                item.secondary_pos,
                SecondaryPos::ProperNoun | SecondaryPos::Abbreviation
            );
            let mut found = self.generate_item(item, &wanted, progressive);
            if found.is_none()
                && item.attributes.contains(RootAttr::COMPOUND_P3SG)
                && wanted.contains(&"A3pl")
                && !wanted.iter().any(|t| POSSESSIVES.contains(t))
            {
                let mut own = wanted.clone();
                let at = own.iter().position(|t| *t == "A3pl").map_or(1, |i| i + 1);
                own.insert(at, "P3sg");
                found = self.generate_item(item, &own, progressive);
            }
            if let Some(surface) = found {
                let key = (
                    capitalised != name,
                    name && item.display_lemma() != written,
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

    /// The word one item makes with the tags, if any path renders them.
    fn generate_item(
        &self,
        item: &DictionaryItem,
        wanted: &[&str],
        progressive: Option<M>,
    ) -> Option<String> {
        let roots = &ENGINE.roots;
        let mut search = Generation {
            wanted,
            progressive,
            surface: String::new(),
            steps: Vec::new(),
            work: 0,
            special: false,
            candidates: Vec::new(),
        };
        for (stem, owner) in stems::item_stems(item, self.lexicon(), roots) {
            let root_morpheme = ENGINE.graph.states[stem.to.0 as usize].morpheme;
            search.surface = stem.surface.to_string();
            // A special root's other stem (içer- of içeri, bur- of bura, di- of de), not a
            // regular alternation (kitab- of kitap, gid- of git).
            search.special = [
                roots.noun_last_vowel_drop,
                roots.adj_last_vowel_drop,
                roots.pron_pers_mod,
                roots.pron_quant_modified,
            ]
            .contains(&stem.to)
                || (stem.to == roots.verb_de_ye && stem.surface.as_ref() != owner.root);
            let ctx = Context::at_root(stem.to, root_morpheme);
            search.walk(&owner, &stem, ctx, stem.attrs, 1);
        }
        // The analyzer must read the word back with the same item, secondary POS (Hak the
        // name is no hak) and tags. Of several spellings of the reading, -Iyor comes before
        // -mAktA (gidiyorum, not gitmekteyim; diyor, not demekte), but -mAktA first before the
        // copula -DIr (gelmektedir: 14,384 times in the court and RagTurk samples, geliyordur
        // once; geliyordu 337 times, gelmekteydi 72); then one on the root as spelt before one
        // on a special stem (yukarıda, not yukarda); then the shortest is the unmarked one
        // (gidin, not gitsenize).
        let mut candidates = search.candidates;
        let prefer_prog2 =
            progressive.is_none() && wanted.windows(2).any(|w| w == ["Prog", "CopPres3sg"]);
        candidates.sort_by_key(|c| (c.prog2 != prefer_prog2, c.special, c.text.chars().count()));
        candidates.into_iter().map(|c| c.text).find(|text| {
            self.analyze_unranked(text).iter().any(|p| {
                p.lemma == item.lemma
                    && p.root == item.root
                    && p.primary_pos == item.primary_pos
                    && p.secondary_pos == item.secondary_pos
                    && p.morpheme_tags
                        .iter()
                        .map(String::as_str)
                        .filter(|t| !DEFAULT_TAGS.contains(t))
                        .eq(wanted.iter().copied())
            })
        })
    }
}

struct Generation<'a> {
    wanted: &'a [&'a str],
    /// The progressive asked for by name (Prog1 or Prog2), if any.
    progressive: Option<M>,
    surface: String,
    steps: Vec<Step>,
    work: usize,
    /// The current stem is a special root's other stem (yukar- of yukarı).
    special: bool,
    candidates: Vec<Candidate>,
}

/// A word found, as written.
struct Candidate {
    text: String,
    /// It comes from a special root's other stem.
    special: bool,
    /// Its progressive is -mAktA.
    prog2: bool,
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
            if rendered.eq(self.wanted.iter().copied()) {
                let text = self.written(item, stem);
                if !self.candidates.iter().any(|c| c.text == text) {
                    let prog2 = self.steps.iter().any(|s| {
                        let to = graph.transitions[s.transition as usize].to;
                        graph.states[to.0 as usize].morpheme == M::Prog2
                    });
                    self.candidates.push(Candidate {
                        text,
                        special: self.special,
                        prog2,
                    });
                }
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
            if matches!(m, M::Prog1 | M::Prog2) && self.progressive.is_some_and(|p| p != m) {
                continue;
            }
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

    /// The word as written: a name or an abbreviation on its root as the lexicon spells it,
    /// with an apostrophe before the suffixes that take one (Ankara'ya, TBMM'de, but
    /// Ankaralı, Türkçenin).
    fn written(&self, item: &DictionaryItem, stem: &StemTransition) -> String {
        let name = matches!(
            item.secondary_pos,
            SecondaryPos::ProperNoun | SecondaryPos::Abbreviation
        );
        if !name || stem.surface.as_ref() != item.root {
            return self.surface.clone();
        }
        let mut text = item.display_lemma().into_owned();
        if !suffixes_without_apostrophe(&ENGINE.graph, item, &self.steps) {
            text.push('\'');
        }
        text.push_str(&self.surface[stem.surface.len()..]);
        text
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
