//! Stems the analyzer looks words up by, generated from the lexicon when it loads (Zemberek's
//! `StemTransitionsBase`).
//!
//! A root whose last letters change before a vowel gets two stems: the plain one, which only a
//! consonant-initial suffix (or nothing) may follow (kitap, kitap-ta), and the changed one,
//! which only a vowel-initial suffix may follow and which cannot end a word (kitab-ı). That
//! makes the alternations obligatory: kitapı and kitabda have no analysis. Irregular stems
//! (ben/ban, de/di, birbiri/birbir) and the possessive-less stem of compounds such as
//! cumhurbaşkan-lar-ı are generated here too.

use super::dictionary::{DictionaryItem, RootLexicon};
use super::morphotactics::StateId;
use super::phonetics::{attributes_of, is_vowel, voice, PhoneticAttrs as P};
use super::pos::{PrimaryPos, RootAttr, SecondaryPos};
use super::turkish_morphotactics::RootStates;
use ahash::AHashMap;

/// A stem: its surface, the item it belongs to (a handle [`StemIndex::item`] resolves), its
/// phonetic attributes and the state the analysis starts in.
#[derive(Debug, Clone, PartialEq)]
pub struct StemTransition {
    pub surface: Box<str>,
    pub item: u32,
    pub attrs: P,
    pub to: StateId,
}

/// Every stem of a lexicon, by surface. A stem's item is the lexicon's own, found by its
/// place there (so the items are not stored twice, review 2 MORPH2-13), or one the index made
/// itself: a compound's possessive-less root, or the rare item whose place does not fit the
/// handle.
#[derive(Clone)]
pub struct StemIndex {
    own: Vec<DictionaryItem>,
    by_surface: AHashMap<Box<str>, Vec<StemTransition>>,
    max_chars: usize,
}

/// A stem's item handle: the index's own item (this bit, and its place in [`StemIndex::own`]),
/// or the lexicon's, by its root bucket (the low 24 bits) and its place in it (7 bits).
const OWN: u32 = 1 << 31;

/// The handle of the lexicon's item at `(bucket, place)`, if it fits.
fn lexicon_handle((bucket, place): (usize, usize)) -> Option<u32> {
    (bucket < 1 << 24 && place < 1 << 7).then_some((place as u32) << 24 | bucket as u32)
}

impl StemIndex {
    pub fn new(lexicon: &RootLexicon, roots: &RootStates) -> Self {
        let mut index = StemIndex {
            own: Vec::new(),
            by_surface: AHashMap::with_capacity(lexicon.count() * 5 / 4),
            max_chars: 0,
        };
        // In lexicon order (homographs in Zemberek's order), so the order of equal analyses
        // does not depend on hash order.
        for (at, item) in lexicon.positions() {
            index.add(at, item, lexicon, roots);
        }
        index
    }

    /// Adds the stems of the lexicon's item at `at` (its bucket and place, as
    /// [`RootLexicon::add_item`] returns them), and for a CompoundP3sg noun its
    /// possessive-less stem.
    pub fn add(
        &mut self,
        at: (usize, usize),
        item: &DictionaryItem,
        lexicon: &RootLexicon,
        roots: &RootStates,
    ) {
        let compound = compound_root_item(item, lexicon);
        let handle = lexicon_handle(at).unwrap_or_else(|| self.keep(item.clone()));
        for s in generate(item, handle, roots) {
            self.insert(s);
        }
        if let Some((surface, dummy)) = compound {
            let handle = self.keep(dummy.clone());
            for s in compound_stems(&surface, &dummy, handle, roots) {
                self.insert(s);
            }
        }
    }

    /// Stores an item of the index's own and returns its handle.
    fn keep(&mut self, item: DictionaryItem) -> u32 {
        self.own.push(item);
        OWN | (self.own.len() - 1) as u32
    }

    fn insert(&mut self, s: StemTransition) {
        self.max_chars = self.max_chars.max(s.surface.chars().count());
        match self.by_surface.get_mut(&s.surface) {
            Some(bucket) => {
                if !bucket.contains(&s) {
                    bucket.push(s);
                }
            }
            None => {
                self.by_surface.insert(s.surface.clone(), vec![s]);
            }
        }
    }

    /// The stems that are prefixes of `word`, shortest first.
    pub fn prefix_matches<'a>(
        &'a self,
        word: &'a str,
    ) -> impl Iterator<Item = &'a StemTransition> + 'a {
        word.char_indices()
            .map(|(i, c)| i + c.len_utf8())
            .take(self.max_chars)
            .filter_map(move |end| self.by_surface.get(&word[..end]))
            .flatten()
    }

    /// The item a stem belongs to, in `lexicon`, the one the index was built from.
    pub fn item<'a>(&'a self, s: &StemTransition, lexicon: &'a RootLexicon) -> &'a DictionaryItem {
        if s.item & OWN != 0 {
            &self.own[(s.item & !OWN) as usize]
        } else {
            lexicon.item_at((s.item & 0xFF_FFFF) as usize, (s.item >> 24) as usize)
        }
    }
}

/// An item's stems, each with the item its analyses render: the item's own, and for a
/// CompoundP3sg noun the possessive-less stem of its dummy item (cumhurbaşkan-ı-na).
pub fn item_stems(
    item: &DictionaryItem,
    lexicon: &RootLexicon,
    roots: &RootStates,
) -> Vec<(StemTransition, DictionaryItem)> {
    let mut out: Vec<(StemTransition, DictionaryItem)> = generate(item, 0, roots)
        .into_iter()
        .map(|s| (s, item.clone()))
        .collect();
    if let Some((surface, dummy)) = compound_root_item(item, lexicon) {
        for s in compound_stems(&surface, &dummy, 0, roots) {
            out.push((s, dummy.clone()));
        }
    }
    out
}

/// The stems of a compound's dummy item, spelled as the compound's root without its
/// possessive ending.
fn compound_stems(
    surface: &str,
    dummy: &DictionaryItem,
    idx: u32,
    roots: &RootStates,
) -> Vec<StemTransition> {
    let mut stems = generate(dummy, idx, roots);
    for s in &mut stems {
        if s.surface.as_ref() == dummy.root.as_str() {
            s.surface = surface.into();
        }
    }
    stems
}

/// The possessive-less stem of a CompoundP3sg noun (cumhurbaşkanı -> cumhurbaşkan), as an
/// item with the head's attributes but no voicing (Zemberek's dummy CompoundP3sgRoot item).
/// The item keeps the compound's lemma and root, so analyses through it name the compound.
fn compound_root_item(
    item: &DictionaryItem,
    lexicon: &RootLexicon,
) -> Option<(String, DictionaryItem)> {
    if !item.attributes.contains(RootAttr::COMPOUND_P3SG) {
        return None;
    }
    let (first, head) = item.compound_roots.as_ref()?;
    let surface = format!("{first}{head}");
    let mut attrs = lexicon
        .get_items(head)
        .and_then(|items| {
            items
                .iter()
                .find(|i| matches!(i.primary_pos, PrimaryPos::Noun | PrimaryPos::Adj))
        })
        .map_or(RootAttr::NONE, |h| {
            h.attributes
                & (RootAttr::VOWEL_DROP
                    | RootAttr::CONSONANT_DOUBLING
                    | RootAttr::INVERSE_HARMONY
                    | RootAttr::NO_VOICING)
        });
    attrs |= RootAttr::COMPOUND_P3SG_ROOT | RootAttr::NO_VOICING;
    let mut dummy = DictionaryItem::new(
        item.lemma.clone(),
        item.root.clone(),
        item.primary_pos,
        item.secondary_pos,
        attrs,
    );
    dummy.pronunciation = Some(surface.clone());
    Some((surface, dummy))
}

fn calculate(seq: &str) -> P {
    attributes_of(seq, P::empty())
}

/// Zemberek's `TurkishMorphotactics.getRootState`.
pub fn root_state(item: &DictionaryItem, attrs: P, roots: &RootStates) -> StateId {
    match (item.lemma.as_str(), item.primary_pos) {
        ("değil", PrimaryPos::Verb) => return roots.verb_degil,
        ("imek", PrimaryPos::Verb) => return roots.imek,
        ("su" | "akarsu", PrimaryPos::Noun) if item.secondary_pos == SecondaryPos::None => {
            return roots.noun_su_root
        }
        ("öyle" | "böyle" | "şöyle", PrimaryPos::Adv) => return roots.adv_for_verb_deriv,
        _ => {}
    }
    if attrs.contains(P::LAST_LETTER_DROPPED) {
        return roots.verb_vowel_drop;
    }
    if item.attributes.contains(RootAttr::RECIPROCAL) {
        return roots.verb_implicit_recip;
    }
    if item.attributes.contains(RootAttr::REFLEXIVE) {
        return roots.verb_implicit_reflex;
    }
    match item.primary_pos {
        PrimaryPos::Noun => match item.secondary_pos {
            SecondaryPos::ProperNoun => roots.noun_proper,
            SecondaryPos::Abbreviation => roots.noun_abbrv,
            _ if item.attributes.contains(RootAttr::COMPOUND_P3SG_ROOT) => roots.noun_compound_root,
            _ => roots.noun,
        },
        PrimaryPos::Adj => roots.adj,
        PrimaryPos::Pron => match item.secondary_pos {
            SecondaryPos::Personal => roots.pron_pers,
            SecondaryPos::Demonstrative => roots.pron_demons,
            SecondaryPos::Question => roots.pron_ques,
            SecondaryPos::Reflexive => roots.pron_reflex,
            _ => roots.pron_quant,
        },
        PrimaryPos::Adv => roots.adv,
        PrimaryPos::Conj => roots.conj,
        PrimaryPos::Q => roots.question,
        PrimaryPos::Interj => roots.interj,
        PrimaryPos::Verb => roots.verb,
        PrimaryPos::Punc => roots.punc,
        PrimaryPos::Det => roots.det,
        PrimaryPos::Postp => roots.postp,
        PrimaryPos::Num => roots.num,
        PrimaryPos::Dup => roots.dup,
        PrimaryPos::Unknown => roots.noun,
    }
}

/// The stems of one item (Zemberek's `StemTransitionsBase.generate`). An abbreviation ending
/// in K is read with the letter's name "ke", as TDK writes it (TDK'nin, KHK'nin), as well as
/// "ka" (HMK'nın), which Zemberek uses.
pub fn generate(item: &DictionaryItem, idx: u32, roots: &RootStates) -> Vec<StemTransition> {
    let mut stems = generate_one(item, idx, roots);
    if item.secondary_pos == SecondaryPos::Abbreviation && item.root.ends_with('k') {
        let pron = item.pronunciation();
        let other = if let Some(p) = pron.strip_suffix("ka") {
            Some(format!("{p}ke"))
        } else {
            pron.strip_suffix("ke").map(|p| format!("{p}ka"))
        };
        if let Some(other) = other {
            let attrs = calculate(&other);
            if !stems.iter().any(|s| s.attrs == attrs) {
                stems.push(StemTransition {
                    surface: item.root.clone().into_boxed_str(),
                    item: idx,
                    attrs,
                    to: root_state(item, attrs, roots),
                });
            }
        }
    }
    stems
}

fn generate_one(item: &DictionaryItem, idx: u32, roots: &RootStates) -> Vec<StemTransition> {
    if let Some(stems) = special_roots(item, idx, roots) {
        return stems;
    }
    let modifiers = RootAttr::CONSONANT_DOUBLING
        | RootAttr::VOWEL_DROP
        | RootAttr::PROGRESSIVE_VOWEL_DROP
        | RootAttr::INVERSE_HARMONY
        | RootAttr::VOICING
        | RootAttr::COMPOUND_P3SG
        | RootAttr::COMPOUND_P3SG_ROOT;
    let pron = item.pronunciation();
    if !item.attributes.intersects(modifiers) {
        let attrs = calculate(pron);
        return vec![StemTransition {
            surface: item.root.clone().into_boxed_str(),
            item: idx,
            attrs,
            to: root_state(item, attrs, roots),
        }];
    }
    modified_root_nodes(item, idx, roots)
}

/// Zemberek's `generateModifiedRootNodes`. The changes apply in Zemberek's attribute order:
/// progressive vowel drop, voicing, inverse harmony, doubling, vowel drop (ret -> red -> redd,
/// nakit -> nakid -> nakd).
fn modified_root_nodes(item: &DictionaryItem, idx: u32, roots: &RootStates) -> Vec<StemTransition> {
    let a = item.attributes;
    let mut seq: Vec<char> = item.pronunciation().chars().collect();
    let mut original = calculate(item.pronunciation());
    let mut modified = original;
    let mut unmodified_state = None;
    let mut modified_state = None;

    if a.contains(RootAttr::PROGRESSIVE_VOWEL_DROP) && seq.len() > 1 {
        seq.pop();
        let s: String = seq.iter().collect();
        if s.chars().any(is_vowel) {
            modified = calculate(&s);
        }
        modified |= P::LAST_LETTER_DROPPED;
    }
    if a.contains(RootAttr::VOICING) {
        if let Some(last) = seq.last_mut() {
            let mut voiced = voice(*last);
            if voiced != *last {
                if item.lemma.ends_with("nk") {
                    voiced = 'g';
                }
                *last = voiced;
                modified -= P::LAST_LETTER_VOICELESS_STOP;
                original |= P::EXPECTS_CONSONANT;
                modified |= P::EXPECTS_VOWEL | P::CANNOT_TERMINATE;
            }
        }
    }
    if a.contains(RootAttr::INVERSE_HARMONY) {
        for attrs in [&mut original, &mut modified] {
            *attrs |= P::LAST_VOWEL_FRONTAL;
            *attrs -= P::LAST_VOWEL_BACK;
        }
    }
    if a.contains(RootAttr::CONSONANT_DOUBLING) {
        if let Some(&last) = seq.last() {
            seq.push(last);
            original |= P::EXPECTS_CONSONANT;
            modified |= P::EXPECTS_VOWEL | P::CANNOT_TERMINATE;
        }
    }
    if a.contains(RootAttr::VOWEL_DROP) && seq.len() >= 2 {
        if seq.last().is_some_and(|&c| is_vowel(c)) {
            seq.pop();
            modified |= P::EXPECTS_CONSONANT | P::CANNOT_TERMINATE;
        } else {
            seq.remove(seq.len() - 2);
            if item.primary_pos == PrimaryPos::Verb {
                unmodified_state = Some(roots.verb_last_vowel_drop_unmod);
                modified_state = Some(roots.verb_last_vowel_drop_mod);
            } else {
                original |= P::EXPECTS_CONSONANT;
            }
            modified |= P::EXPECTS_VOWEL | P::CANNOT_TERMINATE;
        }
    }

    let first = StemTransition {
        surface: item.root.clone().into_boxed_str(),
        item: idx,
        attrs: original,
        to: unmodified_state.unwrap_or_else(|| root_state(item, original, roots)),
    };
    let second = StemTransition {
        surface: seq.iter().collect::<String>().into_boxed_str(),
        item: idx,
        attrs: modified,
        to: modified_state.unwrap_or_else(|| root_state(item, modified, roots)),
    };
    if first.surface == second.surface && first.attrs == second.attrs {
        vec![first]
    } else {
        vec![first, second]
    }
}

/// Zemberek's `handleSpecialRoots`.
fn special_roots(
    item: &DictionaryItem,
    idx: u32,
    roots: &RootStates,
) -> Option<Vec<StemTransition>> {
    let stem = |surface: &str, attrs: P, to: StateId| StemTransition {
        surface: surface.into(),
        item: idx,
        attrs,
        to,
    };
    // Only the special roots need the attributes; computing them for every item cost
    // a sizeable share of the stem index's build.
    let original_attrs = || calculate(item.pronunciation());
    let lemma = item.lemma.as_str();
    let pos = item.primary_pos;
    let sec = item.secondary_pos;
    match (lemma, pos) {
        ("içeri", PrimaryPos::Noun | PrimaryPos::Adj)
        | ("dışarı", PrimaryPos::Noun | PrimaryPos::Adj | PrimaryPos::Postp)
        | ("yukarı", PrimaryPos::Noun | PrimaryPos::Adj)
        | ("ileri" | "şura" | "bura" | "ora", PrimaryPos::Noun)
            if sec == SecondaryPos::None =>
        {
            // içer-de, bur-da, or-dan.
            let original = stem(
                &item.root,
                original_attrs(),
                root_state(item, original_attrs(), roots),
            );
            let target = if pos == PrimaryPos::Noun {
                roots.noun_last_vowel_drop
            } else {
                roots.adj_last_vowel_drop
            };
            let mut chars: Vec<char> = item.root.chars().collect();
            chars.pop();
            let m: String = chars.into_iter().collect();
            let attrs = calculate(&m) | P::EXPECTS_CONSONANT | P::CANNOT_TERMINATE;
            Some(vec![original, stem(&m, attrs, target)])
        }
        ("ben" | "sen", PrimaryPos::Pron) if sec == SecondaryPos::Personal => {
            let original = stem(
                &item.root,
                original_attrs() | P::UNMODIFIED_PRONOUN,
                roots.pron_pers,
            );
            let m = if lemma == "ben" { "ban" } else { "san" };
            Some(vec![
                original,
                stem(m, calculate(m) | P::MODIFIED_PRONOUN, roots.pron_pers_mod),
            ])
        }
        ("demek" | "yemek", PrimaryPos::Verb) => {
            let m = if lemma == "demek" { "di" } else { "yi" };
            Some(vec![
                stem(&item.root, original_attrs(), roots.verb_de_ye),
                stem(m, calculate(m), roots.verb_de_ye),
            ])
        }
        ("imek", PrimaryPos::Verb) => Some(vec![stem(&item.root, original_attrs(), roots.imek)]),
        ("birbiri" | "çoğu" | "öbürü" | "birçoğu", PrimaryPos::Pron)
            if sec == SecondaryPos::Quantitive =>
        {
            let m = match lemma {
                "birbiri" => "birbir",
                "çoğu" => "çok",
                "öbürü" => "öbür",
                _ => "birçok",
            };
            Some(vec![
                stem(
                    &item.root,
                    original_attrs() | P::UNMODIFIED_PRONOUN,
                    roots.pron_quant,
                ),
                stem(
                    m,
                    calculate(m) | P::MODIFIED_PRONOUN,
                    roots.pron_quant_modified,
                ),
            ])
        }
        _ => None,
    }
}
