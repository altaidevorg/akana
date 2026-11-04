//! Morphology profile: how often the words of a text take each part of speech and each
//! morpheme, from the best reading of Akana's morphological analyzer
//! ([`crate::morphology::TurkishMorphology::analyze`]).
//!
//! A model input `m_<key>` is the count of `<key>` per 100 words, where `<key>` is `pos:<POS>`
//! (the word's part of speech after its last derivation), a morpheme tag after the root's own
//! part of speech (`Pass`, `Prog`, `PresPart`, `A2pl`, ...), `deriv` (derivational tags) or
//! `unparsed` (words without a reading). `m_tags_per_word` is the morpheme tags per word.
//! Words are the word tokens of [`super::features::scan_tokens`] on the cleaned text: masks
//! (`<SAYI>`, `<ÖZEL>`, ...) and the suffixes after an apostrophe are no words.
//!
//! The analyzer is shared code: a change to its lexicon, morphotactics, tag names or ranking
//! moves these inputs, which `inputs_match_drift_canary` records.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::LazyLock;

use ahash::AHashMap;

use super::features::{scan_tokens, Tok};
use crate::morphology::{is_derivational_tag, PrimaryPos, TurkishMorphology};

/// The detector's own analyzer, as the NER, PII and normalization modules keep theirs: analyzers
/// share the built-in lexicon, so it costs no memory, and `detect` is called without an
/// auditor instance to borrow one from.
static MORPHOLOGY: LazyLock<TurkishMorphology> = LazyLock::new(TurkishMorphology::new);

/// Prefix of the model inputs this module computes.
pub(in crate::style) const INPUT_PREFIX: &str = "m_";
const TAGS_PER_WORD: &str = "tags_per_word";
/// Keys of [`MorphProfile::counts`] that are no input of their own.
pub(super) const RESERVED_KEYS: [&str; 2] = ["words", "tags"];

/// What one word adds to a profile: its keys (`pos:`, then its tags, each derivational tag
/// preceded by a `deriv`; `unparsed` alone without a reading) and how many of them are tags.
struct WordKeys {
    keys: Vec<String>,
    tags: u32,
}

/// Distinct words per thread whose keys are kept, so that a batch of texts analyses each word
/// once; the cache is emptied when it reaches this size.
const CACHE_WORDS: usize = 20_000;

thread_local! {
    static CACHE: RefCell<AHashMap<String, Rc<WordKeys>>> = RefCell::new(AHashMap::new());
}

fn word_keys(word: &str) -> Rc<WordKeys> {
    if let Some(k) = CACHE.with(|c| c.borrow().get(word).cloned()) {
        return k;
    }
    let k = Rc::new(analyse(word));
    CACHE.with(|c| {
        let mut c = c.borrow_mut();
        if c.len() >= CACHE_WORDS {
            c.clear();
        }
        c.insert(word.to_string(), Rc::clone(&k));
    });
    k
}

fn analyse(word: &str) -> WordKeys {
    let Some(r) = MORPHOLOGY.analyze(word).into_iter().next() else {
        return WordKeys {
            keys: vec!["unparsed".to_string()],
            tags: 0,
        };
    };
    let mut keys = Vec::with_capacity(r.morpheme_tags.len() + 2);
    keys.push(format!("pos:{}", r.pos.as_str()));
    // The first tag names the root's part of speech, which `pos:` already describes.
    let skip = usize::from(
        r.morpheme_tags
            .first()
            .is_some_and(|t| PrimaryPos::from_name(t).is_some()),
    );
    let mut tags = 0;
    for t in r.morpheme_tags.into_iter().skip(skip) {
        tags += 1;
        if is_derivational_tag(&t) {
            keys.push("deriv".to_string());
        }
        keys.push(t);
    }
    WordKeys { keys, tags }
}

/// Counts over the words of one text.
#[derive(Debug, Default)]
pub(super) struct MorphProfile {
    words: u32,
    tags: u32,
    counts: AHashMap<String, u32>,
}

impl MorphProfile {
    /// Value of the input `m_<key>` (see the module docs).
    pub(super) fn value(&self, key: &str) -> f64 {
        let words = f64::from(self.words.max(1));
        if key == TAGS_PER_WORD {
            return f64::from(self.tags) / words;
        }
        100.0 * f64::from(self.counts.get(key).copied().unwrap_or(0)) / words
    }

    /// All counts with `words` and `tags` (for the training-input dump).
    #[cfg(test)]
    pub(super) fn counts(&self) -> std::collections::BTreeMap<String, u32> {
        let mut m: std::collections::BTreeMap<String, u32> =
            self.counts.iter().map(|(k, v)| (k.clone(), *v)).collect();
        m.insert("words".into(), self.words);
        m.insert("tags".into(), self.tags);
        m
    }
}

/// The profile of a cleaned text ([`super::preprocess::preprocess`]).
pub(super) fn morph_profile(clean: &str) -> MorphProfile {
    // Each distinct word is analysed once and counted as often as it occurs.
    let mut occurrences: AHashMap<&str, u32> = AHashMap::new();
    scan_tokens(clean, |t| {
        if let Tok::Word(w) = t {
            *occurrences.entry(w).or_insert(0) += 1;
        }
    });
    let mut p = MorphProfile::default();
    for (word, n) in occurrences {
        let w = word_keys(word);
        p.words += n;
        p.tags += w.tags * n;
        for k in &w.keys {
            match p.counts.get_mut(k.as_str()) {
                Some(c) => *c += n,
                None => {
                    p.counts.insert(k.clone(), n);
                }
            }
        }
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_text_has_zero_inputs() {
        let p = morph_profile("");
        for key in ["pos:Noun", "Loc", TAGS_PER_WORD, "unparsed"] {
            assert_eq!(p.value(key), 0.0, "{key}");
        }
    }

    #[test]
    fn counts_tags_of_the_best_reading_per_occurrence() {
        // `evlerde` twice: Noun+A3pl+Loc (the root's Noun is the `pos:` key, not a tag).
        let p = morph_profile("evlerde evlerde");
        assert_eq!(p.value("pos:Noun"), 100.0);
        assert_eq!(p.value("A3pl"), 100.0);
        assert_eq!(p.value("Loc"), 100.0);
        assert_eq!(p.value("Noun"), 0.0);
        assert_eq!(p.value(TAGS_PER_WORD), 2.0);
        assert_eq!(p.counts()["words"], 2);
    }

    #[test]
    fn masks_and_apostrophe_suffixes_are_no_words() {
        let p = morph_profile("<ÖZEL>'da <SAYI> kitap İstanbul'a");
        assert_eq!(p.counts()["words"], 2);
        assert_eq!(p.value("unparsed"), 0.0);
    }

    #[test]
    fn derivations_are_counted_with_their_tag() {
        // `sağlayan`: sağla:Verb+PresPart, a derivation to an adjective.
        let p = morph_profile("sağlayan");
        assert_eq!(p.value("pos:Adj"), 100.0);
        assert_eq!(p.value("PresPart"), 100.0);
        assert_eq!(p.value("deriv"), 100.0);
    }

    #[test]
    fn words_without_a_reading_are_unparsed() {
        let p = morph_profile("xqzvbn");
        assert_eq!(p.value("unparsed"), 100.0);
        assert_eq!(p.value(TAGS_PER_WORD), 0.0);
    }
}
