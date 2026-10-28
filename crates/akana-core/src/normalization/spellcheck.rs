//! High-speed Turkish Spell Checker with StringZilla SIMD edit distance acceleration and morphology engine.

use crate::morphology::TurkishMorphology;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::{Arc, LazyLock};
use stringzilla::StringZilla;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpellSuggestion {
    pub word: String,
    pub distance: usize,
    pub score: f32,
}

/// The words a checker accepts and suggests: a set for lookups, and the words in the order
/// they were added, which fixes the order of equally good suggestions.
#[derive(Clone, Default)]
struct Vocabulary {
    words: HashSet<String>,
    list: Vec<String>,
}

impl Vocabulary {
    fn add(&mut self, lower: String) {
        if self.words.insert(lower.clone()) {
            self.list.push(lower);
        }
    }
}

/// The vocabulary of the built-in lexicon, built once per process and shared by every checker
/// until it adds a word of its own (copy-on-write): its common words. A nonstandard item is no
/// spelling (herşey, whose lemma is its own), and names and abbreviations are judged as written
/// by the analyzer (`Türkiye`, `TBMM`): in this lower-cased list the corpus names made
/// misspellings words (egitim, varmı) and abbreviations made suggestions (cok: cbk; review 3
/// NORM3-01).
static DEFAULT_VOCABULARY: LazyLock<Arc<Vocabulary>> = LazyLock::new(|| {
    let mut vocabulary = Vocabulary::default();
    for line in crate::morphology::ZEMBEREK_LEXICON.lines() {
        let mut fields = line.split_whitespace();
        if matches!(fields.nth(3), Some("ProperNoun" | "Abbrv"))
            || line.split_whitespace().any(|f| f == "Nonstandard")
        {
            continue;
        }
        if let Some(first_word) = line.split_whitespace().next() {
            vocabulary.add(super::super::phonology::to_turkish_lower(first_word));
        }
    }
    Arc::new(vocabulary)
});

pub struct TurkishSpellChecker {
    vocabulary: Arc<Vocabulary>,
    morphology: TurkishMorphology,
}

impl Default for TurkishSpellChecker {
    fn default() -> Self {
        Self::new()
    }
}

impl TurkishSpellChecker {
    pub fn new() -> Self {
        Self {
            vocabulary: Arc::clone(&DEFAULT_VOCABULARY),
            morphology: TurkishMorphology::new(),
        }
    }

    pub fn with_vocabulary(words: impl IntoIterator<Item = String>) -> Self {
        let mut vocabulary = Vocabulary::default();
        for w in words {
            vocabulary.add(super::super::phonology::to_turkish_lower(&w));
        }
        Self {
            vocabulary: Arc::new(vocabulary),
            morphology: TurkishMorphology::new(),
        }
    }

    /// Adds a word to the spell checker vocabulary.
    pub fn add_word(&mut self, word: &str) {
        let lower = super::super::phonology::to_turkish_lower(word);
        if !self.vocabulary.words.contains(&lower) {
            Arc::make_mut(&mut self.vocabulary).add(lower);
        }
    }

    /// Checks if a word is correctly spelled: in the vocabulary, or read by the morphological
    /// analyzer as TDK spells it, in the case it is written in (a nonstandard reading,
    /// temerrüte or olmıyan, is no spelling, nor a name's reading of ankarada).
    #[inline]
    pub fn is_correct(&self, word: &str) -> bool {
        let lower = super::super::phonology::to_turkish_lower(word);
        if self.vocabulary.words.contains(&lower) {
            return true;
        }
        !self.morphology.analyze_standard(word).is_empty()
    }

    /// Suggests correctly spelled words using StringZilla SIMD-accelerated edit distance.
    pub fn suggest(
        &self,
        word: &str,
        max_distance: usize,
        max_suggestions: usize,
    ) -> Vec<SpellSuggestion> {
        let lower = super::super::phonology::to_turkish_lower(word);
        if self.is_correct(word) {
            return vec![SpellSuggestion {
                word: lower,
                distance: 0,
                score: 1.0,
            }];
        }

        let mut candidates: Vec<SpellSuggestion> = Vec::new();

        for candidate in &self.vocabulary.list {
            // Fast length difference heuristic before SIMD distance check
            let len_diff = (candidate.len() as isize - lower.len() as isize).unsigned_abs();
            if len_diff > max_distance {
                continue;
            }

            // SIMD Edit distance via StringZilla
            let dist = lower.as_str().sz_edit_distance(candidate.as_str());
            if dist <= max_distance {
                let score = 1.0 / (1.0 + dist as f32);
                candidates.push(SpellSuggestion {
                    word: candidate.clone(),
                    distance: dist,
                    score,
                });
            }
        }

        candidates.sort_by(|a, b| {
            a.distance
                .cmp(&b.distance)
                .then_with(|| b.score.partial_cmp(&a.score).unwrap())
        });

        candidates.truncate(max_suggestions);
        candidates
    }
}
