//! High-speed Turkish Spell Checker with StringZilla SIMD edit distance acceleration and morphology engine.

use std::collections::HashSet;
use stringzilla::StringZilla;
use serde::{Deserialize, Serialize};
use crate::morphology::TurkishMorphology;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpellSuggestion {
    pub word: String,
    pub distance: usize,
    pub score: f32,
}

pub struct TurkishSpellChecker {
    vocabulary: HashSet<String>,
    vocab_list: Vec<String>,
    morphology: TurkishMorphology,
}

impl Default for TurkishSpellChecker {
    fn default() -> Self {
        Self::new()
    }
}

impl TurkishSpellChecker {
    pub fn new() -> Self {
        let mut checker = Self {
            vocabulary: HashSet::new(),
            vocab_list: Vec::new(),
            morphology: TurkishMorphology::new(),
        };
        checker.load_default_vocabulary();
        checker
    }

    pub fn with_vocabulary(words: impl IntoIterator<Item = String>) -> Self {
        let mut set = HashSet::new();
        let mut list = Vec::new();
        for w in words {
            let lower = super::super::phonology::to_turkish_lower(&w);
            if set.insert(lower.clone()) {
                list.push(lower);
            }
        }
        Self {
            vocabulary: set,
            vocab_list: list,
            morphology: TurkishMorphology::new(),
        }
    }

    fn load_default_vocabulary(&mut self) {
        let lex_str = include_str!("../morphology/zemberek_lexicon.txt");
        for line in lex_str.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Some(first_word) = line.split_whitespace().next() {
                let lower = super::super::phonology::to_turkish_lower(first_word);
                if self.vocabulary.insert(lower.clone()) {
                    self.vocab_list.push(lower);
                }
            }
        }
    }

    /// Adds a word to the spell checker vocabulary.
    pub fn add_word(&mut self, word: &str) {
        let lower = super::super::phonology::to_turkish_lower(word);
        if self.vocabulary.insert(lower.clone()) {
            self.vocab_list.push(lower);
        }
    }

    /// Checks if a word is correctly spelled (either in root dictionary or recognized by morphological analyzer).
    #[inline]
    pub fn is_correct(&self, word: &str) -> bool {
        let lower = super::super::phonology::to_turkish_lower(word);
        if self.vocabulary.contains(&lower) {
            return true;
        }
        !self.morphology.analyze(&lower).is_empty()
    }

    /// Suggests correctly spelled words using StringZilla SIMD-accelerated edit distance.
    pub fn suggest(&self, word: &str, max_distance: usize, max_suggestions: usize) -> Vec<SpellSuggestion> {
        let lower = super::super::phonology::to_turkish_lower(word);
        if self.is_correct(&lower) {
            return vec![SpellSuggestion {
                word: lower,
                distance: 0,
                score: 1.0,
            }];
        }

        let mut candidates: Vec<SpellSuggestion> = Vec::new();

        for candidate in &self.vocab_list {
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
            a.distance.cmp(&b.distance)
                .then_with(|| b.score.partial_cmp(&a.score).unwrap())
        });

        candidates.truncate(max_suggestions);
        candidates
    }
}
