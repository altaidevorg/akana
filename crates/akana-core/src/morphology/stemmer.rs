//! Morphological Turkish Stemmer and Lemmatizer.

use super::analyzer::TurkishMorphology;
use crate::phonology::to_turkish_lower;

pub struct TurkishStemmer {
    morphology: TurkishMorphology,
}

impl Default for TurkishStemmer {
    fn default() -> Self {
        Self::new()
    }
}

impl TurkishStemmer {
    pub fn new() -> Self {
        Self {
            morphology: TurkishMorphology::new(),
        }
    }

    pub fn with_morphology(morphology: TurkishMorphology) -> Self {
        Self { morphology }
    }

    /// Finds the linguistic root/stem of a word using morphological analysis.
    /// If multiple parses exist, returns the best-scoring root.
    /// Falls back to lowercase surface form if unknown.
    pub fn stem(&self, word: &str) -> String {
        let lower = to_turkish_lower(word);
        let parses = self.morphology.analyze(&lower);
        if let Some(top) = parses.first() {
            top.root.clone()
        } else {
            lower
        }
    }

    /// Stems a list of word tokens.
    pub fn stem_tokens(&self, tokens: &[&str]) -> Vec<String> {
        tokens.iter().map(|&t| self.stem(t)).collect()
    }
}
