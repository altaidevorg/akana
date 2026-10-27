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

    /// The root of the word's best-scoring analysis, or the word in lower case when it has
    /// none. The word is analysed as written, so its capital still tells a name from a common
    /// word (Buhara, not buhar) and a Roman numeral keeps its root (XIX., not xıx.; review 2
    /// MORPH2-11, RANK2-10); a root from the lexicon is in lower case either way.
    pub fn stem(&self, word: &str) -> String {
        let word = word.trim();
        match self.morphology.analyze(word).into_iter().next() {
            Some(top) => top.root,
            None => to_turkish_lower(word),
        }
    }

    /// Stems a list of word tokens.
    pub fn stem_tokens(&self, tokens: &[&str]) -> Vec<String> {
        tokens.iter().map(|&t| self.stem(t)).collect()
    }
}
