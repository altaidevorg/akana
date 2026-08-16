//! Turkish Compound Word Analysis and Decomposition (Bileşik Sözcük Ayrıştırıcı).

use super::analyzer::{MorphParse, TurkishMorphology};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompoundAnalysis {
    pub surface: String,
    pub part1: String,
    pub part2: String,
    pub parse1: MorphParse,
    pub parse2: MorphParse,
}

pub struct CompoundDecomposer {
    morphology: TurkishMorphology,
}

impl Default for CompoundDecomposer {
    fn default() -> Self {
        Self::new()
    }
}

impl CompoundDecomposer {
    pub fn new() -> Self {
        Self {
            morphology: TurkishMorphology::new(),
        }
    }

    pub fn with_morphology(morphology: TurkishMorphology) -> Self {
        Self { morphology }
    }

    /// Attempts to decompose a candidate word into two valid Turkish linguistic roots/words.
    /// Example: `decompose("denizaltı") -> ("deniz", "altı")`
    /// Example: `decompose("gecekondu") -> ("gece", "kondu")`
    /// Example: `decompose("akbaba") -> ("ak", "baba")`
    pub fn decompose(&self, word: &str) -> Vec<CompoundAnalysis> {
        let clean = word.trim();
        let lower = crate::phonology::to_turkish_lower(clean);
        let chars: Vec<char> = lower.chars().collect();
        let n = chars.len();

        if n < 4 {
            return Vec::new();
        }

        let mut results = Vec::new();

        // Try all split points (minimum 2 chars per subword)
        for split in 2..=(n - 2) {
            let left_part: String = chars[..split].iter().collect();
            let right_part: String = chars[split..].iter().collect();

            let left_parses = self.morphology.analyze(&left_part);
            if left_parses.is_empty() {
                continue;
            }

            let right_parses = self.morphology.analyze(&right_part);
            if right_parses.is_empty() {
                continue;
            }

            // Valid decomposition found!
            results.push(CompoundAnalysis {
                surface: clean.to_string(),
                part1: left_part,
                part2: right_part,
                parse1: left_parses[0].clone(),
                parse2: right_parses[0].clone(),
            });
        }

        results
    }
}
