//! Turkish Compound Word Analysis and Decomposition (Bileşik Sözcük Ayrıştırıcı).

use super::analyzer::{is_derivational_tag, MorphParse, TurkishMorphology};
use super::pos::{PrimaryPos, RootAttr, SecondaryPos};
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

        // No real word is this long, even with a long suffix chain; each split analyses both
        // halves, so a long token (a base64 blob) was quadratic or worse.
        if !(4..=128).contains(&n) {
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

    /// Splits a lexicon root into the two roots of a noun compound, from the lexicon alone: a
    /// noun, adjective or numeral, then a noun or adjective either bare (baş+bakan, yüz+yıl,
    /// büyük+şehir) or with the compound's 3rd-person possessive (deniz+alt-ı, zeytin+yağ-ı,
    /// cumhur+başkan-ı, ayak+kab-ı). Returns the two roots (deniz, alt).
    ///
    /// Unlike `decompose`, which accepts any split whose halves have any analysis, neither part
    /// may carry other suffixes, verbs, proper nouns and abbreviations do not count, parts have
    /// at least three letters (two for the possessive stem of a word the lexicon marks
    /// CompoundP3sg: hanım+el-i), and a word that is a shorter root plus a derivation is not a
    /// compound (dava-cı, not dav+acı; yetki-li). Splitting inflected surfaces instead turned
    /// davacıya into dav + acıya and görevlisi into gör + evlisi.
    pub fn split_lexical_compound(
        morphology: &TurkishMorphology,
        word: &str,
    ) -> Option<(String, String)> {
        let lower = crate::phonology::to_turkish_lower(word.trim());
        let chars: Vec<char> = lower.chars().collect();
        let n = chars.len();
        // Two roots are at most twice the longest root; longer tokens (a base64 blob) would
        // only cost a quadratic scan.
        if n < 6 || n > 2 * morphology.max_root_chars() {
            return None;
        }
        let marked = morphology
            .root_items(&lower)
            .iter()
            .any(|i| i.attributes.contains(RootAttr::COMPOUND_P3SG));
        let derived = morphology
            .analyze(&lower)
            .iter()
            .any(|p| p.root != lower && p.morpheme_tags.iter().any(|t| is_derivational_tag(t)));
        if derived {
            return None;
        }

        let is_root = |s: &str, second: bool| {
            morphology.root_items(s).iter().any(|i| {
                !matches!(
                    i.secondary_pos,
                    SecondaryPos::ProperNoun | SecondaryPos::Abbreviation
                ) && match i.primary_pos {
                    PrimaryPos::Noun | PrimaryPos::Adj => true,
                    PrimaryPos::Num => !second,
                    _ => false,
                }
            })
        };
        let possessive_stem = |s: &str| -> Option<String> {
            let min = if marked { 2 } else { 3 };
            ["sı", "si", "su", "sü", "ı", "i", "u", "ü"]
                .iter()
                .filter_map(|p| s.strip_suffix(p))
                .filter(|stem| stem.chars().count() >= min)
                .flat_map(|stem| [stem.to_string(), unvoice(stem)])
                .find(|stem| is_root(stem, true))
        };

        for split in 3..=(n - 3) {
            let left: String = chars[..split].iter().collect();
            if !is_root(&left, false) {
                continue;
            }
            let right: String = chars[split..].iter().collect();
            // A word marked CompoundP3sg ends in the possessive: zeytin+yağ-ı, not the
            // archaic root yağı.
            if marked {
                if let Some(stem) = possessive_stem(&right) {
                    return Some((left, stem));
                }
            }
            if is_root(&right, true) {
                return Some((left, right));
            }
            if let Some(stem) = possessive_stem(&right) {
                return Some((left, stem));
            }
        }
        None
    }
}

/// kab -> kap, reng -> renk, ağac -> ağaç (the stem before a possessive may be voiced).
fn unvoice(stem: &str) -> String {
    let mut chars: Vec<char> = stem.chars().collect();
    if let Some(last) = chars.last_mut() {
        *last = match *last {
            'b' => 'p',
            'c' => 'ç',
            'd' => 't',
            'ğ' | 'g' => 'k',
            c => c,
        };
    }
    chars.into_iter().collect()
}
