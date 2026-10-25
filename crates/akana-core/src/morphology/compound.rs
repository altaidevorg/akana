//! Turkish Compound Word Analysis and Decomposition (Bileşik Sözcük Ayrıştırıcı).
//!
//! A word splits into two roots only when the lexicon says it is a compound (Zemberek's
//! CompoundP3sg roots: cumhur+başkan-ı, zeytin+yağ-ı, deniz+alt-ı) or when it is a productive
//! compound pattern (baş+bakan, ana+yasa, tebliğ+name, huzur+ev-i) on an open-class root.
//! Splitting wherever both halves happen to be words split mahkeme into mah + keme and
//! hareket into hare + ket.

use super::analyzer::{is_derivational_tag, MorphParse, TurkishMorphology};
use super::dictionary::DictionaryItem;
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

/// First parts that make compounds with a following noun (başbakan, anayasa, kuzeybatı,
/// ortaokul, ilköğretim, tümgeneral, altyapı, büyükelçi, yüksekokul). The rest must be an
/// open-class root of at least four letters: başkan is not baş + kan.
const PRODUCTIVE_FIRST: &[&str] = &[
    "baş", "ana", "kuzey", "güney", "doğu", "batı", "orta", "ilk", "tüm", "alt", "üst", "büyük",
    "yüksek",
];

/// Heads that make compounds after an open-class root of at least three letters, with the
/// head's root (tebliğname, tımarhane, huzurevi, karayolu, ustabaşı, vatansever, dilbilim,
/// olağanüstü, yasadışı, yeryüzü, yüzyıl).
const PRODUCTIVE_HEADS: &[(&str, &str)] = &[
    ("name", "name"),
    ("hane", "hane"),
    ("evi", "ev"),
    ("yolu", "yol"),
    ("kurulu", "kurul"),
    ("başı", "baş"),
    ("sever", "sever"),
    ("bilim", "bilim"),
    ("üstü", "üst"),
    ("dışı", "dış"),
    ("yüzü", "yüz"),
    ("yıl", "yıl"),
];

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

    /// Decomposes a (possibly inflected) word whose root is a compound into its first part and
    /// the rest of the word, each with its best analysis.
    /// Example: `decompose("denizaltında") -> ("deniz", "altında")`
    /// Example: `decompose("başbakanlar") -> ("baş", "bakanlar")`
    pub fn decompose(&self, word: &str) -> Vec<CompoundAnalysis> {
        let clean = word.trim();
        let lower = crate::phonology::to_turkish_lower(clean);
        // No real word is this long, even with a long suffix chain; a long token (a base64
        // blob) would only cost a scan.
        if !(4..=128).contains(&lower.chars().count()) {
            return Vec::new();
        }
        let mut roots: Vec<String> = self
            .morphology
            .analyze(&lower)
            .iter()
            .take(3)
            .map(|p| p.root.clone())
            .collect();
        roots.dedup();
        let mut results: Vec<CompoundAnalysis> = Vec::new();
        for root in roots {
            let Some((first, _)) = Self::split_lexical_compound(&self.morphology, &root) else {
                continue;
            };
            let Some(rest) = lower.strip_prefix(first.as_str()) else {
                continue;
            };
            let (Some(parse1), Some(parse2)) = (
                self.morphology.analyze(&first).into_iter().next(),
                self.morphology.analyze(rest).into_iter().next(),
            ) else {
                continue;
            };
            if results.iter().any(|r| r.part1 == first) {
                continue;
            }
            results.push(CompoundAnalysis {
                surface: clean.to_string(),
                part1: first,
                part2: rest.to_string(),
                parse1,
                parse2,
            });
        }
        results
    }

    /// Splits a lexicon root into the roots of its two parts: from the lexicon's compound
    /// data (cumhurbaşkanı: cumhur, başkan; zeytinyağı: zeytin, yağ; denizaltı: deniz, alt),
    /// else from a productive pattern on open-class lexicon roots (başbakan: baş, bakan;
    /// anayasa: ana, yasa; tebliğname: tebliğ, name; huzurevi: huzur, ev). Names,
    /// abbreviations and words the analyzer reads as a shorter root plus a derivation are not
    /// split.
    pub fn split_lexical_compound(
        morphology: &TurkishMorphology,
        word: &str,
    ) -> Option<(String, String)> {
        let lower = crate::phonology::to_turkish_lower(word.trim());
        let items = morphology.root_items(&lower);
        if let Some((first, head)) = items
            .iter()
            .filter(|i| i.attributes.contains(RootAttr::COMPOUND_P3SG))
            .find_map(|i| i.compound_roots.clone())
        {
            return Some((first, head));
        }
        // A shorter root plus a derivation is not a compound (dava-cı, not dav + acı).
        let derived = || {
            morphology
                .analyze(&lower)
                .iter()
                .any(|p| p.root != lower && p.morpheme_tags.iter().any(|t| is_derivational_tag(t)))
        };
        if !items.iter().any(open_class) || derived() {
            return None;
        }
        let is_root = |s: &str| morphology.root_items(s).iter().any(open_class);
        // A longer root across the split makes the word its derivation: başvur-u (from the
        // verb), not baş + vuru.
        let spanned = |first: &str| {
            lower
                .char_indices()
                .map(|(i, _)| i)
                .filter(|&i| i > first.len())
                .any(|i| !morphology.root_items(&lower[..i]).is_empty())
        };
        for first in PRODUCTIVE_FIRST {
            if let Some(rest) = lower.strip_prefix(first) {
                if rest.chars().count() >= 4 && is_root(rest) && !spanned(first) {
                    return Some((first.to_string(), rest.to_string()));
                }
            }
        }
        for (head, head_root) in PRODUCTIVE_HEADS {
            if let Some(first) = lower.strip_suffix(head) {
                if first.chars().count() >= 3 && is_root(first) && !spanned(first) {
                    return Some((first.to_string(), head_root.to_string()));
                }
            }
        }
        None
    }
}

/// A common noun or adjective (not a name or abbreviation).
fn open_class(item: &DictionaryItem) -> bool {
    matches!(item.primary_pos, PrimaryPos::Noun | PrimaryPos::Adj)
        && !matches!(
            item.secondary_pos,
            SecondaryPos::ProperNoun | SecondaryPos::Abbreviation
        )
}
