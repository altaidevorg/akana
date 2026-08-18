//! Turkish Phonological Grammar Rules Engine.
//!
//! Evaluates and corrects:
//! 1. Vowel Harmony violations in suffixation (Major & Minor Vowel Harmony).
//! 2. Consonant Assimilation / Hardening (*fıstıkçı şahap* rule: *kitapda* -> *kitapta*).
//! 3. Consonant Softening / Voicing (*kitapı* -> *kitabı*, *ağaca*).
//! 4. Vowel Dropping / Syncope (*akılı* -> *aklı*, *şehire* -> *şehre*).
//! 5. Vowel Narrowing in progressive tense (*başlayor* -> *başlıyor*).

use std::collections::HashMap;
use lazy_static::lazy_static;

use super::{ErrorCategory, GrammarFinding};
use crate::morphology::TurkishMorphology;
use crate::phonology::{
    harmony_i_type, is_front_vowel, is_hard_consonant, is_turkish_vowel, last_vowel,
    to_turkish_lower,
};

lazy_static! {
    /// Common Turkish roots undergoing vowel drop (syncope) upon vowel-initial suffixation:
    /// akıl -> aklı, burun -> burnu, karın -> karnı, göğüs -> göğsü, şehir -> şehri, boyun -> boynu, vs.
    static ref VOWEL_DROP_ROOTS: HashMap<&'static str, &'static str> = {
        let mut m = HashMap::new();
        m.insert("akıl", "akl");
        m.insert("alın", "aln");
        m.insert("bağır", "bağr");
        m.insert("beyin", "beyn");
        m.insert("boyun", "boyn");
        m.insert("burun", "burn");
        m.insert("karın", "karn");
        m.insert("göğüs", "göğs");
        m.insert("gönül", "gönl");
        m.insert("omuz", "omz");
        m.insert("oğul", "oğl");
        m.insert("resim", "resm");
        m.insert("şehir", "şehr");
        m.insert("isim", "ism");
        m.insert("fikir", "fikr");
        m.insert("sabır", "sabr");
        m.insert("vakit", "vakt");
        m.insert("nesil", "nesl");
        m.insert("metin", "metn");
        m.insert("zehir", "zehr");
        m.insert("hüküm", "hükm");
        m.insert("devir", "devr");
        m
    };
}

pub struct PhonologicalRuleEngine;

impl PhonologicalRuleEngine {
    pub fn check(
        _text: &str,
        tokens: &[&str],
        token_spans: &[(usize, usize)],
        morphology: &TurkishMorphology,
        findings: &mut Vec<GrammarFinding>,
    ) {
        for (i, &tok) in tokens.iter().enumerate() {
            let lower = to_turkish_lower(tok);
            let (start, end) = token_spans[i];

            // Skip punctuation or numbers
            if lower.is_empty() || lower.chars().all(|c| !c.is_alphabetic()) {
                continue;
            }

            // 1. Check Consonant Assimilation (fıstıkçı şahap: e.g. "kitapda", "sokakdan", "yapdı", "gitdi")
            if Self::check_consonant_assimilation(tok, &lower, start, end, morphology, findings) {
                continue;
            }

            // 2. Check Vowel Dropping (Syncope: e.g. "akılı", "şehire", "burunu")
            if Self::check_vowel_drop(tok, &lower, start, end, morphology, findings) {
                continue;
            }

            // 3. Check Vowel Narrowing in progressive tense (e.g. "başlayor", "anlamayor")
            if Self::check_vowel_narrowing(tok, &lower, start, end, morphology, findings) {
                continue;
            }

            // 4. Check Consonant Softening (e.g. "kitapı" -> "kitabı", "ağaca")
            if Self::check_consonant_softening(tok, &lower, start, end, morphology, findings) {
                continue;
            }

            // 5. Check Suffix Vowel Harmony (e.g. "evlar", "arabaler")
            Self::check_suffix_vowel_harmony(tok, &lower, start, end, morphology, findings);
        }
    }

    /// 1. Consonant Assimilation / Hardening:
    /// When a word ends with a voiceless consonant (f, s, t, k, ç, ş, h, p),
    /// suffixes starting with c/d must harden to ç/t (*kitap-da* -> *kitapta*, *sokak-dan* -> *sokaktan*).
    fn check_consonant_assimilation(
        original: &str,
        lower: &str,
        start: usize,
        end: usize,
        morphology: &TurkishMorphology,
        findings: &mut Vec<GrammarFinding>,
    ) -> bool {
        // Look for unassimilated suffix patterns: "da", "de", "dan", "den", "dı", "di", "du", "dü", "dır", "dir", "dur", "dür", "ce", "ca"
        let unassimilated_suffixes = [
            ("da", "ta"), ("de", "te"),
            ("dan", "tan"), ("den", "ten"),
            ("dı", "tı"), ("di", "ti"), ("du", "tu"), ("dü", "tü"),
            ("dır", "tır"), ("dir", "tir"), ("dur", "tur"), ("dür", "tür"),
            ("ca", "ça"), ("ce", "çe"),
        ];

        for (bad_suf, good_suf) in &unassimilated_suffixes {
            if lower.ends_with(bad_suf) && lower.len() > bad_suf.len() + 2 {
                let stem = &lower[..lower.len() - bad_suf.len()];
                if let Some(stem_last_char) = stem.chars().last() {
                    if is_hard_consonant(stem_last_char) {
                        // Validate that stem is a real Turkish root/stem
                        let parses = morphology.analyze(stem);
                        if !parses.is_empty() {
                            let fixed_lower = format!("{}{}", stem, good_suf);
                            // Preserve original casing
                            let replacement = if original.chars().next().map_or(false, |c| c.is_uppercase()) {
                                crate::phonology::to_turkish_title(&fixed_lower)
                            } else {
                                fixed_lower
                            };

                            findings.push(GrammarFinding {
                                category: ErrorCategory::ConsonantAssimilation,
                                start_offset: start,
                                end_offset: end,
                                original_text: original.to_string(),
                                replacement,
                                message_tr: "Sert ünsüzle (f, s, t, k, ç, ş, h, p) biten sözcüklere gelen ekler sertleşir (ünsüz benzeşmesi).".to_string(),
                                message_en: "Suffixes following voiceless consonants must undergo consonant hardening.".to_string(),
                                confidence: 0.98,
                            });
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    /// 2. Vowel Dropping (Syncope):
    /// Words like *akıl*, *burun*, *şehir* drop the second vowel when receiving a vowel-initial suffix (*akıl-ı* -> *aklı*).
    fn check_vowel_drop(
        original: &str,
        lower: &str,
        start: usize,
        end: usize,
        _morphology: &TurkishMorphology,
        findings: &mut Vec<GrammarFinding>,
    ) -> bool {
        for (&root, &contracted) in VOWEL_DROP_ROOTS.iter() {
            if lower.starts_with(root) && lower.len() > root.len() {
                let suf = &lower[root.len()..];
                if let Some(first_suf_char) = suf.chars().next() {
                    if is_turkish_vowel(first_suf_char) {
                        let fixed_lower = format!("{}{}", contracted, suf);
                        let replacement = if original.chars().next().map_or(false, |c| c.is_uppercase()) {
                            crate::phonology::to_turkish_title(&fixed_lower)
                        } else {
                            fixed_lower
                        };

                        findings.push(GrammarFinding {
                            category: ErrorCategory::VowelDropping,
                            start_offset: start,
                            end_offset: end,
                            original_text: original.to_string(),
                            replacement,
                            message_tr: format!("'{}' sözcüğü ünlüyle başlayan ek aldığında ikinci hecedeki dar ünlü düşer (ünlü düşmesi).", root),
                            message_en: format!("The second vowel in '{}' drops when attached to a vowel-initial suffix.", root),
                            confidence: 0.99,
                        });
                        return true;
                    }
                }
            }
        }
        false
    }

    /// 3. Vowel Narrowing:
    /// Progressive tense "-yor" narrows preceding open vowels 'a'/'e' (*başla-yor* -> *başlıyor*, *gözle-yor* -> *gözlüyor*).
    fn check_vowel_narrowing(
        original: &str,
        lower: &str,
        start: usize,
        end: usize,
        morphology: &TurkishMorphology,
        findings: &mut Vec<GrammarFinding>,
    ) -> bool {
        if lower.ends_with("ayor") || lower.ends_with("eyor") {
            let stem = &lower[..lower.len() - 4];
            let ending = &lower[lower.len() - 4..];
            if !stem.is_empty() {
                let verb_root = format!("{}{}", stem, if ending.starts_with('a') { "a" } else { "e" });
                let parses = morphology.analyze(&verb_root);
                if parses.iter().any(|p| p.primary_pos == crate::morphology::pos::PrimaryPos::Verb) {
                    let last_v = last_vowel(stem).unwrap_or('a');
                    let narrow_v = harmony_i_type(last_v);
                    let fixed_lower = format!("{}{}yor", stem, narrow_v);
                    let replacement = if original.chars().next().map_or(false, |c| c.is_uppercase()) {
                        crate::phonology::to_turkish_title(&fixed_lower)
                    } else {
                        fixed_lower
                    };

                    findings.push(GrammarFinding {
                        category: ErrorCategory::VowelHarmony,
                        start_offset: start,
                        end_offset: end,
                        original_text: original.to_string(),
                        replacement,
                        message_tr: "'-yor' şimdiki zaman eki kendinden önceki 'a/e' geniş ünlülerini daraltır (ünlü daralması).".to_string(),
                        message_en: "The progressive suffix '-yor' narrows preceding open vowels 'a/e'.".to_string(),
                        confidence: 0.98,
                    });
                    return true;
                }
            }
        }
        false
    }

    /// 4. Consonant Softening / Voicing:
    /// When words ending in p, ç, t, k take a vowel-initial suffix, they mutate to b, c, d, ğ/g (*kitap-ı* -> *kitabı*).
    fn check_consonant_softening(
        original: &str,
        lower: &str,
        start: usize,
        end: usize,
        _morphology: &TurkishMorphology,
        findings: &mut Vec<GrammarFinding>,
    ) -> bool {
        let softenable_roots = [
            ("kitap", "kitab"), ("ağaç", "ağac"), ("kanat", "kanad"), ("renk", "reng"),
            ("çocuk", "çocuğ"), ("ayak", "ayağ"), ("ekmek", "ekmeğ"), ("köpek", "köpeğ"),
            ("dolap", "dolab"), ("kalp", "kalb"), ("hesap", "hesab"), ("cevap", "cevab"),
            ("kulak", "kulağ"), ("bacak", "bacağ"), ("çiçek", "çiçeğ"), ("tarak", "tarağ"),
        ];

        for (unmutated, mutated) in &softenable_roots {
            if lower.starts_with(unmutated) && lower.len() > unmutated.len() {
                let suf = &lower[unmutated.len()..];
                if let Some(first_suf_char) = suf.chars().next() {
                    if is_turkish_vowel(first_suf_char) {
                        let fixed_lower = format!("{}{}", mutated, suf);
                        let replacement = if original.chars().next().map_or(false, |c| c.is_uppercase()) {
                            crate::phonology::to_turkish_title(&fixed_lower)
                        } else {
                            fixed_lower
                        };

                        findings.push(GrammarFinding {
                            category: ErrorCategory::ConsonantSoftening,
                            start_offset: start,
                            end_offset: end,
                            original_text: original.to_string(),
                            replacement,
                            message_tr: format!("'{}' sözcüğü ünlüyle başlayan ek aldığında sonundaki ünsüz yumuşar (ünsüz yumuşaması).", unmutated),
                            message_en: format!("The final consonant in '{}' undergoes voicing when a vowel-initial suffix is added.", unmutated),
                            confidence: 0.98,
                        });
                        return true;
                    }
                }
            }
        }
        false
    }

    /// 5. Suffix Vowel Harmony (Major/Minor Vowel Harmony):
    /// Suffixes must match backness/frontness of the stem (*araba-ler* -> *arabalar*, *ev-lar* -> *evler*).
    fn check_suffix_vowel_harmony(
        original: &str,
        lower: &str,
        start: usize,
        end: usize,
        morphology: &TurkishMorphology,
        findings: &mut Vec<GrammarFinding>,
    ) {
        // Only check words unrecognized by morphology (potential harmony typo)
        let parses = morphology.analyze(lower);
        if !parses.is_empty() {
            return;
        }

        // Test plural suffix mismatch (-ler vs -lar)
        if lower.ends_with("ler") || lower.ends_with("lar") {
            let stem = &lower[..lower.len() - 3];
            if let Some(stem_v) = last_vowel(stem) {
                let expected_pl = if is_front_vowel(stem_v) { "ler" } else { "lar" };
                let current_pl = &lower[lower.len() - 3..];
                if current_pl != expected_pl {
                    let fixed_lower = format!("{}{}", stem, expected_pl);
                    if !morphology.analyze(&fixed_lower).is_empty() {
                        let replacement = if original.chars().next().map_or(false, |c| c.is_uppercase()) {
                            crate::phonology::to_turkish_title(&fixed_lower)
                        } else {
                            fixed_lower
                        };
                        findings.push(GrammarFinding {
                            category: ErrorCategory::VowelHarmony,
                            start_offset: start,
                            end_offset: end,
                            original_text: original.to_string(),
                            replacement,
                            message_tr: format!("Çoğul eki büyük ünlü uyumuna göre '-{}' olmalıdır.", expected_pl),
                            message_en: format!("The plural suffix must be '-{}' to satisfy vowel harmony.", expected_pl),
                            confidence: 0.97,
                        });
                    }
                }
            }
        }
    }
}
