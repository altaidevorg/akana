//! Turkish Lexicon, Compound Verb, and Tautology / Redundancy Rules.
//!
//! Covers:
//! 1. Auxiliary Compound Verbs (*terketmek* -> *terk etmek*, *hiss etmek* -> *hissetmek*).
//! 2. Tautology and Pleonasm Detection (*henüz hala*, *birlikte beraber*, *geri iade*).

use std::collections::HashMap;
use lazy_static::lazy_static;
use stringzilla::StringZilla;

use super::{ErrorCategory, GrammarFinding};
use crate::phonology::to_turkish_lower;

lazy_static! {
    /// Compound verbs that MUST be written separately (no phonological mutation):
    /// terketmek -> terk etmek, farketmek -> fark etmek, arzetmek -> arz etmek, vs.
    static ref SEPARATE_COMPOUND_VERBS: HashMap<&'static str, &'static str> = {
        let mut m = HashMap::new();
        m.insert("terketmek", "terk etmek");
        m.insert("terketti", "terk etti");
        m.insert("terkettiler", "terk ettiler");
        m.insert("farketmek", "fark etmek");
        m.insert("farkettim", "fark ettim");
        m.insert("farketti", "fark etti");
        m.insert("ayırdetmek", "ayırt etmek");
        m.insert("arzetmek", "arz etmek");
        m.insert("sağol", "sağ ol");
        m.insert("sağolun", "sağ olun");
        m.insert("yokolmak", "yok olmak");
        m.insert("varketmek", "var etmek");
        m.insert("haketmek", "hak etmek");
        m.insert("dahilolmak", "dahil olmak");
        m.insert("yardımetmek", "yardım etmek");
        m.insert("teslimolmak", "teslim olmak");
        m.insert("hayranolmak", "hayran olmak");
        m
    };

    /// Compound verbs that MUST be written attached (due to vowel drop or consonant doubling):
    /// hiss etmek -> hissetmek, red etmek -> reddetmek, kayıp olmak -> kaybolmak, vs.
    static ref ATTACHED_COMPOUND_VERBS: HashMap<&'static str, &'static str> = {
        let mut m = HashMap::new();
        m.insert("hiss etmek", "hissetmek");
        m.insert("hiss etti", "hissetti");
        m.insert("red etmek", "reddetmek");
        m.insert("red etti", "reddetti");
        m.insert("af etmek", "affetmek");
        m.insert("af etti", "affetti");
        m.insert("kayıp olmak", "kaybolmak");
        m.insert("kayıp oldu", "kayboldu");
        m.insert("zehir olmak", "zehrolmak");
        m.insert("zehir oldu", "zehroldu");
        m.insert("hüküm etmek", "hükmetmek");
        m.insert("hüküm etti", "hükmetti");
        m.insert("sabır etmek", "sabretmek");
        m.insert("sabır etti", "sabretti");
        m.insert("devir etmek", "devretmek");
        m.insert("devir etti", "devretti");
        m.insert("hal olmak", "hallolmak");
        m.insert("hal oldu", "halloldu");
        m
    };

    /// Common Turkish tautological / redundant phrase pairs
    static ref TAUTOLOGY_PAIRS: HashMap<&'static str, (&'static str, &'static str)> = {
        let mut m = HashMap::new();
        m.insert("henüz hala", ("henüz", "Eşanlamlı veya çelişkili zarfların bir arada kullanımı gereksiz sözcük tekrarıdır."));
        m.insert("hala henüz", ("hâlâ", "Gereksiz sözcük tekrarı."));
        m.insert("birlikte beraber", ("birlikte", "Gereksiz sözcük tekrarı."));
        m.insert("beraber birlikte", ("beraber", "Gereksiz sözcük tekrarı."));
        m.insert("ilk başlangıç", ("başlangıç", "Gereksiz sözcük kullanımı."));
        m.insert("geri iade", ("iade", "'İade etmek' zaten geri vermektir; 'geri' sözcüğü gereksizdir."));
        m.insert("kendi öz", ("öz", "Gereksiz sözcük kullanımı."));
        m.insert("şüphesiz kuşkusuz", ("şüphesiz", "Gereksiz sözcük tekrarı."));
        m.insert("aynen tamamen", ("tamamen", "Gereksiz sözcük tekrarı."));
        m
    };
}

pub struct LexiconRuleEngine;

impl LexiconRuleEngine {
    pub fn check(
        text: &str,
        tokens: &[&str],
        token_spans: &[(usize, usize)],
        findings: &mut Vec<GrammarFinding>,
    ) {
        Self::check_compound_verbs(tokens, token_spans, findings);
        Self::check_tautologies(text, findings);
    }

    /// 1. Compound Verbs:
    /// Validates correct separate vs attached spelling based on sound change laws.
    fn check_compound_verbs(
        tokens: &[&str],
        token_spans: &[(usize, usize)],
        findings: &mut Vec<GrammarFinding>,
    ) {
        let n = tokens.len();

        // Check single tokens that should be separate (e.g. "terketmek", "farkettim")
        for (i, &tok) in tokens.iter().enumerate() {
            let lower = to_turkish_lower(tok);
            let (start, end) = token_spans[i];

            if let Some(&correct_form) = SEPARATE_COMPOUND_VERBS.get(lower.as_str()) {
                findings.push(GrammarFinding {
                    category: ErrorCategory::CompoundWordOrthography,
                    start_offset: start,
                    end_offset: end,
                    original_text: tok.to_string(),
                    replacement: correct_form.to_string(),
                    message_tr: "Ses düşmesi veya türemesi olmayan birleşik fiiller ayrı yazılır.".to_string(),
                    message_en: "Compound auxiliary verbs without phonological mutation must be written separately.".to_string(),
                    confidence: 0.99,
                });
            }
        }

        // Check two-token pairs that should be attached (e.g. "hiss etmek", "kayıp olmak")
        for i in 0..n {
            if i + 1 < n {
                let pair = format!("{} {}", to_turkish_lower(tokens[i]), to_turkish_lower(tokens[i + 1]));
                if let Some(&attached_form) = ATTACHED_COMPOUND_VERBS.get(pair.as_str()) {
                    let start = token_spans[i].0;
                    let end = token_spans[i + 1].1;
                    findings.push(GrammarFinding {
                        category: ErrorCategory::CompoundWordOrthography,
                        start_offset: start,
                        end_offset: end,
                        original_text: format!("{} {}", tokens[i], tokens[i + 1]),
                        replacement: attached_form.to_string(),
                        message_tr: "Ses düşmesi veya ünsüz türemesine uğrayan birleşik fiiller bitişik yazılır.".to_string(),
                        message_en: "Compound auxiliary verbs with sound changes must be written attached.".to_string(),
                        confidence: 0.99,
                    });
                }
            }
        }
    }

    /// 2. Tautologies & Pleonasms (Accelerated with StringZilla SIMD find):
    fn check_tautologies(
        text: &str,
        findings: &mut Vec<GrammarFinding>,
    ) {
        let text_lower = to_turkish_lower(text);

        for (&bad_phrase, &(replacement, explanation)) in TAUTOLOGY_PAIRS.iter() {
            if let Some(offset) = text_lower.as_str().sz_find(bad_phrase) {
                let end = offset + bad_phrase.len();
                let original_snippet = &text[offset..end];

                findings.push(GrammarFinding {
                    category: ErrorCategory::TautologyRedundancy,
                    start_offset: offset,
                    end_offset: end,
                    original_text: original_snippet.to_string(),
                    replacement: replacement.to_string(),
                    message_tr: explanation.to_string(),
                    message_en: "Tautological redundancy in phrase pairing.".to_string(),
                    confidence: 0.98,
                });
            }
        }
    }
}
