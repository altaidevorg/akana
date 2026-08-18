//! SIMD-Accelerated Turkish Orthographic and Clitic Error Detection and Correction.
//!
//! Covers:
//! 1. `de/da` Conjunction (separate) vs Locative Suffix (attached) and incorrect consonant assimilation (*gitse te* -> *gitse de*).
//! 2. `ki` Conjunction (separate, with SOMBAHÇEMİ exception list) vs Relative/Possessive Suffix (attached).
//! 3. `mi/mı/mu/mü` Question Particle (separate, 4-way vowel harmony).
//! 4. Proper noun & numeric apostrophes (*Ankara'da*, *1923'te*, *2'nci*, *Ahmetlerin*).
//! 5. Sentence-initial & Proper Noun Capitalization.
//! 6. Reduplication orthography (*el ele*, *yan yana*).

use std::collections::HashSet;
use lazy_static::lazy_static;
use stringzilla::StringZilla;

use super::{ErrorCategory, GrammarFinding};
use crate::morphology::pos::PrimaryPos;
use crate::morphology::TurkishMorphology;
use crate::phonology::{
    harmony_i_type, is_front_vowel, last_vowel, to_turkish_lower, to_turkish_title,
};

lazy_static! {
    /// Exception list for `ki` conjunctions written attached (SOMBAHÇEMİ rule):
    /// Sanki, Oysaki, Mademki, Belki, Halbuki, Çünkü, Meğerki, İllaki
    static ref SOMBAHCEMI_EXCEPTIONS: HashSet<&'static str> = {
        let mut s = HashSet::new();
        s.insert("sanki");
        s.insert("oysaki");
        s.insert("mademki");
        s.insert("belki");
        s.insert("halbuki");
        s.insert("çünkü");
        s.insert("meğerki");
        s.insert("illaki");
        s
    };

    /// Common Turkish reduplications that must be written separately
    static ref SEPARATE_REDUPLICATIONS: HashSet<&'static str> = {
        let mut s = HashSet::new();
        s.insert("elele");
        s.insert("yanyana");
        s.insert("gözgöze");
        s.insert("baştanbaşa");
        s.insert("adımadım");
        s.insert("gündengüne");
        s.insert("peşpeşe");
        s.insert("sırasırasına");
        s.insert("artarda");
        s.insert("içiçe");
        s.insert("üstüste");
        s.insert("arkaarkaya");
        s.insert("ardarda");
        s.insert("birebir");
        s
    };
}

pub struct OrthographyRuleEngine;

impl OrthographyRuleEngine {
    /// Checks all surface orthographic, clitic, and capitalization rules across a sentence.
    pub fn check(
        text: &str,
        tokens: &[&str],
        token_spans: &[(usize, usize)],
        morphology: &TurkishMorphology,
        findings: &mut Vec<GrammarFinding>,
    ) {
        if tokens.is_empty() {
            return;
        }

        Self::check_capitalization(text, tokens, token_spans, findings);
        Self::check_question_particle_mi(text, tokens, token_spans, findings);
        Self::check_clitic_de_da(text, tokens, token_spans, morphology, findings);
        Self::check_clitic_ki(text, tokens, token_spans, morphology, findings);
        Self::check_apostrophes(text, tokens, token_spans, morphology, findings);
        Self::check_reduplications(text, tokens, token_spans, findings);
    }

    /// 1. Sentence-initial & Proper Noun Capitalization
    fn check_capitalization(
        _text: &str,
        tokens: &[&str],
        token_spans: &[(usize, usize)],
        findings: &mut Vec<GrammarFinding>,
    ) {
        // Sentence first word must start with uppercase
        if let Some(&first_tok) = tokens.first() {
            if let Some(first_char) = first_tok.chars().next() {
                if first_char.is_alphabetic() && first_char.is_lowercase() {
                    let (start, end) = token_spans[0];
                    let replacement = to_turkish_title(first_tok);
                    findings.push(GrammarFinding {
                        category: ErrorCategory::Capitalization,
                        start_offset: start,
                        end_offset: end,
                        original_text: first_tok.to_string(),
                        replacement,
                        message_tr: "Cümle büyük harfle başlamalıdır.".to_string(),
                        message_en: "Sentences must begin with a capital letter.".to_string(),
                        confidence: 0.99,
                    });
                }
            }
        }
    }

    /// 2. `mi/mı/mu/mü` Question Particle Rules
    fn check_question_particle_mi(
        _text: &str,
        tokens: &[&str],
        token_spans: &[(usize, usize)],
        findings: &mut Vec<GrammarFinding>,
    ) {
        let mi_particles = ["mi", "mı", "mu", "mü"];
        let mi_suffixes = [
            "misin", "mısın", "musun", "müsün", "miyiz", "mıyız", "muyuz", "müyüz",
            "misiniz", "mısınız", "musunuz", "müsünüz", "miler", "mılar",
        ];

        for (i, &tok) in tokens.iter().enumerate() {
            let lower = to_turkish_lower(tok);
            let (start, end) = token_spans[i];

            // Case A: Merged question particle at the end of word (e.g. "geldinmi", "biliyormusun")
            for &p in &mi_particles {
                if lower.ends_with(p) && lower.len() > p.len() + 2 {
                    let stem = &lower[..lower.len() - p.len()];
                    if let Some(stem_last_vowel) = last_vowel(stem) {
                        let expected_vowel = harmony_i_type(stem_last_vowel);
                        let expected_p = match expected_vowel {
                            'ı' => "mı",
                            'i' => "mi",
                            'u' => "mu",
                            'ü' => "mü",
                            _ => "mi",
                        };

                        if p == expected_p && (stem.ends_with('r') || stem.ends_with('n') || stem.ends_with('k') || stem.ends_with('z') || stem.ends_with('d') || stem.ends_with('t') || stem.ends_with('m') || stem.ends_with('ş')) {
                            let original_stem = &tok[..tok.len() - p.len()];
                            let replacement = format!("{} {}", original_stem, expected_p);
                            findings.push(GrammarFinding {
                                category: ErrorCategory::ParticleMi,
                                start_offset: start,
                                end_offset: end,
                                original_text: tok.to_string(),
                                replacement,
                                message_tr: "Soru eki olan 'mı/mi/mu/mü' kendinden önceki sözcükten ayrı yazılır.".to_string(),
                                message_en: "The question particle 'mı/mi/mu/mü' must be written separately.".to_string(),
                                confidence: 0.98,
                            });
                            break;
                        }
                    }
                }
            }

            // Case A2: Merged question particle with person suffix (e.g. "biliyormusun")
            for &suf in &mi_suffixes {
                if lower.ends_with(suf) && lower.len() > suf.len() + 2 {
                    let original_stem = &tok[..tok.len() - suf.len()];
                    let replacement = format!("{} {}", original_stem, suf);
                    findings.push(GrammarFinding {
                        category: ErrorCategory::ParticleMi,
                        start_offset: start,
                        end_offset: end,
                        original_text: tok.to_string(),
                        replacement,
                        message_tr: "Soru eki ve eklenen şahıs ekleri ayrı yazılır.".to_string(),
                        message_en: "Question particles and person suffixes must be written separately.".to_string(),
                        confidence: 0.98,
                    });
                    break;
                }
            }

            // Case B: Standalone question particle with vowel harmony error (e.g., "gitti mı" -> "gitti mi")
            if (lower == "mi" || lower == "mı" || lower == "mu" || lower == "mü") && i > 0 {
                let prev_tok = tokens[i - 1];
                if let Some(prev_vowel) = last_vowel(&to_turkish_lower(prev_tok)) {
                    let expected_vowel = harmony_i_type(prev_vowel);
                    let expected_p = match expected_vowel {
                        'ı' => "mı",
                        'i' => "mi",
                        'u' => "mu",
                        'ü' => "mü",
                        _ => "mi",
                    };
                    if lower != expected_p {
                        findings.push(GrammarFinding {
                            category: ErrorCategory::ParticleMi,
                            start_offset: start,
                            end_offset: end,
                            original_text: tok.to_string(),
                            replacement: expected_p.to_string(),
                            message_tr: format!("Soru eki ünlü uyumuna göre '{}' olmalıdır.", expected_p),
                            message_en: format!("The question particle should be '{}' according to vowel harmony.", expected_p),
                            confidence: 0.99,
                        });
                    }
                }
            }
        }
    }

    /// 3. `de/da` Clitic Rules
    fn check_clitic_de_da(
        _text: &str,
        tokens: &[&str],
        token_spans: &[(usize, usize)],
        morphology: &TurkishMorphology,
        findings: &mut Vec<GrammarFinding>,
    ) {
        for (i, &tok) in tokens.iter().enumerate() {
            let lower = to_turkish_lower(tok);
            let (start, end) = token_spans[i];

            // Case A: Standalone "te" or "ta"
            if lower == "te" || lower == "ta" {
                if i > 0 {
                    let prev_tok = tokens[i - 1];
                    let prev_lower = to_turkish_lower(prev_tok);
                    let correct_clitic = if let Some(v) = last_vowel(&prev_lower) {
                        if is_front_vowel(v) { "de" } else { "da" }
                    } else {
                        "de"
                    };
                    findings.push(GrammarFinding {
                        category: ErrorCategory::CliticDeDa,
                        start_offset: start,
                        end_offset: end,
                        original_text: tok.to_string(),
                        replacement: correct_clitic.to_string(),
                        message_tr: "Bağlaç olan 'de/da' ayrı yazılır ve asla 'te/ta' şeklinde sertleşmez.".to_string(),
                        message_en: "The conjunction 'de/da' is written separately and never undergoes consonant hardening to 'te/ta'.".to_string(),
                        confidence: 0.99,
                    });
                }
            }

            // Case B: Attached "de/da" on finite verbs (e.g. "gittide", "bilsede", "yaparda", "gelmişde")
            if lower.ends_with("de") || lower.ends_with("da") || lower.ends_with("te") || lower.ends_with("ta") {
                if lower.len() > 4 {
                    let stem = &lower[..lower.len() - 2];
                    let parses = morphology.analyze(stem);
                    // Check if stem is a finite verb with clear tense marker (Past, Fut, Aor, Cond, Prog)
                    // and ensure it is not a common noun root that happens to have a rare optative verb parse (e.g. sok-a-k -> sokak)
                    let is_common_noun = parses.iter().any(|p| p.primary_pos == PrimaryPos::Noun && (p.morpheme_tags.is_empty() || p.morpheme_tags == vec!["Noun".to_string()]));
                    let is_finite_verb = !is_common_noun && parses.iter().any(|p| {
                        p.primary_pos == PrimaryPos::Verb && (
                            p.morpheme_tags.iter().any(|t| t.contains("Past") || t.contains("Pres") || t.contains("Fut") || t.contains("Aor") || t.contains("Cond"))
                        )
                    });

                    if is_finite_verb {
                        let original_stem = &tok[..tok.len() - 2];
                        let correct_clitic = if let Some(v) = last_vowel(stem) {
                            if is_front_vowel(v) { "de" } else { "da" }
                        } else {
                            "de"
                        };
                        let replacement = format!("{} {}", original_stem, correct_clitic);
                        findings.push(GrammarFinding {
                            category: ErrorCategory::CliticDeDa,
                            start_offset: start,
                            end_offset: end,
                            original_text: tok.to_string(),
                            replacement,
                            message_tr: "Fiillerden sonra gelen 'de/da' bağlaçtır ve ayrı yazılır.".to_string(),
                            message_en: "The clitic 'de/da' following finite verbs is a conjunction and must be written separately.".to_string(),
                            confidence: 0.96,
                        });
                    }
                }
            }
        }
    }

    /// 4. `ki` Clitic Rules
    fn check_clitic_ki(
        _text: &str,
        tokens: &[&str],
        token_spans: &[(usize, usize)],
        morphology: &TurkishMorphology,
        findings: &mut Vec<GrammarFinding>,
    ) {
        for (i, &tok) in tokens.iter().enumerate() {
            let lower = to_turkish_lower(tok);
            let (start, end) = token_spans[i];

            // Case A: SOMBAHÇEMİ words erroneously written separately (e.g. "san ki", "oysa ki", "madem ki")
            if lower == "ki" && i > 0 {
                let prev_tok = to_turkish_lower(tokens[i - 1]);
                let compound = format!("{}{}", prev_tok, "ki");
                if SOMBAHCEMI_EXCEPTIONS.contains(compound.as_str()) {
                    let prev_start = token_spans[i - 1].0;
                    let is_upper = tokens[i - 1].chars().next().map_or(false, |c| c.is_uppercase());
                    let replacement = if is_upper {
                        to_turkish_title(&compound)
                    } else {
                        compound.clone()
                    };

                    findings.push(GrammarFinding {
                        category: ErrorCategory::CliticKi,
                        start_offset: prev_start,
                        end_offset: end,
                        original_text: format!("{} {}", tokens[i - 1], tok),
                        replacement,
                        message_tr: format!("'{}' sözcüğündeki 'ki' kalıplaşmış olduğu için bitişik yazılır.", compound),
                        message_en: format!("'{}' is a lexicalized exception and must be written attached.", compound),
                        confidence: 0.99,
                    });
                }
            }

            // Case B: Attached "ki" on verbs (e.g. "duydumki", "gördümki", "biliyorki")
            if lower.ends_with("ki") && lower.len() > 4 && !SOMBAHCEMI_EXCEPTIONS.contains(lower.as_str()) {
                let stem = &lower[..lower.len() - 2];
                let parses = morphology.analyze(stem);
                let is_verb = parses.iter().any(|p| p.primary_pos == PrimaryPos::Verb);

                if is_verb {
                    let original_stem = &tok[..tok.len() - 2];
                    let replacement = format!("{} ki", original_stem);
                    findings.push(GrammarFinding {
                        category: ErrorCategory::CliticKi,
                        start_offset: start,
                        end_offset: end,
                        original_text: tok.to_string(),
                        replacement,
                        message_tr: "Fiillerden sonra gelen 'ki' bağlaçtır ve ayrı yazılır.".to_string(),
                        message_en: "The clitic 'ki' following verbs is a conjunction and must be written separately.".to_string(),
                        confidence: 0.97,
                    });
                }
            }
        }
    }

    /// 5. Apostrophe & Proper Noun Rules
    fn check_apostrophes(
        text: &str,
        _tokens: &[&str],
        _token_spans: &[(usize, usize)],
        _morphology: &TurkishMorphology,
        findings: &mut Vec<GrammarFinding>,
    ) {
        // Full-text StringZilla SIMD search for double ordinal suffixes on numbers (e.g. "2.'nci" -> "2'nci")
        let ordinal_double_patterns = [".'nci", ".'ncı", ".'ncu", ".'ncü", ".'inci", ".'ıncı", ".'uncu", ".'üncü"];
        for &pat in &ordinal_double_patterns {
            if let Some(offset) = text.sz_find(pat) {
                let prefix = &text[..offset];
                let digit_start = prefix.rfind(|c: char| !c.is_ascii_digit()).map(|i| i + 1).unwrap_or(0);
                let num_str = &prefix[digit_start..];
                if !num_str.is_empty() {
                    let end = offset + pat.len();
                    let original_snippet = &text[digit_start..end];
                    let fixed_suf = pat.replace('.', "");
                    let replacement = format!("{}{}", num_str, fixed_suf);
                    findings.push(GrammarFinding {
                        category: ErrorCategory::ApostropheNumberDate,
                        start_offset: digit_start,
                        end_offset: end,
                        original_text: original_snippet.to_string(),
                        replacement,
                        message_tr: "Sıra sayıları yazılırken hem nokta hem de kesme eki birlikte kullanılmaz.".to_string(),
                        message_en: "Both period and ordinal suffix should not be used together on numbers.".to_string(),
                        confidence: 0.99,
                    });
                }
            }
        }

        // Full-text StringZilla scan for all apostrophes
        let mut search_idx = 0;
        while let Some(apos_rel) = text[search_idx..].sz_find("'") {
            let apos_pos = search_idx + apos_rel;
            let prefix = &text[..apos_pos];
            let suffix = &text[apos_pos + 1..];

            let start = prefix.rfind(|c: char| c.is_whitespace() || c == '.' || c == ',' || c == '(' || c == '"' || c == '!').map(|i| i + 1).unwrap_or(0);
            let root = &prefix[start..];

            let suf_len = suffix.find(|c: char| !c.is_alphabetic()).unwrap_or(suffix.len());
            let suf = &suffix[..suf_len];
            let end = apos_pos + 1 + suf_len;

            if !root.is_empty() && !suf.is_empty() {
                // A. Number apostrophe assimilation (e.g. 1923'de -> 1923'te)
                if root.chars().all(|c| c.is_ascii_digit()) {
                    let last_digit = root.chars().last().unwrap();
                    let suf_lower = to_turkish_lower(suf);
                    let is_voiceless_digit = matches!(last_digit, '3' | '4' | '5');

                    if is_voiceless_digit {
                        if suf_lower.starts_with("de") || suf_lower.starts_with("da") || suf_lower.starts_with("den") || suf_lower.starts_with("dan") {
                            let hardened_suf = if suf_lower.starts_with("de") {
                                format!("te{}", &suf[2..])
                            } else if suf_lower.starts_with("da") {
                                format!("ta{}", &suf[2..])
                            } else if suf_lower.starts_with("den") {
                                format!("ten{}", &suf[3..])
                            } else {
                                format!("tan{}", &suf[3..])
                            };
                            let replacement = format!("{}'{}", root, hardened_suf);
                            findings.push(GrammarFinding {
                                category: ErrorCategory::ApostropheNumberDate,
                                start_offset: start,
                                end_offset: end,
                                original_text: text[start..end].to_string(),
                                replacement,
                                message_tr: "Sayılara gelen ekler son rakamın okunuşundaki sert ünsüze uymalıdır (ünsüz sertleşmesi).".to_string(),
                                message_en: "Suffixes attached to numbers must follow consonant assimilation.".to_string(),
                                confidence: 0.99,
                            });
                        }
                    }
                } else {
                    // B. Plural / derivational on proper noun (e.g. Ahmetler'in, Türk'ler)
                    let root_lower = to_turkish_lower(root);
                    let suf_lower = to_turkish_lower(suf);
                    let is_plural_or_deriv_root = root_lower.ends_with("ler") || root_lower.ends_with("lar") || root_lower.ends_with("li") || root_lower.ends_with("lı") || root_lower.ends_with("lu") || root_lower.ends_with("lü") || root_lower.ends_with("lik") || root_lower.ends_with("lık") || root_lower.ends_with("luk") || root_lower.ends_with("lük") || root_lower.ends_with("gil");
                    let is_plural_or_deriv_suffix = suf_lower.starts_with("ler") || suf_lower.starts_with("lar") || suf_lower.starts_with("li") || suf_lower.starts_with("lı") || suf_lower.starts_with("lu") || suf_lower.starts_with("lü") || suf_lower.starts_with("siz") || suf_lower.starts_with("sız") || suf_lower.starts_with("suz") || suf_lower.starts_with("süz") || suf_lower.starts_with("lik") || suf_lower.starts_with("lık") || suf_lower.starts_with("luk") || suf_lower.starts_with("lük");

                    if is_plural_or_deriv_root || is_plural_or_deriv_suffix {
                        let replacement = format!("{}{}", root, suf);
                        findings.push(GrammarFinding {
                            category: ErrorCategory::ApostropheProperNoun,
                            start_offset: start,
                            end_offset: end,
                            original_text: text[start..end].to_string(),
                            replacement,
                            message_tr: "Özel isimlere gelen yapım ekleri, çokluk eki (-ler/-lar) ve bunlardan sonra gelen ekler kesme işaretiyle ayrılmaz.".to_string(),
                            message_en: "Plural and derivational suffixes on proper nouns (and following suffixes) do not take an apostrophe.".to_string(),
                            confidence: 0.99,
                        });
                    }
                }
            }

            search_idx = apos_pos + 1;
        }
    }

    /// 6. Reduplications (İkilemeler): Must be written separately (*elele* -> *el ele*, *yanyana* -> *yan yana*)
    fn check_reduplications(
        _text: &str,
        tokens: &[&str],
        token_spans: &[(usize, usize)],
        findings: &mut Vec<GrammarFinding>,
    ) {
        for (i, &tok) in tokens.iter().enumerate() {
            let lower = to_turkish_lower(tok);
            let (start, end) = token_spans[i];

            if SEPARATE_REDUPLICATIONS.contains(lower.as_str()) {
                let split_word = match lower.as_str() {
                    "elele" => "el ele",
                    "yanyana" => "yan yana",
                    "gözgöze" => "göz göze",
                    "baştanbaşa" => "baştan başa",
                    "adımadım" => "adım adım",
                    "gündengüne" => "günden güne",
                    "peşpeşe" => "peş peşe",
                    "sırasırasına" => "sıra sırasına",
                    "artarda" | "ardarda" => "art arda",
                    "içiçe" => "iç içe",
                    "üstüste" => "üst üste",
                    "arkaarkaya" => "arka arkaya",
                    _ => "",
                };

                if !split_word.is_empty() {
                    findings.push(GrammarFinding {
                        category: ErrorCategory::ReduplicationOrthography,
                        start_offset: start,
                        end_offset: end,
                        original_text: tok.to_string(),
                        replacement: split_word.to_string(),
                        message_tr: "İkilemeler ayrı yazılır.".to_string(),
                        message_en: "Reduplications must be written separately.".to_string(),
                        confidence: 0.99,
                    });
                }
            }
        }
    }
}
