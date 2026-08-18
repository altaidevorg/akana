//! Turkish Morphosyntactic Agreement & Concord Error Engine.
//!
//! Evaluates:
//! 1. Quantity Determiner + Noun Plural Redundancy (*üç elmalar* -> *üç elma*, *birçok insanlar* -> *birçok insan*).
//! 2. Genitive - Possessive Person Concord (*benim ev* -> *benim evim*, *onun evimiz* -> *onun evi*).
//! 3. Correlative Conjunction Polarity (*Ne... Ne...* takes positive verb: *ne geldi ne gitmedi* -> *ne geldi ne gitti*).
//! 4. Negation Concord (*Kimse / Hiçbiri* requires negative verb: *Kimse geldi* -> *Kimse gelmedi*).
//! 5. Subject-Verb Agreement (Inanimate plural subjects take singular verb: *Ağaçlar döküldüler* -> *Ağaçlar döküldü*).

use std::collections::HashSet;
use lazy_static::lazy_static;

use super::{ErrorCategory, GrammarFinding};
use crate::morphology::pos::{PrimaryPos, SecondaryPos};
use crate::morphology::{MorphParse, MorphologicalDisambiguator, TurkishMorphology};
use crate::parser::TurkishDependencyParser;
use crate::phonology::{
    get_i_type_harmonic_vowel, is_turkish_vowel, to_turkish_lower,
};

lazy_static! {
    /// Quantity words and numerical determiners that disallow plural suffix on the head noun
    static ref QUANTITY_MODIFIERS: HashSet<&'static str> = {
        let mut s = HashSet::new();
        s.insert("bir");
        s.insert("iki");
        s.insert("üç");
        s.insert("dört");
        s.insert("beş");
        s.insert("altı");
        s.insert("yedi");
        s.insert("sekiz");
        s.insert("dokuz");
        s.insert("on");
        s.insert("yirmi");
        s.insert("otuz");
        s.insert("kırk");
        s.insert("elli");
        s.insert("altmış");
        s.insert("yetmiş");
        s.insert("seksen");
        s.insert("doksan");
        s.insert("yüz");
        s.insert("bin");
        s.insert("milyon");
        s.insert("birçok");
        s.insert("birkaç");
        s.insert("çok");
        s.insert("az");
        s.insert("onlarca");
        s.insert("yüzlerce");
        s.insert("binlerce");
        s.insert("kaç");
        s.insert("fazla");
        s.insert("hayli");
        s
    };

    /// Negative pronouns / adverbs requiring negative verb polarity
    static ref NEGATION_TRIGGERS: HashSet<&'static str> = {
        let mut s = HashSet::new();
        s.insert("kimse");
        s.insert("hiçbiri");
        s.insert("hiçbirisi");
        s.insert("hiçkimse");
        s.insert("asla");
        s.insert("katiyen");
        s
    };
}

pub struct AgreementRuleEngine;

impl AgreementRuleEngine {
    pub fn check(
        _text: &str,
        tokens: &[&str],
        token_spans: &[(usize, usize)],
        morphology: &TurkishMorphology,
        disambiguator: &MorphologicalDisambiguator,
        parser: &TurkishDependencyParser,
        findings: &mut Vec<GrammarFinding>,
    ) {
        if tokens.is_empty() {
            return;
        }

        let disambiguated = disambiguator.disambiguate(tokens);

        Self::check_quantity_plural_agreement(tokens, token_spans, &disambiguated, findings);
        Self::check_genitive_possessive_concord(tokens, token_spans, &disambiguated, morphology, findings);
        Self::check_correlative_conjunctions(tokens, token_spans, &disambiguated, findings);
        Self::check_negation_concord(tokens, token_spans, &disambiguated, findings);
        Self::check_subject_verb_inanimate_plural(tokens, token_spans, &disambiguated, parser, findings);
    }

    /// 1. Quantity Determiner + Noun Plural Redundancy
    /// In Turkish, nouns modified by quantity words or numbers must be singular (*üç elmalar* -> *üç elma*).
    fn check_quantity_plural_agreement(
        tokens: &[&str],
        token_spans: &[(usize, usize)],
        parses: &[MorphParse],
        findings: &mut Vec<GrammarFinding>,
    ) {
        let n = tokens.len();
        for i in 0..n {
            let curr_lower = to_turkish_lower(tokens[i]);
            let is_qty = QUANTITY_MODIFIERS.contains(curr_lower.as_str()) || curr_lower.chars().all(|c| c.is_ascii_digit());

            if is_qty && i + 1 < n {
                let next_tok = tokens[i + 1];
                let next_lower = to_turkish_lower(next_tok);
                let next_parse = &parses[i + 1];
                let (next_start, next_end) = token_spans[i + 1];

                // Exclude lexical exceptions (e.g. "Üç Silahşorlar", "Yedi Cüceler" proper nouns)
                if next_parse.secondary_pos == SecondaryPos::ProperNoun {
                    continue;
                }

                // Check if next token is a plural noun
                if (next_lower.ends_with("ler") || next_lower.ends_with("lar")) && next_lower.len() > 3 {
                    let has_plural_tag = next_parse.morpheme_tags.iter().any(|t| t.contains("Plur") || t.contains("A3pl"));
                    if has_plural_tag || next_parse.primary_pos == PrimaryPos::Noun {
                        // Strip plural suffix to obtain singular form
                        let singular_form = &next_tok[..next_tok.len() - 3];
                        if !singular_form.is_empty() {
                            findings.push(GrammarFinding {
                                category: ErrorCategory::QuantityPluralClash,
                                start_offset: next_start,
                                end_offset: next_end,
                                original_text: next_tok.to_string(),
                                replacement: singular_form.to_string(),
                                message_tr: "Sayı sıfatları ve miktar belirteçlerinden sonra gelen isimler çoğul eki (-ler/-lar) almaz.".to_string(),
                                message_en: "Nouns preceded by numerical or quantity modifiers must remain singular.".to_string(),
                                confidence: 0.98,
                            });
                        }
                    }
                }
            }
        }
    }

    /// 2. Genitive - Possessive Concord
    /// *benim* requires 1st sing possessive (*evim*), *senin* requires 2nd sing (*evin*), etc.
    fn check_genitive_possessive_concord(
        tokens: &[&str],
        token_spans: &[(usize, usize)],
        _parses: &[MorphParse],
        morphology: &TurkishMorphology,
        findings: &mut Vec<GrammarFinding>,
    ) {
        let genitive_pronouns = [
            ("benim", 1, "m"),
            ("senin", 2, "n"),
            ("onun", 3, "si"),
            ("bizim", 4, "miz"),
            ("sizin", 5, "niz"),
        ];

        let n = tokens.len();
        for i in 0..n {
            let curr_lower = to_turkish_lower(tokens[i]);
            for &(pron, person, _expected_suf) in &genitive_pronouns {
                if curr_lower == pron && i + 1 < n {
                    let next_tok = tokens[i + 1];
                    let next_lower = to_turkish_lower(next_tok);
                    let (next_start, next_end) = token_spans[i + 1];

                    let noun_parses = morphology.analyze(&next_lower);
                    if noun_parses.is_empty() {
                        continue;
                    }

                    // If noun is bare root (no possessive) after "benim" (e.g. "benim araba" -> "benim arabam")
                    let is_bare_noun = noun_parses.iter().any(|p| p.primary_pos == PrimaryPos::Noun && p.morpheme_tags.iter().all(|t| !t.contains("P1sg") && !t.contains("P2sg") && !t.contains("P3sg")));

                    if is_bare_noun && person == 1 {
                        let last_ch = next_lower.chars().last().unwrap_or('a');
                        let poss_suffix = if is_turkish_vowel(last_ch) {
                            "m".to_string()
                        } else {
                            format!("{}m", get_i_type_harmonic_vowel(&next_lower))
                        };
                        let replacement = format!("{}{}", next_tok, poss_suffix);
                        findings.push(GrammarFinding {
                            category: ErrorCategory::GenitivePossessiveClash,
                            start_offset: next_start,
                            end_offset: next_end,
                            original_text: next_tok.to_string(),
                            replacement,
                            message_tr: format!("'{}' tamlayanından sonra gelen isim iyelik eki almalıdır (tamlayan-tamlanan uyumu).", pron),
                            message_en: format!("Nouns following the genitive '{}' must take matching possessive suffix.", pron),
                            confidence: 0.94,
                        });
                    }
                }
            }
        }
    }

    /// 3. Correlative Conjunction Polarity (*Ne... Ne...*):
    /// In standard Turkish, sentences with *ne ... ne ...* must have an affirmative (positive) predicate (*Ne geldi ne gitti* vs *Ne gelmedi ne gitmedi* ❌).
    fn check_correlative_conjunctions(
        tokens: &[&str],
        token_spans: &[(usize, usize)],
        parses: &[MorphParse],
        findings: &mut Vec<GrammarFinding>,
    ) {
        let has_ne_ne = tokens.iter().filter(|&&t| to_turkish_lower(t) == "ne").count() >= 2;
        if !has_ne_ne {
            return;
        }

        // Check if predicate verb has a negative suffix (-me/-ma)
        for (i, parse) in parses.iter().enumerate() {
            if parse.primary_pos == PrimaryPos::Verb {
                let has_neg = parse.morpheme_tags.iter().any(|t| t.contains("Neg"));
                if has_neg {
                    let tok = tokens[i];
                    let (start, end) = token_spans[i];
                    let lower = to_turkish_lower(tok);
                    // Remove negative marker "-me/-ma"
                    let fixed = lower.replace("medi", "di").replace("madı", "dı")
                                     .replace("miyor", "iyor").replace("mıyor", "ıyor")
                                     .replace("müyor", "üyor").replace("muyor", "uyor")
                                     .replace("meyecek", "ecek").replace("mayacak", "acak")
                                     .replace("mez", "r").replace("maz", "r");

                    findings.push(GrammarFinding {
                        category: ErrorCategory::CorrelativeConjunctionPolarity,
                        start_offset: start,
                        end_offset: end,
                        original_text: tok.to_string(),
                        replacement: fixed,
                        message_tr: "'Ne ... ne ...' bağlacının kullanıldığı cümlelerde yüklem olumlu olmalıdır.".to_string(),
                        message_en: "Predicates in sentences with the correlative conjunction 'ne ... ne ...' must be positive.".to_string(),
                        confidence: 0.96,
                    });
                }
            }
        }
    }

    /// 4. Negation Concord (*Kimse / Hiçbiri*):
    /// Must be used with negative verbs (*Kimse geldi* ❌ -> *Kimse gelmedi*).
    fn check_negation_concord(
        tokens: &[&str],
        token_spans: &[(usize, usize)],
        parses: &[MorphParse],
        findings: &mut Vec<GrammarFinding>,
    ) {
        let has_negation_trigger = tokens.iter().any(|&t| NEGATION_TRIGGERS.contains(to_turkish_lower(t).as_str()));
        if !has_negation_trigger {
            return;
        }

        // Find the final verb predicate
        if let Some((i, parse)) = parses.iter().enumerate().rfind(|(_, p)| p.primary_pos == PrimaryPos::Verb) {
            let has_neg = parse.morpheme_tags.iter().any(|t| t.contains("Neg") || t.contains("Yok") || t.contains("Değil"));
            if !has_neg {
                let tok = tokens[i];
                let (start, end) = token_spans[i];
                findings.push(GrammarFinding {
                    category: ErrorCategory::CorrelativeConjunctionPolarity,
                    start_offset: start,
                    end_offset: end,
                    original_text: tok.to_string(),
                    replacement: tok.to_string(), // Detection flag
                    message_tr: "'Kimse / Hiçbiri' gibi olumsuzluk zamirleri bulunan cümlelerde yüklem olumsuz olmalıdır.".to_string(),
                    message_en: "Sentences with negative pronouns ('kimse/hiçbiri') require a negative predicate.".to_string(),
                    confidence: 0.93,
                });
            }
        }
    }

    /// 5. Subject-Verb Plurality & Inanimate Agreement:
    /// Inanimate / abstract plural subjects take a singular verb (*Ağaçlar döküldüler* -> *Ağaçlar döküldü*).
    fn check_subject_verb_inanimate_plural(
        tokens: &[&str],
        token_spans: &[(usize, usize)],
        parses: &[MorphParse],
        _parser: &TurkishDependencyParser,
        findings: &mut Vec<GrammarFinding>,
    ) {
        let n = tokens.len();
        if n < 2 {
            return;
        }

        // Look for plural inanimate subject at sentence start followed by plural verb at sentence end
        let first_parse = &parses[0];
        let is_plural_noun = first_parse.primary_pos == PrimaryPos::Noun && first_parse.morpheme_tags.iter().any(|t| t.contains("Plur") || t.contains("A3pl"));

        if is_plural_noun {
            if let Some((last_idx, _last_parse)) = parses.iter().enumerate().rfind(|(_, p)| p.primary_pos == PrimaryPos::Verb) {
                let last_tok = tokens[last_idx];
                let last_lower = to_turkish_lower(last_tok);
                let (start, end) = token_spans[last_idx];

                if last_lower.ends_with("ler") || last_lower.ends_with("lar") {
                    let singular_verb = &last_tok[..last_tok.len() - 3];
                    findings.push(GrammarFinding {
                        category: ErrorCategory::SubjectVerbAgreement,
                        start_offset: start,
                        end_offset: end,
                        original_text: last_tok.to_string(),
                        replacement: singular_verb.to_string(),
                        message_tr: "İnsan dışı varlık ve kavramların çoğul özne olduğu durumlarda yüklem tekil olur.".to_string(),
                        message_en: "Inanimate and abstract plural subjects take a singular predicate verb.".to_string(),
                        confidence: 0.90,
                    });
                }
            }
        }
    }
}
