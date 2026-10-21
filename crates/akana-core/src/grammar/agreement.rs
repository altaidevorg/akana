//! Turkish Morphosyntactic Agreement & Concord Error Engine.
//!
//! Evaluates:
//! 1. Quantity Determiner + Noun Plural Redundancy (*üç elmalar* -> *üç elma*, *birçok insanlar* -> *birçok insan*).
//! 2. Genitive - Possessive Person Concord (*benim ev* -> *benim evim*; only the first person).
//! 3. Correlative Conjunction Polarity (*Ne... Ne...* takes positive verb: *ne geldi ne gitmedi* -> *ne geldi ne gitti*).
//! 4. Negation Concord (*Kimse / Hiçbiri* requires negative verb: *Kimse geldi* -> *Kimse gelmedi*).
//! 5. Subject-Verb Agreement (Inanimate plural subjects take singular verb: *Ağaçlar döküldüler* -> *Ağaçlar döküldü*).
//!
//! The rules read the disambiguated parse of each word, use its final POS (`pos`: a participle
//! is not a verb), and look at one sentence at a time. None of them knows animacy, clause
//! structure or how to generate a form, so they stay experimental.

use lazy_static::lazy_static;
use std::collections::HashSet;
use std::ops::Range;

use super::{GrammarFinding, GrammarRule, RuleSet};
use crate::morphology::pos::{PrimaryPos, SecondaryPos};
use crate::morphology::{MorphParse, MorphologicalDisambiguator, TurkishMorphology};
use crate::parser::TurkishDependencyParser;
use crate::phonology::{get_i_type_harmonic_vowel, is_turkish_vowel, to_turkish_lower};

lazy_static! {
    /// Numbers and quantifiers after which a noun stays singular. Not *bir* (bir şeyler, bir
    /// zamanlar), *çok*, *az*, *fazla* or *hayli*, which are also adverbs (çok güzeller,
    /// fazla sular).
    static ref QUANTITY_MODIFIERS: HashSet<&'static str> = {
        let mut s = HashSet::new();
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
        s.insert("onlarca");
        s.insert("yüzlerce");
        s.insert("binlerce");
        s.insert("kaç");
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

fn has_tag(parse: &MorphParse, tag: &str) -> bool {
    parse.morpheme_tags.iter().any(|t| t == tag)
}

/// A verb after its last derivation: not a participle or verbal noun with a verb root.
fn is_verb(parse: &MorphParse) -> bool {
    parse.pos == PrimaryPos::Verb
}

pub struct AgreementRuleEngine;

impl AgreementRuleEngine {
    #[allow(clippy::too_many_arguments)]
    pub fn check(
        _text: &str,
        tokens: &[&str],
        token_spans: &[(usize, usize)],
        sentences: &[Range<usize>],
        morphology: &TurkishMorphology,
        disambiguator: &MorphologicalDisambiguator,
        parser: &TurkishDependencyParser,
        rules: RuleSet,
        findings: &mut Vec<GrammarFinding>,
    ) {
        const MINE: [GrammarRule; 5] = [
            GrammarRule::QuantityPlural,
            GrammarRule::GenitivePossessive,
            GrammarRule::NeNePolarity,
            GrammarRule::NegationConcord,
            GrammarRule::InanimatePluralVerb,
        ];
        if tokens.is_empty() || !MINE.iter().any(|&r| rules.contains(r)) {
            return;
        }

        let disambiguated = disambiguator.disambiguate(tokens);

        for range in sentences {
            let toks = &tokens[range.clone()];
            let spans = &token_spans[range.clone()];
            let parses = &disambiguated[range.clone()];
            if rules.contains(GrammarRule::QuantityPlural) {
                Self::check_quantity_plural_agreement(toks, spans, parses, findings);
            }
            if rules.contains(GrammarRule::GenitivePossessive) {
                Self::check_genitive_possessive_concord(toks, spans, parses, findings);
            }
            if rules.contains(GrammarRule::NeNePolarity) {
                Self::check_correlative_conjunctions(toks, spans, parses, morphology, findings);
            }
            if rules.contains(GrammarRule::NegationConcord) {
                Self::check_negation_concord(toks, spans, parses, findings);
            }
            if rules.contains(GrammarRule::InanimatePluralVerb) {
                Self::check_subject_verb_inanimate_plural(toks, spans, parses, parser, findings);
            }
        }
    }

    /// 1. Quantity Determiner + Noun Plural Redundancy:
    ///
    /// In Turkish, nouns modified by quantity words or numbers must be singular (*üç elmalar* -> *üç elma*).
    /// Only a bare plural noun (no case or possessive after -ler, so the fix is to drop it),
    /// written in lower case: "Bakanlar Kurulu" and "Üç Silahşorlar" are names, "dolar" is
    /// no plural.
    fn check_quantity_plural_agreement(
        tokens: &[&str],
        token_spans: &[(usize, usize)],
        parses: &[MorphParse],
        findings: &mut Vec<GrammarFinding>,
    ) {
        let n = tokens.len();
        for i in 0..n.saturating_sub(1) {
            let curr_lower = to_turkish_lower(tokens[i]);
            let is_qty = QUANTITY_MODIFIERS.contains(curr_lower.as_str())
                || curr_lower.chars().all(|c| c.is_ascii_digit());
            if !is_qty {
                continue;
            }
            let next_tok = tokens[i + 1];
            let next_parse = &parses[i + 1];
            let (next_start, next_end) = token_spans[i + 1];
            if next_parse.secondary_pos == SecondaryPos::ProperNoun
                || next_tok.chars().next().is_some_and(char::is_uppercase)
            {
                continue;
            }
            let bare_plural = (next_tok.ends_with("ler") || next_tok.ends_with("lar"))
                && next_tok.len() > 3
                && next_parse.pos == PrimaryPos::Noun
                && has_tag(next_parse, "A3pl")
                && !next_parse.morpheme_tags.iter().any(|t| {
                    t.starts_with('P')
                        || matches!(t.as_str(), "Acc" | "Dat" | "Loc" | "Abl" | "Gen" | "Ins")
                });
            if bare_plural {
                let singular_form = &next_tok[..next_tok.len() - 3];
                findings.push(GrammarFinding::new(
                    GrammarRule::QuantityPlural,
                    (next_start, next_end),
                    next_tok.to_string(),
                    singular_form.to_string(),
                    "Sayı sıfatları ve miktar belirteçlerinden sonra gelen isimler çoğul eki (-ler/-lar) almaz.".to_string(),
                    "Nouns preceded by numerical or quantity modifiers must remain singular.".to_string(),
                ));
            }
        }
    }

    /// 2. Genitive - Possessive Concord:
    ///
    /// *benim* requires the 1st person singular possessive (*benim ev* -> *benim evim*). Only
    /// the first person is implemented. The next word must read as a bare noun in the
    /// nominative (not a postposition, conjunction or adjective: "Benim için", "Benim de"),
    /// and no word right after it may carry the possessive ("benim kitap okumam").
    fn check_genitive_possessive_concord(
        tokens: &[&str],
        token_spans: &[(usize, usize)],
        parses: &[MorphParse],
        findings: &mut Vec<GrammarFinding>,
    ) {
        let n = tokens.len();
        for i in 0..n.saturating_sub(1) {
            if to_turkish_lower(tokens[i]) != "benim" {
                continue;
            }
            let next_tok = tokens[i + 1];
            let next_lower = to_turkish_lower(next_tok);
            let next = &parses[i + 1];
            let bare_noun = next.pos == PrimaryPos::Noun
                && next.primary_pos == PrimaryPos::Noun
                && next.secondary_pos != SecondaryPos::ProperNoun
                && has_tag(next, "A3sg")
                && has_tag(next, "Nom")
                && !next.morpheme_tags.iter().any(|t| t.starts_with('P'));
            let possessed_later = parses[i + 2..].iter().take(3).any(|p| has_tag(p, "P1sg"));
            if !bare_noun || possessed_later {
                continue;
            }
            let (next_start, next_end) = token_spans[i + 1];
            let last_ch = next_lower.chars().last().unwrap_or('a');
            let poss_suffix = if is_turkish_vowel(last_ch) {
                "m".to_string()
            } else {
                format!("{}m", get_i_type_harmonic_vowel(&next_lower))
            };
            findings.push(GrammarFinding::new(
                GrammarRule::GenitivePossessive,
                (next_start, next_end),
                next_tok.to_string(),
                format!("{next_tok}{poss_suffix}"),
                "'benim' tamlayanından sonra gelen isim iyelik eki almalıdır (tamlayan-tamlanan uyumu).".to_string(),
                "Nouns following the genitive 'benim' must take matching possessive suffix.".to_string(),
            ));
        }
    }

    /// 3. Correlative Conjunction Polarity (*Ne... Ne...*):
    ///
    /// In standard Turkish, sentences with *ne ... ne ...* must have an affirmative (positive) predicate (*Ne geldi ne gitti* vs *Ne gelmedi ne gitmedi* ❌).
    /// Within one sentence that is not a question, a negative verb after the first *ne* is
    /// flagged, and only when dropping its negative gives a word the analyser knows.
    fn check_correlative_conjunctions(
        tokens: &[&str],
        token_spans: &[(usize, usize)],
        parses: &[MorphParse],
        morphology: &TurkishMorphology,
        findings: &mut Vec<GrammarFinding>,
    ) {
        let ne: Vec<usize> = (0..tokens.len())
            .filter(|&i| to_turkish_lower(tokens[i]) == "ne")
            .collect();
        if ne.len() < 2 || tokens.last() == Some(&"?") {
            return;
        }

        for (i, parse) in parses.iter().enumerate().skip(ne[0] + 1) {
            if !is_verb(parse) || !has_tag(parse, "Neg") {
                continue;
            }
            let tok = tokens[i];
            let lower = to_turkish_lower(tok);
            // Remove negative marker "-me/-ma"
            let fixed = lower
                .replace("medi", "di")
                .replace("madı", "dı")
                .replace("miyor", "iyor")
                .replace("mıyor", "ıyor")
                .replace("müyor", "üyor")
                .replace("muyor", "uyor")
                .replace("meyecek", "ecek")
                .replace("mayacak", "acak")
                .replace("mez", "r")
                .replace("maz", "r");
            let positive = morphology
                .analyze(&fixed)
                .iter()
                .any(|p| is_verb(p) && !has_tag(p, "Neg"));
            if fixed == lower || !positive {
                continue;
            }
            findings.push(GrammarFinding::new(
                GrammarRule::NeNePolarity,
                token_spans[i],
                tok.to_string(),
                fixed,
                "'Ne ... ne ...' bağlacının kullanıldığı cümlelerde yüklem olumlu olmalıdır."
                    .to_string(),
                "Predicates in sentences with the correlative conjunction 'ne ... ne ...' must be positive."
                    .to_string(),
            ));
        }
    }

    /// 4. Negation Concord (*Kimse / Hiçbiri*):
    ///
    /// Must be used with negative verbs (*Kimse geldi* ❌ -> *Kimse gelmedi*). The verb is the
    /// first one after the trigger in its sentence ("Kimse gelmedi ama sonra herkes geldi" is
    /// fine), and *değil*/*yok* after it count as negation. Detection only: the replacement
    /// is the word itself.
    fn check_negation_concord(
        tokens: &[&str],
        token_spans: &[(usize, usize)],
        parses: &[MorphParse],
        findings: &mut Vec<GrammarFinding>,
    ) {
        let Some(trigger) = tokens
            .iter()
            .position(|&t| NEGATION_TRIGGERS.contains(to_turkish_lower(t).as_str()))
        else {
            return;
        };
        let Some(i) = (trigger + 1..tokens.len()).find(|&i| is_verb(&parses[i])) else {
            return;
        };
        let negated = has_tag(&parses[i], "Neg")
            || has_tag(&parses[i], "NegAbil")
            || tokens[i..].iter().take(3).any(|t| {
                let t = to_turkish_lower(t);
                t.starts_with("değil") || t.starts_with("yok")
            });
        if !negated {
            let tok = tokens[i];
            findings.push(GrammarFinding::new(
                GrammarRule::NegationConcord,
                token_spans[i],
                tok.to_string(),
                tok.to_string(),
                "'Kimse / Hiçbiri' gibi olumsuzluk zamirleri bulunan cümlelerde yüklem olumsuz olmalıdır.".to_string(),
                "Sentences with negative pronouns ('kimse/hiçbiri') require a negative predicate.".to_string(),
            ));
        }
    }

    /// 5. Subject-Verb Plurality & Inanimate Agreement:
    ///
    /// Inanimate / abstract plural subjects take a singular verb (*Ağaçlar döküldüler* -> *Ağaçlar döküldü*).
    /// There is no animacy check: any plural noun opening the sentence counts as the subject,
    /// and the last verb of the sentence with a third person plural ending is flagged.
    fn check_subject_verb_inanimate_plural(
        tokens: &[&str],
        token_spans: &[(usize, usize)],
        parses: &[MorphParse],
        _parser: &TurkishDependencyParser,
        findings: &mut Vec<GrammarFinding>,
    ) {
        if tokens.len() < 2 {
            return;
        }
        let first = &parses[0];
        if !(first.pos == PrimaryPos::Noun && has_tag(first, "A3pl")) {
            return;
        }
        let Some(last_idx) = parses.iter().rposition(is_verb) else {
            return;
        };
        let last_tok = tokens[last_idx];
        if has_tag(&parses[last_idx], "A3pl")
            && (last_tok.ends_with("ler") || last_tok.ends_with("lar"))
        {
            findings.push(GrammarFinding::new(
                GrammarRule::InanimatePluralVerb,
                token_spans[last_idx],
                last_tok.to_string(),
                last_tok[..last_tok.len() - 3].to_string(),
                "İnsan dışı varlık ve kavramların çoğul özne olduğu durumlarda yüklem tekil olur."
                    .to_string(),
                "Inanimate and abstract plural subjects take a singular predicate verb."
                    .to_string(),
            ));
        }
    }
}
