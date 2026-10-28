//! SIMD-Accelerated Turkish Orthographic and Clitic Error Detection and Correction.
//!
//! Covers:
//! 1. `de/da` Conjunction (separate) vs Locative Suffix (attached) and incorrect consonant assimilation (*gitse te* -> *gitse de*).
//! 2. `ki` Conjunction (separate, with SOMBAHÇEMİ exception list) vs Relative/Possessive Suffix (attached).
//! 3. `mi/mı/mu/mü` Question Particle (separate, 4-way vowel harmony).
//! 4. Proper noun & numeric apostrophes (*Ankara'da*, *1923'te*, *2'nci*, *Ahmetlerin*).
//! 5. Sentence-initial & Proper Noun Capitalization.
//! 6. Reduplication orthography (*el ele*, *yan yana*).

use lazy_static::lazy_static;
use std::collections::HashSet;
use stringzilla::StringZilla;

use super::{capitalize_first, match_case, GrammarFinding, GrammarRule, RuleSet};
use crate::morphology::pos::{PrimaryPos, RootAttr, SecondaryPos};
use crate::morphology::TurkishMorphology;
use crate::phonology::{
    harmony_i_type, is_front_vowel, last_vowel, origin_offset, to_turkish_lower,
    to_turkish_lower_with_origin,
};

/// Whether the analyser has a reading of `word`, as written: a token with its case (Ankarada
/// is a name's spelling, a lower-cased one is none; see `analyze_standard`), or a lower-cased
/// stem. The rules that split a clitic off a word fire only on a word it does not know:
/// "hükmü", "yetki" and "kararda" end like a merged particle but are words, and "varmı"
/// has no reading but the name Varmı's, which does not count in lower case.
pub(crate) fn is_known(morphology: &TurkishMorphology, word: &str) -> bool {
    !morphology.analyze_standard(word).is_empty()
}

/// Whether `lower` reads as a finite verb (gitti, yapar, gelmiş, bilse): a verb after its
/// last derivation, with a tense, aspect or mood. The root's POS is not enough, as every
/// deverbal noun (sözleşme, yetki) and participle (bulunduğun) has a verb root.
fn is_finite_verb(morphology: &TurkishMorphology, lower: &str) -> bool {
    const FINITE: [&str; 8] = ["Past", "Prog", "Aor", "Fut", "Evid", "Cond", "Opt", "Nec"];
    morphology.analyze_standard(lower).iter().any(|p| {
        p.pos == PrimaryPos::Verb && p.morpheme_tags.iter().any(|t| FINITE.contains(&t.as_str()))
    })
}

/// Whether `lower` reads as a word in the accusative, dative, genitive, ablative or
/// instrumental (onu, eve, kitabın, ulaştığını, yerden): such a word takes no locative, so a
/// -de/-da after it is the conjunction. Not the locative, and not a name or abbreviation
/// read into a lower-case word ("saudade" is no Saud-a-de).
fn is_case_marked(morphology: &TurkishMorphology, lower: &str) -> bool {
    fn case(p: &crate::morphology::MorphParse) -> Option<&str> {
        const CASES: [&str; 5] = ["Acc", "Dat", "Gen", "Abl", "Ins"];
        p.morpheme_tags
            .iter()
            .map(String::as_str)
            .find(|t| CASES.contains(t))
    }
    let common: Vec<_> = morphology
        .analyze_standard(lower)
        .into_iter()
        .filter(|p| {
            !matches!(
                p.secondary_pos,
                SecondaryPos::ProperNoun | SecondaryPos::Abbreviation
            )
        })
        .collect();
    let mut cases = common.iter().filter_map(case).peekable();
    if cases.peek().is_none() {
        return false;
    }
    // A stem that is only a genitive must be one at its best reading.
    if cases.all(|c| c == "Gen") {
        return common.first().and_then(case).is_some();
    }
    true
}

/// Whether a stem's case readings are all genitives (alının, hakkının).
fn only_genitive(morphology: &TurkishMorphology, lower: &str) -> bool {
    let mut cases = morphology
        .analyze_standard(lower)
        .into_iter()
        .filter(|p| {
            !matches!(
                p.secondary_pos,
                SecondaryPos::ProperNoun | SecondaryPos::Abbreviation
            )
        })
        .filter_map(|p| {
            p.morpheme_tags
                .into_iter()
                .find(|t| matches!(t.as_str(), "Acc" | "Dat" | "Gen" | "Abl" | "Ins"))
        })
        .peekable();
    cases.peek().is_some() && cases.all(|c| c == "Gen")
}

/// Lower-case word forms ending in -ndA that the court sample writes at least 50 times
/// (scripts/frequent_locatives.py).
static FREQUENT_LOCATIVES: std::sync::LazyLock<std::collections::HashSet<&'static str>> =
    std::sync::LazyLock::new(|| {
        include_str!("frequent_locatives.txt")
            .lines()
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .collect()
    });

/// Whether a word one letter away from `lower` (a letter changed, dropped or added) is a
/// frequent word's possessed locative, -(s)I-ndA, of a root none of `own_roots`: `lower` is then
/// likelier that word misspelt than a genitive with -da written on it, or than a vowel-drop
/// error (aranında for oran-ı-nda, alınında for alan-ı-nda; review 2 GN2-03). A rare neighbour
/// is no evidence (seninde is no misspelt eninde, ülkeninde no üleninde), nor the token's own
/// root's locative (mahkemeninde is mahkeme-nin-de misspelt, not mahkemesinde; review 3
/// GRAM3-02). Not a neighbour the edit makes through another possessive or a derivation
/// (hakkınızda, fatura-sız-ı-nda for hakkınında and faturasınında, which are genitives).
/// Neighbours are looked up in [`FREQUENT_LOCATIVES`] before they are analysed.
pub(crate) fn misspelt_locative(
    morphology: &TurkishMorphology,
    lower: &str,
    own_roots: &[String],
) -> bool {
    const LETTERS: &str = "abcçdefgğhıijklmnoöprsştuüvyz";
    let chars: Vec<char> = lower.chars().collect();
    let locative = |word: Vec<char>| {
        let word: String = word.into_iter().collect();
        FREQUENT_LOCATIVES.contains(word.as_str())
            && morphology.analyze_standard(&word).iter().any(|p| {
                p.morpheme_tags
                    .ends_with(&["P3sg".to_string(), "Loc".to_string()])
                    && !own_roots.contains(&p.root)
                    && !p.morpheme_tags.iter().any(|t| {
                        crate::morphology::is_derivational_tag(t)
                            && !matches!(t.as_str(), "ActN" | "Inf")
                    })
            })
    };
    for i in 0..chars.len() {
        let mut dropped = chars.clone();
        dropped.remove(i);
        if locative(dropped) {
            return true;
        }
        for letter in LETTERS.chars().filter(|&l| l != chars[i]) {
            let mut changed = chars.clone();
            changed[i] = letter;
            if locative(changed) {
                return true;
            }
        }
    }
    (0..=chars.len()).any(|i| {
        LETTERS.chars().any(|letter| {
            let mut added = chars.clone();
            added.insert(i, letter);
            locative(added)
        })
    })
}

/// The roots of a stem's genitive readings (alının: alın), which a misspelt locative's
/// neighbour must not share ([`misspelt_locative`]).
fn genitive_roots(morphology: &TurkishMorphology, stem: &str) -> Vec<String> {
    morphology
        .analyze_standard(stem)
        .into_iter()
        .filter(|p| p.morpheme_tags.iter().any(|t| t == "Gen"))
        .map(|p| p.root)
        .collect()
}

/// The vowel a suffix harmonises with after `lower`: its last vowel, made front when `lower`
/// is a root that takes front-vowel suffixes despite a back vowel (saat, gol, kalp:
/// InverseHarmony in the lexicon).
fn harmony_vowel(morphology: &TurkishMorphology, lower: &str) -> Option<char> {
    let v = last_vowel(lower)?;
    let inverse = morphology
        .root_items(lower)
        .iter()
        .any(|item| item.attributes.contains(RootAttr::INVERSE_HARMONY));
    Some(match (inverse, v) {
        (true, 'a') => 'e',
        (true, 'ı') => 'i',
        (true, 'o') => 'ö',
        (true, 'u') => 'ü',
        _ => v,
    })
}

/// `tok` split before the characters that make up the last `suffix_len` bytes of
/// `to_turkish_lower(tok)`. Lower-casing changes byte lengths (MI is 2 bytes and mı 3, Mİ 3
/// and mi 2), so a suffix length measured on the lower-cased token cannot cut the token
/// itself: KARIŞMI was cut inside the Ş, and GELDİNMİ became "GELDİNM" + "İ".
fn split_lower_suffix(tok: &str, suffix_len: usize) -> (&str, &str) {
    let (lower, origin) = to_turkish_lower_with_origin(tok);
    let at = origin_offset(&origin, tok.len(), lower.len().saturating_sub(suffix_len));
    tok.split_at(at)
}

/// How the number with these digits ends when read out, for the suffix after it: whether the
/// last word ends in a voiceless consonant, and whether its last vowel is front. 1923 ends in
/// "üç" (voiceless, front: 1923'te), 1940 in "kırk" (1940'ta), 1990 in "doksan" (1990'da),
/// 100 in "yüz" (100'de). `None` past the trillions.
fn number_reading(digits: &str) -> Option<(bool, bool)> {
    let digits = digits.trim_start_matches('0');
    let zeros = digits.len() - digits.trim_end_matches('0').len();
    let last = |d: u8| (d - b'0') as usize;
    let word = match (digits.as_bytes(), zeros) {
        ([], _) => "sıfır",
        (b, 0) => [
            "", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz",
        ][last(b[b.len() - 1])],
        (b, 1) => [
            "", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan",
        ][last(b[b.len() - 2])],
        (_, 2) => "yüz",
        (_, 3..=5) => "bin",
        (_, 6..=8) => "milyon",
        (_, 9..=11) => "milyar",
        (_, 12..=14) => "trilyon",
        _ => return None,
    };
    let end = word.chars().last()?;
    let voiceless = matches!(end, 'ç' | 'f' | 'h' | 'k' | 'p' | 's' | 'ş' | 't');
    Some((voiceless, is_front_vowel(last_vowel(word)?)))
}

/// Longest root and suffix, in bytes, the apostrophe rule looks at. Words are far shorter; a
/// longer run (base64, minified data) is not a word, and looking at all of it for every
/// apostrophe in it made the rule quadratic.
const MAX_APOSTROPHE_WORD: usize = 128;

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
        rules: RuleSet,
        findings: &mut Vec<GrammarFinding>,
    ) {
        if tokens.is_empty() {
            return;
        }

        if rules.contains(GrammarRule::Capitalization) {
            Self::check_capitalization(text, tokens, token_spans, findings);
        }
        Self::check_question_particle_mi(text, tokens, token_spans, morphology, rules, findings);
        Self::check_clitic_de_da(text, tokens, token_spans, morphology, rules, findings);
        Self::check_clitic_ki(text, tokens, token_spans, morphology, rules, findings);
        Self::check_apostrophes(text, tokens, token_spans, morphology, rules, findings);
        if rules.contains(GrammarRule::Reduplication) {
            Self::check_reduplications(text, tokens, token_spans, findings);
        }
        if rules.contains(GrammarRule::EbilSplit) {
            Self::check_modal_verb_ebilmek(text, tokens, token_spans, morphology, findings);
        }
        if rules.contains(GrammarRule::IndefiniteCompound) {
            Self::check_indefinite_compounds(text, tokens, token_spans, morphology, findings);
        }
    }

    /// 1. Sentence-initial & Proper Noun Capitalization
    fn check_capitalization(
        _text: &str,
        tokens: &[&str],
        token_spans: &[(usize, usize)],
        findings: &mut Vec<GrammarFinding>,
    ) {
        // Sentence first word must start with uppercase. A word with a capital inside (iPhone,
        // mRNA, e-Devlet) is written that way on purpose.
        if let Some(&first_tok) = tokens.first() {
            if let Some(first_char) = first_tok.chars().next() {
                if first_char.is_alphabetic()
                    && first_char.is_lowercase()
                    && !first_tok.chars().any(char::is_uppercase)
                {
                    let (start, end) = token_spans[0];
                    let replacement = capitalize_first(first_tok);
                    findings.push(GrammarFinding::new(
                        GrammarRule::Capitalization,
                        (start, end),
                        first_tok.to_string(),
                        replacement,
                        "Cümle büyük harfle başlamalıdır.".to_string(),
                        "Sentences must begin with a capital letter.".to_string(),
                    ));
                }
            }
        }
    }

    /// 2. `mi/mı/mu/mü` Question Particle Rules
    fn check_question_particle_mi(
        _text: &str,
        tokens: &[&str],
        token_spans: &[(usize, usize)],
        morphology: &TurkishMorphology,
        rules: RuleSet,
        findings: &mut Vec<GrammarFinding>,
    ) {
        let mi_particles = ["mi", "mı", "mu", "mü"];
        let mi_suffixes = [
            "misin",
            "mısın",
            "musun",
            "müsün",
            "miyiz",
            "mıyız",
            "muyuz",
            "müyüz",
            "misiniz",
            "mısınız",
            "musunuz",
            "müsünüz",
            "miler",
            "mılar",
        ];

        // Whether a token is a word as written. At the start of a sentence its capital says
        // nothing, and the corpus names Varmı and Yokmu do not make Varmı? a word (review 3
        // GRAM3-03): it must read in lower case too.
        let known_as_written = |i: usize, word: &str| {
            let sentence_start = i == 0 || matches!(tokens[i - 1], "." | "!" | "?" | "…" | ":");
            is_known(morphology, word)
                && (!sentence_start || is_known(morphology, &to_turkish_lower(word)))
        };
        for (i, &tok) in tokens.iter().enumerate() {
            let (start, end) = token_spans[i];
            let trimmed_tok = tok.trim_end_matches(|c: char| {
                c.is_ascii_punctuation()
                    || c == '?'
                    || c == '!'
                    || c == '.'
                    || c == ','
                    || c == ':'
                    || c == '"'
                    || c == '\''
            });
            let punct = &tok[trimmed_tok.len()..];
            let raw_lower = to_turkish_lower(trimmed_tok);
            let lower = raw_lower.as_str();

            // Case A: Merged question particle at the end of word (e.g. "geldinmi", "biliyormusun", "değilmi?")
            for &p in &mi_particles {
                if rules.contains(GrammarRule::MiMerged)
                    && lower.ends_with(p)
                    && lower.len() > p.len() + 2
                {
                    let stem = &lower[..lower.len() - p.len()];
                    if let Some(stem_last_vowel) = harmony_vowel(morphology, stem) {
                        let expected_vowel = harmony_i_type(stem_last_vowel);
                        let expected_p = match expected_vowel {
                            'ı' => "mı",
                            'i' => "mi",
                            'u' => "mu",
                            'ü' => "mü",
                            _ => "mi",
                        };

                        if p == expected_p
                            && (stem.ends_with('r')
                                || stem.ends_with('n')
                                || stem.ends_with('k')
                                || stem.ends_with('z')
                                || stem.ends_with('d')
                                || stem.ends_with('t')
                                || stem.ends_with('m')
                                || stem.ends_with('ş')
                                || stem.ends_with('l'))
                            // a word that ends like it is no merged particle: hükmü, filmi, yirmi;
                            // the stem as written, so that a name's counts (Ahmetmi, Türkiye'denmi;
                            // review 3 GRAM3-03)
                            && !known_as_written(i, trimmed_tok)
                            && is_known(morphology, split_lower_suffix(trimmed_tok, p.len()).0)
                        {
                            // the particle as written (Mİ in GELDİNMİ), which is `expected_p`
                            let (original_stem, original_p) =
                                split_lower_suffix(trimmed_tok, p.len());
                            let replacement = format!("{original_stem} {original_p}{punct}");
                            findings.push(GrammarFinding::new(
                                GrammarRule::MiMerged,
                                (start, end),
                                tok.to_string(),
                                replacement,
                                "Soru eki olan 'mı/mi/mu/mü' kendinden önceki sözcükten ayrı yazılır.".to_string(),
                                "The question particle 'mı/mi/mu/mü' must be written separately.".to_string(),
                            ));
                            break;
                        }
                    }
                }
            }

            // Case A2: Merged question particle with person suffix (e.g. "biliyormusun", "gelirmisiniz?")
            for &suf in &mi_suffixes {
                if rules.contains(GrammarRule::MiMergedPerson)
                    && lower.ends_with(suf)
                    && lower.len() > suf.len() + 2
                    && !known_as_written(i, trimmed_tok)
                    && is_known(morphology, split_lower_suffix(trimmed_tok, suf.len()).0)
                {
                    let (original_stem, original_suf) = split_lower_suffix(trimmed_tok, suf.len());
                    let replacement = format!("{original_stem} {original_suf}{punct}");
                    findings.push(GrammarFinding::new(
                        GrammarRule::MiMergedPerson,
                        (start, end),
                        tok.to_string(),
                        replacement,
                        "Soru eki ve eklenen şahıs ekleri ayrı yazılır.".to_string(),
                        "Question particles and person suffixes must be written separately."
                            .to_string(),
                    ));
                    break;
                }
            }

            // Case B: Standalone question particle with vowel harmony error (e.g., "gitti mı" -> "gitti mi")
            if rules.contains(GrammarRule::MiHarmony)
                && trimmed_tok == lower
                && (lower == "mi" || lower == "mı" || lower == "mu" || lower == "mü")
                && i > 0
            {
                let prev_tok = tokens[i - 1];
                if let Some(prev_vowel) = harmony_vowel(morphology, &to_turkish_lower(prev_tok)) {
                    let expected_vowel = harmony_i_type(prev_vowel);
                    let expected_p = match expected_vowel {
                        'ı' => "mı",
                        'i' => "mi",
                        'u' => "mu",
                        'ü' => "mü",
                        _ => "mi",
                    };
                    if lower != expected_p {
                        findings.push(GrammarFinding::new(
                            GrammarRule::MiHarmony,
                            (start, end),
                            tok.to_string(),
                            format!("{expected_p}{punct}"),
                            format!("Soru eki ünlü uyumuna göre '{expected_p}' olmalıdır."),
                            format!("The question particle should be '{expected_p}' according to vowel harmony."),
                        ));
                    }
                }
            }
        }
    }

    /// 3. `de/da` Clitic Rules
    fn check_clitic_de_da(
        text: &str,
        tokens: &[&str],
        token_spans: &[(usize, usize)],
        morphology: &TurkishMorphology,
        rules: RuleSet,
        findings: &mut Vec<GrammarFinding>,
    ) {
        // Not bende, sende, bizde, sizde or birde: those are locatives too ("Kitap bende kaldı",
        // "Saat birde buluşalım").
        const FREQUENT_CLITIC_MERGES: &[(&str, &str)] = &[
            ("yada", "ya da"),
            ("hemde", "hem de"),
            ("dahada", "daha da"),
            ("şunuda", "şunu da"),
            ("bunuda", "bunu da"),
            ("onuda", "onu da"),
            ("pekde", "pek de"),
            ("hiçde", "hiç de"),
            ("ya'da", "ya da"),
            ("hem'de", "hem de"),
            ("daha'da", "daha da"),
            ("ben'de", "ben de"),
            ("sen'de", "sen de"),
            ("o'da", "o da"),
            ("biz'de", "biz de"),
            ("siz'de", "siz de"),
            ("onlar'da", "onlar da"),
            ("bu'da", "bu da"),
            ("şu'da", "şu da"),
        ];

        // Not için, tarafın, adın, durumun, zamanın, yolun, peşin, boyun or üstün: in edited
        // text they are mostly a word followed by the conjunction ("Bunun için de", "davalı
        // tarafın da kabulünde").
        const SPLIT_LOCATIVE_STEMS: &[&str] = &[
            "hakkın",
            "arasın",
            "konusun",
            "üzerin",
            "yanın",
            "sayesin",
            "karşısın",
            "altın",
            "önün",
            "arkasın",
            "başın",
            "sonun",
            "halin",
            "yüzün",
            "içerisin",
            "dışarısın",
        ];

        for (i, &tok) in tokens.iter().enumerate() {
            let lower = to_turkish_lower(tok);
            let (start, end) = token_spans[i];

            // Case A: Standalone "te" or "ta", a word of its own after a word: not the suffix
            // of "1923'te" (which the tokenizer splits off), nor a name ("Ta"), nor the adverb
            // ta of "ta ki", "ta kendisi", "ta kalbine kadar"
            let adverb_ta = tok == "ta"
                && tokens.get(i + 1).is_some_and(|t| {
                    let t = to_turkish_lower(t);
                    t == "ki" || t.starts_with("kendi")
                })
                || tok == "ta"
                    && tokens[i + 1..]
                        .iter()
                        .take(3)
                        .any(|t| matches!(*t, "kadar" | "beri"));
            if rules.contains(GrammarRule::DeDaHardened)
                && (tok == "te" || tok == "ta")
                && i > 0
                && !adverb_ta
                && text[..start].ends_with(char::is_whitespace)
                // nor a word broken by a hyphen ("te-levizyonu")
                && !text[end..].starts_with('-')
                && tokens[i - 1].chars().all(char::is_alphabetic)
            {
                let prev_tok = tokens[i - 1];
                let prev_lower = to_turkish_lower(prev_tok);
                let correct_clitic = if let Some(v) = harmony_vowel(morphology, &prev_lower) {
                    if is_front_vowel(v) {
                        "de"
                    } else {
                        "da"
                    }
                } else {
                    "de"
                };
                findings.push(GrammarFinding::new(
                        GrammarRule::DeDaHardened,
                        (start, end),
                        tok.to_string(),
                        correct_clitic.to_string(),
                        "Bağlaç olan 'de/da' ayrı yazılır ve asla 'te/ta' şeklinde sertleşmez.".to_string(),
                        "The conjunction 'de/da' is written separately and never undergoes consonant hardening to 'te/ta'.".to_string(),
                    ));
            }

            // Case B: High frequency pronoun/adverb clitic merges & apostrophes (e.g. "yada", "hemde", "ya'da")
            for &(merged, split) in FREQUENT_CLITIC_MERGES {
                if rules.contains(GrammarRule::DeDaMergedFixed) && lower == merged {
                    let replacement = match_case(tok, split);
                    findings.push(GrammarFinding::new(
                        GrammarRule::DeDaMergedFixed,
                        (start, end),
                        tok.to_string(),
                        replacement,
                        "Bağlaç olan 'de/da' ayrı yazılır ve kesme işareti almaz.".to_string(),
                        "The conjunction 'de/da' must be written separately without an apostrophe."
                            .to_string(),
                    ));
                }
            }

            // Case C: Erroneously split locative/ablative suffix on noun stems (e.g. "hakkın da" -> "hakkında", "arasın da" -> "arasında")
            if rules.contains(GrammarRule::LocativeSplit)
                && (lower == "da"
                    || lower == "de"
                    || lower == "dan"
                    || lower == "den"
                    || lower == "nda"
                    || lower == "nde")
                && i > 0
            {
                let prev_tok = tokens[i - 1];
                let prev_lower = to_turkish_lower(prev_tok);
                if SPLIT_LOCATIVE_STEMS.contains(&prev_lower.as_str()) {
                    let prev_start = token_spans[i - 1].0;
                    let replacement = format!("{prev_tok}{tok}");
                    findings.push(GrammarFinding::new(
                        GrammarRule::LocativeSplit,
                        (prev_start, end),
                        format!("{prev_tok} {tok}"),
                        replacement,
                        "Bulunma/ayrılma durum eki (-de/-da/-den/-dan) sözcüğe bitişik yazılır."
                            .to_string(),
                        "The locative/ablative case suffix must be written attached to the noun."
                            .to_string(),
                    ));
                }
            }

            // Case D: Attached "de/da" on finite verbs (e.g. "gittide", "bilsede", "yaparda",
            // "gelmişde") or on a word in a case ("ulaştığınıda", "evede"): only on a word the
            // analyser does not know, whose stem is a finite verb or case-marked. A word that has
            // a reading (kararda, altında, bulunduğunda) is a locative.
            // A word in capitals is no name (ALİ GİTTİDE GELDİ; review 3 GRAM3-06), and a
            // lower-case word with its suffix after an apostrophe right after a capitalised one
            // is part of a name (Rossii po futbolu'da; review 3 GRAM3-01).
            let name_inside_sentence = i > 0
                && !matches!(tokens[i - 1], "." | "!" | "?" | "…")
                && tok.chars().next().is_some_and(char::is_uppercase)
                && tok.chars().any(char::is_lowercase);
            let name_continues = i > 1
                && tok.contains(['\'', '’'])
                && tokens[i - 1].chars().next().is_some_and(char::is_uppercase)
                && !matches!(tokens[i - 2], "." | "!" | "?" | "…");
            if rules.contains(GrammarRule::DeDaAfterVerb)
                && !name_inside_sentence
                && !name_continues
                && (lower.ends_with("de")
                    || lower.ends_with("da")
                    || lower.ends_with("te")
                    || lower.ends_with("ta"))
                && lower.len() > 4
            {
                let stem = &lower[..lower.len() - 2];
                // (not -te/-ta after a case: "modernite" is no moderni-te)
                if !is_known(morphology, tok)
                    && (is_finite_verb(morphology, stem)
                        || (lower.ends_with("de") || lower.ends_with("da"))
                            && is_case_marked(morphology, stem)
                            && !(only_genitive(morphology, stem)
                                && misspelt_locative(
                                    morphology,
                                    &lower,
                                    &genitive_roots(morphology, stem),
                                )))
                {
                    // without the apostrophe the clitic was written after: onu'da is onu da
                    // (review 3 GRAM3-01)
                    let (original_stem, _) = split_lower_suffix(tok, 2);
                    let original_stem = original_stem.trim_end_matches(['\'', '’']);
                    let correct_clitic = if let Some(v) = last_vowel(stem) {
                        if is_front_vowel(v) {
                            "de"
                        } else {
                            "da"
                        }
                    } else {
                        "de"
                    };
                    // in the token's case: KİTABI DA (review 3 GRAM3-06)
                    let all_caps = !tok.chars().any(char::is_lowercase);
                    let correct_clitic = if all_caps {
                        crate::phonology::to_turkish_upper(correct_clitic)
                    } else {
                        correct_clitic.to_string()
                    };
                    let replacement = format!("{original_stem} {correct_clitic}");
                    findings.push(GrammarFinding::new(
                            GrammarRule::DeDaAfterVerb,
                            (start, end),
                            tok.to_string(),
                            replacement,
                            "Fiillerden sonra gelen 'de/da' bağlaçtır ve ayrı yazılır.".to_string(),
                            "The clitic 'de/da' following finite verbs is a conjunction and must be written separately.".to_string(),
                        ));
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
        rules: RuleSet,
        findings: &mut Vec<GrammarFinding>,
    ) {
        const MERGED_KI_EXCEPTIONS: &[(&str, &str)] = &[
            ("yazıkki", "yazık ki"),
            ("demekki", "demek ki"),
            ("tabiiki", "tabii ki"),
            ("tabiki", "tabii ki"),
            ("varki", "var ki"),
            ("iyiki", "iyi ki"),
            ("yeterki", "yeter ki"),
            ("öyleki", "öyle ki"),
        ];

        for (i, &tok) in tokens.iter().enumerate() {
            let lower = to_turkish_lower(tok);
            let (start, end) = token_spans[i];

            // Case A: SOMBAHÇEMİ words erroneously written separately (e.g. "san ki", "oysa ki", "madem ki")
            if rules.contains(GrammarRule::KiLexicalizedSplit) && lower == "ki" && i > 0 {
                let prev_tok = to_turkish_lower(tokens[i - 1]);
                let compound = format!("{}{}", prev_tok, "ki");
                if SOMBAHCEMI_EXCEPTIONS.contains(compound.as_str()) {
                    let prev_start = token_spans[i - 1].0;
                    let replacement = match_case(tokens[i - 1], &compound);

                    findings.push(GrammarFinding::new(
                        GrammarRule::KiLexicalizedSplit,
                        (prev_start, end),
                        format!("{} {}", tokens[i - 1], tok),
                        replacement,
                        format!("'{compound}' sözcüğündeki 'ki' kalıplaşmış olduğu için bitişik yazılır."),
                        format!("'{compound}' is a lexicalized exception and must be written attached."),
                    ));
                }
            }

            // Case B: Merged non-lexicalized ki (e.g. "yazıkki", "demekki", "tabiiki")
            for &(merged, split) in MERGED_KI_EXCEPTIONS {
                if rules.contains(GrammarRule::KiMergedFixed) && lower == merged {
                    let replacement = match_case(tok, split);
                    findings.push(GrammarFinding::new(
                        GrammarRule::KiMergedFixed,
                        (start, end),
                        tok.to_string(),
                        replacement,
                        "Bağlaç olan 'ki' ayrı yazılır.".to_string(),
                        "The conjunction 'ki' must be written separately.".to_string(),
                    ));
                }
            }

            // Case C: Attached "ki" on verbs (e.g. "duydumki", "gördümki", "biliyorki"): only on
            // a word the analyser does not know, whose stem is a finite verb. Words (yetki,
            // tepki, öteki) and the relative -ki (sözleşmedeki, yılki) have a reading.
            if rules.contains(GrammarRule::KiAfterVerb)
                && lower.ends_with("ki")
                && lower.len() > 4
                && !SOMBAHCEMI_EXCEPTIONS.contains(lower.as_str())
            {
                let stem = &lower[..lower.len() - 2];
                if !is_known(morphology, tok) && is_finite_verb(morphology, stem) {
                    // DUYDUMKİ: the İ is two bytes, "ki" in the lower-cased token one each
                    let (original_stem, original_ki) = split_lower_suffix(tok, 2);
                    let replacement = format!("{original_stem} {original_ki}");
                    findings.push(GrammarFinding::new(
                        GrammarRule::KiAfterVerb,
                        (start, end),
                        tok.to_string(),
                        replacement,
                        "Fiillerden sonra gelen 'ki' bağlaçtır ve ayrı yazılır.".to_string(),
                        "The clitic 'ki' following verbs is a conjunction and must be written separately.".to_string(),
                    ));
                }
            }
        }
    }

    /// 5. Apostrophe & Proper Noun Rules
    fn check_apostrophes(
        text: &str,
        _tokens: &[&str],
        _token_spans: &[(usize, usize)],
        morphology: &TurkishMorphology,
        rules: RuleSet,
        findings: &mut Vec<GrammarFinding>,
    ) {
        // Full-text StringZilla SIMD search for double ordinal suffixes on numbers (e.g. "2.'nci" -> "2'nci")
        let ordinal_double_patterns = [
            ".'nci", ".'ncı", ".'ncu", ".'ncü", ".'inci", ".'ıncı", ".'uncu", ".'üncü",
        ];
        for &pat in &ordinal_double_patterns {
            if !rules.contains(GrammarRule::OrdinalDotApostrophe) {
                break;
            }
            if let Some(offset) = text.sz_find(pat) {
                let prefix = &text[..offset];
                // where the digits before the pattern start (the character before them can be
                // more than one byte: "ş2.'nci")
                let digit_start = prefix.trim_end_matches(|c: char| c.is_ascii_digit()).len();
                let num_str = &prefix[digit_start..];
                if !num_str.is_empty() {
                    let end = offset + pat.len();
                    let original_snippet = &text[digit_start..end];
                    let fixed_suf = pat.replace('.', "");
                    let replacement = format!("{num_str}{fixed_suf}");
                    findings.push(GrammarFinding::new(
                        GrammarRule::OrdinalDotApostrophe,
                        (digit_start, end),
                        original_snippet.to_string(),
                        replacement,
                        "Sıra sayıları yazılırken hem nokta hem de kesme eki birlikte kullanılmaz."
                            .to_string(),
                        "Both period and ordinal suffix should not be used together on numbers."
                            .to_string(),
                    ));
                }
            }
        }

        // Every apostrophe, with the root before it back to the last word break and the
        // letters after it, in one pass over the text. The word break is tracked as the scan
        // goes: searching back from each apostrophe was quadratic in a run without breaks
        // ("x'x'x'..."), and it assumed one-byte breaks, so a no-break or thin space before
        // the apostrophe left the root starting inside that space.
        let is_word_break = |c: char| c.is_whitespace() || matches!(c, '.' | ',' | '(' | '"' | '!');
        let mut word_start = 0;
        for (apos_pos, c) in text.char_indices() {
            if is_word_break(c) {
                word_start = apos_pos + c.len_utf8();
                continue;
            }
            if c != '\'' {
                continue;
            }
            let start = word_start;
            let root = &text[start..apos_pos];
            if root.is_empty() || root.len() > MAX_APOSTROPHE_WORD {
                continue;
            }

            // the letters after it: this stops at the next apostrophe, so the scans together
            // read the text once
            let suffix = &text[apos_pos + 1..];
            let suf_len = suffix
                .find(|c: char| !c.is_alphabetic())
                .unwrap_or(suffix.len());
            if suf_len > MAX_APOSTROPHE_WORD {
                continue;
            }
            let suf = &suffix[..suf_len];
            let end = apos_pos + 1 + suf_len;

            if !suf.is_empty() {
                // A. Number apostrophe assimilation (e.g. 1923'de -> 1923'te, 1940'da ->
                // 1940'ta). The digits after the last "." or "," are read as they are (the year
                // of 15.03.1993'te, the minutes of 15.30'da, the 500 of 2.500'de), except a
                // group of zeros, which a larger unit before it names (1.000'de: bin).
                if root.chars().all(|c| c.is_ascii_digit()) {
                    if !rules.contains(GrammarRule::NumberSuffix)
                        || text[..start].ends_with(['.', ',']) && root.bytes().all(|b| b == b'0')
                    {
                        continue;
                    }
                    let Some((voiceless, front)) = number_reading(root) else {
                        continue;
                    };
                    let bytes = suf.as_bytes();
                    let is_loc_abl = bytes.len() >= 2
                        && matches!(bytes[0], b'd' | b't')
                        && matches!(bytes[1], b'e' | b'a');
                    if !is_loc_abl {
                        continue;
                    }
                    let expected = format!(
                        "{}{}",
                        if voiceless { 't' } else { 'd' },
                        if front { 'e' } else { 'a' }
                    );
                    if suf[..2] != expected {
                        let replacement = format!("{root}'{expected}{}", &suf[2..]);
                        findings.push(GrammarFinding::new(
                            GrammarRule::NumberSuffix,
                            (start, end),
                            text[start..end].to_string(),
                            replacement,
                            "Sayılara gelen ekler sayının okunuşuna uyar: son sözcük sert ünsüzle bitiyorsa ek sertleşir, ünlüsü de ünlü uyumuna göre seçilir (1923'te, 1940'ta, 1990'da).".to_string(),
                            "A suffix on a number follows how the number is read: it hardens after a voiceless consonant and its vowel follows vowel harmony (1923'te, 1940'ta, 1990'da).".to_string(),
                        ));
                    }
                } else if rules.contains(GrammarRule::ApostropheDerivational)
                    || rules.contains(GrammarRule::ApostrophePluralName)
                {
                    // B. Plural or derivational suffix on a proper noun (Türk'ler -> Türkler,
                    // Ankara'lı -> Ankaralı, Ahmetler'in -> Ahmetlerin). Only on a name written
                    // as one: a capital and then lower-case letters. Abbreviations (KHK'ler,
                    // CHP'li), units (km'lik) and numbers (2-1'lik) keep the apostrophe.
                    // (two letters is an abbreviation more often than a name: No'lu, Lü)
                    let mut chars = root.chars();
                    let is_name = chars.next().is_some_and(char::is_uppercase)
                        && chars.clone().count() > 1
                        && chars.all(char::is_lowercase);
                    if !is_name {
                        continue;
                    }
                    let root_lower = to_turkish_lower(root);
                    let suf_lower = to_turkish_lower(suf);
                    // a plural name before the apostrophe: the name without -ler is a proper
                    // noun and the whole is not a word (Ahmetler'in; not Hitler'in, Kepler'in)
                    let is_plural_name = rules.contains(GrammarRule::ApostrophePluralName)
                        && ["ler", "lar"].iter().any(|pl| {
                            root_lower.strip_suffix(pl).is_some_and(|base| {
                                morphology
                                    .root_items(base)
                                    .iter()
                                    .any(|item| item.secondary_pos == SecondaryPos::ProperNoun)
                            })
                        })
                        && morphology.root_items(&root_lower).is_empty();
                    let is_plural_or_deriv_suffix = rules
                        .contains(GrammarRule::ApostropheDerivational)
                        && [
                            "ler", "lar", "li", "lı", "lu", "lü", "siz", "sız", "suz", "süz",
                            "lik", "lık", "luk", "lük",
                        ]
                        .iter()
                        .any(|s| suf_lower.starts_with(s));

                    if is_plural_name || is_plural_or_deriv_suffix {
                        let replacement = format!("{root}{suf}");
                        findings.push(GrammarFinding::new(
                            if is_plural_or_deriv_suffix {
                                GrammarRule::ApostropheDerivational
                            } else {
                                GrammarRule::ApostrophePluralName
                            },
                            (start, end),
                            text[start..end].to_string(),
                            replacement,
                            "Özel isimlere gelen yapım ekleri, çokluk eki (-ler/-lar) ve bunlardan sonra gelen ekler kesme işaretiyle ayrılmaz.".to_string(),
                            "Plural and derivational suffixes on proper nouns (and following suffixes) do not take an apostrophe.".to_string(),
                        ));
                    }
                }
            }
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
                    findings.push(GrammarFinding::new(
                        GrammarRule::Reduplication,
                        (start, end),
                        tok.to_string(),
                        match_case(tok, split_word),
                        "İkilemeler ayrı yazılır.".to_string(),
                        "Reduplications must be written separately.".to_string(),
                    ));
                }
            }
        }
    }

    /// 7. Modal Verb `-ebilmek / -abilmek` (Label 20): Must be written attached (*ola bilir* -> *olabilir*, *yapa bilmek* -> *yapabilmek*)
    ///
    /// Only when the two words written together are an ability verb the analyser knows: a word
    /// ending in -a/-e followed by *bile*, *bildir-*, *bilen* or *bilirkişi* is no split
    /// *-ebil* ("Ona bile", "Mahkemeye bildirdi", "Türkçe bilen").
    fn check_modal_verb_ebilmek(
        _text: &str,
        tokens: &[&str],
        token_spans: &[(usize, usize)],
        morphology: &TurkishMorphology,
        findings: &mut Vec<GrammarFinding>,
    ) {
        for (i, &tok) in tokens.iter().enumerate() {
            if i == 0 {
                continue;
            }
            let lower = to_turkish_lower(tok);
            let prev_lower = to_turkish_lower(tokens[i - 1]);

            if (lower.starts_with("bilir")
                || lower.starts_with("bile")
                || lower.starts_with("bildi")
                || lower.starts_with("bilecek")
                || lower.starts_with("bilse")
                || lower.starts_with("bilmiş")
                || lower.starts_with("biliyor")
                || lower.starts_with("bilsin")
                || lower.starts_with("bilen")
                || lower.starts_with("bilip")
                || lower.starts_with("bilmek"))
                && (prev_lower.ends_with('a') || prev_lower.ends_with('e'))
                && prev_lower.chars().all(char::is_alphabetic)
                // "bile bile" (knowingly) is a reduplication, and bile alone is the adverb
                // "even": ona-bile reads as ona-bil-e (onamak), a form nobody means
                && prev_lower != "bile"
                && lower != "bile"
                && morphology
                    .analyze_standard(&format!("{prev_lower}{lower}"))
                    .iter()
                    .any(|p| p.morpheme_tags.iter().any(|t| t == "Abil"))
            {
                let prev_start = token_spans[i - 1].0;
                let end = token_spans[i].1;
                let replacement = format!("{}{}", tokens[i - 1], tok);
                findings.push(GrammarFinding::new(
                    GrammarRule::EbilSplit,
                    (prev_start, end),
                    format!("{} {}", tokens[i - 1], tok),
                    replacement,
                    "Yeterlik fiili (-ebilmek/-abilmek) fiil kök veya gövdesine bitişik yazılır."
                        .to_string(),
                    "The potential modal verb suffix (-ebilmek/-abilmek) must be written attached."
                        .to_string(),
                ));
            }
        }
    }

    /// 8. Indefinite Pronouns & Determiners (Label 22): *bir çok* -> *birçok*, *bir kaç* -> *birkaç*, *hiç bir* -> *hiçbir*
    fn check_indefinite_compounds(
        _text: &str,
        tokens: &[&str],
        token_spans: &[(usize, usize)],
        morphology: &TurkishMorphology,
        findings: &mut Vec<GrammarFinding>,
    ) {
        for (i, &tok) in tokens.iter().enumerate() {
            if i == 0 {
                continue;
            }
            let lower = to_turkish_lower(tok);
            let prev_lower = to_turkish_lower(tokens[i - 1]);
            let prev_start = token_spans[i - 1].0;
            let end = token_spans[i].1;

            // "bir takım" is also "a team" or "a set": only before a plural ("bir takım
            // kurallar"), not in "bir takım gönderip", "baskın bir takım hâline geldi"
            let plural = |t: &str| {
                morphology
                    .analyze_standard(t)
                    .iter()
                    .any(|p| p.morpheme_tags.iter().any(|t| t == "A3pl"))
            };
            let nominal = |t: &str| {
                morphology
                    .analyze_standard(t)
                    .iter()
                    .any(|p| matches!(p.pos, PrimaryPos::Noun | PrimaryPos::Adj))
            };
            let plural_follows = || match tokens.get(i + 1..i + 3) {
                Some([next, after]) => plural(next) || nominal(next) && plural(after),
                _ => tokens.get(i + 1).is_some_and(|t| plural(t)),
            };
            let compound = match (prev_lower.as_str(), lower.as_str()) {
                ("bir", "çok") => Some("birçok"),
                ("bir", "kaç") => Some("birkaç"),
                ("bir", "az") => Some("biraz"),
                ("bir", "takım") if plural_follows() => Some("birtakım"),
                ("hiç", "bir") => Some("hiçbir"),
                ("hiç", "biri") => Some("hiçbiri"),
                ("hiç", "birisi") => Some("hiçbirisi"),
                ("her", "hangi") => Some("herhangi"),
                ("her", "hangisi") => Some("herhangisi"),
                _ => None,
            };

            if let Some(comp_str) = compound {
                let replacement = match_case(tokens[i - 1], comp_str);
                findings.push(GrammarFinding::new(
                    GrammarRule::IndefiniteCompound,
                    (prev_start, end),
                    format!("{} {}", tokens[i - 1], tok),
                    replacement,
                    format!("'{comp_str}' belgisiz sözcüğü bitişik yazılır."),
                    format!("'{comp_str}' is a compound determiner/pronoun and must be written attached."),
                ));
            }
        }
    }
}
