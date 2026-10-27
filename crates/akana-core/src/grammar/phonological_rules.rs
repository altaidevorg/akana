//! Turkish Phonological Grammar Rules Engine.
//!
//! Evaluates and corrects:
//! 1. Vowel Harmony violations in suffixation (Major & Minor Vowel Harmony).
//! 2. Consonant Assimilation / Hardening (*fıstıkçı şahap* rule: *kitapda* -> *kitapta*).
//! 3. Consonant Softening / Voicing (*kitapı* -> *kitabı*, *ağaca*).
//! 4. Vowel Dropping / Syncope (*akılı* -> *aklı*, *şehire* -> *şehre*).
//! 5. Vowel Narrowing in progressive tense (*başlayor* -> *başlıyor*).

use lazy_static::lazy_static;
use std::collections::HashMap;

use super::{match_case, GrammarFinding, GrammarRule, RuleSet};
use crate::morphology::pos::PrimaryPos;
use crate::morphology::{MorphParse, TurkishMorphology};
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

/// Whether `word` (as written) is a misspelling of `fixed`: the analyser knows no word `word`
/// (it rejects the forms a sound change rules out: kitap-da, akıl-ı, kitap-ı) and reads
/// `fixed` through `root` (with that POS, when given). A word with a reading of its own is left
/// alone: takdir and nakden are roots, akılı is akı-lı, kalpak and hükümet have other roots.
fn misspelling_of(
    morphology: &TurkishMorphology,
    word: &str,
    fixed: &str,
    root: &str,
    pos: Option<PrimaryPos>,
) -> bool {
    morphology.analyze_standard(word).is_empty()
        && morphology
            .analyze_standard(fixed)
            .iter()
            .any(|p: &MorphParse| p.root == root && pos.is_none_or(|pos| p.primary_pos == pos))
}

pub struct PhonologicalRuleEngine;

impl PhonologicalRuleEngine {
    pub fn check(
        _text: &str,
        tokens: &[&str],
        token_spans: &[(usize, usize)],
        morphology: &TurkishMorphology,
        rules: RuleSet,
        findings: &mut Vec<GrammarFinding>,
    ) {
        for (i, &tok) in tokens.iter().enumerate() {
            let lower = to_turkish_lower(tok);
            let (start, end) = token_spans[i];

            // Skip punctuation or numbers
            if lower.is_empty() || lower.chars().all(|c| !c.is_alphabetic()) {
                continue;
            }
            // A capital inside a sentence starts a name (Nahda, Selimiye, Altın Orda), which the
            // lexicon's rules for common words do not cover.
            let sentence_start = i == 0 || matches!(tokens[i - 1], "." | "!" | "?" | "…");
            if !sentence_start && tok.chars().next().is_some_and(char::is_uppercase) {
                continue;
            }

            // 1. Check Consonant Assimilation (fıstıkçı şahap: e.g. "kitapda", "sokakdan", "yapdı", "gitdi")
            if rules.contains(GrammarRule::ConsonantAssimilation)
                && Self::check_consonant_assimilation(tok, &lower, start, end, morphology, findings)
            {
                continue;
            }

            // 2. Check Vowel Dropping (Syncope: e.g. "akılı", "şehire", "burunu")
            if rules.contains(GrammarRule::VowelDrop)
                && Self::check_vowel_drop(tok, &lower, start, end, morphology, findings)
            {
                continue;
            }

            // 3. Check Vowel Narrowing in progressive tense (e.g. "başlayor", "anlamayor")
            if Self::check_vowel_narrowing(tok, &lower, start, end, morphology, rules, findings) {
                continue;
            }

            // 4. Check Consonant Softening (e.g. "kitapı" -> "kitabı", "ağaca")
            if rules.contains(GrammarRule::ConsonantSoftening)
                && Self::check_consonant_softening(tok, &lower, start, end, morphology, findings)
            {
                continue;
            }

            // 5. Check Suffix Vowel Harmony (e.g. "evlar", "arabaler")
            if rules.contains(GrammarRule::PluralHarmony) {
                Self::check_suffix_vowel_harmony(tok, &lower, start, end, morphology, findings);
            }
        }
    }

    /// 1. Consonant Assimilation / Hardening:
    ///
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
            ("da", "ta"),
            ("de", "te"),
            ("dan", "tan"),
            ("den", "ten"),
            ("dı", "tı"),
            ("di", "ti"),
            ("du", "tu"),
            ("dü", "tü"),
            ("dır", "tır"),
            ("dir", "tir"),
            ("dur", "tur"),
            ("dür", "tür"),
            ("ca", "ça"),
            ("ce", "çe"),
        ];

        for (bad_suf, good_suf) in &unassimilated_suffixes {
            if lower.ends_with(bad_suf) && lower.len() > bad_suf.len() + 2 {
                let stem = &lower[..lower.len() - bad_suf.len()];
                if let Some(stem_last_char) = stem.chars().last() {
                    if is_hard_consonant(stem_last_char) {
                        // Not a word, and the hardened form is the stem plus the suffix: not
                        // takdir, nakden or akdi, whose roots are takdir, nakit and akit.
                        let fixed_lower = format!("{stem}{good_suf}");
                        if misspelling_of(morphology, original, &fixed_lower, stem, None) {
                            let replacement = match_case(original, &fixed_lower);

                            findings.push(GrammarFinding::new(
                                GrammarRule::ConsonantAssimilation,
                                (start, end),
                                original.to_string(),
                                replacement,
                                "Sert ünsüzle (f, s, t, k, ç, ş, h, p) biten sözcüklere gelen ekler sertleşir (ünsüz benzeşmesi).".to_string(),
                                "Suffixes following voiceless consonants must undergo consonant hardening.".to_string(),
                            ));
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    /// 2. Vowel Dropping (Syncope):
    ///
    /// Words like *akıl*, *burun*, *şehir* drop the second vowel when receiving a vowel-initial suffix (*akıl-ı* -> *aklı*).
    fn check_vowel_drop(
        original: &str,
        lower: &str,
        start: usize,
        end: usize,
        morphology: &TurkishMorphology,
        findings: &mut Vec<GrammarFinding>,
    ) -> bool {
        for (&root, &contracted) in VOWEL_DROP_ROOTS.iter() {
            if lower.starts_with(root) && lower.len() > root.len() {
                let suf = &lower[root.len()..];
                let fixed_lower = format!("{contracted}{suf}");
                if suf.chars().next().is_some_and(is_turkish_vowel)
                    && misspelling_of(
                        morphology,
                        original,
                        &fixed_lower,
                        root,
                        Some(PrimaryPos::Noun),
                    )
                {
                    {
                        let replacement = match_case(original, &fixed_lower);

                        findings.push(GrammarFinding::new(
                            GrammarRule::VowelDrop,
                            (start, end),
                            original.to_string(),
                            replacement,
                            format!("'{root}' sözcüğü ünlüyle başlayan ek aldığında ikinci hecedeki dar ünlü düşer (ünlü düşmesi)."),
                            format!("The second vowel in '{root}' drops when attached to a vowel-initial suffix."),
                        ));
                        return true;
                    }
                }
            }
        }
        false
    }

    /// 3. Vowel Narrowing:
    ///
    /// Progressive tense "-yor" narrows preceding open vowels 'a'/'e' (*başla-yor* -> *başlıyor*, *gözle-yor* -> *gözlüyor*).
    fn check_vowel_narrowing(
        original: &str,
        lower: &str,
        start: usize,
        end: usize,
        morphology: &TurkishMorphology,
        rules: RuleSet,
        findings: &mut Vec<GrammarFinding>,
    ) -> bool {
        // 3A. Check Progressive Tense Under-Narrowing (e.g. "başlayor" -> "başlıyor", "anlamayor" -> "anlamıyor")
        if rules.contains(GrammarRule::YorNarrowing)
            && (lower.ends_with("ayor") || lower.ends_with("eyor"))
        {
            let stem = &lower[..lower.len() - 4];
            let ending = &lower[lower.len() - 4..];
            if !stem.is_empty() {
                let verb_root = format!(
                    "{}{}",
                    stem,
                    if ending.starts_with('a') { "a" } else { "e" }
                );
                if morphology
                    .analyze_standard(&verb_root)
                    .iter()
                    .any(|p| p.primary_pos == crate::morphology::pos::PrimaryPos::Verb)
                {
                    let last_v = last_vowel(stem).unwrap_or('a');
                    let narrow_v = harmony_i_type(last_v);
                    let fixed_lower = format!("{stem}{narrow_v}yor");
                    let replacement = match_case(original, &fixed_lower);

                    findings.push(GrammarFinding::new(
                        GrammarRule::YorNarrowing,
                        (start, end),
                        original.to_string(),
                        replacement,
                        "'-yor' şimdiki zaman eki kendinden önceki 'a/e' geniş ünlülerini daraltır (ünlü daralması).".to_string(),
                        "The progressive suffix '-yor' narrows preceding open vowels 'a/e'.".to_string(),
                    ));
                    return true;
                }
            }
        }

        // 3B. Check Over-Narrowing in non-progressive suffixes (Label 12: e.g. "başlıyan" -> "başlayan", "yapmıya" -> "yapmaya")
        const OVER_NARROWED_PATTERNS: &[(&str, &str)] = &[
            ("mıyanlar", "mayanlar"),
            ("miyenler", "meyenler"),
            ("mıyan", "mayan"),
            ("miyen", "meyen"),
            ("mıyarak", "mayarak"),
            ("miyerek", "meyerek"),
            ("mıyalım", "mayalım"),
            ("miyelim", "meyelim"),
            ("mıya", "maya"),
            ("miye", "meye"),
            ("başlıyan", "başlayan"),
            ("söyliyen", "söyleyen"),
            ("bekliyen", "bekleyen"),
            ("yaşıyan", "yaşayan"),
            ("anlıyan", "anlayan"),
            ("istiyen", "isteyen"),
            ("izliyen", "izleyen"),
            ("dinliyen", "dinleyen"),
            ("kutlıyan", "kutlayan"),
            ("özliyen", "özleyen"),
            ("olmıyan", "olmayan"),
            ("görmiyen", "görmeyen"),
            ("bilmiyen", "bilmeyen"),
            ("gelmiyen", "gelmeyen"),
        ];

        for &(narrowed, corrected) in OVER_NARROWED_PATTERNS {
            if !rules.contains(GrammarRule::OverNarrowing) {
                break;
            }
            if let Some(stem) = lower.strip_suffix(narrowed) {
                let fixed_lower = format!("{stem}{corrected}");
                // a verb form: not the nouns and numerals that end alike (ekonomiye, yevmiye,
                // yirmiye, ikramiye), and the corrected form is a word
                let parses = morphology.analyze_standard(original);
                if parses.iter().any(|p| p.primary_pos != PrimaryPos::Verb)
                    || morphology.analyze_standard(&fixed_lower).is_empty()
                {
                    continue;
                }
                let replacement = match_case(original, &fixed_lower);
                findings.push(GrammarFinding::new(
                    GrammarRule::OverNarrowing,
                    (start, end),
                    original.to_string(),
                    replacement,
                    "Yazı dilinde '-yor' dışındaki eklerde (sıfat-fiil, isim-fiil) ünlü daralması gösterilmez.".to_string(),
                    "Vowel narrowing should only be written before '-yor', not in participles or nominals.".to_string(),
                ));
                return true;
            }
        }

        // 3C. Check Colloquial Pronoun Syncopes (Label 14: e.g. "burda" -> "burada", "dışarda" -> "dışarıda")
        const COLLOQUIAL_SYNCOPES: &[(&str, &str)] = &[
            ("burda", "burada"),
            ("burdan", "buradan"),
            ("burdaki", "buradaki"),
            ("şurda", "şurada"),
            ("şurdan", "şuradan"),
            ("şurdaki", "şuradaki"),
            ("orda", "orada"),
            ("ordan", "oradan"),
            ("ordaki", "oradaki"),
            ("dışarda", "dışarıda"),
            ("dışardan", "dışarıdan"),
            ("dışardaki", "dışarıdaki"),
            ("içerde", "içeride"),
            ("içerden", "içeriden"),
            ("içerdeki", "içerideki"),
            ("yukarda", "yukarıda"),
            ("yukardan", "yukarıdan"),
            ("yukardaki", "yukarıdaki"),
            ("ilerde", "ileride"),
            ("ilerden", "ileriden"),
            ("ilerdeki", "ilerideki"),
            ("yalınız", "yalnız"),
            ("yanlız", "yalnız"),
            ("yalnış", "yanlış"),
        ];

        for &(colloquial, standard) in COLLOQUIAL_SYNCOPES {
            if rules.contains(GrammarRule::ColloquialForm) && lower == colloquial {
                let replacement = match_case(original, standard);
                findings.push(GrammarFinding::new(
                    GrammarRule::ColloquialForm,
                    (start, end),
                    original.to_string(),
                    replacement,
                    "Konuşma dilindeki ünlü düşmesi yazı dilinde gösterilmez.".to_string(),
                    "Colloquial syncope should not be written in standard Turkish orthography."
                        .to_string(),
                ));
                return true;
            }
        }

        false
    }

    /// 4. Consonant Softening / Voicing:
    ///
    /// When words ending in p, ç, t, k take a vowel-initial suffix, they mutate to b, c, d, ğ/g (*kitap-ı* -> *kitabı*).
    fn check_consonant_softening(
        original: &str,
        lower: &str,
        start: usize,
        end: usize,
        morphology: &TurkishMorphology,
        findings: &mut Vec<GrammarFinding>,
    ) -> bool {
        let softenable_roots = [
            ("kitap", "kitab"),
            ("ağaç", "ağac"),
            ("kanat", "kanad"),
            ("renk", "reng"),
            ("çocuk", "çocuğ"),
            ("ayak", "ayağ"),
            ("ekmek", "ekmeğ"),
            ("köpek", "köpeğ"),
            ("dolap", "dolab"),
            ("kalp", "kalb"),
            ("hesap", "hesab"),
            ("cevap", "cevab"),
            ("kulak", "kulağ"),
            ("bacak", "bacağ"),
            ("çiçek", "çiçeğ"),
            ("tarak", "tarağ"),
        ];

        for (unmutated, mutated) in &softenable_roots {
            if lower.starts_with(unmutated) && lower.len() > unmutated.len() {
                let suf = &lower[unmutated.len()..];
                let fixed_lower = format!("{mutated}{suf}");
                // not kalpak, ayaküstü: words with another root
                if suf.chars().next().is_some_and(is_turkish_vowel)
                    && misspelling_of(morphology, original, &fixed_lower, unmutated, None)
                {
                    {
                        let replacement = match_case(original, &fixed_lower);

                        findings.push(GrammarFinding::new(
                            GrammarRule::ConsonantSoftening,
                            (start, end),
                            original.to_string(),
                            replacement,
                            format!("'{unmutated}' sözcüğü ünlüyle başlayan ek aldığında sonundaki ünsüz yumuşar (ünsüz yumuşaması)."),
                            format!("The final consonant in '{unmutated}' undergoes voicing when a vowel-initial suffix is added."),
                        ));
                        return true;
                    }
                }
            }
        }
        false
    }

    /// 5. Suffix Vowel Harmony (Major/Minor Vowel Harmony):
    ///
    /// Suffixes must match backness/frontness of the stem (*araba-ler* -> *arabalar*, *ev-lar* -> *evler*).
    fn check_suffix_vowel_harmony(
        original: &str,
        lower: &str,
        start: usize,
        end: usize,
        morphology: &TurkishMorphology,
        findings: &mut Vec<GrammarFinding>,
    ) {
        // Only check words unrecognized by morphology (potential harmony typo), and not a
        // name or abbreviation with a suffix after an apostrophe (SoC'ler, Beat'ler)
        if lower.contains(['\'', '’']) || !morphology.analyze_standard(original).is_empty() {
            return;
        }

        // Test plural suffix mismatch (-ler vs -lar)
        if lower.ends_with("ler") || lower.ends_with("lar") {
            let stem = &lower[..lower.len() - 3];
            if let Some(stem_v) = last_vowel(stem) {
                let expected_pl = if is_front_vowel(stem_v) { "ler" } else { "lar" };
                let current_pl = &lower[lower.len() - 3..];
                if current_pl != expected_pl {
                    let fixed_lower = format!("{stem}{expected_pl}");
                    if !morphology.analyze_standard(&fixed_lower).is_empty() {
                        let replacement = match_case(original, &fixed_lower);
                        findings.push(GrammarFinding::new(
                            GrammarRule::PluralHarmony,
                            (start, end),
                            original.to_string(),
                            replacement,
                            format!("Çoğul eki büyük ünlü uyumuna göre '-{expected_pl}' olmalıdır."),
                            format!("The plural suffix must be '-{expected_pl}' to satisfy vowel harmony."),
                        ));
                    }
                }
            }
        }
    }
}
