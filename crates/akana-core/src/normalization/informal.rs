//! Normalization for informal Turkish, spoken contractions, slang, and repetitive elongated characters.

use std::collections::HashMap;
use lazy_static::lazy_static;
use crate::morphology::TurkishMorphology;
use crate::phonology::to_turkish_lower;

lazy_static! {
    static ref GLOBAL_MORPHOLOGY: TurkishMorphology = TurkishMorphology::new();

    /// Common Turkish informal spoken contractions -> standard Turkish written forms
    static ref INFORMAL_CONTRACTIONS: HashMap<&'static str, &'static str> = {
        let mut m = HashMap::new();
        let contractions = [
            ("tmm", "tamam"), ("tmmdır", "tamamdır"), ("ok", "tamam"), ("oki", "tamam"),
            ("nbr", "ne haber"), ("naber", "ne haber"), ("slm", "selam"), ("mrb", "merhaba"),
            ("meraba", "merhaba"), ("tşk", "teşekkürler"), ("tskk", "teşekkürler"),
            ("tşkler", "teşekkürler"), ("teşekkür", "teşekkür ederim"), ("eyw", "eyvallah"),
            ("kib", "kendine iyi bak"), ("hoscakal", "hoşça kal"), ("hoşçakal", "hoşça kal"),
            ("aynen", "aynen"), ("aynn", "aynen"), ("ztn", "zaten"), ("bgn", "bugün"),
            ("yarin", "yarın"), ("inş", "inşallah"), ("hçbr", "hiçbir"), ("hç", "hiç"),
            ("bşy", "bir şey"), ("bişey", "bir şey"), ("herşey", "her şey"),
            ("noldu", "ne oldu"), ("nooldu", "ne oldu"), ("nolur", "ne olur"),
            ("naptın", "ne yaptın"), ("napıyon", "ne yapıyorsun"), ("napıyorsun", "ne yapıyorsun"),
            ("napıon", "ne yapıyorsun"), ("nasılsın", "nasılsın"), ("nslsn", "nasılsın"),
            ("napıyonuz", "ne yapıyorsunuz"), ("hadi", "hadi"),
        ];
        for (k, v) in contractions {
            m.insert(k, v);
        }
        m
    };
}

pub struct TurkishInformalNormalizer;

impl TurkishInformalNormalizer {
    /// Deduplicates exaggerated repetitive characters (e.g. "çooookk" -> "çok", "harikaaaa" -> "harika").
    pub fn deduplicate_repeated_chars(word: &str) -> String {
        let mut result = String::with_capacity(word.len());
        let mut last_char = None;
        let mut count = 0;

        for c in word.chars() {
            if Some(c) == last_char {
                count += 1;
                if count <= 2 {
                    result.push(c);
                }
            } else {
                last_char = Some(c);
                count = 1;
                result.push(c);
            }
        }

        // If the 2-char deduped word is not in the dictionary, try reducing to single chars
        if count >= 2 {
            let single_chars = Self::reduce_all_consecutive_duplicates(word);
            if !GLOBAL_MORPHOLOGY.analyze(&to_turkish_lower(&single_chars)).is_empty() {
                return single_chars;
            }
        }

        result
    }

    fn reduce_all_consecutive_duplicates(word: &str) -> String {
        let mut res = String::with_capacity(word.len());
        let mut last_c = None;
        for c in word.chars() {
            if Some(c) != last_c {
                res.push(c);
                last_c = Some(c);
            }
        }
        res
    }

    /// Normalizes a single informal Turkish word if it is a known spoken contraction or elongated form.
    pub fn normalize_word(word: &str) -> String {
        let deduped = Self::deduplicate_repeated_chars(word);
        let lower = to_turkish_lower(&deduped);

        if let Some(&formal) = INFORMAL_CONTRACTIONS.get(lower.as_str()) {
            return formal.to_string();
        }

        // Apply productive future contraction rules (yapcam -> yapacağım, gelcem -> geleceğim, okucam -> okuyacağım)
        if lower.ends_with("cam") || lower.ends_with("can") || lower.ends_with("caz") || lower.ends_with("cak") || lower.ends_with("caklar") ||
           lower.ends_with("cem") || lower.ends_with("cen") || lower.ends_with("cez") || lower.ends_with("cek") || lower.ends_with("cekler") {
            let (stem, ending) = if let Some(idx) = lower.rfind('c') {
                (&lower[..idx], &lower[idx+1..])
            } else {
                ("", "")
            };

            if !stem.is_empty() {
                let last_vowel = stem.chars().rev().find(|&c| super::super::phonology::is_turkish_vowel(c)).unwrap_or('a');
                let is_back = super::super::phonology::is_back_vowel(last_vowel);
                let stem_ends_with_vowel = super::super::phonology::is_turkish_vowel(stem.chars().last().unwrap());

                let buffer = if stem_ends_with_vowel { "y" } else if is_back { "a" } else { "e" };
                let harmonic_c = if is_back { "c" } else { "c" };
                let suffix = match ending {
                    "am" => if is_back { "ağım" } else { "eceğim" },
                    "an" => if is_back { "aksın" } else { "eksin" },
                    "az" => if is_back { "ağız" } else { "eceğiz" },
                    "ak" => if is_back { "ak" } else { "ek" },
                    "aklar" => if is_back { "aklar" } else { "ekler" },
                    "em" => "eceğim",
                    "en" => "eksin",
                    "ez" => "eceğiz",
                    "ek" => "ek",
                    "ekler" => "ekler",
                    _ => ending,
                };

                let candidate = if ending == "am" {
                    format!("{}{}{}", stem, buffer, if is_back { "cağım" } else { "ceğim" })
                } else if ending == "em" {
                    format!("{}{}{}", stem, buffer, "ceğim")
                } else if ending == "az" {
                    format!("{}{}{}", stem, buffer, if is_back { "cağız" } else { "ceğiz" })
                } else if ending == "ez" {
                    format!("{}{}{}", stem, buffer, "ceğiz")
                } else {
                    format!("{}{}{}{}", stem, buffer, harmonic_c, suffix)
                };

                if !GLOBAL_MORPHOLOGY.analyze(&candidate).is_empty() {
                    return candidate;
                }
            }
        }

        // Fast progressive contractions (geliyom -> geliyorum, biliyo -> biliyor, yapıyom -> yapıyorum)
        let prog_endings = [
            ("yolar", "yorlar"),
            ("yonuz", "yorsunuz"),
            ("yom", "yorum"),
            ("yon", "yorsun"),
            ("yoz", "yoruz"),
            ("yo", "yor"),
        ];

        for (end, repl) in prog_endings {
            if lower.ends_with(end) && lower.len() > end.len() {
                let stem = &lower[..lower.len() - end.len()];
                let candidate = format!("{}{}", stem, repl);
                if !GLOBAL_MORPHOLOGY.analyze(&candidate).is_empty() {
                    return candidate;
                }
            }
        }

        // Fast negative progressive contractions (gelmiom -> gelmiyorum, yapmıom -> yapmıyorum)
        let neg_prog_endings = [
            ("mıom", "mıyorum"),
            ("miom", "miyorum"),
            ("muom", "muyorum"),
            ("müom", "müyorum"),
        ];

        for (end, repl) in neg_prog_endings {
            if lower.ends_with(end) && lower.len() > end.len() {
                let stem = &lower[..lower.len() - end.len()];
                let candidate = format!("{}{}", stem, repl);
                if !GLOBAL_MORPHOLOGY.analyze(&candidate).is_empty() {
                    return candidate;
                }
            }
        }

        deduped
    }

    /// Normalizes an entire informal Turkish text.
    pub fn normalize_text(text: &str) -> String {
        let tokens: Vec<&str> = text.split_inclusive(|c: char| !c.is_alphabetic()).collect();
        let mut result = String::with_capacity(text.len());

        for token in tokens {
            let alphabetic_part: String = token.chars().filter(|c| c.is_alphabetic()).collect();
            let non_alphabetic_part: String = token.chars().filter(|c| !c.is_alphabetic()).collect();

            if alphabetic_part.is_empty() {
                result.push_str(token);
                continue;
            }

            let normalized = Self::normalize_word(&alphabetic_part);
            result.push_str(&normalized);
            result.push_str(&non_alphabetic_part);
        }

        result
    }
}
