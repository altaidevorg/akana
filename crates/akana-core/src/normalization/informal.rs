//! Normalization for informal Turkish, spoken contractions, and repetitive elongated characters.

use std::collections::HashMap;
use lazy_static::lazy_static;

lazy_static! {
    /// Common Turkish informal spoken contractions -> standard Turkish written forms
    static ref INFORMAL_CONTRACTIONS: HashMap<&'static str, &'static str> = {
        let mut m = HashMap::new();
        let contractions = [
            ("yapcam", "yapacağım"), ("yapcan", "yapacaksın"),
            ("yapcaz", "yapacağız"), ("yapmicam", "yapmayacağım"), ("yapmicak", "yapmayacak"),
            ("geliyom", "geliyorum"), ("geliyon", "geliyorsun"), ("geliyo", "geliyor"),
            ("gidiyom", "gidiyorum"), ("gidiyon", "gidiyorsun"), ("gidiyo", "gidiyor"),
            ("bakiyom", "bakıyorum"), ("biliyon", "biliyorsun"), ("bilmiom", "bilmiyorum"),
            ("napıyon", "ne yapıyorsun"), ("napıyorsun", "ne yapıyorsun"), ("napıon", "ne yapıyorsun"),
            ("noldu", "ne oldu"), ("nooldu", "ne oldu"), ("nolur", "ne olur"),
            ("naptın", "ne yaptın"), ("bişey", "bir şey"), ("herşey", "her şey"),
            ("naber", "ne haber"), ("nbr", "ne haber"), ("slm", "selam"), ("mrb", "merhaba"),
            ("tşk", "teşekkürler"), ("tskk", "teşekkürler"), ("eyw", "eyvallah"),
            ("kib", "kendine iyi bak"), ("hoscakal", "hoşça kal"),
            ("aynen", "aynen"), ("ztn", "zaten"), ("bgn", "bugün"), ("yarin", "yarın")
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
    /// In standard Turkish words, max repetition of any letter is 2 (e.g. "saat", "hakkı", "milli").
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
        result
    }

    /// Normalizes a single informal Turkish word if it is a known spoken contraction or elongated form.
    pub fn normalize_word(word: &str) -> String {
        let deduped = Self::deduplicate_repeated_chars(word);
        let lower = super::super::phonology::to_turkish_lower(&deduped);

        if let Some(&formal) = INFORMAL_CONTRACTIONS.get(lower.as_str()) {
            formal.to_string()
        } else {
            deduped
        }
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
