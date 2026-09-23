//! Turkish morphological suffix stripping, apostrophe handling, and vowel harmony restoration.
//!
//! Provides:
//! - Exact stem/suffix demarcation for proper nouns taking Turkish inflectional suffixes (`Ahmet'in`, `Ayşe'ye`)
//! - Vowel harmony and consonant mutation adjustment for restoring placeholders into LLM responses

use crate::phonology::to_turkish_lower;
use stringzilla::StringZilla;

/// Result of splitting a word into its stem and suffix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StemmedToken<'a> {
    pub raw: &'a str,
    pub stem: &'a str,
    pub apostrophe: Option<&'a str>,
    pub suffix: Option<&'a str>,
    pub stem_start: usize,
    pub stem_end: usize,
}

/// Splits a token into its base stem and apostrophe-separated suffix.
///
/// Example:
/// - `"Ahmet'in"` -> stem: `"Ahmet"`, apostrophe: `Some("'")`, suffix: `Some("in")`
/// - `"Ayşe’ye"` -> stem: `"Ayşe"`, apostrophe: `Some("’")`, suffix: `Some("ye")`
/// - `"Mehmet"` -> stem: `"Mehmet"`, apostrophe: `None`, suffix: `None`
pub fn split_stem_suffix<'a>(token: &'a str, token_offset: usize) -> StemmedToken<'a> {
    let apo_idx = [token.sz_find("'"), token.sz_find("’"), token.sz_find("´")]
        .into_iter()
        .flatten()
        .min();
    if let Some(apo_idx) = apo_idx {
        let stem = &token[..apo_idx];
        let apo_char_len = token[apo_idx..].chars().next().map_or(1, |c| c.len_utf8());
        let suffix_start = apo_idx + apo_char_len;
        let apostrophe = &token[apo_idx..suffix_start];
        let suffix = if suffix_start < token.len() {
            Some(&token[suffix_start..])
        } else {
            None
        };

        StemmedToken {
            raw: token,
            stem,
            apostrophe: Some(apostrophe),
            suffix,
            stem_start: token_offset,
            stem_end: token_offset + apo_idx,
        }
    } else {
        StemmedToken {
            raw: token,
            stem: token,
            apostrophe: None,
            suffix: None,
            stem_start: token_offset,
            stem_end: token_offset + token.len(),
        }
    }
}

/// Suffix case types for Turkish proper nouns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurkishCase {
    Dative,       // -e / -a (yönelme)
    Genitive,     // -in / -ın / -un / -ün (ilgi / tamlayan)
    Accusative,   // -i / -ı / -u / -ü (belirtme)
    Locative,     // -de / -da / -te / -ta (bulunma)
    Ablative,     // -den / -dan / -ten / -tan (ayrılma)
    Instrumental, // -le / -la (vasıta)
    Plural,       // -ler / -lar (çoğul)
}

/// Detects the inflectional case marker from an existing suffix string.
pub fn detect_suffix_case(suffix: &str) -> Option<TurkishCase> {
    let lower = to_turkish_lower(suffix.trim_matches(|c: char| !c.is_alphabetic()));
    match lower.as_str() {
        "e" | "a" | "ye" | "ya" => Some(TurkishCase::Dative),
        "in" | "ın" | "un" | "ün" | "nin" | "nın" | "nun" | "nün" => {
            Some(TurkishCase::Genitive)
        }
        "i" | "ı" | "u" | "ü" | "yi" | "yı" | "yu" | "yü" => Some(TurkishCase::Accusative),
        "de" | "da" | "te" | "ta" => Some(TurkishCase::Locative),
        "den" | "dan" | "ten" | "tan" => Some(TurkishCase::Ablative),
        "le" | "la" | "yle" | "yla" => Some(TurkishCase::Instrumental),
        "ler" | "lar" => Some(TurkishCase::Plural),
        _ => None,
    }
}

/// Re-harmonizes a suffix onto a target stem according to Turkish vowel harmony
/// and consonant assimilation rules.
///
/// Example:
/// - `harmonize_suffix("Ahmet", TurkishCase::Dative)` -> `"Ahmet'e"`
/// - `harmonize_suffix("Ali", TurkishCase::Dative)` -> `"Ali'ye"`
/// - `harmonize_suffix("Murat", TurkishCase::Dative)` -> `"Murat'a"`
/// - `harmonize_suffix("Mehmet", TurkishCase::Locative)` -> `"Mehmet'te"`
/// - `harmonize_suffix("Can", TurkishCase::Locative)` -> `"Can'da"`
pub fn harmonize_suffix(stem: &str, case: TurkishCase) -> String {
    let lower_stem = to_turkish_lower(stem);
    let last_char = lower_stem.chars().last().unwrap_or('a');
    let last_vowel = lower_stem
        .chars()
        .rev()
        .find(|&c| matches!(c, 'a' | 'e' | 'ı' | 'i' | 'o' | 'ö' | 'u' | 'ü'))
        .unwrap_or('e');

    let is_vowel_ended = matches!(last_char, 'a' | 'e' | 'ı' | 'i' | 'o' | 'ö' | 'u' | 'ü');
    let is_front_vowel = matches!(last_vowel, 'e' | 'i' | 'ö' | 'ü');
    let is_unvoiced_stop = matches!(last_char, 'ç' | 'f' | 'h' | 'k' | 'p' | 's' | 'ş' | 't');

    let suffix: String = match case {
        TurkishCase::Dative => {
            if is_vowel_ended {
                if is_front_vowel {
                    "ye".to_string()
                } else {
                    "ya".to_string()
                }
            } else if is_front_vowel {
                "e".to_string()
            } else {
                "a".to_string()
            }
        }
        TurkishCase::Genitive => {
            let buffer = if is_vowel_ended { "n" } else { "" };
            match last_vowel {
                'e' | 'i' => format!("{buffer}in"),
                'a' | 'ı' => format!("{buffer}ın"),
                'o' | 'u' => format!("{buffer}un"),
                'ö' | 'ü' => format!("{buffer}ün"),
                _ => format!("{buffer}in"),
            }
        }
        TurkishCase::Accusative => {
            let buffer = if is_vowel_ended { "y" } else { "" };
            match last_vowel {
                'e' | 'i' => format!("{buffer}i"),
                'a' | 'ı' => format!("{buffer}ı"),
                'o' | 'u' => format!("{buffer}u"),
                'ö' | 'ü' => format!("{buffer}ü"),
                _ => format!("{buffer}i"),
            }
        }
        TurkishCase::Locative => {
            let d_or_t = if is_unvoiced_stop { "t" } else { "d" };
            let e_or_a = if is_front_vowel { "e" } else { "a" };
            format!("{d_or_t}{e_or_a}")
        }
        TurkishCase::Ablative => {
            let d_or_t = if is_unvoiced_stop { "t" } else { "d" };
            let e_or_a = if is_front_vowel { "e" } else { "a" };
            format!("{d_or_t}{e_or_a}n")
        }
        TurkishCase::Instrumental => {
            let buffer = if is_vowel_ended { "y" } else { "" };
            let e_or_a = if is_front_vowel { "le" } else { "la" };
            format!("{buffer}{e_or_a}")
        }
        TurkishCase::Plural => {
            if is_front_vowel {
                "ler".to_string()
            } else {
                "lar".to_string()
            }
        }
    };

    format!("{stem}'{suffix}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_stem_suffix() {
        let t1 = split_stem_suffix("Ahmet'in", 10);
        assert_eq!(t1.stem, "Ahmet");
        assert_eq!(t1.suffix, Some("in"));
        assert_eq!(t1.stem_start, 10);
        assert_eq!(t1.stem_end, 15);

        let t2 = split_stem_suffix("Ayşe’ye", 0);
        assert_eq!(t2.stem, "Ayşe");
        assert_eq!(t2.suffix, Some("ye"));

        let t3 = split_stem_suffix("Mehmet", 5);
        assert_eq!(t3.stem, "Mehmet");
        assert_eq!(t3.suffix, None);
    }

    #[test]
    fn test_harmonize_suffix() {
        assert_eq!(harmonize_suffix("Ahmet", TurkishCase::Dative), "Ahmet'e");
        assert_eq!(harmonize_suffix("Murat", TurkishCase::Dative), "Murat'a");
        assert_eq!(harmonize_suffix("Ali", TurkishCase::Dative), "Ali'ye");
        assert_eq!(
            harmonize_suffix("Mehmet", TurkishCase::Locative),
            "Mehmet'te"
        );
        assert_eq!(harmonize_suffix("Can", TurkishCase::Locative), "Can'da");
        assert_eq!(harmonize_suffix("Ali", TurkishCase::Genitive), "Ali'nin");
        assert_eq!(harmonize_suffix("Murat", TurkishCase::Genitive), "Murat'ın");
    }
}
