//! TDK-compliant Turkish Syllabification and Hyphenation Engine.
//! Follows standard Turkish phonotactic rules (V, CV, VC, CVC, VCC, CVCC).

use super::is_turkish_vowel;

pub struct TurkishSyllabifier;

impl TurkishSyllabifier {
    /// Syllabifies a single Turkish word according to standard Turkish phonotactics.
    /// Example: "Türkçe" -> ["Türk", "çe"], "araba" -> ["a", "ra", "ba"], "ilkokul" -> ["il", "ko", "kul"].
    pub fn syllabify(word: &str) -> Vec<String> {
        let chars: Vec<char> = word.chars().collect();
        let len = chars.len();
        if len == 0 {
            return Vec::new();
        }

        // Count vowels
        let vowel_indices: Vec<usize> = chars.iter().enumerate()
            .filter(|(_, &c)| is_turkish_vowel(c))
            .map(|(i, _)| i)
            .collect();

        if vowel_indices.is_empty() {
            return vec![word.to_string()];
        }

        if vowel_indices.len() == 1 {
            return vec![word.to_string()];
        }

        let mut cut_points: Vec<usize> = Vec::new();

        for w in 0..vowel_indices.len() - 1 {
            let v1 = vowel_indices[w];
            let v2 = vowel_indices[w + 1];
            let consonants_between = v2 - v1 - 1;

            match consonants_between {
                0 => {
                    // Two consecutive vowels (usually loanwords e.g. "sa-at", "fi-il", "re-is", "şair")
                    cut_points.push(v1 + 1);
                }
                1 => {
                    // Single consonant between vowels goes with the second vowel (e.g. "a-ra-ba", "o-kul", "ba-şöğ-ret-men")
                    cut_points.push(v1 + 1);
                }
                2 => {
                    // Two consonants between vowels: first goes with V1, second with V2 (e.g. "kit-lık", "kar-deş", "Türk-çe")
                    cut_points.push(v1 + 2);
                }
                3 => {
                    // Three consonants between vowels: first two go with V1, third with V2 (e.g. "alt-lık", "sürt-me", "türk-çe")
                    cut_points.push(v1 + 3);
                }
                _ => {
                    // 4+ consonants: split before the last consonant
                    cut_points.push(v2 - 1);
                }
            }
        }

        let mut syllables = Vec::new();
        let mut start = 0;
        for &cut in &cut_points {
            if cut > start && cut <= len {
                syllables.push(chars[start..cut].iter().collect::<String>());
                start = cut;
            }
        }
        if start < len {
            syllables.push(chars[start..len].iter().collect::<String>());
        }

        syllables
    }

    /// Hyphenates a word using a given delimiter (default is "-").
    pub fn hyphenate(word: &str, delimiter: &str) -> String {
        let syllables = Self::syllabify(word);
        syllables.join(delimiter)
    }

    /// Counts syllables in a word (equivalent to vowel count in Turkish).
    #[inline]
    pub fn count_syllables(word: &str) -> usize {
        word.chars().filter(|&c| is_turkish_vowel(c)).count()
    }
}
