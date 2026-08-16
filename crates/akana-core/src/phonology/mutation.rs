//! Phonological mutations in Turkish: consonant softening, vowel drop, doubling, and buffer letters.

use super::alphabet::is_hard_consonant;

/// Consonant mutation (Ünsüz Yumuşaması):
/// - 'p' -> 'b' (kitap -> kitaba)
/// - 'ç' -> 'c' (ağaç -> ağaca)
/// - 't' -> 'd' (kanat -> kanada)
/// - 'k' -> 'ğ' (ayak -> ayağı, bebek -> bebeği)
/// - 'k' -> 'g' (renk -> rengi, cenk -> cengi)
#[inline]
pub fn soften_consonant(c: char, is_nk_cluster: bool) -> char {
    match c {
        'p' => 'b',
        'ç' => 'c',
        't' => 'd',
        'k' => if is_nk_cluster { 'g' } else { 'ğ' },
        'P' => 'B',
        'Ç' => 'C',
        'T' => 'D',
        'K' => if is_nk_cluster { 'G' } else { 'Ğ' },
        _ => c,
    }
}

/// Hardens a consonant (Ünsüz Sertleşmesi / Benzeşmesi):
/// e.g. for locative/ablative '-da/-de' -> '-ta/-te' after hard consonant (kitap-ta, ağaç-tan)
#[inline]
pub fn harden_initial_consonant(c: char) -> char {
    match c {
        'd' => 't',
        'c' => 'ç',
        'g' => 'k',
        'D' => 'T',
        'C' => 'Ç',
        'G' => 'K',
        _ => c,
    }
}

/// Applies consonant softening to the end of a stem if applicable.
pub fn apply_stem_softening(stem: &str) -> String {
    if stem.is_empty() {
        return String::new();
    }
    let mut chars: Vec<char> = stem.chars().collect();
    let len = chars.len();
    let last = chars[len - 1];

    let is_nk = len >= 2 && (chars[len - 2] == 'n' || chars[len - 2] == 'N') && (last == 'k' || last == 'K');
    chars[len - 1] = soften_consonant(last, is_nk);
    chars.into_iter().collect()
}

/// Applies vowel drop (Ünlü Düşmesi) to a 2-syllable noun root when an affix starting with a vowel is added.
/// e.g. "burun" -> "burn", "akıl" -> "akl", "şehir" -> "şehr", "karın" -> "karn"
pub fn apply_vowel_drop(stem: &str) -> Option<String> {
    let chars: Vec<char> = stem.chars().collect();
    let len = chars.len();
    if len < 3 {
        return None;
    }

    // Usually second-to-last char is a close vowel: ı, i, u, ü
    let second_last = chars[len - 2];
    if matches!(second_last, 'ı' | 'i' | 'u' | 'ü') {
        let mut result = String::with_capacity(stem.len());
        for (i, &c) in chars.iter().enumerate() {
            if i != len - 2 {
                result.push(c);
            }
        }
        Some(result)
    } else {
        None
    }
}

/// Applies consonant doubling (Ünsüz Türemesi):
/// e.g. "hak" -> "hakk", "his" -> "hiss", "af" -> "aff", "hat" -> "hatt"
pub fn apply_consonant_doubling(stem: &str) -> String {
    if let Some(last) = stem.chars().last() {
        let mut result = stem.to_string();
        result.push(last);
        result
    } else {
        stem.to_string()
    }
}

/// Determines the buffer consonant (Kaynaştırma Harfi) needed between morphemes.
/// - 'y' for case affixes (kapı-y-a, su-y-u)
/// - 's' for 3rd person possessive (kapı-s-ı)
/// - 'n' for pronominal n / genitive after vowel (kapı-n-ın, o-n-a)
/// - 'ş' for distributive numerals (iki-ş-er, altı-ş-ar)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferConsonant {
    Y,
    S,
    N,
    Sh,
}

impl BufferConsonant {
    #[inline]
    pub fn as_char(&self) -> char {
        match self {
            BufferConsonant::Y => 'y',
            BufferConsonant::S => 's',
            BufferConsonant::N => 'n',
            BufferConsonant::Sh => 'ş',
        }
    }
}

/// Checks if a morpheme boundary needs a buffer consonant (i.e. stem ends with vowel and affix starts with vowel).
#[inline]
pub fn needs_buffer(stem_ends_vowel: bool, affix_starts_vowel: bool) -> bool {
    stem_ends_vowel && affix_starts_vowel
}

/// Checks if an affix initial consonant should harden due to preceding hard consonant.
#[inline]
pub fn should_harden(last_char: char) -> bool {
    is_hard_consonant(last_char)
}
