//! Phonetic attributes of a stem or suffix sequence, which decide the shape of the next suffix
//! (Zemberek's `PhoneticAttribute` and `AttributesHelper`).

bitflags::bitflags! {
    /// What the analyzer knows about the letters matched so far. Suffix templates read the
    /// vowel and voicing bits; the `EXPECTS_*` and `CANNOT_TERMINATE` bits come from the stem
    /// (kitab- must be followed by a vowel, kitap- by a consonant).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
    pub struct PhoneticAttrs: u32 {
        const LAST_LETTER_VOWEL = 1 << 0;
        const LAST_LETTER_CONSONANT = 1 << 1;
        const LAST_VOWEL_FRONTAL = 1 << 2;
        const LAST_VOWEL_BACK = 1 << 3;
        const LAST_VOWEL_ROUNDED = 1 << 4;
        const LAST_VOWEL_UNROUNDED = 1 << 5;
        const LAST_LETTER_VOICELESS = 1 << 6;
        const LAST_LETTER_VOICED = 1 << 7;
        const LAST_LETTER_VOICELESS_STOP = 1 << 8;
        const FIRST_LETTER_VOWEL = 1 << 9;
        const FIRST_LETTER_CONSONANT = 1 << 10;
        const HAS_NO_VOWEL = 1 << 11;
        /// The next suffix must start with a vowel (kitab-, hakk-, burn-).
        const EXPECTS_VOWEL = 1 << 12;
        /// The next suffix must start with a consonant (kitap-, hak-, burun-).
        const EXPECTS_CONSONANT = 1 << 13;
        /// ban-, san-: the modified stems of ben and sen.
        const MODIFIED_PRONOUN = 1 << 14;
        const UNMODIFIED_PRONOUN = 1 << 15;
        /// A verb stem whose final vowel was dropped before -Iyor (bekl-).
        const LAST_LETTER_DROPPED = 1 << 16;
        /// The word may not end here (a modified stem, or a suffix ending in ğ).
        const CANNOT_TERMINATE = 1 << 17;
    }
}

#[inline]
pub fn is_vowel(c: char) -> bool {
    matches!(
        c,
        'a' | 'e' | 'ı' | 'i' | 'o' | 'ö' | 'u' | 'ü' | 'â' | 'î' | 'û'
    )
}

#[inline]
fn is_frontal(c: char) -> bool {
    matches!(c, 'e' | 'i' | 'ö' | 'ü' | 'î' | 'û')
}

#[inline]
fn is_rounded(c: char) -> bool {
    matches!(c, 'o' | 'ö' | 'u' | 'ü' | 'û')
}

/// Voiceless consonants (Fıstıkçı Şahap).
#[inline]
pub fn is_voiceless(c: char) -> bool {
    matches!(c, 'ç' | 'f' | 'h' | 'k' | 'p' | 's' | 'ş' | 't')
}

/// Voiceless stops, the consonants that voicing changes (kitap -> kitab-).
#[inline]
pub fn is_voiceless_stop(c: char) -> bool {
    matches!(c, 'ç' | 'k' | 'p' | 't')
}

/// The voiced counterpart of a final consonant (p -> b, ç -> c, t -> d, k -> ğ, g -> ğ).
#[inline]
pub fn voice(c: char) -> char {
    match c {
        'ç' => 'c',
        'g' | 'k' => 'ğ',
        'p' => 'b',
        't' => 'd',
        c => c,
    }
}

/// The voiceless counterpart of a suffix's first consonant (d -> t, c -> ç, ...).
#[inline]
pub fn devoice(c: char) -> char {
    match c {
        'b' => 'p',
        'c' => 'ç',
        'd' => 't',
        'g' | 'ğ' => 'k',
        c => c,
    }
}

/// The attributes of a letter sequence (a stem, or the suffixes matched so far), given those
/// before it. A sequence without vowels keeps the vowel attributes of what precedes it.
pub fn attributes_of(seq: &str, predecessor: PhoneticAttrs) -> PhoneticAttrs {
    let Some(last) = seq.chars().last() else {
        return predecessor;
    };
    let mut attrs;
    if let Some(last_vowel) = seq.chars().rev().find(|&c| is_vowel(c)) {
        attrs = PhoneticAttrs::empty();
        attrs |= if is_vowel(last) {
            PhoneticAttrs::LAST_LETTER_VOWEL
        } else {
            PhoneticAttrs::LAST_LETTER_CONSONANT
        };
        attrs |= if is_frontal(last_vowel) {
            PhoneticAttrs::LAST_VOWEL_FRONTAL
        } else {
            PhoneticAttrs::LAST_VOWEL_BACK
        };
        attrs |= if is_rounded(last_vowel) {
            PhoneticAttrs::LAST_VOWEL_ROUNDED
        } else {
            PhoneticAttrs::LAST_VOWEL_UNROUNDED
        };
        attrs |= if seq.chars().next().is_some_and(is_vowel) {
            PhoneticAttrs::FIRST_LETTER_VOWEL
        } else {
            PhoneticAttrs::FIRST_LETTER_CONSONANT
        };
    } else {
        attrs = predecessor;
        attrs |= PhoneticAttrs::LAST_LETTER_CONSONANT
            | PhoneticAttrs::FIRST_LETTER_CONSONANT
            | PhoneticAttrs::HAS_NO_VOWEL;
        attrs -= PhoneticAttrs::LAST_LETTER_VOWEL | PhoneticAttrs::EXPECTS_CONSONANT;
    }
    attrs -= PhoneticAttrs::LAST_LETTER_VOICELESS
        | PhoneticAttrs::LAST_LETTER_VOICED
        | PhoneticAttrs::LAST_LETTER_VOICELESS_STOP;
    if is_voiceless(last) {
        attrs |= PhoneticAttrs::LAST_LETTER_VOICELESS;
        if is_voiceless_stop(last) {
            attrs |= PhoneticAttrs::LAST_LETTER_VOICELESS_STOP;
        }
    } else {
        attrs |= PhoneticAttrs::LAST_LETTER_VOICED;
    }
    attrs
}
