//! Turkish vowel harmony rules: 2-way (A/E - Büyük Ünlü Uyumu) and 4-way (I/İ/U/Ü - Küçük Ünlü Uyumu).

use super::alphabet::{is_back_vowel, is_front_vowel, is_rounded_vowel, is_unrounded_vowel, last_vowel};

/// Evaluates 2-way vowel harmony (A-type):
/// - Back vowels (a, ı, o, u) -> 'a'
/// - Front vowels (e, i, ö, ü) -> 'e'
#[inline]
pub fn harmony_a_type(vowel: char) -> char {
    if is_front_vowel(vowel) {
        'e'
    } else {
        'a'
    }
}

/// Evaluates 4-way vowel harmony (I-type):
/// - Unrounded back (a, ı) -> 'ı'
/// - Unrounded front (e, i) -> 'i'
/// - Rounded back (o, u) -> 'u'
/// - Rounded front (ö, ü) -> 'ü'
#[inline]
pub fn harmony_i_type(vowel: char) -> char {
    if is_front_vowel(vowel) {
        if is_rounded_vowel(vowel) {
            'ü'
        } else {
            'i'
        }
    } else {
        if is_rounded_vowel(vowel) {
            'u'
        } else {
            'ı'
        }
    }
}

/// Gets the appropriate A-type harmonic vowel for a given stem.
#[inline]
pub fn get_a_type_harmonic_vowel(stem: &str) -> char {
    match last_vowel(stem) {
        Some(v) => harmony_a_type(v),
        None => 'e', // Default fallback
    }
}

/// Gets the appropriate I-type harmonic vowel for a given stem.
#[inline]
pub fn get_i_type_harmonic_vowel(stem: &str) -> char {
    match last_vowel(stem) {
        Some(v) => harmony_i_type(v),
        None => 'i', // Default fallback
    }
}

/// Checks if a word satisfies Major Vowel Harmony (Büyük Ünlü Uyumu: all vowels either all back or all front).
pub fn check_major_vowel_harmony(word: &str) -> bool {
    let mut has_back = false;
    let mut has_front = false;

    for c in word.chars() {
        if is_back_vowel(c) {
            has_back = true;
        } else if is_front_vowel(c) {
            has_front = true;
        }
        if has_back && has_front {
            return false;
        }
    }
    true
}

/// Checks if a word satisfies Minor Vowel Harmony (Küçük Ünlü Uyumu).
pub fn check_minor_vowel_harmony(word: &str) -> bool {
    let vowels: Vec<char> = word.chars().filter(|&c| super::alphabet::is_turkish_vowel(c)).collect();
    if vowels.len() <= 1 {
        return true;
    }

    for window in vowels.windows(2) {
        let prev = window[0];
        let curr = window[1];

        if is_unrounded_vowel(prev) {
            // Unrounded must be followed by unrounded (a, e, ı, i)
            if !is_unrounded_vowel(curr) {
                return false;
            }
        } else if is_rounded_vowel(prev) {
            // Rounded must be followed by unrounded open (a, e) or rounded close (u, ü)
            let is_open_unrounded = curr == 'a' || curr == 'e';
            let is_close_rounded = curr == 'u' || curr == 'ü';
            if !is_open_unrounded && !is_close_rounded {
                return false;
            }
        }
    }
    true
}
