//! Turkish alphabet characteristics, letter properties, and casing rules.

/// Turkish specific letters
pub const TURKISH_LOWER_ALPHABET: [char; 29] = [
    'a', 'b', 'c', 'ç', 'd', 'e', 'f', 'g', 'ğ', 'h', 'ı', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'ö',
    'p', 'r', 's', 'ş', 't', 'u', 'ü', 'v', 'y', 'z',
];

pub const TURKISH_UPPER_ALPHABET: [char; 29] = [
    'A', 'B', 'C', 'Ç', 'D', 'E', 'F', 'G', 'Ğ', 'H', 'I', 'İ', 'J', 'K', 'L', 'M', 'N', 'O', 'Ö',
    'P', 'R', 'S', 'Ş', 'T', 'U', 'Ü', 'V', 'Y', 'Z',
];

/// Turkish vowels: a, e, ı, i, o, ö, u, ü
pub const VOWELS: [char; 8] = ['a', 'e', 'ı', 'i', 'o', 'ö', 'u', 'ü'];
pub const UPPER_VOWELS: [char; 8] = ['A', 'E', 'I', 'İ', 'O', 'Ö', 'U', 'Ü'];

/// Back vowels (Kalın ünlüler): a, ı, o, u
pub const BACK_VOWELS: [char; 4] = ['a', 'ı', 'o', 'u'];
pub const UPPER_BACK_VOWELS: [char; 4] = ['A', 'I', 'O', 'U'];

/// Front vowels (İnce ünlüler): e, i, ö, ü
pub const FRONT_VOWELS: [char; 4] = ['e', 'i', 'ö', 'ü'];
pub const UPPER_FRONT_VOWELS: [char; 4] = ['E', 'İ', 'Ö', 'Ü'];

/// Unrounded vowels (Düz ünlüler): a, e, ı, i
pub const UNROUNDED_VOWELS: [char; 4] = ['a', 'e', 'ı', 'i'];

/// Rounded vowels (Yuvarlak ünlüler): o, ö, u, ü
pub const ROUNDED_VOWELS: [char; 4] = ['o', 'ö', 'u', 'ü'];

/// Hard (voiceless) consonants (Sert ünsüzler - Fıstıkçı Şahap): f, s, t, k, ç, ş, h, p
pub const HARD_CONSONANTS: [char; 8] = ['f', 's', 't', 'k', 'ç', 'ş', 'h', 'p'];
pub const UPPER_HARD_CONSONANTS: [char; 8] = ['F', 'S', 'T', 'K', 'Ç', 'Ş', 'H', 'P'];

/// Soft / voiced consonants (Yumuşak ünsüzler): b, c, d, g, ğ, j, l, m, n, r, v, y, z
pub const SOFT_CONSONANTS: [char; 13] = [
    'b', 'c', 'd', 'g', 'ğ', 'j', 'l', 'm', 'n', 'r', 'v', 'y', 'z',
];

/// Mutable consonants subject to softening (p, ç, t, k)
pub const MUTABLE_CONSONANTS: [char; 4] = ['p', 'ç', 't', 'k'];

/// The style auditor's AI-text detector was trained on this behaviour: changing it shifts the
/// detector's inputs and fails `style::ai_detector::tests::outputs_match_training_reference`.
#[inline]
pub fn is_turkish_vowel(c: char) -> bool {
    matches!(
        c,
        'a' | 'e'
            | 'ı'
            | 'i'
            | 'o'
            | 'ö'
            | 'u'
            | 'ü'
            | 'A'
            | 'E'
            | 'I'
            | 'İ'
            | 'O'
            | 'Ö'
            | 'U'
            | 'Ü'
            | 'â'
            | 'î'
            | 'û'
            | 'Â'
            | 'Î'
            | 'Û'
    )
}

#[inline]
pub fn is_back_vowel(c: char) -> bool {
    matches!(
        c,
        'a' | 'ı' | 'o' | 'u' | 'A' | 'I' | 'O' | 'U' | 'â' | 'û' | 'Â' | 'Û'
    )
}

#[inline]
pub fn is_front_vowel(c: char) -> bool {
    matches!(c, 'e' | 'i' | 'ö' | 'ü' | 'E' | 'İ' | 'Ö' | 'Ü' | 'î' | 'Î')
}

#[inline]
pub fn is_rounded_vowel(c: char) -> bool {
    matches!(c, 'o' | 'ö' | 'u' | 'ü' | 'O' | 'Ö' | 'U' | 'Ü' | 'û' | 'Û')
}

#[inline]
pub fn is_unrounded_vowel(c: char) -> bool {
    matches!(
        c,
        'a' | 'e' | 'ı' | 'i' | 'A' | 'E' | 'I' | 'İ' | 'â' | 'î' | 'Â' | 'Î'
    )
}

/// The style auditor's AI-text detector was trained on this behaviour: changing it shifts the
/// detector's inputs and fails `style::ai_detector::tests::outputs_match_training_reference`.
#[inline]
pub fn is_hard_consonant(c: char) -> bool {
    matches!(
        c,
        'f' | 's'
            | 't'
            | 'k'
            | 'ç'
            | 'ş'
            | 'h'
            | 'p'
            | 'F'
            | 'S'
            | 'T'
            | 'K'
            | 'Ç'
            | 'Ş'
            | 'H'
            | 'P'
    )
}

#[inline]
pub fn is_soft_consonant(c: char) -> bool {
    matches!(
        c,
        'b' | 'c'
            | 'd'
            | 'g'
            | 'ğ'
            | 'j'
            | 'l'
            | 'm'
            | 'n'
            | 'r'
            | 'v'
            | 'y'
            | 'z'
            | 'B'
            | 'C'
            | 'D'
            | 'G'
            | 'Ğ'
            | 'J'
            | 'L'
            | 'M'
            | 'N'
            | 'R'
            | 'V'
            | 'Y'
            | 'Z'
    )
}

/// Converts a single character to lower case following Turkish casing rules (ÇAĞ -> çağ). A
/// character whose lower case is several characters keeps its first.
#[inline]
pub fn to_turkish_lower_char(c: char) -> char {
    match c {
        'I' => 'ı',
        'İ' => 'i',
        _ => c.to_lowercase().next().unwrap_or(c),
    }
}

/// Converts a single character to upper case following Turkish casing rules (Türkçe ->
/// TÜRKÇE). A character whose upper case is several characters keeps its first.
#[inline]
pub fn to_turkish_upper_char(c: char) -> char {
    match c {
        'ı' => 'I',
        'i' => 'İ',
        _ => c.to_uppercase().next().unwrap_or(c),
    }
}

/// Converts a string slice to lowercase according to Turkish locale rules (e.g. 'I' -> 'ı', 'İ' -> 'i').
///
/// The style auditor's AI-text detector was trained on this behaviour: changing it shifts the
/// detector's inputs and fails `style::ai_detector::tests::outputs_match_training_reference`.
pub fn to_turkish_lower(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            'I' => result.push('ı'),
            'İ' => result.push('i'),
            _ => {
                for lc in c.to_lowercase() {
                    result.push(lc);
                }
            }
        }
    }
    result
}

/// `to_turkish_lower(s)` together with, for every byte of it, the byte offset in `s` of the
/// character it came from.
///
/// Turkish lower-casing changes byte lengths (İ -> i is 2 -> 1 bytes, I -> ı is 1 -> 2), so a
/// position found in the lower-cased copy is not a position in `s`: map it through `origin`
/// (`s.len()` past the end), or the span lands on the wrong bytes after every İ/I and slicing
/// `s` with it can panic inside a character.
pub fn to_turkish_lower_with_origin(s: &str) -> (String, Vec<usize>) {
    let mut lower = String::with_capacity(s.len());
    let mut origin = Vec::with_capacity(s.len());
    for (i, c) in s.char_indices() {
        let before = lower.len();
        match c {
            'I' => lower.push('ı'),
            'İ' => lower.push('i'),
            _ => lower.extend(c.to_lowercase()),
        }
        origin.resize(origin.len() + lower.len() - before, i);
    }
    (lower, origin)
}

/// Byte offset in the original text of byte `i` of a `to_turkish_lower_with_origin` copy.
#[inline]
pub fn origin_offset(origin: &[usize], original_len: usize, i: usize) -> usize {
    origin.get(i).copied().unwrap_or(original_len)
}

/// Converts a string slice to uppercase according to Turkish locale rules (e.g. 'ı' -> 'I', 'i' -> 'İ').
pub fn to_turkish_upper(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            'ı' => result.push('I'),
            'i' => result.push('İ'),
            _ => {
                for uc in c.to_uppercase() {
                    result.push(uc);
                }
            }
        }
    }
    result
}

/// Converts a string to Title Case following Turkish rules.
pub fn to_turkish_title(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut capitalize_next = true;

    for c in s.chars() {
        if c.is_whitespace() || c == '-' || c == '/' || c == '(' || c == '"' || c == '\'' {
            result.push(c);
            capitalize_next = true;
        } else if capitalize_next {
            match c {
                'ı' => result.push('I'),
                'i' => result.push('İ'),
                _ => {
                    for uc in c.to_uppercase() {
                        result.push(uc);
                    }
                }
            }
            capitalize_next = false;
        } else {
            match c {
                'I' => result.push('ı'),
                'İ' => result.push('i'),
                _ => {
                    for lc in c.to_lowercase() {
                        result.push(lc);
                    }
                }
            }
        }
    }
    result
}

/// Finds the last vowel in a word (ignoring circumflex if applicable).
pub fn last_vowel(s: &str) -> Option<char> {
    s.chars().rev().find(|&c| is_turkish_vowel(c))
}

/// Finds the first vowel in a word.
pub fn first_vowel(s: &str) -> Option<char> {
    s.chars().find(|&c| is_turkish_vowel(c))
}

/// Returns the number of syllables (vowels) in a Turkish word.
pub fn syllable_count(s: &str) -> usize {
    s.chars().filter(|&c| is_turkish_vowel(c)).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_the_case_of_every_turkish_letter() {
        let lower: String = "ÇAĞ İŞÖÜI".chars().map(to_turkish_lower_char).collect();
        assert_eq!(lower, "çağ işöüı");
        let upper: String = "çağ işöüı".chars().map(to_turkish_upper_char).collect();
        assert_eq!(upper, "ÇAĞ İŞÖÜI");
    }
}
