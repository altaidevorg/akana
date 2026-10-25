//! How words that are not spelled as they are read are pronounced: abbreviations (ABD is
//! "abede", so ABD'nin, not ABD'nın), words without vowels, and numbers. Suffixes harmonise
//! with the pronunciation, as in Zemberek's PronunciationGuesser and TurkishNumbers.

use crate::phonology::is_turkish_vowel;

/// Turkish letter names (Zemberek's `phonetics/turkish-letter-names.txt`). A final k is read
/// "ka" (TBK, HMK).
fn letter_name(c: char) -> Option<&'static str> {
    Some(match c {
        'a' => "a",
        'b' => "be",
        'c' => "ce",
        'ç' => "çe",
        'd' => "de",
        'e' => "e",
        'f' => "fe",
        'g' => "ge",
        'ğ' => "yumuşakge",
        'h' => "he",
        'ı' => "ı",
        'i' => "i",
        'j' => "je",
        'k' => "ke",
        'l' => "le",
        'm' => "me",
        'n' => "ne",
        'o' => "o",
        'ö' => "ö",
        'p' => "pe",
        'r' => "re",
        's' => "se",
        'ş' => "şe",
        't' => "te",
        'u' => "u",
        'ü' => "ü",
        'v' => "ve",
        'y' => "ye",
        'z' => "ze",
        'w' => "dabılyu",
        'q' => "kü",
        'x' => "iks",
        _ => return None,
    })
}

const ONES: [&str; 10] = [
    "", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz",
];
const TENS: [&str; 10] = [
    "", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan",
];
const POWERS: [&str; 7] = [
    "",
    "bin",
    "milyon",
    "milyar",
    "trilyon",
    "katrilyon",
    "kentilyon",
];

/// A run of decimal digits read out without spaces (2026 -> ikibinyirmialtı, 007 ->
/// sıfırsıfıryedi), as TurkishNumbers reads it. Digits beyond the largest named power are read
/// one by one.
pub fn number_to_words(digits: &str) -> String {
    let mut out = String::new();
    let trimmed = digits.trim_start_matches('0');
    for _ in 0..digits.len() - trimmed.len() {
        out.push_str("sıfır");
    }
    if trimmed.is_empty() {
        return out;
    }
    if trimmed.len() > 3 * POWERS.len() {
        for c in trimmed.chars() {
            let d = c.to_digit(10).unwrap_or(0) as usize;
            out.push_str(if d == 0 { "sıfır" } else { ONES[d] });
        }
        return out;
    }
    let bytes = trimmed.as_bytes();
    let groups = bytes.len().div_ceil(3);
    let first = bytes.len() - 3 * (groups - 1);
    let mut start = 0;
    for g in 0..groups {
        let end = if g == 0 { first } else { start + 3 };
        let value: usize = trimmed[start..end].parse().unwrap_or(0);
        let power = groups - 1 - g;
        if value != 0 {
            if !(power == 1 && value == 1) {
                let (h, rest) = (value / 100, value % 100);
                if h > 1 {
                    out.push_str(ONES[h]);
                }
                if h > 0 {
                    out.push_str("yüz");
                }
                out.push_str(TENS[rest / 10]);
                out.push_str(ONES[rest % 10]);
            }
            out.push_str(POWERS[power]);
        }
        start = end;
    }
    out
}

/// Spells a word letter by letter (ABD -> abede, TBK -> tebeka); digit runs are read as
/// numbers (A4 -> adört).
pub fn letter_pronunciation(word: &str) -> String {
    let chars: Vec<char> = word.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_ascii_digit() {
            let start = i;
            while i < chars.len() && chars[i].is_ascii_digit() {
                i += 1;
            }
            let digits: String = chars[start..i].iter().collect();
            out.push_str(&number_to_words(&digits));
            continue;
        }
        if c == 'k' && i + 1 == chars.len() {
            out.push_str("ka");
        } else if let Some(name) = letter_name(c) {
            out.push_str(name);
        } else if c != '-' {
            out.push(c);
        }
        i += 1;
    }
    out
}

/// Two-consonant codas a Turkish syllable may end in (kırk, sarp, üst, renk).
const FINAL_CLUSTERS: &[&str] = &[
    "rk", "rt", "rp", "rs", "rç", "rd", "rb", "rg", "rf", "rm", "rn", "rz", "rş", "lk", "lt", "lp",
    "lç", "ls", "lm", "lf", "nk", "nt", "nç", "nd", "ns", "nz", "st", "şt", "ks", "ft", "kt", "pt",
    "yt", "yn", "yk", "ym", "yr", "yl", "yş",
];

/// Whether a word reads as Turkish syllables: at most one consonant before the first vowel,
/// at most three between vowels, and at most a common two-consonant coda after the last.
fn strictly_syllabifiable(word: &str) -> bool {
    let chars: Vec<char> = word.chars().collect();
    let Some(first) = chars.iter().position(|&c| is_turkish_vowel(c)) else {
        return false;
    };
    if first > 1 {
        return false;
    }
    let last = chars
        .iter()
        .rposition(|&c| is_turkish_vowel(c))
        .unwrap_or(first);
    let mut run = 0;
    for &c in &chars[first..=last] {
        if is_turkish_vowel(c) {
            run = 0;
        } else {
            run += 1;
            if run > 3 {
                return false;
            }
        }
    }
    let coda: String = chars[last + 1..].iter().collect();
    coda.chars().count() <= 1 || FINAL_CLUSTERS.contains(&coda.as_str())
}

/// Zemberek's `guessForAbbreviation`: an abbreviation is spelled letter by letter (ABD ->
/// abede, AİHM -> aiheme) unless it reads as a word (NATO, TÜBİTAK).
pub fn guess_abbreviation(root: &str) -> String {
    let first_two: String = root.chars().take(2).collect();
    if root.chars().count() < 3
        || !first_two.chars().any(is_turkish_vowel)
        || !strictly_syllabifiable(root)
    {
        letter_pronunciation(root)
    } else {
        root.chars()
            .flat_map(|c| match c {
                'w' => "v".chars().collect::<Vec<_>>(),
                'q' => vec!['k'],
                'x' => vec!['k', 's'],
                c => vec![c],
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_numbers() {
        assert_eq!(number_to_words("0"), "sıfır");
        assert_eq!(number_to_words("1"), "bir");
        assert_eq!(number_to_words("10"), "on");
        assert_eq!(number_to_words("1000"), "bin");
        assert_eq!(number_to_words("1923"), "bindokuzyüzyirmiüç");
        assert_eq!(number_to_words("2026"), "ikibinyirmialtı");
        assert_eq!(number_to_words("1000000"), "birmilyon");
        assert_eq!(number_to_words("007"), "sıfırsıfıryedi");
    }

    #[test]
    fn guesses_abbreviations() {
        assert_eq!(guess_abbreviation("abd"), "abede");
        assert_eq!(guess_abbreviation("tbk"), "tebeka");
        assert_eq!(guess_abbreviation("aihm"), "aiheme");
        assert_eq!(guess_abbreviation("nato"), "nato");
        assert_eq!(guess_abbreviation("tbmm"), "tebememe");
        assert_eq!(guess_abbreviation("a4"), "adört");
    }
}
