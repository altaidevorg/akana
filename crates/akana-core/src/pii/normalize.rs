//! Detection-time folding against evasion.
//!
//! Zero-width and bidirectional format characters are dropped, and full-width ASCII and
//! non-Latin decimal digits are folded to ASCII, so `787\u{200B}05685624`, `７８７０５６８５６２４`
//! and `٧٨٧٠٥٦٨٥٦٢٤` are all seen as the TCKN they spell, and a zero-width space inside an API key
//! no longer splits it in two. Decomposed letters are composed (NFC): macOS and PDF text send
//! ş as s + U+0327, and the tokenizer split words at the mark, so "Şükrü Öztürk" in that form
//! was not a name. Detection runs on the folded text; spans are mapped back to the original
//! bytes, so masking replaces exactly what the caller sent.

use unicode_normalization::char::is_combining_mark;
use unicode_normalization::{is_nfc_quick, IsNormalized, UnicodeNormalization};

/// Folded text plus, for every byte of it, the byte offset in the original of the character it
/// came from (a composed letter: of its base letter).
pub(crate) struct Folded {
    pub text: String,
    origin: Vec<usize>,
    original_len: usize,
}

impl Folded {
    /// Original byte range of the folded range `start..end`. Dropped characters inside the
    /// span, and right after it up to the next kept character, belong to it.
    pub fn original_range(&self, start: usize, end: usize) -> (usize, usize) {
        let s = self.origin.get(start).copied().unwrap_or(self.original_len);
        let e = self.origin.get(end).copied().unwrap_or(self.original_len);
        (s, e.max(s))
    }
}

/// Characters an attacker can put inside a value without changing how it reads.
fn is_invisible(c: char) -> bool {
    matches!(
        c,
        '\u{00AD}'               // soft hyphen
            | '\u{034F}'         // combining grapheme joiner
            | '\u{061C}'         // Arabic letter mark
            | '\u{115F}' | '\u{1160}' | '\u{3164}' | '\u{FFA0}' // Hangul fillers
            | '\u{17B4}' | '\u{17B5}' // Khmer inherent vowels
            | '\u{180B}'..='\u{180F}' // Mongolian variation selectors, vowel separator
            | '\u{200B}'..='\u{200F}' // zero-width space/joiners, LRM, RLM
            | '\u{202A}'..='\u{202E}' // bidi embeddings and overrides
            | '\u{2060}'..='\u{2064}' // word joiner, invisible operators
            | '\u{2066}'..='\u{2069}' // bidi isolates
            | '\u{FE00}'..='\u{FE0F}' // variation selectors
            | '\u{FEFF}' // zero-width no-break space
            | '\u{1D173}'..='\u{1D17A}' // musical format controls
            | '\u{E0000}'..='\u{E007F}' // tag characters
            | '\u{E0100}'..='\u{E01EF}' // variation selectors supplement
    )
}

/// Zero digit of each Unicode block with a contiguous 0-9 run (general category Nd).
const DIGIT_ZEROS: &[u32] = &[
    0x0660, 0x06F0, 0x07C0, 0x0966, 0x09E6, 0x0A66, 0x0AE6, 0x0B66, 0x0BE6, 0x0C66, 0x0CE6, 0x0D66,
    0x0DE6, 0x0E50, 0x0ED0, 0x0F20, 0x1040, 0x1090, 0x17E0, 0x1810, 0x1946, 0x19D0, 0x1A80, 0x1A90,
    0x1B50, 0x1BB0, 0x1C40, 0x1C50, 0xA620, 0xA8D0, 0xA900, 0xA9D0, 0xA9F0, 0xAA50, 0xABF0, 0xFF10,
    0x104A0, 0x10D30, 0x11066, 0x110F0, 0x11136, 0x111D0, 0x112F0, 0x11450, 0x114D0, 0x11650,
    0x116C0, 0x11730, 0x118E0, 0x11950, 0x11C50, 0x11D50, 0x11DA0, 0x11F50, 0x16A60, 0x16AC0,
    0x16B50, 0x1D7CE, 0x1D7D8, 0x1D7E2, 0x1D7EC, 0x1D7F6, 0x1E140, 0x1E2F0, 0x1E4F0, 0x1E950,
    0x1FBF0,
];

fn fold_char(c: char) -> Option<char> {
    let u = c as u32;
    match u {
        // Full-width ASCII (U+FF01..U+FF5E) and the ideographic space.
        0xFF01..=0xFF5E => char::from_u32(u - 0xFEE0),
        0x3000 => Some(' '),
        _ => DIGIT_ZEROS
            .iter()
            .find(|&&zero| (zero..zero + 10).contains(&u))
            .and_then(|&zero| char::from_digit(u - zero, 10)),
    }
}

/// `None` when nothing needs folding (the common case: no copy is made).
pub(crate) fn fold_for_detection(text: &str) -> Option<Folded> {
    if text.is_ascii()
        || (!text
            .chars()
            .any(|c| !c.is_ascii() && (is_invisible(c) || fold_char(c).is_some()))
            && is_nfc_quick(text.chars()) == IsNormalized::Yes)
    {
        return None;
    }
    let mut folded = String::with_capacity(text.len());
    let mut origin = Vec::with_capacity(text.len());
    let mut emit = |c: char, at: usize| {
        origin.extend(std::iter::repeat_n(at, c.len_utf8()));
        folded.push(c);
    };
    let mut chars = text.char_indices().peekable();
    let mut marks: Vec<(usize, char)> = Vec::new();
    while let Some((i, c)) = chars.next() {
        if is_invisible(c) {
            continue;
        }
        let base = fold_char(c).unwrap_or(c);
        // the combining marks after it (an invisible character between them is dropped too)
        marks.clear();
        while let Some(&(j, m)) = chars.peek() {
            if is_invisible(m) {
                chars.next();
            } else if is_combining_mark(m) {
                marks.push((j, m));
                chars.next();
            } else {
                break;
            }
        }
        if marks.is_empty() {
            emit(base, i);
            continue;
        }
        // The letter and its marks as one composed letter, when there is one. The dot above
        // that Unicode lower-casing leaves on i (İ to i + U+0307, so ŞİFRE to "şi\u{307}fre")
        // is dropped: i has its dot already.
        let soft_dotted = matches!(base, 'i' | 'j');
        let composed: String = std::iter::once(base)
            .chain(
                marks
                    .iter()
                    .map(|&(_, m)| m)
                    .filter(|&m| !(soft_dotted && m == '\u{0307}')),
            )
            .nfc()
            .collect();
        let mut one = composed.chars();
        match (one.next(), one.next()) {
            (Some(letter), None) => emit(letter, i),
            // marks that do not all compose are left as they were
            _ => {
                emit(base, i);
                for &(j, m) in &marks {
                    emit(m, j);
                }
            }
        }
    }
    Some(Folded {
        text: folded,
        origin,
        original_len: text.len(),
    })
}

/// A text lower-cased both ways, for trigger gates: Turkish rules turn PIN into "pın" (missing
/// the English "pin"), Rust's `to_lowercase` turns ŞİFRE into "şi\u{307}fre" (missing the
/// Turkish "şifre"). A trigger counts when either form has it.
pub struct Lowered {
    pub turkish: String,
    pub unicode: String,
}

impl Lowered {
    pub fn of(text: &str) -> Self {
        Self {
            turkish: crate::phonology::to_turkish_lower(text),
            unicode: text.to_lowercase(),
        }
    }

    /// Whether either lower-cased form contains `needle` (itself lower case).
    pub fn has(&self, needle: &str) -> bool {
        self.turkish.contains(needle) || self.unicode.contains(needle)
    }
}

/// See `phonology::to_turkish_lower_with_origin`.
pub(crate) fn turkish_lower_with_origin(text: &str) -> (String, Vec<usize>) {
    crate::phonology::to_turkish_lower_with_origin(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lower_with_origin_matches_turkish_lower_and_maps_back() {
        let text = "İLÇE İdaresi DİYABET hastası";
        let (lower, origin) = turkish_lower_with_origin(text);
        assert_eq!(lower, crate::phonology::to_turkish_lower(text));
        let pos = lower.find("diyabet").unwrap();
        let start = origin[pos];
        let end = origin
            .get(pos + "diyabet".len())
            .copied()
            .unwrap_or(text.len());
        assert_eq!(&text[start..end], "DİYABET");
    }

    #[test]
    fn folds_and_maps_back() {
        let text = "TC: ７８７\u{200B}05685624.";
        let f = fold_for_detection(text).expect("needs folding");
        assert_eq!(f.text, "TC: 78705685624.");
        let start = f.text.find('7').unwrap();
        let (s, e) = f.original_range(start, start + 11);
        assert_eq!(&text[s..e], "７８７\u{200B}05685624");
    }

    #[test]
    fn plain_text_is_left_alone() {
        assert!(fold_for_detection("Ahmet Yılmaz 0532 123 45 67").is_none());
        assert!(fold_for_detection("ascii only").is_none());
        assert!(fold_for_detection("Şükrü Öztürk, Çağlayan İstanbul'da").is_none());
    }

    #[test]
    fn decomposed_letters_compose_and_map_back() {
        let text = "S\u{327}u\u{308}kru\u{308} geldi";
        let f = fold_for_detection(text).expect("needs composing");
        assert_eq!(f.text, "Şükrü geldi");
        let (s, e) = f.original_range(0, "Şükrü".len());
        assert_eq!(&text[s..e], "S\u{327}u\u{308}kru\u{308}");
        let at = f.text.find("geldi").unwrap();
        assert_eq!(
            f.original_range(at, at + 5),
            (text.find("geldi").unwrap(), text.len())
        );
        // an invisible character between a letter and its mark does not keep them apart
        assert_eq!(
            fold_for_detection("c\u{200B}\u{327}ay").unwrap().text,
            "çay"
        );
        // capital I with a dot above is İ; the dot that lower-casing leaves on i goes
        assert_eq!(
            fold_for_detection("I\u{307}stanbul").unwrap().text,
            "İstanbul"
        );
        assert_eq!(fold_for_detection("şi\u{307}fre").unwrap().text, "şifre");
        // marks that do not compose into one letter stay as they were
        assert_eq!(
            fold_for_detection("x\u{301}\u{302} ve ş\u{200B}")
                .unwrap()
                .text,
            "x\u{301}\u{302} ve ş"
        );
    }

    #[test]
    fn non_latin_digits_fold() {
        let f = fold_for_detection("٠٥٣٢ ١٢٣").unwrap();
        assert_eq!(f.text, "0532 123");
    }
}
