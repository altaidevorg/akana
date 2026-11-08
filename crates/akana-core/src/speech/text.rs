//! Text to words, clause boundaries and pauses: one `_:` at a clause boundary, `_!` at quotes
//! and brackets and before conjunctions, words split at spaces, punctuation and case changes,
//! and digits the verbaliser left read as numbers.

use super::Pause;
use crate::phonology::to_turkish_lower;
use smallvec::SmallVec;
use std::ops::Range;
use unicode_normalization::UnicodeNormalization;

/// Punctuation that ends a clause: `_:` (and a new clause for the nucleus).
const CLAUSE_PUNCT: [char; 7] = [',', '.', ';', ':', '!', '?', '…'];
/// Quotation marks and brackets: a short pause `_!` where they stand.
const QUOTES: [char; 7] = ['"', '“', '”', '«', '»', '(', ')'];
/// Words before which `tts-v1` plans an optional short pause `_!` (coordinating and adversative
/// conjunctions; the teacher pauses before `ve`, `ki`, `ama` in about 30% of cases), unless they
/// end their clause.
const TTS_SHORT_PAUSE_WORDS: [&str; 13] = [
    "ve", "veya", "yahut", "yoksa", "ama", "fakat", "ancak", "lakin", "çünkü", "ki", "oysa",
    "halbuki", "hatta",
];

/// A word as the text gives it, before phonemisation.
#[derive(Clone, Debug)]
pub(crate) struct TextWord {
    /// Byte span in the input (a joined pair spans both words and the space).
    pub span: Range<usize>,
    /// The lexicon key: Turkish lower case, NFC, `’` as `'`, without a hyphen (`kuvve-i` →
    /// `kuvvei`; a joined pair without its space, `istermisin`).
    pub key: String,
    /// Byte span of the text between the previous word (or the start) and this word.
    pub gap: Range<usize>,
    /// A clause boundary falls before this word.
    pub boundary: bool,
    /// The pauses before this word.
    pub pauses: SmallVec<[Pause; 4]>,
    /// The clause's number (for its mark).
    pub clause: usize,
    /// The word reads a number (`yirmi` of `20`).
    pub number: bool,
}

/// Letters a word is made of: Latin letters, the Turkish ones and the circumflex vowels (and a
/// combining circumflex).
pub(crate) fn is_word_letter(c: char) -> bool {
    c.is_ascii_alphabetic()
        || matches!(
            c,
            'Ç' | 'Ğ'
                | 'İ'
                | 'Ö'
                | 'Ş'
                | 'Ü'
                | 'ç'
                | 'ğ'
                | 'ı'
                | 'ö'
                | 'ş'
                | 'ü'
                | 'â'
                | 'î'
                | 'û'
                | 'Â'
                | 'Î'
                | 'Û'
                | '\u{0302}'
        )
}

/// A combining diacritic (U+0300–U+036F): part of a letter in text written decomposed (NFD).
fn is_combining(c: char) -> bool {
    ('\u{0300}'..='\u{036F}').contains(&c)
}

/// A letter of a word: with `marks`, a combining diacritic too.
fn is_letter_in(c: char, marks: bool) -> bool {
    is_word_letter(c) || (marks && is_combining(c))
}

fn is_apostrophe(c: char) -> bool {
    c == '\'' || c == '’'
}

/// The lexicon key of a word's text.
pub(crate) fn word_key(text: &str) -> String {
    let lower = to_turkish_lower(text);
    lower
        .nfc()
        .filter(|&c| c != '-')
        .map(|c| if c == '’' { '\'' } else { c })
        .collect()
}

/// Scans one word from `start` (a letter): letters, then groups of an apostrophe and letters,
/// then an optional izafet `-i` not followed by a letter. Returns the end.
fn scan_word(text: &str, start: usize, marks: bool) -> usize {
    let bytes_end = text.len();
    let mut end = start;
    let mut chars = text[start..].char_indices().peekable();
    // letters
    while let Some(&(off, c)) = chars.peek() {
        if is_letter_in(c, marks) {
            end = start + off + c.len_utf8();
            chars.next();
        } else {
            break;
        }
    }
    // apostrophe groups
    loop {
        let rest = &text[end..];
        let mut it = rest.chars();
        match it.next() {
            Some(a) if is_apostrophe(a) => {
                let after = end + a.len_utf8();
                let letters: usize = text[after..]
                    .chars()
                    .take_while(|&c| is_letter_in(c, marks))
                    .map(char::len_utf8)
                    .sum();
                if letters == 0 {
                    break;
                }
                end = after + letters;
            }
            _ => break,
        }
    }
    // izafet -i
    let rest = &text[end..];
    let mut it = rest.chars();
    if let (Some('-'), Some(i)) = (it.next(), it.next()) {
        if (i == 'i' || i == 'I') && !it.next().is_some_and(is_word_letter) {
            end += 1 + i.len_utf8();
        }
    }
    debug_assert!(end <= bytes_end);
    end
}

/// Words split at a small letter followed by a capital (observed: `nasıI` is read `nası I`,
/// `iPhone` `i Phone`).
fn split_case(text: &str, span: Range<usize>, out: &mut Vec<Range<usize>>) {
    let word = &text[span.clone()];
    let mut start = span.start;
    let mut prev: Option<char> = None;
    for (off, c) in word.char_indices() {
        if let Some(p) = prev {
            if c.is_uppercase() && p.is_alphabetic() && !p.is_uppercase() {
                out.push(start..span.start + off);
                start = span.start + off;
            }
        }
        prev = Some(c);
    }
    out.push(start..span.end);
}

/// Digit words, for numbers read digit by digit.
const DIGIT_WORDS: [&str; 10] = [
    "sıfır", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz",
];

/// A number in the text: digit groups joined by `.` `,` `:` `/` `-`, with an optional `%`
/// before it, that the verbaliser did not read. Pushes its words and returns its end: each
/// leading zero as `sıfır` (`07` sıfır yedi), a decimal comma as `virgül`, a dot between groups
/// as `nokta` unless the groups are thousands (`1.250`), `:` `/` `-` silent, `%` as `yüzde`.
fn scan_number(text: &str, start: usize, words: &mut Vec<String>) -> usize {
    let b = text.as_bytes();
    let mut i = start;
    if b.get(i) == Some(&b'%') {
        words.push("yüzde".to_string());
        i += 1;
    }
    let mut runs: Vec<Range<usize>> = Vec::new();
    let mut seps: Vec<u8> = Vec::new();
    loop {
        let s = i;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        runs.push(s..i);
        if i + 1 < b.len()
            && matches!(b[i], b'.' | b',' | b':' | b'/' | b'-')
            && b[i + 1].is_ascii_digit()
        {
            seps.push(b[i]);
            i += 1;
        } else {
            break;
        }
    }
    let mut k = 0;
    while k < runs.len() {
        // thousands: a group of 1-3 digits followed by `.`-separated groups of exactly 3
        let mut group = text[runs[k].clone()].to_string();
        let mut j = k;
        if (1..=3).contains(&runs[k].len()) {
            while j < seps.len() && seps[j] == b'.' && runs[j + 1].len() == 3 {
                group.push_str(&text[runs[j + 1].clone()]);
                j += 1;
            }
        }
        read_digits(&group, words);
        if let Some(&sep) = seps.get(j) {
            match sep {
                b',' => words.push("virgül".to_string()),
                b'.' => words.push("nokta".to_string()),
                _ => {}
            }
        }
        k = j + 1;
    }
    i
}

/// A group of digits as words: leading zeros as `sıfır` each, the rest as a cardinal; more than
/// 15 digits digit by digit.
fn read_digits(run: &str, words: &mut Vec<String>) {
    if run.len() > 15 {
        for d in run.bytes() {
            words.push(DIGIT_WORDS[usize::from(d - b'0')].to_string());
        }
        return;
    }
    let zeros = run.bytes().take_while(|&d| d == b'0').count();
    for _ in 0..zeros {
        words.push(DIGIT_WORDS[0].to_string());
    }
    if zeros == run.len() {
        return;
    }
    let n: i64 = run[zeros..].parse().unwrap_or(0);
    for w in crate::normalization::TurkishNumberConverter::number_to_words(n).split(' ') {
        words.push(w.to_string());
    }
}

/// The words of `text`, with their gaps, boundaries and pauses, and the gap after the last word.
/// With `marks` (`tts-v2`), combining diacritics stay inside their word, so text written
/// decomposed (NFD) reads as composed text; the key is composed and drops the dot of `i̇`.
/// Pauses:
/// one `_:` at a clause boundary, one `_!` at a quote or bracket, no pause after a word with an
/// apostrophe (the teacher pauses there in under 5% of cases), and `_!` before a conjunction.
pub(crate) fn segment(text: &str, words: &mut Vec<TextWord>, marks: bool) -> Range<usize> {
    words.clear();
    // (span, key, number word)
    let mut spans: Vec<(Range<usize>, String, bool)> = Vec::new();
    let mut parts: Vec<Range<usize>> = Vec::new();
    let mut number_words: Vec<String> = Vec::new();
    let mut i = 0;
    while i < text.len() {
        let c = match text[i..].chars().next() {
            Some(c) => c,
            None => break,
        };
        let percent_number = c == '%'
            && text[i + 1..]
                .chars()
                .next()
                .is_some_and(|d| d.is_ascii_digit());
        if is_word_letter(c) && c != '\u{0302}' {
            let end = scan_word(text, i, marks);
            parts.clear();
            split_case(text, i..end, &mut parts);
            for s in parts.drain(..) {
                // decomposed text is composed before it is lower-cased (I + dot above is İ, i)
                let mut key = if marks {
                    word_key(&text[s.clone()].nfc().collect::<String>())
                } else {
                    word_key(&text[s.clone()])
                };
                if marks {
                    key.retain(|c| c != '\u{0307}');
                }
                spans.push((s, key, false));
            }
            i = end;
        } else if c.is_ascii_digit() || percent_number {
            number_words.clear();
            let end = scan_number(text, i, &mut number_words);
            for w in number_words.drain(..) {
                spans.push((i..end, w, true));
            }
            i = end;
            // an apostrophe suffix after a number is read as its own word (`1923'te` ... `'te`)
            let mut it = text[i..].chars();
            if let (Some(a), Some(l)) = (it.next(), it.next()) {
                if is_apostrophe(a) && is_word_letter(l) && l != '\u{0302}' {
                    let s = i + a.len_utf8();
                    let e = scan_word(text, s, marks);
                    let key = format!("'{}", word_key(&text[s..e]));
                    spans.push((s..e, key, true));
                    i = e;
                }
            }
        } else {
            i += c.len_utf8();
        }
    }
    let mut prev_end = 0;
    for (span, key, number) in spans {
        // the words of one number share its span: only the first has the gap before it
        let gap = prev_end.min(span.start)..span.start;
        words.push(TextWord {
            gap,
            span: span.clone(),
            key,
            boundary: false,
            pauses: SmallVec::new(),
            clause: 0,
            number,
        });
        prev_end = prev_end.max(span.end);
    }
    let final_gap = prev_end..text.len();
    for k in 1..words.len() {
        let gap = &text[words[k].gap.clone()];
        let mut boundary = gap.chars().any(|c| CLAUSE_PUNCT.contains(&c));
        // A full stop followed by a word in small letters does not end the clause (observed:
        // `vb. farklılıklar`, `karışmam. dedi`).
        if boundary
            && gap.trim() == "."
            && text[words[k].span.clone()]
                .chars()
                .next()
                .is_some_and(char::is_lowercase)
        {
            boundary = false;
        }
        let quote = gap.chars().any(|c| QUOTES.contains(&c));
        let mut pauses = SmallVec::new();
        if boundary {
            pauses.push(Pause::ClauseEnd);
        } else if quote {
            pauses.push(Pause::Short);
        }
        words[k].boundary = boundary;
        words[k].pauses = pauses;
    }
    for k in 1..words.len() {
        if !TTS_SHORT_PAUSE_WORDS.contains(&words[k].key.as_str()) || !words[k].pauses.is_empty() {
            continue;
        }
        let ends_clause = k + 1 == words.len() || words[k + 1].boundary;
        if !ends_clause {
            words[k].pauses.push(Pause::Short);
        }
    }
    let mut clause = 0;
    for w in words.iter_mut() {
        if w.boundary {
            clause += 1;
        }
        w.clause = clause;
    }
    final_gap
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys(text: &str) -> Vec<String> {
        let mut w = Vec::new();
        segment(text, &mut w, false);
        w.into_iter().map(|w| w.key).collect()
    }

    #[test]
    fn words_split_and_keep_apostrophes() {
        assert_eq!(keys("Durumu nasıI?"), ["durumu", "nası", "ı"]);
        assert_eq!(keys("iPhone ayarları"), ["i", "phone", "ayarları"]);
        assert_eq!(keys("Okumak ister misin?"), ["okumak", "ister", "misin"]);
        assert_eq!(keys("Ali’nin kuvve-i"), ["ali'nin", "kuvvei"]);
    }

    #[test]
    fn pauses_follow_the_gaps() {
        let mut w = Vec::new();
        segment(
            "Ali'nin kitabı İstanbul'a, Kemal'e \"ve\" Ankara'dan geldi.",
            &mut w,
            false,
        );
        let p: Vec<&[Pause]> = w.iter().map(|w| w.pauses.as_slice()).collect();
        // no pause after an apostrophe; one `_:` at a comma; `_!` at a quote
        assert_eq!(p[1], []);
        assert_eq!(p[3], [Pause::ClauseEnd]);
        assert_eq!(p[4], [Pause::Short]);
        assert!(w[3].boundary && !w[1].boundary);
    }

    #[test]
    fn short_pause_unless_the_conjunction_ends_its_clause() {
        let mut w = Vec::new();
        segment("Yaptı ama kancıklık. İyi ama, neydi o", &mut w, false);
        assert_eq!(w[1].pauses.as_slice(), [Pause::Short]);
        assert!(w[4].pauses.is_empty());
    }

    #[test]
    fn decomposed_text_keeps_its_words_with_marks() {
        let nfd = "C\u{0327}ic\u{0327}ek gu\u{0308}zel i\u{0307}stanbul I\u{0307}zmir";
        let mut w = Vec::new();
        segment(nfd, &mut w, true);
        let keys: Vec<&str> = w.iter().map(|w| w.key.as_str()).collect();
        assert_eq!(keys, ["çiçek", "güzel", "istanbul", "izmir"]);
        assert_eq!(&nfd[w[0].span.clone()], "C\u{0327}ic\u{0327}ek");
        segment(nfd, &mut w, false);
        assert!(w.len() > 4);
    }

    #[test]
    fn numbers_are_read_as_words() {
        assert_eq!(keys("20:00'de"), ["yirmi", "sıfır", "sıfır", "'de"]);
        assert_eq!(keys("07.30"), ["sıfır", "yedi", "nokta", "otuz"]);
        assert_eq!(keys("%25"), ["yüzde", "yirmi", "beş"]);
    }
}
