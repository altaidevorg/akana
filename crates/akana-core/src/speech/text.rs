//! Text to words, clause boundaries and pauses, as the reference fixtures show eSpeak NG doing
//! it for Turkish. Every rule here was inferred from the fixtures' outputs.

use super::Pause;
use crate::phonology::to_turkish_lower;
use smallvec::SmallVec;
use std::ops::Range;
use unicode_normalization::UnicodeNormalization;

/// Punctuation that ends a clause: `_:` then `_` (and a new clause for the tonic).
const CLAUSE_PUNCT: [char; 7] = [',', '.', ';', ':', '!', '?', '…'];
/// Quotation marks and brackets: a pause pair `_: _:` where they stand (observed for `"`, `(`, `)`).
const QUOTES: [char; 7] = ['"', '“', '”', '«', '»', '(', ')'];
/// Words that take the short pause `_!` before them unless they end their clause.
const SHORT_PAUSE_WORDS: [&str; 7] = ["ve", "veya", "yoksa", "veda", "ama", "ki", "kime"];
/// Word pairs eSpeak reads as one word (observed in the fixtures).
const JOINS: [(&str, &str); 3] = [("ister", "misin"), ("ister", "misiniz"), ("ne", "yaptın")];
/// Words with an apostrophe after which eSpeak makes no pause (observed: `Allah'a`, a word it
/// pronounces as a whole, with clear l).
const NO_APOSTROPHE_PAUSE: [&str; 1] = ["allah'a"];

/// A word as the text gives it, before phonemisation.
#[derive(Clone, Debug)]
pub(crate) struct TextWord {
    /// Byte span in the input (a joined pair spans both words and the space).
    pub span: Range<usize>,
    /// The lexicon key: Turkish lower case, NFC, `’` as `'`, without a hyphen (`kuvve-i` →
    /// `kuvvei`; a joined pair without its space, `istermisin`).
    pub key: String,
    /// An izafet compound written with a hyphen (`kuvve-i`).
    pub izafet: bool,
    /// The word has an apostrophe between letters (`Ali'nin`).
    pub apostrophe: bool,
    /// Byte span of the text between the previous word (or the start) and this word.
    pub gap: Range<usize>,
    /// A clause boundary falls before this word.
    pub boundary: bool,
    /// The pauses before this word.
    pub pauses: SmallVec<[Pause; 4]>,
    /// The clause's number for marks (eSpeak's clause count).
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
fn scan_word(text: &str, start: usize) -> usize {
    let bytes_end = text.len();
    let mut end = start;
    let mut chars = text[start..].char_indices().peekable();
    // letters
    while let Some(&(off, c)) = chars.peek() {
        if is_word_letter(c) {
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
                    .take_while(|&c| is_word_letter(c))
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
/// before it. Pushes its words and returns its end. Read as eSpeak reads them where the probes
/// show it: each leading zero as `sıfır` (`07` sıfır yedi, `00` sıfır sıfır), a decimal comma
/// as `virgül`, a dot between groups as `nokta` unless the groups are thousands (`1.250`), `:`
/// `/` `-` silent, `%` as `yüzde` before the number. eSpeak joins tens and units (`yirmibeş`)
/// and hundreds (`dokuzyüz`) into one word; this reads them as separate words (a recorded
/// divergence).
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
pub(crate) fn segment(text: &str, words: &mut Vec<TextWord>) -> Range<usize> {
    words.clear();
    // (span, key, number word, apostrophe)
    let mut spans: Vec<(Range<usize>, String, bool, bool)> = Vec::new();
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
            let end = scan_word(text, i);
            parts.clear();
            split_case(text, i..end, &mut parts);
            for s in parts.drain(..) {
                let key = word_key(&text[s.clone()]);
                let apostrophe = text[s.clone()].chars().any(is_apostrophe);
                spans.push((s, key, false, apostrophe));
            }
            i = end;
        } else if c.is_ascii_digit() || percent_number {
            number_words.clear();
            let end = scan_number(text, i, &mut number_words);
            for w in number_words.drain(..) {
                spans.push((i..end, w, true, false));
            }
            i = end;
            // an apostrophe suffix after a number is read as its own word, then a pause pair
            // (observed: `1923'te` ... t6 E6 _: _:)
            let mut it = text[i..].chars();
            if let (Some(a), Some(l)) = (it.next(), it.next()) {
                if is_apostrophe(a) && is_word_letter(l) && l != '\u{0302}' {
                    let s = i + a.len_utf8();
                    let e = scan_word(text, s);
                    let key = format!("'{}", word_key(&text[s..e]));
                    spans.push((s..e, key, true, true));
                    i = e;
                }
            }
        } else {
            i += c.len_utf8();
        }
    }
    // joins
    let mut merged: Vec<(Range<usize>, String, bool, bool)> = Vec::with_capacity(spans.len());
    for (s, key, number, apostrophe) in spans {
        if let Some((prev_span, prev_key, prev_number, _)) = merged.last_mut() {
            if !number
                && !*prev_number
                && &text[prev_span.end..s.start] == " "
                && JOINS.iter().any(|(a, b)| a == prev_key && *b == key)
            {
                prev_key.push_str(&key);
                prev_span.end = s.end;
                continue;
            }
        }
        merged.push((s, key, number, apostrophe));
    }
    let mut prev_end = 0;
    for (span, key, number, apostrophe) in merged {
        let izafet = !number && text[span.clone()].contains('-');
        // the words of one number share its span: only the first has the gap before it
        let gap = prev_end.min(span.start)..span.start;
        words.push(TextWord {
            gap,
            span: span.clone(),
            key,
            izafet,
            apostrophe,
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
        let prev = &words[k - 1];
        let apostrophe_pause =
            prev.apostrophe && !boundary && !NO_APOSTROPHE_PAUSE.contains(&prev.key.as_str());
        let mut pauses = SmallVec::new();
        if boundary {
            pauses.push(Pause::ClauseEnd);
            pauses.push(Pause::Edge);
        }
        if quote || apostrophe_pause {
            pauses.push(Pause::ClauseEnd);
            pauses.push(Pause::ClauseEnd);
        }
        words[k].boundary = boundary;
        words[k].pauses = pauses;
    }
    for k in 1..words.len() {
        if !SHORT_PAUSE_WORDS.contains(&words[k].key.as_str()) {
            continue;
        }
        let ends_clause = k + 1 == words.len() || words[k + 1].boundary;
        let has_pair = words[k].pauses.len() >= 2
            && words[k].pauses[words[k].pauses.len() - 1] == Pause::ClauseEnd
            && words[k].pauses[words[k].pauses.len() - 2] == Pause::ClauseEnd;
        if !ends_clause && !has_pair {
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

/// The marks of the text's clauses: the last character of each run of `[,.;:!?]` (EchoCraft's
/// `clause_marks`), indexed by eSpeak's clause number; `.` past the end.
pub(crate) fn clause_marks(text: &str, marks: &mut Vec<char>) {
    marks.clear();
    let mut run_last: Option<char> = None;
    for c in text.chars() {
        if matches!(c, ',' | '.' | ';' | ':' | '!' | '?') {
            run_last = Some(c);
        } else if let Some(m) = run_last.take() {
            marks.push(m);
        }
    }
    if let Some(m) = run_last {
        marks.push(m);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys(text: &str) -> Vec<String> {
        let mut w = Vec::new();
        segment(text, &mut w);
        w.into_iter().map(|w| w.key).collect()
    }

    #[test]
    fn words_split_join_and_keep_apostrophes() {
        assert_eq!(keys("Durumu nasıI?"), ["durumu", "nası", "ı"]);
        assert_eq!(keys("iPhone ayarları"), ["i", "phone", "ayarları"]);
        assert_eq!(keys("Okumak ister misin?"), ["okumak", "istermisin"]);
        assert_eq!(keys("Ne yaptın sen?"), ["neyaptın", "sen"]);
        assert_eq!(keys("Ali’nin kuvve-i"), ["ali'nin", "kuvvei"]);
    }

    #[test]
    fn pauses_follow_the_gaps() {
        let mut w = Vec::new();
        segment(
            "Ali'nin kitabı İstanbul'a, Kemal'e ve Ankara'dan geldi.",
            &mut w,
        );
        let p: Vec<usize> = w.iter().map(|w| w.pauses.len()).collect();
        // kitabı after an apostrophe word: `_: _:`; Kemal'e after a comma: `_: _`; ve after
        // an apostrophe word: `_: _:` and no `_!`
        assert_eq!(p, [0, 2, 0, 2, 2, 0, 2]);
        assert!(w[3].boundary && !w[1].boundary);
        assert_eq!(w[4].pauses.as_slice(), [Pause::ClauseEnd, Pause::ClauseEnd]);
    }

    #[test]
    fn short_pause_unless_the_conjunction_ends_its_clause() {
        let mut w = Vec::new();
        segment("Yaptı ama kancıklık. İyi ama, neydi o", &mut w);
        assert_eq!(w[1].pauses.as_slice(), [Pause::Short]);
        assert!(w[4].pauses.is_empty());
    }

    #[test]
    fn numbers_are_read_as_words() {
        assert_eq!(keys("20:00'de"), ["yirmi", "sıfır", "sıfır", "'de"]);
        assert_eq!(keys("07.30"), ["sıfır", "yedi", "nokta", "otuz"]);
        assert_eq!(
            keys("1.250,75 TL"),
            ["bin", "iki", "yüz", "elli", "virgül", "yetmiş", "beş", "tl"]
        );
        assert_eq!(keys("%25"), ["yüzde", "yirmi", "beş"]);
        let mut w = Vec::new();
        segment("1923'te kuruldu", &mut w);
        // the suffix word is followed by the apostrophe's pause pair
        assert_eq!(w.last().map(|w| w.pauses.len()), Some(2));
    }

    #[test]
    fn marks_are_runs_of_punctuation() {
        let mut m = Vec::new();
        clause_marks("Evet, dedi... Ne?! Peki", &mut m);
        assert_eq!(m, [',', '.', '!']);
    }
}
