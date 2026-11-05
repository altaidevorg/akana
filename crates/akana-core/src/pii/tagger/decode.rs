//! From the tagger models' tag log-probabilities to spans, as scripts/pii_train/tagger.py decodes
//! them: overlapping windows stitched, each token taking its scores from the window where it sits
//! farthest from an edge; the operating point's bias; BIO-constrained Viterbi over the whole text;
//! spans widened to whole words; and the default policy that leaves call-centre and service
//! numbers and role mailboxes unmasked (scripts/pii_eval/systems.py's `public_value`).

use std::sync::LazyLock;

use regex::Regex;
use unicode_categories::UnicodeCategories;

/// The tagger's labels, in the models' tag order: O, then the B- and I- tags of each label.
pub const LABELS: [&str; 9] = [
    "PERSON", "USERNAME", "ADDRESS", "PHONE", "EMAIL", "URL", "ID", "SECRET", "DATE",
];
/// Tags per token: O and two for each label.
pub const N_TAGS: usize = 1 + 2 * LABELS.len();
/// A window's tokens between its special tokens (256 with them), and the tokens two windows share.
pub const WINDOW_TOKENS: usize = 254;
pub const WINDOW_STRIDE: usize = 64;

const PERSON: usize = 0;
const PHONE: usize = 3;
const EMAIL: usize = 4;

/// One token's tag log-probabilities.
pub type Row = [f32; N_TAGS];

/// The B- tag of a label.
pub fn begin_tag(label: usize) -> usize {
    1 + 2 * label
}

fn is_inside_tag(tag: usize) -> bool {
    tag > 0 && tag.is_multiple_of(2)
}

fn label_of(tag: usize) -> usize {
    (tag - 1) / 2
}

/// The content-token ranges of a text's windows, as Hugging Face's tokenizer cuts its overflowing
/// windows: up to 254 tokens each, every window starting 190 tokens after the last, the last ending
/// with the text. A text without tokens still has one (empty) window.
pub fn windows(n_tokens: usize) -> Vec<(usize, usize)> {
    let step = WINDOW_TOKENS - WINDOW_STRIDE;
    let mut out = Vec::new();
    let mut start = 0;
    loop {
        let stop = (start + WINDOW_TOKENS).min(n_tokens);
        out.push((start, stop));
        if stop == n_tokens {
            return out;
        }
        start += step;
    }
}

/// Every token's scores from the window where it sits farthest from an edge (a window's last
/// tokens see no right context and may end inside a word); the first such window on a tie.
/// `rows[w]` holds window `w`'s content tokens.
pub fn stitch(n_tokens: usize, windows: &[(usize, usize)], rows: &[Vec<Row>]) -> Vec<Row> {
    let mut best: Vec<Option<(usize, Row)>> = vec![None; n_tokens];
    for (&(start, _), window) in windows.iter().zip(rows) {
        let n = window.len();
        for (k, row) in window.iter().enumerate() {
            let depth = k.min(n - 1 - k);
            let slot = &mut best[start + k];
            if slot.is_none_or(|(d, _)| depth > d) {
                *slot = Some((depth, *row));
            }
        }
    }
    best.into_iter()
        .map(|slot| slot.map_or([f32::NEG_INFINITY; N_TAGS], |(_, row)| row))
        .collect()
}

/// The operating point: `bias` added to the B- and I- tags of `labels`, or, with no labels,
/// taken off the O tag (which raises every label alike).
pub fn apply_bias(rows: &mut [Row], bias: f32, labels: &[usize]) {
    if bias == 0.0 {
        return;
    }
    for row in rows {
        if labels.is_empty() {
            row[0] -= bias;
        }
        for &label in labels {
            row[begin_tag(label)] += bias;
            row[begin_tag(label) + 1] += bias;
        }
    }
}

/// Whether tag `cur` may follow `prev` (`None`: the text's start): an I- tag only after its own
/// label's B- or I- tag.
fn allowed(prev: Option<usize>, cur: usize) -> bool {
    !is_inside_tag(cur) || prev.is_some_and(|p| p != 0 && label_of(p) == label_of(cur))
}

/// The best tag sequence under the BIO constraints, in f64 like the Python decoder, the first
/// maximum winning a tie (numpy's argmax).
pub fn viterbi(rows: &[Row]) -> Vec<usize> {
    let Some(first) = rows.first() else {
        return Vec::new();
    };
    let mut best = [0f64; N_TAGS];
    for (t, b) in best.iter_mut().enumerate() {
        *b = if allowed(None, t) {
            f64::from(first[t])
        } else {
            f64::NEG_INFINITY
        };
    }
    let mut back = vec![[0usize; N_TAGS]; rows.len()];
    for (i, row) in rows.iter().enumerate().skip(1) {
        let mut next = [0f64; N_TAGS];
        for t in 0..N_TAGS {
            let (mut arg, mut max) = (0, f64::NEG_INFINITY);
            for (p, &score) in best.iter().enumerate() {
                let s = if allowed(Some(p), t) {
                    score
                } else {
                    f64::NEG_INFINITY
                };
                if p == 0 || s > max {
                    (arg, max) = (p, s);
                }
            }
            back[i][t] = arg;
            next[t] = max + f64::from(row[t]);
        }
        best = next;
    }
    let mut t = 0;
    for (k, &b) in best.iter().enumerate() {
        if b > best[t] {
            t = k;
        }
    }
    let mut out = vec![t; rows.len()];
    for i in (1..rows.len()).rev() {
        t = back[i][t];
        out[i - 1] = t;
    }
    out
}

/// Spans (start, end, label) from the tokens' offsets and their tags.
pub fn spans_from_tags(offsets: &[(usize, usize)], tags: &[usize]) -> Vec<(usize, usize, usize)> {
    let mut spans = Vec::new();
    let mut cur: Option<(usize, usize, usize)> = None;
    for (&(a, b), &tag) in offsets.iter().zip(tags) {
        if tag == 0 {
            spans.extend(cur.take());
        } else if !is_inside_tag(tag) || cur.is_none_or(|c| c.2 != label_of(tag)) {
            spans.extend(cur.take());
            cur = Some((a, b, label_of(tag)));
        } else if let Some(c) = cur.as_mut() {
            c.1 = b;
        }
    }
    spans.extend(cur);
    spans
}

/// Characters inside a code, date, e-mail or link that join its parts (29/9/2016, SRM-2021-K2VM6,
/// ad.soyad@...).
const JOINERS: [char; 6] = ['-', '/', '.', '_', '@', '+'];

/// Python's `str.isalnum` for one character: a letter or a number.
fn is_alnum(c: char) -> bool {
    c.is_letter() || c.is_number()
}

/// Spans that start or end inside a word widened to the whole word. A tag sequence can stop after
/// a word's first piece ("Kr" of "Kristof", "pınar" of "pınarla", "29" of "29/9/2016"). Names
/// widen over letters and digits only, so an apostrophe's suffix stays out; other values also over
/// the joiners inside codes, dates, e-mails and links. Offsets are character indices of `text`.
pub fn snap(text: &[char], spans: &[(usize, usize, usize)]) -> Vec<(usize, usize, usize)> {
    let n = text.len();
    let mut out = Vec::with_capacity(spans.len());
    for &(mut s, mut e, label) in spans {
        let joiners: &[char] = if label == PERSON { &[] } else { &JOINERS };
        let inside = |i: usize| {
            let c = text[i];
            is_alnum(c)
                || (joiners.contains(&c)
                    && 0 < i
                    && i + 1 < n
                    && is_alnum(text[i - 1])
                    && is_alnum(text[i + 1]))
        };
        while 0 < s && s < n && inside(s - 1) && inside(s) {
            s -= 1;
        }
        while 0 < e && e < n && inside(e) && inside(e - 1) {
            e += 1;
        }
        out.push((s, e, label));
    }
    merge(out)
}

/// Overlapping spans resolved: the earlier (and, from one start, the longer) one stays, extended
/// by a later one of its own label.
fn merge(mut spans: Vec<(usize, usize, usize)>) -> Vec<(usize, usize, usize)> {
    spans.sort_by_key(|&(s, e, _)| (s, std::cmp::Reverse(e - s)));
    let mut kept: Vec<(usize, usize, usize)> = Vec::with_capacity(spans.len());
    for span in spans {
        if let Some(last) = kept.last_mut() {
            if span.0 < last.1 {
                if span.1 > last.1 && span.2 == last.2 {
                    last.1 = span.1;
                }
                continue;
            }
        }
        kept.push(span);
    }
    kept
}

const SPOKEN_DIGITS: [(&str, &str); 10] = [
    ("sıfır", "0"),
    ("bir", "1"),
    ("iki", "2"),
    ("üç", "3"),
    ("dört", "4"),
    ("beş", "5"),
    ("altı", "6"),
    ("yedi", "7"),
    ("sekiz", "8"),
    ("dokuz", "9"),
];
const NUMBER_WORDS: [&str; 11] = [
    "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan", "yüz", "bin",
];
const ROLE_MAILBOXES: [&str; 17] = [
    "info",
    "destek",
    "iletisim",
    "kvkk",
    "muhasebe",
    "satis",
    "noreply",
    "no-reply",
    "insankaynaklari",
    "basin",
    "support",
    "help",
    "contact",
    "sales",
    "yardim",
    "musterihizmetleri",
    "bilgi",
];

static WORD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\w+").expect("a valid pattern"));
static MAILBOX_END: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\s*(?:@|\[at\]|\(at\)| at )").expect("a valid pattern"));

/// A call-centre or service number (444, 0850, 0800, 0900, or four digits at most), or a role
/// mailbox: public, left unmasked by the default policy. A number read out with tens or hundreds
/// ("beş yüz otuz iki") is not read, so it stays personal.
pub fn public_value(value: &str, label: usize) -> bool {
    match label {
        PHONE => {
            let spoken = WORD.replace_all(value, |m: &regex::Captures| {
                let word = m[0].to_lowercase();
                SPOKEN_DIGITS
                    .iter()
                    .find(|(w, _)| *w == word)
                    .map_or(m[0].to_string(), |(_, d)| d.to_string())
            });
            if WORD
                .find_iter(&spoken)
                .any(|m| NUMBER_WORDS.contains(&m.as_str().to_lowercase().as_str()))
            {
                return false;
            }
            let digits: String = spoken
                .chars()
                .filter(|c| c.is_number_decimal_digit())
                .collect();
            let digits = ["0090", "90"]
                .iter()
                .find_map(|p| {
                    digits
                        .strip_prefix(p)
                        .filter(|rest| rest.chars().count() == 10)
                })
                .unwrap_or(&digits)
                .trim_start_matches('0');
            ["444", "850", "800", "900"]
                .iter()
                .any(|p| digits.starts_with(p))
                || digits.chars().count() <= 4
        }
        EMAIL => {
            let lower = value.to_lowercase();
            let local = MAILBOX_END.split(&lower).next().unwrap_or("").trim();
            ROLE_MAILBOXES.contains(&local)
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tag(name: &str) -> usize {
        if name == "O" {
            return 0;
        }
        let (prefix, label) = name.split_at(2);
        let label = LABELS.iter().position(|l| *l == label).unwrap();
        begin_tag(label) + usize::from(prefix == "I-")
    }

    fn row(best: &str) -> Row {
        let mut r = [-10.0; N_TAGS];
        r[tag(best)] = -0.1;
        r
    }

    #[test]
    fn windows_overlap_by_the_stride_and_end_with_the_text() {
        assert_eq!(windows(0), vec![(0, 0)]);
        assert_eq!(windows(254), vec![(0, 254)]);
        assert_eq!(windows(255), vec![(0, 254), (190, 255)]);
        assert_eq!(windows(600), vec![(0, 254), (190, 444), (380, 600)]);
    }

    #[test]
    fn a_token_takes_its_scores_from_its_deepest_window() {
        let w = vec![(0, 5), (2, 7)];
        let rows = vec![
            vec![row("O"), row("O"), row("O"), row("O"), row("B-DATE")],
            vec![row("O"), row("B-DATE"), row("B-PERSON"), row("O"), row("O")],
        ];
        let s = stitch(7, &w, &rows);
        // token 3: depth 1 in both windows: the first window wins the tie
        assert_eq!(s[3], row("O"));
        // token 4: depth 0 in the first window (its last token), 2 in the second
        assert_eq!(s[4], row("B-PERSON"));
    }

    #[test]
    fn viterbi_never_starts_inside_a_span() {
        let mut r = row("I-PERSON");
        r[tag("B-PERSON")] = -0.5;
        assert_eq!(
            viterbi(&[r, row("I-PERSON")]),
            vec![tag("B-PERSON"), tag("I-PERSON")]
        );
    }

    #[test]
    fn viterbi_keeps_an_inside_tag_with_its_own_label() {
        let mut second = row("I-DATE");
        second[tag("I-PERSON")] = -0.2;
        assert_eq!(
            viterbi(&[row("B-PERSON"), second]),
            vec![tag("B-PERSON"), tag("I-PERSON")]
        );
    }

    #[test]
    fn the_bias_raises_only_the_chosen_labels() {
        let mut rows = vec![row("O")];
        apply_bias(&mut rows, 2.0, &[0]);
        assert_eq!(rows[0][tag("B-PERSON")], -8.0);
        assert_eq!(rows[0][tag("B-DATE")], -10.0);
        apply_bias(&mut rows, 1.0, &[]);
        assert!((rows[0][0] - -1.1).abs() < 1e-6);
    }

    #[test]
    fn spans_follow_the_tags() {
        let offsets = [(0, 5), (6, 12), (13, 15), (16, 20)];
        let tags = [tag("B-PERSON"), tag("I-PERSON"), tag("O"), tag("B-DATE")];
        assert_eq!(
            spans_from_tags(&offsets, &tags),
            vec![(0, 12, 0), (16, 20, 8)]
        );
    }

    fn snapped(text: &str, s: usize, e: usize, label: usize) -> String {
        let chars: Vec<char> = text.chars().collect();
        let out = snap(&chars, &[(s, e, label)]);
        chars[out[0].0..out[0].1].iter().collect()
    }

    #[test]
    fn snap_widens_to_the_whole_word() {
        assert_eq!(snapped("Kristof geldi", 0, 2, 0), "Kristof");
        assert_eq!(snapped("tarih 29/9/2016 idi", 6, 8, 8), "29/9/2016");
        // a name stops at the apostrophe; a date's joiners do not apply to it
        assert_eq!(snapped("Ahmet'in", 0, 3, 0), "Ahmet");
        assert_eq!(snapped("ahmet-ali", 0, 3, 0), "ahmet");
    }

    #[test]
    fn overlapping_spans_merge_into_the_earlier_one() {
        assert_eq!(
            merge(vec![(4, 9, 0), (0, 6, 0), (10, 12, 8)]),
            vec![(0, 9, 0), (10, 12, 8)]
        );
        assert_eq!(merge(vec![(0, 6, 0), (4, 9, 8)]), vec![(0, 6, 0)]);
    }

    #[test]
    fn public_numbers_and_role_mailboxes_stay_unmasked() {
        assert!(public_value("444 0 444", PHONE));
        assert!(public_value("0850 222 33 44", PHONE));
        assert!(public_value("+90 850 222 33 44", PHONE));
        assert!(public_value("112", PHONE));
        assert!(!public_value("0532 111 22 33", PHONE));
        assert!(public_value("dört dört dört sıfır sıfır", PHONE));
        assert!(!public_value("beş yüz otuz iki", PHONE));
        assert!(public_value("Destek@firma.com.tr", EMAIL));
        assert!(public_value("info [at] firma.com", EMAIL));
        assert!(!public_value("ayse.kaya@gmail.com", EMAIL));
        assert!(!public_value("info@x.com", PERSON));
    }
}
