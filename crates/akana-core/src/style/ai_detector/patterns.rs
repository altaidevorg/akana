//! Mined AI patterns: openers ("elbette"), closers, sentence starts and phrases ("önemli bir
//! rol oynar") found far more often in AI answers than in human text.
//!
//! The list and each pattern's AI / human rates come from the embedded model
//! ([`super::model::Model::patterns`]); this module finds them in a text and turns the hits
//! into the five pattern inputs. Word and sentence splitting are frozen to the definitions
//! the patterns were mined with, so they intentionally differ from
//! [`crate::tokenization`].

use std::sync::LazyLock;

use ahash::AHashMap;
use regex::Regex;

use super::model::{PatternPlace, PatternSpec, MODEL};
use super::tokens::MARKDOWN;
use crate::phonology::alphabet::to_turkish_lower;

/// A word: letters (and letter-like numerals), optionally followed by `'` and more letters.
static WORD: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"[\p{L}\p{Nl}\p{No}]+(?:'[\p{L}\p{Nl}\p{No}]+)?").expect("valid regex")
});

static INDEX: LazyLock<AHashMap<(PatternPlace, &'static str), &'static PatternSpec>> =
    LazyLock::new(|| {
        MODEL
            .patterns
            .iter()
            .map(|p| ((p.kind, p.pattern.as_str()), p))
            .collect()
    });

/// Longest pattern (in words) per place; longer candidates cannot match.
static MAX_WORDS: LazyLock<AHashMap<PatternPlace, usize>> = LazyLock::new(|| {
    let mut m = AHashMap::new();
    for p in &MODEL.patterns {
        let n = p.pattern.split(' ').count();
        let e = m.entry(p.kind).or_insert(0);
        *e = (*e).max(n);
    }
    m
});

fn max_words(place: PatternPlace, cap: usize) -> usize {
    MAX_WORDS.get(&place).copied().unwrap_or(0).min(cap)
}

/// Turkish-lowercased words of a text, markdown symbols removed.
pub(super) fn pattern_words(text: &str) -> Vec<String> {
    let low = to_turkish_lower(&MARKDOWN.replace_all(text, " "));
    WORD.find_iter(&low)
        .map(|m| m.as_str().to_string())
        .collect()
}

/// Whitespace as Python's `\s` sees it (the patterns were mined in Python).
fn is_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// Sentence pieces: split after `.!?…` followed by whitespace, and at line breaks.
fn sentences(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut prev: Option<char> = None;
    let mut it = text.char_indices().peekable();
    while let Some((i, c)) = it.next() {
        let after_stop = matches!(prev, Some('.' | '!' | '?' | '…'));
        let sep: Option<fn(char) -> bool> = if after_stop && is_space(c) {
            Some(is_space)
        } else if c == '\n' {
            Some(|c| c == '\n')
        } else {
            None
        };
        match sep {
            Some(cont) => {
                out.push(&text[start..i]);
                let mut end = i + c.len_utf8();
                let mut last = c;
                while let Some(&(j, d)) = it.peek() {
                    if !cont(d) {
                        break;
                    }
                    end = j + d.len_utf8();
                    last = d;
                    it.next();
                }
                start = end;
                prev = Some(last);
            }
            None => prev = Some(c),
        }
    }
    out.push(&text[start..]);
    out
}

/// Mined patterns present in a text, each once, ordered by (place name, pattern): the order
/// the training code summed `pat_evidence` in, so the sum is bit-identical.
pub(super) fn found_patterns(text: &str) -> Vec<&'static PatternSpec> {
    let mut hits: Vec<&'static PatternSpec> = Vec::new();
    let mut buf = String::new();
    let mut look = |place: PatternPlace, words: &[String]| {
        buf.clear();
        for (i, w) in words.iter().enumerate() {
            if i > 0 {
                buf.push(' ');
            }
            buf.push_str(w);
        }
        if let Some(p) = INDEX.get(&(place, buf.as_str())) {
            hits.push(p);
        }
    };
    let w = pattern_words(text);
    for k in 1..=max_words(PatternPlace::Opener, 4).min(w.len()) {
        look(PatternPlace::Opener, &w[..k]);
    }
    for k in 1..=max_words(PatternPlace::Closer, 5).min(w.len()) {
        look(PatternPlace::Closer, &w[w.len() - k..]);
    }
    let max_start = max_words(PatternPlace::SentStart, 3);
    for sent in sentences(text) {
        let sw = pattern_words(sent);
        // A sentence start needs at least one more word after it.
        for k in 1..=max_start.min(sw.len().saturating_sub(1)) {
            look(PatternPlace::SentStart, &sw[..k]);
        }
    }
    for n in 2..=max_words(PatternPlace::Phrase, 4) {
        for win in w.windows(n) {
            look(PatternPlace::Phrase, win);
        }
    }
    hits.sort_by(|a, b| {
        (a.kind.name(), a.pattern.as_str()).cmp(&(b.kind.name(), b.pattern.as_str()))
    });
    hits.dedup_by(|a, b| std::ptr::eq(*a, *b));
    hits
}

/// `[pat_opener, pat_closer, pat_sent_start_n, pat_phrase_n, pat_evidence]`.
pub(super) fn pattern_features(hits: &[&PatternSpec]) -> [f64; 5] {
    let count = |place: PatternPlace| hits.iter().filter(|p| p.kind == place).count() as f64;
    [
        f64::from(u8::from(count(PatternPlace::Opener) > 0.0)),
        f64::from(u8::from(count(PatternPlace::Closer) > 0.0)),
        count(PatternPlace::SentStart),
        count(PatternPlace::Phrase),
        hits.iter().map(|p| p.weight).sum(),
    ]
}
