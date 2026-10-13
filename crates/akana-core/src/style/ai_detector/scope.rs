//! Text types the model was shown to be unreliable on.
//!
//! - Poems (human poems got false alarms, AI poems were mostly missed): no decision.
//! - Academic writing (about 13% false alarms on human abstracts): a warning only, because
//!   the academic cues also occur in ordinary chat answers (about 15 in 100 OpenAI answers).
//!
//! On the evaluation sets the rules caught 80% of human poems and 66% of human abstracts;
//! the poem rule wrongly flagged about 1 in 100 chat answers.

use std::sync::LazyLock;

use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::phonology::alphabet::to_turkish_lower;

/// Which unreliable text type was recognised.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ScopeKind {
    /// A poem: the detector makes no decision.
    Poem,
    /// Academic writing: the decision is made but should be read with care.
    Academic,
}

/// A text type the model is not reliable on, with a Turkish explanation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ScopeWarning {
    pub kind: ScopeKind,
    pub message: String,
}

const ACADEMIC_CUES: &[&str] = &[
    "bu çalışma",
    "bu araştırma",
    "çalışmanın amacı",
    "araştırmanın amacı",
    "yöntem",
    "bulgular",
    "anahtar kelimeler",
    "örneklem",
    "katılımcı",
    "literatür",
    "nicel",
    "nitel",
    "istatistiksel",
    "elde edilen",
    "sonuç olarak",
    "ölçeği",
    "analiz edil",
    "değerlendiril",
    "incelen",
];

/// Cues needed for the academic warning.
const MIN_ACADEMIC_CUES: usize = 2;

/// A bulleted, numbered, heading or bold line: lists are not poems.
static LIST_LINE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\s*([-*•–]|\d+[.)]|#)|\*\*").expect("valid regex"));

/// Poem shape: at least 4 lines, short lines (median ≤ 60 chars, ≤ 9 words on average) that
/// mostly do not end in `.` or `:`, and not a list.
const POEM_MIN_LINES: usize = 4;
const POEM_MAX_MEDIAN_CHARS: usize = 60;
const POEM_MAX_WORDS_PER_LINE: f64 = 9.0;
const POEM_MAX_STOP_SHARE: f64 = 0.5;

fn looks_like_poem(text: &str) -> bool {
    let lines: Vec<&str> = text
        .split(['\n', '\r'])
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    if lines.len() < POEM_MIN_LINES || lines.iter().filter(|l| LIST_LINE.is_match(l)).count() >= 2 {
        return false;
    }
    let mut lens: Vec<usize> = lines.iter().map(|l| l.chars().count()).collect();
    lens.sort_unstable();
    let n = lines.len() as f64;
    let words_per_line = lines
        .iter()
        .map(|l| l.split_whitespace().count())
        .sum::<usize>() as f64
        / n;
    let stops = lines.iter().filter(|l| l.ends_with(['.', ':'])).count() as f64 / n;
    lens[lens.len() / 2] <= POEM_MAX_MEDIAN_CHARS
        && words_per_line <= POEM_MAX_WORDS_PER_LINE
        && stops <= POEM_MAX_STOP_SHARE
}

/// Poem or academic text?
pub(super) fn out_of_scope(text: &str) -> Option<ScopeWarning> {
    if looks_like_poem(text) {
        return Some(ScopeWarning {
            kind: ScopeKind::Poem,
            message: "Bu metin şiir gibi görünüyor. Testlerimizde model şiirde güvenilir değil: \
                      insan şiirlerinin bir kısmına yanlış alarm verdi ve AI şiirlerinin çoğunu \
                      kaçırdı. Bu yüzden karar verilmedi."
                .into(),
        });
    }
    let low = to_turkish_lower(text);
    if ACADEMIC_CUES.iter().filter(|c| low.contains(*c)).count() >= MIN_ACADEMIC_CUES {
        return Some(ScopeWarning {
            kind: ScopeKind::Academic,
            message: "Bu metin akademik bir metin gibi görünüyor. Testlerimizde model akademik \
                      özetlerde daha az güvenilir: insan özetlerinin yaklaşık %13'üne yanlış \
                      alarm verdi. Sonuca dikkatle bakın."
                .into(),
        });
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lists_and_prose_are_not_poems() {
        assert!(!looks_like_poem(
            "Şunları yap:\n- su iç\n- uyu\n- yürü\n- oku"
        ));
        assert!(!looks_like_poem("Bir. İki. Üç."));
    }

    #[test]
    fn carriage_returns_split_lines() {
        let poem = "Gün batarken\rDalgalar söyler\rBir martı süzülür\rAkşam iner usulca";
        assert!(looks_like_poem(poem));
    }

    #[test]
    fn academic_cues_use_turkish_case_rules() {
        let text = "İncelenen örneklemde İstatistiksel anlamlılık arandı.";
        assert_eq!(
            out_of_scope(text).map(|s| s.kind),
            Some(ScopeKind::Academic)
        );
    }
}
