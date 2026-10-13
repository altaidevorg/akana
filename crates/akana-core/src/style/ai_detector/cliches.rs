//! The cliché patterns behind the `cliche_per_100w` input: the `ConclusionHedge`,
//! `RhetoricalCalque` and `IndefiniteArticleInflation` entries of the style auditor's catalogue
//! ([`crate::style::cliches`]) as they were when the model was trained.
//!
//! This is a deliberate frozen copy: the auditor's catalogue is meant to grow, while the
//! model's input must keep its training definition. Only the count is used, with the same
//! matching rules as [`crate::style::ClicheMatcher::find_all`].

use std::sync::LazyLock;

use regex::Regex;

use crate::phonology::alphabet::to_turkish_lower;

/// Literal phrases, matched on the Turkish-lowercased text at word boundaries.
const LITERALS: [&str; 3] = [
    "göz ardı edilmemelidir",
    "unutulmamalıdır ki",
    "özetlemek gerekirse",
];

/// Variable-length patterns, matched on the original text.
static PATTERNS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    [
        r"(?i)\b(?:kritik|önemli|büyük|belirleyici|temel)\s+bir\s+rol\s+(?:oynamaktadır|oynar|üstlenmektedir|oynamaktadırlar)\b",
        r"(?i)\b(?:hayati|büyük|kritik|özel)\s+bir\s+önem\s+(?:taşımaktadır|arz\s+etmektedir|taşır)\b",
        r"(?i)\b(?:büyük|önemli|kritik)\s+bir\s+öneme\s+sahiptir\b",
        r"(?i)\bvazgeçilmez\s+bir\s+(?:unsur|parça|hale\s+gelmiştir|parçasıdır|öğe)\b",
        r"(?i)\b(?:sadece|yalnızca|salt)\s+[^.;!?\n]+?\s+değil,\s*(?:aynı\s+zamanda|bununla\s+birlikte)\b",
        r"(?i)\b(?:sadece|yalnızca)?\s*[^.;!?\n]+?\s+ile\s+sınırlı\s+kalmayıp\b",
        r"(?i)\b[^.;!?\n]+?\s+olmakla\s+birlikte,\s*[^.;!?\n]+?\s+niteliğini\s+(?:de\s+)?taşımaktadır\b",
        r"(?i)\b[^.;!?\n]+?'?[ıinunün]?\s+ötesinde,\s*[^.;!?\n]+?\s+boyutu\b",
        r"(?i)\b(?:önemli|kritik|büyük|belirleyici|temel|merkezi|stratejik|kapsamlı|başarılı|etkili|güçlü)\s+bir\s+(?:adım|rol|aşama|süreç|unsur|faktör|örnek|çalışma|katkı|dönüm\s+noktası)\b",
    ]
    .iter()
    .map(|p| Regex::new(p).expect("valid regex"))
    .collect()
});

/// Non-overlapping occurrences of `needle` in `lower` that are not part of a longer word.
fn count_literal(lower: &str, needle: &str) -> usize {
    let mut n = 0;
    let mut offset = 0;
    while let Some(pos) = lower[offset..].find(needle) {
        let start = offset + pos;
        let end = start + needle.len();
        let prev_ok = !lower[..start]
            .chars()
            .next_back()
            .is_some_and(char::is_alphabetic);
        let next_ok = !lower[end..].chars().next().is_some_and(char::is_alphabetic);
        n += usize::from(prev_ok && next_ok);
        offset = end;
    }
    n
}

/// Number of cliché matches in a text.
pub(super) fn count(text: &str) -> usize {
    let lower = to_turkish_lower(text);
    let literals: usize = LITERALS.iter().map(|l| count_literal(&lower, l)).sum();
    let patterns: usize = PATTERNS.iter().map(|re| re.find_iter(text).count()).sum();
    literals + patterns
}
