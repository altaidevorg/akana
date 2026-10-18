//! Phone numbers in any grouping, by their digits rather than by a pattern per layout.
//!
//! Every run of digits and separators (space, dot, dash, slash, parentheses, a leading `+`)
//! that is not glued to letters or digits is split at list separators (" / ", ", ") and its
//! digits checked against numbering plans: Turkish numbers with or without +90 / 0090 / 90 / 0
//! (geographic 2xx-4xx, mobile 5xx, 0800 / 0850 / 0900, 444 xxxx), and international E.164
//! numbers (8-15 digits) after `+` or `00`. This covers groupings the layout regex did not
//! (0850 222 0 222, 0532/123 45 67, 905321234567, 0 5 3 2 1 2 3 4 5 6 7, wa.me/90532...,
//! +44 20 7946 0958) and keeps dates, amounts, ISBNs, TCKNs and IBANs out by length and prefix.

use super::{PiiEntity, PiiType};
use lazy_static::lazy_static;
use regex::Regex;

lazy_static! {
    static ref DIGIT_RUN: Regex = Regex::new(r"[+(]*\d[\d \t.\-/()]*\d\)?").unwrap();
}

#[derive(Debug, PartialEq, Eq)]
enum Plan {
    Turkish,
    International,
}

/// Which numbering plan the digits of `piece` fit, if any.
fn plan(piece: &str) -> Option<Plan> {
    let digits: String = piece.chars().filter(char::is_ascii_digit).collect();
    let plus = piece.trim_start_matches('(').starts_with('+');
    let separated = piece.chars().any(|c| !c.is_ascii_digit() && c != '+');
    // A slash only after a trunk-prefixed area code (0212/555 12 34): elsewhere it is a file
    // number (2023/102251) or a date (22/11/2001).
    if piece.contains('/') {
        let head = piece.trim_start_matches('(');
        let area_slash = head.len() > 5
            && head.as_bytes()[0] == b'0'
            && head[..4].bytes().all(|b| b.is_ascii_digit())
            && head[4..].trim_start_matches(')').starts_with('/');
        if !area_slash || piece.matches('/').count() > 1 {
            return None;
        }
    }

    // Turkish: strip the country or trunk prefix, then 10 digits of area/operator + subscriber.
    let national = if let Some(rest) = digits.strip_prefix("0090") {
        Some(rest)
    } else if let Some(rest) = digits.strip_prefix("90").filter(|r| plus || r.len() == 10) {
        Some(rest)
    } else if plus {
        None
    } else if let Some(rest) = digits.strip_prefix('0') {
        Some(rest)
    } else if piece.starts_with('(') && piece[1..].find(')') == Some(3) {
        // a parenthesised area code is as clear as a prefix: (212) 555 12 34
        Some(digits.as_str())
    } else {
        None
    };
    if let Some(n) = national {
        let b = n.as_bytes();
        let turkish = n.len() == 10
            && (matches!(b[0], b'2'..=b'5')
                || n.starts_with("800")
                || n.starts_with("850")
                || n.starts_with("900"));
        if turkish {
            return Some(Plan::Turkish);
        }
    }
    // Without a trunk or country prefix only a mobile number (5xx) is taken: a bare 10-digit
    // number starting 2-4 is as likely a file or account number (the layout regex in
    // patterns.rs still matches the usual landline layouts).
    if !plus && digits.len() == 10 && digits.starts_with('5') {
        return Some(Plan::Turkish);
    }
    // 444 xxxx (7 digits, nothing before)
    if !plus && digits.len() == 7 && digits.starts_with("444") && separated {
        return Some(Plan::Turkish);
    }
    // International: + or 00, then 8-15 digits starting 1-9.
    let intl = if plus {
        Some(digits.as_str())
    } else {
        digits.strip_prefix("00")
    };
    if let Some(n) = intl {
        if (8..=15).contains(&n.len()) && !n.starts_with('0') && (separated || n.len() >= 10) {
            return Some(Plan::International);
        }
    }
    None
}

/// Splits a run at separators that list several numbers: " / ", " - ", ", ".
fn pieces(run: &str) -> Vec<(usize, &str)> {
    let mut out = Vec::new();
    let mut start = 0;
    let bytes = run.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if matches!(bytes[i], b'/' | b'-')
            && i > 0
            && i + 1 < bytes.len()
            && bytes[i - 1] == b' '
            && bytes[i + 1] == b' '
        {
            out.push((start, run[start..i - 1].trim_end()));
            start = i + 2;
            i += 2;
            continue;
        }
        // ". " ends a sentence or a list number (26. 22.11.2001): not inside a phone number.
        if bytes[i] == b'.' && i + 1 < bytes.len() && bytes[i + 1] == b' ' {
            out.push((start, &run[start..i]));
            start = i + 2;
            i += 2;
            continue;
        }
        i += 1;
    }
    out.push((start, &run[start..]));
    out
}

fn trim_piece(piece: &str) -> (usize, &str) {
    let trimmed_start = piece.trim_start_matches([' ', '\t', '.', '-', '/']);
    let lead = piece.len() - trimmed_start.len();
    let mut s = trimmed_start.trim_end_matches([' ', '\t', '.', '-', '/']);
    // Parentheses around the whole number belong to the sentence: "(0532 123 45 67)".
    if s.starts_with('(') && s.find(')') == Some(s.len() - 1) {
        return (lead + 1, &s[1..s.len() - 1]);
    }
    // So does an unbalanced one: "0532 123 45 67)".
    if s.ends_with(')') && s.matches('(').count() < s.matches(')').count() {
        s = &s[..s.len() - 1];
    }
    if s.starts_with('(') && s.matches('(').count() > s.matches(')').count() {
        return (lead + 1, &s[1..]);
    }
    (lead, s)
}

/// Detects phone numbers; see the module documentation.
pub fn detect_phones(text: &str, out: &mut Vec<PiiEntity>) {
    for m in DIGIT_RUN.find_iter(text) {
        let before = text[..m.start()].chars().next_back();
        let after = text[m.end()..].chars().next();
        if before.is_some_and(|c| c.is_alphanumeric() || c == '_')
            || after.is_some_and(|c| c.is_alphanumeric() || c == '_')
        {
            continue; // glued to a word or a longer number (TR33 0006..., ABC123)
        }
        for (offset, raw) in pieces(m.as_str()) {
            let (lead, piece) = trim_piece(raw);
            let Some(plan) = plan(piece) else {
                continue;
            };
            let start = m.start() + offset + lead;
            out.push(PiiEntity {
                text: piece.to_string(),
                label: PiiType::Phone.as_str().to_string(),
                pii_type: PiiType::Phone,
                start,
                end: start + piece.len(),
                confidence: if plan == Plan::Turkish { 0.97 } else { 0.95 },
                stem: piece.to_string(),
                suffix: None,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn found(text: &str) -> Vec<String> {
        let mut out = Vec::new();
        detect_phones(text, &mut out);
        out.into_iter().map(|e| e.text).collect()
    }

    #[test]
    fn groupings_and_prefixes() {
        for p in [
            "+90 532 123 45 67",
            "+90(532)1234567",
            "+90-532-123-45-67",
            "0 (532) 123 45 67",
            "0532 123 4567",
            "05321234567",
            "5321234567",
            "90 532 123 45 67",
            "905321234567",
            "0090 532 123 45 67",
            "(+90) 532 123 45 67",
            "0212/555 12 34",
            "0532/123 45 67",
            "(212) 555-1234",
            "0800 211 00 88",
            "0850 222 0 222",
            "444 0 444",
            "444 1234",
            "0 5 3 2 1 2 3 4 5 6 7",
            "+1 (415) 555-2671",
            "+44 20 7946 0958",
            "+49 30 1234567",
            "+33 1 23 45 67 89",
            "+971 50 123 4567",
            "+86 10 1234 5678",
            "+31 6 12345678",
            "+994 50 123 45 67",
            "+7 495 123-45-67",
        ] {
            let text = format!("Beni {p} numarasından arayın.");
            assert!(
                found(&text).iter().any(|f| f == p),
                "{p}: {:?}",
                found(&text)
            );
        }
        assert_eq!(found("https://wa.me/905321234567"), ["905321234567"]);
        assert_eq!(found("Ara: 0532 123 45 67'yi"), ["0532 123 45 67"]);
        assert_eq!(
            found("0532 123 45 67 / 0533 987 65 43"),
            ["0532 123 45 67", "0533 987 65 43"]
        );
        assert_eq!(found("(0532 123 45 67)"), ["0532 123 45 67"]);
    }

    #[test]
    fn numbers_that_are_not_phones() {
        for text in [
            "Toplantı 2024 12 31 tarihinde.",
            "Tutar 1.234.567 TL oldu.",
            "ISBN 978-605-123-45-6",
            "1990-2020 yılları arasında",
            "Sunucu 192.168.1.10 adresinde.",
            "TCKN 10000000146",
            "Nüfus 85 372 377 kişi.",
            "Koordinat 41.0082, 28.9784",
            "Sayfa 123-145 arası okunacak.",
            "IBAN TR33 0006 1005 1978 6457 8413 26",
            "Kart 4111 1111 1111 1111",
            "Oran +5 puan arttı.",
            "Başvuru Numarası: 2022/100238",
            "müşteki hakkında 2017/124968 sayılı soruşturma",
            "26. 22/11/2001 tarihli ve 4721 sayılı Kanun",
            "26. 22.11.2001 tarihli karar",
            "No: 2023/102251, 25/2/2025, § 94",
        ] {
            assert!(found(text).is_empty(), "{text}: {:?}", found(text));
        }
    }
}
