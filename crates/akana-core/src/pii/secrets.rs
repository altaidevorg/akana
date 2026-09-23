//! Advanced Turkish & Enterprise Secret, Password, and API Token Detection.
//!
//! Provides comprehensive detection for:
//! - Contextual passwords & OTPs: "şifrem Limon004_ ile", "parola Kedi358!", "sms doğrulama kodum 321165", "pin 3235"
//! - Structured API keys: OpenAI (`sk-...`), GitHub (`ghp_...`), AWS (`AKIA...`), Slack (`xoxb-...`), Google (`AIza...`)
//! - Cryptographic tokens: JWT (`eyJ...`), Bearer tokens, PEM Private Keys
//! - Shannon entropy-based detection of free-floating high-entropy secrets ($H > 3.4$ bits/char) with false-positive suppression.

use super::{PiiEntity, PiiType};
use lazy_static::lazy_static;
use regex::Regex;
use std::collections::HashMap;
use stringzilla::StringZilla;

lazy_static! {
    /// Contextual passwords, credentials, and OTP verification codes.
    pub static ref CONTEXTUAL_SECRET_REGEX: Regex = Regex::new(
        r"(?ix)
        \b(?:
            ş[iİıI]fre(?:m|s[iİıI]|n[iİıI]z)?
            | parola(?:m|s[ıiIİ])?
            | p[iİıI]n(?:[iİıI]m|s[iİıI]|_kodu|\s+kodu)?
            | kullan[ıiIİ]c[ıiIİ]\s*ş[iİıI]fres[iİıI]
            | (?:sms\s+)?do[ğgĞG]rulama\s+kodu(?:m|nu)?
            | onay\s+kodu(?:m|nu)?
            | tek\s+kullan[ıiIİ]ml[ıiIİ]k\s+ş[iİıI]fre(?:m)?
            | ge[çcÇC][iİıI]c[iİıI]\s+ş[iİıI]fre(?:m)?
            | mob[iİıI]l\s+onay\s+p[iİıI]n(?:[iİıI]m)?
            | otp(?:\s*kodu(?:m)?)?
            | 2fa(?:\s*kodu(?:m)?)?
            | g[iİıI]zl[iİıI]\s*anahtar(?:[ıiIİ]m)?
            | ap[iİıI]\s*(?:anahtar[ıiIİ]|key)
            | secret\s*key
            | access\s*token
            | bearer\s*token
        )
        \b
        [:\s=]+
        ([^\s,;:.]{4,64})"
    ).unwrap();

    /// OpenAI API Keys: sk-..., sk-proj-...
    pub static ref OPENAI_KEY_REGEX: Regex = Regex::new(
        r"\b(sk-(?:proj-)?[A-Za-z0-9_-]{32,128})\b"
    ).unwrap();

    /// GitHub Personal Access Tokens & App tokens (ghp_, gho_, ghu_, ghs_, ghr_)
    pub static ref GITHUB_TOKEN_REGEX: Regex = Regex::new(
        r"\b(gh[pours]_[A-Za-z0-9]{36,40})\b"
    ).unwrap();

    /// AWS Access Key ID (AKIA..., ASIA...)
    pub static ref AWS_KEY_REGEX: Regex = Regex::new(
        r"\b((?:AKIA|ASIA)[0-9A-Z]{16})\b"
    ).unwrap();

    /// Slack Bot & User Tokens (xoxb-, xoxp-, xoxa-, xoxr-, xoxs-)
    pub static ref SLACK_TOKEN_REGEX: Regex = Regex::new(
        r"\b(xox[baprs]-[0-9A-Za-z-]{10,72})\b"
    ).unwrap();

    /// Google Cloud API Keys (AIza...)
    pub static ref GOOGLE_API_KEY_REGEX: Regex = Regex::new(
        r"\b(AIza[0-9A-Za-z_-]{35})\b"
    ).unwrap();

    /// JWT (JSON Web Tokens): eyJ...eyJ...xxx
    pub static ref JWT_TOKEN_REGEX: Regex = Regex::new(
        r"\b(eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})\b"
    ).unwrap();

    /// Bearer Authorization Tokens
    pub static ref BEARER_TOKEN_REGEX: Regex = Regex::new(
        r"(?i)\bBearer\s+([A-Za-z0-9._~+/-]{20,}={0,2})\b"
    ).unwrap();

    /// PEM Private Key Block Header & Digest
    pub static ref PRIVATE_KEY_BLOCK_REGEX: Regex = Regex::new(
        r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----[\s\S]+?-----END (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"
    ).unwrap();
}

/// Calculates the base-2 Shannon entropy of a string (in bits per character).
pub fn shannon_entropy(s: &str) -> f64 {
    if s.is_empty() {
        return 0.0;
    }
    let mut counts: HashMap<char, f64> = HashMap::new();
    let mut total: f64 = 0.0;
    for c in s.chars() {
        *counts.entry(c).or_insert(0.0) += 1.0;
        total += 1.0;
    }
    let mut entropy = 0.0;
    for &count in counts.values() {
        let p = count / total;
        entropy -= p * p.log2();
    }
    entropy
}

/// Checks if a string contains mixed character classes (e.g. uppercase, lowercase, digits, symbols).
pub fn has_mixed_character_classes(s: &str) -> bool {
    let mut has_upper = false;
    let mut has_lower = false;
    let mut has_digit = false;
    let mut has_special = false;
    for c in s.chars() {
        if c.is_ascii_uppercase() {
            has_upper = true;
        } else if c.is_ascii_lowercase() {
            has_lower = true;
        } else if c.is_ascii_digit() {
            has_digit = true;
        } else if !c.is_alphanumeric() {
            has_special = true;
        }
    }
    let classes = (has_upper as u8) + (has_lower as u8) + (has_digit as u8) + (has_special as u8);
    classes >= 3
}

/// Filters out tokens that should not be classified by the entropy scanner.
fn is_suppressed_token(token: &str) -> bool {
    if token.starts_with("http://") || token.starts_with("https://") {
        return true;
    }
    if token.sz_find("@").is_some() || token.sz_find("/").is_some() || token.sz_find("\\").is_some()
    {
        return true;
    }
    if token.ends_with(".com")
        || token.ends_with(".net")
        || token.ends_with(".org")
        || token.ends_with(".tr")
    {
        return true;
    }
    if token.starts_with("0x") && token.len() == 42 {
        return true; // Ethereum address
    }
    if token.starts_with("bc1") {
        return true; // Bitcoin address
    }
    if token.chars().all(|c| c.is_ascii_digit()) {
        return true; // Pure numbers handled by other modules (phone, card, account)
    }
    if token.chars().all(|c| c.is_ascii_lowercase())
        || token.chars().all(|c| c.is_ascii_uppercase())
    {
        return true; // Ordinary words
    }
    false
}

/// Detects all secrets, passwords, OTPs, API keys, and high-entropy credentials.
pub fn detect_secrets(text: &str, out: &mut Vec<PiiEntity>) {
    // 1. Contextual Secrets & OTPs (Fast SIMD pre-check for triggers before running unicode regex)
    let lower = text.to_lowercase();
    const CONTEXTUAL_TRIGGERS: &[&str] = &[
        "şifre", "sifre", "parola", "pin", "kodu", "anahtar", "token", "key", "otp", "2fa",
    ];
    let has_context_trigger = CONTEXTUAL_TRIGGERS
        .iter()
        .any(|&trig| lower.sz_find(trig).is_some());
    if has_context_trigger {
        for cap in CONTEXTUAL_SECRET_REGEX.captures_iter(text) {
            if let Some(val) = cap.get(1) {
                let val_str = val.as_str().trim_matches(|c: char| {
                    c == '.' || c == ',' || c == ';' || c == ':' || c == '?'
                });
                if val_str.len() >= 4 {
                    out.push(PiiEntity {
                        text: val_str.to_string(),
                        label: "SIFRE".to_string(),
                        pii_type: PiiType::Credentials,
                        start: val.start(),
                        end: val.start() + val_str.len(),
                        confidence: 0.98,
                        stem: val_str.to_string(),
                        suffix: None,
                    });
                }
            }
        }
    }

    // 2. OpenAI API Keys (SIMD anchor: sk-)
    if text.sz_find("sk-").is_some() {
        for cap in OPENAI_KEY_REGEX.captures_iter(text) {
            let val = cap.get(1).unwrap();
            out.push(PiiEntity {
                text: val.as_str().to_string(),
                label: "SIFRE".to_string(),
                pii_type: PiiType::Credentials,
                start: val.start(),
                end: val.end(),
                confidence: 0.99,
                stem: val.as_str().to_string(),
                suffix: None,
            });
        }
    }

    // 3. GitHub Tokens (SIMD anchor: gh)
    if text.sz_find("gh").is_some() {
        for cap in GITHUB_TOKEN_REGEX.captures_iter(text) {
            let val = cap.get(1).unwrap();
            out.push(PiiEntity {
                text: val.as_str().to_string(),
                label: "SIFRE".to_string(),
                pii_type: PiiType::Credentials,
                start: val.start(),
                end: val.end(),
                confidence: 0.99,
                stem: val.as_str().to_string(),
                suffix: None,
            });
        }
    }

    // 4. AWS Keys (SIMD anchor: AKIA or ASIA)
    if text.sz_find("AKIA").is_some() || text.sz_find("ASIA").is_some() {
        for cap in AWS_KEY_REGEX.captures_iter(text) {
            let val = cap.get(1).unwrap();
            out.push(PiiEntity {
                text: val.as_str().to_string(),
                label: "SIFRE".to_string(),
                pii_type: PiiType::Credentials,
                start: val.start(),
                end: val.end(),
                confidence: 0.99,
                stem: val.as_str().to_string(),
                suffix: None,
            });
        }
    }

    // 5. Slack Tokens (SIMD anchor: xox)
    if text.sz_find("xox").is_some() {
        for cap in SLACK_TOKEN_REGEX.captures_iter(text) {
            let val = cap.get(1).unwrap();
            out.push(PiiEntity {
                text: val.as_str().to_string(),
                label: "SIFRE".to_string(),
                pii_type: PiiType::Credentials,
                start: val.start(),
                end: val.end(),
                confidence: 0.99,
                stem: val.as_str().to_string(),
                suffix: None,
            });
        }
    }

    // 6. Google API Keys (SIMD anchor: AIza)
    if text.sz_find("AIza").is_some() {
        for cap in GOOGLE_API_KEY_REGEX.captures_iter(text) {
            let val = cap.get(1).unwrap();
            out.push(PiiEntity {
                text: val.as_str().to_string(),
                label: "SIFRE".to_string(),
                pii_type: PiiType::Credentials,
                start: val.start(),
                end: val.end(),
                confidence: 0.99,
                stem: val.as_str().to_string(),
                suffix: None,
            });
        }
    }

    // 7. JWT Tokens (SIMD anchor: eyJ)
    if text.sz_find("eyJ").is_some() {
        for cap in JWT_TOKEN_REGEX.captures_iter(text) {
            let val = cap.get(1).unwrap();
            out.push(PiiEntity {
                text: val.as_str().to_string(),
                label: "SIFRE".to_string(),
                pii_type: PiiType::Credentials,
                start: val.start(),
                end: val.end(),
                confidence: 0.99,
                stem: val.as_str().to_string(),
                suffix: None,
            });
        }
    }

    // 8. Bearer Authorization Tokens (SIMD anchor: Bearer / bearer)
    if text.sz_find("Bearer ").is_some() || text.sz_find("bearer ").is_some() {
        for cap in BEARER_TOKEN_REGEX.captures_iter(text) {
            let val = cap.get(1).unwrap();
            out.push(PiiEntity {
                text: val.as_str().to_string(),
                label: "SIFRE".to_string(),
                pii_type: PiiType::Credentials,
                start: val.start(),
                end: val.end(),
                confidence: 0.99,
                stem: val.as_str().to_string(),
                suffix: None,
            });
        }
    }

    // 9. PEM Private Key Blocks (SIMD anchor: -----BEGIN)
    if text.sz_find("-----BEGIN").is_some() {
        for mat in PRIVATE_KEY_BLOCK_REGEX.find_iter(text) {
            let span_str = mat.as_str();
            out.push(PiiEntity {
                text: span_str.to_string(),
                label: "SIFRE".to_string(),
                pii_type: PiiType::Credentials,
                start: mat.start(),
                end: mat.end(),
                confidence: 1.0,
                stem: span_str.to_string(),
                suffix: None,
            });
        }
    }

    // 10. Free-Floating High-Entropy Secret Scanner
    for (start_idx, token) in tokenize_words_with_offsets(text) {
        let clean_token =
            token.trim_matches(|c: char| c == '.' || c == ',' || c == ';' || c == ':' || c == '?');
        if clean_token.len() < 8 || clean_token.len() > 64 {
            continue;
        }
        if is_suppressed_token(clean_token) {
            continue;
        }

        let entropy = shannon_entropy(clean_token);
        let mixed = has_mixed_character_classes(clean_token);

        let is_candidate = (clean_token.len() >= 12 && entropy >= 3.4)
            || (clean_token.len() >= 8 && mixed && entropy >= 2.85);

        if is_candidate {
            out.push(PiiEntity {
                text: clean_token.to_string(),
                label: "SIFRE".to_string(),
                pii_type: PiiType::Credentials,
                start: start_idx,
                end: start_idx + clean_token.len(),
                confidence: ((0.80 + (entropy / 5.0) * 0.19).min(0.99)) as f32,
                stem: clean_token.to_string(),
                suffix: None,
            });
        }
    }
}

/// Tokenizes text into (byte_offset, token) pairs by splitting on whitespace.
fn tokenize_words_with_offsets(text: &str) -> Vec<(usize, &str)> {
    let mut tokens = Vec::new();
    let mut in_token = false;
    let mut start = 0;

    for (idx, c) in text.char_indices() {
        if c.is_whitespace() {
            if in_token {
                tokens.push((start, &text[start..idx]));
                in_token = false;
            }
        } else if !in_token {
            start = idx;
            in_token = true;
        }
    }
    if in_token {
        tokens.push((start, &text[start..]));
    }
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shannon_entropy() {
        let e1 = shannon_entropy("Limon004_");
        assert!(e1 > 2.8, "Entropy of Limon004_ should be > 2.8, got {}", e1);

        let e2 = shannon_entropy("sk-proj-7839218947219847129847192847");
        assert!(e2 > 3.4, "Entropy of API key should be > 3.4, got {}", e2);
    }

    #[test]
    fn test_mixed_character_classes() {
        assert!(has_mixed_character_classes("Limon004_"));
        assert!(has_mixed_character_classes("Kedi7019!"));
        assert!(!has_mixed_character_classes("istanbul"));
        assert!(!has_mixed_character_classes("12345678"));
    }

    #[test]
    fn test_detect_secrets_contextual() {
        let text = "Şifrem Limon004_ ile giremiyorum, parola Kedi358! kayıtta duruyor ve mobil onay pinim 45606 oldu.";
        let mut out = Vec::new();
        detect_secrets(text, &mut out);

        let detected_values: Vec<&str> = out.iter().map(|e| e.text.as_str()).collect();
        assert!(
            detected_values.contains(&"Limon004_"),
            "Missing Limon004_: {:?}",
            detected_values
        );
        assert!(
            detected_values.contains(&"Kedi358!"),
            "Missing Kedi358!: {:?}",
            detected_values
        );
        assert!(
            detected_values.contains(&"45606"),
            "Missing 45606: {:?}",
            detected_values
        );
    }

    #[test]
    fn test_detect_api_keys() {
        let text = "Burada OpenAI anahtarı sk-proj-98421094821098412094812098412094 ve GitHub token ghp_ABCDEF0123456789abcdef0123456789abcd var.";
        let mut out = Vec::new();
        detect_secrets(text, &mut out);

        let detected_values: Vec<&str> = out.iter().map(|e| e.text.as_str()).collect();
        assert!(detected_values.contains(&"sk-proj-98421094821098412094812098412094"));
        assert!(detected_values.contains(&"ghp_ABCDEF0123456789abcdef0123456789abcd"));
    }
}
