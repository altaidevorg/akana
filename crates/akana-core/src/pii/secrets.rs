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

    /// Anthropic API Keys: sk-ant-api03-..., sk-ant-admin01-...
    pub static ref ANTHROPIC_KEY_REGEX: Regex = Regex::new(
        r"\b(sk-ant-[A-Za-z0-9_-]{20,128})\b"
    ).unwrap();

    /// GitLab Personal Access Tokens (glpat-...)
    pub static ref GITLAB_TOKEN_REGEX: Regex = Regex::new(
        r"\b(glpat-[0-9A-Za-z_-]{20,40})\b"
    ).unwrap();

    /// GitHub Fine-Grained Personal Access Tokens (github_pat_...)
    pub static ref GITHUB_FINE_GRAINED_PAT_REGEX: Regex = Regex::new(
        r"\b(github_pat_[0-9A-Za-z_]{60,100})\b"
    ).unwrap();

    /// GitHub Personal Access Tokens & App tokens (ghp_, gho_, ghu_, ghs_, ghr_)
    pub static ref GITHUB_TOKEN_REGEX: Regex = Regex::new(
        r"\b(gh[pours]_[A-Za-z0-9]{36,40})\b"
    ).unwrap();

    /// Hugging Face User Access Tokens (hf_...)
    pub static ref HUGGINGFACE_TOKEN_REGEX: Regex = Regex::new(
        r"\b(hf_[0-9A-Za-z]{30,50})\b"
    ).unwrap();

    /// Groq API Keys (gsk_...)
    pub static ref GROQ_KEY_REGEX: Regex = Regex::new(
        r"\b(gsk_[0-9A-Za-z]{40,70})\b"
    ).unwrap();

    /// npm Access Tokens (npm_...)
    pub static ref NPM_TOKEN_REGEX: Regex = Regex::new(
        r"\b(npm_[0-9A-Za-z]{32,45})\b"
    ).unwrap();

    /// Stripe API Keys (sk_live_..., sk_test_..., rk_live_..., rk_test_...)
    pub static ref STRIPE_KEY_REGEX: Regex = Regex::new(
        r"\b((?:sk|rk)_(?:live|test)_[0-9A-Za-z]{24,99})\b"
    ).unwrap();

    /// SendGrid API Keys (SG.xxx.yyy)
    pub static ref SENDGRID_KEY_REGEX: Regex = Regex::new(
        r"\b(SG\.[0-9A-Za-z_-]{16,32}\.[0-9A-Za-z_-]{32,64})\b"
    ).unwrap();

    /// Slack and Discord Incoming Webhooks
    pub static ref WEBHOOK_URL_REGEX: Regex = Regex::new(
        r"(?i)\bhttps?://(?:hooks\.slack\.com/services/[0-9A-Za-z_-]+/[0-9A-Za-z_-]+/[0-9A-Za-z_-]+|(?:ptb\.|canary\.)?discord(?:app)?\.com/api/webhooks/\d+/[0-9A-Za-z_-]+)\b"
    ).unwrap();

    /// Sensitive Database Connection URLs (PostgreSQL, MySQL, MongoDB, Redis, AMQP, MSSQL)
    pub static ref DATABASE_URL_REGEX: Regex = Regex::new(
        r#"(?i)\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|rediss|amqps?|mssql|sqlserver)://[^\s<>'"`)]+"#
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
    if token.sz_find("://").is_some() {
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

    // 2. Anthropic API Keys (SIMD anchor: sk-ant)
    if text.sz_find("sk-ant").is_some() {
        for cap in ANTHROPIC_KEY_REGEX.captures_iter(text) {
            let val = cap.get(1).unwrap();
            out.push(PiiEntity {
                text: val.as_str().to_string(),
                label: "SIFRE".to_string(),
                pii_type: PiiType::Credentials,
                start: val.start(),
                end: val.end(),
                confidence: 1.0,
                stem: val.as_str().to_string(),
                suffix: None,
            });
        }
    }

    // 3. OpenAI API Keys (SIMD anchor: sk-)
    if text.sz_find("sk-").is_some() {
        for cap in OPENAI_KEY_REGEX.captures_iter(text) {
            let val = cap.get(1).unwrap();
            if val.as_str().starts_with("sk-ant-") {
                continue;
            }
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

    // 10. GitLab Personal Access Tokens (SIMD anchor: glpat-)
    if text.sz_find("glpat-").is_some() {
        for cap in GITLAB_TOKEN_REGEX.captures_iter(text) {
            let val = cap.get(1).unwrap();
            out.push(PiiEntity {
                text: val.as_str().to_string(),
                label: "SIFRE".to_string(),
                pii_type: PiiType::Credentials,
                start: val.start(),
                end: val.end(),
                confidence: 1.0,
                stem: val.as_str().to_string(),
                suffix: None,
            });
        }
    }

    // 11. GitHub Fine-Grained Personal Access Tokens (SIMD anchor: github_pat_)
    if text.sz_find("github_pat_").is_some() {
        for cap in GITHUB_FINE_GRAINED_PAT_REGEX.captures_iter(text) {
            let val = cap.get(1).unwrap();
            out.push(PiiEntity {
                text: val.as_str().to_string(),
                label: "SIFRE".to_string(),
                pii_type: PiiType::Credentials,
                start: val.start(),
                end: val.end(),
                confidence: 1.0,
                stem: val.as_str().to_string(),
                suffix: None,
            });
        }
    }

    // 12. Hugging Face Tokens (SIMD anchor: hf_)
    if text.sz_find("hf_").is_some() {
        for cap in HUGGINGFACE_TOKEN_REGEX.captures_iter(text) {
            let val = cap.get(1).unwrap();
            out.push(PiiEntity {
                text: val.as_str().to_string(),
                label: "SIFRE".to_string(),
                pii_type: PiiType::Credentials,
                start: val.start(),
                end: val.end(),
                confidence: 1.0,
                stem: val.as_str().to_string(),
                suffix: None,
            });
        }
    }

    // 13. Groq API Keys (SIMD anchor: gsk_)
    if text.sz_find("gsk_").is_some() {
        for cap in GROQ_KEY_REGEX.captures_iter(text) {
            let val = cap.get(1).unwrap();
            out.push(PiiEntity {
                text: val.as_str().to_string(),
                label: "SIFRE".to_string(),
                pii_type: PiiType::Credentials,
                start: val.start(),
                end: val.end(),
                confidence: 1.0,
                stem: val.as_str().to_string(),
                suffix: None,
            });
        }
    }

    // 14. npm Access Tokens (SIMD anchor: npm_)
    if text.sz_find("npm_").is_some() {
        for cap in NPM_TOKEN_REGEX.captures_iter(text) {
            let val = cap.get(1).unwrap();
            out.push(PiiEntity {
                text: val.as_str().to_string(),
                label: "SIFRE".to_string(),
                pii_type: PiiType::Credentials,
                start: val.start(),
                end: val.end(),
                confidence: 1.0,
                stem: val.as_str().to_string(),
                suffix: None,
            });
        }
    }

    // 15. Stripe API Keys (SIMD anchor: sk_live_, sk_test_, rk_live_, rk_test_)
    if text.sz_find("sk_live_").is_some()
        || text.sz_find("sk_test_").is_some()
        || text.sz_find("rk_live_").is_some()
        || text.sz_find("rk_test_").is_some()
    {
        for cap in STRIPE_KEY_REGEX.captures_iter(text) {
            let val = cap.get(1).unwrap();
            out.push(PiiEntity {
                text: val.as_str().to_string(),
                label: "SIFRE".to_string(),
                pii_type: PiiType::Credentials,
                start: val.start(),
                end: val.end(),
                confidence: 1.0,
                stem: val.as_str().to_string(),
                suffix: None,
            });
        }
    }

    // 16. SendGrid API Keys (SIMD anchor: SG.)
    if text.sz_find("SG.").is_some() {
        for cap in SENDGRID_KEY_REGEX.captures_iter(text) {
            let val = cap.get(1).unwrap();
            out.push(PiiEntity {
                text: val.as_str().to_string(),
                label: "SIFRE".to_string(),
                pii_type: PiiType::Credentials,
                start: val.start(),
                end: val.end(),
                confidence: 1.0,
                stem: val.as_str().to_string(),
                suffix: None,
            });
        }
    }

    // 17. Webhooks (Slack & Discord) (SIMD anchor: hooks.slack.com/services or /api/webhooks/)
    if text.sz_find("hooks.slack.com/services").is_some()
        || text.sz_find("/api/webhooks/").is_some()
    {
        for mat in WEBHOOK_URL_REGEX.find_iter(text) {
            let val_str = mat.as_str().trim_end_matches(|c: char| {
                c == '.'
                    || c == ','
                    || c == ';'
                    || c == ':'
                    || c == ')'
                    || c == ']'
                    || c == '}'
                    || c == '>'
                    || c == '"'
                    || c == '\''
            });
            out.push(PiiEntity {
                text: val_str.to_string(),
                label: "SIFRE".to_string(),
                pii_type: PiiType::Credentials,
                start: mat.start(),
                end: mat.start() + val_str.len(),
                confidence: 1.0,
                stem: val_str.to_string(),
                suffix: None,
            });
        }
    }

    // 18. Database Connection URLs (SIMD anchor: :// and protocol keywords)
    if text.sz_find("://").is_some()
        && (text.sz_find("postgres").is_some()
            || text.sz_find("mysql").is_some()
            || text.sz_find("mongodb").is_some()
            || text.sz_find("redis").is_some()
            || text.sz_find("amqp").is_some()
            || text.sz_find("mssql").is_some()
            || text.sz_find("sqlserver").is_some())
    {
        for mat in DATABASE_URL_REGEX.find_iter(text) {
            let val_str = mat.as_str().trim_end_matches(|c: char| {
                c == '.'
                    || c == ','
                    || c == ';'
                    || c == ':'
                    || c == ')'
                    || c == ']'
                    || c == '}'
                    || c == '>'
                    || c == '"'
                    || c == '\''
            });
            out.push(PiiEntity {
                text: val_str.to_string(),
                label: "SIFRE".to_string(),
                pii_type: PiiType::Credentials,
                start: mat.start(),
                end: mat.start() + val_str.len(),
                confidence: 1.0,
                stem: val_str.to_string(),
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
        let openai_key = ["sk-proj-", "98421094821098412094812098412094"].concat();
        let gh_token = ["ghp_", "ABCDEF0123456789abcdef0123456789abcd"].concat();
        let text = format!("Burada OpenAI anahtarı {openai_key} ve GitHub token {gh_token} var.");
        let mut out = Vec::new();
        detect_secrets(&text, &mut out);

        let detected_values: Vec<&str> = out.iter().map(|e| e.text.as_str()).collect();
        assert!(detected_values.contains(&openai_key.as_str()));
        assert!(detected_values.contains(&gh_token.as_str()));
    }

    #[test]
    fn test_detect_new_enterprise_secret_families() {
        let sk_ant = [
            "sk-ant-",
            "api03-",
            "abcdefghijklmnopqrstuvwxyz0123456789_ABCD",
        ]
        .concat();
        let glpat = ["glpat-", "abcdefghijklmnopqrst"].concat();
        let gh_pat = [
            "github_",
            "pat_11ABCDEFG0123456789abcdefghijklmnopqrstuvwxyz_0123456789abcdefghijklmnopq",
        ]
        .concat();
        let hf_tok = ["hf_", "abcdefghijklmnopqrstuvwxyz01234567"].concat();
        let gsk_key = ["gsk_", "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLM"].concat();
        let npm_tok = ["npm_", "abcdefghijklmnopqrstuvwxyz0123456789"].concat();
        let stripe_live = ["sk_live_", "51ABCDEF0123456789abcdefghijklmnop"].concat();
        let stripe_test = ["rk_test_", "51ABCDEF0123456789abcdefghijklmnop"].concat();
        let sendgrid = [
            "SG.",
            "abcdefghijklmnopqrstuv.",
            "0123456789abcdefghijklmnopqrstuvwxyz0123456789",
        ]
        .concat();
        let slack_hook = [
            "https://hooks.",
            "slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX",
        ]
        .concat();
        let discord_hook = [
            "https://discord.",
            "com/api/webhooks/123456789012345678/abcdefghijklmnopqrstuvwxyz_0123456789",
        ]
        .concat();
        let pg_url = [
            "postgresql://admin:super_secret@",
            "localhost:5432/my_database",
        ]
        .concat();
        let mongo_url = [
            "mongodb+srv://app_user:p%40ssword@",
            "cluster0.abcde.mongodb.net/prod_db",
        ]
        .concat();
        let redis_url = ["redis://:mypassword@", "cache.internal:6379/0"].concat();

        let text = format!(
            "Anthropic: {sk_ant}\n\
             GitLab: {glpat}\n\
             GitHub Fine-Grained: {gh_pat}\n\
             HuggingFace: {hf_tok}\n\
             Groq: {gsk_key}\n\
             npm: {npm_tok}\n\
             Stripe Live: {stripe_live}\n\
             Stripe Restricted: {stripe_test}\n\
             SendGrid: {sendgrid}\n\
             Slack Webhook: {slack_hook}\n\
             Discord Webhook: {discord_hook}\n\
             Postgres URL: {pg_url}\n\
             MongoDB URL: {mongo_url}\n\
             Redis URL: {redis_url}"
        );

        let mut out = Vec::new();
        detect_secrets(&text, &mut out);

        let detected: Vec<&str> = out.iter().map(|e| e.text.as_str()).collect();
        assert!(detected.iter().any(|s| s.starts_with("sk-ant-")));
        assert!(detected.iter().any(|s| s.starts_with("glpat-")));
        assert!(detected.iter().any(|s| s.starts_with("github_")));
        assert!(detected.iter().any(|s| s.starts_with("hf_")));
        assert!(detected.iter().any(|s| s.starts_with("gsk_")));
        assert!(detected.iter().any(|s| s.starts_with("npm_")));
        assert!(detected.iter().any(|s| s.starts_with("sk_live_")));
        assert!(detected.iter().any(|s| s.starts_with("rk_test_")));
        assert!(detected.iter().any(|s| s.starts_with("SG.")));
        assert!(detected
            .iter()
            .any(|s| s.contains("hooks.slack.com/services/")));
        assert!(detected
            .iter()
            .any(|s| s.contains("discord.com/api/webhooks/")));
        assert!(detected
            .iter()
            .any(|s| s.starts_with("postgresql://admin:super_secret@")));
        assert!(detected
            .iter()
            .any(|s| s.starts_with("mongodb+srv://app_user:")));
        assert!(detected
            .iter()
            .any(|s| s.starts_with("redis://:mypassword@")));

        for entity in &out {
            assert_eq!(entity.label, "SIFRE");
            assert_eq!(entity.pii_type, PiiType::Credentials);
        }
    }
}
