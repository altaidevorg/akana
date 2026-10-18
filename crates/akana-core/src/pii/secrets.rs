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
            g[iİıI]zl[iİıI]\s*anahtar(?:[ıiIİ]m)?
            | ap[iİıI]\s*(?:anahtar[ıiIİ]|key)
            | secret\s*key
            | private\s*key
            | [öÖ]zel\s*anahtar(?:[ıiIİ]m)?
            | access\s*token
            | bearer\s*token
        )
        \b
        [:\s=]+
        (\S{4,128})"
    ).unwrap();

    /// OpenAI API Keys: sk-..., sk-proj-..., sk-svcacct-...
    pub static ref OPENAI_KEY_REGEX: Regex = Regex::new(
        r"\b(sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{32,128})\b"
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

    /// Stripe API Keys (sk_live_..., sk_test_..., rk_live_..., rk_test_..., pk_live_..., pk_test_...)
    pub static ref STRIPE_KEY_REGEX: Regex = Regex::new(
        r"\b((?:sk|rk|pk)_(?:live|test)_[0-9A-Za-z]{24,99})\b"
    ).unwrap();

    /// DigitalOcean Personal Access Tokens (dop_v1_ + 64 lowercase hex digits)
    pub static ref DIGITALOCEAN_TOKEN_REGEX: Regex = Regex::new(
        r"\b(dop_v1_[0-9a-f]{64})\b"
    ).unwrap();

    /// Shopify Tokens (shpat_, shpca_, shppa_, shpss_, shcap_ + >=32 lowercase hex digits)
    pub static ref SHOPIFY_TOKEN_REGEX: Regex = Regex::new(
        r"\b(sh(?:pat|pca|ppa|pss|cap)_[0-9a-f]{32,})\b"
    ).unwrap();

    /// Square Tokens (sq0atp- and sq0csp- + token >=22 characters)
    pub static ref SQUARE_TOKEN_REGEX: Regex = Regex::new(
        r"\b(sq0(?:atp|csp)-[0-9A-Za-z_-]{22,})\b"
    ).unwrap();

    /// Telegram Bot Tokens (8 to 10 digits:AA + 33 token characters)
    pub static ref TELEGRAM_BOT_TOKEN_REGEX: Regex = Regex::new(
        r"\b(\d{8,10}:AA[0-9A-Za-z_-]{33})\b"
    ).unwrap();

    /// Azure Storage Connection String AccountKey value
    pub static ref AZURE_STORAGE_KEY_REGEX: Regex = Regex::new(
        r#"(?i)\bAccountKey\s*=\s*([^;\s"'\r\n]{16,128})"#
    ).unwrap();

    /// Assignment values whose key name ends in a secret keyword: password=, DB_PASSWORD=,
    /// SECRET_KEY=, aws_secret_access_key =, "password": "...", x-api-key:. The keyword has to
    /// end the key, so max_tokens: 1000 and tokenizer: bert are not secrets.
    pub static ref ASSIGNMENT_SECRET_REGEX: Regex = Regex::new(
        r#"(?i)\b[A-Za-z0-9_.-]*?(?:password|passwd|pwd|passphrase|secret|token|credentials?|api[_-]?key|(?:access|secret|private|signing|encryption|client|master)[_-]?key|client[_-]?secret)["']?\s*[:=]\s*(?:"([^"\r\n]{2,256})"|'([^'\r\n]{2,256})'|([^\s,;\r\n"'>}]+))"#
    ).unwrap();

    /// Long structured tokens the entropy scanner (64 characters at most) cannot see, and a few
    /// prefixed ones: PyPI, Atlassian, age, Postman, Google OAuth access tokens, Slack app
    /// tokens, and hex private keys (0x + 64 hex; addresses are 0x + 40).
    pub static ref LONG_PREFIXED_TOKEN_REGEX: Regex = Regex::new(
        r"\b(pypi-AgEIcHlwaS5vcmc[0-9A-Za-z_-]{50,}|ATATT3[0-9A-Za-z_\-=]{150,}|AGE-SECRET-KEY-1[0-9A-Z]{58}|PMAK-[0-9a-f]{24}-[0-9a-f]{34}|ya29\.[0-9A-Za-z_-]{20,}|xapp-[0-9]-[0-9A-Za-z-]{20,}|0x[0-9a-fA-F]{64})\b"
    ).unwrap();

    /// Wallet recovery phrases after an explicit trigger: 12-24 lower-case words.
    pub static ref SEED_PHRASE_REGEX: Regex = Regex::new(
        r"(?i)\b(?:seed\s*phrase|mnemonic(?:\s*phrase)?|(?:secret\s*)?recovery\s*phrase|kurtarma\s*(?:ifadesi|ifadem|kelimeleri(?:m)?|cümlesi)|gizli\s*kelimeler(?:im|i)?|cüzdan\s*kelimeleri(?:m)?)\b[^\n\w]{0,10}((?:[a-z]{3,8}[ \t]+){11,23}[a-z]{3,8})\b"
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

    /// PEM private key blocks of any kind (RSA, EC, OPENSSH, ENCRYPTED, PGP ... BLOCK), also with
    /// JSON-escaped newlines, and up to the end of the key material when the END line is missing.
    pub static ref PRIVATE_KEY_BLOCK_REGEX: Regex = Regex::new(
        r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----(?:[\s\S]*?-----END (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----|(?:[A-Za-z0-9+/=:,\-\s]|\\[nr])+)"
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

/// Detects all secrets, passwords, OTPs, API keys, and high-entropy credentials.
pub fn detect_secrets(text: &str, out: &mut Vec<PiiEntity>) {
    // Both lower-cased forms for every trigger gate below (see `Lowered`): with Rust's alone
    // "GİZLİ KELİMELER" never contained "kelime", with the Turkish alone "MNEMONIC" never
    // contained "mnemonic".
    let lowered = super::normalize::Lowered::of(text);

    // 0. Passwords, PINs and one-time codes (sentences, commands, configs, URLs, headers)
    super::passwords::detect_passwords_in(text, &lowered, out);

    // 1. Contextual key material: gizli/özel anahtar, API key, secret/private key, tokens
    const CONTEXTUAL_TRIGGERS: &[&str] = &["anahtar", "token", "key"];
    let has_context_trigger = CONTEXTUAL_TRIGGERS.iter().any(|&trig| lowered.has(trig));
    if has_context_trigger {
        for cap in CONTEXTUAL_SECRET_REGEX.captures_iter(text) {
            if let Some(val) = cap.get(1) {
                // Sentence punctuation and quotes around the value are not part of it; inside
                // it they are (Kedi.Mavi.99).
                let (lead, val_str) = trim_value(val.as_str());
                // The value ends where the next field begins (api key: X,user=ali; X&lang=tr):
                // the capture takes any non-space run so that dots inside it stay (Kedi.Mavi.99).
                let val_str = trim_value(cut_at_next_field(val_str)).1;
                let separator = &text[cap.get(0).unwrap().start()..val.start()];
                let strong = separator.contains([':', '=']);
                if val_str.len() >= 4
                    && (strong || super::passwords::looks_like_password(val_str, false))
                {
                    out.push(PiiEntity {
                        text: val_str.to_string(),
                        label: "SIFRE".to_string(),
                        pii_type: PiiType::Credentials,
                        start: val.start() + lead,
                        end: val.start() + lead + val_str.len(),
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

    // 3. GitHub Tokens (SIMD anchors: the token prefixes; "gh" alone is in high, eight, ...)
    if ["ghp_", "gho_", "ghu_", "ghs_", "ghr_"]
        .iter()
        .any(|p| text.sz_find(p).is_some())
    {
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

    // 15. Stripe API Keys (SIMD anchor: sk_live_, sk_test_, rk_live_, rk_test_, pk_live_, pk_test_)
    if text.sz_find("sk_live_").is_some()
        || text.sz_find("sk_test_").is_some()
        || text.sz_find("rk_live_").is_some()
        || text.sz_find("rk_test_").is_some()
        || text.sz_find("pk_live_").is_some()
        || text.sz_find("pk_test_").is_some()
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

    // 19. DigitalOcean Personal Access Tokens (SIMD anchor: dop_v1_)
    if text.sz_find("dop_v1_").is_some() {
        for cap in DIGITALOCEAN_TOKEN_REGEX.captures_iter(text) {
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

    // 20. Shopify Tokens (SIMD anchor: shpat_, shpca_, shppa_, shpss_, shcap_)
    if text.sz_find("shpat_").is_some()
        || text.sz_find("shpca_").is_some()
        || text.sz_find("shppa_").is_some()
        || text.sz_find("shpss_").is_some()
        || text.sz_find("shcap_").is_some()
    {
        for cap in SHOPIFY_TOKEN_REGEX.captures_iter(text) {
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

    // 21. Square Tokens (SIMD anchor: sq0atp-, sq0csp-)
    if text.sz_find("sq0atp-").is_some() || text.sz_find("sq0csp-").is_some() {
        for cap in SQUARE_TOKEN_REGEX.captures_iter(text) {
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

    // 22. Telegram Bot Tokens (SIMD anchor: :AA)
    if text.sz_find(":AA").is_some() {
        for cap in TELEGRAM_BOT_TOKEN_REGEX.captures_iter(text) {
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

    // 23. Azure Storage Connection String AccountKey value (AccountName is not a secret)
    if text.sz_find("AccountKey").is_some() || lowered.has("accountkey") {
        for cap in AZURE_STORAGE_KEY_REGEX.captures_iter(text) {
            if let Some(val) = cap.get(1) {
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
    }

    // 24. Assignment values whose key ends in a secret keyword
    if ["pass", "pwd", "secret", "token", "credential", "key"]
        .iter()
        .any(|k| lowered.has(k))
    {
        for cap in ASSIGNMENT_SECRET_REGEX.captures_iter(text) {
            let matched = cap.get(1).or_else(|| cap.get(2)).or_else(|| cap.get(3));
            if let Some(val) = matched {
                let val_str = val.as_str().trim();
                // Short numbers are settings (timeout = 30, max_token = 512), not secrets.
                let short_number = val_str.len() < 6 && val_str.chars().all(|c| c.is_ascii_digit());
                if val_str.len() >= 2
                    && !short_number
                    && !val_str.eq_ignore_ascii_case("null")
                    && !val_str.eq_ignore_ascii_case("none")
                    && !val_str.eq_ignore_ascii_case("true")
                    && !val_str.eq_ignore_ascii_case("false")
                {
                    out.push(PiiEntity {
                        text: val_str.to_string(),
                        label: "SIFRE".to_string(),
                        pii_type: PiiType::Credentials,
                        start: val.start(),
                        end: val.end(),
                        confidence: 1.0,
                        stem: val_str.to_string(),
                        suffix: None,
                    });
                }
            }
        }
    }

    // 25. Long structured tokens and hex private keys
    for cap in LONG_PREFIXED_TOKEN_REGEX.captures_iter(text) {
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

    // 26. Wallet recovery phrases
    if has_context_trigger
        || ["seed", "mnemonic", "recovery", "kurtarma", "kelime"]
            .iter()
            .any(|k| lowered.has(k))
    {
        for cap in SEED_PHRASE_REGEX.captures_iter(text) {
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

    // 27. Wallet recovery phrases without a trigger word
    detect_bare_seed_phrases(text, out);

    // 10. Free-Floating High-Entropy Secret Scanner
    let fields = tokenize_words_with_offsets(text)
        .into_iter()
        .flat_map(|(start, token)| {
            value_fields(token)
                .into_iter()
                .map(move |(offset, field)| (start + offset, field))
        });
    for (token_start, token) in fields {
        // Quotes, brackets and sentence punctuation around a token are not part of it (JSON
        // "key": "value", markdown `value`); the span starts after what was trimmed.
        let (lead, clean_token) = trim_value(token);
        let start_idx = token_start + lead;
        if clean_token.len() < 8 || clean_token.len() > 64 {
            continue;
        }
        if is_suppressed_entropy_token(clean_token) {
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

/// Whether `key` reads as the name of a setting or a field (api_key, DB_PASS, user, x-token).
fn is_field_key(key: &str) -> bool {
    !key.is_empty()
        && key.len() <= 32
        && key.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_')
        && key
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'))
}

/// The byte offsets of the separators (`,` `;` `&` `|`) in `value` that start another field:
/// followed by a field name and `=` or `:` (X,user=ali; X;lang:tr). Separators inside a value
/// (Ab;cd!) are not among them.
fn field_starts(value: &str) -> Vec<usize> {
    let mut out = Vec::new();
    for (i, c) in value.char_indices() {
        if !matches!(c, ',' | ';' | '&' | '|') {
            continue;
        }
        let rest = value[i + 1..].trim_start();
        let key_len = rest
            .find(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.')))
            .unwrap_or(rest.len());
        let after = rest[key_len..].trim_start();
        if key_len > 0 && is_field_key(&rest[..key_len]) && after.starts_with(['=', ':']) {
            out.push(i);
        }
    }
    out
}

/// A field's value without its `key=` or `key:` (api_key=X is X), with its offset; a base64
/// value with its padding (c2VjcmV0==) or a field without a key is itself.
fn field_value(field: &str) -> (usize, &str) {
    let split = field.find(['=', ':']).and_then(|i| {
        let (key, value) = (&field[..i], &field[i + 1..]);
        (is_field_key(key) && !value.trim_start_matches('=').is_empty()).then_some((i + 1, value))
    });
    split.unwrap_or((0, field))
}

/// The values of a token made of fields (api_key=X;user=ali, DB_PASS=X;HOST=db,
/// sk9Xq2Lm7Pz,user=ali), each with its offset and without its key; any other token whole.
/// Scored whole, such a token was masked with its keys and the fields after the secret.
fn value_fields(token: &str) -> Vec<(usize, &str)> {
    let starts = field_starts(token);
    if starts.is_empty() {
        let (offset, value) = field_value(token);
        return vec![(offset, value)];
    }
    let mut out = Vec::new();
    let mut begin = 0;
    for end in starts.into_iter().chain(std::iter::once(token.len())) {
        let (offset, value) = field_value(&token[begin..end]);
        out.push((begin + offset, value));
        begin = (end + 1).min(token.len());
    }
    out
}

/// `value` up to where another field starts (see `field_starts`).
pub(crate) fn cut_at_next_field(value: &str) -> &str {
    match field_starts(value).first() {
        Some(&i) => &value[..i],
        None => value,
    }
}

/// A wallet recovery phrase with no word before it that names it: 12, 15, 18, 21 or 24
/// lower-case ASCII words of three to eight letters (the shape of BIP-39 words), one after the
/// other or numbered one per line (1. abandon 2. ability), with none of the function words
/// that every English or Turkish sentence of that length has. Two-letter words (of, to, is, ve,
/// da) end a run by their length.
fn detect_bare_seed_phrases(text: &str, out: &mut Vec<PiiEntity>) {
    const FUNCTION_WORDS: &[&str] = &[
        "the", "and", "for", "that", "this", "with", "from", "have", "was", "were", "are", "not",
        "but", "his", "her", "its", "they", "she", "their", "which", "would", "could", "should",
        "there", "been", "has", "had", "you", "your", "them", "than", "then", "these", "those",
        "what", "when", "where", "who", "why", "how", "into", "our", "bir", "ile", "icin", "gibi",
        "daha", "cok", "olan", "olarak", "kadar", "sonra", "ama", "veya", "ise", "her", "hem",
        "ben", "sen", "biz", "siz", "onlar", "bunu", "buna", "sey", "var", "yok", "degil",
    ];
    let is_word = |t: &str| {
        let t = t.trim_end_matches(['.', ',']);
        (3..=8).contains(&t.len())
            && t.bytes().all(|b| b.is_ascii_lowercase())
            && !FUNCTION_WORDS.contains(&t)
    };
    // "1." / "12)" before a word of a numbered list
    let is_number = |t: &str| {
        let digits = t.trim_end_matches(['.', ')', ':']);
        !digits.is_empty()
            && digits.len() <= 2
            && t.len() > digits.len()
            && digits.bytes().all(|b| b.is_ascii_digit())
    };
    let tokens = tokenize_words_with_offsets(text);
    let mut run: Vec<(usize, &str)> = Vec::new();
    let flush = |run: &mut Vec<(usize, &str)>, out: &mut Vec<PiiEntity>| {
        // at least half the words of a phrase are distinct; chants repeat theirs
        let distinct = run
            .iter()
            .map(|(_, w)| *w)
            .collect::<std::collections::HashSet<_>>()
            .len();
        if matches!(run.len(), 12 | 15 | 18 | 21 | 24) && distinct * 2 >= run.len() {
            let (start, _) = run[0];
            let (last_start, last) = run[run.len() - 1];
            let end = last_start + last.len();
            out.push(PiiEntity {
                text: text[start..end].to_string(),
                label: "SIFRE".to_string(),
                pii_type: PiiType::Credentials,
                start,
                end,
                confidence: 0.9,
                stem: text[start..end].to_string(),
                suffix: None,
            });
        }
        run.clear();
    };
    for (raw_start, raw) in tokens {
        // a label before a value (kullanma:, mnemonic=) is not part of the phrase after it
        if raw.ends_with([':', '=']) {
            flush(&mut run, out);
            continue;
        }
        // quotes and brackets around a word are not part of it ('abandon ... accident'})
        let (lead, token) = trim_value(raw);
        let start = raw_start + lead;
        if is_word(token) {
            run.push((start, token));
            // a full stop, comma or semicolon after a word ends the run
            let bare = raw.trim_end_matches(['"', '\'', ')', ']', '}', '`']);
            if bare.ends_with(['.', ',', ';']) {
                flush(&mut run, out);
            }
        } else if is_number(raw) {
            continue; // the numbers of a list, one word per line
        } else {
            flush(&mut run, out);
        }
    }
    flush(&mut run, out);
}

/// Filters out tokens that should not be classified by the free-floating entropy scanner.
///
/// NOTE: Structured secrets (API keys, JWT, Webhooks, and Database connection URLs) are extracted
/// by dedicated detectors (e.g. `DATABASE_URL_REGEX`) prior to this step and are NOT affected
/// by this suppression filter.
fn is_suppressed_entropy_token(token: &str) -> bool {
    if token.starts_with("http://") || token.starts_with("https://") {
        return true;
    }
    if token.sz_find("://").is_some() {
        return true; // Protocol URIs (handled by dedicated database/webhook extractors)
    }
    if token.sz_find("@").is_some() || token.sz_find("\\").is_some() {
        return true;
    }
    // Paths (src/main.rs, /usr/bin) are not secrets, but base64 key material such as an AWS
    // secret access key contains '/' too: only a pure base64 run of 30+ characters with upper,
    // lower and digits is kept.
    if token.sz_find("/").is_some() && !is_base64_run(token) {
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
    // Suppress IBAN candidates (alphanumeric 15..34 starting with 2 letters and 2 digits)
    let b = token.as_bytes();
    if b.len() >= 15
        && b.len() <= 34
        && b[0].is_ascii_alphabetic()
        && b[1].is_ascii_alphabetic()
        && b[2].is_ascii_digit()
        && b[3].is_ascii_digit()
    {
        return true;
    }
    // Suppress structured secrets handled by dedicated extractors
    if token.starts_with("sk-")
        || token.starts_with("gh")
        || token.starts_with("glpat-")
        || token.starts_with("dop_v1_")
        || token.starts_with("shpat_")
        || token.starts_with("shpca_")
        || token.starts_with("shppa_")
        || token.starts_with("shpss_")
        || token.starts_with("shcap_")
        || token.starts_with("sq0atp-")
        || token.starts_with("sq0csp-")
        || token.starts_with("hf_")
        || token.starts_with("npm_")
        || token.starts_with("AIza")
        || token.starts_with("eyJ")
        || token.starts_with("SG.")
        || token.starts_with("AKIA")
        || token.starts_with("ASIA")
        || token.starts_with("xox")
    {
        return true;
    }
    if token.chars().all(|c| c.is_ascii_digit()) {
        return true; // Pure numbers handled by other modules (phone, card, account)
    }
    if token.chars().all(|c| c.is_ascii_lowercase())
        || token.chars().all(|c| c.is_ascii_uppercase())
    {
        return true; // Ordinary words
    }
    // Words in any script and case shape, also hyphenated, with an apostrophe suffix or in
    // brackets/quotes (telekomünikasyon, Başsavcılık, Coudenhove-Kalergi, Ergenekon'un): the
    // ASCII-only check above let every long Turkish word through as a password.
    if is_word_shaped(token) {
        return true;
    }
    if is_glued_reference(token) {
        return true;
    }
    // a setting with a number (PORT=6379, TIMEOUT=30, retries:5): secret settings are found by
    // name (ASSIGNMENT_SECRET_REGEX), and a number alone is not key material
    if let Some((key, value)) = token.split_once(['=', ':']) {
        let key_ok = key
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
            && key
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-');
        if key_ok && !value.is_empty() && value.chars().all(|c| c.is_ascii_digit()) {
            return true;
        }
    }
    false
}

/// Prose with a number glued to it, as text copied from web pages and PDFs has: footnote and
/// list numbers (Kupası16, Anayasa'nın46, Basketball-Reference.com21, 1Hillsborough), and a
/// word before an amount or a time (gerekçesiyle20.000, göre16.40, 5.000,00TL'dir). A word
/// with a plain number after it (kedi2020, Galatasaray1905) is a common password and stays a
/// candidate unless the word is an inflected form (kupası, gerekçesiyle), carries an apostrophe
/// suffix or is a domain.
fn is_glued_reference(token: &str) -> bool {
    let token = token.trim_end_matches(['.', ',', ';', ':', '…']);
    let split = |s: &str| -> Option<(usize, usize)> {
        // (end of the leading letters, start of the trailing number)
        let letters_end = s.find(|c: char| c.is_ascii_digit())?;
        Some((letters_end, letters_end))
    };
    // a number then a word: 1Hillsborough, 5.000,00TL'dir, 20.000TL
    if token.starts_with(|c: char| c.is_ascii_digit()) {
        let num_end = token
            .find(|c: char| !(c.is_ascii_digit() || c == '.' || c == ','))
            .unwrap_or(token.len());
        let (num, word) = token.split_at(num_end);
        let separated = num.contains(['.', ',']);
        let word_ok = !word.is_empty() && is_word_shaped(word);
        let list_number = num.len() <= 2 && word.chars().count() >= 5;
        return word_ok && (separated || list_number);
    }
    let Some((letters_end, _)) = split(token) else {
        return false;
    };
    let (word, num) = token.split_at(letters_end);
    if word.is_empty()
        || !is_word_shaped(word)
        || !num
            .chars()
            .all(|c| c.is_ascii_digit() || c == '.' || c == ',')
        || !num.starts_with(|c: char| c.is_ascii_digit())
    {
        return false;
    }
    if num.contains(['.', ',']) {
        return true; // an amount or a time after a word: gerekçesiyle20.000, göre16.40
    }
    if num.len() > 3 {
        return false;
    }
    let lower = crate::phonology::to_turkish_lower(word);
    lower.contains(['\'', '’'])
        || [".com", ".org", ".net", ".tr", ".edu", ".gov"]
            .iter()
            .any(|tld| lower.ends_with(tld))
        || super::names::is_inflected_word(&lower)
}

/// Every part between hyphens, dashes, apostrophes, dots, commas, quotes and brackets is a
/// word (all lower case, capitalised, or all upper case letters) or a number (1980'lerde,
/// %50'sinden, 20.000). Random mixed-case or letter-digit strings are not.
pub(crate) fn is_word_shaped(token: &str) -> bool {
    if is_code_shaped(token) {
        return false;
    }
    const SEPARATORS: &[char] = &[
        '-', '–', '—', '\'', '’', '‘', '.', ',', '"', '“', '”', '«', '»', '(', ')', '[', ']', '…',
        '!', '?', ':', ';',
    ];
    // a number of eight digits or more is an identifier with a suffix (4799585158750652'nin),
    // not prose like 1980'lerde
    if token
        .split(SEPARATORS)
        .any(|p| p.len() >= 8 && p.bytes().all(|b| b.is_ascii_digit()))
    {
        return false;
    }
    token.split(SEPARATORS).all(|part| {
        let mut chars = part.chars();
        let Some(first) = chars.next() else {
            return true;
        };
        let rest = chars.as_str();
        let word = first.is_alphabetic()
            && rest.chars().all(char::is_alphabetic)
            && (rest.chars().all(char::is_lowercase)
                || (first.is_uppercase() && rest.chars().all(char::is_uppercase)));
        let number = part.chars().all(|c| c.is_ascii_digit() || c == '%');
        word || number
    })
}

/// A reference, account or access code, or a password of that shape: letter parts and number
/// parts joined by hyphens or underscores, with three digits or more in a number part or two
/// number parts (FAT-2025-26486, REF-9962-QK, DEMO-TOKEN-153, Temp-4577!, ACC-TR-98026-181).
/// Words with a short number stay words (COVID-19, 1980'lerde).
fn is_code_shaped(token: &str) -> bool {
    let token = token.trim_end_matches(['!', '?', '.', ',', '*', '#']);
    if !token.contains(['-', '_']) || token.contains(['\'', '’']) {
        return false;
    }
    let parts: Vec<&str> = token.split(['-', '_']).filter(|p| !p.is_empty()).collect();
    let letters = parts
        .iter()
        .any(|p| p.len() >= 2 && p.chars().all(|c| c.is_ascii_alphabetic()));
    let numbers: Vec<&&str> = parts
        .iter()
        .filter(|p| p.chars().all(|c| c.is_ascii_digit()))
        .collect();
    letters && (numbers.len() >= 2 || numbers.iter().any(|p| p.len() >= 3))
}

/// Strips surrounding sentence punctuation, quotes and brackets from a candidate value;
/// returns the number of leading bytes removed and the value.
pub(crate) fn trim_value(raw: &str) -> (usize, &str) {
    const EDGE: &[char] = &[
        '.', ',', ';', ':', '?', '"', '\'', '`', '(', ')', '[', ']', '{', '}', '<', '>', '“', '”',
        '‘', '’', '«', '»',
    ];
    let start_trimmed = raw.trim_start_matches(EDGE);
    let lead = raw.len() - start_trimmed.len();
    (lead, start_trimmed.trim_end_matches(EDGE))
}

/// A base64 run that is not a path: 30+ characters of the base64 alphabet, not starting with
/// '/', with upper case, lower case and digits.
fn is_base64_run(token: &str) -> bool {
    token.len() >= 30
        && !token.starts_with('/')
        && token
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'+' | b'/' | b'='))
        && token.bytes().any(|b| b.is_ascii_uppercase())
        && token.bytes().any(|b| b.is_ascii_lowercase())
        && token.bytes().any(|b| b.is_ascii_digit())
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

    #[test]
    fn test_detect_v052_credentials() {
        let dop = [
            "dop_v1_",
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        ]
        .concat();
        let shp = ["shpat_", "0123456789abcdef0123456789abcdef"].concat();
        let sq0 = ["sq0atp-", "0123456789ABCDEFGHIJKLMNOPQRSTUV"].concat();
        let tg = ["123456789", ":AA", "abcdefghijklmnopqrstuvwxyz0123456"].concat();
        let azure_key = ["bXlfc3VwZXJfc2VjcmV0X2F6dXJlX2tleV9mb3JfdGVzdGluZw", "=="].concat();
        let azure_cs = format!("DefaultEndpointsProtocol=https;AccountName=myacc;AccountKey={azure_key};EndpointSuffix=core.windows.net");
        let openai_svc = ["sk-svcacct-", "0123456789abcdef0123456789abcdef0123456789"].concat();
        let stripe_pk_live = ["pk_live_", "51ABCDEF0123456789abcdefghijklmnop"].concat();
        let stripe_pk_test = ["pk_test_", "51ABCDEF0123456789abcdefghijklmnop"].concat();
        let sk_ant = [
            "sk-ant-",
            "api03-",
            "abcdefghijklmnopqrstuvwxyz0123456789_ABCD",
        ]
        .concat();

        let text = format!(
            "DigitalOcean: {dop}\n\
             Shopify: {shp}\n\
             Square: {sq0}\n\
             Telegram: {tg}\n\
             Azure: {azure_cs}\n\
             OpenAI Svc: {openai_svc}\n\
             Stripe PK Live: {stripe_pk_live}\n\
             Stripe PK Test: {stripe_pk_test}\n\
             Anthropic: {sk_ant}\n\
             password = \"SecretPass123!\"\n\
             pwd: 'pass456'\n\
             secret = super_secret_token\n\
             api_key = \"my_api_key_val\"\n\
             api-key: my_api_key_val2"
        );

        let mut out = Vec::new();
        detect_secrets(&text, &mut out);

        let detected: Vec<&str> = out.iter().map(|e| e.text.as_str()).collect();
        assert!(detected.contains(&dop.as_str()));
        assert!(detected.contains(&shp.as_str()));
        assert!(detected.contains(&sq0.as_str()));
        assert!(detected.contains(&tg.as_str()));
        assert!(detected.contains(&azure_key.as_str()));
        // Verify AccountName is NOT in detected secrets
        assert!(!detected.contains(&"myacc"));
        assert!(detected.contains(&openai_svc.as_str()));
        assert!(detected.contains(&stripe_pk_live.as_str()));
        assert!(detected.contains(&stripe_pk_test.as_str()));
        // Anthropic key is one single credential span
        assert!(detected.contains(&sk_ant.as_str()));
        let ant_count = detected.iter().filter(|&&s| s == sk_ant.as_str()).count();
        assert_eq!(ant_count, 1);

        // Assignment values
        assert!(detected.contains(&"SecretPass123!"));
        assert!(detected.contains(&"pass456"));
        assert!(detected.contains(&"super_secret_token"));
        assert!(detected.contains(&"my_api_key_val"));
        assert!(detected.contains(&"my_api_key_val2"));

        for entity in &out {
            assert_eq!(entity.label, "SIFRE");
            assert_eq!(entity.pii_type, PiiType::Credentials);
        }
    }
}
