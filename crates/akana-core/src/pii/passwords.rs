//! Passwords, PINs and one-time codes as an AI-agent gateway sees them: in Turkish and English
//! sentences (şifrem X, şifremi X olarak değiştirdim, şifre olarak X, X şifresiyle, PIN'im X,
//! my password is X), command lines (mysql -pX, sshpass -p X, docker login -p X, curl -u u:X,
//! --password X, redis-cli -a X), environment variables (DB_PASS=X), config formats (.netrc,
//! XML), URLs (user:X@host, ?sig=X), HTTP headers (Authorization: Token X) and session cookies.
//!
//! After an explicit separator (`:`, `=`, "is", "olarak", quotes) the word after a password
//! trigger is taken as the password. After a plain space it must look like one
//! (`looks_like_password`), so sentences about passwords ("şifre politikası güncellendi",
//! "şifremi unuttum", "password policies") are left alone.

use super::normalize::Lowered;
use super::secrets::{is_word_shaped, trim_value};
use super::{PiiEntity, PiiType};
use crate::morphology::TurkishMorphology;
use crate::phonology::to_turkish_lower;
use lazy_static::lazy_static;
use regex::Regex;

lazy_static! {
    /// A password word in a form that introduces its value: şifre, şifrem, şifresi, şifreniz,
    /// şifremi, şifresini, şifreyi, parolam, parolası, PIN, PIN'im, pinim, PIN kodu, kullanıcı
    /// şifresi, password, passwd, passcode, passphrase, pwd. Plurals and derivations (şifreler,
    /// şifreleme, şifreli, şifresiz) introduce none and do not match.
    static ref TRIGGER_AFTER: Regex = Regex::new(
        r"(?i)\b(?:((?:[şs][iİı]fre|parola)(?:m|n|s[iİı]|m[iİı]z|n[iİı]z)?(?:[iİıI]|y[iİıI]|n[iİıI])?|kullan[ıi]c[ıi]\s+[şs][iİı]fres[iİı])|(p[iİı]n(?:\s*kodu(?:m|nuz|n)?|['’]?(?:[iİı]m|[iİı]n[iİı]z|[iİı]))?)|(password|passwd|passcode|passphrase|pwd)|(pass))\b"
    ).unwrap();

    /// "X şifresiyle / şifreyle / parolasıyla / parolayla": the value comes first.
    static ref TRIGGER_BEFORE: Regex = Regex::new(
        r"(?i)(\S+)\s+(?:[şs][iİı]fre(?:s[iİı])?yle|parola(?:s[ıI])?yla)\b"
    ).unwrap();

    /// One-time codes after an explicit context word.
    static ref OTP_REGEX: Regex = Regex::new(
        r"(?i)\b(?:do[ğg]rulama|sms|tek\s+kullan[ıi]ml[ıi]k|aktivasyon|verification|confirmation|one[- ]?time|2fa|otp|mfa|(?:onay|g[üu]venlik|giri[şs]|login|security|auth)\s+(?:kod\w*|code|pin))\b[^\n\d]{0,30}?\b(\d{4,8})\b"
    ).unwrap();

    /// Credentials in command lines, environment variables, config files, URLs and headers.
    /// Each has one value group per quoting style: '...', "...", or bare.
    static ref STRUCTURED: Vec<Regex> = [
        // mysql -pSECRET (attached; a bare -p prompts)
        r"\b(?:mysql|mysqldump|mariadb|mysqladmin|mysqlimport)\b[^\n|;&]*?\s-p(?:'([^'\n]+)'|([^\s'][^\s]*))",
        r"\bsshpass\s+-p\s*(?:'([^'\n]+)'|\x22([^\x22\n]+)\x22|(\S+))",
        r"\b(?:docker|podman|nerdctl|buildah|skopeo)\s+login\b[^\n|;&]*?\s(?:-p|--password)(?:=|\s+)(?:'([^'\n]+)'|\x22([^\x22\n]+)\x22|(\S+))",
        r"\bredis-cli\b[^\n|;&]*?\s-a\s+(?:'([^'\n]+)'|\x22([^\x22\n]+)\x22|(\S+))",
        r"\bcurl\b[^\n|;&]*?\s(?:-u|--user)\s*['\x22]?[^:\s'\x22]+:([^\s'\x22]+)",
        r"(?i)(?:^|\s)--(?:password|passwd|pass|pwd|token|api[-_]?key|secret|client[-_]secret|auth[-_]token|access[-_]token)(?:=|\s+)(?:'([^'\n]+)'|\x22([^\x22\n]+)\x22|(\S+))",
        // DB_PASS=..., export REDIS_PASS=... (names ending in PASS; PASSWORD, SECRET and TOKEN
        // are the key=value rule's)
        r"\b(?:export\s+|set\s+)?[A-Z][A-Z0-9_]*PASS\s*=\s*(?:'([^'\n]*)'|\x22([^\x22\n]*)\x22|(\S+))",
        // .netrc
        r"(?i)\bmachine\s+\S+\s+login\s+\S+\s+password\s+(\S+)",
        // <password>...</password>
        r"(?i)<(?:password|passwd|pwd|secret|token|api[_-]?key|access[_-]?key|client[_-]?secret)>([^<\n]{1,256})</",
        // scheme://user:password@host
        r"\b[a-zA-Z][a-zA-Z0-9+.-]*://[^\s:/@]+:([^\s@/]+)@",
        // ?sig=..., &session=... (key=value covers api_key, token, secret, password)
        r"(?i)[?&](?:sig|signature|session|sessionid|sid|auth|apikey|key)=([^&\s#]+)",
        // Authorization: Token / ApiKey / Basic / Digest / JWT ... (Bearer has its own rule)
        r"(?i)\bAuthorization\s*:\s*(?:Token|ApiKey|Api-Key|Key|Basic|Digest|JWT|SSWS|Signature)\s+([^\s,]+)",
        // session cookies
        r"(?i)\b(?:session[_-]?id|sessionid|sess|sid|auth[_-]?token|jwt|csrf[_-]?token|csrftoken|connect\.sid|phpsessid|jsessionid|_session|remember[_-]?token)=([^;\s]+)",
        // root:toor, admin/admin123: a default account name glued to its password
        // (not inside a path: /home/ubuntu/project, /root/.ssh/config)
        r"(?i)(?:^|[\s\x22'(=,;])(?:root|admin|administrator|sa|postgres|ubuntu|pi|guest|oracle|ftp|sysadmin)[:/]([^\s,;:/.][^\s,;:/]{2,63})(?:$|[\s\x22'),;])",
        // Login with admin / password, giriş: ali / şifre
        r"(?i)\b(?:log\s?in|sign\s?in|giri[şs](?:\s+yap\w*)?)\b\s*(?:with|as|ile|:)\s*[^\s/:]+\s*/\s*(\S+)",
    ]
    .iter()
    .map(|p| Regex::new(p).unwrap())
    .collect();

    /// Words for "no value" that follow a password word in prose or templates.
    static ref NON_VALUES: std::collections::HashSet<&'static str> = [
        "yok", "none", "null", "nil", "boş", "bos", "empty", "n/a", "-", "gizli", "hidden",
        "required", "optional", "zorunlu", "isteğe", "unknown", "bilinmiyor",
    ]
    .into_iter()
    .collect();

    /// Passwords people actually use that read as ordinary words.
    static ref COMMON_PASSWORDS: std::collections::HashSet<&'static str> = [
        "password", "parola", "sifre", "şifre", "admin", "root", "toor", "qwerty", "qwertyuiop",
        "asdfgh", "asdfghjkl", "asdasd", "zxcvbn", "letmein", "welcome", "iloveyou", "monkey",
        "dragon", "master", "sunshine", "princess", "football", "baseball", "shadow", "superman",
        "galatasaray", "fenerbahce", "fenerbahçe", "besiktas", "beşiktaş", "trabzonspor",
        "bursaspor", "goztepe", "göztepe", "ankara", "istanbul", "izmir", "turkiye", "türkiye",
        "sevgilim", "askim", "aşkım", "canim", "canım", "bebegim", "bebeğim", "hayatim",
        "hayatım", "gizlisifre", "gizlişifre", "deneme", "test", "guest", "misafir",
    ]
    .into_iter()
    .collect();

    /// Shared with names.rs, which asks whether a word is a common word.
    pub(super) static ref MORPHOLOGY: TurkishMorphology = TurkishMorphology::new();
}

/// Connectors between a password word and its value; the first group makes the value
/// explicit (şifre olarak X, password is X).
const STRONG_CONNECTORS: &[&str] = &["is", "was", "olarak", "şudur", "sudur", "ise"];
const WEAK_CONNECTORS: &[&str] = &[
    "şu",
    "su",
    "de",
    "da",
    "yeni",
    "new",
    "current",
    "the",
    "numarası",
    "numaram",
    "numaranız",
    "numarasi",
    "bilgisi",
    "bilgim",
    "bilgileri",
    "kodu",
    "kodum",
    "kodunuz",
];
/// How far past a password word an explicit separator may come (password for the db is X).
const LOOKAHEAD_TOKENS: usize = 6;

/// Whether `value` (already trimmed) reads as a password rather than a word of the sentence.
pub(crate) fn looks_like_password(value: &str, turkish_trigger: bool) -> bool {
    let lower = to_turkish_lower(value);
    if value.chars().any(|c| c.is_ascii_digit()) || COMMON_PASSWORDS.contains(lower.as_str()) {
        return true;
    }
    if !is_word_shaped(value) {
        return true; // symbols inside, or mixed case: P@ss, kEdİmaVi
    }
    // A lower-case word Turkish morphology does not know (qwerty, asdf) after a Turkish
    // password word; a known word (konusu, politikası, unuttum) or a capitalised one (Gmail)
    // is part of the sentence.
    turkish_trigger
        && value.chars().next().is_some_and(char::is_lowercase)
        && value.chars().count() >= 4
        && MORPHOLOGY.analyze(&lower).is_empty()
}

fn accept(value: &str, strong: bool, turkish_trigger: bool, pin_trigger: bool) -> bool {
    let n = value.chars().count();
    if !(3..=128).contains(&n) || value.starts_with("{{") {
        return false;
    }
    let lower = to_turkish_lower(value);
    if NON_VALUES.contains(lower.as_str()) {
        return false;
    }
    if pin_trigger {
        // A PIN is digits; a PIN word followed by anything else is prose (pin the issue).
        let digits = value.chars().all(|c| c.is_ascii_digit());
        return (digits && (4..=8).contains(&n)) || (strong && !digits && n >= 4);
    }
    strong || looks_like_password(value, turkish_trigger)
}

/// The value after a password word ending at `from`, as (start, end, strong). An explicit
/// separator (`:`, `=`, is, was, olarak) a few words on wins ("the admin password for the db is
/// X", "Parolanı unutma: X"); otherwise separators and connectors are skipped and the first
/// word is the candidate.
fn value_after(text: &str, from: usize) -> Option<(usize, usize, bool)> {
    let mut pos = from;
    for i in 0..LOOKAHEAD_TOKENS {
        let rest = &text[pos..];
        let trimmed = rest.trim_start();
        let token_len = trimmed.find(char::is_whitespace).unwrap_or(trimmed.len());
        // past the first word the lookahead stays on the line (Şifre politikası
        // güncellendi\nToplantı: Pazartesi is not a password)
        let skipped = &rest[..rest.len() - trimmed.len()];
        if token_len == 0 || (i > 0 && skipped.contains(['\n', '\r'])) {
            break;
        }
        let token_start = pos + (rest.len() - trimmed.len());
        if let Some(span) = quoted_span(text, token_start) {
            return Some((span.0, span.1, true)); // a quoted value is explicit
        }
        let token = &trimmed[..token_len];
        let word = to_turkish_lower(token);
        // "word:" or "word=" with the value after it, or a separator token of its own
        if let Some(sep) = token.find([':', '=']) {
            if sep + 1 < token.len() {
                // "unutma:Deniz2024" (glued): the value is after the separator
                let value_from = token_start + sep + 1;
                let (lead, value) = trim_value(&text[value_from..token_start + token_len]);
                if !value.is_empty() && sep > 0 {
                    return Some((value_from + lead, value_from + lead + value.len(), true));
                }
            } else if sep > 0 || token.len() == 1 {
                return explicit_value(text, token_start + token_len);
            }
        }
        if STRONG_CONNECTORS.contains(&word.as_str()) {
            // "şifre olarak X" (right after the password word) and "... is X" introduce the
            // value; a later "olarak" follows it (şifremi X olarak değiştirdim).
            if i == 0 || word != "olarak" {
                return explicit_value(text, token_start + token_len);
            }
            break;
        }
        pos = token_start + token_len;
    }
    first_value(text, from)
}

/// The next word after an explicit separator, as a strong value.
fn explicit_value(text: &str, from: usize) -> Option<(usize, usize, bool)> {
    let rest = &text[from..];
    let trimmed = rest.trim_start();
    let start = from + (rest.len() - trimmed.len());
    if let Some((s, e)) = quoted_span(text, start) {
        return Some((s, e, true));
    }
    let token_len = trimmed.find(char::is_whitespace).unwrap_or(trimmed.len());
    let (lead, value) = trim_value(&trimmed[..token_len]);
    (!value.is_empty()).then_some((start + lead, start + lead + value.len(), true))
}

/// The inside of a quoted value starting at `at`: up to the matching closing quote on the same
/// line, at most 128 bytes ("mavi kedi 42", “correct horse battery staple”). A value of
/// several words was cut at its first space.
fn quoted_span(text: &str, at: usize) -> Option<(usize, usize)> {
    let open = text[at..].chars().next()?;
    let close = match open {
        '"' | '\'' | '`' => open,
        '“' => '”',
        '‘' => '’',
        '«' => '»',
        _ => return None,
    };
    let inner = at + open.len_utf8();
    let rest = &text[inner..];
    let end = rest.find([close, '\n', '\r'])?;
    let ok = rest[end..].starts_with(close) && (1..=128).contains(&end);
    ok.then_some((inner, inner + end))
}

/// The first word after the password word, skipping separators and connectors.
fn first_value(text: &str, from: usize) -> Option<(usize, usize, bool)> {
    let mut pos = from;
    let mut strong = false;
    for _ in 0..4 {
        let rest = &text[pos..];
        let trimmed = rest.trim_start();
        pos += rest.len() - trimmed.len();
        let first = trimmed.chars().next()?;
        if matches!(first, ':' | '=' | '-' | '–' | '>' | '→') {
            strong |= matches!(first, ':' | '=');
            pos += first.len_utf8();
            continue;
        }
        let token_len = trimmed.find(char::is_whitespace).unwrap_or(trimmed.len());
        let token = &trimmed[..token_len];
        let word = to_turkish_lower(token.trim_end_matches([',', '.', ':', ';']));
        if STRONG_CONNECTORS.contains(&word.as_str()) {
            strong = true;
            pos += token_len;
            continue;
        }
        if WEAK_CONNECTORS.contains(&word.as_str()) {
            pos += token_len;
            continue;
        }
        if let Some((s, e)) = quoted_span(text, pos) {
            return Some((s, e, true));
        }
        let quoted = token.starts_with(['"', '\'', '“', '‘', '`']);
        let (lead, value) = trim_value(token);
        if value.is_empty() {
            return None;
        }
        let start = pos + lead;
        let end = start + value.len();
        // "Şifremi X olarak değiştirdim": olarak after the value makes it explicit too.
        let after = text[pos + token_len..].trim_start();
        let olarak_after = to_turkish_lower(after.split_whitespace().next().unwrap_or(""))
            .trim_end_matches([',', '.'])
            == "olarak";
        return Some((start, end, strong || quoted || olarak_after));
    }
    None
}

fn push(out: &mut Vec<PiiEntity>, text: &str, start: usize, end: usize, confidence: f32) {
    // the value ends where the next field begins (DB_PASS=X;HOST=db, password=X,user=ali)
    let value = super::secrets::cut_at_next_field(&text[start..end]);
    let end = start + value.len();
    out.push(PiiEntity {
        text: value.to_string(),
        label: "SIFRE".to_string(),
        pii_type: PiiType::Credentials,
        start,
        end,
        confidence,
        stem: value.to_string(),
        suffix: None,
    });
}

/// Detects passwords, PINs and one-time codes; see the module documentation.
pub fn detect_passwords(text: &str, out: &mut Vec<PiiEntity>) {
    detect_passwords_in(text, &Lowered::of(text), out);
}

/// `detect_passwords` with the text's lower-cased forms already made (`detect_secrets` makes
/// them once for all its gates).
pub(crate) fn detect_passwords_in(text: &str, lowered: &Lowered, out: &mut Vec<PiiEntity>) {
    let has = |k: &str| lowered.has(k);
    if has("şifre") || has("sifre") || has("parola") || has("pin") || has("pass") || has("pwd") {
        for cap in TRIGGER_AFTER.captures_iter(text) {
            let whole = cap.get(0).unwrap();
            let (turkish, pin, bare_pass) = (
                cap.get(1).is_some(),
                cap.get(2).is_some(),
                cap.get(4).is_some(),
            );
            if let Some((start, end, strong)) = value_after(text, whole.end()) {
                // "pass" alone is a common word: only "pass: X" with a password-like X.
                if bare_pass && !(strong && looks_like_password(&text[start..end], false)) {
                    continue;
                }
                if accept(&text[start..end], strong, turkish || pin, pin) {
                    push(out, text, start, end, 0.97);
                }
            }
        }
        for cap in TRIGGER_BEFORE.captures_iter(text) {
            let m = cap.get(1).unwrap();
            let (lead, value) = trim_value(m.as_str());
            if !value.is_empty() && accept(value, false, true, false) {
                push(
                    out,
                    text,
                    m.start() + lead,
                    m.start() + lead + value.len(),
                    0.95,
                );
            }
        }
    }

    for cap in OTP_REGEX.captures_iter(text) {
        let m = cap.get(1).unwrap();
        // A year after "SMS" is not a code.
        let is_year = m.as_str().len() == 4 && matches!(&m.as_str()[..2], "19" | "20");
        if !is_year {
            push(out, text, m.start(), m.end(), 0.97);
        }
    }

    for re in STRUCTURED.iter() {
        for cap in re.captures_iter(text) {
            if let Some(m) = (1..cap.len()).find_map(|i| cap.get(i)) {
                let (lead, value) = trim_value(m.as_str());
                let v = to_turkish_lower(value);
                if value.chars().count() >= 2
                    && !matches!(v.as_str(), "null" | "none" | "true" | "false")
                {
                    push(
                        out,
                        text,
                        m.start() + lead,
                        m.start() + lead + value.len(),
                        0.98,
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn found(text: &str) -> Vec<String> {
        let mut out = Vec::new();
        detect_passwords(text, &mut out);
        out.into_iter().map(|e| e.text).collect()
    }

    #[test]
    fn sentences_with_a_password() {
        for (text, value) in [
            ("Şifremi ankara06 olarak değiştirdim.", "ankara06"),
            ("Şifre olarak qwerty kullan.", "qwerty"),
            ("ankara06 şifresiyle giriş yap.", "ankara06"),
            ("Hesabın şifresini Gizli.2024 yaptım.", "Gizli.2024"),
            (
                "Giriş bilgileri: kullanıcı adı ali, şifre galatasaray",
                "galatasaray",
            ),
            ("my password is qwerty", "qwerty"),
            ("The passcode is 123456.", "123456"),
            ("use passphrase sifre123 to unlock", "sifre123"),
            ("PIN'im 4321", "4321"),
            ("mobil onay pinim 45606 oldu", "45606"),
            ("Doğrulama kodunuz: 482913", "482913"),
            ("SMS ile gelen kod 482913", "482913"),
            ("Your verification code is 482913", "482913"),
            ("PIN numarası 1234", "1234"),
            ("the admin password for the db is hunter2", "hunter2"),
            ("Parolanı unutma: Deniz2024!", "Deniz2024!"),
            ("ŞİFREM: KARTAL1903", "KARTAL1903"),
            ("pass: hunter2", "hunter2"),
            ("Gerekirse root:toor ile bağlan", "toor"),
            ("Modem arayüzü admin/admin123", "admin123"),
        ] {
            assert!(
                found(text).iter().any(|v| v == value),
                "{text}: {:?}",
                found(text)
            );
        }
    }

    #[test]
    fn commands_configs_and_headers() {
        for (text, value) in [
            ("mysql -u root -pankara06 prod", "ankara06"),
            ("sshpass -p 'qwerty' ssh root@10.0.0.5", "qwerty"),
            (
                "docker login -u ali -p 123456 registry.example.com",
                "123456",
            ),
            ("curl -u admin:P@ssw0rd https://api.example.com", "P@ssw0rd"),
            ("export DB_PASS=P@ssw0rd", "P@ssw0rd"),
            ("redis-cli -h cache -a hunter22 ping", "hunter22"),
            ("app --password=s3cr3t --verbose", "s3cr3t"),
            (
                "machine api.example.com login ali password qwerty",
                "qwerty",
            ),
            ("<password>fenerbahce1907</password>", "fenerbahce1907"),
            (
                "git clone https://ali:Ali1990!@github.com/org/repo.git",
                "Ali1990!",
            ),
            ("Authorization: Token qwerty123", "qwerty123"),
            ("Cookie: session_id=P@ssw0rd; theme=dark", "P@ssw0rd"),
            (
                "https://x.blob.core.windows.net/c?sv=2022&sig=AbC%2Bd3f&se=2026",
                "AbC%2Bd3f",
            ),
            ("Login with admin / qwerty", "qwerty"),
        ] {
            assert!(
                found(text).iter().any(|v| v == value),
                "{text}: {:?}",
                found(text)
            );
        }
    }

    #[test]
    fn sentences_about_passwords_are_left_alone() {
        for text in [
            "Şifre konusu toplantıda konuşuldu ve parola politikası güncellendi.",
            "Kullanıcılar şifrelerini düzenli olarak değiştirmelidir.",
            "Şifreleme algoritması AES-256 kullanır.",
            "Şifremi unuttum, yardım eder misiniz?",
            "Şifrenizi kimseyle paylaşmayın.",
            "Gmail şifresiyle giriş yapamıyorum.",
            "Password policies should require at least 12 characters.",
            "Please pin the issue to the board.",
            "SMS 2024 yılında yaygınlaştı.",
            "Cookie: theme=dark; lang=tr",
            "mkdir -p build && cd build",
            "Bu şifre çok zayıf.",
            "PIN kodu en az 6 haneli olmalı.",
            "You shall not pass: the bridge is closed.",
            "admin: Ahmet Yılmaz",
        ] {
            assert!(found(text).is_empty(), "{text}: {:?}", found(text));
        }
    }
}
