//! Session Vault for reversible pseudonymization and de-pseudonymization in secure LLM gateways.
//!
//! Tracks bidirectional mappings between sensitive personal data and placeholders/surrogates,
//! preserving grammatical case and adjusting vowel harmony upon restoration.

use super::morphology::{detect_suffix_case, harmonize_suffix};
use super::validators::{validate_iban, validate_tckn};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Masking / Pseudonymization mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PiiMode {
    /// Structured token mode: `{{NAME_1}}`, `{{TCKN_1}}`, `{{IBAN_1}}` (unambiguous, gateway standard).
    Placeholder,
    /// Bracketed tag mode: `[AD]`, `[TCKN]`, `[IBAN]` (benchmark & dataset standard).
    Tag,
    /// Redaction mode: replaces sensitive entities with category label tag (`[AD]`, `[TCKN]`, `[SIFRE]`).
    Anonymize,
    /// Synthetic surrogate mode: replaces with realistic fake Turkish data (`Ahmet` -> `Can`).
    SyntheticSurrogate,
}

/// Stored entry in the vault for a single pseudonymized entity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VaultEntry {
    pub placeholder: String,
    pub original_text: String,
    pub pii_label: String,
    pub original_stem: String,
    pub original_suffix: Option<String>,
}

/// Bidirectional Session Vault holding mappings for prompt anonymization and response restoration.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PiiVault {
    /// Maps placeholder (e.g. `"{{NAME_1}}"`) to VaultEntry
    pub placeholder_to_entry: HashMap<String, VaultEntry>,
    /// Maps original text to placeholder (for consistent re-occurrence replacement within a session)
    pub original_to_placeholder: HashMap<String, String>,
    /// Counts per entity label to produce sequential IDs (`NAME_1`, `NAME_2`, etc.)
    label_counters: HashMap<String, usize>,
    /// Tags (`[AD]`) issued for more than one original: restoring them would pick one of the
    /// originals for every occurrence, so `restore` leaves them as they are.
    #[serde(default)]
    ambiguous: HashSet<String>,
}

impl PiiVault {
    /// Creates a new empty vault.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a detected entity into the vault, generating a consistent placeholder or surrogate.
    pub fn register(
        &mut self,
        original_text: &str,
        label: &str,
        stem: &str,
        suffix: Option<&str>,
        mode: PiiMode,
    ) -> String {
        self.register_in("", original_text, label, stem, suffix, mode)
    }

    /// `register` for an entity of `text`: a new placeholder or surrogate is never one already
    /// issued in this vault for another value, nor a string that occurs in `text` (a surrogate
    /// "Can Demir" for a prompt that also names the real Can Demir would restore him as the
    /// masked person).
    pub fn register_in(
        &mut self,
        text: &str,
        original_text: &str,
        label: &str,
        stem: &str,
        suffix: Option<&str>,
        mode: PiiMode,
    ) -> String {
        let taken = issued_shapes_in(text);
        self.register_avoiding(&taken, original_text, label, stem, suffix, mode)
    }

    /// `register_in` with the placeholder- and surrogate-shaped strings of the text already
    /// collected by `issued_shapes_in`, so that masking a text scans it once rather than once
    /// per candidate.
    pub(crate) fn register_avoiding(
        &mut self,
        taken: &HashSet<String>,
        original_text: &str,
        label: &str,
        stem: &str,
        suffix: Option<&str>,
        mode: PiiMode,
    ) -> String {
        // If this exact original text was already pseudonymized in this session, reuse its token
        if let Some(existing) = self.original_to_placeholder.get(original_text) {
            return existing.clone();
        }
        if !stem.is_empty() && stem != original_text {
            if let Some(existing) = self.original_to_placeholder.get(stem) {
                return existing.clone();
            }
        }

        let placeholder = if matches!(mode, PiiMode::Tag | PiiMode::Anonymize) {
            // One tag for every value of a label: it is restorable only while it stands for a
            // single original.
            let tag = format!("[{}]", label.to_uppercase());
            if self
                .placeholder_to_entry
                .get(&tag)
                .is_some_and(|e| e.original_text != original_text)
            {
                self.placeholder_to_entry.remove(&tag);
                self.ambiguous.insert(tag.clone());
            }
            if self.ambiguous.contains(&tag) {
                self.original_to_placeholder
                    .insert(original_text.to_string(), tag.clone());
                return tag;
            }
            tag
        } else {
            // Every rejected candidate is in `taken` or in the vault, so within `limit` tries a
            // family either gives a free one or has repeated itself: the surrogate families
            // are cyclic (dates repeat after 13,440, phones after 9 million), and a text that
            // holds a whole cycle kept this loop running forever. Placeholders do not repeat,
            // so after `limit` tries they always end it.
            let limit = taken.len() + self.placeholder_to_entry.len() + 1;
            let mut tries = 0usize;
            loop {
                let count = self.label_counters.entry(label.to_string()).or_insert(0);
                *count += 1;
                let index = *count;
                tries += 1;
                let candidate = match mode {
                    PiiMode::SyntheticSurrogate if tries <= limit => {
                        get_synthetic_surrogate(label, index)
                    }
                    _ => format!("{{{{{}_{}}}}}", label.to_uppercase(), index),
                };
                if !self.placeholder_to_entry.contains_key(&candidate)
                    && !taken.contains(&candidate)
                {
                    break candidate;
                }
            }
        };

        let entry = VaultEntry {
            placeholder: placeholder.clone(),
            original_text: original_text.to_string(),
            pii_label: label.to_string(),
            original_stem: stem.to_string(),
            original_suffix: suffix.map(|s| s.to_string()),
        };

        self.placeholder_to_entry.insert(placeholder.clone(), entry);
        self.original_to_placeholder
            .insert(original_text.to_string(), placeholder.clone());
        if !stem.is_empty() && stem != original_text {
            self.original_to_placeholder
                .insert(stem.to_string(), placeholder.clone());
        }

        placeholder
    }

    /// Restores placeholders in an inbound LLM response back to the original sensitive data,
    /// performing vowel harmony adjustment if the LLM appended suffixes to the placeholder.
    ///
    /// One left-to-right pass over the response, longest placeholder first at each position:
    /// a restored value is never searched again, so an original that contains another
    /// placeholder's text (or a surrogate's) cannot pull a second value in. Placeholders with an
    /// empty key are ignored.
    pub fn restore(&self, llm_response: &str) -> String {
        // Which keys occur in the response: a key of a placeholder or surrogate shape by one
        // scan of the response (searching it once per key was quadratic in long sessions),
        // any other key (tags, keys from `from_mapping`) by searching for it.
        let shapes = issued_shapes_in(llm_response);
        let mut keys: Vec<&str> = self
            .placeholder_to_entry
            .keys()
            .map(String::as_str)
            .filter(|k| {
                !k.is_empty()
                    && (shapes.contains(*k)
                        || (!issued_shapes_in(k).contains(*k) && llm_response.contains(*k)))
            })
            .collect();
        if keys.is_empty() {
            return llm_response.to_string();
        }
        // The regex alternation is leftmost-first, so longer placeholders must come first
        // ({{AD_10}} before {{AD_1}}).
        keys.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
        let alternation = keys
            .iter()
            .map(|k| regex::escape(k))
            .collect::<Vec<_>>()
            .join("|");
        // Placeholder, then optionally an apostrophe and the suffix the LLM attached to it:
        // `{{AD_1}}'e`, `{{AD_1}}'in`, `Can Demir'e`.
        // Thousands of distinct placeholders in one response exceed the default compiled size;
        // if even this limit is exceeded the response is returned unrestored (placeholders kept).
        let pattern = match regex::RegexBuilder::new(&format!(
            r"({alternation})(?:(['’´])(\p{{Alphabetic}}*))?"
        ))
        .size_limit(256 << 20)
        .build()
        {
            Ok(re) => re,
            Err(_) => return llm_response.to_string(),
        };
        let bytes_before = |i: usize| llm_response[..i].chars().next_back();
        pattern
            .replace_all(llm_response, |caps: &regex::Captures| {
                // A key that begins or ends with a letter or digit (a surrogate) must stand as
                // a word: Can Demir is not restored inside Can Demirtaş, nor a surrogate TCKN
                // inside a longer number.
                let key = caps.get(1).unwrap();
                let (first, last) = (
                    key.as_str().chars().next(),
                    key.as_str().chars().next_back(),
                );
                let glued_before = first.is_some_and(char::is_alphanumeric)
                    && bytes_before(key.start()).is_some_and(char::is_alphanumeric);
                let glued_after = last.is_some_and(char::is_alphanumeric)
                    && llm_response[key.end()..]
                        .chars()
                        .next()
                        .is_some_and(char::is_alphanumeric);
                if glued_before || glued_after {
                    return caps[0].to_string();
                }
                let entry = &self.placeholder_to_entry[&caps[1]];
                match caps.get(3) {
                    Some(found_suffix) => {
                        let found_suffix = found_suffix.as_str();
                        // If a case suffix is detected on the placeholder, harmonize it onto the original stem!
                        if let Some(case) = detect_suffix_case(found_suffix) {
                            harmonize_suffix(&entry.original_stem, case)
                        } else {
                            format!("{}'{}", entry.original_stem, found_suffix)
                        }
                    }
                    None => entry.original_text.clone(),
                }
            })
            .into_owned()
    }

    /// Returns a direct key-value mapping from placeholder/surrogate to the original sensitive text.
    pub fn get_mapping(&self) -> HashMap<String, String> {
        let mut map = HashMap::with_capacity(self.placeholder_to_entry.len());
        for (k, v) in &self.placeholder_to_entry {
            map.insert(k.clone(), v.original_text.clone());
        }
        map
    }

    /// Returns the detailed mapping containing full VaultEntry metadata.
    pub fn get_detailed_mapping(&self) -> HashMap<String, VaultEntry> {
        self.placeholder_to_entry.clone()
    }

    /// Creates a PiiVault from a simple mapping { placeholder_or_surrogate: original_text }.
    pub fn from_mapping(mapping: &HashMap<String, String>) -> Self {
        let mut vault = Self::new();
        for (placeholder, original) in mapping {
            vault.placeholder_to_entry.insert(
                placeholder.clone(),
                VaultEntry {
                    placeholder: placeholder.clone(),
                    original_text: original.clone(),
                    pii_label: "PII".to_string(),
                    original_stem: original.clone(),
                    original_suffix: None,
                },
            );
            vault
                .original_to_placeholder
                .insert(original.clone(), placeholder.clone());
        }
        vault
    }

    /// Restores placeholders in an LLM response using a plain mapping { placeholder: original_text }.
    pub fn restore_with_mapping(llm_response: &str, mapping: &HashMap<String, String>) -> String {
        let vault = Self::from_mapping(mapping);
        vault.restore(llm_response)
    }

    /// Number of distinct entities registered in the vault.
    pub fn len(&self) -> usize {
        self.placeholder_to_entry.len()
    }

    /// Checks if the vault is empty.
    pub fn is_empty(&self) -> bool {
        self.placeholder_to_entry.is_empty()
    }
}

/// Generates valid-looking, realistic Turkish synthetic surrogates for LLM fluency, a different
/// one for every index: they used to repeat after 3-5 values (index modulo a short list), so
/// the 6th person got the 1st person's surrogate and restore turned both into the 6th.
const FIRST_NAMES: &[&str] = &[
    "Can", "Burak", "Merve", "Zeynep", "Emre", "Elif", "Kerem", "Selin", "Onur", "Derya", "Arda",
    "Esra",
];
const SURNAMES: &[&str] = &[
    "Demir", "Şahin", "Kaya", "Çelik", "Yılmaz", "Aydın", "Arslan", "Doğan", "Kılıç", "Koç",
    "Kurt", "Özdemir",
];

lazy_static::lazy_static! {
    /// The shapes `get_synthetic_surrogate` and placeholders take, but for the digit runs of
    /// TCKNs and IBANs, which `issued_shapes_in` windows itself. Surrogates are ASCII, so the
    /// digits are [0-9], not the Unicode \d.
    static ref ISSUED_SHAPE: regex::Regex = regex::Regex::new(&format!(
        r"\{{\{{[^{{}}\s]{{1,64}}\}}\}}|\[[^\[\]\s]{{1,40}}_[0-9]+\]|(?:{})\x20(?:{})|0555\x20[0-9]{{3}}\x20[0-9]{{2}}\x20[0-9]{{2}}|[0-9]{{2}}\.[0-9]{{2}}\.[0-9]{{4}}|https://portal\.example\.com/dosya/[0-9]{{6}}",
        FIRST_NAMES.join("|"),
        SURNAMES.join("|"),
    ))
    .unwrap();
    // ASCII digits only: the windows are cut by bytes, and the original text (not the folded
    // one) can hold full-width digits.
    static ref DIGIT_RUN: regex::Regex = regex::Regex::new(r"(?:TR)?[0-9]{11,64}").unwrap();
}

/// Every string in `text` that a placeholder or a surrogate could be: placeholders ({{AD_1}}),
/// bracket forms ([AD_145]), surrogate names, phones, dates and URLs, and each 11-digit window
/// of a digit run (TCKNs) and the 24 digits after a "TR" (IBANs). A new placeholder or
/// surrogate must not be one of them, or restore would also turn that text into the original.
/// Digit runs longer than 64 digits are not windowed.
pub(crate) fn issued_shapes_in(text: &str) -> HashSet<String> {
    let mut out: HashSet<String> = ISSUED_SHAPE
        .find_iter(text)
        .map(|m| m.as_str().to_string())
        .collect();
    for m in DIGIT_RUN.find_iter(text) {
        let run = m.as_str();
        if let Some(digits) = run.strip_prefix("TR") {
            if digits.len() >= 24 {
                out.insert(format!("TR{}", &digits[..24]));
            }
        }
        let digits = run.trim_start_matches("TR");
        for i in 0..=digits.len().saturating_sub(11) {
            if digits.len() >= i + 11 {
                out.insert(digits[i..i + 11].to_string());
            }
        }
    }
    out
}

fn get_synthetic_surrogate(label: &str, index: usize) -> String {
    let i = index.saturating_sub(1);

    match label {
        "NAME" | "PERSON" | "AD" | "KISI" => {
            // i -> (i mod 12, (i + i / 12) mod 12) visits all 144 pairs before repeating.
            let combos = FIRST_NAMES.len() * SURNAMES.len();
            if i >= combos {
                return format!("[{}_{}]", label, index);
            }
            let first = i % FIRST_NAMES.len();
            let last = (i + i / FIRST_NAMES.len()) % SURNAMES.len();
            format!("{} {}", FIRST_NAMES[first], SURNAMES[last])
        }
        "TCKN" | "TC" => surrogate_tckn(i),
        "IBAN" => surrogate_iban(i),
        "PHONE" | "TEL" => {
            let n = 9_876_543 - (i as u64 % 9_000_000);
            format!(
                "0555 {:03} {:02} {:02}",
                n / 10_000,
                (n / 100) % 100,
                n % 100
            )
        }
        "OZEL_URL" => format!("https://portal.example.com/dosya/{:06}", 100000 + index),
        "OZEL_TARIH" | "DOGUM_TARIHI" => {
            // Distinct day/month/year for every index (28 days x 12 months x 40 years).
            let (day, month, year) = (1 + i % 28, 1 + (i / 28) % 12, 1990 - (i / 336) as i64 % 40);
            format!("{day:02}.{month:02}.{year}")
        }
        _ => format!("[{}_{}]", label, index),
    }
}

/// A TCKN with valid check digits, distinct for every index (first nine digits 9xxxxxxxx).
fn surrogate_tckn(i: usize) -> String {
    let base = format!("{:09}", 900_000_000 + (i as u64 * 7_919) % 99_999_999);
    let d: Vec<u32> = base.chars().filter_map(|c| c.to_digit(10)).collect();
    let odd: u32 = d[0] + d[2] + d[4] + d[6] + d[8];
    let even: u32 = d[1] + d[3] + d[5] + d[7];
    let d10 = (odd * 7 + 10 * 10 - even) % 10;
    let d11 = (d.iter().sum::<u32>() + d10) % 10;
    let tckn = format!("{base}{d10}{d11}");
    debug_assert!(validate_tckn(&tckn));
    tckn
}

/// A Turkish IBAN with valid check digits, distinct for every index (bank code 00099).
fn surrogate_iban(i: usize) -> String {
    let bban = format!("000990{:016}", 1_000_000 + i as u64);
    // ISO 13616 check digits: move "TR00" to the end, letters to numbers (T=29, R=27), mod 97.
    let rearranged = format!("{bban}292700");
    let rem = rearranged
        .chars()
        .filter_map(|c| c.to_digit(10))
        .fold(0u64, |acc, d| (acc * 10 + d as u64) % 97);
    let iban = format!("TR{:02}{bban}", 98 - rem);
    debug_assert!(validate_iban(&iban));
    iban
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_surrogate_shape_is_recognised_in_text() {
        // a surrogate is avoided only when issued_shapes_in finds it in the prompt
        for label in [
            "AD",
            "TCKN",
            "IBAN",
            "TEL",
            "OZEL_URL",
            "DOGUM_TARIHI",
            "SIFRE",
        ] {
            for index in [1, 2, 77, 144, 145, 5_000, 13_441, 123_456] {
                let s = get_synthetic_surrogate(label, index);
                let text = format!("önce {s} sonra");
                assert!(issued_shapes_in(&text).contains(&s), "{label} {index}: {s}");
            }
        }
        let p = "{{AD_7}}";
        assert!(issued_shapes_in(&format!("x {p} y")).contains(p));
    }

    #[test]
    fn a_text_holding_a_whole_surrogate_cycle_does_not_hang() {
        // all 13,440 date surrogates in the prompt: the date family has nothing left
        let dates: Vec<String> = (1..=13_440)
            .map(|i| get_synthetic_surrogate("DOGUM_TARIHI", i))
            .collect();
        let text = dates.join(" ");
        let mut vault = PiiVault::new();
        let s = vault.register_in(
            &text,
            "12.03.1985",
            "DOGUM_TARIHI",
            "12.03.1985",
            None,
            PiiMode::SyntheticSurrogate,
        );
        assert!(!text.contains(&s), "{s}");
        assert_eq!(vault.restore(&s), "12.03.1985");
    }

    #[test]
    fn many_placeholders_in_the_text_are_skipped_quickly() {
        let text: String = (1..=20_000).map(|i| format!("{{{{AD_{i}}}}} ")).collect();
        let started = std::time::Instant::now();
        let mut vault = PiiVault::new();
        let p = vault.register_in(&text, "Ahmet", "AD", "Ahmet", None, PiiMode::Placeholder);
        assert_eq!(p, "{{AD_20001}}");
        assert!(started.elapsed().as_secs() < 5, "{:?}", started.elapsed());
    }

    #[test]
    fn test_vault_placeholder_and_restore() {
        let mut vault = PiiVault::new();
        let p1 = vault.register(
            "Ahmet Yılmaz",
            "NAME",
            "Ahmet Yılmaz",
            None,
            PiiMode::Placeholder,
        );
        let p2 = vault.register(
            "0532 123 45 67",
            "PHONE",
            "0532 123 45 67",
            None,
            PiiMode::Placeholder,
        );

        assert_eq!(p1, "{{NAME_1}}");
        assert_eq!(p2, "{{PHONE_1}}");

        // Simulate LLM response containing the placeholders
        let llm_resp = "Sayın {{NAME_1}}, talebiniz alındı. {{PHONE_1}} numarasına SMS gönderildi.";
        let restored = vault.restore(llm_resp);

        assert_eq!(
            restored,
            "Sayın Ahmet Yılmaz, talebiniz alındı. 0532 123 45 67 numarasına SMS gönderildi."
        );
    }

    #[test]
    fn test_vault_suffix_vowel_harmony_restoration() {
        let mut vault = PiiVault::new();
        vault.register("Murat", "NAME", "Murat", Some("a"), PiiMode::Placeholder);

        // LLM generates placeholder with suffix: {{NAME_1}}'a
        let llm_resp = "{{NAME_1}}'a başarıyla iletildi.";
        let restored = vault.restore(llm_resp);

        assert_eq!(restored, "Murat'a başarıyla iletildi.");
    }

    #[test]
    fn test_vault_mapping_export_and_restore_with_mapping() {
        let mut vault = PiiVault::new();
        vault.register(
            "Ahmet Yılmaz",
            "NAME",
            "Ahmet Yılmaz",
            None,
            PiiMode::Placeholder,
        );
        vault.register(
            "0532 123 45 67",
            "PHONE",
            "0532 123 45 67",
            None,
            PiiMode::Placeholder,
        );

        let mapping = vault.get_mapping();
        assert_eq!(mapping.get("{{NAME_1}}").unwrap(), "Ahmet Yılmaz");
        assert_eq!(mapping.get("{{PHONE_1}}").unwrap(), "0532 123 45 67");

        let llm_resp = "Sayın {{NAME_1}}, telefonunuz {{PHONE_1}} güncellendi.";
        let restored = PiiVault::restore_with_mapping(llm_resp, &mapping);
        assert_eq!(
            restored,
            "Sayın Ahmet Yılmaz, telefonunuz 0532 123 45 67 güncellendi."
        );
    }

    #[test]
    fn test_vault_redos_and_adversarial_patterns() {
        let mut vault = PiiVault::new();
        // Self-referential stem: replacement contains the exact placeholder prefix
        vault.register(
            "{{NAME_1}}'",
            "NAME",
            "{{NAME_1}}'",
            None,
            PiiMode::Placeholder,
        );

        let adversarial_input = "{{NAME_1}}'a ".repeat(500);
        let start = std::time::Instant::now();
        let restored = vault.restore(&adversarial_input);
        let elapsed = start.elapsed();

        // Must complete instantaneously without hanging or infinite loop
        assert!(
            elapsed.as_millis() < 500,
            "Restore took too long: {:?}",
            elapsed
        );
        assert!(!restored.is_empty());

        // Extreme repetition test exceeding MAX_ITERATIONS (9999)
        let extreme_input = "{{NAME_1}}'a ".repeat(15_000);
        let start2 = std::time::Instant::now();
        let restored2 = vault.restore(&extreme_input);
        let elapsed2 = start2.elapsed();
        assert!(
            elapsed2.as_millis() < 1000,
            "Extreme input took too long: {:?}",
            elapsed2
        );
        assert!(!restored2.is_empty());
    }
}
