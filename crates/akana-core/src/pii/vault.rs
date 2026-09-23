//! Session Vault for reversible pseudonymization and de-pseudonymization in secure LLM gateways.
//!
//! Tracks bidirectional mappings between sensitive personal data and placeholders/surrogates,
//! preserving grammatical case and adjusting vowel harmony upon restoration.

use super::morphology::{detect_suffix_case, harmonize_suffix};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Masking / Pseudonymization mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PiiMode {
    /// Structured token mode: `{{NAME_1}}`, `{{TCKN_1}}`, `{{IBAN_1}}` (unambiguous, gateway standard).
    Placeholder,
    /// Bracketed tag mode: `[AD]`, `[TCKN]`, `[IBAN]` (benchmark & dataset standard).
    Tag,
    /// Irreversible redaction: replaces with `***` or `[GİZLENDİ]`.
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
        // If this exact original text was already pseudonymized in this session, reuse its token
        if let Some(existing) = self.original_to_placeholder.get(original_text) {
            return existing.clone();
        }

        let count = self.label_counters.entry(label.to_string()).or_insert(0);
        *count += 1;
        let index = *count;

        let placeholder = match mode {
            PiiMode::Placeholder => format!("{{{{{}_{}}}}}", label.to_uppercase(), index),
            PiiMode::Tag => format!("[{}]", label.to_uppercase()),
            PiiMode::Anonymize => format!("[{}]", label.to_uppercase()),
            PiiMode::SyntheticSurrogate => get_synthetic_surrogate(label, index),
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

        placeholder
    }

    /// Restores placeholders in an inbound LLM response back to the original sensitive data,
    /// performing vowel harmony adjustment if the LLM appended suffixes to the placeholder.
    pub fn restore(&self, llm_response: &str) -> String {
        let mut result = llm_response.to_string();

        // Sort entries by placeholder length descending to avoid substring collision
        let mut entries: Vec<(&String, &VaultEntry)> = self.placeholder_to_entry.iter().collect();
        entries.sort_by_key(|a| std::cmp::Reverse(a.0.len()));

        for (placeholder, entry) in entries {
            // Check for placeholder with attached suffix: e.g. `{{NAME_1}}'e`, `{{NAME_1}}'in`, `Can Demir'e`
            let patterns_to_check = [
                format!("{placeholder}'"),
                format!("{placeholder}’"),
                format!("{placeholder}´"),
            ];

            for prefix in &patterns_to_check {
                while let Some(pos) = result.find(prefix.as_str()) {
                    let after_prefix = pos + prefix.len();
                    let suffix_len: usize = result[after_prefix..]
                        .chars()
                        .take_while(|c| c.is_alphabetic())
                        .map(|c| c.len_utf8())
                        .sum();

                    let found_suffix = &result[after_prefix..after_prefix + suffix_len];
                    let full_match_len: usize = prefix.len() + suffix_len;

                    // If a case suffix is detected on the placeholder, harmonize it onto the original stem!
                    let replacement = if let Some(case) = detect_suffix_case(found_suffix) {
                        harmonize_suffix(&entry.original_stem, case)
                    } else {
                        format!("{}'{}", entry.original_stem, found_suffix)
                    };

                    result.replace_range(pos..pos + full_match_len, &replacement);
                }
            }

            // Replace standard isolated placeholder
            result = result.replace(placeholder.as_str(), &entry.original_text);
        }

        result
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

/// Generates valid-looking, realistic Turkish synthetic surrogates for LLM fluency.
fn get_synthetic_surrogate(label: &str, index: usize) -> String {
    static FAKE_NAMES: &[&str] = &[
        "Can Demir",
        "Burak Şahin",
        "Merve Kaya",
        "Zeynep Çelik",
        "Emre Yılmaz",
    ];
    static FAKE_TCKNS: &[&str] = &["10000000146", "51980838902", "12345678950", "98765432108"];
    static FAKE_IBANS: &[&str] = &[
        "TR330006100519786457841326",
        "TR440001000000000000000001",
        "TR550006200000000000000002",
    ];
    static FAKE_PHONES: &[&str] = &["0555 987 65 43", "0544 123 99 88", "0533 555 44 33"];

    match label {
        "NAME" | "PERSON" | "AD" | "KISI" => FAKE_NAMES[(index - 1) % FAKE_NAMES.len()].to_string(),
        "TCKN" | "TC" => FAKE_TCKNS[(index - 1) % FAKE_TCKNS.len()].to_string(),
        "IBAN" => FAKE_IBANS[(index - 1) % FAKE_IBANS.len()].to_string(),
        "PHONE" | "TEL" => FAKE_PHONES[(index - 1) % FAKE_PHONES.len()].to_string(),
        "OZEL_URL" => format!("https://portal.example.com/dosya/{:06}", 100000 + index),
        "OZEL_TARIH" | "DOGUM_TARIHI" => format!("{:02}.05.1990", 10 + index % 18),
        _ => format!("[{}_{}]", label, index),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
