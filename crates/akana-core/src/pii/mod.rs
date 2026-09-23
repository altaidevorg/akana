//! Advanced Turkish PII (Personally Identifiable Information) Recognition & Pseudonymization Engine.
//!
//! Aligned with KVKK (Kişisel Verilerin Korunması Kanunu), Saturday Labs 50-label taxonomy,
//! and high-throughput secure LLM gateway requirements.
//!
//! Features:
//! - Algorithmic checksum validators (TCKN, VKN, IBAN, Luhn Card, Plate, VIN, IMEI)
//! - Turkish morphology & apostrophe-aware stem/suffix demarcation
//! - Polysemy filter for Turkish names (e.g., Deniz, Barış, Gül)
//! - Comprehensive pattern recognizers (Phone, Email, IP, Port, Age, Addresses, Credentials)
//! - Reversible Session Vault with vowel-harmony aware restoration for LLM gateways

pub mod embedding_scorer;
pub mod gazetteer;
pub mod morphology;
pub mod patterns;
pub mod secrets;
pub mod spelled_numbers;
pub mod validators;
pub mod vault;

pub use embedding_scorer::*;
pub use gazetteer::*;
pub use morphology::*;
pub use patterns::*;
pub use secrets::*;
pub use spelled_numbers::*;
pub use validators::*;
pub use vault::*;

use crate::phonology::to_turkish_lower;
use crate::tokenization::TurkishTokenizer;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use stringzilla::StringZilla;

/// Primary PII Entity Types supported by Akana.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PiiType {
    Tckn,
    Vkn,
    Iban,
    CreditCard,
    Phone,
    Email,
    Name,
    Person,
    Address,
    Passport,
    DriverLicense,
    Plate,
    Vin,
    Imei,
    IpAddress,
    Port,
    Age,
    AgeRange,
    BirthDate,
    PrivateDate,
    PrivateUrl,
    BloodType,
    Health,
    Gender,
    Salary,
    Credentials,
    AccountNo,
    Ssn,
}

impl PiiType {
    pub fn as_str(&self) -> &'static str {
        match self {
            PiiType::Tckn => "TCKN",
            PiiType::Vkn => "VKN",
            PiiType::Iban => "IBAN",
            PiiType::CreditCard => "KART",
            PiiType::Phone => "TEL",
            PiiType::Email => "EMAIL",
            PiiType::Name => "AD",
            PiiType::Person => "KISI",
            PiiType::Address => "ADRES",
            PiiType::Passport => "PASAPORT",
            PiiType::DriverLicense => "EHLIYET",
            PiiType::Plate => "PLAKA",
            PiiType::Vin => "SASI_NO",
            PiiType::Imei => "IMEI",
            PiiType::IpAddress => "IP_ADRES",
            PiiType::Port => "PORT",
            PiiType::Age => "YAS",
            PiiType::AgeRange => "YAS_ARALIGI",
            PiiType::BirthDate => "DOGUM_TARIHI",
            PiiType::PrivateDate => "OZEL_TARIH",
            PiiType::PrivateUrl => "OZEL_URL",
            PiiType::BloodType => "KAN_GRUBU",
            PiiType::Health => "SAGLIK",
            PiiType::Gender => "CINSIYET",
            PiiType::Salary => "MAAS",
            PiiType::Credentials => "SIFRE",
            PiiType::AccountNo => "HESAP_NO",
            PiiType::Ssn => "SSN",
        }
    }

    pub fn from_name(s: &str) -> Option<Self> {
        let normalized = s.trim().to_uppercase();
        match normalized.as_str() {
            "TCKN" => Some(PiiType::Tckn),
            "VKN" => Some(PiiType::Vkn),
            "IBAN" => Some(PiiType::Iban),
            "KART" | "CREDITCARD" | "CREDIT_CARD" => Some(PiiType::CreditCard),
            "TEL" | "PHONE" => Some(PiiType::Phone),
            "EMAIL" | "E_MAIL" | "EPOSTA" | "E-POSTA" => Some(PiiType::Email),
            "AD" | "NAME" => Some(PiiType::Name),
            "KISI" | "PERSON" => Some(PiiType::Person),
            "ADRES" | "ADDRESS" => Some(PiiType::Address),
            "PASAPORT" | "PASSPORT" => Some(PiiType::Passport),
            "EHLIYET" | "DRIVER_LICENSE" | "DRIVERLICENSE" => Some(PiiType::DriverLicense),
            "PLAKA" | "PLATE" => Some(PiiType::Plate),
            "SASI_NO" | "VIN" => Some(PiiType::Vin),
            "IMEI" => Some(PiiType::Imei),
            "IP_ADRES" | "IP" | "IPADDRESS" | "IP_ADDRESS" => Some(PiiType::IpAddress),
            "PORT" => Some(PiiType::Port),
            "YAS" | "AGE" => Some(PiiType::Age),
            "YAS_ARALIGI" | "AGE_RANGE" | "AGERANGE" => Some(PiiType::AgeRange),
            "DOGUM_TARIHI" | "BIRTH_DATE" | "BIRTHDATE" => Some(PiiType::BirthDate),
            "OZEL_TARIH" | "PRIVATE_DATE" | "PRIVATEDATE" => Some(PiiType::PrivateDate),
            "OZEL_URL" | "PRIVATE_URL" | "PRIVATEURL" => Some(PiiType::PrivateUrl),
            "KAN_GRUBU" | "BLOOD_TYPE" | "BLOODTYPE" => Some(PiiType::BloodType),
            "SAGLIK" | "HEALTH" => Some(PiiType::Health),
            "CINSIYET" | "GENDER" => Some(PiiType::Gender),
            "MAAS" | "SALARY" => Some(PiiType::Salary),
            "SIFRE" | "CREDENTIALS" | "CREDENTIAL" | "SECRET" => Some(PiiType::Credentials),
            "HESAP_NO" | "ACCOUNT_NO" | "ACCOUNTNO" => Some(PiiType::AccountNo),
            "SSN" => Some(PiiType::Ssn),
            _ => None,
        }
    }
}

/// A detected PII span in Turkish text.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PiiEntity {
    pub text: String,
    pub label: String,
    pub pii_type: PiiType,
    pub start: usize,
    pub end: usize,
    pub confidence: f32,
    pub stem: String,
    pub suffix: Option<String>,
}

/// Result of PII detection and masking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PiiResult {
    pub original_text: String,
    pub masked_text: String,
    pub entities: Vec<PiiEntity>,
    /// Plain mapping of { placeholder_or_surrogate: original_text } for downstream clients
    pub mapping: HashMap<String, String>,
    pub vault: PiiVault,
}

lazy_static::lazy_static! {
    static ref TCKN_CANDIDATE_REGEX: regex::Regex = regex::Regex::new(r"\b\d{11}\b").unwrap();
    static ref GROUPED_TCKN_REGEX: regex::Regex = regex::Regex::new(r"\b(?:\d{3}[- ]\d{3}[- ]\d{3}[- ]\d{2}|\d{4}[- ]\d{4}[- ]\d{3})\b").unwrap();
    static ref TCKN_TRIGGER_REGEX: regex::Regex = regex::Regex::new(r"(?i)\b(?:tc|tckn|tc\s*no|tc\s*kimlik|kimlik\s*no|tc\s*si)[:\s]*([0-9- ]{11,16})\b").unwrap();
    static ref VKN_CANDIDATE_REGEX: regex::Regex = regex::Regex::new(r"\b\d{10}\b").unwrap();
    static ref IBAN_CANDIDATE_REGEX: regex::Regex = regex::Regex::new(r"(?i)\b[A-Z]{2}\d{2}(?:[0-9A-Z]{11,30}|(?:\s+[0-9A-Z]{1,4}){3,8})\b").unwrap();
    static ref IBAN_TRIGGER_REGEX: regex::Regex = regex::Regex::new(r"(?i)\b(?:iban|iban\s*no)[:\s]*([A-Z]{2}\d{2}(?:[0-9A-Z]{11,30}|(?:\s+[0-9A-Z]{1,4}){3,8}))\b").unwrap();
    static ref SSN_CANDIDATE_REGEX: regex::Regex = regex::Regex::new(r"\b\d{3}-\d{2}-\d{4}\b").unwrap();
    static ref CARD_CANDIDATE_REGEX: regex::Regex = regex::Regex::new(r"\b(?:\d[ -]?){13,19}\b").unwrap();
    static ref CARD_TRIGGER_REGEX: regex::Regex = regex::Regex::new(r"(?i)\b(?:kart\s*no|kredi\s*kart[ıi]|kart)[:\s]*((?:\d[ -]?){13,19})\b").unwrap();
    static ref PLATE_CANDIDATE_REGEX: regex::Regex = regex::Regex::new(r"(?i)\b\d{2}\s*[A-ZÇĞİÖŞÜ]{1,3}\s*\d{2,5}\b").unwrap();
    static ref VIN_CANDIDATE_REGEX: regex::Regex = regex::Regex::new(r"\b[A-HJ-NPR-Z0-9]{17}\b").unwrap();
    static ref ADDRESS_REGEX: regex::Regex = regex::Regex::new(
        r"(?ix)
        \b[A-ZÇĞİÖŞÜ][a-zçğıöşü]+\s+(?:Mahallesi|Mah\.|Mah)\b
        (?:[,\s]+[A-ZÇĞİÖŞÜ][a-zçğıöşü\s]+(?:Caddesi|Cad\.|Cad|Sokağı|Sok\.|Sok|Bulvarı|Bulv\.|Bulv))?
        (?:[,\s]+(?:No|Kapı\s*No|Daire|Kat)[:\s]*[0-9/A-Za-z]+)?
        (?:[,\s]+[A-ZÇĞİÖŞÜ][a-zçğıöşü]+(?:\s*/\s*[A-ZÇĞİÖŞÜ][a-zçğıöşü]+)?)?
        "
    ).unwrap();
}

/// High-performance Turkish PII Detection and Pseudonymization Engine.
pub struct TurkishPiiEngine {
    embedding_scorer: Option<PiiEmbeddingScorer>,
    preserve_corporate_emails: bool,
    disabled_types: std::collections::HashSet<PiiType>,
}

impl Default for TurkishPiiEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl TurkishPiiEngine {
    /// Creates a new PII engine without embedding scorer (pure rule/checksum/morphology mode).
    pub fn new() -> Self {
        Self {
            embedding_scorer: None,
            preserve_corporate_emails: true,
            disabled_types: std::collections::HashSet::new(),
        }
    }

    /// Creates a new PII engine initialized with Turkish embeddings for semantic prototype matching.
    pub fn with_embeddings(
        embeddings: std::sync::Arc<crate::embeddings::TurkishEmbeddings>,
    ) -> Self {
        Self {
            embedding_scorer: Some(PiiEmbeddingScorer::new(embeddings)),
            preserve_corporate_emails: true,
            disabled_types: std::collections::HashSet::new(),
        }
    }

    /// Disables detection and masking for a specific `PiiType`.
    /// When disabled, spans of this type are left verbatim in detect, mask, and vault.
    pub fn disable_type(&mut self, pii_type: PiiType) -> &mut Self {
        self.disabled_types.insert(pii_type);
        self
    }

    /// Enables detection and masking for a specific `PiiType`.
    pub fn enable_type(&mut self, pii_type: PiiType) -> &mut Self {
        self.disabled_types.remove(&pii_type);
        self
    }

    /// Sets whether a specific `PiiType` is enabled or disabled.
    pub fn set_type_enabled(&mut self, pii_type: PiiType, enabled: bool) -> &mut Self {
        if enabled {
            self.disabled_types.remove(&pii_type);
        } else {
            self.disabled_types.insert(pii_type);
        }
        self
    }

    /// Returns `true` if the given `PiiType` is enabled.
    pub fn is_type_enabled(&self, pii_type: PiiType) -> bool {
        !self.disabled_types.contains(&pii_type)
    }

    /// Configures the engine with an initial set of disabled `PiiType`s.
    pub fn with_disabled_types<I: IntoIterator<Item = PiiType>>(mut self, types: I) -> Self {
        self.disabled_types = types.into_iter().collect();
        self
    }

    /// Configures whether functional non-PII corporate support emails (e.g. info@, destek@, satis@)
    /// should be preserved rather than masked as individual personal data.
    pub fn set_preserve_corporate_emails(&mut self, preserve: bool) {
        self.preserve_corporate_emails = preserve;
    }

    /// Returns a reference to the embedding scorer, if initialized.
    pub fn embedding_scorer(&self) -> Option<&PiiEmbeddingScorer> {
        self.embedding_scorer.as_ref()
    }

    /// Detects all PII entities in the given text.
    pub fn detect(&self, text: &str) -> Vec<PiiEntity> {
        let mut candidates = Vec::new();

        // 1. Algorithmic Checksum Detections
        self.detect_checksum_entities(text, &mut candidates);

        // 2. Pattern and Regex Detections (Phone, Email, IP, Port, Age, SSN, Triggers)
        self.detect_pattern_entities(text, &mut candidates);

        // 3. Spelled-Out Numbers (Sözle Yazılmış Değerler)
        detect_spelled_numbers(text, &mut candidates);

        // 4. Secrets, Passwords, OTPs & API Tokens
        if self.is_type_enabled(PiiType::Credentials) {
            detect_secrets(text, &mut candidates);
        }

        // 5. Morphological & Lexical Person Name Detection
        if self.is_type_enabled(PiiType::Name) || self.is_type_enabled(PiiType::Person) {
            self.detect_names_and_persons(text, &mut candidates);
        }

        // 6. Address Detection
        if self.is_type_enabled(PiiType::Address) {
            self.detect_addresses(text, &mut candidates);
        }

        // 7. KVKK Article 6 Sensitive Data (Blood type, Health, Religion)
        if self.is_type_enabled(PiiType::BloodType)
            || self.is_type_enabled(PiiType::Health)
            || self.is_type_enabled(PiiType::Gender)
        {
            self.detect_sensitive_categories(text, &mut candidates);
        }

        // Retain only entities whose PiiType is currently enabled
        candidates.retain(|e| self.is_type_enabled(e.pii_type));

        // 8. Conflict Resolution & Non-Overlapping Span Arbitration
        resolve_conflicts(candidates)
    }

    /// Intercepts outbound prompt, detects PII, and returns masked text with a freshly populated session vault.
    pub fn mask(&self, text: &str, mode: PiiMode) -> PiiResult {
        let mut vault = PiiVault::new();
        self.mask_with_vault(text, mode, &mut vault)
    }

    /// Intercepts outbound prompt, detects PII, and masks text using an existing, persistent session vault.
    /// Reuses existing placeholders for recurrent values and assigns sequential counters across conversation turns.
    pub fn mask_with_vault(&self, text: &str, mode: PiiMode, vault: &mut PiiVault) -> PiiResult {
        let entities = self.detect(text);
        let mut masked = String::with_capacity(text.len());
        let mut last_end = 0;

        for entity in &entities {
            if entity.start < last_end {
                continue; // Skip overlapping spans already handled
            }
            masked.push_str(&text[last_end..entity.start]);

            let placeholder = vault.register(
                &entity.text,
                entity.pii_type.as_str(),
                &entity.stem,
                entity.suffix.as_deref(),
                mode,
            );
            masked.push_str(&placeholder);

            last_end = entity.end;
        }

        if last_end < text.len() {
            masked.push_str(&text[last_end..]);
        }

        let mapping = vault.get_mapping();

        PiiResult {
            original_text: text.to_string(),
            masked_text: masked,
            entities,
            mapping,
            vault: vault.clone(),
        }
    }

    /// Inbound gateway response restoration: replaces placeholders in LLM response with original PII values.
    pub fn restore_response(&self, llm_response: &str, vault: &PiiVault) -> String {
        vault.restore(llm_response)
    }

    /// Restores placeholders in an LLM response using a plain mapping { placeholder: original_text }.
    pub fn restore_with_mapping(
        &self,
        llm_response: &str,
        mapping: &HashMap<String, String>,
    ) -> String {
        PiiVault::restore_with_mapping(llm_response, mapping)
    }

    // --- Private Extraction Steps ---

    fn detect_checksum_entities(&self, text: &str, out: &mut Vec<PiiEntity>) {
        let has_digits = text.bytes().any(|b| b.is_ascii_digit());
        if !has_digits {
            return;
        }

        // TCKN candidates (11-digit numbers)
        for mat in TCKN_CANDIDATE_REGEX.find_iter(text) {
            let span_str = mat.as_str();
            if validate_tckn(span_str) {
                out.push(PiiEntity {
                    text: span_str.to_string(),
                    label: PiiType::Tckn.as_str().to_string(),
                    pii_type: PiiType::Tckn,
                    start: mat.start(),
                    end: mat.end(),
                    confidence: 1.0,
                    stem: span_str.to_string(),
                    suffix: None,
                });
            }
        }

        // Grouped TCKN candidates (e.g. 645-175-336-66, 3782 5513 262)
        for mat in GROUPED_TCKN_REGEX.find_iter(text) {
            let span_str = mat.as_str();
            let clean_digits: String = span_str.chars().filter(|c| c.is_ascii_digit()).collect();
            if clean_digits.len() == 11 && validate_tckn(&clean_digits) {
                out.push(PiiEntity {
                    text: span_str.to_string(),
                    label: PiiType::Tckn.as_str().to_string(),
                    pii_type: PiiType::Tckn,
                    start: mat.start(),
                    end: mat.end(),
                    confidence: 0.99,
                    stem: span_str.to_string(),
                    suffix: None,
                });
            }
        }

        // Trigger-preceded TCKN (e.g. TC Sİ 645-175-336-66, tc 12345678901)
        let lower = to_turkish_lower(text);
        if lower.sz_find("tc").is_some() || lower.sz_find("kimlik").is_some() {
            for cap in TCKN_TRIGGER_REGEX.captures_iter(text) {
                if let Some(val) = cap.get(1) {
                    let span_str = val.as_str().trim();
                    let clean_digits: String =
                        span_str.chars().filter(|c| c.is_ascii_digit()).collect();
                    if clean_digits.len() == 11 {
                        let conf = if validate_tckn(&clean_digits) {
                            1.0
                        } else {
                            0.95
                        };
                        out.push(PiiEntity {
                            text: span_str.to_string(),
                            label: PiiType::Tckn.as_str().to_string(),
                            pii_type: PiiType::Tckn,
                            start: val.start(),
                            end: val.end(),
                            confidence: conf,
                            stem: span_str.to_string(),
                            suffix: None,
                        });
                    }
                }
            }
        }

        // VKN candidates (10-digit numbers)
        for mat in VKN_CANDIDATE_REGEX.find_iter(text) {
            let span_str = mat.as_str();
            if validate_vkn(span_str) {
                out.push(PiiEntity {
                    text: span_str.to_string(),
                    label: PiiType::Vkn.as_str().to_string(),
                    pii_type: PiiType::Vkn,
                    start: mat.start(),
                    end: mat.end(),
                    confidence: 1.0,
                    stem: span_str.to_string(),
                    suffix: None,
                });
            }
        }

        // IBAN candidates (any country with or without spaces, strictly verified via ISO 13616 & MOD 97)
        for mat in IBAN_CANDIDATE_REGEX.find_iter(text) {
            let matched_str = mat.as_str();
            let mut curr = matched_str.trim();
            loop {
                if validate_iban(curr) {
                    let start = mat.start();
                    let end = start + curr.len();
                    out.push(PiiEntity {
                        text: curr.to_string(),
                        label: PiiType::Iban.as_str().to_string(),
                        pii_type: PiiType::Iban,
                        start,
                        end,
                        confidence: 1.0,
                        stem: curr.to_string(),
                        suffix: None,
                    });
                    break;
                }
                if let Some(last_space_idx) = curr.rfind(|c: char| c.is_whitespace()) {
                    curr = curr[..last_space_idx].trim_end();
                    if curr.len() < 15 {
                        break;
                    }
                } else {
                    break;
                }
            }
        }

        // Trigger-preceded IBAN
        if lower.sz_find("iban").is_some() {
            for cap in IBAN_TRIGGER_REGEX.captures_iter(text) {
                if let Some(val) = cap.get(1) {
                    let mut curr = val.as_str().trim();
                    loop {
                        if validate_iban(curr) {
                            let start = val.start();
                            let end = start + curr.len();
                            out.push(PiiEntity {
                                text: curr.to_string(),
                                label: PiiType::Iban.as_str().to_string(),
                                pii_type: PiiType::Iban,
                                start,
                                end,
                                confidence: 1.0,
                                stem: curr.to_string(),
                                suffix: None,
                            });
                            break;
                        }
                        if let Some(last_space_idx) = curr.rfind(|c: char| c.is_whitespace()) {
                            curr = curr[..last_space_idx].trim_end();
                            if curr.len() < 15 {
                                break;
                            }
                        } else {
                            break;
                        }
                    }
                }
            }
        }

        // Trigger-preceded Vehicle Plates
        if lower.sz_find("plaka").is_some() {
            for cap in PLATE_TRIGGER_REGEX.captures_iter(text) {
                if let Some(val) = cap.get(1) {
                    let span_str = val.as_str().trim();
                    out.push(PiiEntity {
                        text: span_str.to_string(),
                        label: PiiType::Plate.as_str().to_string(),
                        pii_type: PiiType::Plate,
                        start: val.start(),
                        end: val.end(),
                        confidence: 0.98,
                        stem: span_str.to_string(),
                        suffix: None,
                    });
                }
            }
        }

        // Credit/Debit Card candidates (13-19 digits, possibly space/dash grouped)
        for mat in CARD_CANDIDATE_REGEX.find_iter(text) {
            let span_str = mat.as_str();
            if let Some(_brand) = validate_credit_card(span_str) {
                out.push(PiiEntity {
                    text: span_str.to_string(),
                    label: PiiType::CreditCard.as_str().to_string(),
                    pii_type: PiiType::CreditCard,
                    start: mat.start(),
                    end: mat.end(),
                    confidence: 1.0,
                    stem: span_str.to_string(),
                    suffix: None,
                });
            }
        }

        // Trigger-preceded Card numbers (e.g. kart no 4147462686590570, kart 4500591135632828)
        if lower.sz_find("kart").is_some() {
            for cap in CARD_TRIGGER_REGEX.captures_iter(text) {
                if let Some(val) = cap.get(1) {
                    let span_str = val.as_str().trim();
                    let clean_digits: String =
                        span_str.chars().filter(|c| c.is_ascii_digit()).collect();
                    if (13..=19).contains(&clean_digits.len()) {
                        let conf = if validate_credit_card(span_str).is_some() {
                            1.0
                        } else {
                            0.95
                        };
                        out.push(PiiEntity {
                            text: span_str.to_string(),
                            label: PiiType::CreditCard.as_str().to_string(),
                            pii_type: PiiType::CreditCard,
                            start: val.start(),
                            end: val.end(),
                            confidence: conf,
                            stem: span_str.to_string(),
                            suffix: None,
                        });
                    }
                }
            }
        }

        // Vehicle Plate candidates (e.g. 34 ABC 123)
        for mat in PLATE_CANDIDATE_REGEX.find_iter(text) {
            let span_str = mat.as_str();
            if validate_plate(span_str) {
                out.push(PiiEntity {
                    text: span_str.to_string(),
                    label: PiiType::Plate.as_str().to_string(),
                    pii_type: PiiType::Plate,
                    start: mat.start(),
                    end: mat.end(),
                    confidence: 0.98,
                    stem: span_str.to_string(),
                    suffix: None,
                });
            }
        }

        // VIN candidates (17 alphanumeric)
        for mat in VIN_CANDIDATE_REGEX.find_iter(text) {
            let span_str = mat.as_str();
            if validate_vin(span_str) {
                out.push(PiiEntity {
                    text: span_str.to_string(),
                    label: PiiType::Vin.as_str().to_string(),
                    pii_type: PiiType::Vin,
                    start: mat.start(),
                    end: mat.end(),
                    confidence: 1.0,
                    stem: span_str.to_string(),
                    suffix: None,
                });
            }
        }
    }

    fn detect_pattern_entities(&self, text: &str, out: &mut Vec<PiiEntity>) {
        let has_digits = text.bytes().any(|b| b.is_ascii_digit());
        let lower = to_turkish_lower(text);

        // Phone numbers
        if has_digits {
            for mat in PHONE_REGEX.find_iter(text) {
                let span_str = mat.as_str();
                out.push(PiiEntity {
                    text: span_str.to_string(),
                    label: PiiType::Phone.as_str().to_string(),
                    pii_type: PiiType::Phone,
                    start: mat.start(),
                    end: mat.end(),
                    confidence: 0.98,
                    stem: span_str.to_string(),
                    suffix: None,
                });
            }
        }

        // Private URLs (URLs with sensitive query parameters, auth tokens, or private paths)
        if text.sz_find("http://").is_some() || text.sz_find("https://").is_some() {
            for mat in PRIVATE_URL_REGEX.find_iter(text) {
                let span_str = mat.as_str();
                out.push(PiiEntity {
                    text: span_str.to_string(),
                    label: PiiType::PrivateUrl.as_str().to_string(),
                    pii_type: PiiType::PrivateUrl,
                    start: mat.start(),
                    end: mat.end(),
                    confidence: 0.98,
                    stem: span_str.to_string(),
                    suffix: None,
                });
            }
            for cap in PRIVATE_URL_TRIGGER_REGEX.captures_iter(text) {
                let val = cap.get(1).unwrap();
                let span_str = val.as_str();
                out.push(PiiEntity {
                    text: span_str.to_string(),
                    label: PiiType::PrivateUrl.as_str().to_string(),
                    pii_type: PiiType::PrivateUrl,
                    start: val.start(),
                    end: val.end(),
                    confidence: 0.98,
                    stem: span_str.to_string(),
                    suffix: None,
                });
            }
        }

        // Email addresses (standard + obfuscated, preserving corporate support emails when configured)
        if text.sz_find("@").is_some()
            || lower.sz_find("[at]").is_some()
            || lower.sz_find("(at)").is_some()
        {
            for mat in EMAIL_REGEX.find_iter(text) {
                let span_str = mat.as_str();
                if self.preserve_corporate_emails && is_corporate_email(span_str) {
                    continue;
                }
                out.push(PiiEntity {
                    text: span_str.to_string(),
                    label: PiiType::Email.as_str().to_string(),
                    pii_type: PiiType::Email,
                    start: mat.start(),
                    end: mat.end(),
                    confidence: 1.0,
                    stem: span_str.to_string(),
                    suffix: None,
                });
            }
            for mat in EMAIL_OBFUSCATED_REGEX.find_iter(text) {
                let span_str = mat.as_str();
                if self.preserve_corporate_emails && is_corporate_email(span_str) {
                    continue;
                }
                out.push(PiiEntity {
                    text: span_str.to_string(),
                    label: PiiType::Email.as_str().to_string(),
                    pii_type: PiiType::Email,
                    start: mat.start(),
                    end: mat.end(),
                    confidence: 0.95,
                    stem: span_str.to_string(),
                    suffix: None,
                });
            }
        }

        // IP addresses
        if has_digits && text.sz_find(".").is_some() {
            for mat in IPV4_REGEX.find_iter(text) {
                let span_str = mat.as_str();
                out.push(PiiEntity {
                    text: span_str.to_string(),
                    label: PiiType::IpAddress.as_str().to_string(),
                    pii_type: PiiType::IpAddress,
                    start: mat.start(),
                    end: mat.end(),
                    confidence: 0.95,
                    stem: span_str.to_string(),
                    suffix: None,
                });
            }
        }

        // Port numbers
        if has_digits && (lower.sz_find("port").is_some() || text.sz_find(":").is_some()) {
            for cap in PORT_REGEX.captures_iter(text) {
                let full = cap.get(0).unwrap();
                let port_str = cap.get(1).or_else(|| cap.get(2)).unwrap().as_str();
                if let Ok(p) = port_str.parse::<u32>() {
                    if (1..=65535).contains(&p) {
                        out.push(PiiEntity {
                            text: port_str.to_string(),
                            label: PiiType::Port.as_str().to_string(),
                            pii_type: PiiType::Port,
                            start: full.start(),
                            end: full.end(),
                            confidence: 0.95,
                            stem: port_str.to_string(),
                            suffix: None,
                        });
                    }
                }
            }
        }

        // Passport numbers
        if has_digits
            && (lower.sz_find("pasaport").is_some()
                || text.sz_find("U").is_some()
                || text.sz_find("u").is_some()
                || text.sz_find("A").is_some()
                || text.sz_find("a").is_some()
                || text.sz_find("EP").is_some()
                || text.sz_find("ep").is_some())
        {
            for cap in PASSPORT_REGEX.captures_iter(text) {
                let full = cap.get(0).unwrap();
                let val = cap.get(1).unwrap().as_str();
                out.push(PiiEntity {
                    text: val.to_string(),
                    label: PiiType::Passport.as_str().to_string(),
                    pii_type: PiiType::Passport,
                    start: full.start(),
                    end: full.end(),
                    confidence: 0.95,
                    stem: val.to_string(),
                    suffix: None,
                });
            }
        }

        // Age & Age range
        if has_digits && (lower.sz_find("yaş").is_some() || lower.sz_find("yas").is_some()) {
            for cap in AGE_REGEX.captures_iter(text) {
                let val = cap.get(1).unwrap();
                out.push(PiiEntity {
                    text: val.as_str().to_string(),
                    label: PiiType::Age.as_str().to_string(),
                    pii_type: PiiType::Age,
                    start: val.start(),
                    end: val.end(),
                    confidence: 0.95,
                    stem: val.as_str().to_string(),
                    suffix: None,
                });
            }
            for mat in AGE_RANGE_REGEX.find_iter(text) {
                out.push(PiiEntity {
                    text: mat.as_str().to_string(),
                    label: PiiType::AgeRange.as_str().to_string(),
                    pii_type: PiiType::AgeRange,
                    start: mat.start(),
                    end: mat.end(),
                    confidence: 0.95,
                    stem: mat.as_str().to_string(),
                    suffix: None,
                });
            }
        }

        // CVV / CVC
        if has_digits
            && (lower.sz_find("cvv").is_some()
                || lower.sz_find("cvc").is_some()
                || lower.sz_find("güvenlik").is_some()
                || lower.sz_find("guvenlik").is_some())
        {
            for cap in CVV_REGEX.captures_iter(text) {
                let val = cap.get(1).unwrap();
                out.push(PiiEntity {
                    text: val.as_str().to_string(),
                    label: "CVV".to_string(),
                    pii_type: PiiType::CreditCard,
                    start: val.start(),
                    end: val.end(),
                    confidence: 0.98,
                    stem: val.as_str().to_string(),
                    suffix: None,
                });
            }
        }

        // Card Expiry (SKT)
        if has_digits
            && (lower.sz_find("skt").is_some()
                || lower.sz_find("son kullanma").is_some()
                || lower.sz_find("exp").is_some())
        {
            for cap in CARD_EXPIRY_REGEX.captures_iter(text) {
                let full = cap.get(0).unwrap();
                out.push(PiiEntity {
                    text: full.as_str().to_string(),
                    label: "KART_SKT".to_string(),
                    pii_type: PiiType::CreditCard,
                    start: full.start(),
                    end: full.end(),
                    confidence: 0.95,
                    stem: full.as_str().to_string(),
                    suffix: None,
                });
            }
        }

        // Account / Customer No
        if lower.sz_find("müşteri").is_some()
            || lower.sz_find("musteri").is_some()
            || lower.sz_find("hesap").is_some()
            || lower.sz_find("ödeme").is_some()
            || lower.sz_find("odeme").is_some()
        {
            for cap in ACCOUNT_NO_REGEX.captures_iter(text) {
                if let Some(val) = cap.get(1) {
                    out.push(PiiEntity {
                        text: val.as_str().to_string(),
                        label: "MUSTERI_NO".to_string(),
                        pii_type: PiiType::AccountNo,
                        start: val.start(),
                        end: val.end(),
                        confidence: 0.95,
                        stem: val.as_str().to_string(),
                        suffix: None,
                    });
                } else if let Some(val) = cap.get(2).or_else(|| cap.get(3)) {
                    out.push(PiiEntity {
                        text: val.as_str().to_string(),
                        label: "HESAP_NO".to_string(),
                        pii_type: PiiType::AccountNo,
                        start: val.start(),
                        end: val.end(),
                        confidence: 0.95,
                        stem: val.as_str().to_string(),
                        suffix: None,
                    });
                }
            }
        }

        // Bank account 4-7-3 format (e.g. 5427-5551073-146)
        if has_digits && text.sz_find("-").is_some() {
            for mat in BANK_ACCOUNT_FORMAT_REGEX.find_iter(text) {
                let span_str = mat.as_str();
                out.push(PiiEntity {
                    text: span_str.to_string(),
                    label: "HESAP_NO".to_string(),
                    pii_type: PiiType::AccountNo,
                    start: mat.start(),
                    end: mat.end(),
                    confidence: 0.94,
                    stem: span_str.to_string(),
                    suffix: None,
                });
            }
        }

        // US Social Security Number (hyphenated form only: 000-00-0000)
        if has_digits && text.sz_find("-").is_some() {
            for mat in SSN_CANDIDATE_REGEX.find_iter(text) {
                let span_str = mat.as_str();
                if validate_ssn(span_str) {
                    out.push(PiiEntity {
                        text: span_str.to_string(),
                        label: PiiType::Ssn.as_str().to_string(),
                        pii_type: PiiType::Ssn,
                        start: mat.start(),
                        end: mat.end(),
                        confidence: 1.0,
                        stem: span_str.to_string(),
                        suffix: None,
                    });
                }
            }
        }

        // Tax ID (VKN / Vergi No)
        if has_digits && (lower.sz_find("vergi").is_some() || lower.sz_find("vkn").is_some()) {
            for cap in VERGI_NO_REGEX.captures_iter(text) {
                let val = cap.get(1).unwrap();
                out.push(PiiEntity {
                    text: val.as_str().to_string(),
                    label: "VERGI_NO".to_string(),
                    pii_type: PiiType::Tckn,
                    start: val.start(),
                    end: val.end(),
                    confidence: 0.98,
                    stem: val.as_str().to_string(),
                    suffix: None,
                });
            }
        }

        // Disability Status (KVKK Article 6)
        if lower.sz_find("engel").is_some()
            || lower.sz_find("özür").is_some()
            || lower.sz_find("ozur").is_some()
        {
            for cap in DISABILITY_REGEX.captures_iter(text) {
                let val = cap.get(1).unwrap();
                let span_str = val.as_str().trim();
                if !span_str.is_empty() {
                    out.push(PiiEntity {
                        text: span_str.to_string(),
                        label: "ENGEL_DURUMU".to_string(),
                        pii_type: PiiType::Health,
                        start: val.start(),
                        end: val.end(),
                        confidence: 0.98,
                        stem: span_str.to_string(),
                        suffix: None,
                    });
                }
            }
        }

        // Credentials / Passwords
        if lower.sz_find("şifre").is_some()
            || lower.sz_find("sifre").is_some()
            || lower.sz_find("parola").is_some()
            || lower.sz_find("pin").is_some()
        {
            for cap in CREDENTIALS_REGEX.captures_iter(text) {
                let val = cap.get(1).unwrap();
                let val_lower = val.as_str().to_lowercase();
                if matches!(
                    val_lower.as_str(),
                    "sıfırlama"
                        | "sifirlama"
                        | "değiştirme"
                        | "degistirme"
                        | "güncelleme"
                        | "guncelleme"
                        | "işlemi"
                        | "islemi"
                        | "talebi"
                        | "isteği"
                        | "hatası"
                        | "ekranı"
                ) {
                    continue;
                }
                out.push(PiiEntity {
                    text: val.as_str().to_string(),
                    label: PiiType::Credentials.as_str().to_string(),
                    pii_type: PiiType::Credentials,
                    start: val.start(),
                    end: val.end(),
                    confidence: 0.95,
                    stem: val.as_str().to_string(),
                    suffix: None,
                });
            }
        }

        // Salary / Income
        if has_digits
            && (lower.sz_find("maaş").is_some()
                || lower.sz_find("maas").is_some()
                || lower.sz_find("gelir").is_some()
                || lower.sz_find("ücret").is_some()
                || lower.sz_find("ucret").is_some())
        {
            for cap in SALARY_REGEX.captures_iter(text) {
                let full = cap.get(0).unwrap();
                out.push(PiiEntity {
                    text: full.as_str().to_string(),
                    label: PiiType::Salary.as_str().to_string(),
                    pii_type: PiiType::Salary,
                    start: full.start(),
                    end: full.end(),
                    confidence: 0.95,
                    stem: full.as_str().to_string(),
                    suffix: None,
                });
            }
        }

        // Driver's License
        if lower.sz_find("ehliyet").is_some()
            || lower.sz_find("sürücü").is_some()
            || lower.sz_find("surucu").is_some()
            || lower.sz_find("şoför").is_some()
            || lower.sz_find("sofor").is_some()
        {
            for cap in DRIVER_LICENSE_REGEX.captures_iter(text) {
                let val = cap.get(1).unwrap();
                out.push(PiiEntity {
                    text: val.as_str().to_string(),
                    label: PiiType::DriverLicense.as_str().to_string(),
                    pii_type: PiiType::DriverLicense,
                    start: val.start(),
                    end: val.end(),
                    confidence: 0.95,
                    stem: val.as_str().to_string(),
                    suffix: None,
                });
            }
        }

        // SGK No
        if has_digits
            && (lower.sz_find("sgk").is_some()
                || lower.sz_find("ssk").is_some()
                || lower.sz_find("bağkur").is_some()
                || lower.sz_find("bagkur").is_some())
        {
            for cap in SGK_NO_REGEX.captures_iter(text) {
                let val = cap.get(1).unwrap();
                out.push(PiiEntity {
                    text: val.as_str().to_string(),
                    label: "SGK_NO".to_string(),
                    pii_type: PiiType::Tckn,
                    start: val.start(),
                    end: val.end(),
                    confidence: 0.95,
                    stem: val.as_str().to_string(),
                    suffix: None,
                });
            }
        }

        // Positive Date of birth (explicit birth triggers)
        if has_digits && (lower.sz_find("doğ").is_some() || lower.sz_find("dog").is_some()) {
            for cap in BIRTH_DATE_POSITIVE_REGEX.captures_iter(text) {
                let date_match = cap.get(1).or_else(|| cap.get(4)).or_else(|| cap.get(0));
                if let Some(mat) = date_match {
                    out.push(PiiEntity {
                        text: mat.as_str().to_string(),
                        label: PiiType::BirthDate.as_str().to_string(),
                        pii_type: PiiType::BirthDate,
                        start: mat.start(),
                        end: mat.end(),
                        confidence: 0.98,
                        stem: mat.as_str().to_string(),
                        suffix: None,
                    });
                }
            }
        }

        // Positive Private Dates (fatura kesim, randevu, teslimat, mezuniyet, abonelik, etc.)
        const DATE_TRIGGERS: &[&str] = &[
            "fatura",
            "ödeme",
            "odeme",
            "itiraz",
            "iptal",
            "servis",
            "abonelik",
            "üyelik",
            "uyelik",
            "randevu",
            "kurulum",
            "muayene",
            "teslim",
            "rezervasyon",
            "işlem",
            "islem",
            "mezuniyet",
            "sınav",
            "sinav",
            "kayıt",
            "kayit",
            "üniversite",
            "universite",
            "işe",
            "ise",
            "işten",
            "isten",
            "sözleşme",
            "sozlesme",
            "başvuru",
            "basvuru",
            "talep",
            "doğum",
            "dogum",
            "tarih",
            "günü",
            "gunu",
        ];
        if has_digits
            && DATE_TRIGGERS
                .iter()
                .any(|&trig| lower.sz_find(trig).is_some())
        {
            for cap in PRIVATE_DATE_TRIGGER_REGEX.captures_iter(text) {
                if let Some(mat) = cap.get(1) {
                    out.push(PiiEntity {
                        text: mat.as_str().to_string(),
                        label: "OZEL_TARIH".to_string(),
                        pii_type: PiiType::PrivateDate,
                        start: mat.start(),
                        end: mat.end(),
                        confidence: 0.98,
                        stem: mat.as_str().to_string(),
                        suffix: None,
                    });
                }
            }
            if lower.sz_find("olan").is_some() {
                for cap in PRIVATE_DATE_POST_TRIGGER_REGEX.captures_iter(text) {
                    if let Some(mat) = cap.get(1) {
                        out.push(PiiEntity {
                            text: mat.as_str().to_string(),
                            label: "OZEL_TARIH".to_string(),
                            pii_type: PiiType::PrivateDate,
                            start: mat.start(),
                            end: mat.end(),
                            confidence: 0.98,
                            stem: mat.as_str().to_string(),
                            suffix: None,
                        });
                    }
                }
            }
        }

        // Semantic embedding disambiguation for candidate dates (when embeddings are initialized)
        if has_digits {
            if let Some(scorer) = self.embedding_scorer.as_ref() {
                for mat in TURKISH_DATE_CANDIDATE_REGEX.find_iter(text) {
                    let ctx_start = if mat.start() >= 40 {
                        mat.start() - 40
                    } else {
                        0
                    };
                    let ctx_end = std::cmp::min(text.len(), mat.end() + 40);
                    let ctx = &text[ctx_start..ctx_end];
                    if scorer.is_private_date_context(ctx) {
                        out.push(PiiEntity {
                            text: mat.as_str().to_string(),
                            label: PiiType::BirthDate.as_str().to_string(),
                            pii_type: PiiType::BirthDate,
                            start: mat.start(),
                            end: mat.end(),
                            confidence: 0.90,
                            stem: mat.as_str().to_string(),
                            suffix: None,
                        });
                    }
                }
                if text.sz_find(".").is_some()
                    || text.sz_find("/").is_some()
                    || text.sz_find("-").is_some()
                {
                    for mat in NUMERIC_DATE_CANDIDATE_REGEX.find_iter(text) {
                        let ctx_start = if mat.start() >= 40 {
                            mat.start() - 40
                        } else {
                            0
                        };
                        let ctx_end = std::cmp::min(text.len(), mat.end() + 40);
                        let ctx = &text[ctx_start..ctx_end];
                        if scorer.is_private_date_context(ctx) {
                            out.push(PiiEntity {
                                text: mat.as_str().to_string(),
                                label: PiiType::BirthDate.as_str().to_string(),
                                pii_type: PiiType::BirthDate,
                                start: mat.start(),
                                end: mat.end(),
                                confidence: 0.90,
                                stem: mat.as_str().to_string(),
                                suffix: None,
                            });
                        }
                    }
                }
            }
        }

        // IMEI device numbers
        if has_digits && (lower.sz_find("imei").is_some() || text.len() >= 15) {
            for cap in IMEI_REGEX.captures_iter(text) {
                let val = cap.get(1).unwrap();
                let span_str = val.as_str();
                let conf = if validate_imei(span_str) { 1.0 } else { 0.95 };
                out.push(PiiEntity {
                    text: span_str.to_string(),
                    label: "IMEI".to_string(),
                    pii_type: PiiType::Imei,
                    start: val.start(),
                    end: val.end(),
                    confidence: conf,
                    stem: span_str.to_string(),
                    suffix: None,
                });
            }
        }

        // MAC Addresses
        if text.sz_find(":").is_some() || text.sz_find("-").is_some() {
            for cap in MAC_ADDRESS_REGEX.captures_iter(text) {
                let val = cap.get(1).unwrap();
                out.push(PiiEntity {
                    text: val.as_str().to_string(),
                    label: "MAC_ADRES".to_string(),
                    pii_type: PiiType::IpAddress,
                    start: val.start(),
                    end: val.end(),
                    confidence: 0.98,
                    stem: val.as_str().to_string(),
                    suffix: None,
                });
            }
        }

        // Cryptocurrency Wallets
        if text.sz_find("bc1").is_some()
            || text.sz_find("0x").is_some()
            || text.sz_find("0X").is_some()
            || text.sz_find("T").is_some()
        {
            for mat in CRYPTO_WALLET_REGEX.find_iter(text) {
                out.push(PiiEntity {
                    text: mat.as_str().to_string(),
                    label: "KRIPTO_CUZDAN".to_string(),
                    pii_type: PiiType::AccountNo,
                    start: mat.start(),
                    end: mat.end(),
                    confidence: 0.98,
                    stem: mat.as_str().to_string(),
                    suffix: None,
                });
            }
        }

        // Vehicle Registration (Ruhsat No)
        if lower.sz_find("ruhsat").is_some() {
            for cap in RUHSAT_NO_REGEX.captures_iter(text) {
                let val = cap.get(1).unwrap();
                out.push(PiiEntity {
                    text: val.as_str().to_string(),
                    label: "RUHSAT_NO".to_string(),
                    pii_type: PiiType::Plate,
                    start: val.start(),
                    end: val.end(),
                    confidence: 0.95,
                    stem: val.as_str().to_string(),
                    suffix: None,
                });
            }
        }

        // Contract No
        if lower.sz_find("sözleşme").is_some() || lower.sz_find("sozlesme").is_some() {
            for cap in CONTRACT_NO_REGEX.captures_iter(text) {
                let val = cap.get(1).unwrap();
                out.push(PiiEntity {
                    text: val.as_str().to_string(),
                    label: "SOZLESME_NO".to_string(),
                    pii_type: PiiType::AccountNo,
                    start: val.start(),
                    end: val.end(),
                    confidence: 0.95,
                    stem: val.as_str().to_string(),
                    suffix: None,
                });
            }
        }

        // Policy No
        if lower.sz_find("poliçe").is_some() || lower.sz_find("police").is_some() {
            for cap in POLICY_NO_REGEX.captures_iter(text) {
                let val = cap.get(1).unwrap();
                out.push(PiiEntity {
                    text: val.as_str().to_string(),
                    label: "POLICE_NO".to_string(),
                    pii_type: PiiType::AccountNo,
                    start: val.start(),
                    end: val.end(),
                    confidence: 0.95,
                    stem: val.as_str().to_string(),
                    suffix: None,
                });
            }
        }

        // Reference No
        if lower.sz_find("referans").is_some()
            || text.sz_find("REF-").is_some()
            || text.sz_find("RF").is_some()
        {
            for cap in REFERENCE_NO_REGEX.captures_iter(text) {
                let val = cap.get(1).unwrap();
                out.push(PiiEntity {
                    text: val.as_str().to_string(),
                    label: "REFERANS".to_string(),
                    pii_type: PiiType::AccountNo,
                    start: val.start(),
                    end: val.end(),
                    confidence: 0.95,
                    stem: val.as_str().to_string(),
                    suffix: None,
                });
            }
        }

        // Vehicle Chassis / VIN No in context
        if lower.sz_find("şasi").is_some()
            || lower.sz_find("sasi").is_some()
            || lower.sz_find("vin").is_some()
        {
            for cap in CHASSIS_NO_REGEX.captures_iter(text) {
                let val = cap.get(1).unwrap();
                out.push(PiiEntity {
                    text: val.as_str().to_string(),
                    label: "SASI_NO".to_string(),
                    pii_type: PiiType::Vin,
                    start: val.start(),
                    end: val.end(),
                    confidence: 0.98,
                    stem: val.as_str().to_string(),
                    suffix: None,
                });
            }
        }

        // Engine / Motor No in context
        if lower.sz_find("motor").is_some() {
            for cap in MOTOR_NO_REGEX.captures_iter(text) {
                let val = cap.get(1).unwrap();
                out.push(PiiEntity {
                    text: val.as_str().to_string(),
                    label: "MOTOR_NO".to_string(),
                    pii_type: PiiType::Plate,
                    start: val.start(),
                    end: val.end(),
                    confidence: 0.95,
                    stem: val.as_str().to_string(),
                    suffix: None,
                });
            }
        }

        // PIN in context
        if has_digits && lower.sz_find("pin").is_some() {
            for cap in PIN_REGEX.captures_iter(text) {
                let val = cap.get(1).unwrap();
                out.push(PiiEntity {
                    text: val.as_str().to_string(),
                    label: "PIN".to_string(),
                    pii_type: PiiType::Credentials,
                    start: val.start(),
                    end: val.end(),
                    confidence: 0.95,
                    stem: val.as_str().to_string(),
                    suffix: None,
                });
            }
        }

        // Credit rating / Findeks score
        if has_digits && (lower.sz_find("kredi").is_some() || lower.sz_find("findeks").is_some()) {
            for cap in CREDIT_SCORE_REGEX.captures_iter(text) {
                let val = cap.get(1).unwrap();
                out.push(PiiEntity {
                    text: val.as_str().to_string(),
                    label: "KREDI_NOTU".to_string(),
                    pii_type: PiiType::AccountNo,
                    start: val.start(),
                    end: val.end(),
                    confidence: 0.95,
                    stem: val.as_str().to_string(),
                    suffix: None,
                });
            }
        }

        // Mother's maiden name
        if lower.sz_find("kızlık").is_some() || lower.sz_find("kizlik").is_some() {
            for cap in MOTHER_MAIDEN_REGEX.captures_iter(text) {
                let val = cap.get(1).unwrap();
                out.push(PiiEntity {
                    text: val.as_str().to_string(),
                    label: "ANNE_KIZLIK".to_string(),
                    pii_type: PiiType::Name,
                    start: val.start(),
                    end: val.end(),
                    confidence: 0.95,
                    stem: val.as_str().to_string(),
                    suffix: None,
                });
            }
        }

        // Gender
        if lower.sz_find("cinsiyet").is_some()
            || lower.sz_find("kadın").is_some()
            || lower.sz_find("kadin").is_some()
            || lower.sz_find("erkek").is_some()
            || lower.sz_find("bayan").is_some()
            || lower.sz_find("dişi").is_some()
            || lower.sz_find("disi").is_some()
        {
            for cap in GENDER_REGEX.captures_iter(text) {
                if let Some(val) = cap.get(1).or_else(|| cap.get(2)) {
                    out.push(PiiEntity {
                        text: val.as_str().to_string(),
                        label: "CINSIYET".to_string(),
                        pii_type: PiiType::Gender,
                        start: val.start(),
                        end: val.end(),
                        confidence: 0.95,
                        stem: val.as_str().to_string(),
                        suffix: None,
                    });
                }
            }
        }

        // Device ID
        if lower.sz_find("cihaz").is_some() || lower.sz_find("device").is_some() {
            for cap in DEVICE_ID_REGEX.captures_iter(text) {
                let val = cap.get(1).unwrap();
                out.push(PiiEntity {
                    text: val.as_str().to_string(),
                    label: "CIHAZ_ID".to_string(),
                    pii_type: PiiType::AccountNo,
                    start: val.start(),
                    end: val.end(),
                    confidence: 0.95,
                    stem: val.as_str().to_string(),
                    suffix: None,
                });
            }
        }

        // Sicil No
        if lower.sz_find("sicil").is_some() {
            for cap in SICIL_NO_REGEX.captures_iter(text) {
                let val = cap.get(1).unwrap();
                out.push(PiiEntity {
                    text: val.as_str().to_string(),
                    label: "SICIL_NO".to_string(),
                    pii_type: PiiType::Tckn,
                    start: val.start(),
                    end: val.end(),
                    confidence: 0.95,
                    stem: val.as_str().to_string(),
                    suffix: None,
                });
            }
        }

        // Digital Signature / E-Signature
        if lower.sz_find("imza").is_some() {
            for mat in SIGNATURE_REGEX.find_iter(text) {
                let span_str = mat.as_str();
                out.push(PiiEntity {
                    text: span_str.to_string(),
                    label: "IMZA".to_string(),
                    pii_type: PiiType::Credentials,
                    start: mat.start(),
                    end: mat.end(),
                    confidence: 0.98,
                    stem: span_str.to_string(),
                    suffix: None,
                });
            }
        }

        // Criminal / Judicial record
        if lower.sz_find("ceza").is_some()
            || lower.sz_find("kabahat").is_some()
            || lower.sz_find("serbestlik").is_some()
            || lower.sz_find("icra").is_some()
            || lower.sz_find("sicil").is_some()
            || lower.sz_find("sabıka").is_some()
            || lower.sz_find("sabika").is_some()
        {
            for mat in CRIMINAL_RECORD_REGEX.find_iter(text) {
                let span_str = mat.as_str();
                out.push(PiiEntity {
                    text: span_str.to_string(),
                    label: "CEZA_KAYDI".to_string(),
                    pii_type: PiiType::Health,
                    start: mat.start(),
                    end: mat.end(),
                    confidence: 0.98,
                    stem: span_str.to_string(),
                    suffix: None,
                });
            }
        }

        // Nationality
        if lower.sz_find("uyruk").is_some()
            || lower.sz_find("uyruğu").is_some()
            || lower.sz_find("uyrugu").is_some()
            || lower.sz_find("vatandaş").is_some()
            || lower.sz_find("vatandas").is_some()
        {
            for cap in NATIONALITY_REGEX.captures_iter(text) {
                let val = cap.get(1).unwrap();
                out.push(PiiEntity {
                    text: val.as_str().to_string(),
                    label: "UYRUK".to_string(),
                    pii_type: PiiType::Person,
                    start: val.start(),
                    end: val.end(),
                    confidence: 0.95,
                    stem: val.as_str().to_string(),
                    suffix: None,
                });
            }
        }

        // Geolocation coordinates
        if has_digits && text.sz_find(".").is_some() {
            for mat in COORDINATES_REGEX.find_iter(text) {
                let span_str = mat.as_str();
                out.push(PiiEntity {
                    text: span_str.to_string(),
                    label: "KONUM".to_string(),
                    pii_type: PiiType::Address,
                    start: mat.start(),
                    end: mat.end(),
                    confidence: 0.98,
                    stem: span_str.to_string(),
                    suffix: None,
                });
            }
        }

        // Health record
        if lower.sz_find("sağlık").is_some()
            || lower.sz_find("saglik").is_some()
            || lower.sz_find("rahatsızlık").is_some()
            || lower.sz_find("rahatsizlik").is_some()
            || lower.sz_find("kronik").is_some()
            || lower.sz_find("engel").is_some()
        {
            for cap in HEALTH_RECORD_REGEX.captures_iter(text) {
                let val = cap.get(1).unwrap();
                let span_str = val.as_str().trim();
                if !span_str.is_empty() {
                    out.push(PiiEntity {
                        text: span_str.to_string(),
                        label: "SAGLIK".to_string(),
                        pii_type: PiiType::Health,
                        start: val.start(),
                        end: val.end(),
                        confidence: 0.95,
                        stem: span_str.to_string(),
                        suffix: None,
                    });
                }
            }
        }

        // Trade union membership
        if lower.sz_find("sendika").is_some()
            || lower.sz_find("sendikası").is_some()
            || lower.sz_find("sendikasi").is_some()
            || lower.sz_find("-iş").is_some()
            || lower.sz_find("-is").is_some()
            || lower.sz_find("sen ").is_some()
            || lower.sz_find("sen\n").is_some()
        {
            for mat in TRADE_UNION_REGEX.find_iter(text) {
                let span_str = mat.as_str();
                out.push(PiiEntity {
                    text: span_str.to_string(),
                    label: "SENDIKA".to_string(),
                    pii_type: PiiType::Person,
                    start: mat.start(),
                    end: mat.end(),
                    confidence: 0.98,
                    stem: span_str.to_string(),
                    suffix: None,
                });
            }
        }

        // Username
        if lower.sz_find("kullanıcı").is_some()
            || lower.sz_find("kullanici").is_some()
            || lower.sz_find("username").is_some()
            || lower.sz_find("oturum").is_some()
        {
            for cap in USERNAME_REGEX.captures_iter(text) {
                if let Some(val) = cap.get(1).or_else(|| cap.get(2)) {
                    out.push(PiiEntity {
                        text: val.as_str().to_string(),
                        label: "KULLANICI_ADI".to_string(),
                        pii_type: PiiType::AccountNo,
                        start: val.start(),
                        end: val.end(),
                        confidence: 0.95,
                        stem: val.as_str().to_string(),
                        suffix: None,
                    });
                }
            }
        }

        // Biometric data
        if lower.sz_find("damar").is_some()
            || lower.sz_find("parmak").is_some()
            || lower.sz_find("yüz").is_some()
            || lower.sz_find("yuz").is_some()
            || lower.sz_find("iris").is_some()
            || lower.sz_find("retina").is_some()
            || lower.sz_find("biyometrik").is_some()
        {
            for mat in BIOMETRIC_REGEX.find_iter(text) {
                let span_str = mat.as_str();
                out.push(PiiEntity {
                    text: span_str.to_string(),
                    label: "BIYOMETRIK".to_string(),
                    pii_type: PiiType::Health,
                    start: mat.start(),
                    end: mat.end(),
                    confidence: 0.98,
                    stem: span_str.to_string(),
                    suffix: None,
                });
            }
        }

        // Birthplace
        if lower.sz_find("doğum").is_some() || lower.sz_find("dogum").is_some() {
            for cap in BIRTHPLACE_REGEX.captures_iter(text) {
                let val = cap.get(1).unwrap();
                out.push(PiiEntity {
                    text: val.as_str().to_string(),
                    label: "DOGUM_YERI".to_string(),
                    pii_type: PiiType::Address,
                    start: val.start(),
                    end: val.end(),
                    confidence: 0.95,
                    stem: val.as_str().to_string(),
                    suffix: None,
                });
            }
        }

        // Family status
        if lower.sz_find("aile").is_some() || lower.sz_find("medeni").is_some() {
            for cap in FAMILY_STATUS_REGEX.captures_iter(text) {
                let val = cap.get(1).unwrap();
                let span_str = val.as_str().trim();
                if !span_str.is_empty() {
                    out.push(PiiEntity {
                        text: span_str.to_string(),
                        label: "AILE".to_string(),
                        pii_type: PiiType::Person,
                        start: val.start(),
                        end: val.end(),
                        confidence: 0.95,
                        stem: span_str.to_string(),
                        suffix: None,
                    });
                }
            }
        }

        // Ethnic origin
        if lower.sz_find("etnik").is_some() {
            for cap in ETHNIC_ORIGIN_REGEX.captures_iter(text) {
                let val = cap.get(1).unwrap();
                out.push(PiiEntity {
                    text: val.as_str().to_string(),
                    label: "ETNIK_KOKEN".to_string(),
                    pii_type: PiiType::Person,
                    start: val.start(),
                    end: val.end(),
                    confidence: 0.95,
                    stem: val.as_str().to_string(),
                    suffix: None,
                });
            }
        }

        // Workplace / Institution
        const WORKPLACE_TRIGGERS: &[&str] = &[
            "işyeri",
            "isyeri",
            "kurum",
            "şirket",
            "sirket",
            "hastane",
            "hastanes",
            "belediye",
            "belediyes",
            "müdürlük",
            "müdürlüğ",
            "mudurluk",
            "mudurlug",
            "bakanlık",
            "bakanlığ",
            "bakanlik",
            "bakanlig",
            "a.ş",
            "aş",
            "ltd",
            "üniversite",
            "universite",
            "holding",
        ];
        if WORKPLACE_TRIGGERS
            .iter()
            .any(|&trig| lower.sz_find(trig).is_some())
        {
            for mat in WORKPLACE_REGEX.find_iter(text) {
                let span_str = mat.as_str();
                out.push(PiiEntity {
                    text: span_str.to_string(),
                    label: "ISYERI".to_string(),
                    pii_type: PiiType::Address,
                    start: mat.start(),
                    end: mat.end(),
                    confidence: 0.95,
                    stem: span_str.to_string(),
                    suffix: None,
                });
            }
        }

        // Mother / Father given names
        if lower.sz_find("anne").is_some() || lower.sz_find("baba").is_some() {
            for cap in PARENT_NAMES_REGEX.captures_iter(text) {
                let full_match = cap.get(0).unwrap().as_str().to_lowercase();
                let val = cap.get(1).unwrap();
                let label = if full_match.starts_with("anne") {
                    "ANNE_ADI"
                } else {
                    "BABA_ADI"
                };
                out.push(PiiEntity {
                    text: val.as_str().to_string(),
                    label: label.to_string(),
                    pii_type: PiiType::Name,
                    start: val.start(),
                    end: val.end(),
                    confidence: 0.95,
                    stem: val.as_str().to_string(),
                    suffix: None,
                });
            }
        }
    }

    fn detect_names_and_persons(&self, text: &str, out: &mut Vec<PiiEntity>) {
        let tokens = TurkishTokenizer::tokenize(text);
        let n = tokens.len();
        let mut i = 0;

        while i < n {
            let tok = &tokens[i];
            let clean_tok = tok
                .text
                .trim_matches(|c: char| !c.is_alphabetic() && c != '\'' && c != '’');
            if clean_tok.is_empty() {
                i += 1;
                continue;
            }

            let stemmed = split_stem_suffix(clean_tok, tok.start);
            let lower_stem = to_turkish_lower(stemmed.stem);

            // Check honorific title preceding: e.g. `Dr. Ayşe`, `Sayın Ahmet Bey`
            let is_title = TITLE_TRIGGERS.contains(lower_stem.as_str());
            let preceded_by_title = i > 0
                && TITLE_TRIGGERS.contains(
                    to_turkish_lower(
                        tokens[i - 1]
                            .text
                            .trim_matches(|c: char| !c.is_alphabetic()),
                    )
                    .as_str(),
                );
            let followed_by_title = i + 1 < n
                && TITLE_TRIGGERS.contains(
                    to_turkish_lower(
                        tokens[i + 1]
                            .text
                            .trim_matches(|c: char| !c.is_alphabetic()),
                    )
                    .as_str(),
                );

            let is_in_given = GIVEN_NAMES.contains(lower_stem.as_str());
            let _is_in_surnames = SURNAMES.contains(lower_stem.as_str());
            let is_polysemous = POLYSEMOUS_NAMES.contains(lower_stem.as_str());

            let is_capitalized = stemmed
                .stem
                .chars()
                .next()
                .is_some_and(|c| c.is_uppercase());

            // Polysemy Guard: If name is polysemous (e.g. Deniz, Barış, Gül), require capitalization
            // and either a title, another capitalized name following (e.g. Deniz Yılmaz), or clear proper noun context
            let mut name_confirmed = false;
            let mut span_start = tok.start;
            let mut span_end = stemmed.stem_end;
            let mut span_text = stemmed.stem.to_string();

            if is_title && i + 1 < n {
                // Preceded by title / role trigger: e.g. `Dr. Ayşe`, `Sayın Ahmet Yılmaz`, `Müşteri Caner Çetin`
                let next_tok = &tokens[i + 1];
                let next_stemmed = split_stem_suffix(next_tok.text, next_tok.start);
                if next_stemmed
                    .stem
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_uppercase())
                {
                    name_confirmed = true;
                    // The title itself remains untouched, the name span starts at the first name token
                    span_start = next_tok.start;
                    span_end = next_stemmed.stem_end;
                    span_text = next_stemmed.stem.to_string();
                    // Check if second name / surname follows: e.g. `Müşteri Caner Çetin`
                    if i + 2 < n
                        && tokens[i + 2]
                            .text
                            .chars()
                            .next()
                            .is_some_and(|c| c.is_uppercase())
                    {
                        let third_stemmed =
                            split_stem_suffix(tokens[i + 2].text, tokens[i + 2].start);
                        span_end = third_stemmed.stem_end;
                        span_text = format!("{} {}", span_text, third_stemmed.stem);
                        i += 1;
                    }
                    i += 1;
                }
            } else if is_capitalized {
                if is_polysemous {
                    if preceded_by_title || followed_by_title {
                        name_confirmed = true;
                    } else if i + 1 < n
                        && tokens[i + 1]
                            .text
                            .chars()
                            .next()
                            .is_some_and(|c| c.is_uppercase())
                    {
                        // `Deniz Kaya`
                        let next_stemmed =
                            split_stem_suffix(tokens[i + 1].text, tokens[i + 1].start);
                        let next_lower = to_turkish_lower(next_stemmed.stem);
                        if SURNAMES.contains(next_lower.as_str())
                            || GIVEN_NAMES.contains(next_lower.as_str())
                        {
                            name_confirmed = true;
                            span_end = next_stemmed.stem_end;
                            span_text = format!("{} {}", stemmed.stem, next_stemmed.stem);
                            i += 1;
                        }
                    } else if let Some(scorer) = self.embedding_scorer.as_ref() {
                        let ctx_start = span_start.saturating_sub(40);
                        let ctx_end = std::cmp::min(text.len(), span_end + 40);
                        // Exclude the ambiguous word itself so its literal noun sense does not bias the context embedding
                        let surrounding = format!(
                            "{} {}",
                            &text[ctx_start..span_start],
                            &text[span_end..ctx_end]
                        );
                        if scorer.is_person_context(surrounding.trim()) {
                            name_confirmed = true;
                        }
                    }
                } else if is_in_given {
                    // Non-polysemous given name (e.g. Mehmet, Mustafa, Ahmet, Zeynep, Fatma, Caner)
                    name_confirmed = true;
                    // Check if followed by surname
                    if i + 1 < n
                        && tokens[i + 1]
                            .text
                            .chars()
                            .next()
                            .is_some_and(|c| c.is_uppercase())
                    {
                        let next_stemmed =
                            split_stem_suffix(tokens[i + 1].text, tokens[i + 1].start);
                        let next_lower = to_turkish_lower(next_stemmed.stem);
                        if SURNAMES.contains(next_lower.as_str())
                            || GIVEN_NAMES.contains(next_lower.as_str())
                            || preceded_by_title
                        {
                            span_end = next_stemmed.stem_end;
                            span_text = format!("{} {}", stemmed.stem, next_stemmed.stem);
                            i += 1;
                        }
                    }
                } else if i + 1 < n
                    && tokens[i + 1]
                        .text
                        .chars()
                        .next()
                        .is_some_and(|c| c.is_uppercase())
                {
                    // Two consecutive capitalized words where word 2 is in SURNAMES (e.g. Caner Çetin, Ada Demir)
                    let next_stemmed = split_stem_suffix(tokens[i + 1].text, tokens[i + 1].start);
                    let next_lower = to_turkish_lower(next_stemmed.stem);
                    if SURNAMES.contains(next_lower.as_str()) {
                        name_confirmed = true;
                        span_end = next_stemmed.stem_end;
                        span_text = format!("{} {}", stemmed.stem, next_stemmed.stem);
                        i += 1;
                    }
                }
            }

            if name_confirmed {
                out.push(PiiEntity {
                    text: span_text.clone(),
                    label: PiiType::Name.as_str().to_string(),
                    pii_type: PiiType::Name,
                    start: span_start,
                    end: span_end,
                    confidence: 0.95,
                    stem: span_text,
                    suffix: stemmed.suffix.map(|s| s.to_string()),
                });
            }

            i += 1;
        }
    }

    fn detect_addresses(&self, text: &str, out: &mut Vec<PiiEntity>) {
        for mat in ADDRESS_REGEX.find_iter(text) {
            let span_str = mat.as_str().trim();
            out.push(PiiEntity {
                text: span_str.to_string(),
                label: PiiType::Address.as_str().to_string(),
                pii_type: PiiType::Address,
                start: mat.start(),
                end: mat.end(),
                confidence: 0.90,
                stem: span_str.to_string(),
                suffix: None,
            });
        }
    }

    fn detect_sensitive_categories(&self, text: &str, out: &mut Vec<PiiEntity>) {
        let lower = to_turkish_lower(text);

        // Blood types
        for bt in BLOOD_TYPES.iter() {
            if let Some(pos) = lower.sz_find(bt.as_str()) {
                out.push(PiiEntity {
                    text: text[pos..pos + bt.len()].to_string(),
                    label: PiiType::BloodType.as_str().to_string(),
                    pii_type: PiiType::BloodType,
                    start: pos,
                    end: pos + bt.len(),
                    confidence: 0.98,
                    stem: text[pos..pos + bt.len()].to_string(),
                    suffix: None,
                });
            }
        }

        // Health terms
        for ht in HEALTH_TERMS.iter() {
            let mut start_idx = 0;
            while let Some(pos) = lower[start_idx..].sz_find(ht.as_str()) {
                let actual_pos = start_idx + pos;
                let end_pos = actual_pos + ht.len();
                out.push(PiiEntity {
                    text: text[actual_pos..end_pos].to_string(),
                    label: PiiType::Health.as_str().to_string(),
                    pii_type: PiiType::Health,
                    start: actual_pos,
                    end: end_pos,
                    confidence: 0.95,
                    stem: text[actual_pos..end_pos].to_string(),
                    suffix: None,
                });
                start_idx = end_pos;
            }
        }

        // Religion terms
        for rt in RELIGION_TERMS.iter() {
            if let Some(pos) = lower.sz_find(rt.as_str()) {
                out.push(PiiEntity {
                    text: text[pos..pos + rt.len()].to_string(),
                    label: "DIN".to_string(),
                    pii_type: PiiType::Health, // Sensitive KVKK 6
                    start: pos,
                    end: pos + rt.len(),
                    confidence: 0.90,
                    stem: text[pos..pos + rt.len()].to_string(),
                    suffix: None,
                });
            }
        }
        // Clinical trigger semantic classification for unknown health conditions
        if let Some(scorer) = self.embedding_scorer.as_ref() {
            static CLINICAL_TRIGGERS: &[&str] = &[
                "tanısı",
                "tanisi",
                "teşhisi",
                "teshisi",
                "tedavisi",
                "ameliyatı",
                "ameliyati",
                "hastalığı",
                "hastaligi",
                "hastası",
                "hastasi",
            ];
            for trig in CLINICAL_TRIGGERS {
                let mut start_idx = 0;
                while let Some(pos) = lower[start_idx..].sz_find(trig) {
                    let actual_pos = start_idx + pos;
                    let window_start = actual_pos.saturating_sub(40);
                    let candidate_context = text[window_start..actual_pos].trim();
                    if let Some(last_word) = candidate_context.split_whitespace().last() {
                        let clean_word = last_word.trim_matches(|c: char| !c.is_alphabetic());
                        if clean_word.len() >= 4
                            && scorer.score_similarity(clean_word, PiiPrototypeCategory::Health)
                                >= 0.25
                        {
                            if let Some(word_pos) =
                                text[window_start..actual_pos].sz_rfind(clean_word)
                            {
                                let w_start = window_start + word_pos;
                                let w_end = w_start + clean_word.len();
                                out.push(PiiEntity {
                                    text: clean_word.to_string(),
                                    label: PiiType::Health.as_str().to_string(),
                                    pii_type: PiiType::Health,
                                    start: w_start,
                                    end: w_end,
                                    confidence: 0.90,
                                    stem: clean_word.to_string(),
                                    suffix: None,
                                });
                            }
                        }
                    }
                    start_idx = actual_pos + trig.len();
                }
            }
        }
    }
}

/// Resolves overlapping spans, keeping the longest, most specific, or highest confidence match.
fn resolve_conflicts(mut candidates: Vec<PiiEntity>) -> Vec<PiiEntity> {
    if candidates.is_empty() {
        return candidates;
    }

    // Sort by start position ascending, then length descending
    candidates.sort_by(|a, b| {
        if a.start != b.start {
            a.start.cmp(&b.start)
        } else {
            (b.end - b.start).cmp(&(a.end - a.start))
        }
    });

    let mut resolved: Vec<PiiEntity> = Vec::with_capacity(candidates.len());
    for cand in candidates {
        if let Some(last) = resolved.last_mut() {
            // Overlap condition
            if cand.start < last.end {
                // If candidate has strictly higher confidence, replace
                if cand.confidence > last.confidence {
                    *last = cand;
                }
                continue;
            }
        }
        resolved.push(cand);
    }

    resolved
}

/// Helper function to check if an email local-part represents a public corporate or support desk.
fn is_corporate_email(email: &str) -> bool {
    let lower = email.to_lowercase();
    if let Some(at_idx) = lower.sz_find("@") {
        let prefix = &lower[..at_idx];
        if crate::pii::gazetteer::CORPORATE_EMAIL_PREFIXES.contains(prefix) {
            return true;
        }
        let norm_prefix = prefix.replace(['.', '_', '-'], "");
        if crate::pii::gazetteer::CORPORATE_EMAIL_PREFIXES.contains(&norm_prefix) {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_tckn_and_phone_masking() {
        let engine = TurkishPiiEngine::new();
        let prompt = "Müşteri Ahmet Yılmaz TC: 10000000146, tel: 0532 123 45 67.";
        let res = engine.mask(prompt, PiiMode::Placeholder);

        assert!(res.masked_text.contains("{{AD_1}}"));
        assert!(res.masked_text.contains("{{TCKN_1}}"));
        assert!(res.masked_text.contains("{{TEL_1}}"));
        assert!(!res.masked_text.contains("10000000146"));
        assert!(!res.masked_text.contains("0532 123 45 67"));

        // Test gateway response restoration
        let llm_reply = "{{AD_1}} isimli müşterinin {{TEL_1}} nolu hattına SMS iletildi.";
        let restored = engine.restore_response(llm_reply, &res.vault);
        assert_eq!(
            restored,
            "Ahmet Yılmaz isimli müşterinin 0532 123 45 67 nolu hattına SMS iletildi."
        );
    }

    #[test]
    fn test_polysemy_disambiguation() {
        let engine = TurkishPiiEngine::new();

        // Polysemous name "Deniz" with title -> SHOULD BE MASKED
        let text1 = "Sayın Deniz Bey yarın toplantıya katılacak.";
        let res1 = engine.mask(text1, PiiMode::Tag);
        assert!(res1.masked_text.contains("[AD]"));

        // Polysemous word "deniz" as common noun -> SHOULD NOT BE MASKED
        let text2 = "Yaz tatilinde deniz kenarında oturduk.";
        let res2 = engine.mask(text2, PiiMode::Tag);
        assert!(!res2.masked_text.contains("[AD]"));
        assert_eq!(res2.masked_text, text2);
    }

    #[test]
    fn test_suffix_apostrophe_preservation() {
        let engine = TurkishPiiEngine::new();
        let prompt = "Ahmet'in 10000000146 nolu TCKN kaydı silindi.";
        let res = engine.mask(prompt, PiiMode::Placeholder);

        // Span for Ahmet is masked, leaving 'in outside
        assert!(res.masked_text.contains("{{AD_1}}'in"));

        // Restore
        let llm_resp = "{{AD_1}}'in kaydı başarıyla silinmiştir.";
        let restored = engine.restore_response(llm_resp, &res.vault);
        assert_eq!(restored, "Ahmet'in kaydı başarıyla silinmiştir.");
    }

    #[test]
    fn test_mapping_export_and_restore_with_mapping() {
        let engine = TurkishPiiEngine::new();
        let prompt = "Müşteri Mehmet Öztürk, TC: 10000000146, tel: 0532 123 45 67.";
        let res = engine.mask(prompt, PiiMode::Placeholder);

        assert!(!res.mapping.is_empty());
        assert_eq!(res.mapping.get("{{AD_1}}").unwrap(), "Mehmet Öztürk");
        assert_eq!(res.mapping.get("{{TCKN_1}}").unwrap(), "10000000146");

        // Restore using just the mapping (simulating client-side or gateway service without PiiVault instance)
        let llm_resp = "Sayın {{AD_1}}, {{TCKN_1}} nolu başvurunuz onaylandı.";
        let restored = engine.restore_with_mapping(llm_resp, &res.mapping);
        assert_eq!(
            restored,
            "Sayın Mehmet Öztürk, 10000000146 nolu başvurunuz onaylandı."
        );
    }

    #[test]
    fn test_private_dates_vs_public_dates() {
        let engine = TurkishPiiEngine::new();

        // 1. Positive private dates MUST be masked
        let priv1 = "Abonelik başlangıcım 23 ocak görünmesine rağmen bildirim gelmedi";
        let res1 = engine.mask(priv1, PiiMode::Tag);
        assert!(
            res1.masked_text.contains("[OZEL_TARIH]")
                || res1.masked_text.contains("[DOGUM_TARIHI]")
        );

        let priv2 =
            "talep ekranında fatura kesim tarihim: 01.10.2025 yazıyor fakat işlem ilerlemiyor";
        let res2 = engine.mask(priv2, PiiMode::Tag);
        assert!(
            res2.masked_text.contains("[OZEL_TARIH]")
                || res2.masked_text.contains("[DOGUM_TARIHI]")
        );

        let priv3 = "randevu günüm 2 mayıs 2011 görünmesine rağmen bildirim gelmedi";
        let res3 = engine.mask(priv3, PiiMode::Tag);
        assert!(
            res3.masked_text.contains("[OZEL_TARIH]")
                || res3.masked_text.contains("[DOGUM_TARIHI]")
        );

        let priv4 = "2 Ekim 2006'da doğanlara özel tanımlansın";
        let res4 = engine.mask(priv4, PiiMode::Tag);
        assert!(res4.masked_text.contains("[DOGUM_TARIHI]"));

        // 2. Public calendar dates & campaign validity dates MUST NOT be masked
        let pub1 = "yılbaşı fırsatı kodu 31.12.2026 ye kadar geçerli";
        let res_pub1 = engine.mask(pub1, PiiMode::Tag);
        assert_eq!(res_pub1.masked_text, pub1);

        let pub2 = "30 ağustos duyurusu 30 ağustos 2026 tarihinde başlayacak";
        let res_pub2 = engine.mask(pub2, PiiMode::Tag);
        assert_eq!(res_pub2.masked_text, pub2);
    }

    #[test]
    fn test_private_urls() {
        let engine = TurkishPiiEngine::new();

        // URL with private token/session query param
        let text1 = "dosyanıza https://portal.example.org/dosya/658033-bulutbugun linkinden erişebilirsiniz";
        let res1 = engine.mask(text1, PiiMode::Tag);
        assert!(res1.masked_text.contains("[OZEL_URL]"));

        // Contextual link trigger
        let text2 = "özel dosya linki https://portal.example.org/paylas/mor-bugun/3035";
        let res2 = engine.mask(text2, PiiMode::Tag);
        assert!(res2.masked_text.contains("[OZEL_URL]"));

        // Public URL without token/private route should not be masked
        let pub_url = "daha fazla bilgi için https://google.com adresini ziyaret ediniz";
        let res_pub = engine.mask(pub_url, PiiMode::Tag);
        assert_eq!(res_pub.masked_text, pub_url);
    }

    #[test]
    fn test_corporate_email_preservation() {
        let mut engine = TurkishPiiEngine::new();
        engine.set_preserve_corporate_emails(true);

        let text = "hesabım derin.demir@example.net ile açılmış, satis@hotelgo.net adresine yazdım ama destek@otel.com yanıt vermedi";
        let res = engine.mask(text, PiiMode::Placeholder);

        // Personal email is masked
        assert!(res.masked_text.contains("{{EMAIL_1}}"));
        assert!(!res.masked_text.contains("derin.demir@example.net"));

        // Corporate functional emails are preserved
        assert!(res.masked_text.contains("satis@hotelgo.net"));
        assert!(res.masked_text.contains("destek@otel.com"));
    }

    #[test]
    fn test_embedding_polysemy_disambiguation() {
        let emb = std::sync::Arc::new(crate::embeddings::TurkishEmbeddings::new());
        let engine = TurkishPiiEngine::with_embeddings(emb);

        // Polysemous word "Deniz" in human action context -> detected as person
        let text1 = "Deniz dün ofise geç geldi ve raporu teslim etti.";
        let res1 = engine.mask(text1, PiiMode::Tag);
        assert!(res1.masked_text.contains("[AD]"));

        // Polysemous word "deniz" in nature context -> NOT detected
        let text2 = "Yaz tatilinde deniz kenarında yürüyüş yaptık.";
        let res2 = engine.mask(text2, PiiMode::Tag);
        assert!(!res2.masked_text.contains("[AD]"));
    }

    #[test]
    fn test_simd_fast_path_non_numeric_and_triggers() {
        let engine = TurkishPiiEngine::new();

        // 1. Text with zero digits: SIMD bypasses all numeric regexes fast
        let text_no_digits = "Sayın Ahmet Bey Ankara ilindeki toplantıya katılacağını bildirdi.";
        let res1 = engine.mask(text_no_digits, PiiMode::Tag);
        assert!(res1.masked_text.contains("[AD]"));

        // 2. Text with secrets matching via StringZilla SIMD anchors
        let sk_key = ["sk-proj-", "12345678901234567890123456789012"].concat();
        let gh_tok = ["ghp_", "1234567890abcdef1234567890abcdef1234"].concat();
        let secret_text = format!("API key: {sk_key} ve token: {gh_tok}");
        let res_secret = engine.mask(&secret_text, PiiMode::Tag);
        assert!(res_secret.masked_text.contains("[SIFRE]"));

        // 3. Text with KVKK Article 6 sensitive data via StringZilla SIMD
        let sensitive_text = "Hastanın kan grubu 0 rh pozitif olup diyabet tanısı mevcuttur.";
        let res_sens = engine.mask(sensitive_text, PiiMode::Tag);
        assert!(res_sens.masked_text.contains("[KAN_GRUBU]"));
        assert!(res_sens.masked_text.contains("[SAGLIK]"));
    }

    #[test]
    fn test_mask_with_vault_conversational_continuity() {
        let engine = TurkishPiiEngine::new();
        let mut session_vault = PiiVault::new();

        // Turn 1: User introduces TCKN and a secret password
        let turn1_input =
            "Merhaba, TCKN numaram 10000000146 ve şifrem Limon004_ ile işlem yapamıyorum.";
        let res1 = engine.mask_with_vault(turn1_input, PiiMode::Placeholder, &mut session_vault);
        assert!(res1.masked_text.contains("{{TCKN_1}}"));
        assert!(res1.masked_text.contains("{{SIFRE_1}}"));

        // Simulate LLM response for Turn 1
        let llm_resp1 = "Anladım, {{TCKN_1}} nolu kullanıcımız için şifre {{SIFRE_1}} sıfırlama işlemi başlatıldı.";
        let restored1 = engine.restore_response(llm_resp1, &session_vault);
        assert!(restored1.contains("10000000146"));
        assert!(restored1.contains("Limon004_"));

        // Turn 2: User repeats the SAME TCKN, but adds a NEW IBAN
        let turn2_input = "Ayrıca 10000000146 nolu TCKN hesabıma TR33 0006 1005 1978 6457 8413 26 IBAN numaramı bağlar mısınız?";
        let res2 = engine.mask_with_vault(turn2_input, PiiMode::Placeholder, &mut session_vault);
        // The SAME TCKN must reuse {{TCKN_1}}, not create {{TCKN_2}}!
        assert!(res2.masked_text.contains("{{TCKN_1}}"));
        assert!(!res2.masked_text.contains("{{TCKN_2}}"));
        // New entity IBAN receives sequential index 1
        assert!(res2.masked_text.contains("{{IBAN_1}}"));

        // Turn 3: A different user / new TCKN is mentioned -> sequential index increments
        let turn3_input = "Eşim için de 10000000214 nolu TCKN kaydını kontrol edin.";
        let res3 = engine.mask_with_vault(turn3_input, PiiMode::Placeholder, &mut session_vault);
        assert!(res3.masked_text.contains("{{TCKN_2}}"));

        // All placeholders from across the conversation can be restored from session_vault
        let llm_resp_final = "İşlemler {{TCKN_1}}, {{IBAN_1}} ve {{TCKN_2}} için tamamlanmıştır.";
        let restored_final = engine.restore_response(llm_resp_final, &session_vault);
        assert!(restored_final.contains("10000000146"));
        assert!(restored_final.contains("TR33 0006 1005 1978 6457 8413 26"));
        assert!(restored_final.contains("10000000214"));
    }

    #[test]
    fn test_per_type_switch() {
        let mut engine = TurkishPiiEngine::new();
        let text = "Ahmet Yılmaz, 35 yaşında, ahmet@example.com ve IP adresi 192.168.1.1 üzerinden bağlandı.";

        // Default: everything is enabled
        let full_res = engine.mask(text, PiiMode::Placeholder);
        assert!(full_res.masked_text.contains("{{AD_1}}"));
        assert!(full_res.masked_text.contains("{{YAS_1}}"));
        assert!(full_res.masked_text.contains("{{EMAIL_1}}"));
        assert!(full_res.masked_text.contains("{{IP_ADRES_1}}"));

        // Disable Name and Person
        engine.disable_type(PiiType::Name);
        engine.disable_type(PiiType::Person);
        let name_disabled_res = engine.mask(text, PiiMode::Placeholder);
        assert!(!name_disabled_res.masked_text.contains("{{AD_1}}"));
        assert!(name_disabled_res.masked_text.contains("Ahmet Yılmaz")); // Left verbatim!
        assert!(!name_disabled_res
            .entities
            .iter()
            .any(|e| e.pii_type == PiiType::Name || e.pii_type == PiiType::Person));
        assert!(!name_disabled_res
            .vault
            .get_mapping()
            .values()
            .any(|v| v.contains("Ahmet")));

        // Disable Age, Email, IP as well
        engine.disable_type(PiiType::Age);
        engine.disable_type(PiiType::Email);
        engine.disable_type(PiiType::IpAddress);
        let all_disabled_res = engine.mask(text, PiiMode::Placeholder);
        assert_eq!(all_disabled_res.masked_text, text); // Completely verbatim!
        assert!(all_disabled_res.entities.is_empty());
        assert!(all_disabled_res.vault.get_mapping().is_empty());

        // Re-enable Name
        engine.enable_type(PiiType::Name);
        let re_enabled_res = engine.mask(text, PiiMode::Placeholder);
        assert!(re_enabled_res.masked_text.contains("{{AD_1}}"));
    }

    #[test]
    fn test_international_iban_detection() {
        let engine = TurkishPiiEngine::new();

        // Valid UK IBAN
        let uk_valid = "Hesap transferi için GB82WEST12345698765432 numarasını kullanınız.";
        let res_uk = engine.mask(uk_valid, PiiMode::Placeholder);
        assert!(res_uk.masked_text.contains("{{IBAN_1}}"));
        assert_eq!(res_uk.entities.len(), 1);
        assert_eq!(res_uk.entities[0].pii_type, PiiType::Iban);
        assert_eq!(res_uk.entities[0].label, "IBAN");

        // Invalid UK IBAN (changed final digit from 2 to 3) -> MUST NOT be detected!
        let uk_invalid = "Hesap transferi için GB82WEST12345698765433 numarasını kullanınız.";
        let res_uk_invalid = engine.mask(uk_invalid, PiiMode::Placeholder);
        assert!(!res_uk_invalid.masked_text.contains("{{IBAN_1}}"));
        assert!(res_uk_invalid
            .masked_text
            .contains("GB82WEST12345698765433"));
        assert!(res_uk_invalid.entities.is_empty());

        // Spaced UK IBAN
        let uk_spaced = "IBAN: GB82 WEST 1234 5698 7654 32 lütfen gönderin.";
        let res_uk_spaced = engine.mask(uk_spaced, PiiMode::Placeholder);
        assert!(res_uk_spaced.masked_text.contains("{{IBAN_1}}"));

        // Turkish IBAN stays single PiiType::Iban
        let tr_valid = "TR IBAN: TR330006100519786457841326";
        let res_tr = engine.mask(tr_valid, PiiMode::Placeholder);
        assert!(res_tr.masked_text.contains("{{IBAN_1}}"));
        assert_eq!(res_tr.entities[0].pii_type, PiiType::Iban);
    }

    #[test]
    fn test_us_ssn_detection() {
        let engine = TurkishPiiEngine::new();

        // Valid SSN: 219-09-9999 masks as {{SSN_1}} in Placeholder mode
        let valid_text = "US customer SSN is 219-09-9999 for verification.";
        let res = engine.mask(valid_text, PiiMode::Placeholder);
        assert!(res.masked_text.contains("{{SSN_1}}"));
        assert_eq!(res.entities.len(), 1);
        assert_eq!(res.entities[0].pii_type, PiiType::Ssn);
        assert_eq!(res.entities[0].label, "SSN");

        // Invalid SSNs must NOT be detected
        for invalid in [
            "000-09-9999",
            "666-09-9999",
            "950-09-9999",
            "219-00-9999",
            "219-09-0000",
        ] {
            let inv_text = format!("Testing {invalid}");
            let inv_res = engine.mask(&inv_text, PiiMode::Placeholder);
            assert!(!inv_res.masked_text.contains("{{SSN_1}}"));
            assert!(inv_res.entities.is_empty());
        }
    }
}
