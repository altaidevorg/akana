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
    static ref IBAN_CANDIDATE_REGEX: regex::Regex = regex::Regex::new(r"(?i)\bTR(?:\s*\d){24}\b").unwrap();
    static ref IBAN_TRIGGER_REGEX: regex::Regex = regex::Regex::new(r"(?i)\b(?:iban|iban\s*no)[:\s]*(TR(?:\s*[0-9]){24})\b").unwrap();
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
        }
    }

    /// Creates a new PII engine initialized with Turkish embeddings for semantic prototype matching.
    pub fn with_embeddings(
        embeddings: std::sync::Arc<crate::embeddings::TurkishEmbeddings>,
    ) -> Self {
        Self {
            embedding_scorer: Some(PiiEmbeddingScorer::new(embeddings)),
            preserve_corporate_emails: true,
        }
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

        // 2. Pattern and Regex Detections (Phone, Email, IP, Port, Age, Triggers)
        self.detect_pattern_entities(text, &mut candidates);

        // 3. Spelled-Out Numbers (Sözle Yazılmış Değerler)
        detect_spelled_numbers(text, &mut candidates);

        // 4. Secrets, Passwords, OTPs & API Tokens
        detect_secrets(text, &mut candidates);

        // 5. Morphological & Lexical Person Name Detection
        self.detect_names_and_persons(text, &mut candidates);

        // 4. Address Detection
        self.detect_addresses(text, &mut candidates);

        // 5. KVKK Article 6 Sensitive Data (Blood type, Health, Religion)
        self.detect_sensitive_categories(text, &mut candidates);

        // 6. Conflict Resolution & Non-Overlapping Span Arbitration
        resolve_conflicts(candidates)
    }

    /// Intercepts outbound prompt, detects PII, and returns masked text with a populated session vault.
    pub fn mask(&self, text: &str, mode: PiiMode) -> PiiResult {
        let entities = self.detect(text);
        let mut vault = PiiVault::new();
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
            vault,
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

        // IBAN candidates (TR + 24 digits, with or without spaces)
        for mat in IBAN_CANDIDATE_REGEX.find_iter(text) {
            let span_str = mat.as_str().trim();
            let clean_digits: String = span_str.chars().filter(|c| c.is_ascii_digit()).collect();
            if clean_digits.len() == 24 {
                let conf = if validate_iban(span_str) { 1.0 } else { 0.95 };
                out.push(PiiEntity {
                    text: span_str.to_string(),
                    label: PiiType::Iban.as_str().to_string(),
                    pii_type: PiiType::Iban,
                    start: mat.start(),
                    end: mat.end(),
                    confidence: conf,
                    stem: span_str.to_string(),
                    suffix: None,
                });
            }
        }

        // Trigger-preceded IBAN (e.g. IBAN TR0325 8143... or aidat iadesi için iban paylaşıyorum TR...)
        for cap in IBAN_TRIGGER_REGEX.captures_iter(text) {
            if let Some(val) = cap.get(1) {
                let span_str = val.as_str().trim();
                let clean_digits: String =
                    span_str.chars().filter(|c| c.is_ascii_digit()).collect();
                if clean_digits.len() == 24 {
                    let conf = if validate_iban(span_str) { 1.0 } else { 0.95 };
                    out.push(PiiEntity {
                        text: span_str.to_string(),
                        label: PiiType::Iban.as_str().to_string(),
                        pii_type: PiiType::Iban,
                        start: val.start(),
                        end: val.end(),
                        confidence: conf,
                        stem: span_str.to_string(),
                        suffix: None,
                    });
                }
            }
        }

        // Trigger-preceded Vehicle Plates
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
        // Phone numbers
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

        // Private URLs (URLs with sensitive query parameters, auth tokens, or private paths)
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

        // Email addresses (standard + obfuscated, preserving corporate support emails when configured)
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

        // IP addresses
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

        // Port numbers
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

        // Passport numbers
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

        // Age
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

        // Age range
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

        // CVV / CVC
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

        // Card Expiry (SKT)
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

        // Account / Customer No
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

        // Bank account 4-7-3 format (e.g. 5427-5551073-146)
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

        // Tax ID (VKN / Vergi No)
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

        // Disability Status (KVKK Article 6)
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

        // Credentials / Passwords
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

        // Salary / Income
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

        // Driver's License
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

        // SGK No
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

        // Positive Date of birth (explicit birth triggers)
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

        // Positive Private Dates (fatura kesim, randevu, teslimat, mezuniyet, abonelik, etc.)
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

        // Semantic embedding disambiguation for candidate dates (when embeddings are initialized)
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

        // IMEI device numbers
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

        // MAC Addresses
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

        // Cryptocurrency Wallets
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

        // Vehicle Registration (Ruhsat No)
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

        // Contract No
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

        // Policy No
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

        // Reference No
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

        // Vehicle Chassis / VIN No in context
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

        // Engine / Motor No in context
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

        // PIN in context
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

        // Credit rating / Findeks score
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

        // Mother's maiden name
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

        // Gender
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

        // Device ID
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

        // Sicil No
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

        // Digital Signature / E-Signature
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

        // Criminal / Judicial record
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

        // Nationality
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

        // Geolocation coordinates
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

        // Health record
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

        // Trade union membership
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

        // Username
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

        // Biometric data
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

        // Birthplace
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

        // Family status
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

        // Ethnic origin
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

        // Workplace / Institution
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

        // Mother / Father given names
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
            if let Some(pos) = lower.find(bt.as_str()) {
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
            while let Some(pos) = lower[start_idx..].find(ht.as_str()) {
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
            if let Some(pos) = lower.find(rt.as_str()) {
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
                while let Some(pos) = lower[start_idx..].find(trig) {
                    let actual_pos = start_idx + pos;
                    let window_start = actual_pos.saturating_sub(40);
                    let candidate_context = text[window_start..actual_pos].trim();
                    if let Some(last_word) = candidate_context.split_whitespace().last() {
                        let clean_word = last_word.trim_matches(|c: char| !c.is_alphabetic());
                        if clean_word.len() >= 4
                            && scorer.score_similarity(clean_word, PiiPrototypeCategory::Health)
                                >= 0.25
                        {
                            if let Some(word_pos) = text[window_start..actual_pos].rfind(clean_word)
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
    if let Some(at_idx) = lower.find('@') {
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
}
