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

mod addresses;
pub mod embedding_scorer;
pub mod gazetteer;
mod ids;
pub mod morphology;
mod names;
pub(crate) use names::{is_given_name, is_person_name, is_place_name};
#[rustfmt::skip]
mod names_data;
mod normalize;
pub mod passwords;
pub mod patterns;
mod phones;
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
    /// Plain mapping of { placeholder_or_surrogate: original_text } of this call's masked text,
    /// a placeholder reused from an earlier turn included; the vault holds the whole session
    /// (review 4 BIND4-06: it held every entry of the session, 20,002 for a call that masked 2).
    pub mapping: HashMap<String, String>,
    pub vault: PiiVault,
}

lazy_static::lazy_static! {
    static ref TCKN_CANDIDATE_REGEX: regex::Regex = regex::Regex::new(r"\b\d{11}\b").unwrap();
    static ref GROUPED_TCKN_REGEX: regex::Regex = regex::Regex::new(r"\b(?:\d{3}[- ]\d{3}[- ]\d{3}[- ]\d{2}|\d{4}[- ]\d{4}[- ]\d{3})\b").unwrap();
    static ref TCKN_TRIGGER_REGEX: regex::Regex = regex::Regex::new(&turkish_i_insensitive(r"(?i)\b(?:tc|tckn|tc\s*no|tc\s*kimlik|kimlik\s*no|tc\s*si)[:\s]*([0-9- ]{11,16})\b")).unwrap();
    static ref VKN_CANDIDATE_REGEX: regex::Regex = regex::Regex::new(r"\b\d{10}\b").unwrap();
    static ref IBAN_CANDIDATE_REGEX: regex::Regex = regex::Regex::new(r"(?i)\b[A-Z]{2}\d{2}(?:[0-9A-Z]{11,30}|(?:\s+[0-9A-Z]{1,4}){3,8})\b").unwrap();
    static ref IBAN_TRIGGER_REGEX: regex::Regex = regex::Regex::new(&turkish_i_insensitive(r"(?i)\b(?:iban|iban\s*no)[:\s]*([A-Z]{2}\d{2}(?:[0-9A-Z]{11,30}|(?:\s+[0-9A-Z]{1,4}){3,8}))\b")).unwrap();
    static ref SSN_CANDIDATE_REGEX: regex::Regex = regex::Regex::new(r"\b\d{3}-\d{2}-\d{4}\b").unwrap();
    static ref CARD_CANDIDATE_REGEX: regex::Regex = regex::Regex::new(r"\b(?:\d[ -]?){13,19}\b").unwrap();
    static ref CARD_TRIGGER_REGEX: regex::Regex = regex::Regex::new(&turkish_i_insensitive(r"(?i)\b(?:kart\s*no|kredi\s*kart[ıi]|kart)[:\s]*((?:\d[ -]?){13,19})\b")).unwrap();
    static ref PLATE_CANDIDATE_REGEX: regex::Regex = regex::Regex::new(r"(?i)\b\d{2}\s*[A-ZÇĞİÖŞÜ]{1,3}\s*\d{2,5}\b").unwrap();
    static ref VIN_CANDIDATE_REGEX: regex::Regex = regex::Regex::new(r"\b[A-HJ-NPR-Z0-9]{17}\b").unwrap();
    /// Text in the input that has the shape of a vault placeholder ({{AD_1}}, {{IBAN_1}}).
    static ref PLACEHOLDER_LITERAL_REGEX: regex::Regex = regex::Regex::new(r"\{\{[^{}\s]{1,64}\}\}").unwrap();
    /// A tag of the tag and anonymize modes ([AD], [TCKN], [OZEL_TARIH]).
    static ref TAG_LITERAL_REGEX: regex::Regex = regex::Regex::new(r"\[[A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜ_0-9]{0,39}\]").unwrap();
}

/// High-performance Turkish PII Detection and Pseudonymization Engine.
#[derive(Clone)]
pub struct TurkishPiiEngine {
    embedding_scorer: Option<PiiEmbeddingScorer>,
    preserve_corporate_emails: bool,
    preserve_public_figures: bool,
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
            preserve_public_figures: false,
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
            preserve_public_figures: false,
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

    /// Configures whether the names of well-known public figures (Atatürk, Mustafa Kemal
    /// Atatürk, Nazım Hikmet, Fatih Sultan Mehmet, Orhan Pamuk; see `names_data.rs`) are left
    /// unmasked. Off by default. A detected name is kept only when it lies wholly inside such a
    /// phrase, so "Mustafa Kemal Yılmaz" or a private "Nazım" are still masked.
    pub fn set_preserve_public_figures(&mut self, preserve: bool) {
        self.preserve_public_figures = preserve;
    }

    /// Returns a reference to the embedding scorer, if initialized.
    pub fn embedding_scorer(&self) -> Option<&PiiEmbeddingScorer> {
        self.embedding_scorer.as_ref()
    }

    /// Detects all PII entities in the given text.
    ///
    /// Zero-width and bidi format characters, full-width ASCII and non-Latin digits are folded
    /// and decomposed letters composed first (see `normalize`), so they cannot hide a value from
    /// the detectors; the returned spans and texts are those of `text` itself.
    pub fn detect(&self, text: &str) -> Vec<PiiEntity> {
        let Some(folded) = normalize::fold_for_detection(text) else {
            return self.detect_folded(text);
        };
        self.detect_folded(&folded.text)
            .into_iter()
            .map(|mut e| {
                let folded_text = e.text.clone();
                // A stem that begins the value is mapped back like it: the vault restores the
                // stem when the reply puts a suffix on the placeholder, and it must be what the
                // caller sent (the decomposed letters, not the composed ones).
                let stem_end = (e.stem != folded_text && folded_text.starts_with(&e.stem))
                    .then(|| folded.original_range(e.start, e.start + e.stem.len()).1);
                let (start, end) = folded.original_range(e.start, e.end);
                e.start = start;
                e.end = end;
                e.text = text[start..end].to_string();
                if e.stem == folded_text {
                    e.stem = e.text.clone();
                } else if let Some(stem_end) = stem_end {
                    e.stem = text[start..stem_end.clamp(start, end)].to_string();
                }
                e
            })
            .collect()
    }

    fn detect_folded(&self, text: &str) -> Vec<PiiEntity> {
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

        // Names of well-known public figures, when asked to keep them
        if self.preserve_public_figures
            && candidates
                .iter()
                .any(|e| matches!(e.pii_type, PiiType::Name | PiiType::Person))
        {
            let figures = names::public_figure_spans(text);
            if !figures.is_empty() {
                candidates.retain(|e| {
                    !matches!(e.pii_type, PiiType::Name | PiiType::Person)
                        || !figures.iter().any(|&(s, t)| s <= e.start && e.end <= t)
                });
            }
        }

        // 8. Conflict Resolution & Non-Overlapping Span Arbitration
        resolve_conflicts(candidates, text)
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
        Self::mask_detected(text, entities, mode, vault)
    }

    /// [`mask_with_vault`](Self::mask_with_vault) with the entities [`detect`](Self::detect)
    /// found in `text`, so that a vault shared between threads is held only while the values
    /// are registered, not while the text is read (review 4 BIND4-06).
    pub fn mask_detected(
        text: &str,
        entities: Vec<PiiEntity>,
        mode: PiiMode,
        vault: &mut PiiVault,
    ) -> PiiResult {
        let (masked_text, used) = Self::mask_into(text, &entities, mode, vault);
        let mapping = used
            .iter()
            .filter_map(|p| {
                let entry = vault.placeholder_to_entry.get(p)?;
                Some((p.clone(), entry.original_text.clone()))
            })
            .collect();
        PiiResult {
            original_text: text.to_string(),
            masked_text,
            entities,
            mapping,
            vault: vault.clone(),
        }
    }

    /// The masking of [`mask_detected`](Self::mask_detected) without its result's copy of the
    /// vault: the masked text, and the placeholders and surrogates it holds, each once, in text
    /// order. A caller with a long session vault reads this call's entries from it.
    pub fn mask_into(
        text: &str,
        entities: &[PiiEntity],
        mode: PiiMode,
        vault: &mut PiiVault,
    ) -> (String, Vec<String>) {
        let mut used: Vec<String> = Vec::new();
        let mut masked = String::with_capacity(text.len());
        let mut last_end = 0;

        // the placeholder- and surrogate-shaped strings of the text, collected once
        let taken = vault::issued_shapes_in(text);

        // Text already in the input that restore would turn into a real value is masked too,
        // as a placeholder of its own that restores to that literal text:
        // - placeholder-shaped text ({{IBAN_1}} in a web page an agent read), in every mode;
        // - any key the vault issued in an earlier turn, in every mode: an [EMAIL_1] or a
        //   surrogate date of a surrogate-mode turn, an [AD] of a tag-mode turn;
        // - in tag and anonymize modes, a tag in the input ([AD] in a template): every [AD]
        //   restores to the one original, so it would have become the person.
        let tag_literals = matches!(mode, PiiMode::Tag | PiiMode::Anonymize)
            .then(|| TAG_LITERAL_REGEX.find_iter(text))
            .into_iter()
            .flatten();
        // the entities are sorted and do not overlap: a literal overlaps one if the last entity
        // starting before its end reaches past its start (one search, not a scan per literal)
        let overlaps_entity = |s: usize, e: usize| {
            let i = entities.partition_point(|x| x.start < e);
            i > 0 && entities[i - 1].end > s
        };
        let literals = PLACEHOLDER_LITERAL_REGEX
            .find_iter(text)
            .chain(tag_literals)
            .map(|m| (m.start(), m.end()))
            .chain(vault.restorable_spans(text, &taken))
            .filter(|&(s, e)| !overlaps_entity(s, e));
        let mut spans: Vec<(usize, usize, Option<&PiiEntity>)> = entities
            .iter()
            .map(|e| (e.start, e.end, Some(e)))
            .chain(literals.map(|(s, e)| (s, e, None)))
            .collect();
        // by start, the longest first: of overlapping literals the loop below keeps the first
        spans.sort_by_key(|&(start, end, _)| (start, std::cmp::Reverse(end)));

        for (start, end, entity) in spans {
            if start < last_end {
                continue; // Skip overlapping spans already handled
            }
            masked.push_str(&text[last_end..start]);

            let placeholder = match entity {
                Some(entity) => vault.register_avoiding(
                    &taken,
                    &entity.text,
                    entity.pii_type.as_str(),
                    &entity.stem,
                    entity.suffix.as_deref(),
                    mode,
                ),
                None => {
                    let literal = &text[start..end];
                    vault.register_avoiding(
                        &taken,
                        literal,
                        "YER_TUTUCU",
                        literal,
                        None,
                        PiiMode::Placeholder,
                    )
                }
            };
            masked.push_str(&placeholder);
            if !used.contains(&placeholder) {
                used.push(placeholder);
            }

            last_end = end;
        }

        if last_end < text.len() {
            masked.push_str(&text[last_end..]);
        }
        (masked, used)
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
        // IBAN, TCKN and VKN by shape and trigger, checksum or not (ids.rs)
        ids::detect_ids(text, out);

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
        let lower = normalize::Lowered::of(text);
        if lower.has("tc") || lower.has("kimlik") {
            for cap in TCKN_TRIGGER_REGEX.captures_iter(text) {
                if let Some(val) = cap.get(1) {
                    let (val_start, val_end, span_str) = trimmed_match(val);
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
                            start: val_start,
                            end: val_end,
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
        if lower.has("iban") {
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
        if lower.has("plaka") {
            for cap in PLATE_TRIGGER_REGEX.captures_iter(text) {
                if let Some(val) = cap.get(1) {
                    let (val_start, val_end, span_str) = trimmed_match(val);
                    out.push(PiiEntity {
                        text: span_str.to_string(),
                        label: PiiType::Plate.as_str().to_string(),
                        pii_type: PiiType::Plate,
                        start: val_start,
                        end: val_end,
                        confidence: 0.98,
                        stem: span_str.to_string(),
                        suffix: None,
                    });
                }
            }
        }

        // Credit/Debit Card candidates (13-19 digits, possibly space/dash grouped)
        for mat in CARD_CANDIDATE_REGEX.find_iter(text) {
            // The match is greedy and may end in a separator: "4111 1111 1111 1111 123" failed
            // Luhn as a whole. The longest prefix that ends at a group and passes is the card.
            let whole = mat.as_str().trim_end_matches([' ', '-']);
            let mut cuts: Vec<usize> = whole
                .char_indices()
                .filter(|&(_, c)| c == ' ' || c == '-')
                .map(|(i, _)| i)
                .collect();
            cuts.push(whole.len());
            let card = cuts.iter().rev().map(|&i| &whole[..i]).find(|prefix| {
                let digits = prefix.bytes().filter(u8::is_ascii_digit).count();
                (13..=19).contains(&digits) && validate_credit_card(prefix).is_some()
            });
            if let Some(span_str) = card {
                out.push(PiiEntity {
                    text: span_str.to_string(),
                    label: PiiType::CreditCard.as_str().to_string(),
                    pii_type: PiiType::CreditCard,
                    start: mat.start(),
                    end: mat.start() + span_str.len(),
                    confidence: 1.0,
                    stem: span_str.to_string(),
                    suffix: None,
                });
            }
        }

        // Trigger-preceded Card numbers (e.g. kart no 4147462686590570, kart 4500591135632828)
        if lower.has("kart") {
            for cap in CARD_TRIGGER_REGEX.captures_iter(text) {
                if let Some(val) = cap.get(1) {
                    let (val_start, val_end, span_str) = trimmed_match(val);
                    let span_str = span_str.trim_end_matches('-');
                    let val_end = val_end.min(val_start + span_str.len());
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
                            start: val_start,
                            end: val_end,
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
        let lower = normalize::Lowered::of(text);

        // Phone numbers: the layout regex, then any grouping by numbering plan (phones.rs)
        if has_digits {
            phones::detect_phones(text, out);
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
        if text.sz_find("@").is_some() || lower.has("[at]") || lower.has("(at)") {
            for mat in EMAIL_REGEX.find_iter(text) {
                let span_str = mat.as_str();
                if self.preserve_corporate_emails && is_corporate_email(span_str) {
                    continue;
                }
                if is_file_like_email(span_str) {
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
                if is_file_like_email(span_str) {
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
        if has_digits && (lower.has("port") || text.sz_find(":").is_some()) {
            for cap in PORT_REGEX.captures_iter(text) {
                let full = cap.get(0).unwrap();
                let Some(port) = (1..=3).find_map(|g| cap.get(g)) else {
                    continue;
                };
                let port_str = port.as_str();
                // host:2013/772, host:10.1017 or ]:2016-17 is a reference or a range, not a port
                let after = &text.as_bytes()[full.end()..];
                if after.len() > 1
                    && matches!(after[0], b'/' | b'.' | b',' | b'-')
                    && after[1].is_ascii_digit()
                {
                    continue;
                }
                if let Ok(p) = port_str.parse::<u32>() {
                    if (1..=65535).contains(&p) {
                        // the span is the number's: masking "8080 portu" whole lost "portu" on
                        // restore, which put back only the number
                        out.push(PiiEntity {
                            text: port_str.to_string(),
                            label: PiiType::Port.as_str().to_string(),
                            pii_type: PiiType::Port,
                            start: port.start(),
                            end: port.end(),
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
            && (lower.has("pasaport")
                || text.sz_find("U").is_some()
                || text.sz_find("u").is_some()
                || text.sz_find("A").is_some()
                || text.sz_find("a").is_some()
                || text.sz_find("EP").is_some()
                || text.sz_find("ep").is_some())
        {
            for cap in PASSPORT_REGEX.captures_iter(text) {
                // the number, not "pasaport no: " before it (restore put back only the number)
                let number = cap.get(1).unwrap();
                let val = number.as_str();
                out.push(PiiEntity {
                    text: val.to_string(),
                    label: PiiType::Passport.as_str().to_string(),
                    pii_type: PiiType::Passport,
                    start: number.start(),
                    end: number.end(),
                    confidence: 0.95,
                    stem: val.to_string(),
                    suffix: None,
                });
            }
        }

        // Age & Age range
        if has_digits && (lower.has("yaş") || lower.has("yas")) {
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
            && (lower.has("cvv")
                || lower.has("cvc")
                || lower.has("güvenlik")
                || lower.has("guvenlik"))
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
        if has_digits && (lower.has("skt") || lower.has("son kullanma") || lower.has("exp")) {
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
        if lower.has("müşteri")
            || lower.has("musteri")
            || lower.has("hesap")
            || lower.has("ödeme")
            || lower.has("odeme")
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
        if has_digits && (lower.has("vergi") || lower.has("vkn")) {
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
        if lower.has("engel") || lower.has("özür") || lower.has("ozur") {
            for cap in DISABILITY_REGEX.captures_iter(text) {
                let Some(val) = cap.get(1).or_else(|| cap.get(2)) else {
                    continue;
                };
                let span_str = val.as_str().trim_end();
                if !span_str.is_empty() {
                    out.push(PiiEntity {
                        text: span_str.to_string(),
                        label: "ENGEL_DURUMU".to_string(),
                        pii_type: PiiType::Health,
                        start: val.start(),
                        end: val.start() + span_str.len(),
                        confidence: 0.98,
                        stem: span_str.to_string(),
                        suffix: None,
                    });
                }
            }
        }

        // Passwords, PINs and one-time codes: passwords::detect_passwords (via detect_secrets).

        // Salary / Income
        if has_digits
            && (lower.has("maaş")
                || lower.has("maas")
                || lower.has("gelir")
                || lower.has("ücret")
                || lower.has("ucret"))
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
        if lower.has("ehliyet")
            || lower.has("sürücü")
            || lower.has("surucu")
            || lower.has("şoför")
            || lower.has("sofor")
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
            && (lower.has("sgk") || lower.has("ssk") || lower.has("bağkur") || lower.has("bagkur"))
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
        if has_digits
            && ["doğ", "dog", "d.t.", "d. t.", "birth", "dob", "born"]
                .iter()
                .any(|t| lower.has(t))
        {
            // on the Turkish lower-cased text, whose offsets are mapped back: the regex is
            // case-insensitive only in ASCII, and missed every İ (DOĞUM TARİHİ)
            let (low, origin) = normalize::turkish_lower_with_origin(text);
            let orig = |i: usize| origin.get(i).copied().unwrap_or(text.len());
            for cap in BIRTH_DATE_POSITIVE_REGEX.captures_iter(&low) {
                let date_match = cap
                    .name("date")
                    .or_else(|| cap.name("date2"))
                    .or_else(|| cap.name("year"));
                if let Some(mat) = date_match {
                    let (start, end) = (orig(mat.start()), orig(mat.end()));
                    out.push(PiiEntity {
                        text: text[start..end].to_string(),
                        label: PiiType::BirthDate.as_str().to_string(),
                        pii_type: PiiType::BirthDate,
                        start,
                        end,
                        confidence: 0.98,
                        stem: text[start..end].to_string(),
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
            "ameliyat",
            "kontrol",
            "tahlil",
            "kargo",
            "sipariş",
            "siparis",
            "tören",
            "toren",
            "düğün",
            "dugun",
            "nikah",
            "nikâh",
            "nişan",
            "nisan",
            "uçuş",
            "ucus",
            "vefat",
            "ölüm",
            "olum",
            "taburcu",
            "yatış",
            "yatis",
        ];
        if has_digits && DATE_TRIGGERS.iter().any(|&trig| lower.has(trig)) {
            let (low, origin) = normalize::turkish_lower_with_origin(text);
            let orig = |i: usize| origin.get(i).copied().unwrap_or(text.len());
            let push_private = |m: regex::Match<'_>, out: &mut Vec<PiiEntity>| {
                let (start, end) = (orig(m.start()), orig(m.end()));
                out.push(PiiEntity {
                    text: text[start..end].to_string(),
                    label: "OZEL_TARIH".to_string(),
                    pii_type: PiiType::PrivateDate,
                    start,
                    end,
                    confidence: 0.98,
                    stem: text[start..end].to_string(),
                    suffix: None,
                });
            };
            for cap in PRIVATE_DATE_TRIGGER_REGEX.captures_iter(&low) {
                if let Some(mat) = cap.get(1) {
                    push_private(mat, out);
                }
            }
            for cap in PRIVATE_EVENT_DATE_REGEX.captures_iter(&low) {
                if let Some(mat) = cap.name("date").or_else(|| cap.name("date2")) {
                    push_private(mat, out);
                }
            }
            if lower.has("olan") {
                for cap in PRIVATE_DATE_POST_TRIGGER_REGEX.captures_iter(&low) {
                    if let Some(mat) = cap.get(1) {
                        push_private(mat, out);
                    }
                }
            }
        }

        // Dates are personal by their trigger words above. The embedding check of a date's
        // context (PiiEmbeddingScorer::is_private_date_context) was taken out: on 90 labeled
        // dates it found none the rules miss and took 9 of 45 public ones, and it marked 238 of
        // 1,461 dates in court decisions and 62 of 1,012 in Wikipedia as dates of birth.

        // IMEI device numbers
        if has_digits && (lower.has("imei") || text.len() >= 15) {
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
        if lower.has("ruhsat") {
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
        if lower.has("sözleşme") || lower.has("sozlesme") {
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
        if lower.has("poliçe") || lower.has("police") {
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
        if lower.has("referans") || text.sz_find("REF-").is_some() || text.sz_find("RF").is_some() {
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
        if lower.has("şasi") || lower.has("sasi") || lower.has("vin") {
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
        if lower.has("motor") {
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
        if has_digits && lower.has("pin") {
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
        if has_digits && (lower.has("kredi") || lower.has("findeks")) {
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
        if lower.has("kızlık") || lower.has("kizlik") {
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
        if lower.has("cinsiyet")
            || lower.has("kadın")
            || lower.has("kadin")
            || lower.has("erkek")
            || lower.has("bayan")
            || lower.has("dişi")
            || lower.has("disi")
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
        if lower.has("cihaz") || lower.has("device") {
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
        if lower.has("sicil") {
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
        if lower.has("imza") {
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
        if lower.has("ceza")
            || lower.has("kabahat")
            || lower.has("serbestlik")
            || lower.has("icra")
            || lower.has("sicil")
            || lower.has("sabıka")
            || lower.has("sabika")
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
        if lower.has("uyruk")
            || lower.has("uyruğu")
            || lower.has("uyrugu")
            || lower.has("vatandaş")
            || lower.has("vatandas")
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
        // every trigger of the regex: "rahatsızlığı" does not contain "rahatsızlık", and
        // tanı / teşhis / hastalığı had no gate word at all
        if [
            "sağlık",
            "saglik",
            "rahatsızl",
            "rahatsizl",
            "hastalığ",
            "hastalig",
            "tanı",
            "tani",
            "teşhis",
            "teshis",
            "kronik",
            "engel",
        ]
        .iter()
        .any(|k| lower.has(k))
        {
            for cap in HEALTH_RECORD_REGEX.captures_iter(text) {
                let Some(val) = cap.get(1).or_else(|| cap.get(2)) else {
                    continue;
                };
                let span_str = val.as_str().trim_end();
                if !span_str.is_empty() {
                    out.push(PiiEntity {
                        text: span_str.to_string(),
                        label: "SAGLIK".to_string(),
                        pii_type: PiiType::Health,
                        start: val.start(),
                        end: val.start() + span_str.len(),
                        confidence: 0.95,
                        stem: span_str.to_string(),
                        suffix: None,
                    });
                }
            }
        }

        // Trade union membership
        if lower.has("sendika")
            || lower.has("sendikası")
            || lower.has("sendikasi")
            || lower.has("-iş")
            || lower.has("-is")
            || lower.has("sen ")
            || lower.has("sen\n")
        {
            for cap in TRADE_UNION_REGEX.captures_iter(text) {
                // the union, not "sendika bilgisi" before it
                let Some(mat) = cap.get(1).or_else(|| cap.get(2)).or_else(|| cap.get(3)) else {
                    continue;
                };
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
        if lower.has("kullanıcı")
            || lower.has("kullanici")
            || lower.has("username")
            || lower.has("oturum")
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
        if lower.has("damar")
            || lower.has("parmak")
            || lower.has("yüz")
            || lower.has("yuz")
            || lower.has("iris")
            || lower.has("retina")
            || lower.has("biyometrik")
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
        if lower.has("doğum") || lower.has("dogum") {
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
        if lower.has("aile") || lower.has("medeni") {
            for cap in FAMILY_STATUS_REGEX.captures_iter(text) {
                let (val_start, val_end, span_str) = trimmed_match(cap.get(1).unwrap());
                if !span_str.is_empty() {
                    out.push(PiiEntity {
                        text: span_str.to_string(),
                        label: "AILE".to_string(),
                        pii_type: PiiType::Person,
                        start: val_start,
                        end: val_end,
                        confidence: 0.95,
                        stem: span_str.to_string(),
                        suffix: None,
                    });
                }
            }
        }

        // Ethnic origin
        if lower.has("etnik") {
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
        if WORKPLACE_TRIGGERS.iter().any(|&trig| lower.has(trig)) {
            for cap in WORKPLACE_REGEX.captures_iter(text) {
                // the name, not the "İşyeri:" before it or "bünyesinde" after it
                let Some(mat) = cap.get(1).or_else(|| cap.get(2)) else {
                    continue;
                };
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
        if lower.has("anne") || lower.has("baba") {
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
        names::detect_names(text, self.embedding_scorer.as_ref(), out);
    }

    fn detect_addresses(&self, text: &str, out: &mut Vec<PiiEntity>) {
        // addresses.rs; the regex that was here (Mahallesi + one street word, case-insensitive)
        // ran across lines and cut inside words (Sok in "sokakta", Cad in "caddede")
        addresses::detect_addresses(text, out);
    }

    fn detect_sensitive_categories(&self, text: &str, out: &mut Vec<PiiEntity>) {
        // Terms are found in the lower-cased text, whose byte offsets differ from `text`'s after
        // every İ or I: using them directly masked the wrong bytes and panicked inside İ.
        let (lower, origin) = normalize::turkish_lower_with_origin(text);
        let orig = |i: usize| origin.get(i).copied().unwrap_or(text.len());

        // Blood types
        for bt in BLOOD_TYPES.iter() {
            for (pos, pos_end) in term_spans(&lower, bt) {
                let (start, end) = (orig(pos), orig(pos_end));
                out.push(PiiEntity {
                    text: text[start..end].to_string(),
                    label: PiiType::BloodType.as_str().to_string(),
                    pii_type: PiiType::BloodType,
                    start,
                    end,
                    confidence: 0.98,
                    stem: text[start..end].to_string(),
                    suffix: None,
                });
            }
        }

        // Health terms
        for ht in HEALTH_TERMS.iter() {
            for (pos, pos_end) in term_spans(&lower, ht) {
                let (start, end) = (orig(pos), orig(pos_end));
                out.push(PiiEntity {
                    text: text[start..end].to_string(),
                    label: PiiType::Health.as_str().to_string(),
                    pii_type: PiiType::Health,
                    start,
                    end,
                    confidence: 0.95,
                    stem: text[start..end].to_string(),
                    suffix: None,
                });
            }
        }

        // Religion terms
        for rt in RELIGION_TERMS.iter() {
            for (pos, pos_end) in term_spans(&lower, rt) {
                let (start, end) = (orig(pos), orig(pos_end));
                out.push(PiiEntity {
                    text: text[start..end].to_string(),
                    label: "DIN".to_string(),
                    pii_type: PiiType::Health, // Sensitive KVKK 6
                    start,
                    end,
                    confidence: 0.90,
                    stem: text[start..end].to_string(),
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
                "ilacı",
                "ilaci",
                "rahatsızlığı",
                "rahatsizligi",
                "nöbeti",
                "nobeti",
            ];
            let stopwords = crate::morphology::TurkishStopwords::new();
            for trig in CLINICAL_TRIGGERS {
                let mut start_idx = 0;
                while let Some(pos) = lower[start_idx..].sz_find(trig) {
                    let lower_pos = start_idx + pos;
                    let actual_pos = orig(lower_pos);
                    let window_start = floor_char_boundary(text, actual_pos.saturating_sub(40));
                    let candidate_context = text[window_start..actual_pos].trim();
                    if let Some(last_word) = candidate_context.split_whitespace().last() {
                        let clean_word = last_word.trim_matches(|c: char| !c.is_alphabetic());
                        let letters = clean_word.chars().count();
                        // three letters or more (gut, lupus), or an acronym (MS, ALS, OKB)
                        let long_enough = letters >= 3
                            || (letters == 2 && clean_word.chars().all(char::is_uppercase));
                        let lower_word = to_turkish_lower(clean_word);
                        if long_enough
                            && !stopwords.is_stopword(&lower_word)
                            && names::is_bare_nominal_or_unknown(&lower_word)
                            && scorer.is_health_condition(clean_word)
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
                    start_idx = lower_pos + trig.len();
                }
            }
        }
    }
}

/// Every occurrence of `term` in `lower` as a word: starting at a word boundary, and for a
/// term of four letters or fewer also ending at one (or at an apostrophe), so that a term is
/// not found inside another word (sara in hasara or Saraçhane, hiv in archived) while longer
/// terms keep their suffixes (kanserli, diyabetim, nörolojik). The previous version searched
/// for substrings, and religion and blood-type terms only for their first occurrence.
fn term_spans(lower: &str, term: &str) -> Vec<(usize, usize)> {
    let short = term.chars().filter(|c| c.is_alphanumeric()).count() <= 4;
    let mut out = Vec::new();
    let mut from = 0;
    while let Some(pos) = lower[from..].sz_find(term) {
        let start = from + pos;
        let end = start + term.len();
        from = end;
        let before = lower[..start].chars().next_back();
        if before.is_some_and(char::is_alphanumeric) {
            continue;
        }
        let ends_in_word = term.chars().next_back().is_some_and(char::is_alphanumeric);
        if short
            && ends_in_word
            && lower[end..]
                .chars()
                .next()
                .is_some_and(char::is_alphanumeric)
        {
            continue;
        }
        out.push((start, end));
    }
    out
}

/// The span of `m` without the whitespace around it, and its text: the entity text was trimmed
/// but the span was not, so masking took the space after the value and restore lost it
/// ("TC 12345678901 ve" became "TC {{TCKN_1}}ve").
fn trimmed_match<'t>(m: regex::Match<'t>) -> (usize, usize, &'t str) {
    let raw = m.as_str();
    let lead = raw.len() - raw.trim_start().len();
    let t = raw.trim();
    (m.start() + lead, m.start() + lead + t.len(), t)
}

/// `'in`, `’e`: an apostrophe and the letters of a suffix.
fn is_apostrophe_suffix(s: &str) -> bool {
    let mut chars = s.chars();
    matches!(chars.next(), Some('\'' | '’')) && {
        let rest = chars.as_str();
        !rest.is_empty() && rest.chars().all(char::is_alphabetic)
    }
}

/// The largest char boundary of `text` at or before `i`.
fn floor_char_boundary(text: &str, i: usize) -> usize {
    let mut i = i.min(text.len());
    while !text.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// The smallest char boundary of `text` at or after `i`.
fn ceil_char_boundary(text: &str, i: usize) -> usize {
    let mut i = i.min(text.len());
    while !text.is_char_boundary(i) {
        i += 1;
    }
    i
}

/// Resolves overlapping spans into non-overlapping ones. Overlapping detections are merged
/// into one span covering both, labelled as the more confident one: keeping only the more
/// confident span unmasked the rest of the other (a credential around an e-mail-like part of it).
fn resolve_conflicts(mut candidates: Vec<PiiEntity>, text: &str) -> Vec<PiiEntity> {
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
            // Overlap condition (sorted by start, so last.start <= cand.start)
            if cand.start < last.end {
                let start = last.start;
                let mut end = last.end.max(cand.end);
                let winner = if cand.confidence > last.confidence {
                    cand
                } else {
                    last.clone()
                };
                // A case suffix after an apostrophe (Ahmet'in) stays outside the masked name, as
                // the text it is restored into: do not widen over it.
                if end > winner.end && is_apostrophe_suffix(&text[winner.end..end]) {
                    end = winner.end;
                }
                // Keep the winner's own text, stem and suffix when its span is the merged one.
                let span_changed = (winner.start, winner.end) != (start, end);
                *last = PiiEntity {
                    start,
                    end,
                    ..winner
                };
                if span_changed {
                    last.text = text[start..end].to_string();
                    last.stem = last.text.clone();
                    last.suffix = None;
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
    let prefix = if let Some(at_idx) = lower.sz_find("@") {
        &lower[..at_idx]
    } else if let Some(at_idx) = lower.sz_find("[at]") {
        &lower[..at_idx]
    } else if let Some(at_idx) = lower.sz_find("(at)") {
        &lower[..at_idx]
    } else {
        return false;
    };
    let clean_prefix: String = prefix.chars().filter(|c| !c.is_whitespace()).collect();
    if crate::pii::gazetteer::CORPORATE_EMAIL_PREFIXES.contains(clean_prefix.as_str()) {
        return true;
    }
    let norm_prefix = clean_prefix.replace(['.', '_', '-'], "");
    if crate::pii::gazetteer::CORPORATE_EMAIL_PREFIXES.contains(norm_prefix.as_str()) {
        return true;
    }
    false
}

const NON_DOMAIN_FILE_EXTENSIONS: &[&str] = &[
    "ext", "txt", "toml", "json", "yaml", "yml", "xml", "html", "htm", "css", "scss", "sass",
    "less", "js", "jsx", "ts", "tsx", "mjs", "cjs", "py", "pyc", "pyd", "rb", "php", "c", "cpp",
    "h", "hpp", "cc", "cxx", "cs", "go", "java", "kt", "swift", "bash", "zsh", "bat", "cmd", "ps1",
    "sql", "graphql", "csv", "tsv", "parquet", "png", "jpg", "jpeg", "gif", "svg", "webp", "ico",
    "zip", "tar", "gz", "tgz", "7z", "rar", "pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx",
    "lock", "wasm", "log", "env", "bak", "tmp", "temp", "cfg", "ini", "conf", "bin", "exe", "dll",
    "dylib", "patch", "diff", "rst", "adoc",
];

const MARKDOWN_OR_CODE_EXTENSIONS: &[&str] = &["md", "rs", "sh"];

const FILE_LIKE_BASE_NAMES: &[&str] = &[
    "readme",
    "handoff",
    "changelog",
    "license",
    "contributing",
    "dockerfile",
    "makefile",
    "cargo",
    "package",
    "tsconfig",
    "file",
    "index",
    "main",
    "lib",
    "mod",
    "build",
    "deploy",
    "setup",
    "install",
    "test",
    "tests",
    "doc",
    "docs",
    "notes",
    "todo",
    "guide",
    "summary",
    "spec",
    "architecture",
    "arch",
    "walkthrough",
    "plan",
    "rfc",
];

/// Helper function to check if an email candidate looks like a file mention or filename.
fn is_file_like_email(email: &str) -> bool {
    let lower = email.to_lowercase();

    // Legitimate email addresses cannot have whitespace adjacent to literal '@'
    if lower.contains(" @") || lower.contains("@ ") || lower.starts_with('@') {
        return true;
    }

    let domain_part = if let Some(idx) = lower.sz_rfind("@") {
        &lower[idx + 1..]
    } else if let Some(idx) = lower.sz_rfind("[at]") {
        &lower[idx + 4..]
    } else if let Some(idx) = lower.sz_rfind("(at)") {
        &lower[idx + 4..]
    } else {
        return false;
    };

    let domain_part = domain_part.replace("[dot]", ".").replace("(dot)", ".");
    let clean_domain: String = domain_part.chars().filter(|c| !c.is_whitespace()).collect();

    let parts: Vec<&str> = clean_domain.split('.').collect();
    if parts.len() < 2 {
        return false;
    }

    let ext = parts.last().copied().unwrap_or("");
    let base = parts[parts.len() - 2];

    if NON_DOMAIN_FILE_EXTENSIONS.contains(&ext) {
        return true;
    }

    if MARKDOWN_OR_CODE_EXTENSIONS.contains(&ext) && FILE_LIKE_BASE_NAMES.contains(&base) {
        return true;
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
        }
    }

    #[test]
    fn test_file_mentions_not_detected_as_email() {
        let engine = TurkishPiiEngine::new();

        // Exact false positives reported by Yada team
        let prompt1 = "update @README.md";
        let res1 = engine.mask(prompt1, PiiMode::Placeholder);
        assert_eq!(res1.masked_text, "update @README.md");
        assert!(res1.entities.is_empty());

        let prompt2 = "and @handoff.md";
        let res2 = engine.mask(prompt2, PiiMode::Placeholder);
        assert_eq!(res2.masked_text, "and @handoff.md");
        assert!(res2.entities.is_empty());

        let prompt3 = "see @file.ext";
        let res3 = engine.mask(prompt3, PiiMode::Placeholder);
        assert_eq!(res3.masked_text, "see @file.ext");
        assert!(res3.entities.is_empty());

        // Other common file mentions
        let prompt4 = "Please review @Cargo.toml and check @main.rs with @script.py";
        let res4 = engine.mask(prompt4, PiiMode::Placeholder);
        assert_eq!(
            res4.masked_text,
            "Please review @Cargo.toml and check @main.rs with @script.py"
        );
        assert!(res4.entities.is_empty());

        // Valid emails must still be detected
        let email_prompt =
            "İletişim için ahmet.yilmaz@example.com veya ali [at] domain [dot] com kullanın.";
        let res_email = engine.mask(email_prompt, PiiMode::Placeholder);
        assert!(res_email.masked_text.contains("{{EMAIL_1}}"));
        assert!(res_email.masked_text.contains("{{EMAIL_2}}"));
        assert_eq!(res_email.entities.len(), 2);
    }

    #[test]
    fn test_is_file_like_email_helper() {
        assert!(is_file_like_email("update @README.md"));
        assert!(is_file_like_email("and @handoff.md"));
        assert!(is_file_like_email("see @file.ext"));
        assert!(is_file_like_email("user@README.md"));
        assert!(is_file_like_email("user@Cargo.toml"));
        assert!(is_file_like_email("user@script.py"));
        assert!(is_file_like_email("user@sample.json"));

        assert!(!is_file_like_email("user@example.com"));
        assert!(!is_file_like_email("ahmet.yilmaz@example.com"));
        assert!(!is_file_like_email("ali [at] domain [dot] com"));
        assert!(!is_file_like_email("user@gmail [dot] com"));
        assert!(!is_file_like_email("support@company.md"));
        assert!(!is_file_like_email("contact@gov.md"));
        assert!(!is_file_like_email("qa@test.com"));
    }

    fn entity(text: &str, start: usize, end: usize, label: &str, confidence: f32) -> PiiEntity {
        PiiEntity {
            text: text[start..end].to_string(),
            label: label.to_string(),
            pii_type: PiiType::Credentials,
            start,
            end,
            confidence,
            stem: text[start..end].to_string(),
            suffix: None,
        }
    }

    #[test]
    fn overlapping_detections_are_merged_not_shrunk() {
        // A more confident e-mail-like part inside a credential used to replace it, leaving the
        // rest of the credential unmasked.
        let text = "url deploy:VhVWE6pSMTnz6pJu@git.example.com son";
        let long = entity(text, 4, 43, "SIFRE", 0.9);
        let inner = entity(text, 11, 43, "EMAIL", 0.99);
        let out = resolve_conflicts(vec![long, inner], text);
        assert_eq!(out.len(), 1);
        assert_eq!((out[0].start, out[0].end), (4, 43));
        assert_eq!(out[0].label, "EMAIL");
        assert_eq!(out[0].text, "deploy:VhVWE6pSMTnz6pJu@git.example.com");

        // A less confident span reaching past the winner widens it...
        let text = "x ab12cd34ef56@host.com.tr y";
        let a = entity(text, 2, 19, "EMAIL", 0.99);
        let b = entity(text, 9, 26, "SIFRE", 0.8);
        let out = resolve_conflicts(vec![a, b], text);
        assert_eq!(
            (out[0].start, out[0].end, out[0].label.as_str()),
            (2, 26, "EMAIL")
        );

        // ...but not over a case suffix after an apostrophe, which restoration needs.
        let text = "Ahmet'in kaydı";
        let mut name = entity(text, 0, 5, "AD", 0.95);
        name.suffix = Some("in".to_string());
        let entropy = entity(text, 0, 8, "SIFRE", 0.9);
        let out = resolve_conflicts(vec![entropy, name], text);
        assert_eq!(
            (out[0].start, out[0].end, out[0].label.as_str()),
            (0, 5, "AD")
        );
        assert_eq!(out[0].suffix.as_deref(), Some("in"));
    }
}
