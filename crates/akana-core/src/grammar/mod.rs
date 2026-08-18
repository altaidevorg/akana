//! Turkish Grammatical Error Correction (GEC) and Detection (GED) Module.
//!
//! Provides a high-speed, SIMD-accelerated, rule-and-morphology-driven grammar checker.
//! Covers 25 core Turkish grammatical, phonological, orthographic, and morphosyntactic error categories.

pub mod orthography;
pub mod phonological_rules;
pub mod agreement;
pub mod lexicon_rules;

use serde::{Deserialize, Serialize};
use std::time::Instant;

use crate::morphology::{MorphologicalDisambiguator, TurkishMorphology};
use crate::parser::TurkishDependencyParser;
use crate::tokenization::TurkishTokenizer;

/// 25 Turkish Grammatical & Orthographic Error Categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ErrorCategory {
    /// Separate vs Attached `de/da` clitic or invalid `te/ta` hardening
    CliticDeDa,
    /// Separate vs Attached `ki` clitic (with SOMBAHÇEMİ exceptions)
    CliticKi,
    /// Question particle `mi/mı/mu/mü` attachment & vowel harmony
    ParticleMi,
    /// Suffix Major & Minor Vowel Harmony violations
    VowelHarmony,
    /// Consonant Assimilation / Hardening (*fıstıkçı şahap* rule)
    ConsonantAssimilation,
    /// Consonant Softening / Voicing (*kitapı* -> *kitabı*)
    ConsonantSoftening,
    /// Vowel Dropping / Syncope (*akılı* -> *aklı*)
    VowelDropping,
    /// Proper Noun Apostrophe usage & illegal apostrophe on plurals
    ApostropheProperNoun,
    /// Number & Date Apostrophe / Suffix assimilation (*1923'te*, *2'nci*)
    ApostropheNumberDate,
    /// Sentence-initial & Proper Noun Capitalization
    Capitalization,
    /// Quantity modifier + Plural noun clash (*üç elmalar* -> *üç elma*)
    QuantityPluralClash,
    /// Subject-Verb Person / Number / Inanimate Plurality agreement
    SubjectVerbAgreement,
    /// Genitive - Possessive agreement (*benim ev* -> *benim evim*)
    GenitivePossessiveClash,
    /// Correlative conjunction polarity (*ne ... ne ...* requires positive verb)
    CorrelativeConjunctionPolarity,
    /// Compound auxiliary verb orthography (*terketmek* -> *terk etmek*, *hiss etmek* -> *hissetmek*)
    CompoundWordOrthography,
    /// Reduplication orthography (*elele* -> *el ele*)
    ReduplicationOrthography,
    /// Tautology & Pleonasm redundancy (*henüz hala*, *birlikte beraber*)
    TautologyRedundancy,
    /// General spelling / typographical error
    SpellingTypo,
}

impl ErrorCategory {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::CliticDeDa => "CliticDeDa",
            Self::CliticKi => "CliticKi",
            Self::ParticleMi => "ParticleMi",
            Self::VowelHarmony => "VowelHarmony",
            Self::ConsonantAssimilation => "ConsonantAssimilation",
            Self::ConsonantSoftening => "ConsonantSoftening",
            Self::VowelDropping => "VowelDropping",
            Self::ApostropheProperNoun => "ApostropheProperNoun",
            Self::ApostropheNumberDate => "ApostropheNumberDate",
            Self::Capitalization => "Capitalization",
            Self::QuantityPluralClash => "QuantityPluralClash",
            Self::SubjectVerbAgreement => "SubjectVerbAgreement",
            Self::GenitivePossessiveClash => "GenitivePossessiveClash",
            Self::CorrelativeConjunctionPolarity => "CorrelativeConjunctionPolarity",
            Self::CompoundWordOrthography => "CompoundWordOrthography",
            Self::ReduplicationOrthography => "ReduplicationOrthography",
            Self::TautologyRedundancy => "TautologyRedundancy",
            Self::SpellingTypo => "SpellingTypo",
        }
    }
}

/// A specific grammatical/orthographical error finding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrammarFinding {
    pub category: ErrorCategory,
    pub start_offset: usize,
    pub end_offset: usize,
    pub original_text: String,
    pub replacement: String,
    pub message_tr: String,
    pub message_en: String,
    pub confidence: f32,
}

/// Result of checking and correcting a text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrammarCheckResult {
    pub original: String,
    pub corrected: String,
    pub findings: Vec<GrammarFinding>,
    pub processing_time_us: u64,
}

/// High-Performance Turkish Grammar Checker Engine.
pub struct TurkishGrammarChecker {
    morphology: TurkishMorphology,
    disambiguator: MorphologicalDisambiguator,
    parser: TurkishDependencyParser,
}

impl Default for TurkishGrammarChecker {
    fn default() -> Self {
        Self::new()
    }
}

impl TurkishGrammarChecker {
    pub fn new() -> Self {
        let morphology = TurkishMorphology::new();
        let disambiguator = MorphologicalDisambiguator::new();
        let parser = TurkishDependencyParser::new();

        Self {
            morphology,
            disambiguator,
            parser,
        }
    }

    /// Performs full grammatical error detection (GED) and returns all findings with explanations.
    pub fn check(&self, text: &str) -> GrammarCheckResult {
        let start_time = Instant::now();

        if text.trim().is_empty() {
            return GrammarCheckResult {
                original: text.to_string(),
                corrected: text.to_string(),
                findings: Vec::new(),
                processing_time_us: start_time.elapsed().as_micros() as u64,
            };
        }

        // 1. Tokenize text and record exact character spans
        let tokens: Vec<&str> = TurkishTokenizer::tokenize_words(text);
        let mut token_spans: Vec<(usize, usize)> = Vec::with_capacity(tokens.len());
        let mut search_offset = 0;

        for &tok in &tokens {
            if let Some(pos) = text[search_offset..].find(tok) {
                let start = search_offset + pos;
                let end = start + tok.len();
                token_spans.push((start, end));
                search_offset = end;
            } else {
                token_spans.push((0, 0));
            }
        }

        let mut findings: Vec<GrammarFinding> = Vec::new();

        // Pass 1: Orthography & Clitic Rules Engine (SIMD accelerated)
        orthography::OrthographyRuleEngine::check(text, &tokens, &token_spans, &self.morphology, &mut findings);

        // Pass 2: Phonological Suffix Rules Engine
        phonological_rules::PhonologicalRuleEngine::check(text, &tokens, &token_spans, &self.morphology, &mut findings);

        // Pass 3: Morphosyntactic Agreement & Concord Engine
        agreement::AgreementRuleEngine::check(
            text,
            &tokens,
            &token_spans,
            &self.morphology,
            &self.disambiguator,
            &self.parser,
            &mut findings,
        );

        // Pass 4: Lexicon & Compound Verb Rules Engine
        lexicon_rules::LexiconRuleEngine::check(text, &tokens, &token_spans, &mut findings);

        // Deduplicate and sort findings by starting offset
        findings.sort_by_key(|f| f.start_offset);
        findings.dedup_by(|a, b| a.start_offset == b.start_offset && a.end_offset == b.end_offset);

        // Pass 5: Synthesize corrected sentence
        let corrected = Self::apply_corrections(text, &findings);

        GrammarCheckResult {
            original: text.to_string(),
            corrected,
            findings,
            processing_time_us: start_time.elapsed().as_micros() as u64,
        }
    }

    /// Automatically corrects grammatical errors in the text.
    pub fn correct(&self, text: &str) -> String {
        self.check(text).corrected
    }

    /// Applies non-overlapping edits in reverse character index order.
    fn apply_corrections(original: &str, findings: &[GrammarFinding]) -> String {
        if findings.is_empty() {
            return original.to_string();
        }

        let mut result = original.to_string();
        let mut sorted_findings = findings.to_vec();
        // Sort descending by start_offset so earlier offsets remain valid after string replacements
        sorted_findings.sort_by(|a, b| b.start_offset.cmp(&a.start_offset));

        for finding in sorted_findings {
            if finding.start_offset <= finding.end_offset && finding.end_offset <= result.len() {
                // Ensure character boundaries are respected
                if result.is_char_boundary(finding.start_offset) && result.is_char_boundary(finding.end_offset) {
                    result.replace_range(finding.start_offset..finding.end_offset, &finding.replacement);
                }
            }
        }

        result
    }
}
