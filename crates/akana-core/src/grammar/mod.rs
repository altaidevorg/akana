//! Turkish Grammatical Error Correction (GEC) and Detection (GED) Module.
//!
//! Provides a high-speed, SIMD-accelerated, rule-and-morphology-driven grammar checker.
//! Covers 25 core Turkish grammatical, phonological, orthographic, and morphosyntactic error categories.

pub mod agreement;
pub mod lexicon_rules;
pub mod orthography;
pub mod phonological_rules;

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
        orthography::OrthographyRuleEngine::check(
            text,
            &tokens,
            &token_spans,
            &self.morphology,
            &mut findings,
        );

        // Pass 2: Phonological Suffix Rules Engine
        phonological_rules::PhonologicalRuleEngine::check(
            text,
            &tokens,
            &token_spans,
            &self.morphology,
            &mut findings,
        );

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

        // Sort findings by span, and drop a second finding of the same category on the same
        // span. Findings of different categories on one span are all reported (a sentence-
        // initial "duydumki" needs both its capital and its "ki"); which of them are applied
        // is decided by `apply_corrections`.
        findings.sort_by_key(|f| (f.start_offset, f.end_offset));
        findings.dedup_by(|a, b| {
            a.start_offset == b.start_offset
                && a.end_offset == b.end_offset
                && a.category == b.category
        });

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

    /// Applies the findings' edits to `original`.
    ///
    /// Findings can overlap: a split of "Ulaştığınıda" and a merge of "Ulaştığınıda biliyoruz"
    /// both rewrite the first word. Replaying them one after the other, each on the text the
    /// previous one changed, corrupted it ("Ulaştığınıdabiliyoruzz."). So the edits are chosen
    /// first: the most confident one wins a conflict (then the narrower one, then the earlier),
    /// and an edit that overlaps an edit already chosen is dropped. A capitalisation finding
    /// is not an edit of its own when another edit starts at the same word: that edit's
    /// replacement is capitalised instead, so "duydumki" gets both its capital and its "ki".
    /// The chosen edits are then applied in one pass against the original offsets.
    fn apply_corrections(original: &str, findings: &[GrammarFinding]) -> String {
        let applicable = |f: &&GrammarFinding| {
            f.start_offset <= f.end_offset
                && f.end_offset <= original.len()
                && original.is_char_boundary(f.start_offset)
                && original.is_char_boundary(f.end_offset)
                && original[f.start_offset..f.end_offset] != f.replacement
        };
        let (capitals, mut candidates): (Vec<&GrammarFinding>, Vec<&GrammarFinding>) = findings
            .iter()
            .filter(applicable)
            .partition(|f| f.category == ErrorCategory::Capitalization);
        candidates.sort_by(|a, b| {
            b.confidence
                .total_cmp(&a.confidence)
                .then_with(|| (a.end_offset - a.start_offset).cmp(&(b.end_offset - b.start_offset)))
                .then_with(|| a.start_offset.cmp(&b.start_offset))
        });

        // Two edits conflict when their spans share a byte, or when they insert at, or
        // replace, the same position.
        let overlaps =
            |a: (usize, usize), b: (usize, usize)| (a.0 < b.1 && b.0 < a.1) || a.0 == b.0;
        let mut edits: Vec<(usize, usize, String)> = Vec::new();
        for f in candidates {
            let span = (f.start_offset, f.end_offset);
            if !edits.iter().any(|&(s, e, _)| overlaps((s, e), span)) {
                edits.push((span.0, span.1, f.replacement.clone()));
            }
        }
        for f in capitals {
            let span = (f.start_offset, f.end_offset);
            if let Some(edit) = edits.iter_mut().find(|(s, _, _)| *s == span.0) {
                edit.2 = capitalize_first(&edit.2);
            } else if !edits.iter().any(|&(s, e, _)| overlaps((s, e), span)) {
                edits.push((span.0, span.1, f.replacement.clone()));
            }
        }

        edits.sort_by_key(|&(s, _, _)| s);
        let mut result = String::with_capacity(original.len());
        let mut last = 0;
        for (start, end, replacement) in &edits {
            result.push_str(&original[last..*start]);
            result.push_str(replacement);
            last = *end;
        }
        result.push_str(&original[last..]);
        result
    }
}

/// `s` with its first character in Turkish upper case (i -> İ, ı -> I).
fn capitalize_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_lowercase() => {
            let mut out = crate::phonology::to_turkish_upper(&c.to_string());
            out.push_str(chars.as_str());
            out
        }
        _ => s.to_string(),
    }
}
