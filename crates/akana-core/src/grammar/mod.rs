//! Turkish Grammatical Error Correction (GEC) and Detection (GED) Module.
//!
//! A rule-and-morphology-driven grammar checker: 33 rules ([`GrammarRule`]) in 17 error
//! categories. A check runs a [`RuleSet`]; the default one holds only the rules that measured
//! precise enough on correct text, and the rest are experimental (see [`rules`]).

pub mod agreement;
pub mod lexicon_rules;
pub mod orthography;
pub mod phonological_rules;
pub mod rules;

pub use rules::{GrammarRule, RuleSet, UnknownRule};

use serde::{Deserialize, Serialize};
use std::ops::Range;
use std::time::Instant;

use crate::morphology::{MorphologicalDisambiguator, TurkishMorphology};
use crate::parser::TurkishDependencyParser;
use crate::tokenization::TurkishTokenizer;

/// Turkish grammatical and orthographic error categories; a category can hold several rules.
/// `SpellingTypo` has none yet.
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
#[serde(try_from = "FindingRecord")]
pub struct GrammarFinding {
    /// The rule that found it.
    pub rule: GrammarRule,
    pub category: ErrorCategory,
    pub start_offset: usize,
    pub end_offset: usize,
    pub original_text: String,
    pub replacement: String,
    pub message_tr: String,
    pub message_en: String,
    /// The rule's measured precision ([`GrammarRule::confidence`]).
    pub confidence: f32,
}

/// A [`GrammarFinding`] as it is read back. Findings serialized by akana 0.8.0 have no `rule`;
/// they take the first rule of their category, which every category 0.8.0 reported has.
#[derive(Deserialize)]
struct FindingRecord {
    #[serde(default)]
    rule: Option<GrammarRule>,
    category: ErrorCategory,
    start_offset: usize,
    end_offset: usize,
    original_text: String,
    replacement: String,
    message_tr: String,
    message_en: String,
    confidence: f32,
}

impl TryFrom<FindingRecord> for GrammarFinding {
    type Error = String;

    fn try_from(r: FindingRecord) -> Result<Self, String> {
        let rule = match r.rule {
            Some(rule) => rule,
            None => GrammarRule::ALL
                .iter()
                .copied()
                .find(|rule| rule.category() == r.category)
                .ok_or_else(|| format!("a {} finding without a rule", r.category.as_str()))?,
        };
        Ok(Self {
            rule,
            category: r.category,
            start_offset: r.start_offset,
            end_offset: r.end_offset,
            original_text: r.original_text,
            replacement: r.replacement,
            message_tr: r.message_tr,
            message_en: r.message_en,
            confidence: r.confidence,
        })
    }
}

impl GrammarFinding {
    /// A finding of `rule` on the bytes `span`, with the rule's category and confidence.
    pub fn new(
        rule: GrammarRule,
        span: (usize, usize),
        original_text: impl Into<String>,
        replacement: impl Into<String>,
        message_tr: impl Into<String>,
        message_en: impl Into<String>,
    ) -> Self {
        Self {
            rule,
            category: rule.category(),
            start_offset: span.0,
            end_offset: span.1,
            original_text: original_text.into(),
            replacement: replacement.into(),
            message_tr: message_tr.into(),
            message_en: message_en.into(),
            confidence: rule.confidence(),
        }
    }
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
///
/// [`TurkishGrammarChecker::new`] runs the default rules. [`with_rules`](Self::with_rules),
/// [`enable_rule`](Self::enable_rule), [`disable_rule`](Self::disable_rule) and
/// [`enable_experimental`](Self::enable_experimental) change them, and
/// [`check_with_rules`](Self::check_with_rules) runs other rules for one call.
pub struct TurkishGrammarChecker {
    morphology: TurkishMorphology,
    disambiguator: MorphologicalDisambiguator,
    parser: TurkishDependencyParser,
    rules: RuleSet,
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
            rules: RuleSet::default(),
        }
    }

    /// A checker that runs `rules`.
    pub fn with_rules(rules: RuleSet) -> Self {
        Self {
            rules,
            ..Self::new()
        }
    }

    /// The rules [`check`](Self::check) runs.
    pub fn rules(&self) -> RuleSet {
        self.rules
    }

    pub fn set_rules(&mut self, rules: RuleSet) {
        self.rules = rules;
    }

    pub fn enable_rule(&mut self, rule: GrammarRule) {
        self.rules.enable(rule);
    }

    pub fn disable_rule(&mut self, rule: GrammarRule) {
        self.rules.disable(rule);
    }

    /// Also runs every experimental rule.
    pub fn enable_experimental(&mut self) {
        self.rules = self.rules.with_experimental();
    }

    /// Performs full grammatical error detection (GED) and returns all findings with explanations.
    pub fn check(&self, text: &str) -> GrammarCheckResult {
        self.check_with_rules(text, self.rules)
    }

    /// [`check`](Self::check) with `rules` instead of the checker's own.
    pub fn check_with_rules(&self, text: &str, rules: RuleSet) -> GrammarCheckResult {
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
            rules,
            &mut findings,
        );

        // Pass 2: Phonological Suffix Rules Engine
        phonological_rules::PhonologicalRuleEngine::check(
            text,
            &tokens,
            &token_spans,
            &self.morphology,
            rules,
            &mut findings,
        );

        // Pass 3: Morphosyntactic Agreement & Concord Engine, one sentence at a time
        agreement::AgreementRuleEngine::check(
            text,
            &tokens,
            &token_spans,
            &sentence_ranges(&tokens),
            &self.morphology,
            &self.disambiguator,
            &self.parser,
            rules,
            &mut findings,
        );

        // Pass 4: Lexicon & Compound Verb Rules Engine
        lexicon_rules::LexiconRuleEngine::check(text, &tokens, &token_spans, rules, &mut findings);

        // The engines skip the rules that are off; this keeps a finding of one from slipping out.
        findings.retain(|f| rules.contains(f.rule));

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

        // A finding's original_text is the text its offsets cover, whatever spacing the rule
        // read between its words ("hiç\nbir"; review 3 BIND3-09).
        for f in &mut findings {
            if let Some(span) = text.get(f.start_offset..f.end_offset) {
                if f.original_text != span {
                    f.original_text = span.to_string();
                }
            }
        }

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

    /// [`correct`](Self::correct) with `rules` instead of the checker's own.
    pub fn correct_with_rules(&self, text: &str, rules: RuleSet) -> String {
        self.check_with_rules(text, rules).corrected
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

/// The token ranges of the sentences of `tokens`: a sentence ends after a run of `.`, `!`,
/// `?` or `…` tokens. The tokenizer keeps abbreviations such as "Dr." in one token.
fn sentence_ranges(tokens: &[&str]) -> Vec<Range<usize>> {
    let is_end = |t: &str| matches!(t, "." | "!" | "?" | "…");
    let mut ranges = Vec::new();
    let mut start = 0;
    for i in 0..tokens.len() {
        if is_end(tokens[i]) && !tokens.get(i + 1).is_some_and(|t| is_end(t)) {
            ranges.push(start..i + 1);
            start = i + 1;
        }
    }
    if start < tokens.len() {
        ranges.push(start..tokens.len());
    }
    ranges
}

/// `fixed`, a lower-case correction of `original`, in `original`'s case: all capitals for
/// "KİTAPDA", a capital first letter for "Kitapda", as it is otherwise. Only the first letter
/// of a multi-word correction is capitalised ("Yada" -> "Ya da", not "Ya Da").
pub(crate) fn match_case(original: &str, fixed: &str) -> String {
    let mut letters = original.chars().filter(|c| c.is_alphabetic());
    let first_upper = letters.next().is_some_and(char::is_uppercase);
    let rest: Vec<char> = letters.collect();
    if first_upper && !rest.is_empty() && rest.iter().all(|c| c.is_uppercase()) {
        crate::phonology::to_turkish_upper(fixed)
    } else if first_upper {
        capitalize_first(fixed)
    } else {
        fixed.to_string()
    }
}

/// `s` with its first character in Turkish upper case (i -> İ, ı -> I).
pub(crate) fn capitalize_first(s: &str) -> String {
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
