//! Turkish AI-generated-text detector.
//!
//! **The output is not proof of authorship.** [`ai_text_detect`] returns a calibrated
//! probability from 500 gradient-boosted trees (model `akana_v1`, trained on OpenAI and
//! Claude answers versus Turkish human text written before November 2022), the decision and
//! its explanation. Measured on held-out golden sets at 1% false alarms on human chat text:
//! about 95 of 100 OpenAI texts caught. Poems get no decision, academic text only a warning,
//! texts under 65 content words are not judged.
//!
//! Inputs of the trees (75 numbers):
//!
//! 1. 68 content features ([`features`]) on cleaned text ([`preprocess()`]).
//! 2. 5 features from mined AI patterns ("elbette", "önemli bir rol oynar") on the raw text.
//! 3. 2 BERTurk token-transition features: how much more likely each piece transition is
//!    under the AI bigram model than under the human one.
//!
//! The explanation holds per-feature path contributions, the found patterns, per-piece
//! token scores and, for long texts, sliding 80-word sections scored on their own (display
//! only). [`why_ai_report`] renders it as a Turkish "why AI" report; the humanizer
//! ([`crate::style::TurkishHumanizer::humanize`]) builds its rewrite prompt from it.
//!
//! Embedded data (`data/aidetect/`, exported by `tr-ai-detect/scripts/export_akana_model.py`):
//! `model.json` (features, calibration, thresholds, patterns, trees, token list) and
//! `tokens.bin.zz` (per-class token counts and the BERTurk WordPiece vocabulary of
//! `dbmdz/bert-base-turkish-uncased`, MIT License, see `data/aidetect/LICENSE-BERTurk`).

// Cleaning and content features are public for the Python bindings (model training) and
// the tests only; the API is `ai_text_detect` and `why_ai_report`.
#[doc(hidden)]
pub mod features;
#[doc(hidden)]
pub mod preprocess;

#[doc(hidden)]
pub use features::{content_features, extract_features, FEATURE_NAMES};
#[doc(hidden)]
pub use preprocess::{preprocess, PreprocessOptions};

use ahash::{AHashMap, AHashSet};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::io::Read;
use std::sync::LazyLock;

// ======================================================================================
// Detection and explanation
// ======================================================================================

/// Section length and step (words) of the sliding sections; texts shorter than
/// `SECTION_MIN_WORDS` get none.
const SECTION_WORDS: usize = 80;
const SECTION_STEP: usize = 40;
const SECTION_MIN_WORDS: usize = 120;

/// Decision of the detector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AiVerdict {
    /// Probability at or above the flag threshold.
    Ai,
    /// Between the human cut-off and the flag threshold.
    Uncertain,
    /// Below the human cut-off.
    Human,
    /// Fewer content words than the model's minimum: no decision.
    TooShort,
    /// Poem: no decision (see [`ScopeWarning`]). Academic text only gets a warning.
    OutOfScope,
}

impl AiVerdict {
    /// Turkish label.
    pub fn label_tr(self) -> &'static str {
        match self {
            Self::Ai => "AI",
            Self::Uncertain => "emin değil",
            Self::Human => "insan",
            Self::TooShort => "çok kısa",
            Self::OutOfScope => "kapsam dışı",
        }
    }
}

/// Text type the model is not reliable on.
#[derive(Debug, Clone, Serialize)]
pub struct ScopeWarning {
    /// `"şiir"` or `"akademik"`.
    pub kind: String,
    pub message: String,
}

/// One feature's value and its share of the decision.
#[derive(Debug, Clone, Serialize)]
pub struct FeatureContribution {
    pub key: String,
    pub name: String,
    pub group: String,
    pub desc: String,
    pub value: f64,
    pub ai_mean: Option<f64>,
    pub human_mean: Option<f64>,
    /// Path contribution to the raw score: positive pushes towards AI.
    pub contribution: f64,
}

/// A mined AI pattern found in the text.
#[derive(Debug, Clone, Serialize)]
pub struct FoundPattern {
    /// Turkish kind: giriş, kapanış, cümle başı, kalıp ifade.
    pub kind: String,
    pub pattern: String,
    /// Share of AI answers / human texts containing it.
    pub ai_rate: f64,
    pub human_rate: f64,
}

/// One BERTurk piece and its transition score (`> 0`: the AI model finds it likelier).
#[derive(Debug, Clone, Serialize)]
pub struct TokenScore {
    pub token: String,
    pub score: f64,
}

/// A sliding section of a long text, scored on its own.
#[derive(Debug, Clone, Serialize)]
pub struct Section {
    /// 1-based word positions (whitespace-separated words) and character offsets.
    pub start_word: usize,
    pub end_word: usize,
    pub start_char: usize,
    pub end_char: usize,
    pub text: String,
    pub ai_probability: f64,
}

/// Full result of [`ai_text_detect`].
#[derive(Debug, Clone, Serialize)]
pub struct AiAnalysis {
    pub model_version: String,
    /// Calibrated AI probability (0–1). Reported even when no decision is made.
    pub ai_probability: f64,
    pub verdict: AiVerdict,
    pub scope: Option<ScopeWarning>,
    pub threshold: f64,
    pub uncertain_low: f64,
    pub n_words: usize,
    pub min_words: usize,
    /// Raw tree score and the score before any split (`bias + Σ contributions = raw_score`).
    pub raw_score: f64,
    pub bias: f64,
    /// All model features, largest absolute contribution first.
    pub features: Vec<FeatureContribution>,
    /// Found patterns, most AI-typical first.
    pub patterns: Vec<FoundPattern>,
    pub tokens: Vec<TokenScore>,
    pub sections: Vec<Section>,
    /// Turkish "why AI" report (markdown), see [`why_ai_report`].
    pub report: String,
}

impl AiAnalysis {
    pub fn is_ai(&self) -> bool {
        self.verdict == AiVerdict::Ai
    }
}

struct Scored {
    x: Vec<f64>,
    hits: Vec<&'static PatternSpec>,
    tokens: Vec<(String, f64)>,
    clean: String,
}

fn score_parts(text: &str) -> Scored {
    let clean = preprocess(text, &PreprocessOptions::default());
    let content = extract_features(&clean);
    let hits = found_patterns(text);
    let pat = pattern_features(&hits);
    let tokens = transition_scores(text);
    let lm = transition_features(&tokens);
    let value = |key: &str| -> f64 {
        if let Some(i) = FEATURE_NAMES.iter().position(|n| *n == key) {
            return content.get(i).copied().unwrap_or(0.0);
        }
        match key {
            "pat_opener" => pat[0],
            "pat_closer" => pat[1],
            "pat_sent_start_n" => pat[2],
            "pat_phrase_n" => pat[3],
            "pat_evidence" => pat[4],
            "wordlm_llr_mean" => lm[0],
            "wordlm_ai_share" => lm[1],
            _ => 0.0,
        }
    };
    let x = MODEL.features.iter().map(|f| value(&f.key)).collect();
    Scored {
        x,
        hits,
        tokens,
        clean,
    }
}

#[doc(hidden)]
/// Raw (unscaled) model feature vector of a text, in model feature order
/// ([`Model::features`]).
pub fn ai_feature_vector(text: &str) -> Vec<f64> {
    score_parts(text).x
}

#[doc(hidden)]
/// Calibrated AI probability of a text, without explanation or decision.
pub fn ai_probability(text: &str) -> f64 {
    let z = MODEL.standardize(&score_parts(text).x);
    MODEL.probability(MODEL.raw_score(&z))
}

const ACADEMIC_CUES: &[&str] = &[
    "bu çalışma",
    "bu araştırma",
    "çalışmanın amacı",
    "araştırmanın amacı",
    "yöntem",
    "bulgular",
    "anahtar kelimeler",
    "örneklem",
    "katılımcı",
    "literatür",
    "nicel",
    "nitel",
    "istatistiksel",
    "elde edilen",
    "sonuç olarak",
    "ölçeği",
    "analiz edil",
    "değerlendiril",
    "incelen",
];

static LIST_LINE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\s*([-*•–]|\d+[.)]|#)|\*\*").unwrap());

#[doc(hidden)]
/// Poem or academic text? A poem blocks the decision, academic text only gets the warning.
/// On the golden sets the rules catch 80% of human poems and 66% of human abstracts; the
/// poem rule wrongly flags about 1 in 100 chat answers, the academic rule about 15 in 100
/// OpenAI answers.
pub fn out_of_scope(text: &str) -> Option<ScopeWarning> {
    let lines: Vec<&str> = text
        .split('\n')
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    let is_list = lines.iter().filter(|l| LIST_LINE.is_match(l)).count() >= 2;
    if lines.len() >= 4 && !is_list {
        let mut lens: Vec<usize> = lines.iter().map(|l| l.chars().count()).collect();
        lens.sort_unstable();
        let med = lens[lens.len() / 2];
        let n = lines.len() as f64;
        let words_per_line = lines
            .iter()
            .map(|l| l.split_whitespace().count())
            .sum::<usize>() as f64
            / n;
        let ends = lines
            .iter()
            .filter(|l| l.ends_with('.') || l.ends_with(':'))
            .count() as f64
            / n;
        if med <= 60 && words_per_line <= 9.0 && ends <= 0.5 {
            return Some(ScopeWarning {
                kind: "şiir".into(),
                message: "Bu metin şiir gibi görünüyor. Testlerimizde model şiirde güvenilir \
                          değil: insan şiirlerinin bir kısmına yanlış alarm verdi ve AI \
                          şiirlerinin çoğunu kaçırdı. Sonucu kullanmayın."
                    .into(),
            });
        }
    }
    let low = text.to_lowercase();
    if ACADEMIC_CUES.iter().filter(|c| low.contains(*c)).count() >= 2 {
        return Some(ScopeWarning {
            kind: "akademik".into(),
            message: "Bu metin akademik bir metin gibi görünüyor. Testlerimizde model akademik \
                      özetlerde güvenilir değil: insan özetlerinin yaklaşık %13'üne yanlış \
                      alarm verdi. Sonuca dikkatle bakın."
                .into(),
        });
    }
    None
}

fn kind_tr(kind: &str) -> &'static str {
    match kind {
        "opener" => "giriş",
        "closer" => "kapanış",
        "sent_start" => "cümle başı",
        _ => "kalıp ifade",
    }
}

/// Sliding sections of a long text, each scored on its own (display only).
fn sections(text: &str) -> Vec<Section> {
    let spans: Vec<(usize, usize)> = text
        .split_whitespace()
        .map(|w| {
            let start = w.as_ptr() as usize - text.as_ptr() as usize;
            (start, start + w.len())
        })
        .collect();
    let n = spans.len();
    if n < SECTION_MIN_WORDS {
        return Vec::new();
    }
    let mut starts: Vec<usize> = (0..=n - SECTION_WORDS).step_by(SECTION_STEP).collect();
    if starts.last() != Some(&(n - SECTION_WORDS)) {
        starts.push(n - SECTION_WORDS);
    }
    starts
        .into_iter()
        .map(|a| {
            let b = a + SECTION_WORDS;
            let (c0, c1) = (spans[a].0, spans[b - 1].1);
            let piece = &text[c0..c1];
            Section {
                start_word: a + 1,
                end_word: b,
                start_char: text[..c0].chars().count(),
                end_char: text[..c1].chars().count(),
                text: piece.to_string(),
                ai_probability: ai_probability(piece),
            }
        })
        .collect()
}

/// Detects AI-written Turkish text and explains the result, including the Turkish "why AI"
/// report ([`AiAnalysis::report`]). **Not proof of authorship.**
pub fn ai_text_detect(text: &str) -> AiAnalysis {
    let mut a = analyze(text, true);
    a.report = why_ai_report(&a);
    a
}

/// [`ai_text_detect`] without the report; `with_sections` adds the sliding-section scores
/// of long texts (one extra model run per 40 words).
fn analyze(text: &str, with_sections: bool) -> AiAnalysis {
    let m = &*MODEL;
    let s = score_parts(text);
    let z = m.standardize(&s.x);
    let raw = m.raw_score(&z);
    let contrib = m.contributions(&z);
    let p = m.probability(raw);
    let n_words = pattern_words(&s.clean).len();
    let scope = out_of_scope(text);
    // A poem blocks the decision; academic text is only flagged (the academic cues also
    // occur in ~15 of 100 OpenAI chat answers, which would otherwise go undecided).
    let verdict = if scope.as_ref().is_some_and(|s| s.kind == "şiir") {
        AiVerdict::OutOfScope
    } else if n_words < m.min_words {
        AiVerdict::TooShort
    } else if p >= m.threshold {
        AiVerdict::Ai
    } else if p < m.uncertain_low {
        AiVerdict::Human
    } else {
        AiVerdict::Uncertain
    };
    let mut features: Vec<FeatureContribution> = m
        .features
        .iter()
        .zip(&s.x)
        .zip(&contrib)
        .map(|((f, &value), &c)| FeatureContribution {
            key: f.key.clone(),
            name: f.name.clone(),
            group: f.group.clone(),
            desc: f.desc.clone(),
            value,
            ai_mean: f.ai_mean,
            human_mean: f.human_mean,
            contribution: c,
        })
        .collect();
    features.sort_by(|a, b| b.contribution.abs().total_cmp(&a.contribution.abs()));
    let mut hits = s.hits;
    hits.sort_by(|a, b| b.ai_rate.total_cmp(&a.ai_rate));
    let mut seen = std::collections::HashSet::new();
    let patterns = hits
        .into_iter()
        .filter(|p| seen.insert(p.pattern.as_str()))
        .map(|p| FoundPattern {
            kind: kind_tr(&p.kind).into(),
            pattern: p.pattern.clone(),
            ai_rate: p.ai_rate,
            human_rate: p.human_rate,
        })
        .collect();
    AiAnalysis {
        model_version: m.model_version.clone(),
        ai_probability: p,
        verdict,
        scope,
        threshold: m.threshold,
        uncertain_low: m.uncertain_low,
        n_words,
        min_words: m.min_words,
        raw_score: raw,
        bias: m.bias(),
        features,
        patterns,
        tokens: s
            .tokens
            .into_iter()
            .map(|(token, score)| TokenScore { token, score })
            .collect(),
        sections: if with_sections {
            sections(text)
        } else {
            Vec::new()
        },
        report: String::new(),
    }
}

fn pct(x: f64) -> String {
    let v = x * 100.0;
    if v > 0.0 && v < 0.1 {
        "%0,1'den az".into()
    } else if v > 99.5 && v < 100.0 {
        "%99'dan fazla".into()
    } else if v >= 10.0 || v == 0.0 {
        format!("%{v:.0}")
    } else {
        format!("%{v:.1}").replace('.', ",")
    }
}

fn num(x: f64) -> String {
    let s = if x.abs() >= 100.0 {
        format!("{x:.0}")
    } else if x.abs() >= 10.0 {
        format!("{x:.1}")
    } else {
        format!("{x:.2}")
    };
    s.replace('.', ",")
}

/// One sentence about a feature: its value here and the typical AI / human values.
pub(crate) fn feature_sentence(f: &FeatureContribution) -> String {
    match (f.ai_mean, f.human_mean) {
        (Some(a), Some(h)) => format!(
            "**{}**: bu metinde {}; AI metinlerinde ortalama {}, insan metinlerinde {}. ({})",
            f.name,
            num(f.value),
            num(a),
            num(h),
            f.desc.trim_end_matches('.')
        ),
        _ => format!("**{}**: bu metinde {}. ({})", f.name, num(f.value), f.desc),
    }
}

/// Features pushing towards AI, strongest first (at most `k`).
pub(crate) fn ai_reasons(a: &AiAnalysis, k: usize) -> Vec<&FeatureContribution> {
    a.features
        .iter()
        .filter(|f| f.contribution > 0.0)
        .take(k)
        .collect()
}

/// Turkish markdown "why AI" report of an analysis.
pub fn why_ai_report(a: &AiAnalysis) -> String {
    let mut out = String::from("# AI tespit raporu\n\n");
    out.push_str(&format!(
        "**Sonuç:** {} — AI olasılığı {} (AI demek için eşik {}, {} altı insan).\n\n",
        a.verdict.label_tr(),
        pct(a.ai_probability),
        pct(a.threshold),
        pct(a.uncertain_low)
    ));
    if a.verdict == AiVerdict::TooShort {
        out.push_str(&format!(
            "Metin {} kelime; model en az {} kelimede karar veriyor.\n\n",
            a.n_words, a.min_words
        ));
    }
    if let Some(s) = &a.scope {
        out.push_str(&format!("⚠ {}\n\n", s.message));
    }
    let reasons = ai_reasons(a, 6);
    if !reasons.is_empty() {
        out.push_str("## Neden AI'a benziyor?\n\n");
        for f in reasons {
            out.push_str(&format!("- {}\n", feature_sentence(f)));
        }
        out.push('\n');
    }
    if !a.patterns.is_empty() {
        out.push_str("## Bulunan AI kalıpları\n\n");
        for p in a.patterns.iter().take(10) {
            out.push_str(&format!(
                "- «{}» ({}): AI cevaplarında {}, insan metinlerinde {} oranında geçiyor.\n",
                p.pattern,
                p.kind,
                pct(p.ai_rate),
                pct(p.human_rate)
            ));
        }
        out.push('\n');
    }
    let hot: Vec<&Section> = a
        .sections
        .iter()
        .filter(|s| s.ai_probability >= a.threshold)
        .collect();
    if !hot.is_empty() {
        out.push_str("## En çok AI'a benzeyen bölümler\n\n");
        for s in hot.iter().take(3) {
            out.push_str(&format!(
                "- {}–{}. kelimeler ({}): «{}…»\n",
                s.start_word,
                s.end_word,
                pct(s.ai_probability),
                s.text.chars().take(80).collect::<String>()
            ));
        }
        out.push('\n');
    }
    let human: Vec<&FeatureContribution> = a
        .features
        .iter()
        .filter(|f| f.contribution < 0.0)
        .take(3)
        .collect();
    if !human.is_empty() {
        out.push_str("## İnsana benzeyen yanlar\n\n");
        for f in human {
            out.push_str(&format!("- {}\n", feature_sentence(f)));
        }
        out.push('\n');
    }
    out.push_str(
        "_Bu bir olasılıktır, kanıt değildir. Model OpenAI ve Claude metinleriyle eğitildi; \
         hiç görmediği yeni modellerde daha zayıftır._\n",
    );
    out
}

// ======================================================================================
// Tree model
// ======================================================================================
// Embedded gradient-boosted tree model of the AI-text detector (LightGBM, exported by
// `tr-ai-detect/scripts/export_akana_model.py` into `data/aidetect/model.json`).
//
// Prediction is the sum of one leaf value per tree. Explanations are path contributions
// (Saabas): on the path of every tree, the change of the expected value from a node to its
// child is credited to the node's split feature. Contributions plus [`Model::bias`] add up
// to the raw score exactly.

#[doc(hidden)]
/// One model feature with its scaling and the texts shown in reports.
#[derive(Debug, Clone, Deserialize)]
pub struct FeatureSpec {
    /// Machine name (e.g. `commas_per_sentence`, `pat_evidence`, `wordlm_llr_mean`).
    pub key: String,
    /// Turkish display name.
    pub name: String,
    /// Turkish group name.
    pub group: String,
    /// Turkish one-sentence description.
    pub desc: String,
    pub mean: f64,
    pub scale: f64,
    /// Mean value on AI / human texts of the evaluation set (for report sentences).
    pub ai_mean: Option<f64>,
    pub human_mean: Option<f64>,
}

#[doc(hidden)]
/// A mined AI pattern (opener, closer, sentence start or phrase).
#[derive(Debug, Clone, Deserialize)]
pub struct PatternSpec {
    pub kind: String,
    pub pattern: String,
    pub ai_rate: f64,
    pub human_rate: f64,
    /// log((ai_rate + 1e-4) / (human_rate + 1e-4)).
    pub weight: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct Tree {
    split_feature: Vec<usize>,
    threshold: Vec<f64>,
    /// Child index: >= 0 internal node, < 0 leaf `!child`.
    left: Vec<i64>,
    right: Vec<i64>,
    internal_value: Vec<f64>,
    leaf_value: Vec<f64>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct Platt {
    pub a: f64,
    pub b: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct LmSpec {
    pub lambda: f64,
    pub vocab_size: usize,
    /// Token strings of ids 2..vocab_size (0 = start, 1 = unknown).
    pub tokens: Vec<String>,
}

#[doc(hidden)]
/// The embedded detector model.
#[derive(Debug, Clone, Deserialize)]
pub struct Model {
    pub model_version: String,
    pub features: Vec<FeatureSpec>,
    pub(crate) calibration: Platt,
    /// Flag threshold on the calibrated probability (1% false alarms on held-out humans).
    pub threshold: f64,
    /// Below this probability the text is called human; between it and `threshold`: uncertain.
    pub uncertain_low: f64,
    /// Content words below which the detector abstains.
    pub min_words: usize,
    pub patterns: Vec<PatternSpec>,
    pub(crate) trees: Vec<Tree>,
    pub(crate) lm: LmSpec,
}

static MODEL_JSON: &str = include_str!("../../data/aidetect/model.json");

/// The model embedded in this build. Invalid embedded data is a build defect, so it
/// panics on first use instead of silently scoring with an empty model.
#[doc(hidden)]
pub static MODEL: LazyLock<Model> = LazyLock::new(|| {
    let m: Model =
        serde_json::from_str(MODEL_JSON).expect("embedded data/aidetect/model.json is invalid");
    assert!(
        m.features.len() == N_MODEL_FEATURES && !m.trees.is_empty(),
        "embedded data/aidetect/model.json does not match this build"
    );
    m
});

/// Number of model inputs: 68 content + 5 pattern + 2 token-transition features.
const N_MODEL_FEATURES: usize = 75;

impl Model {
    /// Standardises a raw feature vector (model feature order).
    pub fn standardize(&self, x: &[f64]) -> Vec<f64> {
        self.features
            .iter()
            .zip(x)
            .map(|(f, &v)| (v - f.mean) / f.scale)
            .collect()
    }

    /// Raw score (sum of leaf values) of a standardised vector.
    pub fn raw_score(&self, z: &[f64]) -> f64 {
        self.trees.iter().map(|t| t.leaf_value[t.leaf(z)]).sum()
    }

    /// Sum of the trees' root expected values: the score before any split is applied.
    pub fn bias(&self) -> f64 {
        self.trees
            .iter()
            .map(|t| t.internal_value.first().copied().unwrap_or(0.0))
            .sum()
    }

    /// Per-feature path contributions (model feature order); `bias() + sum == raw_score`.
    pub fn contributions(&self, z: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.features.len()];
        for t in &self.trees {
            t.contribute(z, &mut out);
        }
        out
    }

    /// Calibrated AI probability of a raw score.
    pub fn probability(&self, raw: f64) -> f64 {
        1.0 / (1.0 + (-(self.calibration.a * raw + self.calibration.b)).exp())
    }
}

impl Tree {
    fn go(&self, node: usize, z: &[f64]) -> i64 {
        let f = self.split_feature[node];
        let v = z.get(f).copied().unwrap_or(0.0);
        if v <= self.threshold[node] {
            self.left[node]
        } else {
            self.right[node]
        }
    }

    fn leaf(&self, z: &[f64]) -> usize {
        if self.split_feature.is_empty() {
            return 0;
        }
        let mut node = 0usize;
        loop {
            let next = self.go(node, z);
            if next < 0 {
                return (!next) as usize;
            }
            node = next as usize;
        }
    }

    fn contribute(&self, z: &[f64], out: &mut [f64]) {
        if self.split_feature.is_empty() {
            return;
        }
        let mut node = 0usize;
        loop {
            let parent_value = self.internal_value[node];
            let f = self.split_feature[node];
            let next = self.go(node, z);
            let child_value = if next < 0 {
                self.leaf_value[(!next) as usize]
            } else {
                self.internal_value[next as usize]
            };
            if let Some(slot) = out.get_mut(f) {
                *slot += child_value - parent_value;
            }
            if next < 0 {
                return;
            }
            node = next as usize;
        }
    }
}

// ======================================================================================
// Token-transition signal
// ======================================================================================
// Token-transition signal of the AI-text detector.
//
// Text is cut into BERTurk WordPiece pieces (`dbmdz/bert-base-turkish-uncased`, MIT; the
// same algorithm as Hugging Face `BertWordPieceTokenizer` with `lowercase=False,
// strip_accents=False, clean_text=True`) after flattening formatting: markdown symbols and
// list markers removed, digit runs → `0`, whitespace collapsed, Turkish lowercase.
//
// Two bigram models (human texts, AI texts) score every transition `prev → cur`:
// `P(cur | prev) = λ·big/hist + (1 − λ)·(uni + 1)/(N + V)` (λ = 0 after the start symbol or
// an unseen history). The per-transition score is `d = log P_AI − log P_human`; the
// detector uses its mean and the share of transitions with `d > 0`.

static TOKENS_BIN: &[u8] = include_bytes!("../../data/aidetect/tokens.bin.zz");

const MAX_WORD_CHARS: usize = 100;
const UNK: &str = "[UNK]";

static MD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[*_`#>]+").unwrap());
static BULLET: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^\s*(?:[-*•–]|\d+[.)])\s+").unwrap());
static DIGITS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\d+").unwrap());
static SPACES: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\s+").unwrap());
static PUNCT: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^[\p{P}!-/:-@\[-`{-~]$").unwrap());
static OTHER: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\p{C}$").unwrap());

/// Turkish lowercase as in the training code: `I → ı`, `İ → i`, then Unicode lowercase.
fn tr_lower(s: &str) -> String {
    s.replace('I', "ı").replace('İ', "i").to_lowercase()
}

/// Formatting removed before tokenising.
fn flatten(text: &str) -> String {
    let t = MD.replace_all(text, " ");
    let t = BULLET.replace_all(&t, " ");
    let t = DIGITS.replace_all(&t, "0");
    tr_lower(SPACES.replace_all(&t, " ").trim())
}

fn is_cjk(c: char) -> bool {
    matches!(c as u32,
        0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0x20000..=0x2A6DF | 0x2A700..=0x2B73F
        | 0x2B740..=0x2B81F | 0x2B820..=0x2CEAF | 0xF900..=0xFAFF | 0x2F800..=0x2FA1F)
}

fn char_matches(re: &Regex, c: char) -> bool {
    let mut buf = [0u8; 4];
    re.is_match(c.encode_utf8(&mut buf))
}

/// BERT normalisation and pre-tokenisation: control characters dropped, whitespace splits,
/// every punctuation and CJK character is a word of its own.
fn bert_words(text: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut cur = String::new();
    for c in text.chars() {
        let space = matches!(c, ' ' | '\t' | '\n' | '\r') || c.is_whitespace();
        if !space && (c == '\0' || c == '\u{fffd}' || char_matches(&OTHER, c)) {
            continue;
        }
        if space {
            if !cur.is_empty() {
                words.push(std::mem::take(&mut cur));
            }
        } else if is_cjk(c) || char_matches(&PUNCT, c) {
            if !cur.is_empty() {
                words.push(std::mem::take(&mut cur));
            }
            words.push(c.to_string());
        } else {
            cur.push(c);
        }
    }
    if !cur.is_empty() {
        words.push(cur);
    }
    words
}

/// Greedy longest-match-first WordPiece; a word with an unmatchable part becomes `[UNK]`.
fn wordpiece(vocab: &AHashSet<String>, word: &str, out: &mut Vec<String>) {
    let chars: Vec<(usize, char)> = word.char_indices().collect();
    if chars.len() > MAX_WORD_CHARS {
        out.push(UNK.into());
        return;
    }
    let mut pieces = Vec::new();
    let mut start = 0;
    while start < chars.len() {
        let mut end = chars.len();
        let mut found = None;
        while start < end {
            let a = chars[start].0;
            let b = chars.get(end).map_or(word.len(), |x| x.0);
            let piece = if start > 0 {
                format!("##{}", &word[a..b])
            } else {
                word[a..b].to_string()
            };
            if vocab.contains(piece.as_str()) {
                found = Some(piece);
                break;
            }
            end -= 1;
        }
        match found {
            Some(p) => pieces.push(p),
            None => {
                out.push(UNK.into());
                return;
            }
        }
        start = end;
    }
    out.extend(pieces);
}

#[doc(hidden)]
/// BERTurk pieces of a text, as seen by the token-transition model.
pub fn berturk_tokens(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for w in bert_words(&flatten(text)) {
        wordpiece(&LM.vocab, &w, &mut out);
    }
    out
}

struct ClassCounts {
    n_tok: f64,
    uni: Vec<u32>,
    big_keys: Vec<u32>,
    big_counts: Vec<u32>,
}

impl ClassCounts {
    fn bigram(&self, key: u32) -> f64 {
        self.big_keys
            .binary_search(&key)
            .map_or(0.0, |i| f64::from(self.big_counts[i]))
    }
}

struct TokenLm {
    lambda: f64,
    v: usize,
    ids: HashMap<&'static str, u32>,
    /// 0 = human, 1 = AI.
    classes: Vec<ClassCounts>,
    /// BERTurk WordPiece vocabulary.
    vocab: AHashSet<String>,
}

fn parse_lm() -> Option<TokenLm> {
    let mut raw = Vec::new();
    flate2::read::ZlibDecoder::new(TOKENS_BIN)
        .read_to_end(&mut raw)
        .ok()?;
    let mut pos = 0usize;
    let mut take = |n: usize| -> Option<&[u8]> {
        let s = raw.get(pos..pos + n)?;
        pos += n;
        Some(s)
    };
    if take(5)? != b"AKLM2" {
        return None;
    }
    let u32s = |b: &[u8]| -> Vec<u32> {
        b.as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect()
    };
    let v = u32::from_le_bytes(take(4)?.try_into().ok()?) as usize;
    let mut classes = Vec::new();
    for _ in 0..2 {
        let n_tok = u64::from_le_bytes(take(8)?.try_into().ok()?) as f64;
        let uni = u32s(take(4 * v)?);
        let n_big = u32::from_le_bytes(take(4)?.try_into().ok()?) as usize;
        let big_keys = u32s(take(4 * n_big)?);
        let big_counts = u32s(take(4 * n_big)?);
        classes.push(ClassCounts {
            n_tok,
            uni,
            big_keys,
            big_counts,
        });
    }
    let n_vocab = u32::from_le_bytes(take(4)?.try_into().ok()?) as usize;
    let vocab = std::str::from_utf8(take(n_vocab)?)
        .ok()?
        .lines()
        .map(|l| l.trim_end_matches('\r').to_string())
        .collect();
    let spec = &MODEL.lm;
    if v != spec.vocab_size || spec.tokens.len() + 2 != v {
        return None;
    }
    let ids = spec
        .tokens
        .iter()
        .enumerate()
        .map(|(i, t)| (t.as_str(), (i + 2) as u32))
        .collect();
    Some(TokenLm {
        lambda: spec.lambda,
        v,
        ids,
        classes,
        vocab,
    })
}

static LM: LazyLock<TokenLm> = LazyLock::new(|| {
    parse_lm()
        .expect("embedded data/aidetect/tokens.bin.zz is invalid or does not match model.json")
});

impl TokenLm {
    fn logprob(&self, c: usize, prev: u32, cur: u32) -> f64 {
        let k = &self.classes[c];
        let p_uni = (f64::from(k.uni[cur as usize]) + 1.0) / (k.n_tok + self.v as f64);
        let hist = if prev == 0 {
            0.0
        } else {
            f64::from(k.uni[prev as usize])
        };
        if hist <= 0.0 {
            return p_uni.ln();
        }
        let big = k.bigram(prev * self.v as u32 + cur);
        (self.lambda * big / hist + (1.0 - self.lambda) * p_uni).ln()
    }
}

/// Per-piece transition scores: `(piece, log P_AI − log P_human)` for every piece of the
/// text (the first one is scored after the start symbol).
pub(crate) fn transition_scores(text: &str) -> Vec<(String, f64)> {
    let lm = &*LM;
    let mut prev = 0u32;
    berturk_tokens(text)
        .into_iter()
        .map(|t| {
            let cur = lm.ids.get(t.as_str()).copied().unwrap_or(1);
            let d = lm.logprob(1, prev, cur) - lm.logprob(0, prev, cur);
            prev = cur;
            (t, d)
        })
        .collect()
}

/// `[wordlm_llr_mean, wordlm_ai_share]` of a list of transition scores (zeros if empty).
fn transition_features(scores: &[(String, f64)]) -> [f64; 2] {
    if scores.is_empty() {
        return [0.0, 0.0];
    }
    let n = scores.len() as f64;
    let mean = scores.iter().map(|s| s.1).sum::<f64>() / n;
    let share = scores.iter().filter(|s| s.1 > 0.0).count() as f64 / n;
    [mean, share]
}

// ======================================================================================
// Mined AI patterns
// ======================================================================================
// Mined AI patterns of the detector: openers ("elbette"), closers, sentence starts and
// phrases ("önemli bir rol oynar") found far more often in AI answers than in human text.
//
// The list and each pattern's AI / human rate come from the embedded model
// ([`Model::patterns`]); this module finds them in a text and turns the hits
// into the five pattern features.

static WORD: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[\p{L}\p{Nl}\p{No}]+(?:'[\p{L}\p{Nl}\p{No}]+)?").unwrap());

static INDEX: LazyLock<AHashMap<(&'static str, &'static str), &'static PatternSpec>> =
    LazyLock::new(|| {
        MODEL
            .patterns
            .iter()
            .map(|p| ((p.kind.as_str(), p.pattern.as_str()), p))
            .collect()
    });

/// Turkish-lowercased words of a text, markdown symbols removed.
fn pattern_words(text: &str) -> Vec<String> {
    let low = tr_lower(&MD.replace_all(text, " "));
    WORD.find_iter(&low)
        .map(|m| m.as_str().to_string())
        .collect()
}

fn py_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// Sentence pieces: split after `.!?…` followed by whitespace, and at line breaks.
fn sentences(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut prev: Option<char> = None;
    let mut it = text.char_indices().peekable();
    while let Some((i, c)) = it.next() {
        let after_stop = matches!(prev, Some('.' | '!' | '?' | '…'));
        let sep = if after_stop && py_space(c) {
            Some(py_space as fn(char) -> bool)
        } else if c == '\n' {
            Some((|c| c == '\n') as fn(char) -> bool)
        } else {
            None
        };
        match sep {
            Some(cont) => {
                out.push(&text[start..i]);
                let mut end = i + c.len_utf8();
                let mut last = c;
                while let Some(&(j, d)) = it.peek() {
                    if !cont(d) {
                        break;
                    }
                    end = j + d.len_utf8();
                    last = d;
                    it.next();
                }
                start = end;
                prev = Some(last);
            }
            None => prev = Some(c),
        }
    }
    out.push(&text[start..]);
    out
}

/// Every candidate (kind, pattern) of a text, before matching against the mined list.
fn candidates(text: &str) -> BTreeSet<(&'static str, String)> {
    let mut found = BTreeSet::new();
    let w = pattern_words(text);
    for k in 1..=4.min(w.len()) {
        found.insert(("opener", w[..k].join(" ")));
    }
    for k in 1..=5.min(w.len()) {
        found.insert(("closer", w[w.len() - k..].join(" ")));
    }
    for sent in sentences(text) {
        let sw = pattern_words(sent);
        for k in 1..=3 {
            if sw.len() > k {
                found.insert(("sent_start", sw[..k].join(" ")));
            }
        }
    }
    for n in 2..=4 {
        for win in w.windows(n) {
            found.insert(("phrase", win.join(" ")));
        }
    }
    found
}

/// Mined patterns present in a text (each at most once).
fn found_patterns(text: &str) -> Vec<&'static PatternSpec> {
    candidates(text)
        .iter()
        .filter_map(|(k, p)| INDEX.get(&(*k, p.as_str())).copied())
        .collect()
}

/// `[pat_opener, pat_closer, pat_sent_start_n, pat_phrase_n, pat_evidence]`.
fn pattern_features(hits: &[&PatternSpec]) -> [f64; 5] {
    let count = |kind: &str| hits.iter().filter(|p| p.kind == kind).count() as f64;
    [
        f64::from(u8::from(count("opener") > 0.0)),
        f64::from(u8::from(count("closer") > 0.0)),
        count("sent_start"),
        count("phrase"),
        hits.iter().map(|p| p.weight).sum(),
    ]
}
