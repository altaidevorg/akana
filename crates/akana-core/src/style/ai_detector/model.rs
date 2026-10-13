//! Embedded gradient-boosted tree model (`data/ai_detector/model.json`).
//!
//! The file is generated from a trained LightGBM booster by the detector's training code
//! (kept outside this repository, see the module docs of [`super`]) and holds everything the
//! runtime needs besides the token counts: the model inputs in order with their
//! standardisation and report texts, the Platt calibration, the decision thresholds, the
//! mined AI patterns and the trees.
//!
//! Trees are flattened arrays: internal node `k` splits on `split_feature[k]` at
//! `threshold[k]` (`z <= threshold` goes left, as in LightGBM numerical splits; the model
//! inputs are always finite, so missing-value handling never applies). A child index `c >= 0`
//! is an internal node, `c < 0` is the leaf `!c`. `internal_value[k]` is the expected value
//! at node `k`, used for path (Saabas) contributions: on the path of every tree, the change
//! of the expected value from a node to its child is credited to the node's split feature,
//! so `bias() + Σ contributions == raw_score()` exactly.

use std::sync::LazyLock;

use serde::Deserialize;

/// Number of model inputs: 68 content features, 5 pattern features and 2 token-transition
/// features.
pub(super) const N_INPUTS: usize = 75;

/// One model input with its standardisation and the texts shown in reports.
#[derive(Debug, Deserialize)]
pub(super) struct InputSpec {
    /// Machine name (e.g. `commas_per_sentence`, `pat_evidence`, `wordlm_llr_mean`).
    pub key: String,
    /// Turkish display name.
    pub name: String,
    /// Turkish one-sentence description.
    pub desc: String,
    pub mean: f64,
    pub scale: f64,
    /// Mean value on AI / human texts of the training data (for report sentences).
    pub ai_mean: f64,
    pub human_mean: f64,
}

/// Where a mined pattern is matched.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum PatternPlace {
    /// First 1–4 words of the text.
    Opener,
    /// Last 1–5 words of the text.
    Closer,
    /// First 1–3 words of a sentence.
    SentStart,
    /// Any 2–4 consecutive words.
    Phrase,
}

impl PatternPlace {
    /// Name in `model.json`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Opener => "opener",
            Self::Closer => "closer",
            Self::SentStart => "sent_start",
            Self::Phrase => "phrase",
        }
    }
}

/// A mined AI pattern.
#[derive(Debug, Deserialize)]
pub(super) struct PatternSpec {
    pub kind: PatternPlace,
    pub pattern: String,
    /// Share of AI answers / human texts containing it in the mining data.
    pub ai_rate: f64,
    pub human_rate: f64,
    /// `ln((ai_rate + 1e-4) / (human_rate + 1e-4))`.
    pub weight: f64,
}

#[derive(Debug, Deserialize)]
struct Tree {
    split_feature: Vec<usize>,
    threshold: Vec<f64>,
    left: Vec<i32>,
    right: Vec<i32>,
    internal_value: Vec<f64>,
    leaf_value: Vec<f64>,
}

#[derive(Debug, Deserialize)]
struct Platt {
    a: f64,
    b: f64,
}

/// Parameters of the token-transition model (counts are in `tokens.bin.zz`).
#[derive(Debug, Deserialize)]
pub(super) struct LmSpec {
    /// Tokenizer the counts were made with; must be `"akana"` (the embedding tokenizer).
    pub tokenizer: String,
    /// Interpolation weight of the bigram estimate.
    pub lambda: f64,
    /// Number of model token ids, including 0 (start) and 1 (unknown).
    pub vocab_size: usize,
    /// Tokenizer ids of model ids `2..vocab_size`.
    pub tokens: Vec<u32>,
}

/// The embedded detector model.
#[derive(Debug, Deserialize)]
pub(super) struct Model {
    pub model_version: String,
    pub features: Vec<InputSpec>,
    calibration: Platt,
    /// Flag threshold on the calibrated probability (1% false alarms on held-out humans).
    pub threshold: f64,
    /// Below this probability the text is called human; between it and `threshold`: uncertain.
    pub uncertain_low: f64,
    /// Content words below which the detector makes no decision.
    pub min_words: usize,
    pub patterns: Vec<PatternSpec>,
    trees: Vec<Tree>,
    pub lm: LmSpec,
}

static MODEL_JSON: &str = include_str!("../../../data/ai_detector/model.json");

/// The model embedded in this build. Invalid embedded data is a build defect, so it panics
/// on first use (like the embedded embedding model) instead of scoring with a broken model.
pub(super) static MODEL: LazyLock<Model> = LazyLock::new(|| {
    let model: Model =
        serde_json::from_str(MODEL_JSON).expect("embedded data/ai_detector/model.json is invalid");
    if let Err(e) = model.validate() {
        panic!("embedded data/ai_detector/model.json: {e}");
    }
    model
});

impl Model {
    /// Checks the invariants the evaluation code indexes by.
    fn validate(&self) -> Result<(), String> {
        if self.features.len() != N_INPUTS {
            return Err(format!(
                "{} inputs, expected {N_INPUTS}",
                self.features.len()
            ));
        }
        if let Some(f) = self
            .features
            .iter()
            .find(|f| !(f.mean.is_finite() && f.scale.is_finite() && f.scale > 0.0))
        {
            return Err(format!("input {} has an invalid standardisation", f.key));
        }
        if self.trees.is_empty() {
            return Err("no trees".into());
        }
        for (i, t) in self.trees.iter().enumerate() {
            let n = t.split_feature.len();
            // Internal children point forward (`c > k`), so evaluation always ends in a leaf.
            let ok_child = |k: usize, c: i32| {
                (c >= 0 && (c as usize) > k && (c as usize) < n)
                    || (c < 0 && ((!c) as usize) < t.leaf_value.len())
            };
            let shapes_ok = t.threshold.len() == n
                && t.left.len() == n
                && t.right.len() == n
                && t.internal_value.len() == n
                && t.leaf_value.len() == n + 1;
            if !shapes_ok
                || t.split_feature.iter().any(|&f| f >= N_INPUTS)
                || !(0..n).all(|k| ok_child(k, t.left[k]) && ok_child(k, t.right[k]))
                || !t
                    .threshold
                    .iter()
                    .chain(&t.leaf_value)
                    .all(|v| v.is_finite())
            {
                return Err(format!("tree {i} is malformed"));
            }
        }
        Ok(())
    }

    /// Standardises a raw input vector (model input order).
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
    #[cfg(test)]
    pub fn bias(&self) -> f64 {
        self.trees
            .iter()
            .map(|t| t.internal_value.first().copied().unwrap_or(t.leaf_value[0]))
            .sum()
    }

    /// Per-input path contributions (model input order); `bias() + Σ == raw_score()`.
    pub fn contributions(&self, z: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; N_INPUTS];
        for t in &self.trees {
            t.contribute(z, &mut out);
        }
        out
    }

    /// Calibrated AI probability of a raw score (Platt scaling).
    pub fn probability(&self, raw: f64) -> f64 {
        1.0 / (1.0 + (-(self.calibration.a * raw + self.calibration.b)).exp())
    }
}

/// A step from an internal node: the next internal node or a leaf.
enum Step {
    Node(usize),
    Leaf(usize),
}

impl Tree {
    fn step(&self, node: usize, z: &[f64]) -> Step {
        let child = if z[self.split_feature[node]] <= self.threshold[node] {
            self.left[node]
        } else {
            self.right[node]
        };
        if child >= 0 {
            Step::Node(child as usize)
        } else {
            Step::Leaf((!child) as usize)
        }
    }

    fn leaf(&self, z: &[f64]) -> usize {
        if self.split_feature.is_empty() {
            return 0;
        }
        let mut node = 0;
        loop {
            match self.step(node, z) {
                Step::Node(next) => node = next,
                Step::Leaf(leaf) => return leaf,
            }
        }
    }

    fn contribute(&self, z: &[f64], out: &mut [f64]) {
        if self.split_feature.is_empty() {
            return;
        }
        let mut node = 0;
        loop {
            let feature = self.split_feature[node];
            let parent = self.internal_value[node];
            match self.step(node, z) {
                Step::Node(next) => {
                    out[feature] += self.internal_value[next] - parent;
                    node = next;
                }
                Step::Leaf(leaf) => {
                    out[feature] += self.leaf_value[leaf] - parent;
                    return;
                }
            }
        }
    }
}
