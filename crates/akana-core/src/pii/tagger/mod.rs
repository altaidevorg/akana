//! akana's learned PII tagger: the token classifiers trained by scripts/pii_train (round P3, on data
//! that may be published: ELECTRA-small for the fast mode, and BERTurk averaged with it on every
//! window for the accurate mode; published under CC BY-SA 4.0 as altaidevorg/akana-pii-tagger on
//! the Hugging Face Hub), run from their ONNX exports (scripts/pii_train/export_onnx.py) with RTen,
//! a pure-Rust CPU runtime, and decoded as scripts/pii_train/tagger.py decodes them.
//! scripts/pii_eval/rust_parity.py checks on the dev sets that the spans are the Python tagger's.
//!
//! The strict modes raise person, username and date tags by the operating point that conformal
//! risk control picked for α = 0.04 on the two real-text dev golds ([`STRICT_FAST_BIAS`],
//! [`STRICT_ACCURATE_BIAS`]).
//!
//! With the `pii-download` feature, [`PiiTagger::from_hub`] downloads the published models on
//! first use and caches them ([`hub`]).

mod decode;
#[cfg(feature = "pii-download")]
pub mod hub;
mod wordpiece;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use rten::{Model, RunOptions, ThreadPool, ValueView};

pub use decode::LABELS;
use decode::{Row, N_TAGS};
use wordpiece::WordPiece;

/// The fast mode's strict operating point (ELECTRA-small P3).
pub const STRICT_FAST_BIAS: f32 = 5.25;
/// The accurate mode's strict operating point (ELECTRA-small P3 averaged with BERTurk P3).
pub const STRICT_ACCURATE_BIAS: f32 = 2.75;
/// The labels a strict operating point raises.
pub const STRICT_LABELS: [&str; 3] = ["PERSON", "USERNAME", "DATE"];

/// The tagger's two modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// ELECTRA-small alone.
    Fast,
    /// ELECTRA-small and BERTurk, averaged on every window.
    Accurate,
}

impl Mode {
    /// This mode's strict operating point ([`TaggerOptions::strict`]).
    pub fn strict_bias(self) -> f32 {
        match self {
            Mode::Fast => STRICT_FAST_BIAS,
            Mode::Accurate => STRICT_ACCURATE_BIAS,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum TaggerError {
    #[error("{path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("{path}: {message}")]
    Tokenizer { path: PathBuf, message: String },
    #[error("{path}: {message}")]
    Model { path: PathBuf, message: String },
    #[error("no model directory was given")]
    NoModel,
    #[error("{0} tokenizes differently from the first model")]
    VocabularyMismatch(PathBuf),
    #[error("{0} is not one of the tagger's labels")]
    UnknownLabel(String),
    #[error("inference: {0}")]
    Inference(String),
    #[error("no cache directory for the models: set AKANA_CACHE_DIR")]
    NoCacheDir,
    #[error("{0} is not in the cache, and downloads are off (AKANA_OFFLINE or HF_HUB_OFFLINE)")]
    Offline(PathBuf),
    #[error("downloading {url}: {message}")]
    Download { url: String, message: String },
    #[error("{path}: {message}")]
    Integrity { path: PathBuf, message: String },
}

/// A value the tagger found: byte offsets in the text, and its label (one of [`LABELS`]).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct TaggedSpan {
    pub start: usize,
    pub end: usize,
    pub label: &'static str,
}

/// How the tagger decodes.
#[derive(Debug, Clone)]
pub struct TaggerOptions {
    /// The operating point: added to the B- and I- tags of `bias_labels`, or, when that is empty,
    /// taken off the O tag (raising every label alike). 0 is the trained point.
    pub bias: f32,
    pub bias_labels: Vec<String>,
    /// Leave call-centre and service numbers and role mailboxes unmasked (the default policy).
    pub public_policy: bool,
    /// Inference threads; `None` uses RTen's default, the machine's physical cores.
    pub threads: Option<usize>,
}

impl Default for TaggerOptions {
    fn default() -> Self {
        Self {
            bias: 0.0,
            bias_labels: Vec::new(),
            public_policy: true,
            threads: None,
        }
    }
}

impl TaggerOptions {
    /// A strict mode: `bias` ([`STRICT_FAST_BIAS`] or [`STRICT_ACCURATE_BIAS`]) on person, username
    /// and date tags.
    pub fn strict(bias: f32) -> Self {
        Self {
            bias,
            bias_labels: STRICT_LABELS.iter().map(|l| l.to_string()).collect(),
            ..Self::default()
        }
    }
}

/// The tagger: one small model, or the small model and bigger ones averaged on every window.
pub struct PiiTagger {
    tokenizer: WordPiece,
    /// The small model first, then the bigger ones.
    models: Vec<Model>,
    bias: f32,
    bias_labels: Vec<usize>,
    public_policy: bool,
    run_options: Option<RunOptions>,
}

impl PiiTagger {
    /// Loads the models from their directories, each holding `model.onnx` and `tokenizer.json`:
    /// the small model's first, then those of any bigger models to average with it. All must
    /// share one vocabulary.
    pub fn load<P: AsRef<Path>>(
        model_dirs: &[P],
        options: TaggerOptions,
    ) -> Result<Self, TaggerError> {
        let mut tokenizer: Option<WordPiece> = None;
        let mut models = Vec::with_capacity(model_dirs.len());
        for dir in model_dirs {
            let dir = dir.as_ref();
            let path = dir.join("tokenizer.json");
            let json = std::fs::read_to_string(&path).map_err(|source| TaggerError::Io {
                path: path.clone(),
                source,
            })?;
            let t = WordPiece::from_json(&json)
                .map_err(|message| TaggerError::Tokenizer { path, message })?;
            match &tokenizer {
                None => tokenizer = Some(t),
                Some(first) if !first.same_vocabulary(&t) => {
                    return Err(TaggerError::VocabularyMismatch(dir.to_path_buf()))
                }
                Some(_) => {}
            }
            let path = dir.join("model.onnx");
            let model = Model::load_file(&path).map_err(|e| TaggerError::Model {
                message: e.to_string(),
                path,
            })?;
            models.push(model);
        }
        let tokenizer = tokenizer.ok_or(TaggerError::NoModel)?;
        let bias_labels = options
            .bias_labels
            .iter()
            .map(|l| {
                LABELS
                    .iter()
                    .position(|x| x == l)
                    .ok_or_else(|| TaggerError::UnknownLabel(l.clone()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let run_options = options.threads.map(|n| {
            RunOptions::default().with_thread_pool(Some(Arc::new(ThreadPool::with_num_threads(n))))
        });
        Ok(Self {
            tokenizer,
            models,
            bias: options.bias,
            bias_labels,
            public_policy: options.public_policy,
            run_options,
        })
    }

    /// The personal values in `text`, in order.
    pub fn tag(&self, text: &str) -> Result<Vec<TaggedSpan>, TaggerError> {
        let chars: Vec<char> = text.chars().collect();
        let tokens = self.tokenizer.tokenize(&wordpiece::model_text(&chars));
        let windows = decode::windows(tokens.len());
        let mut rows = Vec::with_capacity(windows.len());
        for &(start, end) in &windows {
            let mut ids = Vec::with_capacity(end - start + 2);
            ids.push(token_id(self.tokenizer.cls)?);
            for t in &tokens[start..end] {
                ids.push(token_id(t.id)?);
            }
            ids.push(token_id(self.tokenizer.sep)?);
            rows.push(self.window_rows(&ids)?);
        }
        let mut scores = decode::stitch(tokens.len(), &windows, &rows);
        decode::apply_bias(&mut scores, self.bias, &self.bias_labels);
        let tags = decode::viterbi(&scores);
        let offsets: Vec<(usize, usize)> = tokens.iter().map(|t| (t.start, t.end)).collect();
        let spans = decode::snap(&chars, &decode::spans_from_tags(&offsets, &tags));
        let mut byte_at: Vec<usize> = text.char_indices().map(|(b, _)| b).collect();
        byte_at.push(text.len());
        Ok(spans
            .into_iter()
            .filter_map(|(s, e, label)| {
                let (start, end) = (byte_at[s], byte_at[e]);
                let public = self.public_policy && decode::public_value(&text[start..end], label);
                (!public).then_some(TaggedSpan {
                    start,
                    end,
                    label: LABELS[label],
                })
            })
            .collect())
    }

    /// A window's content-token log-probabilities: each model's, averaged in probability as the
    /// Python cascade averages them (the bigger models first, then the small one).
    fn window_rows(&self, ids: &[i32]) -> Result<Vec<Row>, TaggerError> {
        let mut members = self
            .models
            .iter()
            .map(|m| self.log_probs(m, ids))
            .collect::<Result<Vec<_>, _>>()?;
        if members.len() == 1 {
            return Ok(members.remove(0));
        }
        members.rotate_left(1);
        let n = members.len();
        let mut rows = members.remove(0);
        for other in &members {
            for (row, o) in rows.iter_mut().zip(other) {
                for (a, &b) in row.iter_mut().zip(o) {
                    *a = log_add_exp(*a, b);
                }
            }
        }
        let ln_n = (n as f64).ln() as f32;
        for row in &mut rows {
            for a in row.iter_mut() {
                *a -= ln_n;
            }
        }
        Ok(rows)
    }

    /// One model's log-probabilities for a window's content tokens (its special tokens dropped).
    fn log_probs(&self, model: &Model, ids: &[i32]) -> Result<Vec<Row>, TaggerError> {
        let fail = |e: &dyn std::fmt::Display| TaggerError::Inference(e.to_string());
        let n = ids.len();
        let mask = vec![1i32; n];
        let inputs = vec![
            (
                model.node_id("input_ids").map_err(|e| fail(&e))?,
                ValueView::from_shape([1, n], ids)
                    .map_err(|e| fail(&e))?
                    .into(),
            ),
            (
                model.node_id("attention_mask").map_err(|e| fail(&e))?,
                ValueView::from_shape([1, n], &mask[..])
                    .map_err(|e| fail(&e))?
                    .into(),
            ),
        ];
        let logits = model.node_id("logits").map_err(|e| fail(&e))?;
        let [out] = model
            .run_n(inputs, [logits], self.run_options.clone())
            .map_err(|e| fail(&e))?;
        let (shape, data) = out.into_shape_vec::<f32, 3>().map_err(|e| fail(&e))?;
        if shape != [1, n, N_TAGS] {
            return Err(TaggerError::Inference(format!(
                "logits of shape {shape:?} for {n} tokens"
            )));
        }
        Ok(data
            .as_chunks::<N_TAGS>()
            .0
            .iter()
            .skip(1)
            .take(n.saturating_sub(2))
            .map(log_softmax)
            .collect())
    }
}

fn token_id(id: u32) -> Result<i32, TaggerError> {
    i32::try_from(id)
        .map_err(|_| TaggerError::Inference(format!("token id {id} does not fit an i32")))
}

/// PyTorch's log-softmax of one token's logits.
fn log_softmax(logits: &[f32; N_TAGS]) -> Row {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let log_sum = logits.iter().map(|&x| (x - max).exp()).sum::<f32>().ln();
    let mut row = [0f32; N_TAGS];
    for (r, &x) in row.iter_mut().zip(logits) {
        *r = x - max - log_sum;
    }
    row
}

/// numpy's `logaddexp` in f32.
fn log_add_exp(x: f32, y: f32) -> f32 {
    if x == y {
        return x + std::f32::consts::LN_2;
    }
    let d = x - y;
    if d > 0.0 {
        x + (-d).exp().ln_1p()
    } else {
        y + d.exp().ln_1p()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_softmax_sums_to_one() {
        let row = log_softmax(&[1.0; N_TAGS]);
        let total: f32 = row.iter().map(|x| x.exp()).sum();
        assert!((total - 1.0).abs() < 1e-5);
    }

    #[test]
    fn averaging_two_equal_distributions_keeps_them() {
        let x = -1.5f32;
        assert!((log_add_exp(x, x) - (2f64.ln() as f32) - x).abs() < 1e-6);
        assert!((log_add_exp(0.0, f32::NEG_INFINITY) - 0.0).abs() < 1e-6);
    }

    /// The models are not in the repository: point AKANA_PII_MODEL (and optionally AKANA_PII_BIG)
    /// at exported model directories to run this.
    #[test]
    #[ignore]
    fn tags_a_name_and_a_phone_number() {
        let small = std::env::var("AKANA_PII_MODEL").expect("AKANA_PII_MODEL");
        let mut dirs = vec![PathBuf::from(small)];
        if let Ok(big) = std::env::var("AKANA_PII_BIG") {
            dirs.push(PathBuf::from(big));
        }
        let tagger = PiiTagger::load(&dirs, TaggerOptions::default()).unwrap();
        let text = "Ahmet Yılmaz'ı 0532 111 22 33 numarasından arayın.";
        let spans = tagger.tag(text).unwrap();
        let found: Vec<(&str, &str)> = spans
            .iter()
            .map(|s| (&text[s.start..s.end], s.label))
            .collect();
        assert!(found.contains(&("Ahmet Yılmaz", "PERSON")), "{found:?}");
        assert!(found.contains(&("0532 111 22 33", "PHONE")), "{found:?}");
    }
}
