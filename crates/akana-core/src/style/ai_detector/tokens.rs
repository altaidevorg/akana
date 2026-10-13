//! Token-transition signal: how much more likely the text's token sequence is under a bigram
//! model of AI answers than under one of human text.
//!
//! The text is flattened (markdown symbols and list markers removed, digit runs → `0`,
//! whitespace collapsed, Turkish lowercase) and cut by Akana's embedding tokenizer
//! ([`crate::embeddings`]). Every transition `prev → cur` is scored per class with
//! `P(cur | prev) = λ·big/hist + (1 − λ)·(uni + 1)/(N + V)` (λ = 0 after the start symbol
//! or an unseen history); the score is `d = ln P_AI − ln P_human`. The model inputs are the
//! mean of `d` and the share of transitions with `d > 0`.
//!
//! # `data/ai_detector/tokens.bin.zz`
//!
//! zlib-compressed, all integers little-endian:
//!
//! ```text
//! b"AKLM3"   magic and format version
//! u32 V      model vocabulary size (ids 0 = start, 1 = unknown, 2.. = model.json lm.tokens)
//! then for class 0 (human) and class 1 (AI):
//!   u64      N, total number of counted tokens
//!   V × u32  unigram counts of model ids (as the `cur` side of a transition)
//!   u32 B    number of bigrams
//!   B × u32  bigram keys `prev * V + cur`, strictly increasing
//!   B × u32  bigram counts (pairs seen fewer than 5 times are dropped)
//! (end of data)
//! ```

use std::io::Read;
use std::sync::LazyLock;

use regex::Regex;

use super::model::MODEL;
use crate::embeddings::TOKENIZER;
use crate::phonology::alphabet::to_turkish_lower;

static TOKENS_BIN: &[u8] = include_bytes!("../../../data/ai_detector/tokens.bin.zz");

/// Markdown symbols, removed before tokenising and before matching patterns.
pub(super) static MARKDOWN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[*_`#>]+").expect("valid regex"));
static BULLET: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^\s*(?:[-*•–]|\d+[.)])\s+").expect("valid regex"));
static DIGITS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\d+").expect("valid regex"));
static SPACES: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\s+").expect("valid regex"));

/// Formatting removed before tokenising (frozen: the counts were made on this form).
fn flatten(text: &str) -> String {
    let t = MARKDOWN.replace_all(text, " ");
    let t = BULLET.replace_all(&t, " ");
    let t = DIGITS.replace_all(&t, "0");
    to_turkish_lower(SPACES.replace_all(&t, " ").trim())
}

/// Tokenizer ids of a text, as seen by the token-transition model.
pub(super) fn transition_tokens(text: &str) -> Vec<usize> {
    TOKENIZER.encode(&flatten(text))
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
    v: u32,
    /// Model id of every tokenizer id (1 = unknown to the model).
    ids: Vec<u32>,
    /// 0 = human, 1 = AI.
    classes: [ClassCounts; 2],
}

/// Little-endian reader straight from the decompressor, so the decompressed file is never
/// held in memory as a whole.
struct Reader<R: Read>(R);

impl<R: Read> Reader<R> {
    fn bytes(&mut self, n: usize) -> Result<Vec<u8>, &'static str> {
        let mut buf = vec![0u8; n];
        self.0
            .read_exact(&mut buf)
            .map_err(|_| "tokens.bin.zz is truncated or not valid zlib")?;
        Ok(buf)
    }

    fn u32(&mut self) -> Result<u32, &'static str> {
        let b = self.bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self) -> Result<u64, &'static str> {
        let b = self.bytes(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn u32s(&mut self, n: usize) -> Result<Vec<u32>, &'static str> {
        Ok(self
            .bytes(4 * n)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect())
    }

    fn at_end(&mut self) -> bool {
        matches!(self.0.read(&mut [0u8; 1]), Ok(0))
    }
}

fn class_counts<R: Read>(r: &mut Reader<R>, v: usize) -> Result<ClassCounts, &'static str> {
    let n_tok = r.u64()? as f64;
    let uni = r.u32s(v)?;
    let n_big = r.u32()? as usize;
    let big_keys = r.u32s(n_big)?;
    let big_counts = r.u32s(n_big)?;
    if !big_keys.windows(2).all(|w| w[0] < w[1]) {
        return Err("bigram keys are not strictly increasing");
    }
    Ok(ClassCounts {
        n_tok,
        uni,
        big_keys,
        big_counts,
    })
}

fn parse_lm() -> Result<TokenLm, &'static str> {
    let mut r = Reader(flate2::read::ZlibDecoder::new(TOKENS_BIN));
    if r.bytes(5)? != b"AKLM3" {
        return Err("tokens.bin.zz has an unknown format version");
    }
    let v = r.u32()?;
    let classes = [
        class_counts(&mut r, v as usize)?,
        class_counts(&mut r, v as usize)?,
    ];
    if !r.at_end() {
        return Err("tokens.bin.zz has trailing data");
    }
    let spec = &MODEL.lm;
    if spec.tokenizer != "akana" {
        return Err("model.json was trained on another tokenizer");
    }
    if v as usize != spec.vocab_size || spec.tokens.len() + 2 != spec.vocab_size {
        return Err("tokens.bin.zz does not match model.json");
    }
    // Bigram keys are `prev * V + cur` in u32.
    if u64::from(v) * u64::from(v) > u64::from(u32::MAX) {
        return Err("vocabulary too large for u32 bigram keys");
    }
    let mut ids = vec![1u32; TOKENIZER.vocab_size()];
    for (model_id, &tok) in (2u32..).zip(&spec.tokens) {
        *ids.get_mut(tok as usize)
            .ok_or("model.json token id out of range")? = model_id;
    }
    Ok(TokenLm {
        lambda: spec.lambda,
        v,
        ids,
        classes,
    })
}

static LM: LazyLock<TokenLm> = LazyLock::new(|| match parse_lm() {
    Ok(lm) => lm,
    Err(e) => panic!("embedded data/ai_detector: {e}"),
});

impl TokenLm {
    fn logprob(&self, class: usize, prev: u32, cur: u32) -> f64 {
        let k = &self.classes[class];
        let p_uni = (f64::from(k.uni[cur as usize]) + 1.0) / (k.n_tok + f64::from(self.v));
        let hist = if prev == 0 {
            0.0
        } else {
            f64::from(k.uni[prev as usize])
        };
        if hist <= 0.0 {
            return p_uni.ln();
        }
        let big = k.bigram(prev * self.v + cur);
        (self.lambda * big / hist + (1.0 - self.lambda) * p_uni).ln()
    }
}

/// `[wordlm_llr_mean, wordlm_ai_share]` of a text (zeros for a text without tokens).
pub(super) fn transition_features(text: &str) -> [f64; 2] {
    let lm = &*LM;
    let (mut prev, mut n, mut sum, mut ai) = (0u32, 0usize, 0.0, 0usize);
    for t in transition_tokens(text) {
        let cur = lm.ids.get(t).copied().unwrap_or(1);
        let d = lm.logprob(1, prev, cur) - lm.logprob(0, prev, cur);
        prev = cur;
        n += 1;
        sum += d;
        ai += usize::from(d > 0.0);
    }
    if n == 0 {
        return [0.0, 0.0];
    }
    [sum / n as f64, ai as f64 / n as f64]
}
