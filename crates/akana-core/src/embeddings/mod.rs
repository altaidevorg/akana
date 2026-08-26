//! Turkish sentence embedding engine using the bundled Model2Vec TurboQuant 2-bit model.
//!
//! Provides fast, zero-config Turkish text embeddings:
//! - **2.5 MB** TurboQuant 2-bit quantized weights (embedded in binary)
//! - **256-dimensional** vectors
//! - **~20K sentences/sec** on CPU
//! - **92.19%** STSb-TR accuracy
//!
//! # Usage
//!
//! ```rust
//! use akana_core::embeddings::TurkishEmbeddings;
//!
//! let emb = TurkishEmbeddings::new();
//! let vec = emb.embed("Ankara Türkiye'nin başkentidir.");
//! let score = emb.similarity("kedi", "köpek");
//! ```

pub mod tokenizer;
pub mod weights;

use tokenizer::UnigramTokenizer;
use weights::{EmbeddingWeights, EMBEDDING_DIM};

/// Embedded model data (compiled into the binary).
static WEIGHTS_NPZ: &[u8] = include_bytes!("../../../../data/embeddings/turboquant_weights.npz");
static TOKENIZER_JSON: &[u8] = include_bytes!("../../../../data/embeddings/tokenizer.json");

/// Turkish sentence embedding engine.
///
/// Wraps the bundled Model2Vec TurboQuant 2-bit model for zero-config
/// Turkish text embeddings. Thread-safe and reusable.
pub struct TurkishEmbeddings {
    tokenizer: UnigramTokenizer,
    weights: EmbeddingWeights,
}

impl Default for TurkishEmbeddings {
    fn default() -> Self {
        Self::new()
    }
}

impl TurkishEmbeddings {
    /// Create a new embedding engine, loading the bundled model.
    ///
    /// This dequantizes all 39,655 token embeddings from 2-bit to f32 at init,
    /// using ~38 MB of RAM. The init cost is a one-time ~50-100ms operation.
    pub fn new() -> Self {
        let tokenizer = UnigramTokenizer::from_json_bytes(TOKENIZER_JSON)
            .expect("failed to load embedded tokenizer");
        let weights = EmbeddingWeights::from_npz_bytes(WEIGHTS_NPZ)
            .expect("failed to load embedded weights");

        Self { tokenizer, weights }
    }

    /// Embed a single text string into a 256-dimensional f32 vector.
    ///
    /// The text is tokenized, each token's embedding is looked up, and the
    /// results are mean-pooled and L2-normalized.
    pub fn embed(&self, text: &str) -> Vec<f32> {
        let token_ids = self.tokenizer.encode(text);
        self.embed_token_ids(&token_ids)
    }

    /// Embed multiple texts into vectors (batch processing).
    pub fn embed_batch(&self, texts: &[&str]) -> Vec<Vec<f32>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    /// Compute cosine similarity between two texts.
    pub fn similarity(&self, text_a: &str, text_b: &str) -> f32 {
        let vec_a = self.embed(text_a);
        let vec_b = self.embed(text_b);
        cosine_similarity(&vec_a, &vec_b)
    }

    /// Compute cosine similarity between two pre-computed embedding vectors.
    pub fn similarity_vectors(vec_a: &[f32], vec_b: &[f32]) -> f32 {
        cosine_similarity(vec_a, vec_b)
    }

    /// Get the embedding dimension (256).
    pub const fn embedding_dim(&self) -> usize {
        EMBEDDING_DIM
    }

    /// Internal: compute the embedding for a sequence of token IDs.
    fn embed_token_ids(&self, token_ids: &[usize]) -> Vec<f32> {
        if token_ids.is_empty() {
            return vec![0.0; EMBEDDING_DIM];
        }

        // Mean pooling: average all token embeddings (including BOS/EOS)
        let mut sum = vec![0.0f32; EMBEDDING_DIM];
        let mut count = 0usize;

        for &id in token_ids {
            if id < self.weights.vocab_size {
                let emb = self.weights.get_embedding(id);
                for (s, &e) in sum.iter_mut().zip(emb.iter()) {
                    *s += e;
                }
                count += 1;
            }
        }

        if count == 0 {
            return vec![0.0; EMBEDDING_DIM];
        }

        let count_f = count as f32;
        for s in sum.iter_mut() {
            *s /= count_f;
        }

        // L2 normalize
        l2_normalize(&mut sum);

        sum
    }
}

/// L2-normalize a vector in place.
#[inline]
fn l2_normalize(vec: &mut [f32]) {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        let inv = 1.0 / norm;
        for v in vec.iter_mut() {
            *v *= inv;
        }
    }
}

/// Compute cosine similarity between two vectors.
///
/// Assumes vectors are L2-normalized (as produced by `embed`), so this
/// reduces to a dot product.
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vectors must have same length");

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a < 1e-12 || norm_b < 1e-12 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embed_produces_correct_dim() {
        let emb = TurkishEmbeddings::new();
        let vec = emb.embed("merhaba dünya");
        assert_eq!(vec.len(), EMBEDDING_DIM);
    }

    #[test]
    fn test_embed_is_normalized() {
        let emb = TurkishEmbeddings::new();
        let vec = emb.embed("Türkçe doğal dil işleme");

        // L2 norm should be ~1.0
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01,
            "expected L2 norm ~1.0, got {}", norm);
    }

    #[test]
    fn test_similarity_identical() {
        let emb = TurkishEmbeddings::new();
        let score = emb.similarity("kedi", "kedi");
        assert!((score - 1.0).abs() < 0.01,
            "identical texts should have similarity ~1.0, got {}", score);
    }

    #[test]
    fn test_similarity_related() {
        let emb = TurkishEmbeddings::new();
        let score_related = emb.similarity("ev", "evler");
        let score_unrelated = emb.similarity("ev", "araba");

        // Related words should have higher similarity than unrelated
        assert!(score_related > score_unrelated,
            "expected 'ev-evler' ({}) > 'ev-araba' ({})",
            score_related, score_unrelated);
    }

    #[test]
    fn test_embed_batch() {
        let emb = TurkishEmbeddings::new();
        let vecs = emb.embed_batch(&["merhaba", "dünya", "nasılsınız"]);
        assert_eq!(vecs.len(), 3);
        for v in &vecs {
            assert_eq!(v.len(), EMBEDDING_DIM);
        }
    }

    #[test]
    fn test_empty_text() {
        let emb = TurkishEmbeddings::new();
        let vec = emb.embed("");
        assert_eq!(vec.len(), EMBEDDING_DIM);
        // Even empty text gets BOS + EOS, so it won't be all zeros
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 0.0).abs() < 1e-6);

        let c = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &c) - 1.0).abs() < 1e-6);
    }
}
