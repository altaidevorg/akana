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

pub mod hybrid;
pub mod rrf;
pub mod sparse;
pub mod tokenizer;
pub mod usearch_metrics;
pub mod weights;

pub use hybrid::{
    get_usearch_metric_pointer_hybrid, hybrid_distance_128, pack_hybrid_vector,
    set_hybrid_metric_alpha, TurkishHybridEmbeddings, HYBRID_DENSE_BYTES, HYBRID_MAX_SPARSE_SLOTS,
    HYBRID_SPARSE_BYTES, HYBRID_VECTOR_BYTES,
};
pub use rrf::{reciprocal_rank_fusion, FusedResult};
pub use sparse::{hash_term_24bit, MorphologicalSparseEncoder, MorphologicalSparseVector};
use tokenizer::UnigramTokenizer;
pub use usearch_metrics::{
    akana_usearch_metric_1bit, akana_usearch_metric_2bit, cosine_distance_2bit_64,
    dot_product_2bit_64, get_usearch_metric_pointer_1bit, get_usearch_metric_pointer_2bit,
    hamming_distance_1bit_32,
};
pub use weights::{EmbeddingWeights, EMBEDDING_DIM};

/// Embedded model data (compiled into the binary).
static WEIGHTS_NPZ: &[u8] = include_bytes!("../../data/embeddings/turboquant_weights.npz");
static TOKENIZER_JSON: &[u8] = include_bytes!("../../data/embeddings/tokenizer.json");

/// Global coordinate standard deviation for 256-dim unit-normalized Haar-rotated vectors.
/// sigma = 1 / sqrt(256) = 0.0625.
pub const EMBEDDING_SCALE_2BIT: f32 = 0.0625;

/// Number of bytes for 256-dimensional 2-bit packed embeddings (64 bytes = 1 CPU cache line).
pub const PACKED_2BIT_BYTES: usize = 64;

/// Number of bytes for 256-dimensional 1-bit binary embeddings (32 bytes = 256 bits).
pub const PACKED_1BIT_BYTES: usize = 32;

/// Quantize a single float coordinate into 2-bit index (0, 1, 2, 3) using Lloyd-Max boundaries.
#[inline]
pub fn quantize_f32_to_2bit(val: f32) -> u8 {
    if val < -EMBEDDING_SCALE_2BIT {
        0
    } else if val < 0.0 {
        1
    } else if val < EMBEDDING_SCALE_2BIT {
        2
    } else {
        3
    }
}

/// Pack a 256-dimensional float vector into 64 bytes of 2-bit values (4 coords/byte).
pub fn pack_2bit_vector(vec: &[f32]) -> [u8; PACKED_2BIT_BYTES] {
    assert_eq!(vec.len(), EMBEDDING_DIM, "vector must have 256 dimensions");
    let mut packed = [0u8; PACKED_2BIT_BYTES];
    for (byte_idx, chunk) in vec.as_chunks::<4>().0.iter().enumerate() {
        let q0 = quantize_f32_to_2bit(chunk[0]);
        let q1 = quantize_f32_to_2bit(chunk[1]);
        let q2 = quantize_f32_to_2bit(chunk[2]);
        let q3 = quantize_f32_to_2bit(chunk[3]);
        packed[byte_idx] = q0 | (q1 << 2) | (q2 << 4) | (q3 << 6);
    }
    packed
}

/// Dequantize 64 bytes of 2-bit values back to a 256-dimensional float vector.
pub fn dequantize_2bit_vector(packed: &[u8; PACKED_2BIT_BYTES]) -> [f32; EMBEDDING_DIM] {
    let mut out = [0.0f32; EMBEDDING_DIM];
    for (i, val) in out.iter_mut().enumerate() {
        let byte_idx = i / 4;
        let bit_offset = (i % 4) * 2;
        let q = ((packed[byte_idx] >> bit_offset) & 0b11) as f32;
        *val = (q - 1.5) * EMBEDDING_SCALE_2BIT;
    }
    out
}

/// Pack a 256-dimensional float vector into 32 bytes of 1-bit values (8 coords/byte, sign bit).
pub fn pack_1bit_vector(vec: &[f32]) -> [u8; PACKED_1BIT_BYTES] {
    assert_eq!(vec.len(), EMBEDDING_DIM, "vector must have 256 dimensions");
    let mut packed = [0u8; PACKED_1BIT_BYTES];
    for (byte_idx, chunk) in vec.as_chunks::<8>().0.iter().enumerate() {
        let mut b = 0u8;
        for (bit_idx, &val) in chunk.iter().enumerate() {
            if val > 0.0 {
                b |= 1 << bit_idx;
            }
        }
        packed[byte_idx] = b;
    }
    packed
}

/// Dequantize 32 bytes of 1-bit values back to a 256-dimensional float vector (+/- sigma).
pub fn dequantize_1bit_vector(packed: &[u8; PACKED_1BIT_BYTES]) -> [f32; EMBEDDING_DIM] {
    let mut out = [0.0f32; EMBEDDING_DIM];
    for (i, val) in out.iter_mut().enumerate() {
        let byte_idx = i / 8;
        let bit_offset = i % 8;
        let bit = (packed[byte_idx] >> bit_offset) & 1;
        *val = if bit == 1 {
            EMBEDDING_SCALE_2BIT
        } else {
            -EMBEDDING_SCALE_2BIT
        };
    }
    out
}

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
        let weights =
            EmbeddingWeights::from_npz_bytes(WEIGHTS_NPZ).expect("failed to load embedded weights");

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

    /// Embed a text string directly into 64 bytes of 2-bit packed values.
    ///
    /// Ideal for ultra-low-memory storage (64 MB for 1 million entries)
    /// and high-speed retrieval using USearch's compiled 2-bit metric.
    pub fn embed_packed_2bit(&self, text: &str) -> [u8; PACKED_2BIT_BYTES] {
        let f32_vec = self.embed(text);
        pack_2bit_vector(&f32_vec)
    }

    /// Embed a text string directly into 32 bytes of 1-bit sign binary values.
    ///
    /// Ideal for ultra-compact storage (32 MB for 1 million entries)
    /// and hardware-accelerated Hamming distance search.
    pub fn embed_packed_1bit(&self, text: &str) -> [u8; PACKED_1BIT_BYTES] {
        let f32_vec = self.embed(text);
        pack_1bit_vector(&f32_vec)
    }

    /// Embed multiple texts into 2-bit packed byte arrays (batch processing).
    pub fn embed_batch_packed_2bit(&self, texts: &[&str]) -> Vec<[u8; PACKED_2BIT_BYTES]> {
        texts.iter().map(|t| self.embed_packed_2bit(t)).collect()
    }

    /// Embed multiple texts into 1-bit packed byte arrays (batch processing).
    pub fn embed_batch_packed_1bit(&self, texts: &[&str]) -> Vec<[u8; PACKED_1BIT_BYTES]> {
        texts.iter().map(|t| self.embed_packed_1bit(t)).collect()
    }

    /// Tokenize a text string into model token IDs.
    pub fn tokenize(&self, text: &str) -> Vec<usize> {
        self.tokenizer.encode(text)
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

    /// Compute cosine similarity between two 2-bit packed vectors.
    pub fn similarity_packed_2bit(a: &[u8; PACKED_2BIT_BYTES], b: &[u8; PACKED_2BIT_BYTES]) -> f32 {
        dot_product_2bit_64(a, b)
    }

    /// Compute cosine distance between two 2-bit packed vectors.
    pub fn distance_packed_2bit(a: &[u8; PACKED_2BIT_BYTES], b: &[u8; PACKED_2BIT_BYTES]) -> f32 {
        cosine_distance_2bit_64(a, b)
    }

    /// Compute Hamming distance between two 1-bit packed vectors.
    pub fn distance_packed_1bit(a: &[u8; PACKED_1BIT_BYTES], b: &[u8; PACKED_1BIT_BYTES]) -> f32 {
        hamming_distance_1bit_32(a, b)
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
        assert!(
            (norm - 1.0).abs() < 0.01,
            "expected L2 norm ~1.0, got {norm}"
        );
    }

    #[test]
    fn test_similarity_identical() {
        let emb = TurkishEmbeddings::new();
        let score = emb.similarity("kedi", "kedi");
        assert!(
            (score - 1.0).abs() < 0.01,
            "identical texts should have similarity ~1.0, got {score}"
        );
    }

    #[test]
    fn test_similarity_related() {
        let emb = TurkishEmbeddings::new();
        let score_related = emb.similarity("ev", "evler");
        let score_unrelated = emb.similarity("ev", "araba");

        // Related words should have higher similarity than unrelated
        assert!(
            score_related > score_unrelated,
            "expected 'ev-evler' ({score_related}) > 'ev-araba' ({score_unrelated})"
        );
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

    #[test]
    fn test_packed_2bit_embed_and_similarity() {
        let emb = TurkishEmbeddings::new();
        let packed_a = emb.embed_packed_2bit("ev");
        let packed_b = emb.embed_packed_2bit("evler");
        let packed_c = emb.embed_packed_2bit("araba");

        assert_eq!(packed_a.len(), PACKED_2BIT_BYTES);
        assert_eq!(packed_b.len(), PACKED_2BIT_BYTES);

        let sim_identical = TurkishEmbeddings::similarity_packed_2bit(&packed_a, &packed_a);
        assert!(
            (sim_identical - 1.0).abs() < 1e-4,
            "identical 2-bit vectors must have sim ~1.0, got {sim_identical}"
        );

        let dist_identical = TurkishEmbeddings::distance_packed_2bit(&packed_a, &packed_a);
        assert!(
            dist_identical < 1e-4,
            "identical 2-bit vectors must have dist ~0.0, got {dist_identical}"
        );

        let sim_related = TurkishEmbeddings::similarity_packed_2bit(&packed_a, &packed_b);
        let sim_unrelated = TurkishEmbeddings::similarity_packed_2bit(&packed_a, &packed_c);
        assert!(
            sim_related > sim_unrelated,
            "expected related 'ev-evler' ({sim_related}) > unrelated 'ev-araba' ({sim_unrelated})"
        );
    }

    #[test]
    fn test_packed_1bit_embed_and_distance() {
        let emb = TurkishEmbeddings::new();
        let packed_a = emb.embed_packed_1bit("ev");
        let packed_b = emb.embed_packed_1bit("evler");
        let packed_c = emb.embed_packed_1bit("araba");

        assert_eq!(packed_a.len(), PACKED_1BIT_BYTES);
        assert_eq!(packed_b.len(), PACKED_1BIT_BYTES);

        let dist_identical = TurkishEmbeddings::distance_packed_1bit(&packed_a, &packed_a);
        assert_eq!(dist_identical, 0.0);

        let dist_related = TurkishEmbeddings::distance_packed_1bit(&packed_a, &packed_b);
        let dist_unrelated = TurkishEmbeddings::distance_packed_1bit(&packed_a, &packed_c);
        assert!(
            dist_related < dist_unrelated,
            "expected related 'ev-evler' Hamming distance ({dist_related}) < unrelated 'ev-araba' ({dist_unrelated})"
        );
    }

    #[test]
    fn test_pack_dequantize_roundtrip() {
        let emb = TurkishEmbeddings::new();
        let f32_vec = emb.embed("Türkiye'nin başkenti Ankara'dır.");
        let packed_2bit = pack_2bit_vector(&f32_vec);
        let dequant_2bit = dequantize_2bit_vector(&packed_2bit);

        assert_eq!(dequant_2bit.len(), EMBEDDING_DIM);
        // Cosine similarity between original f32 and dequantized 2-bit should be high (> 0.85)
        let cos_sim = cosine_similarity(&f32_vec, &dequant_2bit);
        assert!(
            cos_sim > 0.85,
            "expected high fidelity between f32 and dequantized 2-bit, got {cos_sim}"
        );
    }
}
