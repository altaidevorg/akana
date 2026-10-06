//! Unified 128-Byte Single-Buffer Hybrid Embedding & Compiled USearch Metric.
//!
//! Packs dense 2-bit semantic embeddings and unhashed top-16 morphological
//! BM25 terms into a single 128-byte buffer (exactly two 64-byte CPU cache lines).
//!
//! ## Memory Layout
//! - Bytes 0..63 (64 bytes): Dense 2-bit TurboQuant embedding (256 dims @ 2 bits)
//! - Bytes 64..127 (64 bytes): Top-16 salient unhashed sparse terms (16 slots x 4 bytes)
//!   - Slot: `[id_hi, id_mid, id_lo, weight_u8]` (24-bit term ID + 8-bit BM25 weight)
//!   - Sorted ascending by `term_id` for O(K) two-pointer merge intersection.

use super::sparse::{MorphologicalSparseEncoder, MorphologicalSparseVector};
use super::usearch_metrics::cosine_distance_2bit_64;
use super::{pack_2bit_vector, TurkishEmbeddings, PACKED_2BIT_BYTES};
use std::sync::atomic::{AtomicU32, Ordering};

/// Total byte size of the unified hybrid vector (128 bytes = 2 CPU cache lines).
pub const HYBRID_VECTOR_BYTES: usize = 128;

/// Byte size of the dense portion (64 bytes).
pub const HYBRID_DENSE_BYTES: usize = PACKED_2BIT_BYTES;

/// Byte size of the sparse portion (64 bytes).
pub const HYBRID_SPARSE_BYTES: usize = 64;

/// Maximum number of unhashed sparse terms packed in the buffer (16 terms).
pub const HYBRID_MAX_SPARSE_SLOTS: usize = 16;

/// Global tunable alpha weight for the C-ABI hybrid metric (default 0.5).
/// Stored as bits of f32 for thread-safe lock-free atomic access.
static HYBRID_ALPHA_BITS: AtomicU32 = AtomicU32::new(0x3f000000); // 0.5f32

/// Set the global alpha parameter for `akana_usearch_metric_hybrid`.
///
/// Alpha controls the balance between dense (semantic) and sparse (lexical) distance:
/// - `alpha = 0.0`: Pure dense 2-bit search
/// - `alpha = 0.5`: Balanced hybrid (default)
/// - `alpha = 1.0`: Pure sparse morphological BM25 search
pub fn set_hybrid_metric_alpha(alpha: f32) {
    let clamped = alpha.clamp(0.0, 1.0);
    HYBRID_ALPHA_BITS.store(clamped.to_bits(), Ordering::Relaxed);
}

/// Get the current global alpha parameter.
pub fn get_hybrid_metric_alpha() -> f32 {
    f32::from_bits(HYBRID_ALPHA_BITS.load(Ordering::Relaxed))
}

/// Unified Hybrid Embedding Engine.
pub struct TurkishHybridEmbeddings {
    dense: TurkishEmbeddings,
    sparse: MorphologicalSparseEncoder,
}

impl Default for TurkishHybridEmbeddings {
    fn default() -> Self {
        Self::new()
    }
}

impl TurkishHybridEmbeddings {
    /// Create a new hybrid embedding engine.
    pub fn new() -> Self {
        Self {
            dense: TurkishEmbeddings::new(),
            sparse: MorphologicalSparseEncoder::new(),
        }
    }

    /// Embed Turkish text into a 128-byte unified hybrid buffer.
    pub fn embed_hybrid(&self, text: &str) -> [u8; HYBRID_VECTOR_BYTES] {
        let dense_vec = self.dense.embed(text);
        let sparse_vec = self.sparse.encode_sparse(text);
        pack_hybrid_vector(&dense_vec, &sparse_vec)
    }

    /// Embed multiple texts into 128-byte hybrid vectors (batch processing).
    pub fn embed_hybrid_batch(&self, texts: &[&str]) -> Vec<[u8; HYBRID_VECTOR_BYTES]> {
        texts.iter().map(|t| self.embed_hybrid(t)).collect()
    }

    /// Compute fused hybrid distance between two 128-byte vectors.
    pub fn distance_hybrid(
        &self,
        a: &[u8; HYBRID_VECTOR_BYTES],
        b: &[u8; HYBRID_VECTOR_BYTES],
        alpha: f32,
    ) -> f32 {
        hybrid_distance_128(a, b, alpha)
    }
}

/// Pack a dense 256-dim f32 vector and a sparse morphological vector into 128 bytes.
pub fn pack_hybrid_vector(
    dense_vec: &[f32],
    sparse_vec: &MorphologicalSparseVector,
) -> [u8; HYBRID_VECTOR_BYTES] {
    let mut out = [0u8; HYBRID_VECTOR_BYTES];

    // 1. Pack dense 2-bit embedding into bytes 0..64
    let packed_dense = pack_2bit_vector(dense_vec);
    out[..64].copy_from_slice(&packed_dense);

    // 2. Select top-16 sparse terms sorted by BM25 weight descending
    let mut term_pairs: Vec<(u32, f32)> = sparse_vec
        .indices
        .iter()
        .copied()
        .zip(sparse_vec.values.iter().copied())
        .collect();

    // Sort by weight descending to take most salient terms
    term_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    term_pairs.truncate(HYBRID_MAX_SPARSE_SLOTS);

    // Sort selected top-16 by term_id ascending for O(K) two-pointer merge intersection
    term_pairs.sort_by_key(|p| p.0);

    // 3. Write into bytes 64..128
    for (slot_idx, (term_id, weight)) in term_pairs.iter().enumerate() {
        let offset = 64 + slot_idx * 4;
        let id24 = term_id & 0x00FF_FFFF;
        // Quantize BM25 weight into u8 (clamped to [1, 255])
        let q_weight = ((weight / 4.0) * 255.0).clamp(1.0, 255.0) as u8;

        out[offset] = ((id24 >> 16) & 0xFF) as u8;
        out[offset + 1] = ((id24 >> 8) & 0xFF) as u8;
        out[offset + 2] = (id24 & 0xFF) as u8;
        out[offset + 3] = q_weight;
    }

    out
}

/// Compute distance between two 128-byte hybrid vectors.
///
/// Returns a distance scalar in `[0.0, 2.0]`. Identical vectors return `0.0`.
#[inline]
pub fn hybrid_distance_128(
    a: &[u8; HYBRID_VECTOR_BYTES],
    b: &[u8; HYBRID_VECTOR_BYTES],
    alpha: f32,
) -> f32 {
    let dense_a: &[u8; 64] = a[..64].try_into().unwrap();
    let dense_b: &[u8; 64] = b[..64].try_into().unwrap();

    let dense_dist = cosine_distance_2bit_64(dense_a, dense_b);

    // If alpha is 0, skip sparse computation
    if alpha <= 0.0 {
        return dense_dist;
    }

    // Sparse two-pointer intersection over sorted 16-slot arrays
    let mut i = 0usize;
    let mut j = 0usize;
    let mut sparse_dot: u32 = 0;
    let mut norm_a: u32 = 0;
    let mut norm_b: u32 = 0;

    while i < HYBRID_MAX_SPARSE_SLOTS {
        let off = 64 + i * 4;
        let id_a = ((a[off] as u32) << 16) | ((a[off + 1] as u32) << 8) | (a[off + 2] as u32);
        if id_a == 0 {
            break;
        }
        let w_a = a[off + 3] as u32;
        norm_a += w_a * w_a;
        i += 1;
    }

    while j < HYBRID_MAX_SPARSE_SLOTS {
        let off = 64 + j * 4;
        let id_b = ((b[off] as u32) << 16) | ((b[off + 1] as u32) << 8) | (b[off + 2] as u32);
        if id_b == 0 {
            break;
        }
        let w_b = b[off + 3] as u32;
        norm_b += w_b * w_b;
        j += 1;
    }

    let total_slots_a = i;
    let total_slots_b = j;
    i = 0;
    j = 0;

    while i < total_slots_a && j < total_slots_b {
        let off_a = 64 + i * 4;
        let off_b = 64 + j * 4;

        let id_a =
            ((a[off_a] as u32) << 16) | ((a[off_a + 1] as u32) << 8) | (a[off_a + 2] as u32);
        let id_b =
            ((b[off_b] as u32) << 16) | ((b[off_b + 1] as u32) << 8) | (b[off_b + 2] as u32);

        if id_a == id_b {
            let w_a = a[off_a + 3] as u32;
            let w_b = b[off_b + 3] as u32;
            sparse_dot += w_a * w_b;
            i += 1;
            j += 1;
        } else if id_a < id_b {
            i += 1;
        } else {
            j += 1;
        }
    }

    let sparse_dist = if norm_a == 0 || norm_b == 0 {
        1.0
    } else {
        let denom = ((norm_a as f32) * (norm_b as f32)).sqrt();
        if denom < 1e-12 {
            1.0
        } else {
            let cos_sim = (sparse_dot as f32 / denom).clamp(0.0, 1.0);
            1.0 - cos_sim
        }
    };

    let a_clamped = alpha.clamp(0.0, 1.0);
    (1.0 - a_clamped) * dense_dist + a_clamped * sparse_dist
}

// ── C-ABI Exported Function for USearch CompiledMetric ───────────────────────

/// C-ABI compiled metric for 128-byte unified hybrid embeddings in USearch.
///
/// Signature matches USearch `MetricSignature.ArrayArraySize`:
/// `float metric(const void* a, const void* b, size_t dim)`
///
/// # Safety
/// Pointers `a` and `b` must point to at least 128 valid bytes.
#[no_mangle]
pub unsafe extern "C" fn akana_usearch_metric_hybrid(
    a: *const u8,
    b: *const u8,
    _dim: usize,
) -> f32 {
    if a.is_null() || b.is_null() {
        return 1.0;
    }
    let slice_a = &*(a as *const [u8; HYBRID_VECTOR_BYTES]);
    let slice_b = &*(b as *const [u8; HYBRID_VECTOR_BYTES]);
    let alpha = get_hybrid_metric_alpha();
    hybrid_distance_128(slice_a, slice_b, alpha)
}

/// Retrieve the raw function pointer for `akana_usearch_metric_hybrid` as an integer address.
pub fn get_usearch_metric_pointer_hybrid(alpha: Option<f32>) -> usize {
    if let Some(a) = alpha {
        set_hybrid_metric_alpha(a);
    }
    akana_usearch_metric_hybrid as *const () as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hybrid_embed_and_distance_identical() {
        let engine = TurkishHybridEmbeddings::new();
        let vec_a = engine.embed_hybrid("Ankara Türkiye'nin başkentidir.");
        assert_eq!(vec_a.len(), HYBRID_VECTOR_BYTES);

        let dist_self = engine.distance_hybrid(&vec_a, &vec_a, 0.5);
        assert!(
            dist_self < 1e-4,
            "identical hybrid vectors must have distance ~0.0, got {dist_self}"
        );
    }

    #[test]
    fn test_hybrid_distance_related_vs_unrelated() {
        let engine = TurkishHybridEmbeddings::new();
        let vec_a = engine.embed_hybrid("ev ve konut kredisi");
        let vec_b = engine.embed_hybrid("evler için kredi faizleri");
        let vec_c = engine.embed_hybrid("denizaltı gemisi ve torpido");

        let dist_related = engine.distance_hybrid(&vec_a, &vec_b, 0.5);
        let dist_unrelated = engine.distance_hybrid(&vec_a, &vec_c, 0.5);

        assert!(
            dist_related < dist_unrelated,
            "expected related 'ev/kredi' distance ({dist_related}) < unrelated ({dist_unrelated})"
        );
    }

    #[test]
    fn test_c_abi_pointer_hybrid() {
        let ptr = get_usearch_metric_pointer_hybrid(Some(0.6));
        assert_ne!(ptr, 0);
        assert!((get_hybrid_metric_alpha() - 0.6).abs() < 1e-5);

        unsafe {
            let engine = TurkishHybridEmbeddings::new();
            let vec_a = engine.embed_hybrid("deneme metni");
            let dist = akana_usearch_metric_hybrid(vec_a.as_ptr(), vec_a.as_ptr(), 128);
            assert!(dist < 1e-4);
        }
    }
}
