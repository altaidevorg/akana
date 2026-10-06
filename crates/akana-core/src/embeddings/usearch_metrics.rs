//! Compiled distance metrics for USearch integration.
//!
//! Provides ultra-fast C-ABI metric function pointers compatible with
//! `usearch.index.CompiledMetric(signature=MetricSignature.ArrayArraySize)`.
//!
//! Includes:
//! - 2-Bit TurboQuant cosine distance with 64 KB L2-cache Lookup Table (LUT)
//! - 1-Bit binary Hamming distance with hardware `POPCNT`

/// Compile-time precomputed 256x256 lookup table for partial dot products.
///
/// For byte `u` and byte `v`, each packs 4 coordinates with centered values
/// in `{-3, -1, 1, 3}`. The entry is the integer dot product:
/// `sum_{k=0..3} (2*q_u[k] - 3) * (2*q_v[k] - 3)`
pub static BYTE_DOT_LUT: [[i8; 256]; 256] = build_byte_dot_lut();

const fn build_byte_dot_lut() -> [[i8; 256]; 256] {
    let mut table = [[0i8; 256]; 256];
    let mut u = 0;
    while u < 256 {
        let mut v = 0;
        while v < 256 {
            let mut dot: i32 = 0;
            let mut k = 0;
            while k < 4 {
                let qu = ((u >> (k * 2)) & 0b11) as i32;
                let qv = ((v >> (k * 2)) & 0b11) as i32;
                let cu = 2 * qu - 3;
                let cv = 2 * qv - 3;
                dot += cu * cv;
                k += 1;
            }
            table[u][v] = dot as i8;
            v += 1;
        }
        u += 1;
    }
    table
}

/// Compute cosine similarity between two 64-byte (256-dim) 2-bit packed vectors.
///
/// Returns cosine similarity in `[-1.0, 1.0]`. Self-similarity is `1.0`.
#[inline]
pub fn dot_product_2bit_64(a: &[u8; 64], b: &[u8; 64]) -> f32 {
    let mut dot_int: i32 = 0;
    let mut norm_a_sq: i32 = 0;
    let mut norm_b_sq: i32 = 0;

    for i in 0..64 {
        let byte_a = a[i] as usize;
        let byte_b = b[i] as usize;
        dot_int += BYTE_DOT_LUT[byte_a][byte_b] as i32;
        norm_a_sq += BYTE_DOT_LUT[byte_a][byte_a] as i32;
        norm_b_sq += BYTE_DOT_LUT[byte_b][byte_b] as i32;
    }

    if norm_a_sq <= 0 || norm_b_sq <= 0 {
        return 0.0;
    }

    let denom = ((norm_a_sq as f32) * (norm_b_sq as f32)).sqrt();
    if denom < 1e-12 {
        0.0
    } else {
        (dot_int as f32 / denom).clamp(-1.0, 1.0)
    }
}

/// Compute cosine distance (1.0 - cosine_similarity) between two 64-byte vectors.
///
/// Returns distance in `[0.0, 2.0]`. Identical vectors return `0.0`.
#[inline]
pub fn cosine_distance_2bit_64(a: &[u8; 64], b: &[u8; 64]) -> f32 {
    let sim = dot_product_2bit_64(a, b);
    (1.0 - sim).max(0.0)
}

/// Compute normalized Hamming distance between two 32-byte (256-dim) 1-bit packed vectors.
///
/// Uses hardware 64-bit popcount (`_mm_popcnt` / `count_ones`).
/// Returns distance in `[0.0, 1.0]`. Identical vectors return `0.0`.
#[inline]
pub fn hamming_distance_1bit_32(a: &[u8; 32], b: &[u8; 32]) -> f32 {
    let mut diff: u32 = 0;
    for i in 0..4 {
        let chunk_a = u64::from_ne_bytes(a[i * 8..(i + 1) * 8].try_into().unwrap());
        let chunk_b = u64::from_ne_bytes(b[i * 8..(i + 1) * 8].try_into().unwrap());
        diff += (chunk_a ^ chunk_b).count_ones();
    }
    diff as f32 / 256.0
}

// ── C-ABI Exported Functions for USearch CompiledMetric ──────────────────────

/// C-ABI metric for 2-bit packed embeddings in USearch.
///
/// Signature matches USearch `MetricSignature.ArrayArraySize`:
/// `float metric(const void* a, const void* b, size_t dim)`
///
/// # Safety
/// Pointers `a` and `b` must point to at least 64 valid bytes.
#[no_mangle]
pub unsafe extern "C" fn akana_usearch_metric_2bit(
    a: *const u8,
    b: *const u8,
    _dim: usize,
) -> f32 {
    if a.is_null() || b.is_null() {
        return 1.0;
    }
    let slice_a = &*(a as *const [u8; 64]);
    let slice_b = &*(b as *const [u8; 64]);
    cosine_distance_2bit_64(slice_a, slice_b)
}

/// C-ABI metric for 1-bit binary embeddings in USearch.
///
/// Signature matches USearch `MetricSignature.ArrayArraySize`:
/// `float metric(const void* a, const void* b, size_t dim)`
///
/// # Safety
/// Pointers `a` and `b` must point to at least 32 valid bytes.
#[no_mangle]
pub unsafe extern "C" fn akana_usearch_metric_1bit(
    a: *const u8,
    b: *const u8,
    _dim: usize,
) -> f32 {
    if a.is_null() || b.is_null() {
        return 1.0;
    }
    let slice_a = &*(a as *const [u8; 32]);
    let slice_b = &*(b as *const [u8; 32]);
    hamming_distance_1bit_32(slice_a, slice_b)
}

/// Retrieve the raw function pointer for `akana_usearch_metric_2bit` as an integer address.
pub fn get_usearch_metric_pointer_2bit() -> usize {
    akana_usearch_metric_2bit as *const () as usize
}

/// Retrieve the raw function pointer for `akana_usearch_metric_1bit` as an integer address.
pub fn get_usearch_metric_pointer_1bit() -> usize {
    akana_usearch_metric_1bit as *const () as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_byte_dot_lut_self_product() {
        // A byte with identical values [0, 0, 0, 0] (-3 in each)
        // dot = 4 * (-3 * -3) = 36
        assert_eq!(BYTE_DOT_LUT[0][0], 36);

        // A byte with [3, 3, 3, 3] (+3 in each, byte = 0xFF = 255)
        // dot = 4 * (3 * 3) = 36
        assert_eq!(BYTE_DOT_LUT[255][255], 36);

        // Orthogonal-ish bytes: [0, 0, 0, 0] vs [3, 3, 3, 3]
        // dot = 4 * (-3 * 3) = -36
        assert_eq!(BYTE_DOT_LUT[0][255], -36);
    }

    #[test]
    fn test_dot_product_identical() {
        let vec_a = [0x55u8; 64]; // Pattern 01010101
        let sim = dot_product_2bit_64(&vec_a, &vec_a);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "identical vectors must have cosine similarity ~1.0, got {sim}"
        );
        let dist = cosine_distance_2bit_64(&vec_a, &vec_a);
        assert!(dist.abs() < 1e-5, "identical vectors must have distance 0.0");
    }

    #[test]
    fn test_dot_product_opposite() {
        // 0 -> bin 0 (-3), 3 -> bin 3 (+3)
        // byte 0x00 has all 0s (-3)
        // byte 0xFF has all 3s (+3)
        let vec_a = [0x00u8; 64];
        let vec_b = [0xFFu8; 64];
        let sim = dot_product_2bit_64(&vec_a, &vec_b);
        assert!(
            (sim - (-1.0)).abs() < 1e-5,
            "opposite vectors must have similarity -1.0, got {sim}"
        );
        let dist = cosine_distance_2bit_64(&vec_a, &vec_b);
        assert!(
            (dist - 2.0).abs() < 1e-5,
            "opposite vectors must have distance 2.0, got {dist}"
        );
    }

    #[test]
    fn test_hamming_distance_1bit() {
        let vec_a = [0xAAu8; 32];
        let dist_same = hamming_distance_1bit_32(&vec_a, &vec_a);
        assert_eq!(dist_same, 0.0);

        let vec_b = [0x55u8; 32]; // exactly inverted bits
        let dist_diff = hamming_distance_1bit_32(&vec_a, &vec_b);
        assert_eq!(dist_diff, 1.0);
    }

    #[test]
    fn test_c_abi_pointers_valid() {
        let ptr2 = get_usearch_metric_pointer_2bit();
        assert_ne!(ptr2, 0);

        let ptr1 = get_usearch_metric_pointer_1bit();
        assert_ne!(ptr1, 0);

        unsafe {
            let vec_a = [0x55u8; 64];
            let dist = akana_usearch_metric_2bit(vec_a.as_ptr(), vec_a.as_ptr(), 64);
            assert!(dist < 1e-5);
        }
    }
}
