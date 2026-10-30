//! Reciprocal Rank Fusion (RRF) for Hybrid Retrieval.
//!
//! Blends ranked result lists from dense and sparse retrieval systems
//! without requiring scale normalization:
//!
//! `RRF(d) = 1.0 / (k + rank_dense) + alpha / (k + rank_sparse)`

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Fused ranking result containing document ID and combined RRF score.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FusedResult {
    pub id: u64,
    pub score: f32,
}

/// Compute Reciprocal Rank Fusion between dense and sparse ranked result lists.
///
/// Both input lists are expected to be ordered best-to-worst (rank 1 is index 0).
///
/// - `dense_results`: `[(doc_id, score), ...]`
/// - `sparse_results`: `[(doc_id, score), ...]`
/// - `k`: smoothing constant (default: 60.0), positive and finite
/// - `alpha`: relative weight for the sparse ranking (default: 1.0), finite
///
/// The result is deterministic (review 2 EMB-12): an id listed twice in one list counts once,
/// at its best rank, and equal scores are ordered by the id's best rank in either list, then
/// by id.
///
/// # Panics
///
/// If `k` is not positive and finite, or `alpha` is not finite.
pub fn reciprocal_rank_fusion(
    dense_results: &[(u64, f32)],
    sparse_results: &[(u64, f32)],
    k: f32,
    alpha: f32,
) -> Vec<FusedResult> {
    assert!(
        k > 0.0 && k.is_finite(),
        "RRF k must be positive and finite, got {k}"
    );
    assert!(alpha.is_finite(), "RRF alpha must be finite, got {alpha}");
    // (id, score, best rank), in order of first appearance.
    let mut entries: Vec<(u64, f32, usize)> = Vec::new();
    let mut position: HashMap<u64, usize> = HashMap::new();
    for (weight, results) in [(1.0, dense_results), (alpha, sparse_results)] {
        let mut seen = HashSet::new();
        for (rank, &(doc_id, _)) in results.iter().enumerate() {
            if !seen.insert(doc_id) {
                continue;
            }
            let rrf = weight / (k + (rank as f32) + 1.0);
            match position.get(&doc_id) {
                Some(&i) => {
                    entries[i].1 += rrf;
                    entries[i].2 = entries[i].2.min(rank);
                }
                None => {
                    position.insert(doc_id, entries.len());
                    entries.push((doc_id, rrf, rank));
                }
            }
        }
    }
    // Highest score first; ties by best rank, then id.
    entries.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.2.cmp(&b.2)).then(a.0.cmp(&b.0)));
    entries
        .into_iter()
        .map(|(id, score, _)| FusedResult { id, score })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rrf_both_top_rank() {
        let dense = vec![(10, 0.95), (20, 0.80), (30, 0.70)];
        let sparse = vec![(10, 4.5), (40, 3.2), (20, 2.1)];

        let fused = reciprocal_rank_fusion(&dense, &sparse, 60.0, 1.0);

        // Doc 10 is rank 1 in both lists, must be top ranked
        assert_eq!(fused[0].id, 10);
        // Doc 20 is rank 2 in dense and rank 3 in sparse, must beat doc 40 (only in sparse)
        assert_eq!(fused[1].id, 20);
    }

    #[test]
    fn test_rrf_alpha_tuning() {
        let dense = vec![(1, 0.9), (2, 0.8)];
        let sparse = vec![(2, 5.0), (1, 1.0)];

        // With alpha = 5.0, sparse rank heavily dominates
        let fused = reciprocal_rank_fusion(&dense, &sparse, 60.0, 5.0);
        assert_eq!(fused[0].id, 2);

        // With alpha = 0.1, dense rank heavily dominates
        let fused_dense = reciprocal_rank_fusion(&dense, &sparse, 60.0, 0.1);
        assert_eq!(fused_dense[0].id, 1);
    }

    /// Review 2 EMB-12: one order for equal scores, duplicates counted once, k validated.
    #[test]
    fn rrf_is_deterministic() {
        let dense = vec![(1, 0.9), (2, 0.8), (3, 0.7), (4, 0.6)];
        let sparse = vec![(2, 4.0), (1, 3.0), (4, 2.0), (3, 1.0)];
        let ids = |f: Vec<FusedResult>| f.into_iter().map(|r| r.id).collect::<Vec<_>>();
        let first = ids(reciprocal_rank_fusion(&dense, &sparse, 60.0, 1.0));
        assert_eq!(first, [1, 2, 3, 4]);
        for _ in 0..50 {
            assert_eq!(
                ids(reciprocal_rank_fusion(&dense, &sparse, 60.0, 1.0)),
                first
            );
        }
        // An id listed twice counts once, at its best rank.
        let fused = reciprocal_rank_fusion(&[(7, 0.9), (7, 0.9), (8, 0.5)], &[(8, 1.0)], 60.0, 1.0);
        assert_eq!(ids(fused), [8, 7]);
    }

    #[test]
    #[should_panic(expected = "k must be positive")]
    fn rrf_rejects_a_k_that_divides_by_zero() {
        reciprocal_rank_fusion(&[(1, 1.0)], &[], -1.0, 1.0);
    }
}
