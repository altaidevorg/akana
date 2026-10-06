//! Reciprocal Rank Fusion (RRF) for Hybrid Retrieval.
//!
//! Blends ranked result lists from dense and sparse retrieval systems
//! without requiring scale normalization:
//!
//! `RRF(d) = 1.0 / (k + rank_dense) + alpha / (k + rank_sparse)`

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
/// - `k`: smoothing constant (default: 60.0)
/// - `alpha`: relative weight for the sparse ranking (default: 1.0)
pub fn reciprocal_rank_fusion(
    dense_results: &[(u64, f32)],
    sparse_results: &[(u64, f32)],
    k: f32,
    alpha: f32,
) -> Vec<FusedResult> {
    let mut scores: HashMap<u64, f32> = HashMap::new();

    for (rank, &(doc_id, _)) in dense_results.iter().enumerate() {
        let rrf = 1.0 / (k + (rank as f32) + 1.0);
        *scores.entry(doc_id).or_insert(0.0) += rrf;
    }

    for (rank, &(doc_id, _)) in sparse_results.iter().enumerate() {
        let rrf = alpha / (k + (rank as f32) + 1.0);
        *scores.entry(doc_id).or_insert(0.0) += rrf;
    }

    let mut fused: Vec<FusedResult> = scores
        .into_iter()
        .map(|(id, score)| FusedResult { id, score })
        .collect();

    // Sort descending by combined RRF score (highest score first)
    fused.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    fused
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
}
