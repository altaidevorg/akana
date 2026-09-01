//! Similarity calculations, windowing, smoothing, and adaptive threshold estimation.

use super::types::ThresholdMode;
use crate::embeddings::cosine_similarity;

/// Computes cosine similarities between consecutive sentence embeddings with optional windowing.
///
/// When `window_size` is 1, it computes pairwise similarity between sentence `i` and sentence `i+1`.
/// When `window_size` > 1, it pools embeddings within the left window `[i - window_size + 1 ..= i]`
/// and right window `[i + 1 ..= i + window_size]` before calculating cosine similarity.
pub fn compute_windowed_similarities(
    embeddings: &[Vec<f32>],
    window_size: usize,
) -> Vec<f32> {
    let n = embeddings.len();
    if n < 2 {
        return Vec::new();
    }

    let w = window_size.max(1);
    let mut similarities = Vec::with_capacity(n - 1);

    for i in 0..(n - 1) {
        if w == 1 {
            let sim = cosine_similarity(&embeddings[i], &embeddings[i + 1]);
            similarities.push(sim);
        } else {
            // Left window: max(0, i + 1 - w) ..= i
            let left_start = if i + 1 >= w { i + 1 - w } else { 0 };
            let left_slice = &embeddings[left_start..=i];
            let left_vec = mean_pool_vectors(left_slice);

            // Right window: i + 1 ..= min(n - 1, i + w)
            let right_end = (i + w).min(n - 1);
            let right_slice = &embeddings[(i + 1)..=right_end];
            let right_vec = mean_pool_vectors(right_slice);

            let sim = cosine_similarity(&left_vec, &right_vec);
            similarities.push(sim);
        }
    }

    similarities
}

/// Averages multiple vectors and L2-normalizes the result.
fn mean_pool_vectors(vectors: &[Vec<f32>]) -> Vec<f32> {
    if vectors.is_empty() {
        return Vec::new();
    }
    let dim = vectors[0].len();
    let mut pooled = vec![0.0f32; dim];

    for v in vectors {
        for (i, val) in v.iter().enumerate() {
            pooled[i] += val;
        }
    }

    let n = vectors.len() as f32;
    for val in &mut pooled {
        *val /= n;
    }

    // L2 normalize
    let norm = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        for val in &mut pooled {
            *val /= norm;
        }
    }

    pooled
}

/// Applies a 1D moving average filter to smooth similarity scores.
pub fn moving_average_filter(scores: &[f32], window_size: usize) -> Vec<f32> {
    let n = scores.len();
    if n == 0 || window_size <= 1 {
        return scores.to_vec();
    }

    let half = window_size / 2;
    let mut smoothed = Vec::with_capacity(n);

    for i in 0..n {
        let start = if i >= half { i - half } else { 0 };
        let end = (i + half + 1).min(n);
        let slice = &scores[start..end];
        let sum: f32 = slice.iter().sum();
        smoothed.push(sum / slice.len() as f32);
    }

    smoothed
}

/// Applies a Savitzky-Golay smoothing filter (5-point quadratic/cubic filter) to similarity scores.
///
/// Convolution weights for standard 5-point Savitzky-Golay: `[-3, 12, 17, 12, -3] / 35`.
pub fn savitzky_golay_filter(scores: &[f32]) -> Vec<f32> {
    let n = scores.len();
    if n < 5 {
        return moving_average_filter(scores, 3);
    }

    let weights = [-3.0f32, 12.0, 17.0, 12.0, -3.0];
    let norm = 35.0f32;
    let mut smoothed = Vec::with_capacity(n);

    // Boundary elements (first 2 and last 2) use moving average
    smoothed.push((scores[0] * 2.0 + scores[1]) / 3.0);
    smoothed.push((scores[0] + scores[1] + scores[2]) / 3.0);

    for i in 2..(n - 2) {
        let val = (scores[i - 2] * weights[0]
            + scores[i - 1] * weights[1]
            + scores[i] * weights[2]
            + scores[i + 1] * weights[3]
            + scores[i + 2] * weights[4])
            / norm;
        smoothed.push(val.clamp(0.0, 1.0));
    }

    smoothed.push((scores[n - 3] + scores[n - 2] + scores[n - 1]) / 3.0);
    smoothed.push((scores[n - 2] + scores[n - 1] * 2.0) / 3.0);

    smoothed
}

/// Calculates the similarity cutoff threshold based on the configured mode and score distribution.
///
/// Scores below this threshold represent topic transitions (valleys/breakpoints).
pub fn calculate_threshold(scores: &[f32], mode: &ThresholdMode) -> f32 {
    if scores.is_empty() {
        return 0.70;
    }

    match mode {
        ThresholdMode::Similarity(val) => *val,

        ThresholdMode::Percentile(p) => {
            // If p = 0.75 (75th percentile of drops), we split at the bottom 25% similarity scores.
            let percentile = (1.0 - p.clamp(0.0, 1.0)).clamp(0.0, 1.0);
            calculate_quantile(scores, percentile)
        }

        ThresholdMode::StandardDeviation(k) => {
            let mean = scores.iter().sum::<f32>() / scores.len() as f32;
            let variance = scores
                .iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f32>()
                / scores.len() as f32;
            let std_dev = variance.sqrt();
            (mean - k * std_dev).clamp(0.0, 1.0)
        }

        ThresholdMode::Interquartile(k) => {
            let q1 = calculate_quantile(scores, 0.25);
            let q3 = calculate_quantile(scores, 0.75);
            let iqr = (q3 - q1).max(0.0);
            (q1 - k * iqr).clamp(0.0, 1.0)
        }

        ThresholdMode::Auto => {
            if scores.len() >= 6 {
                // If enough samples, standard deviation with k=0.8 works remarkably well
                calculate_threshold(scores, &ThresholdMode::StandardDeviation(0.8))
            } else {
                calculate_threshold(scores, &ThresholdMode::Percentile(0.75))
            }
        }
    }
}

/// Calculates the quantile value (0.0 <= q <= 1.0) using linear interpolation.
pub fn calculate_quantile(scores: &[f32], q: f32) -> f32 {
    if scores.is_empty() {
        return 0.0;
    }
    if scores.len() == 1 {
        return scores[0];
    }

    let mut sorted = scores.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let clamped_q = q.clamp(0.0, 1.0);
    let index = clamped_q * (sorted.len() - 1) as f32;
    let lower = index.floor() as usize;
    let upper = index.ceil() as usize;

    if lower == upper {
        sorted[lower]
    } else {
        let weight = index - lower as f32;
        sorted[lower] * (1.0 - weight) + sorted[upper] * weight
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_moving_average_filter() {
        let scores = vec![0.9, 0.1, 0.9];
        let smoothed = moving_average_filter(&scores, 3);
        assert_eq!(smoothed.len(), 3);
        assert!((smoothed[1] - 0.6333).abs() < 0.05);
    }

    #[test]
    fn test_calculate_quantile() {
        let scores = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        let median = calculate_quantile(&scores, 0.5);
        assert!((median - 0.5).abs() < 1e-4);

        let q25 = calculate_quantile(&scores, 0.25);
        assert!((q25 - 0.3).abs() < 1e-4);
    }

    #[test]
    fn test_calculate_threshold_modes() {
        let scores = vec![0.85, 0.82, 0.40, 0.88, 0.35, 0.90];
        let sim_thresh = calculate_threshold(&scores, &ThresholdMode::Similarity(0.70));
        assert_eq!(sim_thresh, 0.70);

        let std_thresh = calculate_threshold(&scores, &ThresholdMode::StandardDeviation(1.0));
        assert!(std_thresh > 0.3 && std_thresh < 0.8);
    }
}
