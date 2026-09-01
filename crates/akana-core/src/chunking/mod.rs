//! High-performance text chunking strategies for Turkish NLP and RAG pipelines.
//!
//! Inspired by Chonkie's lightweight and efficient design, Akana provides:
//! - [`SemanticChunker`]: Splits text based on semantic similarity drops using Turkish embeddings.
//! - [`SentenceChunker`]: Sentence-boundary-aware chunker with token limits and overlap.
//! - [`SDPMChunker`]: Semantic Double-Pass Merge for dense, cohesive chunks.
//! - [`ThresholdMode`]: Adaptive thresholding (Similarity, Percentile, StandardDeviation, Interquartile, Auto).
//!
//! # Examples
//!
//! ```rust
//! use akana_core::chunking::{SemanticChunker, ThresholdMode};
//!
//! let chunker = SemanticChunker::new(512, ThresholdMode::Similarity(0.70));
//! let text = "Yapay zeka hızla gelişiyor. Doğal dil işleme modelleri çok popüler. \
//!             Galatasaray dün akşam şampiyonluk yolunda önemli bir adım attı.";
//! let chunks = chunker.chunk(text);
//!
//! for chunk in chunks {
//!     println!("Chunk ({} tokens, {}-{}): {}", chunk.token_count, chunk.start_index, chunk.end_index, chunk.text);
//! }
//! ```

pub mod sdpm;
pub mod semantic;
pub mod sentence;
pub mod similarity;
pub mod types;

pub use sdpm::SDPMChunker;
pub use semantic::SemanticChunker;
pub use sentence::SentenceChunker;
pub use similarity::{
    calculate_quantile, calculate_threshold, compute_windowed_similarities, moving_average_filter,
    savitzky_golay_filter,
};
pub use types::{Chunk, SentenceSpan, ThresholdMode};
