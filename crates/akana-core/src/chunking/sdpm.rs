//! Semantic Double-Pass Merge (SDPM) Chunker.
//!
//! Inspired by Chonkie's SDPM algorithm:
//! - **Pass 1**: Performs initial fine-grained semantic boundary detection.
//! - **Pass 2**: Merges adjacent cohesive chunks if their semantic similarity >= merge_threshold
//!   and their combined token count <= chunk_size.

use super::semantic::SemanticChunker;
use super::types::{Chunk, ThresholdMode};
use crate::embeddings::TurkishEmbeddings;
use std::sync::Arc;

/// Semantic Double-Pass Merge Chunker for optimal chunk density and semantic coherence.
pub struct SDPMChunker {
    chunk_size: usize,
    threshold_mode: ThresholdMode,
    merge_threshold: f32,
    similarity_window: usize,
    min_chunk_size: usize,
    embeddings: Arc<TurkishEmbeddings>,
}

impl Default for SDPMChunker {
    fn default() -> Self {
        Self::new(512, ThresholdMode::Percentile(0.75), 0.32)
    }
}

impl SDPMChunker {
    /// Creates a new `SDPMChunker`.
    pub fn new(chunk_size: usize, threshold_mode: ThresholdMode, merge_threshold: f32) -> Self {
        Self {
            chunk_size: chunk_size.max(1),
            threshold_mode,
            merge_threshold,
            similarity_window: 1,
            min_chunk_size: 20,
            embeddings: Arc::new(TurkishEmbeddings::new()),
        }
    }

    /// Creates an `SDPMChunker` sharing an existing `TurkishEmbeddings` instance.
    pub fn with_embeddings(
        chunk_size: usize,
        threshold_mode: ThresholdMode,
        merge_threshold: f32,
        embeddings: Arc<TurkishEmbeddings>,
    ) -> Self {
        Self {
            chunk_size: chunk_size.max(1),
            threshold_mode,
            merge_threshold,
            similarity_window: 1,
            min_chunk_size: 20,
            embeddings,
        }
    }

    /// Builder method to set context window size for similarity calculations.
    pub fn with_similarity_window(mut self, window_size: usize) -> Self {
        self.similarity_window = window_size.max(1);
        self
    }

    /// Builder method to set minimum chunk size for the first pass.
    pub fn with_min_chunk_size(mut self, min_chunk_size: usize) -> Self {
        self.min_chunk_size = min_chunk_size;
        self
    }

    /// Chunks input text using Semantic Double-Pass Merge.
    pub fn chunk(&self, text: &str) -> Vec<Chunk> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Vec::new();
        }

        // Pass 1: Fine-grained semantic chunking
        let pass1_chunker = SemanticChunker::with_embeddings(
            self.chunk_size,
            self.threshold_mode.clone(),
            Arc::clone(&self.embeddings),
        )
        .with_similarity_window(self.similarity_window)
        .with_min_chunk_size(self.min_chunk_size)
        .with_min_sentences_per_chunk(1);

        let initial_chunks = pass1_chunker.chunk(text);
        if initial_chunks.len() <= 1 {
            return initial_chunks;
        }

        // Pass 2: Merge adjacent chunks that are semantically coherent and fit in chunk_size
        // Pre-build character-to-byte boundary map for O(1) byte-slice text extraction
        let char_byte_offsets: Vec<usize> = text
            .char_indices()
            .map(|(b, _)| b)
            .chain(std::iter::once(text.len()))
            .collect();

        let mut merged_chunks: Vec<Chunk> = Vec::new();

        for chunk in initial_chunks {
            if let Some(prev) = merged_chunks.last_mut() {
                let combined_tokens = prev.token_count + chunk.token_count;

                if combined_tokens <= self.chunk_size {
                    // Check semantic similarity between the two chunks
                    let sim = self.embeddings.similarity(&prev.text, &chunk.text);

                    if sim >= self.merge_threshold {
                        // Merge `chunk` into `prev`
                        let start_char = prev
                            .start_index
                            .min(char_byte_offsets.len().saturating_sub(1));
                        let end_char = chunk
                            .end_index
                            .min(char_byte_offsets.len().saturating_sub(1));
                        let byte_start = char_byte_offsets[start_char];
                        let byte_end = char_byte_offsets[end_char];
                        let full_text = text[byte_start..byte_end].to_string();

                        prev.text = full_text;
                        prev.end_index = chunk.end_index;
                        prev.token_count = combined_tokens;
                        prev.sentences.extend(chunk.sentences);
                        continue;
                    }
                }
            }

            merged_chunks.push(chunk);
        }

        merged_chunks
    }

    /// Chunks a batch of texts using SDPM.
    pub fn chunk_batch(&self, texts: &[&str]) -> Vec<Vec<Chunk>> {
        texts.iter().map(|t| self.chunk(t)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sdpm_chunker_merging() {
        let chunker = SDPMChunker::new(512, ThresholdMode::Similarity(0.85), 0.50);
        let text = "Yapay zeka sistemleri hızla gelişiyor. Derin öğrenme bu gelişimin motorudur. \
                    Büyük dil modelleri metin üretiminde oldukça başarılıdır. \
                    İtalya mutfağında pizza ve makarna çok sevilir. Espresso kahve kültürü de yaygındır.";

        let chunks = chunker.chunk(text);
        assert!(!chunks.is_empty());
        assert!(chunks.len() <= 3);
    }
}
