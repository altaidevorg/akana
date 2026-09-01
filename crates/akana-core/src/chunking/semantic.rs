//! Semantic text chunker using embedding similarity and boundary detection.
//!
//! Segments text at semantic topic transitions while respecting size and coherence constraints.

use super::similarity::{calculate_threshold, moving_average_filter, savitzky_golay_filter};
use super::types::{Chunk, SentenceSpan, ThresholdMode};
use crate::embeddings::{cosine_similarity, TurkishEmbeddings};
use crate::tokenization::SentenceSegmenter;
use std::sync::Arc;

/// Semantic Chunker that splits text based on semantic similarity drops.
pub struct SemanticChunker {
    chunk_size: usize,
    threshold_mode: ThresholdMode,
    similarity_window: usize,
    min_chunk_size: usize,
    min_sentences_per_chunk: usize,
    use_smoothing: bool,
    embeddings: Arc<TurkishEmbeddings>,
}

impl Default for SemanticChunker {
    fn default() -> Self {
        Self::new(512, ThresholdMode::Percentile(0.75))
    }
}

impl SemanticChunker {
    /// Creates a new `SemanticChunker` with default window and smoothing settings.
    pub fn new(chunk_size: usize, threshold_mode: ThresholdMode) -> Self {
        Self {
            chunk_size: chunk_size.max(1),
            threshold_mode,
            similarity_window: 1,
            min_chunk_size: 30,
            min_sentences_per_chunk: 1,
            use_smoothing: true,
            embeddings: Arc::new(TurkishEmbeddings::new()),
        }
    }

    /// Creates a `SemanticChunker` sharing an existing `TurkishEmbeddings` instance.
    pub fn with_embeddings(
        chunk_size: usize,
        threshold_mode: ThresholdMode,
        embeddings: Arc<TurkishEmbeddings>,
    ) -> Self {
        Self {
            chunk_size: chunk_size.max(1),
            threshold_mode,
            similarity_window: 1,
            min_chunk_size: 30,
            min_sentences_per_chunk: 1,
            use_smoothing: true,
            embeddings,
        }
    }

    /// Builder method to set context window size for similarity calculations.
    pub fn with_similarity_window(mut self, window_size: usize) -> Self {
        self.similarity_window = window_size.max(1);
        self
    }

    /// Builder method to set minimum chunk size in tokens before a split is allowed.
    pub fn with_min_chunk_size(mut self, min_chunk_size: usize) -> Self {
        self.min_chunk_size = min_chunk_size;
        self
    }

    /// Builder method to set minimum sentences per chunk.
    pub fn with_min_sentences_per_chunk(mut self, min_sentences: usize) -> Self {
        self.min_sentences_per_chunk = min_sentences.max(1);
        self
    }

    /// Builder method to toggle similarity smoothing filter.
    pub fn with_smoothing(mut self, use_smoothing: bool) -> Self {
        self.use_smoothing = use_smoothing;
        self
    }

    /// Chunks input text semantically into a list of `Chunk`s.
    pub fn chunk(&self, text: &str) -> Vec<Chunk> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Vec::new();
        }

        let raw_sentences = SentenceSegmenter::segment(text);
        if raw_sentences.is_empty() {
            let tokens = self.embeddings.tokenize(text);
            return vec![Chunk::new(
                text.to_string(),
                0,
                text.len(),
                tokens.len(),
                vec![text.to_string()],
            )];
        }

        if raw_sentences.len() == 1 {
            let s = &raw_sentences[0];
            let token_count = self.embeddings.tokenize(s.text).len();
            let char_start = text[..s.start].chars().count();
            let char_end = text[..s.end].chars().count();
            return vec![Chunk::new(
                text[s.start..s.end].to_string(),
                char_start,
                char_end,
                token_count.max(1),
                vec![s.text.to_string()],
            )];
        }

        // 1. Compute spans, token counts, and embeddings for all sentences
        let spans: Vec<SentenceSpan> = raw_sentences
            .into_iter()
            .map(|s| {
                let token_count = self.embeddings.tokenize(s.text).len().max(1);
                let emb = self.embeddings.embed(s.text);
                let char_start = text[..s.start].chars().count();
                let char_end = text[..s.end].chars().count();
                SentenceSpan::new(
                    s.text.to_string(),
                    s.start,
                    s.end,
                    char_start,
                    char_end,
                    token_count,
                    Some(emb),
                )
            })
            .collect();

        let n = spans.len();
        let mut pairwise_sims = Vec::with_capacity(n.saturating_sub(1));
        let mut has_para_break = Vec::with_capacity(n.saturating_sub(1));

        for i in 0..n.saturating_sub(1) {
            let sim = cosine_similarity(
                spans[i].embedding.as_ref().unwrap(),
                spans[i + 1].embedding.as_ref().unwrap(),
            );
            pairwise_sims.push(sim);

            let inter_slice = &text[spans[i].byte_end..spans[i + 1].byte_start];
            has_para_break.push(inter_slice.contains('\n'));
        }

        // 2. Smooth similarities to eliminate high-frequency noise if enabled
        let similarities = if self.use_smoothing && pairwise_sims.len() >= 10 {
            savitzky_golay_filter(&pairwise_sims)
        } else if self.use_smoothing && pairwise_sims.len() >= 6 {
            moving_average_filter(&pairwise_sims, 3)
        } else {
            pairwise_sims.clone()
        };

        // 3. Calculate threshold cutoff
        let threshold = calculate_threshold(&similarities, &self.threshold_mode);

        // 4. Identify true breakpoint valleys (raw local minima drops OR paragraph breaks with similarity drop)
        let mut is_breakpoint = vec![false; n];
        for i in 0..pairwise_sims.len() {
            let raw_sim = pairwise_sims[i];
            let is_drop = raw_sim < threshold;

            let is_raw_valley = (i == 0 || raw_sim <= pairwise_sims[i - 1])
                && (i == pairwise_sims.len() - 1 || raw_sim <= pairwise_sims[i + 1]);

            let is_para_drop = has_para_break[i] && raw_sim < threshold * 1.25;

            if (is_drop && is_raw_valley) || is_para_drop {
                is_breakpoint[i + 1] = true; // Cut before span i + 1
            }
        }

        // 5. Build chunks based on breakpoints and token limit constraints
        let mut chunks = Vec::new();
        let mut current_sentences = Vec::new();
        let mut current_tokens = 0;
        let mut byte_start = spans[0].byte_start;
        let mut byte_end = spans[0].byte_end;
        let mut char_start = spans[0].char_start;
        let mut char_end = spans[0].char_end;

        for (i, span) in spans.iter().enumerate() {
            // Check if we should split before adding this sentence
            if !current_sentences.is_empty() {
                let exceeds_size = current_tokens + span.token_count > self.chunk_size;
                let meets_min_size = current_tokens >= self.min_chunk_size;
                let meets_min_sentences = current_sentences.len() >= self.min_sentences_per_chunk;

                let is_cut = is_breakpoint[i] && meets_min_size && meets_min_sentences;

                if exceeds_size || is_cut {
                    let chunk_text = text[byte_start..byte_end].to_string();
                    chunks.push(Chunk::new(
                        chunk_text,
                        char_start,
                        char_end,
                        current_tokens,
                        std::mem::take(&mut current_sentences),
                    ));

                    current_tokens = 0;
                    byte_start = span.byte_start;
                    char_start = span.char_start;
                }
            }

            // Append sentence to current chunk
            current_sentences.push(span.text.clone());
            current_tokens += span.token_count;
            byte_end = span.byte_end;
            char_end = span.char_end;
        }

        // Push leftover chunk
        if !current_sentences.is_empty() {
            let chunk_text = text[byte_start..byte_end].to_string();
            chunks.push(Chunk::new(
                chunk_text,
                char_start,
                char_end,
                current_tokens,
                current_sentences,
            ));
        }

        chunks
    }

    /// Chunks a batch of texts semantically.
    pub fn chunk_batch(&self, texts: &[&str]) -> Vec<Vec<Chunk>> {
        texts.iter().map(|t| self.chunk(t)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_chunker_multi_topic() {
        let chunker = SemanticChunker::new(512, ThresholdMode::Percentile(0.75))
            .with_min_chunk_size(10);

        let text = "Kuantum fiziği, atom ve atom altı parçacıkların davranışlarını inceler. Dalga-parçacık ikiliği bu teorinin temelidir. \
                    Fenerbahçe, Süper Lig'de dün akşam kritik bir galibiyet aldı. Forvet oyuncusu iki gol atarak maça damgasını vurdu.";

        let chunks = chunker.chunk(text);
        assert!(chunks.len() >= 2, "Expected at least 2 topic chunks, got {}", chunks.len());
        assert!(chunks[0].text.contains("Kuantum"));
        assert!(chunks.iter().any(|c| c.text.contains("Fenerbahçe")));
    }
}
