//! Sentence-based text chunker.
//!
//! Groups sentences up to a maximum token count (`chunk_size`) with optional token overlap.

use super::types::Chunk;
use crate::embeddings::TurkishEmbeddings;
use crate::tokenization::SentenceSegmenter;
use std::sync::Arc;

/// Rule-based Sentence Chunker that respects sentence boundaries and token limits.
pub struct SentenceChunker {
    chunk_size: usize,
    chunk_overlap: usize,
    min_sentences_per_chunk: usize,
    embeddings: Arc<TurkishEmbeddings>,
}

impl Default for SentenceChunker {
    fn default() -> Self {
        Self::new(512, 0, 1)
    }
}

impl SentenceChunker {
    /// Creates a new `SentenceChunker` with specified parameters and a newly initialized embedding engine.
    pub fn new(chunk_size: usize, chunk_overlap: usize, min_sentences_per_chunk: usize) -> Self {
        Self {
            chunk_size: chunk_size.max(1),
            chunk_overlap,
            min_sentences_per_chunk: min_sentences_per_chunk.max(1),
            embeddings: Arc::new(TurkishEmbeddings::new()),
        }
    }

    /// Creates a `SentenceChunker` sharing an existing `TurkishEmbeddings` instance.
    pub fn with_embeddings(
        chunk_size: usize,
        chunk_overlap: usize,
        min_sentences_per_chunk: usize,
        embeddings: Arc<TurkishEmbeddings>,
    ) -> Self {
        Self {
            chunk_size: chunk_size.max(1),
            chunk_overlap,
            min_sentences_per_chunk: min_sentences_per_chunk.max(1),
            embeddings,
        }
    }

    /// Chunks input text into a list of `Chunk` objects.
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

        // Structure for internal sentence spans
        struct Span {
            text: String,
            byte_start: usize,
            byte_end: usize,
            char_start: usize,
            char_end: usize,
            token_count: usize,
        }

        let spans: Vec<Span> = raw_sentences
            .into_iter()
            .map(|s| {
                let token_count = self.embeddings.tokenize(s.text).len();
                let char_start = text[..s.start].chars().count();
                let char_end = text[..s.end].chars().count();
                Span {
                    text: s.text.to_string(),
                    byte_start: s.start,
                    byte_end: s.end,
                    char_start,
                    char_end,
                    token_count: token_count.max(1),
                }
            })
            .collect();

        let mut chunks = Vec::new();
        let mut i = 0;
        let n = spans.len();

        while i < n {
            let mut current_sentences = Vec::new();
            let mut current_tokens = 0;
            let byte_start = spans[i].byte_start;
            let mut byte_end = spans[i].byte_end;
            let char_start = spans[i].char_start;
            let mut char_end = spans[i].char_end;
            let mut j = i;

            while j < n {
                let s = &spans[j];

                // If adding next sentence exceeds chunk_size and we have met min_sentences_per_chunk
                if current_tokens + s.token_count > self.chunk_size
                    && current_sentences.len() >= self.min_sentences_per_chunk
                {
                    break;
                }

                current_sentences.push(s.text.clone());
                current_tokens += s.token_count;
                byte_end = s.byte_end;
                char_end = s.char_end;
                j += 1;
            }

            // Fallback for single oversized sentence that exceeds chunk_size
            if current_sentences.is_empty() && j < n {
                let s = &spans[j];
                current_sentences.push(s.text.clone());
                current_tokens += s.token_count;
                byte_end = s.byte_end;
                char_end = s.char_end;
                j += 1;
            }

            let chunk_text = text[byte_start..byte_end].to_string();
            chunks.push(Chunk::new(
                chunk_text,
                char_start,
                char_end,
                current_tokens,
                current_sentences,
            ));

            if j >= n {
                break;
            }

            // Calculate overlap: step back sentences up to chunk_overlap tokens
            if self.chunk_overlap > 0 && j > i {
                let mut overlap_tokens = 0;
                let mut next_i = j;
                while next_i > i + 1 {
                    let prev_tokens = spans[next_i - 1].token_count;
                    if overlap_tokens + prev_tokens > self.chunk_overlap {
                        break;
                    }
                    overlap_tokens += prev_tokens;
                    next_i -= 1;
                }
                i = if next_i < j { next_i } else { j };
            } else {
                i = j;
            }
        }

        chunks
    }

    /// Chunks a batch of texts.
    pub fn chunk_batch(&self, texts: &[&str]) -> Vec<Vec<Chunk>> {
        texts.iter().map(|t| self.chunk(t)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sentence_chunker_basic() {
        let chunker = SentenceChunker::new(50, 0, 1);
        let text = "Türkiye'nin başkenti Ankara'dır. İstanbul ise en kalabalık şehridir. İzmir Ege'nin incisidir.";
        let chunks = chunker.chunk(text);

        assert!(!chunks.is_empty());
        assert_eq!(chunks[0].start_index, 0);
        assert_eq!(chunks.last().unwrap().end_index, text.chars().count());
    }

    #[test]
    fn test_sentence_chunker_overlap() {
        let chunker = SentenceChunker::new(15, 5, 1);
        let text = "Birinci cümle buradadır. İkinci cümle buradadır. Üçüncü cümle buradadır. Dördüncü cümle buradadır.";
        let chunks = chunker.chunk(text);
        assert!(chunks.len() >= 2);
    }
}
