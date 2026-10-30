//! Common types and data structures for text chunking.

use serde::{Deserialize, Serialize};

/// Represents a single text chunk produced by any chunker.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Chunk {
    /// The full text content of the chunk.
    pub text: String,
    /// Starting byte/character offset in the original document.
    pub start_index: usize,
    /// Ending byte/character offset in the original document.
    pub end_index: usize,
    /// Number of tokens contained in this chunk.
    pub token_count: usize,
    /// Individual sentences that constitute this chunk.
    pub sentences: Vec<String>,
}

impl Chunk {
    pub fn new(
        text: String,
        start_index: usize,
        end_index: usize,
        token_count: usize,
        sentences: Vec<String>,
    ) -> Self {
        Self {
            text,
            start_index,
            end_index,
            token_count,
            sentences,
        }
    }
}

/// Threshold calculation mode for semantic boundary detection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ThresholdMode {
    /// Direct cosine similarity cutoff (e.g. 0.7).
    /// Splits occur when similarity between adjacent windows falls below this value.
    Similarity(f32),
    /// Percentile-based cutoff on similarities/drops (0.0 to 1.0, e.g. 0.75 for 75th percentile).
    Percentile(f32),
    /// Standard deviation cutoff (`mean - k * std_dev`, e.g. k = 1.0).
    StandardDeviation(f32),
    /// Interquartile range cutoff (`Q1 - k * IQR`, e.g. k = 1.5).
    Interquartile(f32),
    /// Automatically determines optimal threshold based on similarity distribution.
    Auto,
}

impl Default for ThresholdMode {
    fn default() -> Self {
        ThresholdMode::Percentile(0.75)
    }
}

/// Internal sentence span representation with cached embeddings and token counts.
#[derive(Debug, Clone)]
pub struct SentenceSpan {
    pub text: String,
    pub byte_start: usize,
    pub byte_end: usize,
    pub char_start: usize,
    pub char_end: usize,
    pub token_count: usize,
    pub embedding: Option<Vec<f32>>,
}

impl SentenceSpan {
    pub fn new(
        text: String,
        byte_start: usize,
        byte_end: usize,
        char_start: usize,
        char_end: usize,
        token_count: usize,
        embedding: Option<Vec<f32>>,
    ) -> Self {
        Self {
            text,
            byte_start,
            byte_end,
            char_start,
            char_end,
            token_count,
            embedding,
        }
    }
}

/// Converts byte offsets of a text into char offsets, front to back: each call counts only the
/// chars since the previous offset, so converting every sentence boundary of a document is
/// linear in its length. Counting the whole prefix per sentence made 8 M chars take 53 s.
pub(crate) struct CharOffsets<'a> {
    text: &'a str,
    byte: usize,
    chars: usize,
}

impl<'a> CharOffsets<'a> {
    pub(crate) fn new(text: &'a str) -> Self {
        Self {
            text,
            byte: 0,
            chars: 0,
        }
    }

    /// `text[..byte].chars().count()`. Offsets are expected in increasing order; an earlier one
    /// is counted from the start again.
    pub(crate) fn at(&mut self, byte: usize) -> usize {
        if byte < self.byte {
            (self.byte, self.chars) = (0, 0);
        }
        self.chars += self.text[self.byte..byte].chars().count();
        self.byte = byte;
        self.chars
    }
}
