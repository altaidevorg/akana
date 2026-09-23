//! Minimal Unigram (SentencePiece-style) tokenizer for the Model2Vec embedding model.
//!
//! This implements just enough of the Unigram algorithm to tokenize Turkish text
//! for embedding lookup. The vocabulary and scores are loaded from the embedded
//! HuggingFace `tokenizer.json`.

use ahash::AHashMap;

/// Special token IDs.
pub const BOS_ID: usize = 0; // <s>
pub const PAD_ID: usize = 1; // <pad>
pub const EOS_ID: usize = 2; // </s>
pub const UNK_ID: usize = 3; // <unk>

/// The Unicode replacement character used by SentencePiece Metaspace (▁ = U+2581).
const METASPACE: char = '\u{2581}';

/// Maximum subword length to consider during Viterbi segmentation.
const MAX_PIECE_LEN: usize = 64;

/// A vocabulary entry: the piece string and its log-probability score.
#[derive(Debug, Clone)]
struct VocabEntry {
    piece: String,
    score: f64,
}

/// Minimal Unigram tokenizer.
pub struct UnigramTokenizer {
    /// Vocabulary entries indexed by token ID.
    vocab: Vec<VocabEntry>,
    /// Piece string → token ID lookup.
    piece_to_id: AHashMap<String, usize>,
    /// UNK token ID.
    unk_id: usize,
}

impl UnigramTokenizer {
    /// Load the tokenizer from the embedded `tokenizer.json` bytes.
    pub fn from_json_bytes(data: &[u8]) -> Result<Self, TokenizerError> {
        let json_str = std::str::from_utf8(data)
            .map_err(|_| TokenizerError::InvalidJson("non-utf8 tokenizer json"))?;

        // Parse the JSON manually with serde_json
        let root: serde_json::Value = serde_json::from_str(json_str)
            .map_err(|e| TokenizerError::InvalidJson(Box::leak(e.to_string().into_boxed_str())))?;

        let model = root
            .get("model")
            .ok_or(TokenizerError::InvalidJson("missing 'model' key"))?;

        let model_type = model
            .get("type")
            .and_then(|v| v.as_str())
            .ok_or(TokenizerError::InvalidJson("missing model type"))?;

        if model_type != "Unigram" {
            return Err(TokenizerError::InvalidJson("expected Unigram model type"));
        }

        let unk_id = model
            .get("unk_id")
            .and_then(|v| v.as_u64())
            .unwrap_or(UNK_ID as u64) as usize;

        let vocab_array = model
            .get("vocab")
            .and_then(|v| v.as_array())
            .ok_or(TokenizerError::InvalidJson("missing 'vocab' array"))?;

        let mut vocab = Vec::with_capacity(vocab_array.len());
        let mut piece_to_id = AHashMap::with_capacity(vocab_array.len());

        for (idx, entry) in vocab_array.iter().enumerate() {
            let arr = entry
                .as_array()
                .ok_or(TokenizerError::InvalidJson("vocab entry is not an array"))?;

            if arr.len() < 2 {
                return Err(TokenizerError::InvalidJson(
                    "vocab entry has fewer than 2 elements",
                ));
            }

            let piece = arr[0]
                .as_str()
                .ok_or(TokenizerError::InvalidJson("vocab piece is not a string"))?
                .to_string();
            let score = arr[1]
                .as_f64()
                .ok_or(TokenizerError::InvalidJson("vocab score is not a number"))?;

            piece_to_id.insert(piece.clone(), idx);
            vocab.push(VocabEntry { piece, score });
        }

        Ok(Self {
            vocab,
            piece_to_id,
            unk_id,
        })
    }

    /// Tokenize a text string into token IDs without special tokens (standard for Model2Vec static embeddings).
    ///
    /// Applies Metaspace pre-tokenization (prepend ▁, split on whitespace),
    /// then Viterbi segmentation per word.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut token_ids = Vec::new();

        // Normalize: NFKC (lightweight — just handle basic cases for Turkish)
        let normalized = normalize_nfkc_light(text);

        // Pre-tokenize: Metaspace + WhitespaceSplit
        let words = metaspace_pretokenize(&normalized);

        for word in &words {
            let segmented = self.viterbi_segment(word);
            token_ids.extend(segmented);
        }

        token_ids
    }

    /// Tokenize a text string into token IDs wrapped with BOS (<s>) and EOS (</s>).
    pub fn encode_with_special(&self, text: &str) -> Vec<usize> {
        let mut token_ids = Vec::with_capacity(2);
        token_ids.push(BOS_ID);
        token_ids.extend(self.encode(text));
        token_ids.push(EOS_ID);
        token_ids
    }

    /// Viterbi best-path segmentation of a single pre-tokenized word.
    fn viterbi_segment(&self, text: &str) -> Vec<usize> {
        if text.is_empty() {
            return vec![];
        }

        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();

        // best_score[i] = best log-probability to segment chars[0..i]
        // best_end[i] = length of the last piece ending at position i
        let mut best_score = vec![f64::NEG_INFINITY; n + 1];
        let mut best_end = vec![0usize; n + 1];
        best_score[0] = 0.0;

        for i in 1..=n {
            // Try all possible last pieces ending at position i
            let max_len = i.min(MAX_PIECE_LEN);
            for len in 1..=max_len {
                let start = i - len;
                if best_score[start] == f64::NEG_INFINITY {
                    continue;
                }

                let piece: String = chars[start..i].iter().collect();
                let (score, _id) = if let Some(&id) = self.piece_to_id.get(&piece) {
                    (self.vocab[id].score, id)
                } else {
                    continue; // This piece is not in vocab
                };

                let candidate = best_score[start] + score;
                if candidate > best_score[i] {
                    best_score[i] = candidate;
                    best_end[i] = len;
                }
            }

            // If no piece was found ending at i, try single-char UNK fallback
            if best_score[i] == f64::NEG_INFINITY && best_score[i - 1] > f64::NEG_INFINITY {
                // Use UNK for single character
                best_score[i] = best_score[i - 1] + self.vocab[self.unk_id].score;
                best_end[i] = 1;
            }
        }

        // Backtrace to get the segmentation
        let mut result = Vec::new();
        let mut pos = n;
        while pos > 0 {
            let len = best_end[pos];
            if len == 0 {
                // Fallback: consume one character as UNK
                result.push(self.unk_id);
                pos -= 1;
            } else {
                let piece: String = chars[pos - len..pos].iter().collect();
                let id = self.piece_to_id.get(&piece).copied().unwrap_or(self.unk_id);
                result.push(id);
                pos -= len;
            }
        }

        result.reverse();
        result.dedup_by(|a, b| *a == self.unk_id && *b == self.unk_id);
        result
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Get the string piece for a given token ID.
    pub fn id_to_piece(&self, id: usize) -> Option<&str> {
        self.vocab.get(id).map(|e| e.piece.as_str())
    }
}

/// Metaspace pre-tokenization: split on whitespace and prepend ▁ to each word.
///
/// "merhaba dünya" → ["▁merhaba", "▁dünya"]
fn metaspace_pretokenize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|word| format!("{METASPACE}{word}"))
        .collect()
}

use unicode_normalization::UnicodeNormalization;

/// NFKC Unicode normalization (matches HuggingFace Tokenizer normalizer pipeline).
fn normalize_nfkc_light(text: &str) -> String {
    text.nfkc().collect()
}

/// Errors from tokenizer operations.
#[derive(Debug)]
pub enum TokenizerError {
    InvalidJson(&'static str),
}

impl std::fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TokenizerError::InvalidJson(msg) => write!(f, "invalid tokenizer JSON: {msg}"),
        }
    }
}

impl std::error::Error for TokenizerError {}

#[cfg(test)]
mod tests {
    use super::*;

    static TOKENIZER_DATA: &[u8] = include_bytes!("../../../../data/embeddings/tokenizer.json");

    #[test]
    fn test_load_tokenizer() {
        let tok =
            UnigramTokenizer::from_json_bytes(TOKENIZER_DATA).expect("failed to load tokenizer");
        assert_eq!(tok.vocab_size(), 39655);
    }

    #[test]
    fn test_special_tokens() {
        let tok =
            UnigramTokenizer::from_json_bytes(TOKENIZER_DATA).expect("failed to load tokenizer");
        assert_eq!(tok.id_to_piece(BOS_ID), Some("<s>"));
        assert_eq!(tok.id_to_piece(PAD_ID), Some("<pad>"));
        assert_eq!(tok.id_to_piece(EOS_ID), Some("</s>"));
        assert_eq!(tok.id_to_piece(UNK_ID), Some("<unk>"));
    }

    #[test]
    fn test_known_tokens() {
        let tok =
            UnigramTokenizer::from_json_bytes(TOKENIZER_DATA).expect("failed to load tokenizer");

        // Known from our verification: ▁merhaba -> id 38114
        assert_eq!(tok.piece_to_id.get("▁merhaba"), Some(&38114));
        // ▁dünya -> id 7275
        assert_eq!(tok.piece_to_id.get("▁dünya"), Some(&7275));
    }

    #[test]
    fn test_encode_wraps_with_bos_eos() {
        let tok =
            UnigramTokenizer::from_json_bytes(TOKENIZER_DATA).expect("failed to load tokenizer");
        let ids_special = tok.encode_with_special("merhaba dünya");

        // Should start with BOS and end with EOS
        assert_eq!(ids_special[0], BOS_ID);
        assert_eq!(*ids_special.last().unwrap(), EOS_ID);

        let ids = tok.encode("merhaba dünya");
        assert_ne!(ids[0], BOS_ID);
    }

    #[test]
    fn test_encode_contains_known_tokens() {
        let tok =
            UnigramTokenizer::from_json_bytes(TOKENIZER_DATA).expect("failed to load tokenizer");
        let ids = tok.encode("merhaba dünya");

        // ▁merhaba (38114) should be in the output (if it's a single piece)
        // or the constituent pieces should be present
        assert!(
            ids.contains(&38114) || ids.len() > 4,
            "expected ▁merhaba token or sub-pieces, got {ids:?}"
        );
    }

    #[test]
    fn test_metaspace_pretokenize() {
        let words = metaspace_pretokenize("merhaba dünya");
        assert_eq!(words, vec!["▁merhaba", "▁dünya"]);
    }
}
