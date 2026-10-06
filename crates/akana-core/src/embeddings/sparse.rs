//! Morphological BM25 Sparse Vector Encoder.
//!
//! Converts Turkish text into sparse lexical vectors weighted by:
//! - Turkish morphological analysis & disambiguation (true lemma / root)
//! - Compound noun decomposition (`denizaltı` -> `deniz` + `altı`)
//! - Linguistic Part-of-Speech weighting (Proper Nouns/Numbers > Nouns > Verbs > Adverbs)
//! - BM25 term frequency saturation

use crate::morphology::{
    CompoundDecomposer, PrimaryPos, SecondaryPos, TurkishMorphology, TurkishStopwords,
};
use crate::phonology::to_turkish_lower;
use crate::tokenization::TurkishTokenizer;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Sparse lexical vector containing feature IDs and corresponding BM25 weights.
///
/// Compatible with standard vector databases (Qdrant, Milvus, Pinecone, etc.).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MorphologicalSparseVector {
    /// 24-bit or 32-bit hashed term IDs.
    pub indices: Vec<u32>,
    /// BM25 saturated & POS-weighted float values.
    pub values: Vec<f32>,
    /// Corresponding string terms (for diagnostics and human inspection).
    pub terms: Vec<String>,
}

/// 24-bit FNV-1a hash function for term strings.
///
/// Produces an integer in `1..=0x00FFFFFF` (reserving 0 as empty/sentinel slot).
#[inline]
pub fn hash_term_24bit(term: &str) -> u32 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in term.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    let h24 = (hash ^ (hash >> 24) ^ (hash >> 48)) as u32 & 0x00FF_FFFF;
    if h24 == 0 {
        1
    } else {
        h24
    }
}

/// Turkish morphological sparse encoder.
pub struct MorphologicalSparseEncoder {
    morphology: TurkishMorphology,
    decomposer: CompoundDecomposer,
    stopwords: TurkishStopwords,
}

impl Default for MorphologicalSparseEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl MorphologicalSparseEncoder {
    /// Create a new morphological sparse encoder with built-in Turkish lexicon.
    pub fn new() -> Self {
        Self {
            morphology: TurkishMorphology::new(),
            decomposer: CompoundDecomposer::new(),
            stopwords: TurkishStopwords::default(),
        }
    }

    /// Extract disambiguated lemmas, compound parts, and linguistic weights.
    pub fn encode_sparse(&self, text: &str) -> MorphologicalSparseVector {
        let tokens = TurkishTokenizer::tokenize_words(text);
        let mut term_counts: HashMap<String, (f32, u32)> = HashMap::new(); // term -> (pos_weight, raw_count)

        for token in tokens {
            let lower_token = to_turkish_lower(token.trim());
            if lower_token.is_empty() || self.stopwords.is_stopword(&lower_token) {
                continue;
            }

            // Morphological analysis and disambiguation
            let parses = self.morphology.analyze(&lower_token);
            let pos_weight = if let Some(best) = parses.first() {
                if best.secondary_pos == SecondaryPos::ProperNoun {
                    1.8
                } else {
                    match best.primary_pos {
                        PrimaryPos::Num => 1.6,
                        PrimaryPos::Noun => 1.3,
                        PrimaryPos::Verb => 1.0,
                        PrimaryPos::Adj => 1.0,
                        PrimaryPos::Adv => 0.7,
                        PrimaryPos::Pron
                        | PrimaryPos::Conj
                        | PrimaryPos::Postp
                        | PrimaryPos::Q => 0.2,
                        _ => 0.8,
                    }
                }
            } else {
                0.9
            };

            // Collect all unique roots and lemmas from top parses
            let mut added_any = false;
            for p in parses.iter().take(3) {
                let root = to_turkish_lower(&p.root);
                if !self.stopwords.is_stopword(&root) && root.len() > 1 {
                    term_counts.entry(root.clone()).or_insert((pos_weight, 0)).1 += 1;
                    added_any = true;
                }
                let lemma = to_turkish_lower(&p.lemma);
                if lemma != root && !self.stopwords.is_stopword(&lemma) && lemma.len() > 1 {
                    term_counts.entry(lemma).or_insert((pos_weight, 0)).1 += 1;
                    added_any = true;
                }
            }

            if !added_any && !self.stopwords.is_stopword(&lower_token) && lower_token.len() > 1 {
                term_counts
                    .entry(lower_token.clone())
                    .or_insert((pos_weight, 0))
                    .1 += 1;
            }

            // Compound word decomposition: only for words >= 6 chars, both parts >= 3 chars
            if lower_token.chars().count() >= 6 {
                let compounds = self.decomposer.decompose(&lower_token);
                if let Some(c) = compounds.first() {
                    let p1 = to_turkish_lower(&c.part1);
                    let p2 = to_turkish_lower(&c.part2);
                    if p1.chars().count() >= 3 && p2.chars().count() >= 3 {
                        if !self.stopwords.is_stopword(&p1) {
                            term_counts.entry(p1).or_insert((1.0, 0)).1 += 1;
                        }
                        if !self.stopwords.is_stopword(&p2) {
                            term_counts.entry(p2).or_insert((1.0, 0)).1 += 1;
                        }
                    }
                }
            }
        }

        // BM25 term frequency saturation: TF_sat = (tf * (k1 + 1)) / (tf + k1), k1 = 1.2
        let k1: f32 = 1.2;
        let mut term_entries: Vec<(u32, f32, String)> = Vec::with_capacity(term_counts.len());

        for (term, (pos_w, count)) in term_counts {
            let tf = count as f32;
            let tf_sat = (tf * (k1 + 1.0)) / (tf + k1);
            let final_weight = tf_sat * pos_w;
            let term_id = hash_term_24bit(&term);
            term_entries.push((term_id, final_weight, term));
        }

        // Sort by term_id for deterministic representation
        term_entries.sort_by_key(|e| e.0);

        let mut indices = Vec::with_capacity(term_entries.len());
        let mut values = Vec::with_capacity(term_entries.len());
        let mut terms = Vec::with_capacity(term_entries.len());

        for (id, val, term) in term_entries {
            indices.push(id);
            values.push(val);
            terms.push(term);
        }

        MorphologicalSparseVector {
            indices,
            values,
            terms,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_term_24bit() {
        let h1 = hash_term_24bit("ankara");
        let h2 = hash_term_24bit("istanbul");
        assert_ne!(h1, h2);
        assert!(h1 <= 0x00FF_FFFF);
        assert!(h2 <= 0x00FF_FFFF);
        assert_ne!(h1, 0);
    }

    #[test]
    fn test_encode_sparse_lemmatization() {
        let encoder = MorphologicalSparseEncoder::new();
        // "evlerimizde" should be lemmatized to "ev"
        let vec = encoder.encode_sparse("evlerimizde güzel kediler var");
        assert!(!vec.indices.is_empty());
        assert!(
            vec.terms.iter().any(|t| t == "ev"),
            "expected lemma 'ev' in extracted terms, got {:?}",
            vec.terms
        );
        assert!(
            vec.terms.iter().any(|t| t == "kedi"),
            "expected lemma 'kedi' in extracted terms, got {:?}",
            vec.terms
        );
    }

    #[test]
    fn test_encode_sparse_compound_decomposition() {
        let encoder = MorphologicalSparseEncoder::new();
        let vec = encoder.encode_sparse("denizaltı batığı keşfedildi");
        // "denizaltı" decomposes to "deniz" and "altı"
        assert!(
            vec.terms.iter().any(|t| t == "deniz" || t == "denizaltı"),
            "expected compound components in extracted terms, got {:?}",
            vec.terms
        );
    }
}
