//! Akana: Modern and blazingly fast Turkish NLP toolkit in Rust.
//!
//! # Cargo features
//!
//! The default feature, `full`, builds everything a build without features had before the
//! split (0.11.0). Phonology, tokenization, morphology, `syntactic_morphology`, normalization
//! (without the spell checker), `lexicon`, `analysis` and `readability` are always built. The
//! other modules each have a feature: `speech`, `parser` (with [`AkanaEngine`]), `ner`,
//! `embeddings` (with `chunking`), and the ones that compile StringZilla, which is C:
//! `spellcheck`, `pii`, `grammar` and `style`. `pii-tagger` and `pii-download` stay opt-in.
//!
//! `default-features = false, features = ["speech"]` is the speech front end alone: pure Rust,
//! and it builds for `wasm32-unknown-unknown`.

pub mod analysis;
#[cfg(feature = "embeddings")]
pub mod chunking;
#[cfg(feature = "embeddings")]
pub mod embeddings;
// `fingerprint::of` serves the embeddings' spec and `SentenceSegmenter::first_sentence_end` the AI
// detector, so they are dead in builds without those. Both modules' sources are hashed into the
// encoders' fingerprints (build.rs), so the allowance sits here instead of on the items.
#[cfg_attr(not(feature = "embeddings"), allow(dead_code))]
mod fingerprint;
#[cfg(feature = "grammar")]
pub mod grammar;
pub mod lexicon;
pub mod morphology;
#[cfg(feature = "ner")]
pub mod ner;
pub mod normalization;
#[cfg(feature = "parser")]
pub mod parser;
pub mod phonology;
#[cfg(feature = "pii")]
pub mod pii;
pub mod readability;
#[cfg(feature = "speech")]
pub mod speech;
#[cfg(feature = "style")]
pub mod style;
pub mod syntactic_morphology;
#[cfg_attr(not(feature = "style"), allow(dead_code))]
pub mod tokenization;

pub use analysis::*;
#[cfg(feature = "embeddings")]
pub use chunking::{Chunk, SDPMChunker, SemanticChunker, SentenceChunker, ThresholdMode};
#[cfg(feature = "embeddings")]
pub use embeddings::{
    cosine_distance_2bit_64, cosine_similarity, dot_product_2bit_64,
    get_usearch_metric_pointer_1bit, get_usearch_metric_pointer_2bit,
    get_usearch_metric_pointer_hybrid, hamming_distance_1bit_32, hybrid_distance_128,
    reciprocal_rank_fusion, set_hybrid_metric_alpha, FusedResult, MorphologicalSparseEncoder,
    MorphologicalSparseVector, TurkishEmbeddings, TurkishHybridEmbeddings, EMBEDDING_SCALE_2BIT,
    HYBRID_VECTOR_BYTES, PACKED_1BIT_BYTES, PACKED_2BIT_BYTES,
};
#[cfg(feature = "grammar")]
pub use grammar::*;
pub use morphology::*;
#[cfg(feature = "ner")]
pub use ner::*;
pub use normalization::*;
#[cfg(feature = "parser")]
pub use parser::*;
pub use phonology::*;
#[cfg(feature = "pii")]
pub use pii::*;
pub use readability::*;
#[cfg(feature = "style")]
pub use style::*;
pub use syntactic_morphology::*;
pub use tokenization::*;

/// High-level document model for Turkish NLP analysis.
#[cfg(feature = "parser")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TurkishAnalysisDoc {
    pub text: String,
    pub sentences: Vec<TurkishSentenceDoc>,
}

#[cfg(feature = "parser")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TurkishSentenceDoc {
    pub text: String,
    pub tokens: Vec<String>,
    pub parses: Vec<morphology::MorphParse>,
    pub dependency_tree: parser::DependencyTree,
}

/// The end-to-end pipeline: sentence segmentation, tokenization, morphological
/// disambiguation and dependency parsing. Its components share one morphology.
#[cfg(feature = "parser")]
pub struct AkanaEngine {
    pub morphology: morphology::TurkishMorphology,
    pub disambiguator: morphology::MorphologicalDisambiguator,
    pub parser: parser::TurkishDependencyParser,
}

#[cfg(feature = "parser")]
impl Default for AkanaEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "parser")]
impl AkanaEngine {
    pub fn new() -> Self {
        let morphology = morphology::TurkishMorphology::new();
        let disambiguator =
            morphology::MorphologicalDisambiguator::with_morphology(morphology.clone());
        let parser = parser::TurkishDependencyParser::with_morphology(morphology.clone());

        Self {
            morphology,
            disambiguator,
            parser,
        }
    }

    /// Performs full end-to-end NLP analysis (sentence segmentation, tokenization, morphological disambiguation, and dependency parsing).
    pub fn analyze_document(&self, text: &str) -> TurkishAnalysisDoc {
        let raw_sentences = tokenization::SentenceSegmenter::segment(text);
        let mut sentence_docs = Vec::with_capacity(raw_sentences.len());

        for sent in raw_sentences {
            let tokens: Vec<&str> = tokenization::TurkishTokenizer::tokenize_words(sent.text);
            let token_strings: Vec<String> = tokens.iter().map(|&s| s.to_string()).collect();
            let disambiguated = self.disambiguator.disambiguate(&tokens);
            let dep_tree = self.parser.parse_disambiguated(&tokens, &disambiguated);

            sentence_docs.push(TurkishSentenceDoc {
                text: sent.text.to_string(),
                tokens: token_strings,
                parses: disambiguated,
                dependency_tree: dep_tree,
            });
        }

        TurkishAnalysisDoc {
            text: text.to_string(),
            sentences: sentence_docs,
        }
    }
}
