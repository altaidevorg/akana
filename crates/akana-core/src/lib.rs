//! Akana: Modern and blazingly fast Turkish NLP toolkit in Rust.

pub mod phonology;
pub mod tokenization;
pub mod normalization;
pub mod morphology;
pub mod parser;
pub mod readability;
pub mod ner;
pub mod analysis;
pub mod style;

pub use phonology::*;
pub use tokenization::*;
pub use normalization::*;
pub use morphology::*;
pub use parser::*;
pub use readability::*;
pub use ner::*;
pub use analysis::*;
pub use style::*;

/// High-level document model for Turkish NLP analysis.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TurkishAnalysisDoc {
    pub text: String,
    pub sentences: Vec<TurkishSentenceDoc>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TurkishSentenceDoc {
    pub text: String,
    pub tokens: Vec<String>,
    pub parses: Vec<morphology::MorphParse>,
    pub dependency_tree: parser::DependencyTree,
}

pub struct AkanaEngine {
    pub morphology: morphology::TurkishMorphology,
    pub disambiguator: morphology::MorphologicalDisambiguator,
    pub parser: parser::TurkishDependencyParser,
    pub spell_checker: normalization::TurkishSpellChecker,
}

impl Default for AkanaEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl AkanaEngine {
    pub fn new() -> Self {
        let morphology = morphology::TurkishMorphology::new();
        let disambiguator = morphology::MorphologicalDisambiguator::new();
        let parser = parser::TurkishDependencyParser::new();
        let spell_checker = normalization::TurkishSpellChecker::new();

        Self {
            morphology,
            disambiguator,
            parser,
            spell_checker,
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
            let dep_tree = self.parser.parse(&tokens);

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
