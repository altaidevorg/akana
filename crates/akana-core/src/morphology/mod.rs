//! Turkish Morphology engine.

/// The Zemberek root lexicon (one entry per line), embedded once and shared by every
/// component that reads it.
///
/// The style auditor's AI-text detector was trained on this behaviour: changing it shifts the
/// detector's inputs and fails `style::ai_detector::tests::outputs_match_training_reference`.
pub(crate) static ZEMBEREK_LEXICON: &str = include_str!("zemberek_lexicon.txt");

pub mod analyzer;
pub mod compound;
pub mod dictionary;
pub mod disambiguator;
pub mod generator;
pub mod graph;
pub mod pos;
pub mod stemmer;
pub mod stopwords;
pub mod suffixes;

pub use analyzer::*;
pub use compound::*;
pub use dictionary::*;
pub use disambiguator::*;
pub use generator::*;
pub use graph::*;
pub use pos::*;
pub use stemmer::*;
pub use stopwords::*;
pub use suffixes::*;
