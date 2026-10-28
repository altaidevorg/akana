//! Turkish Morphology engine.

/// The Zemberek root lexicon (one entry per line), embedded once and shared by every
/// component that reads it.
///
/// The style auditor's AI-text detector was trained on this behaviour: changing it shifts the
/// detector's inputs and fails `style::ai_detector::tests::inputs_match_drift_canary` (and
/// `outputs_match_training_reference` when its sample shows it).
pub(crate) static ZEMBEREK_LEXICON: &str = include_str!("zemberek_lexicon.txt");

pub mod analyzer;
pub mod compound;
pub mod dictionary;
pub mod disambiguator;
pub mod generator;
pub mod morphotactics;
pub mod phonetics;
pub mod pos;
pub mod pronunciation;
pub mod stemmer;
pub mod stems;
pub mod stopwords;
pub mod turkish_morphotactics;
pub mod verb_form;

pub use analyzer::*;
pub use compound::*;
pub use dictionary::*;
pub use disambiguator::*;
pub use generator::*;
pub use pos::*;
pub use stemmer::*;
pub use stopwords::*;
pub use verb_form::{verb_form, verb_form_of, NonFinite, VerbForm};
