//! Syntactic morphology for Turkish: the morphological analyzer's readings split into
//! inflectional groups (IGs) with Universal Dependencies features, in the format of Öztürel,
//! Kayadelen and Demirşahin, "A Syntactically Expressive Morphological Analyzer for Turkish"
//! (FSMNLP 2019).

mod analyzer;
pub mod types;

pub use analyzer::TurkishSyntacticMorphology;
pub use types::{InflectionalGroup, SyntacticParse};
