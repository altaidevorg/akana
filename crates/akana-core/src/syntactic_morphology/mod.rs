//! Syntactic & Expressive Morphology Engine for Turkish (FSMNLP 2019 / Google-style).
//! Structured around Inflectional Groups (IG) and Universal Dependencies (UD) features.

pub mod types;
pub mod lexicon;
mod analyzer;

pub use types::{InflectionalGroup, SyntacticParse};
pub use lexicon::{SyntacticLexicon, SyntacticPOS, SyntacticRootFlags, SyntacticRootEntry};
pub use analyzer::TurkishSyntacticMorphology;
