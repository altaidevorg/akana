//! Syntactic & Expressive Morphology Engine for Turkish (FSMNLP 2019 / Google-style).
//! Structured around Inflectional Groups (IG) and Universal Dependencies (UD) features.

mod analyzer;
pub mod lexicon;
pub mod types;

pub use analyzer::TurkishSyntacticMorphology;
pub use lexicon::{SyntacticLexicon, SyntacticPOS, SyntacticRootEntry, SyntacticRootFlags};
pub use types::{InflectionalGroup, SyntacticParse};
