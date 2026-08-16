//! Turkish Morphology engine.

pub mod pos;
pub mod dictionary;
pub mod suffixes;
pub mod graph;
pub mod analyzer;
pub mod generator;
pub mod disambiguator;
pub mod compound;

pub use pos::*;
pub use dictionary::*;
pub use suffixes::*;
pub use graph::*;
pub use analyzer::*;
pub use generator::*;
pub use disambiguator::*;
pub use compound::*;
