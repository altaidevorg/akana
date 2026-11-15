//! Normalization, spell checking, and de-asciification module.

pub mod asciifier;
pub mod deasciifier;
pub mod informal;
pub mod numbers;
#[cfg(feature = "spellcheck")]
pub mod spellcheck;

pub use asciifier::*;
pub use deasciifier::*;
pub use informal::*;
pub use numbers::*;
#[cfg(feature = "spellcheck")]
pub use spellcheck::*;
