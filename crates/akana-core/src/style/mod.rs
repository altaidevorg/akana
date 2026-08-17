//! Turkish Style Auditor, AI Writing Signature Detection, and Humanizer Engine.

pub mod cliches;
pub mod style_metrics;
pub mod detector;
pub mod humanizer;

pub use cliches::*;
pub use style_metrics::*;
pub use detector::*;
pub use humanizer::*;
