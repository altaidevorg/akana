//! Turkish Style Auditor, AI Writing Signature Detection, and Humanizer Engine.

pub mod cliches;
pub mod detector;
pub mod humanizer;
pub mod style_metrics;

pub use cliches::*;
pub use detector::*;
pub use humanizer::*;
pub use style_metrics::*;
