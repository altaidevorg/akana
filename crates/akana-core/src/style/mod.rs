//! Turkish Style Auditor, AI Writing Signature Detection, and Humanizer Engine.

mod ai_detector;
pub mod cliches;
pub mod detector;
pub mod humanizer;
pub mod style_metrics;

pub use ai_detector::{
    AiDetection, AiPattern, AiSignal, AiVerdict, PatternKind, ScopeKind, ScopeWarning,
};
pub use cliches::*;
pub use detector::*;
pub use humanizer::*;
pub use style_metrics::*;
