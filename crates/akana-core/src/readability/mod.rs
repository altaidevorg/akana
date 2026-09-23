//! Turkish Readability Analysis Engine.
//!
//! Provides the modern 2025 Kalyoncu multi-regression readability equations (up to R²=0.99)
//! along with classical Turkish metrics (Ateşman, Çetinkaya-Uzun, Bezirci-Yılmaz).

pub mod kalyoncu;
pub mod legacy;
pub mod metrics;

pub use kalyoncu::KalyoncuAnalyzer;
pub use legacy::LegacyReadability;
pub use metrics::*;

/// High-level convenience function to analyze Turkish text readability.
pub fn analyze_readability(text: &str) -> ReadabilityReport {
    let analyzer = KalyoncuAnalyzer::new();
    let stats = analyzer.calculate_statistics(text);
    let kalyoncu = analyzer.evaluate(&stats);
    let legacy = LegacyReadability::evaluate(&stats);

    ReadabilityReport {
        statistics: stats,
        kalyoncu,
        legacy,
    }
}
