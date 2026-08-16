//! Classic Turkish Readability Formulas: Ateşman (1997), Çetinkaya-Uzun (2010), Bezirci-Yılmaz (2010).

use super::metrics::{FormulaResult, LegacyResults, TextStatistics};

pub struct LegacyReadability;

impl LegacyReadability {
    /// Evaluates text readability using classical Turkish formulas.
    pub fn evaluate(stats: &TextStatistics) -> LegacyResults {
        let osu = stats.syllables_per_word;
        let ocu = stats.words_per_sentence;
        let poly_ratio = stats.polysyllabic_word_ratio;

        // 1. Ateşman (1997)
        // Skor = 198.825 - (40.175 * OSU) - (2.610 * OCU)
        let atesman_score = 198.825 - (40.175 * osu) - (2.610 * ocu);
        let atesman_grade = Self::grade_atesman(atesman_score);

        // 2. Çetinkaya-Uzun (2010)
        // Skor = 118.823 - (25.987 * OSU) - (0.971 * OCU)
        let cetinkaya_score = 118.823 - (25.987 * osu) - (0.971 * ocu);
        let cetinkaya_grade = Self::grade_cetinkaya(cetinkaya_score);

        // 3. Bezirci-Yılmaz (2010)
        // Skor = sqrt(OCU * Polysyllabic_Ratio)
        let bezirci_score = (ocu * poly_ratio).sqrt();
        let bezirci_grade = Self::grade_bezirci(bezirci_score);

        LegacyResults {
            atesman: FormulaResult {
                score: (atesman_score * 100.0).round() / 100.0,
                grade_level: atesman_grade.to_string(),
                description: "Ateşman Okunabilirlik Formülü (1997)".to_string(),
            },
            cetinkaya_uzun: FormulaResult {
                score: (cetinkaya_score * 100.0).round() / 100.0,
                grade_level: cetinkaya_grade.to_string(),
                description: "Çetinkaya-Uzun Okunabilirlik Formülü (2010)".to_string(),
            },
            bezirci_yilmaz: FormulaResult {
                score: (bezirci_score * 100.0).round() / 100.0,
                grade_level: bezirci_grade.to_string(),
                description: "Bezirci-Yılmaz Okunabilirlik Formülü (2010)".to_string(),
            },
        }
    }

    fn grade_atesman(score: f64) -> &'static str {
        if score >= 90.0 {
            "Çok Kolay (4. Sınıf ve Altı)"
        } else if score >= 70.0 {
            "Kolay (5 ve 6. Sınıf)"
        } else if score >= 50.0 {
            "Orta Güçlükte (7 ve 8. Sınıf)"
        } else if score >= 30.0 {
            "Zor (9 ve 10. Sınıf)"
        } else {
            "Çok Zor (Lisans ve Lisansüstü)"
        }
    }

    fn grade_cetinkaya(score: f64) -> &'static str {
        if score >= 51.0 {
            "Bağımsız Okuma Düzeyi (8. Sınıf ve Altı)"
        } else if score >= 38.0 {
            "Öğretim Düzeyi (9 ve 12. Sınıf)"
        } else {
            "Endişe / Zorluk Düzeyi (Lisans ve Üstü)"
        }
    }

    fn grade_bezirci(score: f64) -> &'static str {
        if score <= 8.0 {
            "İlkokul Düzeyi"
        } else if score <= 12.0 {
            "Ortaokul Düzeyi"
        } else if score <= 16.0 {
            "Lise Düzeyi"
        } else {
            "Üniversite ve Akademik Düzey"
        }
    }
}
