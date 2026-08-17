//! Multi-factor Turkish AI Style Detector and Diagnostic Pinpointer.

use serde::{Deserialize, Serialize};
use crate::morphology::TurkishMorphology;
use super::cliches::{ClicheMatcher, ClicheCategory};
use super::style_metrics::{StyleMetricsExtractor, StyleMetricsSummary};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiagnosticFinding {
    pub category: String,
    pub severity: String, // "Info", "Warning", "Critical"
    pub message: String,
    pub suggestion: Option<String>,
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleAuditReport {
    pub ai_score: f32, // 0.0 to 100.0
    pub verdict: String,
    pub findings: Vec<DiagnosticFinding>,
    pub metrics: StyleMetricsSummary,
}

pub struct TurkishStyleAuditor {
    morphology: TurkishMorphology,
}

impl Default for TurkishStyleAuditor {
    fn default() -> Self {
        Self::new()
    }
}

impl TurkishStyleAuditor {
    pub fn new() -> Self {
        Self {
            morphology: TurkishMorphology::new(),
        }
    }

    /// Performs comprehensive style audit and AI signature detection on a Turkish text.
    pub fn audit(&self, text: &str) -> StyleAuditReport {
        let punct_metrics = StyleMetricsExtractor::extract_punctuation_metrics(text);
        let rhythm_metrics = StyleMetricsExtractor::extract_rhythm_metrics(text);
        let predicate_metrics = StyleMetricsExtractor::extract_predicate_metrics(text, &self.morphology);
        let cliche_matches = ClicheMatcher::find_all(text);

        let mut findings = Vec::new();
        let mut score: f32 = 0.0;

        // 1. Cliché and Calque Findings
        let mut bureaucratic_count = 0;
        let mut conclusion_count = 0;
        let mut calque_count = 0;

        for m in cliche_matches {
            let (cat_str, sev, penalty) = match m.category {
                ClicheCategory::BureaucraticConnector => {
                    bureaucratic_count += 1;
                    ("BureaucraticConnector", "Warning", 12.0)
                }
                ClicheCategory::ConclusionHedge => {
                    conclusion_count += 1;
                    ("ConclusionHedge", "Critical", 18.0)
                }
                ClicheCategory::RhetoricalCalque => {
                    calque_count += 1;
                    ("RhetoricalCalque", "Critical", 22.0)
                }
                ClicheCategory::SemicolonConjunction => {
                    ("SemicolonConjunction", "Critical", 18.0)
                }
                ClicheCategory::InflatedHedging => {
                    ("InflatedHedging", "Warning", 8.0)
                }
                ClicheCategory::PassiveInflation => {
                    ("PassiveInflation", "Warning", 10.0)
                }
                ClicheCategory::IndefiniteArticleInflation => {
                    ("IndefiniteArticleInflation", "Info", 6.0)
                }
                ClicheCategory::TricolonParallelList => {
                    ("TricolonParallelList", "Warning", 8.0)
                }
                ClicheCategory::HypophoraQuestion => {
                    ("HypophoraQuestion", "Warning", 10.0)
                }
                ClicheCategory::CliticAnomaly => {
                    ("CliticAnomaly", "Warning", 8.0)
                }
                ClicheCategory::HumanAuthenticityMarker => {
                    ("HumanAuthenticityMarker", "Info", -10.0)
                }
                _ => ("Cliché", "Warning", 10.0),
            };

            score += penalty;
            findings.push(DiagnosticFinding {
                category: cat_str.to_string(),
                severity: sev.to_string(),
                message: format!("'{}': {}", m.text, m.explanation),
                suggestion: Some(m.suggestion.to_string()),
                start: m.start,
                end: m.end,
            });
        }

        // 2. Punctuation Findings
        if punct_metrics.em_dash_count > 0 {
            score += punct_metrics.em_dash_count as f32 * 15.0;
            findings.push(DiagnosticFinding {
                category: "PunctuationAnomaly".to_string(),
                severity: "Critical".to_string(),
                message: format!("Metinde {} adet cümle-içi uzun tire (—) tespit edildi. TDK kurallarına ve doğal Türkçeye aykırıdır; İngilizce em-dash çeviri alışkanlığıdır.", punct_metrics.em_dash_count),
                suggestion: Some("Uzun tireleri virgüle çevirin veya cümleyi iki ayrı bağımsız cümleye bölün.".to_string()),
                start: 0,
                end: 0,
            });
        }

        if punct_metrics.semicolon_count >= 2 {
            score += punct_metrics.semicolon_count as f32 * 8.0;
            findings.push(DiagnosticFinding {
                category: "PunctuationAnomaly".to_string(),
                severity: "Warning".to_string(),
                message: format!("Metinde {} adet noktalı virgül (;) tespit edildi. AI genellikle iki bağımsız fikri bağlamak için noktalı virgülü suistimal eder.", punct_metrics.semicolon_count),
                suggestion: Some("Noktalı virgülle ayrılan cümleleri bağımsız noktalı cümlelere dönüştürün.".to_string()),
                start: 0,
                end: 0,
            });
        }

        if punct_metrics.inline_colon_count >= 2 {
            score += punct_metrics.inline_colon_count as f32 * 8.0;
            findings.push(DiagnosticFinding {
                category: "PunctuationAnomaly".to_string(),
                severity: "Warning".to_string(),
                message: format!("Cümle içinde {} adet açıklama amaçlı iki nokta (:) bulundu.", punct_metrics.inline_colon_count),
                suggestion: Some("İki noktaları kaldırıp doğal Türkçe açıklama yan cümlesi kurun.".to_string()),
                start: 0,
                end: 0,
            });
        }

        // 3. Rhythm and Burstiness Findings
        if rhythm_metrics.is_low_burstiness {
            score += 20.0;
            findings.push(DiagnosticFinding {
                category: "RhythmMonotony".to_string(),
                severity: "Critical".to_string(),
                message: format!("Cümle uzunluğu varyansı çok düşük (CV = {:.2}, Ortalama = {:.1} kelime). Cümleler nefessiz ve tekdüze bir uzunlukta takılmıştır.", rhythm_metrics.coefficient_of_variation, rhythm_metrics.mean_sentence_length),
                suggestion: Some("Uzun cümlelerin arasına 1-4 kelimelik kısa, vurgulu nefes cümleleri ekleyin veya bir uzun cümleyi bölün.".to_string()),
                start: 0,
                end: 0,
            });
        }

        if rhythm_metrics.total_sentences >= 3 && rhythm_metrics.short_sentences_count == 0 && rhythm_metrics.mean_sentence_length > 15.0 {
            score += 10.0;
            findings.push(DiagnosticFinding {
                category: "RhythmMonotony".to_string(),
                severity: "Warning".to_string(),
                message: "Metinde hiç kısa (< 5 kelime) cümle bulunmuyor. İnsan yazısı kısa ve uzun cümleleri doğal bir ritimle harmanlar.".to_string(),
                suggestion: Some("Paragraf aralarına kısa cümle patlamaları ekleyin (Örn: 'Durum basitti.', 'Oysa öyle değildi.').".to_string()),
                start: 0,
                end: 0,
            });
        }

        // 4. Predicate Morphology Findings
        if predicate_metrics.mektedir_count >= 2 {
            let ratio_pct = predicate_metrics.mektedir_ratio * 100.0;
            if ratio_pct >= 50.0 {
                score += 25.0;
                findings.push(DiagnosticFinding {
                    category: "PredicateRepetition".to_string(),
                    severity: "Critical".to_string(),
                    message: format!("Yüklemlerin %{:.0}'ı '-mektedir/-maktadır' ile bitiyor ({} adet). Doğal Türkçede bu oran %25'in altındadır.", ratio_pct, predicate_metrics.mektedir_count),
                    suggestion: Some("Yüklemleri geçmiş zaman (-di), geniş zaman (-r) ve şimdiki zaman (-yor) ile çeşitlendirin.".to_string()),
                    start: 0,
                    end: 0,
                });
            } else if ratio_pct >= 25.0 {
                score += 12.0;
                findings.push(DiagnosticFinding {
                    category: "PredicateRepetition".to_string(),
                    severity: "Warning".to_string(),
                    message: format!("Yüklemlerin %{:.0}'ı '-mektedir/-maktadır' eki taşıyor ({} adet).", ratio_pct, predicate_metrics.mektedir_count),
                    suggestion: Some("Tense çeşitliliği sağlayın.".to_string()),
                    start: 0,
                    end: 0,
                });
            }
        }

        let ai_score = score.clamp(0.0, 100.0);
        let verdict = if ai_score >= 75.0 {
            "Ağır AI / Robotik Türkçe İmzalı".to_string()
        } else if ai_score >= 50.0 {
            "Belirgin AI / Makine Dokunuşu".to_string()
        } else if ai_score >= 25.0 {
            "Muhtemel İnsan / Hafif AI Düzenlemesi".to_string()
        } else {
            "Doğal İnsan Metni".to_string()
        };

        let metrics = StyleMetricsSummary {
            punctuation: punct_metrics,
            rhythm: rhythm_metrics,
            predicates: predicate_metrics,
            bureaucratic_connectors_count: bureaucratic_count,
            conclusion_cliches_count: conclusion_count,
            rhetorical_calques_count: calque_count,
        };

        StyleAuditReport {
            ai_score,
            verdict,
            findings,
            metrics,
        }
    }
}
