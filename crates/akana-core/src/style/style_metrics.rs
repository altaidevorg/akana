//! Style, rhythm, punctuation, and morphological predicate metric extractors.

use serde::{Deserialize, Serialize};
use crate::tokenization::{SentenceSegmenter, TurkishTokenizer};
use crate::morphology::TurkishMorphology;
use crate::phonology::to_turkish_lower;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PunctuationMetrics {
    pub em_dash_count: usize,
    pub semicolon_count: usize,
    pub inline_colon_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RhythmMetrics {
    pub total_sentences: usize,
    pub total_words: usize,
    pub mean_sentence_length: f32,
    pub std_dev_sentence_length: f32,
    pub coefficient_of_variation: f32, // CV = std_dev / mean
    pub short_sentences_count: usize,  // < 5 words
    pub long_sentences_count: usize,   // > 28 words
    pub is_low_burstiness: bool,       // CV < 0.30 and total_sentences >= 3
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredicateMetrics {
    pub total_predicates: usize,
    pub mektedir_count: usize,
    pub mektedir_ratio: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleMetricsSummary {
    pub punctuation: PunctuationMetrics,
    pub rhythm: RhythmMetrics,
    pub predicates: PredicateMetrics,
    pub bureaucratic_connectors_count: usize,
    pub conclusion_cliches_count: usize,
    pub rhetorical_calques_count: usize,
}

pub struct StyleMetricsExtractor;

impl StyleMetricsExtractor {
    /// Extracts punctuation anomaly counts from the text.
    pub fn extract_punctuation_metrics(text: &str) -> PunctuationMetrics {
        let em_dash_count = text.matches('—').count() + text.matches(" -- ").count();
        let semicolon_count = text.matches(';').count();

        let mut inline_colon_count = 0;
        let tokens = TurkishTokenizer::tokenize(text);
        for (i, tok) in tokens.iter().enumerate() {
            if tok.text == ":" && i > 0 && i + 1 < tokens.len() {
                inline_colon_count += 1;
            }
        }

        PunctuationMetrics {
            em_dash_count,
            semicolon_count,
            inline_colon_count,
        }
    }

    /// Extracts sentence rhythm, length variance, and burstiness metrics.
    pub fn extract_rhythm_metrics(text: &str) -> RhythmMetrics {
        let sentences = SentenceSegmenter::segment(text);
        let total_sentences = sentences.len();
        if total_sentences == 0 {
            return RhythmMetrics {
                total_sentences: 0,
                total_words: 0,
                mean_sentence_length: 0.0,
                std_dev_sentence_length: 0.0,
                coefficient_of_variation: 0.0,
                short_sentences_count: 0,
                long_sentences_count: 0,
                is_low_burstiness: false,
            };
        }

        let mut word_counts = Vec::with_capacity(total_sentences);
        let mut short_sentences_count = 0;
        let mut long_sentences_count = 0;
        let mut total_words = 0;

        for s in &sentences {
            let tokens = TurkishTokenizer::tokenize(s.text);
            let count = tokens.into_iter()
                .filter(|t| !matches!(t.token_type, crate::tokenization::TokenType::Punctuation | crate::tokenization::TokenType::Symbol))
                .count();

            total_words += count;
            word_counts.push(count as f32);

            if count < 5 {
                short_sentences_count += 1;
            } else if count > 28 {
                long_sentences_count += 1;
            }
        }

        let mean = total_words as f32 / total_sentences as f32;
        let variance = if total_sentences > 1 {
            let sum_sq_diff: f32 = word_counts.iter().map(|&c| (c - mean).powi(2)).sum();
            sum_sq_diff / (total_sentences - 1) as f32
        } else {
            0.0
        };
        let std_dev = variance.sqrt();
        let cv = if mean > 0.0 { std_dev / mean } else { 0.0 };

        let is_low_burstiness = total_sentences >= 3 && cv < 0.30 && mean >= 16.0;

        RhythmMetrics {
            total_sentences,
            total_words,
            mean_sentence_length: (mean * 10.0).round() / 10.0,
            std_dev_sentence_length: (std_dev * 10.0).round() / 10.0,
            coefficient_of_variation: (cv * 100.0).round() / 100.0,
            short_sentences_count,
            long_sentences_count,
            is_low_burstiness,
        }
    }

    /// Analyzes predicate verb tense variety and counts `-mektedir / -maktadır` occurrences.
    pub fn extract_predicate_metrics(text: &str, morphology: &TurkishMorphology) -> PredicateMetrics {
        let sentences = SentenceSegmenter::segment(text);
        let total_sentences = sentences.len();
        if total_sentences == 0 {
            return PredicateMetrics {
                total_predicates: 0,
                mektedir_count: 0,
                mektedir_ratio: 0.0,
            };
        }

        let mut mektedir_count = 0;
        let mut total_predicates = 0;

        for s in &sentences {
            let tokens = TurkishTokenizer::tokenize(s.text);
            let words: Vec<&str> = tokens.into_iter()
                .filter(|t| matches!(t.token_type, crate::tokenization::TokenType::Word | crate::tokenization::TokenType::ProperNounWithApostrophe))
                .map(|t| t.text)
                .collect();

            if let Some(&last_word) = words.last() {
                let clean = last_word.trim_matches(|c: char| !c.is_alphabetic());
                let lower = to_turkish_lower(clean);

                if lower.ends_with("mektedir") || lower.ends_with("maktadır") ||
                   lower.ends_with("mektedirler") || lower.ends_with("maktadırlar") {
                    mektedir_count += 1;
                    total_predicates += 1;
                } else {
                    let parses = morphology.analyze(&lower);
                    if !parses.is_empty() {
                        total_predicates += 1;
                    }
                }
            }
        }

        let mektedir_ratio = if total_predicates > 0 {
            (mektedir_count as f32 / total_predicates as f32 * 100.0).round() / 100.0
        } else {
            0.0
        };

        PredicateMetrics {
            total_predicates,
            mektedir_count,
            mektedir_ratio,
        }
    }
}
