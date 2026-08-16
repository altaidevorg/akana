//! Turkish Keyword and Keyphrase Extraction using Turkish-adapted RAKE and Morphological Stemming.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use crate::morphology::{TurkishStopwords, TurkishStemmer};
use crate::tokenization::TurkishTokenizer;
use crate::phonology::to_turkish_lower;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KeywordScore {
    pub keyword: String,
    pub score: f32,
}

pub struct TurkishKeywordExtractor {
    stopwords: TurkishStopwords,
    stemmer: TurkishStemmer,
}

impl Default for TurkishKeywordExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl TurkishKeywordExtractor {
    pub fn new() -> Self {
        Self {
            stopwords: TurkishStopwords::new(),
            stemmer: TurkishStemmer::new(),
        }
    }

    /// Extracts top-k keywords and keyphrases from Turkish text.
    pub fn extract_keywords(&self, text: &str, top_k: usize) -> Vec<KeywordScore> {
        let tokens = TurkishTokenizer::tokenize(text);
        if tokens.is_empty() {
            return Vec::new();
        }

        // 1. Partition tokens into candidate keyphrases separated by stopwords or punctuation
        let mut candidate_phrases: Vec<Vec<String>> = Vec::new();
        let mut current_phrase: Vec<String> = Vec::new();

        for tok in tokens {
            let clean = tok.text.trim_matches(|c: char| !c.is_alphabetic());
            if clean.is_empty() {
                if !current_phrase.is_empty() {
                    candidate_phrases.push(std::mem::take(&mut current_phrase));
                }
                continue;
            }

            let lower = to_turkish_lower(clean);
            if self.stopwords.is_stopword(&lower) {
                if !current_phrase.is_empty() {
                    candidate_phrases.push(std::mem::take(&mut current_phrase));
                }
            } else {
                let stem = self.stemmer.stem(&lower);
                current_phrase.push(stem);
            }
        }
        if !current_phrase.is_empty() {
            candidate_phrases.push(current_phrase);
        }

        if candidate_phrases.is_empty() {
            return Vec::new();
        }

        // 2. Build word frequency and word degree co-occurrence matrix
        let mut word_freq: HashMap<String, usize> = HashMap::new();
        let mut word_degree: HashMap<String, usize> = HashMap::new();

        for phrase in &candidate_phrases {
            let phrase_len = phrase.len();
            for word in phrase {
                *word_freq.entry(word.clone()).or_insert(0) += 1;
                *word_degree.entry(word.clone()).or_insert(0) += phrase_len;
            }
        }

        // 3. Compute individual word scores: deg(w) / freq(w)
        let mut word_scores: HashMap<String, f32> = HashMap::new();
        for (w, freq) in &word_freq {
            let deg = *word_degree.get(w).unwrap_or(&1) as f32;
            word_scores.insert(w.clone(), deg / (*freq as f32));
        }

        // 4. Score each candidate phrase
        let mut phrase_scores: HashMap<String, f32> = HashMap::new();
        for phrase in candidate_phrases {
            let phrase_text = phrase.join(" ");
            let score: f32 = phrase.iter()
                .map(|w| *word_scores.get(w).unwrap_or(&1.0))
                .sum();

            let entry = phrase_scores.entry(phrase_text).or_insert(0.0);
            if score > *entry {
                *entry = score;
            }
        }

        let mut results: Vec<KeywordScore> = phrase_scores.into_iter()
            .map(|(keyword, score)| KeywordScore { keyword, score })
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);

        results
    }
}
