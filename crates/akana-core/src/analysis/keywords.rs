//! Turkish Keyword and Keyphrase Extraction using Turkish-adapted RAKE and Morphological Stemming.

use crate::morphology::{TurkishStemmer, TurkishStopwords};
use crate::phonology::to_turkish_lower;
use crate::tokenization::TurkishTokenizer;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
            // A number with its suffixes is no word (1923'te is no "te", 3'er no "er"; review 3
            // NORM3-03).
            if clean.is_empty()
                || tok
                    .text
                    .starts_with(|c: char| c.is_ascii_digit() || c == '%')
            {
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
                // As written: the capital tells a name from a common word.
                let stem = self.stemmer.stem(clean);
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

        // 4. Score each candidate phrase, remembering where it first occurs
        let mut phrase_scores: HashMap<String, (f32, usize)> = HashMap::new();
        for (at, phrase) in candidate_phrases.into_iter().enumerate() {
            let phrase_text = phrase.join(" ");
            let score: f32 = phrase
                .iter()
                .map(|w| *word_scores.get(w).unwrap_or(&1.0))
                .sum();

            let entry = phrase_scores.entry(phrase_text).or_insert((0.0, at));
            if score > entry.0 {
                entry.0 = score;
            }
        }

        let mut results: Vec<(KeywordScore, usize)> = phrase_scores
            .into_iter()
            .map(|(keyword, (score, at))| (KeywordScore { keyword, score }, at))
            .collect();

        // Equal scores in the order of the text, so that the same text gives the same keywords
        // (they came in hash order, and top_k cut inside the tie; review 3 NORM3-03).
        results.sort_by(|(a, at_a), (b, at_b)| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(at_a.cmp(at_b))
        });
        let mut results: Vec<KeywordScore> = results.into_iter().map(|(k, _)| k).collect();
        results.truncate(top_k);

        results
    }
}
