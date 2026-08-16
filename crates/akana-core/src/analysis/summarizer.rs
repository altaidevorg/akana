//! Extractive Turkish Document Summarizer using Morphological Graph-based TextRank.

use std::collections::HashSet;
use crate::tokenization::{SentenceSegmenter, TurkishTokenizer};
use crate::morphology::{TurkishStemmer, TurkishStopwords};
use crate::phonology::to_turkish_lower;

pub struct TurkishSummarizer {
    stemmer: TurkishStemmer,
    stopwords: TurkishStopwords,
}

impl Default for TurkishSummarizer {
    fn default() -> Self {
        Self::new()
    }
}

impl TurkishSummarizer {
    pub fn new() -> Self {
        Self {
            stemmer: TurkishStemmer::new(),
            stopwords: TurkishStopwords::new(),
        }
    }

    /// Extracts the top-k most important sentences from a Turkish text in chronological order.
    pub fn summarize(&self, text: &str, max_sentences: usize) -> Vec<String> {
        let raw_sentences = SentenceSegmenter::segment(text);
        if raw_sentences.is_empty() {
            return Vec::new();
        }
        if raw_sentences.len() <= max_sentences {
            return raw_sentences.into_iter().map(|s| s.text.to_string()).collect();
        }

        // 1. Preprocess and lemmatize sentences into stemmed token bags
        let mut sentence_bags: Vec<HashSet<String>> = Vec::new();
        for sent in &raw_sentences {
            let tokens = TurkishTokenizer::tokenize(sent.text);
            let mut bag = HashSet::new();
            for tok in tokens {
                let clean = tok.text.trim_matches(|c: char| !c.is_alphabetic());
                if clean.is_empty() {
                    continue;
                }
                let lower = to_turkish_lower(clean);
                if !self.stopwords.is_stopword(&lower) {
                    let stem = self.stemmer.stem(&lower);
                    bag.insert(stem);
                }
            }
            sentence_bags.push(bag);
        }

        let n = raw_sentences.len();
        // 2. Build Sentence Similarity Adjacency Matrix
        let mut weights = vec![vec![0.0f32; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let bag_i = &sentence_bags[i];
                let bag_j = &sentence_bags[j];
                if bag_i.is_empty() || bag_j.is_empty() {
                    continue;
                }

                let intersection = bag_i.intersection(bag_j).count();
                if intersection > 0 {
                    // Normalizer: log(len(i)) + log(len(j))
                    let norm = (bag_i.len() as f32).ln() + (bag_j.len() as f32).ln();
                    let sim = if norm > 0.0 {
                        intersection as f32 / norm
                    } else {
                        intersection as f32
                    };
                    weights[i][j] = sim;
                    weights[j][i] = sim;
                }
            }
        }

        // 3. Run PageRank / TextRank (20 iterations, damping factor d = 0.85)
        let d = 0.85f32;
        let mut scores = vec![1.0f32; n];
        for _ in 0..25 {
            let mut next_scores = vec![1.0 - d; n];
            for i in 0..n {
                for j in 0..n {
                    if i == j || weights[j][i] == 0.0 {
                        continue;
                    }
                    let sum_out: f32 = weights[j].iter().sum();
                    if sum_out > 0.0 {
                        next_scores[i] += d * (weights[j][i] / sum_out) * scores[j];
                    }
                }
            }
            scores = next_scores;
        }

        // 4. Select top-k sentences
        let mut ranked_indices: Vec<usize> = (0..n).collect();
        ranked_indices.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
        ranked_indices.truncate(max_sentences);

        // Sort selected indices back to chronological order
        ranked_indices.sort();

        ranked_indices.into_iter()
            .map(|idx| raw_sentences[idx].text.to_string())
            .collect()
    }
}
