//! Extractive Turkish Document Summarizer using Morphological Graph-based TextRank.

use crate::morphology::{TurkishStemmer, TurkishStopwords};
use crate::phonology::to_turkish_lower;
use crate::tokenization::{SentenceSegmenter, TurkishTokenizer};
use std::collections::{HashMap, HashSet};

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
            return raw_sentences
                .into_iter()
                .map(|s| s.text.to_string())
                .collect();
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
        // 2. Build the sentence similarity graph as adjacency lists: weights[i] holds (j, w_ij)
        // for every j that shares a stem with i, in increasing j. Pairs are found through an
        // inverted index on stems, so memory follows the number of edges instead of n².
        let mut stem_ids: HashMap<&str, usize> = HashMap::new();
        let mut postings: Vec<Vec<usize>> = Vec::new(); // sentences of each stem, in order
        for (i, bag) in sentence_bags.iter().enumerate() {
            for stem in bag {
                let next_id = postings.len();
                let id = *stem_ids.entry(stem.as_str()).or_insert(next_id);
                if id == next_id {
                    postings.push(Vec::new());
                }
                postings[id].push(i);
            }
        }
        let mut weights: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
        let mut shared = vec![0usize; n];
        let mut later: Vec<usize> = Vec::new(); // the j > i sharing a stem with i
        for (i, bag_i) in sentence_bags.iter().enumerate() {
            for stem in bag_i {
                let list = &postings[stem_ids[stem.as_str()]];
                for &j in &list[list.partition_point(|&j| j <= i)..] {
                    if shared[j] == 0 {
                        later.push(j);
                    }
                    shared[j] += 1;
                }
            }
            later.sort_unstable();
            for &j in &later {
                let intersection = std::mem::take(&mut shared[j]);
                let bag_j = &sentence_bags[j];
                // Normalizer: log(len(i)) + log(len(j))
                let norm = (bag_i.len() as f32).ln() + (bag_j.len() as f32).ln();
                let sim = if norm > 0.0 {
                    intersection as f32 / norm
                } else {
                    intersection as f32
                };
                weights[i].push((j, sim));
                weights[j].push((i, sim));
            }
            later.clear();
        }

        // 3. Run PageRank / TextRank (25 iterations, damping factor d = 0.85)
        // The out-weight sums are computed once: recomputing them inside the i × j loop made
        // this O(25·n³), and 1,000 sentences took 20 s. Adding a row's non-zero weights in order
        // gives the same f32 as adding the whole dense row, so the scores are unchanged.
        let sum_out: Vec<f32> = weights
            .iter()
            .map(|row| row.iter().map(|&(_, w)| w).sum())
            .collect();
        let d = 0.85f32;
        let mut scores = vec![1.0f32; n];
        for _ in 0..25 {
            let mut next_scores = vec![1.0 - d; n];
            for (next_score, row) in next_scores.iter_mut().zip(&weights) {
                for &(j, w) in row {
                    if sum_out[j] > 0.0 {
                        *next_score += d * (w / sum_out[j]) * scores[j];
                    }
                }
            }
            // A fixed point: every further iteration would return the same scores.
            if next_scores == scores {
                break;
            }
            scores = next_scores;
        }

        // 4. Select top-k sentences
        let mut ranked_indices: Vec<usize> = (0..n).collect();
        ranked_indices.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
        ranked_indices.truncate(max_sentences);

        // Sort selected indices back to chronological order
        ranked_indices.sort();

        ranked_indices
            .into_iter()
            .map(|idx| raw_sentences[idx].text.to_string())
            .collect()
    }
}
