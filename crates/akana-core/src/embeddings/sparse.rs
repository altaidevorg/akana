//! Morphological BM25 Sparse Vector Encoder.
//!
//! Converts Turkish text into sparse lexical vectors. The text is split into sentences and each
//! sentence is disambiguated, so every word occurrence counts once, as the lemma of the reading
//! its context picks (kanunu -> kanun, bıraktım -> bırakmak), plus the parts of its root when
//! the root is a compound (denizaltı -> deniz + alt). A term's value is
//!
//! - its BM25-saturated occurrence count, `tf * (k1 + 1) / (tf + k1)` with `k1 = 1.2`,
//! - times a part-of-speech weight (proper nouns and numbers > nouns > verbs > adverbs),
//! - times an IDF weight, below 1 only for the function lemmas: those in more than one document
//!   in ten of both a court and a general sample (`function_lemmas.tsv`; bir weighs 0.19).
//!
//! The IDF is gated on purpose. Scaling every term by a static IDF, from court decisions alone
//! or from the two samples together, cost up to 0.04 MRR@10 on RagTurk and Turkuaz: a passage's
//! repeated topic words lost their hybrid slots to words it mentions once, and an inverted index
//! that applies its own corpus IDF counted the IDF twice.

use crate::morphology::{
    CompoundDecomposer, MorphParse, MorphologicalDisambiguator, PrimaryPos, SecondaryPos,
    TurkishMorphology, TurkishStopwords,
};
use crate::phonology::to_turkish_lower;
use crate::tokenization::{SentenceSegmenter, TurkishTokenizer};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::LazyLock;

/// Sparse lexical vector containing feature IDs and corresponding BM25 weights.
///
/// Compatible with standard vector databases (Qdrant, Milvus, Pinecone, etc.).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MorphologicalSparseVector {
    /// 32-bit hashed term IDs ([`hash_term_32bit`]), ascending.
    pub indices: Vec<u32>,
    /// BM25-saturated, POS- and IDF-weighted values.
    pub values: Vec<f32>,
    /// Corresponding string terms (for diagnostics and human inspection).
    pub terms: Vec<String>,
}

/// 32-bit FNV-1a hash of a term: its sparse index.
///
/// Never 0 (an empty slot) or `u32::MAX` (Milvus requires indices below it). Of the 99,472
/// distinct terms of the court sample, 564 share a 24-bit id with another and none a 32-bit one.
#[inline]
pub fn hash_term_32bit(term: &str) -> u32 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in term.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    match (hash ^ (hash >> 32)) as u32 {
        0 => 1,
        u32::MAX => u32::MAX - 1,
        h => h,
    }
}

/// The 24-bit id a term takes in the hybrid vector's 4-byte slots (`hybrid.rs`): its 32-bit
/// id folded, never 0 (an empty slot).
#[inline]
pub fn term_id_24bit(id: u32) -> u32 {
    match (id ^ (id >> 24)) & 0x00FF_FFFF {
        0 => 1,
        h => h,
    }
}

/// BM25 term-frequency saturation constant.
const K1: f32 = 1.2;
/// The weight of a compound's parts (deniz, alt of denizaltı), whatever the word's POS.
const COMPOUND_PART_WEIGHT: f32 = 1.0;

/// A term is a function lemma when more than this fraction of the documents of both samples
/// have it.
pub const FUNCTION_LEMMA_DF: f64 = 0.1;

/// The function-lemma table (`examples/sparse_idf.rs`).
pub(crate) const FUNCTION_LEMMAS_TSV: &str = include_str!("function_lemmas.tsv");

/// IDF weights of the function lemmas by term id, from `function_lemmas.tsv`.
static FUNCTION_LEMMAS: LazyLock<HashMap<u32, f32>> =
    LazyLock::new(|| function_lemma_weights(FUNCTION_LEMMAS_TSV));

/// Reads `function_lemmas.tsv`, written by `examples/sparse_idf.rs`: a `#docs` line with the
/// sizes of the court and the general sample, then `term<TAB>court df<TAB>general df`.
///
/// A term's document frequency `p` is the smaller of its two samples' fractions, so it is common
/// only if common in both: tarih, in 92 % of court decisions and 2 % of the general sample, is
/// no function lemma. A function lemma weighs `ln p / ln FUNCTION_LEMMA_DF`, its IDF `ln (1/p)`
/// relative to the IDF at the threshold: 1 there, 0 for a term in every document.
fn function_lemma_weights(table: &str) -> HashMap<u32, f32> {
    let mut sizes: Option<(f64, f64)> = None;
    let mut weights = HashMap::new();
    for line in table.lines() {
        let mut fields = line.split('\t');
        let (Some(term), Some(court), Some(general)) =
            (fields.next(), fields.next(), fields.next())
        else {
            continue;
        };
        let (Ok(court), Ok(general)) = (court.parse::<f64>(), general.parse::<f64>()) else {
            continue;
        };
        if term == "#docs" {
            sizes = Some((court, general)).filter(|&(c, g)| c > 0.0 && g > 0.0);
            continue;
        }
        let Some((court_docs, general_docs)) = sizes else {
            continue;
        };
        let p = (court / court_docs).min(general / general_docs).min(1.0);
        if p > FUNCTION_LEMMA_DF {
            let weight = p.ln() / FUNCTION_LEMMA_DF.ln();
            weights.insert(hash_term_32bit(term), weight as f32);
        }
    }
    weights
}

/// Turkish morphological sparse encoder.
pub struct MorphologicalSparseEncoder {
    morphology: TurkishMorphology,
    disambiguator: MorphologicalDisambiguator,
    stopwords: TurkishStopwords,
}

impl Default for MorphologicalSparseEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl MorphologicalSparseEncoder {
    /// Create a new morphological sparse encoder with built-in Turkish lexicon.
    pub fn new() -> Self {
        let morphology = TurkishMorphology::new();
        Self {
            disambiguator: MorphologicalDisambiguator::with_morphology(morphology.clone()),
            morphology,
            stopwords: TurkishStopwords::default(),
        }
    }

    /// The text's terms, one per word occurrence and compound part, in text order, each with
    /// the POS weight of the reading it comes from. Stopwords, punctuation and one-character
    /// terms are left out.
    pub fn term_occurrences(&self, text: &str) -> Vec<(String, f32)> {
        let mut out = Vec::new();
        // Compound splits of the roots seen so far: splitting may analyse the root again.
        let mut splits: HashMap<String, Option<(String, String)>> = HashMap::new();
        for sentence in SentenceSegmenter::segment(text) {
            let tokens = TurkishTokenizer::tokenize_words(sentence.text);
            let parses = self.disambiguator.disambiguate(&tokens);
            for (token, parse) in tokens.iter().zip(&parses) {
                let surface = to_turkish_lower(token.trim());
                if !self.is_term(&surface) {
                    continue;
                }
                if parse.primary_pos == PrimaryPos::Unknown {
                    // A word the analyzer does not know is its own term.
                    out.push((surface, pos_weight(parse)));
                    continue;
                }
                let lemma = to_turkish_lower(&parse.lemma);
                if self.is_term(&lemma) {
                    out.push((lemma, pos_weight(parse)));
                }
                let root = to_turkish_lower(&parse.root);
                let parts = splits.entry(root).or_insert_with_key(|root| {
                    CompoundDecomposer::split_lexical_compound(&self.morphology, root)
                });
                if let Some((first, head)) = parts {
                    for part in [first, head] {
                        if self.is_term(part) {
                            out.push((part.clone(), COMPOUND_PART_WEIGHT));
                        }
                    }
                }
            }
        }
        out
    }

    /// Extract disambiguated lemmas, compound parts, and linguistic weights.
    pub fn encode_sparse(&self, text: &str) -> MorphologicalSparseVector {
        // term -> (highest POS weight, occurrences)
        let mut counts: HashMap<String, (f32, u32)> = HashMap::new();
        for (term, weight) in self.term_occurrences(text) {
            let entry = counts.entry(term).or_insert((weight, 0));
            entry.0 = entry.0.max(weight);
            entry.1 += 1;
        }

        let idf = &*FUNCTION_LEMMAS;
        let mut entries: Vec<(u32, f32, String)> = counts
            .into_iter()
            .map(|(term, (pos_w, count))| {
                let tf = count as f32;
                let tf_sat = (tf * (K1 + 1.0)) / (tf + K1);
                let id = hash_term_32bit(&term);
                (id, tf_sat * pos_w * idf.get(&id).unwrap_or(&1.0), term)
            })
            .collect();
        // Sorted by id (and by term on the rare shared id) for a deterministic vector.
        entries.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.2.cmp(&b.2)));

        let mut indices = Vec::with_capacity(entries.len());
        let mut values = Vec::with_capacity(entries.len());
        let mut terms = Vec::with_capacity(entries.len());
        for (id, value, term) in entries {
            indices.push(id);
            values.push(value);
            terms.push(term);
        }
        MorphologicalSparseVector {
            indices,
            values,
            terms,
        }
    }

    /// Whether a lower-cased word or lemma can be a term: two characters or more, not only
    /// punctuation, not a stopword.
    fn is_term(&self, word: &str) -> bool {
        word.chars().nth(1).is_some()
            && word.chars().any(char::is_alphanumeric)
            && !self.stopwords.is_stopword(word)
    }
}

/// How much a word's reading counts: names and numbers most, function words least.
/// The weight of a proper noun's term.
const PROPER_NOUN_WEIGHT: f32 = 1.8;

/// The weight of a term by its reading's POS; [`OTHER_POS_WEIGHT`] for the rest.
const POS_WEIGHTS: &[(PrimaryPos, f32)] = &[
    (PrimaryPos::Num, 1.6),
    (PrimaryPos::Noun, 1.3),
    (PrimaryPos::Verb, 1.0),
    (PrimaryPos::Adj, 1.0),
    (PrimaryPos::Unknown, 0.9),
    (PrimaryPos::Adv, 0.7),
    (PrimaryPos::Pron, 0.2),
    (PrimaryPos::Conj, 0.2),
    (PrimaryPos::Postp, 0.2),
    (PrimaryPos::Q, 0.2),
];

/// The weight of a term whose POS [`POS_WEIGHTS`] does not list.
const OTHER_POS_WEIGHT: f32 = 0.8;

fn pos_weight(parse: &MorphParse) -> f32 {
    if parse.secondary_pos == SecondaryPos::ProperNoun {
        return PROPER_NOUN_WEIGHT;
    }
    POS_WEIGHTS
        .iter()
        .find(|(pos, _)| *pos == parse.primary_pos)
        .map_or(OTHER_POS_WEIGHT, |&(_, w)| w)
}

/// The constants that define terms' values, as text for the encoder fingerprint
/// ([`super::spec::encoder_info`]).
pub(crate) fn fingerprint_constants() -> String {
    let mut s = format!(
        "k1 {K1} compound-part {COMPOUND_PART_WEIGHT} function-df {FUNCTION_LEMMA_DF} \
         proper {PROPER_NOUN_WEIGHT} other {OTHER_POS_WEIGHT}"
    );
    for (pos, w) in POS_WEIGHTS {
        s.push_str(&format!(" {}={w}", pos.as_str()));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn term_values(vec: MorphologicalSparseVector) -> HashMap<String, f32> {
        vec.terms.into_iter().zip(vec.values).collect()
    }

    #[test]
    fn test_hash_term_32bit() {
        let h1 = hash_term_32bit("ankara");
        let h2 = hash_term_32bit("istanbul");
        assert_ne!(h1, h2);
        assert_ne!(h1, 0);
        assert_ne!(h1, u32::MAX);
        // The hybrid slot id is never 0 and fits in 24 bits.
        for id in [0, 1, h1, h2, 0x0100_0001, u32::MAX] {
            let short = term_id_24bit(id);
            assert!(short != 0 && short <= 0x00FF_FFFF, "{id:#x} -> {short:#x}");
        }
    }

    #[test]
    fn test_encode_sparse_lemmatization() {
        let encoder = MorphologicalSparseEncoder::new();
        // "evlerimizde" should be lemmatized to "ev"
        let vec = encoder.encode_sparse("evlerimizde güzel kediler var");
        assert!(!vec.indices.is_empty());
        assert!(
            vec.terms.iter().any(|t| t == "ev"),
            "expected lemma 'ev' in extracted terms, got {:?}",
            vec.terms
        );
        assert!(
            vec.terms.iter().any(|t| t == "kedi"),
            "expected lemma 'kedi' in extracted terms, got {:?}",
            vec.terms
        );
        assert!(vec.indices.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(vec.indices.len(), vec.values.len());
        assert_eq!(vec.indices.len(), vec.terms.len());
    }

    #[test]
    fn term_frequency_counts_occurrences_not_parses() {
        // kanunu is kanun+P3sg or kanun+Acc: one occurrence, one term.
        let encoder = MorphologicalSparseEncoder::new();
        let terms = encoder.term_occurrences("Kanunu okudu.");
        let kanun = terms.iter().filter(|(t, _)| t == "kanun").count();
        assert_eq!(kanun, 1, "{terms:?}");
        let terms = encoder.term_occurrences("Kanunu okudu. Bu kanun yeni bir kanundur.");
        let kanun = terms.iter().filter(|(t, _)| t == "kanun").count();
        assert_eq!(kanun, 3, "{terms:?}");
        // Each word gives its reading's lemma only, not the lemmas of its other readings.
        let terms: Vec<String> = encoder
            .term_occurrences("Yüzü güldü.")
            .into_iter()
            .map(|(t, _)| t)
            .collect();
        assert_eq!(terms, ["yüz", "gülmek"]);
    }

    #[test]
    fn values_saturate_with_occurrences() {
        let encoder = MorphologicalSparseEncoder::new();
        let once = term_values(encoder.encode_sparse("Tapu iptal edildi."))["tapu"];
        let twice =
            term_values(encoder.encode_sparse("Tapu iptal edildi. Tapu yenilendi."))["tapu"];
        let ratio = twice / once;
        // (2 * 2.2 / 3.2) / (2.2 / 2.2) = 1.375
        assert!((ratio - 1.375).abs() < 1e-4, "{once} {twice}");
    }

    #[test]
    fn function_lemmas_weigh_less() {
        let encoder = MorphologicalSparseEncoder::new();
        // bir is in most documents of both samples, kooperatif in 2 % of court decisions.
        let values = term_values(encoder.encode_sparse("Bir kooperatif kuruldu."));
        assert!(values["bir"] < values["kooperatif"] / 3.0, "{values:?}");
        // tarih is in 92 % of court decisions but few general texts: no function lemma.
        let values = term_values(encoder.encode_sparse("Eski tarih kitabı."));
        assert_eq!(values["tarih"], 1.3, "{values:?}");
        assert!(values.values().all(|&v| v > 0.0 && v < 4.0), "{values:?}");
    }

    #[test]
    fn compound_parts_come_from_the_lexicon_not_from_inflected_surfaces() {
        let encoder = MorphologicalSparseEncoder::new();
        let terms = encoder
            .encode_sparse(
                "Bu taşınmaz davacıya aittir ve görevlisi tarafından el konulmuştur; \
                 Danıştay kararına göre edilmelidir.",
            )
            .terms;
        for junk in [
            "dav", "acıya", "gör", "evlisi", "kar", "arına", "tay", "melidir", "muştur",
        ] {
            assert!(!terms.iter().any(|t| t == junk), "{junk} in {terms:?}");
        }
        let terms = encoder
            .encode_sparse("Cumhurbaşkanı zeytinyağı ve denizaltında başbakan")
            .terms;
        for part in [
            "cumhur", "başkan", "zeytin", "yağ", "deniz", "alt", "baş", "bakan",
        ] {
            assert!(terms.iter().any(|t| t == part), "{part} not in {terms:?}");
        }
    }

    #[test]
    fn test_encode_sparse_compound_decomposition() {
        let encoder = MorphologicalSparseEncoder::new();
        let vec = encoder.encode_sparse("denizaltı batığı keşfedildi");
        // "denizaltı" decomposes to "deniz" and "altı"
        assert!(
            vec.terms.iter().any(|t| t == "deniz" || t == "denizaltı"),
            "expected compound components in extracted terms, got {:?}",
            vec.terms
        );
    }

    #[test]
    fn punctuation_stopwords_and_unknown_words() {
        let encoder = MorphologicalSparseEncoder::new();
        let terms = encoder.encode_sparse("Ve bu – xqzvbt , ( ) ...").terms;
        assert_eq!(terms, ["xqzvbt"]);
        assert!(encoder.encode_sparse("").indices.is_empty());
    }
}
