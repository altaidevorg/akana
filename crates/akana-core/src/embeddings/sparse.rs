//! Morphological BM25 Sparse Vector Encoder.
//!
//! Converts Turkish text into sparse lexical vectors. The text is split into sentences and each
//! sentence is disambiguated, so every word occurrence counts once, as the lemma of the reading
//! its context picks (kanunu -> kanun, bıraktım -> bırakmak), plus the parts of its root when
//! the root is a compound (denizaltı -> deniz + alt). A word the dictionary lists as a
//! postposition or an adverb has a fixed term instead, whatever reading its context picks
//! (tarafından, içinde; bilerek -> bilmek: `frozen_form`). A term's value is
//!
//! - its BM25-saturated occurrence count, `tf * (k1 + 1) / (tf + k1)` with `k1 = 1.2`,
//! - times a part-of-speech weight (proper nouns and numbers > nouns > verbs > adverbs): the
//!   class of the lemma in the lexicon, not of the reading the disambiguator picked, so that a
//!   ranking change (yeni the adverb or the adjective, devam the interjection or the noun) does
//!   not move the vector,
//! - times an IDF weight, below 1 only for the function lemmas: those in more than one
//!   256-token chunk in ten of both a court and a general sample (`function_lemmas.tsv`, frozen
//!   with the sparse spec).
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
use std::collections::{HashMap, HashSet};
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

/// Reads `function_lemmas.tsv`, written by `examples/sparse_idf.rs`: `term<TAB>weight in 64ths`
/// and the term's chunk counts in the two samples (for the record). The weights are stored as the
/// encoder uses them, so nothing is computed from them, and the table changes only with the
/// sparse spec's major version (`#spec`).
fn function_lemma_weights(table: &str) -> HashMap<u32, f32> {
    table
        .lines()
        .filter(|line| !line.starts_with('#'))
        .filter_map(|line| {
            let mut fields = line.split('\t');
            let (term, weight) = (fields.next()?, fields.next()?.parse::<u16>().ok()?);
            Some((hash_term_32bit(term), f32::from(weight) / 64.0))
        })
        .collect()
}

/// The sparse spec major the function-lemma table was made for (its `#spec` line).
#[cfg(test)]
fn function_lemma_spec(table: &str) -> Option<u16> {
    table
        .lines()
        .find_map(|l| l.strip_prefix("#spec\t"))
        .and_then(|v| v.trim().parse().ok())
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
        // The terms of the unknown words seen so far, whose backoff tries every split.
        let mut backoff: HashMap<String, String> = HashMap::new();
        // The fixed terms of the adverbs and postpositions seen so far ([`frozen_form`]).
        let mut frozen: HashMap<String, Option<(String, f32)>> = HashMap::new();
        // The last token of the sentence before, which a suffix written apart may follow.
        let mut before: Option<&str> = None;
        for sentence in SentenceSegmenter::segment(text) {
            let tokens = TurkishTokenizer::tokenize_words(sentence.text);
            let parses = self.disambiguator.disambiguate(&tokens);
            for (i, (token, parse)) in tokens.iter().zip(&parses).enumerate() {
                let surface = to_turkish_lower(token.trim());
                if !self.is_term(&surface) || suffix_fragment(&tokens, i, before) {
                    continue;
                }
                if parse.primary_pos == PrimaryPos::Unknown {
                    // A word the analyzer does not know is its backoff stem (hiperbariğe and
                    // hiperbarik are one term), or else itself, without the period of an
                    // abbreviation the tokenizer keeps (krş. and krş are one term). A name
                    // the lexicon lacks, written with its capital inside a sentence (Larnaca,
                    // Zelenski), keeps every letter: a name's suffixes follow an apostrophe,
                    // and the backoff made junk of it (larnaç; review 3 MORPH3-02).
                    // A name's suffixes after its apostrophe are no part of its term
                    // (Trump'ın, Trump'a: trump).
                    let bare = surface.trim_end_matches('.');
                    let bare = bare
                        .split(['\'', '’', '‘', '`', '´', '′', '＇'])
                        .next()
                        .unwrap_or(bare);
                    let name = i > 0
                        && token.starts_with(char::is_uppercase)
                        && token.chars().any(char::is_lowercase);
                    let term = if name {
                        bare.to_string()
                    } else {
                        backoff
                            .entry(bare.to_string())
                            .or_insert_with_key(|w| {
                                self.morphology.backoff_stem(w).unwrap_or_else(|| w.clone())
                            })
                            .clone()
                    };
                    // A stem that is a function word is no term (itibarende: itibaren; review 3
                    // MSTAB3-07).
                    if self.is_lemma_term(&term) {
                        out.push((term, pos_weight(parse)));
                    }
                    continue;
                }
                let fixed = frozen
                    .entry(surface.clone())
                    .or_insert_with_key(|w| frozen_form(&self.morphology, w));
                if let Some((term, weight)) = fixed {
                    if self.is_lemma_term(term) {
                        out.push((term.clone(), *weight));
                    }
                    continue;
                }
                let lemma = to_turkish_lower(&parse.lemma);
                if self.is_lemma_term(&lemma) {
                    let weight = lemma_weight(&lemma, parse);
                    out.push((lemma, weight));
                }
                let root = to_turkish_lower(&parse.root);
                let parts = splits.entry(root).or_insert_with_key(|root| {
                    CompoundDecomposer::split_lexical_compound(&self.morphology, root)
                });
                if let Some((first, head)) = parts {
                    for part in [first, head] {
                        if self.is_lemma_term(part) {
                            out.push((part.clone(), COMPOUND_PART_WEIGHT));
                        }
                    }
                }
            }
            before = tokens.last().copied().or(before);
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

    /// Whether a lemma can be a term: [`is_term`](Self::is_term), and not a function word
    /// ([`STOP_LEMMAS`]), whichever of its readings the context picks.
    fn is_lemma_term(&self, lemma: &str) -> bool {
        self.is_term(lemma) && !STOP_LEMMAS.contains(lemma)
    }
}

/// Suffixes that, written apart from their word, are no word of their own.
const SUFFIX_FRAGMENTS: &[&str] = &[
    "da", "de", "ta", "te", "dan", "den", "tan", "ten", "nda", "nde", "ndan", "nden", "ın", "in",
    "un", "ün", "nın", "nin", "nun", "nün", "ya", "ye", "na", "ne", "yı", "yi", "yu", "yü", "nı",
    "ni", "nu", "nü", "la", "le", "yla", "yle", "lı", "li", "lu", "lü", "lık", "lik", "luk", "lük",
    "lar", "ler", "dır", "dir", "dur", "dür", "tır", "tir", "tur", "tür", "ki", "daki", "deki",
    "taki", "teki",
];

/// Whether the token at `i` is a suffix written apart, which is no term: after an apostrophe
/// (X ' nin, ... 'nin), a closing quote right before it (Sözleşmesi"nin), the period of an
/// abbreviation or initials in capitals (TCK. nun, A.Ş. nin), or an acronym in capitals
/// (TTK nın). `before` is the last token of the sentence before, for a sentence that starts
/// with the suffix (review 3 MSTAB3-05).
fn suffix_fragment(tokens: &[&str], i: usize, before: Option<&str>) -> bool {
    let previous = match i.checked_sub(1) {
        Some(p) => tokens.get(p).copied(),
        None => before,
    };
    let Some(previous) = previous else {
        return false;
    };
    if !SUFFIX_FRAGMENTS.contains(&to_turkish_lower(tokens[i]).as_str()) {
        return false;
    }
    // Both are slices of the same text: the quote touches the suffix.
    let adjacent = previous.as_ptr() as usize + previous.len() == tokens[i].as_ptr() as usize;
    let after_apostrophe = matches!(previous, "'" | "’" | "‘" | "`" | "´" | "′" | "＇");
    let after_quote = adjacent && matches!(previous, "\"" | "”" | "“" | "»");
    let after_abbreviation = previous.strip_suffix('.').is_some_and(|a| {
        !a.is_empty()
            && a.chars().any(char::is_uppercase)
            && a.chars().all(|c| c.is_uppercase() || c == '.')
    });
    // In lower case only: in text written in capitals the word after one is a word (GIDA UN
    // SANAYİ).
    let after_acronym = previous.chars().nth(1).is_some()
        && previous.chars().all(char::is_uppercase)
        && !tokens[i].chars().any(char::is_uppercase);
    after_apostrophe || after_quote || after_abbreviation || after_acronym
}

/// How much a word's reading counts: names and numbers most, function words least.
/// The weight of a proper noun's term.
const PROPER_NOUN_WEIGHT: f32 = 1.8;

/// The weight of a term by its reading's POS; [`OTHER_POS_WEIGHT`] for the rest.
///
/// Numbers weigh 1.6, which gives them 11.8 % of court hybrid slots (terms with a digit). Plan
/// step G5 (review 2 MSTAB-08) measured 1.3, 1.0 and a cap of two number slots against it:
/// each lowered exact-search hybrid MRR@10 on RagTurk (0.7900 at 1.6 against 0.7877, 0.7807 and
/// 0.7870); on Turkuaz 1.3 and the cap were lower and 1.0 within noise above (0.4400 against
/// 0.4384, 0.4392 and 0.4404). 1.6 is kept and frozen with spec 1.
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
    class_weight(parse.primary_pos, parse.secondary_pos)
}

/// The weight of a POS class: proper nouns by [`PROPER_NOUN_WEIGHT`], the rest by
/// [`POS_WEIGHTS`].
fn class_weight(pos: PrimaryPos, secondary: SecondaryPos) -> f32 {
    if secondary == SecondaryPos::ProperNoun {
        return PROPER_NOUN_WEIGHT;
    }
    POS_WEIGHTS
        .iter()
        .find(|(p, _)| *p == pos)
        .map_or(OTHER_POS_WEIGHT, |&(_, w)| w)
}

/// When the lexicon's items for one lemma fall in different classes, the class its term
/// weighs as: the first of these that one of them has. Common items come before proper nouns
/// (kanun the noun, not a surname Kanun), content classes before function classes, and the
/// interjection last (devam, ileri, ay).
const CLASS_PRECEDENCE: &[PrimaryPos] = &[
    PrimaryPos::Noun,
    PrimaryPos::Adj,
    PrimaryPos::Verb,
    PrimaryPos::Num,
    PrimaryPos::Adv,
    PrimaryPos::Det,
    PrimaryPos::Pron,
    PrimaryPos::Postp,
    PrimaryPos::Conj,
    PrimaryPos::Q,
    PrimaryPos::Interj,
];

/// `lemma_weights.tsv`: [`LEMMA_WEIGHTS`] as committed.
pub(crate) const LEMMA_WEIGHTS_TSV: &str = include_str!("lemma_weights.tsv");
/// `stop_lemmas.txt`: [`STOP_LEMMAS`] as committed.
pub(crate) const STOP_LEMMAS_TXT: &str = include_str!("stop_lemmas.txt");

/// The weight of each lemma whose lexicon items fall in more than one weight class (yeni Adj and
/// Adv, devam Noun and Interj, Kanun Noun and Prop), by [`CLASS_PRECEDENCE`]. A lemma not in it
/// has items of one class, which its reading's class gives. Read from `lemma_weights.tsv`, which
/// a test derives from the lexicon (`committed_tables_are_what_the_lexicon_gives`): a lexicon
/// edit does not move a weight unless the table is rewritten on purpose (review 3 MSTAB3-08).
static LEMMA_WEIGHTS: LazyLock<HashMap<String, f32>> = LazyLock::new(|| {
    LEMMA_WEIGHTS_TSV
        .lines()
        .filter(|l| !l.starts_with('#'))
        .filter_map(|l| {
            let (lemma, weight) = l.split_once('\t')?;
            Some((lemma.to_string(), weight.parse().ok()?))
        })
        .collect()
});

/// [`LEMMA_WEIGHTS`] as the built-in lexicon gives it.
#[cfg(test)]
fn derive_lemma_weights() -> HashMap<String, f32> {
    let mut classes: HashMap<String, Vec<(PrimaryPos, SecondaryPos)>> = HashMap::new();
    let morphology = TurkishMorphology::new();
    for item in morphology.lexicon().items() {
        let entry = classes.entry(to_turkish_lower(&item.lemma)).or_default();
        if !entry.contains(&(item.primary_pos, item.secondary_pos)) {
            entry.push((item.primary_pos, item.secondary_pos));
        }
    }
    classes
        .into_iter()
        .filter_map(|(lemma, items)| {
            let weights: Vec<f32> = items.iter().map(|&(p, s)| class_weight(p, s)).collect();
            if weights.iter().all(|&w| w == weights[0]) {
                return None;
            }
            let common: Vec<PrimaryPos> = items
                .iter()
                .filter(|(_, s)| *s != SecondaryPos::ProperNoun)
                .map(|&(p, _)| p)
                .collect();
            let weight = if common.is_empty() {
                PROPER_NOUN_WEIGHT
            } else {
                CLASS_PRECEDENCE
                    .iter()
                    .find(|p| common.contains(p))
                    .map_or(OTHER_POS_WEIGHT, |&p| class_weight(p, SecondaryPos::None))
            };
            Some((lemma, weight))
        })
        .collect()
}

/// Classes of function words: a lemma whose dictionary items are all in them is no term (ve,
/// için, bu, kendi, mi).
const FUNCTION_CLASSES: &[PrimaryPos] = &[
    PrimaryPos::Pron,
    PrimaryPos::Conj,
    PrimaryPos::Postp,
    PrimaryPos::Q,
    PrimaryPos::Det,
];

/// Function words whose items include a content class, which are no term either: the article
/// and numeral bir, the negative değil, and the adverbs and postpositions of court prose
/// (review 2 RANK2-11, LEX-20).
const LISTED_STOP_LEMMAS: &[&str] = &[
    "bir",
    "değil",
    "diye",
    "kendi",
    "önce",
    "sonra",
    "tarafından",
    "üzere",
];

/// The lemmas that are no term ([`FUNCTION_CLASSES`], [`LISTED_STOP_LEMMAS`]): decided
/// by the lemma, so a word's term does not come and go with the reading its context picks
/// (tarafından the adverb or taraf-ı-ndan the noun). Read from `stop_lemmas.txt`, which the
/// same test derives from the lexicon (review 3 MSTAB3-08).
static STOP_LEMMAS: LazyLock<HashSet<String>> = LazyLock::new(|| {
    STOP_LEMMAS_TXT
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(str::to_string)
        .collect()
});

/// [`STOP_LEMMAS`] as the built-in lexicon gives it.
#[cfg(test)]
fn derive_stop_lemmas() -> HashSet<String> {
    let mut classes: HashMap<String, bool> = HashMap::new();
    let morphology = TurkishMorphology::new();
    for item in morphology.lexicon().items() {
        let function = FUNCTION_CLASSES.contains(&item.primary_pos);
        *classes.entry(to_turkish_lower(&item.lemma)).or_insert(true) &= function;
    }
    classes
        .into_iter()
        .filter_map(|(lemma, function)| function.then_some(lemma))
        .chain(LISTED_STOP_LEMMAS.iter().map(|l| l.to_string()))
        .collect()
}

/// The term of a word the dictionary lists as an adverb or a postposition, written as that word,
/// whichever reading its context picks: the readings are in every such word, and their order
/// turned with every re-estimation of the prior, and the term with it (review 2 RANK2-04).
/// - A postposition is itself (göre, kadar, üzere), never the verb some spell (gör-e).
/// - An adverb that is also a verb's converb or participle is that verb (bil-erek
///   "knowingly", anlaş-ıl-an "apparently", which court decisions write as the participle):
///   the best such reading's lemma, as the verb's other forms are indexed. A finite form the
///   adverb happens to spell (kes-in "cut!", yerin-e, bil-e) does not count.
/// - Any other adverb is itself, over the noun's case form it spells (tarafından, içinde,
///   yerine, yanında, zamanında, başta for taraf-ı-ndan, iç-i-nde, yer-i-ne ...).
///
/// Each with its lexicon class's weight.
fn frozen_form(morphology: &TurkishMorphology, surface: &str) -> Option<(String, f32)> {
    let folded: String = surface
        .chars()
        .map(|c| match c {
            'â' => 'a',
            'î' => 'i',
            'û' => 'u',
            c => c,
        })
        .collect();
    let items = morphology.root_items(&folded);
    let item = items
        .iter()
        .find(|item| item.primary_pos == PrimaryPos::Postp)
        .or_else(|| {
            items
                .iter()
                .find(|item| item.primary_pos == PrimaryPos::Adv)
        })?;
    let weigh = |term: &str, pos, secondary| {
        LEMMA_WEIGHTS
            .get(term)
            .copied()
            .unwrap_or_else(|| class_weight(pos, secondary))
    };
    if item.primary_pos == PrimaryPos::Adv {
        if let Some(verb) = morphology
            .analyze(surface)
            .into_iter()
            .find(|p| p.primary_pos == PrimaryPos::Verb && p.pos != PrimaryPos::Verb)
        {
            let term = to_turkish_lower(&verb.lemma);
            let weight = weigh(&term, verb.primary_pos, verb.secondary_pos);
            return Some((term, weight));
        }
    }
    let term = to_turkish_lower(&item.lemma);
    let weight = weigh(&term, item.primary_pos, item.secondary_pos);
    Some((term, weight))
}

/// The POS weight of a lemma's term: its lexicon class when its items differ in class
/// ([`LEMMA_WEIGHTS`]), else the reading's.
fn lemma_weight(lemma: &str, parse: &MorphParse) -> f32 {
    LEMMA_WEIGHTS
        .get(lemma)
        .copied()
        .unwrap_or_else(|| pos_weight(parse))
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
    s.push_str(" precedence");
    for pos in CLASS_PRECEDENCE {
        s.push_str(&format!(" {}", pos.as_str()));
    }
    s.push_str(" function-classes");
    for pos in FUNCTION_CLASSES {
        s.push_str(&format!(" {}", pos.as_str()));
    }
    s.push_str(" function-lemmas");
    for lemma in LISTED_STOP_LEMMAS {
        s.push_str(&format!(" {lemma}"));
    }
    s.push_str(" fragments");
    for fragment in SUFFIX_FRAGMENTS {
        s.push_str(&format!(" {fragment}"));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Review 3 MSTAB3-08: the committed tables are what the lexicon gives. A lexicon edit that
    /// moves a stop lemma or a weight fails here; such a change moves vectors and follows the
    /// bump policy (`spec.rs`). `AKANA_GOLD_WRITE=sparse_tables` rewrites both files.
    #[test]
    fn committed_tables_are_what_the_lexicon_gives() {
        let mut stop: Vec<String> = derive_stop_lemmas().into_iter().collect();
        stop.sort();
        let mut weights: Vec<(String, f32)> = derive_lemma_weights().into_iter().collect();
        weights.sort_by(|a, b| a.0.cmp(&b.0));
        let stop_txt = format!(
            "# The lemmas that are no sparse term (embeddings::sparse::STOP_LEMMAS), derived from \
             the built-in lexicon; AKANA_GOLD_WRITE=sparse_tables rewrites this file.\n{}\n",
            stop.join("\n")
        );
        let weights_tsv = format!(
            "# Lemmas whose lexicon items differ in weight class, and their weight \
             (embeddings::sparse::LEMMA_WEIGHTS), derived from the built-in lexicon; \
             AKANA_GOLD_WRITE=sparse_tables rewrites this file.\n{}\n",
            weights
                .iter()
                .map(|(l, w)| format!("{l}\t{w}"))
                .collect::<Vec<_>>()
                .join("\n")
        );
        if std::env::var("AKANA_GOLD_WRITE")
            .is_ok_and(|v| v.split(',').any(|s| s.trim() == "sparse_tables"))
        {
            let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/src/embeddings/");
            std::fs::write(format!("{dir}stop_lemmas.txt"), &stop_txt).unwrap();
            std::fs::write(format!("{dir}lemma_weights.tsv"), &weights_tsv).unwrap();
            return;
        }
        assert!(
            stop_txt == STOP_LEMMAS_TXT && weights_tsv == LEMMA_WEIGHTS_TSV,
            "the lexicon gives other stop lemmas or lemma weights than the committed tables: \
             measure the vector change (scripts/encoder_churn.py), then rewrite them with \
             AKANA_GOLD_WRITE=sparse_tables cargo test -p akana-core --lib committed_tables"
        );
        assert_eq!(STOP_LEMMAS.len(), stop.len());
        assert_eq!(LEMMA_WEIGHTS.len(), weights.len());
    }

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
    fn a_lemma_weighs_by_its_lexicon_class_whatever_reading_is_chosen() {
        // yazık is a noun, an adverb and an interjection, yeni an adjective and an adverb, kanun
        // a noun and a (corpus) proper noun: each weighs by its first class, whichever reading
        // the disambiguator takes.
        assert_eq!(LEMMA_WEIGHTS.get("yazık"), Some(&1.3));
        assert_eq!(LEMMA_WEIGHTS.get("yeni"), Some(&1.0));
        assert_eq!(LEMMA_WEIGHTS.get("kanun"), Some(&1.3));
        // A lemma whose items share a class is left to the reading (bırakmak: one verb).
        assert_eq!(LEMMA_WEIGHTS.get("bırakmak"), None);
        let encoder = MorphologicalSparseEncoder::new();
        let alone = term_values(encoder.encode_sparse("Yazık."));
        let in_clause = term_values(encoder.encode_sparse("Emekler çok yazık oldu."));
        assert_eq!(alone["yazık"], in_clause["yazık"]);
    }

    #[test]
    fn function_lemma_table_belongs_to_this_spec() {
        // The table is frozen per spec major: a new one goes with a new major.
        assert_eq!(
            function_lemma_spec(FUNCTION_LEMMAS_TSV),
            Some(crate::embeddings::SPARSE_SPEC.0)
        );
        assert!(!FUNCTION_LEMMAS.is_empty());
        assert!(FUNCTION_LEMMAS.values().all(|&w| w > 0.0 && w <= 1.0));
    }

    #[test]
    fn function_lemmas_weigh_less() {
        let encoder = MorphologicalSparseEncoder::new();
        // almak is in most documents of both samples, kooperatif in 2 % of court decisions.
        let values = term_values(encoder.encode_sparse("Kooperatif karar aldı."));
        assert!(values["almak"] < values["kooperatif"] / 2.0, "{values:?}");
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

    /// Review 2 RANK2-11, LEX-20: function words are no term, decided by the lemma.
    #[test]
    fn function_lemmas_are_no_terms() {
        for lemma in [
            "ve",
            "için",
            "bu",
            "kendi",
            "gibi",
            "göre",
            "bir",
            "değil",
            "tarafından",
            "sonra",
            "üzere",
        ] {
            assert!(STOP_LEMMAS.contains(lemma), "{lemma}");
        }
        // A lemma with a content reading is no stop lemma (çok, karşı and kadar are still no
        // term: the surface stopword list drops them; review 3 MSTAB3-07b).
        for lemma in ["çok", "karar", "devam", "mahkeme", "karşı", "kadar"] {
            assert!(!STOP_LEMMAS.contains(lemma), "{lemma}");
        }
        let encoder = MorphologicalSparseEncoder::new();
        let terms = encoder
            .encode_sparse("Dosya bir kez daha mahkeme tarafından incelendi ve karar verildi.")
            .terms;
        for absent in ["bir", "tarafından", "ve"] {
            assert!(!terms.iter().any(|t| t == absent), "{absent} in {terms:?}");
        }
        for present in ["dosya", "mahkeme", "karar"] {
            assert!(
                terms.iter().any(|t| t == present),
                "{present} not in {terms:?}"
            );
        }
    }

    #[test]
    fn punctuation_stopwords_and_unknown_words() {
        let encoder = MorphologicalSparseEncoder::new();
        let terms = encoder.encode_sparse("Ve bu – xqzvbt , ( ) ...").terms;
        assert_eq!(terms, ["xqzvbt"]);
        assert!(encoder.encode_sparse("").indices.is_empty());
    }

    #[test]
    fn abbreviations_have_one_term_and_suffix_fragments_none() {
        let encoder = MorphologicalSparseEncoder::new();
        let terms = |text| {
            let mut t = encoder.encode_sparse(text).terms;
            t.sort();
            t
        };
        // An abbreviation with or without its period is one term; so is a number with its
        // suffixes (1924'te) and an abbreviation with them after a period (TCK.nun).
        assert_eq!(terms("Av. Av"), ["av"]);
        assert_eq!(terms("1924'te 1924"), ["1924"]);
        assert_eq!(terms("TCK.nun TCK'nun"), ["tck"]);
        assert_eq!(terms("krş. krş"), ["krş"]);
        // Suffixes written apart after an apostrophe or an abbreviation's period are no terms.
        assert_eq!(terms("Ankara ' nın"), ["ankara"]);
        assert_eq!(terms("TCK. nun"), ["tck"]);
        // Review 3: after a name left out (...'nin), a quote, initials and an acronym.
        assert_eq!(terms("Davacı ...'nin vekili"), ["davacı", "vekil"]);
        assert_eq!(terms("Sözleşmesi\"nin"), ["sözleşme"]);
        assert_eq!(terms("TTK nın"), ["ttk"]);
        assert_eq!(terms("GIDA UN SANAYİ"), ["gıda", "sanayi", "un"]);
    }

    #[test]
    fn an_unknown_word_is_its_backoff_stem() {
        // Plan step D6: an inflected unknown word and its bare form are one term.
        let encoder = MorphologicalSparseEncoder::new();
        let terms = |text| encoder.encode_sparse(text).terms;
        assert_eq!(
            terms("hiperbarik hiperbariğe hiperbarikten"),
            ["hiperbarik"]
        );
        // A word with no inflection to take off is its own term.
        assert_eq!(terms("xqzvbt"), ["xqzvbt"]);
        // Review 3: a vowel-final word and its inflected forms are one term; a name the lexicon
        // lacks keeps its letters inside a sentence; a function word is no term.
        assert_eq!(terms("zorbanta zorbantanın zorbantalar"), ["zorbanta"]);
        assert_eq!(terms("Biz Larnaca gördük"), ["larnaca", "görmek"]);
        assert_eq!(terms("Biz Zxqwert'ın ve Zxqwert'a"), ["zxqwert"]);
        assert_eq!(terms("itibarende"), Vec::<String>::new());
    }
}
