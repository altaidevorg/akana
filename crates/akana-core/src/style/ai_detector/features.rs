//! Content features for the Turkish AI-text detector.
//!
//! [`content_features`] maps **already preprocessed** text (the output of
//! [`super::preprocess()`]: markup stripped, NFKC, chatbot opener dropped, masks such as
//! `<YIL>`, `<SAYI>`, `<ÖZEL>`, `<URL>`, `<EPOSTA>`) to a fixed-length vector in the order
//! of [`FEATURE_NAMES`]. The definitions are frozen to the detector's training code.
//!
//! Design rules:
//! - No formatting features: markup, emoji and typography are stripped before.
//! - Every feature is length-normalised (per 100 words, per sentence, per 100 sentences,
//!   per paragraph or a share), except `log_words`, which is the length itself.
//! - Mask tokens (and their apostrophe suffix, `<ÖZEL>'da`) are excluded from every
//!   lexical statistic and from the word count used as the per-100-words denominator. They
//!   only count as structural tokens for sentence and paragraph length.
//! - The morphology analyzer is **not** used (it overgenerates on this task). Suffix
//!   features are surface heuristics.
//! - Degenerate input (empty, one word, emoji only, masks only) yields 0 instead of NaN or
//!   infinity. The output is deterministic: no hash-map iteration order reaches a float.
//!
//! Words are maximal runs of letters (combining marks included) in the Turkish-lower-cased
//! text; an apostrophe suffix after a word is skipped, digits and punctuation separate
//! words. Sentences come from [`SentenceSegmenter`] run on each line separately, so a
//! heading or list item without final punctuation is its own sentence; paragraphs are
//! separated by a blank line (`\n\n`, as produced by [`super::preprocess::normalize`]).

use ahash::AHashMap;

use super::lexicons::{
    function_word_index, ClicheGate, AI_LEXICON, FUNCTION_SET, HUMAN_LEXICON, MASKS,
    N_FUNCTION_WORD_FEATURES,
};
use super::markup::is_letter;
use super::morph::{
    is_dik, is_known, is_makta, is_mis, is_passive, label_colon_counts, lex_hit, predicate_tense,
    Chars, Tense,
};
use crate::phonology::alphabet::{is_turkish_vowel, to_turkish_lower};
use crate::readability::legacy::atesman_score;
use crate::tokenization::SentenceSegmenter;

/// Number of content features.
pub(super) const N_FEATURES: usize = 68;

/// Features before the function-word profile (including `aorist_ratio`).
const N_BASE_FEATURES: usize = 25;
/// Stylometry features after the function-word profile.
const N_STYLE_TAIL_FEATURES: usize = 11;
const _: () =
    assert!(N_BASE_FEATURES + N_FUNCTION_WORD_FEATURES + N_STYLE_TAIL_FEATURES == N_FEATURES);

/// Feature names, in the order of [`ContentFeatures::to_array`].
///
/// | name | definition |
/// |---|---|
/// | `commas_per_sentence` | `,` count / sentences |
/// | `semicolons_per_100w` | `;` count per 100 words |
/// | `questions_per_100s` | sentences containing `?`, per 100 sentences |
/// | `exclamations_per_100s` | sentences containing `!`, per 100 sentences |
/// | `colons_per_100w` | `:` followed by whitespace or end of text, per 100 words, **without** colons that end a line- or sentence-leading label (`Adım 1:`, `Örnek:`) (`label_colon_counts`) |
/// | `sent_len_mean` | mean sentence length in tokens (words + masks) |
/// | `sent_len_cv` | population std / mean of sentence length (0 with < 2 sentences) |
/// | `long_sent_share` | share of sentences longer than 30 tokens |
/// | `mattr_50` | moving-average type/token ratio, window 50 (plain TTR below 50 words) |
/// | `hapax_ratio` | types occurring once / types |
/// | `yules_k` | Yule's K = 10⁴ · (Σ f² − N) / N² over word frequencies f |
/// | `mean_word_len` | mean word length in characters |
/// | `lexical_density` | share of words not in [`super::lexicons::FUNCTION_WORDS`] |
/// | `ai_lex_per_100w` | [`AI_LEXICON`] hits per 100 words |
/// | `human_lex_per_100w` | [`HUMAN_LEXICON`] hits per 100 words |
/// | `ve_per_100w` | `ve` per 100 words |
/// | `ile_per_100w` | `ile` per 100 words |
/// | `cliche_per_100w` | cliché matches ([`super::cliches`]: `ConclusionHedge`, `RhetoricalCalque`, `IndefiniteArticleInflation` of the style catalogue at training time) per 100 words |
/// | `mis_per_100w` | words with `-mIş` (+ person/copula ending) per 100 words |
/// | `makta_per_100w` | words with `-mAktA(dIr)` per 100 words |
/// | `dik_per_100w` | words with a `-DIk` participle (`-DIğ`+vowel, `-DIk`+`çA/tAn/lAr`) per 100 words |
/// | `passive_per_100w` | words with a passive `-(I)l-` / `-(I)n-` + verbal suffix per 100 words |
/// | `oov_rate` | share of word tokens not explained by the Zemberek root lexicon + surface suffix chain |
/// | `log_words` | ln(1 + words) |
/// | `aorist_ratio` | among sentences whose last word is a present-tense predicate, the share in the aorist (`-Ir/-Ar/-r`, `-mAz`, + person) rather than the progressive `-Iyor`; 0 without such a sentence (`predicate_tense`) |
/// | `fw_<word>` (32) | [`super::lexicons::FUNCTION_WORD_FEATURES`] hits per 100 words, one feature per closed-class word |
/// | `lower_start_share` | share of sentences whose first token is a word starting with a lower-case letter |
/// | `short_word_share` | share of words with at most 3 letters |
/// | `long_word_share` | share of words with at least 10 letters |
/// | `multi_excl` | runs of 2+ `!` (no `?`), per 100 sentences |
/// | `interrobang` | runs of `!`/`?` that mix both (`?!`, `!?`, `?!!`), per 100 sentences |
/// | `ellipsis` | runs of 3+ `.` (or `…`), per 100 sentences |
/// | `multi_q` | runs of 2+ `?` (no `!`), per 100 sentences |
/// | `word_entropy` | Shannon entropy of the word distribution / ln(words) (0 below 2 words) |
/// | `heaps_exponent` | least-squares slope of ln(types seen) on ln(words read), over every prefix of the text (0 below 10 words) |
/// | `compression_ratio` | DEFLATE-compressed size / raw size of the lower-cased text (0 for empty text) |
/// | `atesman_readability` | Ateşman (1997) score 198.825 − 40.175 · syllables/word − 2.610 · words/sentence; syllables = vowels (at least 1 per word); 0 without words or sentences |
pub(super) const FEATURE_NAMES: &[&str] = &[
    "commas_per_sentence",
    "semicolons_per_100w",
    "questions_per_100s",
    "exclamations_per_100s",
    "colons_per_100w",
    "sent_len_mean",
    "sent_len_cv",
    "long_sent_share",
    "mattr_50",
    "hapax_ratio",
    "yules_k",
    "mean_word_len",
    "lexical_density",
    "ai_lex_per_100w",
    "human_lex_per_100w",
    "ve_per_100w",
    "ile_per_100w",
    "cliche_per_100w",
    "mis_per_100w",
    "makta_per_100w",
    "dik_per_100w",
    "passive_per_100w",
    "oov_rate",
    "log_words",
    "aorist_ratio",
    "fw_icin",
    "fw_gore",
    "fw_gibi",
    "fw_kadar",
    "fw_ama",
    "fw_fakat",
    "fw_cunku",
    "fw_ancak",
    "fw_ki",
    "fw_ise",
    "fw_ya",
    "fw_veya",
    "fw_hem",
    "fw_ne",
    "fw_degil",
    "fw_var",
    "fw_yok",
    "fw_bu",
    "fw_su",
    "fw_o",
    "fw_bir",
    "fw_her",
    "fw_daha",
    "fw_en",
    "fw_cok",
    "fw_ben",
    "fw_biz",
    "fw_sen",
    "fw_siz",
    "fw_kendi",
    "fw_de",
    "fw_mi",
    "lower_start_share",
    "short_word_share",
    "long_word_share",
    "multi_excl",
    "interrobang",
    "ellipsis",
    "multi_q",
    "word_entropy",
    "heaps_exponent",
    "compression_ratio",
    "atesman_readability",
];

/// Content features of one document; field order equals [`FEATURE_NAMES`].
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub(super) struct ContentFeatures {
    pub commas_per_sentence: f64,
    pub semicolons_per_100w: f64,
    pub questions_per_100s: f64,
    pub exclamations_per_100s: f64,
    pub colons_per_100w: f64,
    pub sent_len_mean: f64,
    pub sent_len_cv: f64,
    pub long_sent_share: f64,
    pub mattr_50: f64,
    pub hapax_ratio: f64,
    pub yules_k: f64,
    pub mean_word_len: f64,
    pub lexical_density: f64,
    pub ai_lex_per_100w: f64,
    pub human_lex_per_100w: f64,
    pub ve_per_100w: f64,
    pub ile_per_100w: f64,
    pub cliche_per_100w: f64,
    pub mis_per_100w: f64,
    pub makta_per_100w: f64,
    pub dik_per_100w: f64,
    pub passive_per_100w: f64,
    pub oov_rate: f64,
    pub log_words: f64,
    pub aorist_ratio: f64,
    /// `fw_*` values in [`super::lexicons::FUNCTION_WORD_FEATURES`] order, per 100 words.
    pub function_words_per_100w: [f64; N_FUNCTION_WORD_FEATURES],
    pub lower_start_share: f64,
    pub short_word_share: f64,
    pub long_word_share: f64,
    pub multi_excl: f64,
    pub interrobang: f64,
    pub ellipsis: f64,
    pub multi_q: f64,
    pub word_entropy: f64,
    pub heaps_exponent: f64,
    pub compression_ratio: f64,
    pub atesman_readability: f64,
    /// Words counted for the per-100-words rates (masks excluded) and how many are
    /// different; not model inputs.
    pub n_words: usize,
    pub n_types: usize,
}

impl ContentFeatures {
    /// Values in [`FEATURE_NAMES`] order; non-finite values are replaced by 0.
    pub fn to_array(self) -> [f64; N_FEATURES] {
        let base: [f64; N_BASE_FEATURES] = [
            self.commas_per_sentence,
            self.semicolons_per_100w,
            self.questions_per_100s,
            self.exclamations_per_100s,
            self.colons_per_100w,
            self.sent_len_mean,
            self.sent_len_cv,
            self.long_sent_share,
            self.mattr_50,
            self.hapax_ratio,
            self.yules_k,
            self.mean_word_len,
            self.lexical_density,
            self.ai_lex_per_100w,
            self.human_lex_per_100w,
            self.ve_per_100w,
            self.ile_per_100w,
            self.cliche_per_100w,
            self.mis_per_100w,
            self.makta_per_100w,
            self.dik_per_100w,
            self.passive_per_100w,
            self.oov_rate,
            self.log_words,
            self.aorist_ratio,
        ];
        let tail: [f64; N_STYLE_TAIL_FEATURES] = [
            self.lower_start_share,
            self.short_word_share,
            self.long_word_share,
            self.multi_excl,
            self.interrobang,
            self.ellipsis,
            self.multi_q,
            self.word_entropy,
            self.heaps_exponent,
            self.compression_ratio,
            self.atesman_readability,
        ];
        let mut out = [0.0; N_FEATURES];
        let values = base
            .iter()
            .chain(&self.function_words_per_100w)
            .chain(&tail);
        for (o, &v) in out.iter_mut().zip(values) {
            *o = if v.is_finite() { v } else { 0.0 };
        }
        out
    }

    /// `(name, value)` pairs in [`FEATURE_NAMES`] order.
    #[cfg(test)]
    pub fn named(&self) -> Vec<(&'static str, f64)> {
        FEATURE_NAMES.iter().copied().zip(self.to_array()).collect()
    }
}

// ------------------------------------------------------------------- tokens

#[derive(Clone, Copy)]
pub(super) enum Tok<'a> {
    Word(&'a str),
    Mask,
}

fn skip_apostrophe_suffix(text: &str, i: usize) -> usize {
    let mut it = text[i..].char_indices();
    match it.next() {
        Some((_, '\'' | '’')) => {}
        _ => return i,
    }
    let mut end = i;
    for (k, c) in it {
        if is_letter(c) {
            end = i + k + c.len_utf8();
        } else {
            break;
        }
    }
    end
}

/// Calls `f` for every word and mask token of `text`, left to right.
pub(super) fn scan_tokens<'a>(text: &'a str, mut f: impl FnMut(Tok<'a>)) {
    let bytes = text.as_bytes();
    let n = text.len();
    let mut i = 0;
    while i < n {
        if bytes[i] == b'<' {
            let rest = &text[i..];
            if let Some(m) = MASKS.iter().find(|m| rest.starts_with(m.as_str())) {
                f(Tok::Mask);
                i = skip_apostrophe_suffix(text, i + m.len());
                continue;
            }
        }
        let Some(c) = text[i..].chars().next() else {
            break;
        };
        if is_letter(c) {
            let end = text[i..]
                .char_indices()
                .find(|&(_, ch)| !is_letter(ch))
                .map_or(n, |(k, _)| i + k);
            f(Tok::Word(&text[i..end]));
            i = skip_apostrophe_suffix(text, end);
        } else {
            i += c.len_utf8();
        }
    }
}

/// One pass over a sentence's tokens: count, first token and the last two.
#[derive(Default)]
struct SentenceScan<'a> {
    tokens: usize,
    first: Option<Tok<'a>>,
    prev: Option<Tok<'a>>,
    last: Option<Tok<'a>>,
}

impl<'a> SentenceScan<'a> {
    fn of(sentence: &'a str) -> Self {
        let mut s = Self::default();
        scan_tokens(sentence, |tok| {
            s.tokens += 1;
            if s.first.is_none() {
                s.first = Some(tok);
            }
            s.prev = s.last.replace(tok);
        });
        s
    }

    /// The first token is a word starting with a lower-case letter (`bugün geldim.`).
    fn starts_lower(&self) -> bool {
        matches!(self.first, Some(Tok::Word(w)) if w.chars().next().is_some_and(char::is_lowercase))
    }
}

/// Maximal punctuation runs: `[multi_excl, interrobang, multi_q, ellipsis]`. A run of `!`
/// and `?` counts once: as `interrobang` if it mixes both, else as `multi_excl` / `multi_q`
/// when at least 2 long. A run of at least 3 `.` (a `…` counts as 3) is an ellipsis.
fn punct_runs(text: &str) -> [usize; 4] {
    let mut out = [0usize; 4];
    let (mut excl, mut quest, mut dots) = (0usize, 0usize, 0usize);
    for c in text.chars().chain(std::iter::once(' ')) {
        match c {
            '!' => excl += 1,
            '?' => quest += 1,
            _ => {
                if excl > 0 && quest > 0 {
                    out[1] += 1;
                } else if excl >= 2 {
                    out[0] += 1;
                } else if quest >= 2 {
                    out[2] += 1;
                }
                excl = 0;
                quest = 0;
            }
        }
        match c {
            '.' => dots += 1,
            '…' => dots += 3,
            _ => {
                if dots >= 3 {
                    out[3] += 1;
                }
                dots = 0;
            }
        }
    }
    out
}

/// Turkish lower-casing; also drops U+0307 left by a non-Turkish lower-casing of `İ`.
pub(super) fn lower_text(text: &str) -> String {
    let lower = to_turkish_lower(text);
    if lower.contains('\u{0307}') {
        lower.replace('\u{0307}', "")
    } else {
        lower
    }
}

// ------------------------------------------------------------------- statistics

fn rate(count: usize, denom: usize, scale: f64) -> f64 {
    if denom == 0 {
        0.0
    } else {
        count as f64 * scale / denom as f64
    }
}

/// Population coefficient of variation; 0 for fewer than two values or a zero mean.
fn cv(values: &[usize]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<usize>() as f64 / n;
    if mean <= 0.0 {
        return 0.0;
    }
    let var = values
        .iter()
        .map(|&v| (v as f64 - mean).powi(2))
        .sum::<f64>()
        / n;
    var.sqrt() / mean
}

/// Moving-average type/token ratio over windows of `w` tokens; plain TTR when the text is
/// shorter than the window.
fn mattr(seq: &[u32], n_types: usize, w: usize) -> f64 {
    let n = seq.len();
    if n == 0 {
        return 0.0;
    }
    if n <= w {
        return n_types as f64 / n as f64;
    }
    let mut counts = vec![0u32; n_types];
    let mut distinct = 0usize;
    for &t in &seq[..w] {
        if let Some(c) = counts.get_mut(t as usize) {
            if *c == 0 {
                distinct += 1;
            }
            *c += 1;
        }
    }
    let mut total = distinct;
    for i in w..n {
        if let Some(c) = counts.get_mut(seq[i - w] as usize) {
            *c -= 1;
            if *c == 0 {
                distinct -= 1;
            }
        }
        if let Some(c) = counts.get_mut(seq[i] as usize) {
            if *c == 0 {
                distinct += 1;
            }
            *c += 1;
        }
        total += distinct;
    }
    total as f64 / ((n - w + 1) * w) as f64
}

/// Shannon entropy of the word frequencies, normalised by ln(words) so that it does not
/// grow with length alone; 0 below two words.
fn word_entropy(freq: &[u64], n_words: usize) -> f64 {
    if n_words < 2 {
        return 0.0;
    }
    let n = n_words as f64;
    let h: f64 = freq
        .iter()
        .filter(|&&f| f > 0)
        .map(|&f| {
            let p = f as f64 / n;
            -p * p.ln()
        })
        .sum();
    h / n.ln()
}

/// Heaps exponent: least-squares slope of ln(distinct words so far) on ln(words read),
/// over every prefix of the word sequence; 0 below 10 words.
fn heaps_exponent(seq: &[u32], n_types: usize) -> f64 {
    let n = seq.len();
    if n < 10 {
        return 0.0;
    }
    let mut seen = vec![false; n_types];
    let mut distinct = 0usize;
    let (mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0);
    for (i, &t) in seq.iter().enumerate() {
        if let Some(s) = seen.get_mut(t as usize) {
            if !*s {
                *s = true;
                distinct += 1;
            }
        }
        let x = ((i + 1) as f64).ln();
        let y = (distinct as f64).ln();
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
    }
    let m = n as f64;
    let den = m * sxx - sx * sx;
    if den <= 0.0 {
        0.0
    } else {
        (m * sxy - sx * sy) / den
    }
}

/// DEFLATE-compressed size / raw size; 0 for empty text or if compression fails.
fn compression_ratio(text: &str) -> f64 {
    use std::io::Write;
    if text.is_empty() {
        return 0.0;
    }
    let mut enc = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::default());
    if enc.write_all(text.as_bytes()).is_err() {
        return 0.0;
    }
    match enc.finish() {
        Ok(out) => out.len() as f64 / text.len() as f64,
        Err(_) => 0.0,
    }
}

/// Ateşman (1997) readability ([`atesman_score`]), 0 without words or sentences.
fn atesman(syllables: usize, n_words: usize, n_sent: usize) -> f64 {
    if n_words == 0 || n_sent == 0 {
        return 0.0;
    }
    atesman_score(
        syllables as f64 / n_words as f64,
        n_words as f64 / n_sent as f64,
    )
}

// ------------------------------------------------------------------- extraction

/// Computes all content features of one preprocessed document.
pub(super) fn content_features(clean: &str) -> ContentFeatures {
    let lower = lower_text(clean);

    // Lexical pass over word tokens (masks excluded).
    let mut ids: AHashMap<&str, u32> = AHashMap::new();
    let mut seq: Vec<u32> = Vec::new();
    let mut char_sum = 0usize;
    let mut content = 0usize;
    let (mut ai, mut human, mut ve, mut ile) = (0usize, 0usize, 0usize, 0usize);
    let (mut mis, mut makta, mut dik, mut passive) = (0usize, 0usize, 0usize, 0usize);
    let (mut short, mut long) = (0usize, 0usize);
    let mut syllables = 0usize;
    let mut fw = [0usize; N_FUNCTION_WORD_FEATURES];
    let function = &*FUNCTION_SET;
    let mut gate = ClicheGate::default();
    scan_tokens(&lower, |tok| {
        let Tok::Word(w) = tok else {
            return;
        };
        let next_id = ids.len() as u32;
        seq.push(*ids.entry(w).or_insert(next_id));
        let chars: Chars = w.chars().collect();
        char_sum += chars.len();
        if !function.contains(w) {
            content += 1;
        }
        ai += usize::from(lex_hit(w, AI_LEXICON));
        human += usize::from(lex_hit(w, HUMAN_LEXICON));
        ve += usize::from(w == "ve");
        ile += usize::from(w == "ile");
        mis += usize::from(is_mis(w));
        makta += usize::from(is_makta(w));
        dik += usize::from(is_dik(&chars));
        passive += usize::from(is_passive(&chars));
        short += usize::from(chars.len() <= 3);
        long += usize::from(chars.len() >= 10);
        // Words are lower-cased here; vowels per word, as `phonology::alphabet::syllable_count`.
        syllables += chars
            .iter()
            .filter(|&&c| is_turkish_vowel(c))
            .count()
            .max(1);
        if let Some(k) = function_word_index(w) {
            fw[k] += 1;
        }
        gate.push(w);
    });
    let n_words = seq.len();
    let n_types = ids.len();

    let mut freq = vec![0u64; n_types];
    for &t in &seq {
        if let Some(f) = freq.get_mut(t as usize) {
            *f += 1;
        }
    }
    let hapax = freq.iter().filter(|&&f| f == 1).count();
    let sum_sq: u64 = freq.iter().map(|&f| f * f).sum();
    let yules_k = if n_words == 0 {
        0.0
    } else {
        let n = n_words as f64;
        1e4 * (sum_sq as f64 - n) / (n * n)
    };
    let oov_tokens: u64 = ids
        .iter()
        .filter(|(w, _)| !is_known(w))
        .map(|(_, &id)| freq.get(id as usize).copied().unwrap_or(0))
        .sum();

    // Sentence structure, on the original text.
    let mut sent_lens: Vec<usize> = Vec::new();
    let (mut questions, mut exclamations) = (0usize, 0usize);
    let (mut aorist, mut progressive) = (0usize, 0usize);
    let mut lower_starts = 0usize;
    for para in clean.split("\n\n") {
        for line in para.split('\n') {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            for s in SentenceSegmenter::segment(line) {
                let scan = SentenceScan::of(s.text);
                let len = scan.tokens;
                if len == 0 {
                    continue;
                }
                sent_lens.push(len);
                questions += usize::from(s.text.contains('?'));
                exclamations += usize::from(s.text.contains('!'));
                lower_starts += usize::from(scan.starts_lower());
                match predicate_tense(scan.last, scan.prev) {
                    Some(Tense::Aorist) => aorist += 1,
                    Some(Tense::Progressive) => progressive += 1,
                    None => {}
                }
            }
        }
    }
    let n_sent = sent_lens.len();

    // Punctuation.
    let commas = clean.matches(',').count();
    let semicolons = clean.matches(';').count();
    let (colons, label_colons) = label_colon_counts(clean);
    let [multi_excl, interrobang, multi_q, ellipsis] = punct_runs(clean);

    let cliches = if n_words == 0 || !gate.open() {
        0
    } else {
        super::cliches::count(clean)
    };

    let per100 = |c: usize| rate(c, n_words, 100.0);
    ContentFeatures {
        commas_per_sentence: rate(commas, n_sent, 1.0),
        semicolons_per_100w: per100(semicolons),
        questions_per_100s: rate(questions, n_sent, 100.0),
        exclamations_per_100s: rate(exclamations, n_sent, 100.0),
        colons_per_100w: per100(colons.saturating_sub(label_colons)),
        sent_len_mean: rate(sent_lens.iter().sum(), n_sent, 1.0),
        sent_len_cv: cv(&sent_lens),
        long_sent_share: rate(sent_lens.iter().filter(|&&l| l > 30).count(), n_sent, 1.0),
        mattr_50: mattr(&seq, n_types, 50),
        hapax_ratio: rate(hapax, n_types, 1.0),
        yules_k,
        mean_word_len: rate(char_sum, n_words, 1.0),
        lexical_density: rate(content, n_words, 1.0),
        ai_lex_per_100w: per100(ai),
        human_lex_per_100w: per100(human),
        ve_per_100w: per100(ve),
        ile_per_100w: per100(ile),
        cliche_per_100w: per100(cliches),
        mis_per_100w: per100(mis),
        makta_per_100w: per100(makta),
        dik_per_100w: per100(dik),
        passive_per_100w: per100(passive),
        oov_rate: if n_words == 0 {
            0.0
        } else {
            oov_tokens as f64 / n_words as f64
        },
        log_words: (n_words as f64).ln_1p(),
        n_words,
        n_types,
        aorist_ratio: rate(aorist, aorist + progressive, 1.0),
        function_words_per_100w: fw.map(per100),
        lower_start_share: rate(lower_starts, n_sent, 1.0),
        short_word_share: rate(short, n_words, 1.0),
        long_word_share: rate(long, n_words, 1.0),
        multi_excl: rate(multi_excl, n_sent, 100.0),
        interrobang: rate(interrobang, n_sent, 100.0),
        ellipsis: rate(ellipsis, n_sent, 100.0),
        multi_q: rate(multi_q, n_sent, 100.0),
        word_entropy: word_entropy(&freq, n_words),
        heaps_exponent: heaps_exponent(&seq, n_types),
        compression_ratio: compression_ratio(&lower),
        atesman_readability: atesman(syllables, n_words, n_sent),
    }
}

#[cfg(test)]
mod tests {
    use super::super::lexicons::{
        FwMatch, AI_LEXICON, FUNCTION_WORD_FEATURES, HUMAN_LEXICON, N_FUNCTION_WORD_FEATURES,
    };
    use super::super::preprocess::preprocess;
    use super::*;

    fn extract_features(clean: &str) -> Vec<f64> {
        content_features(clean).to_array().to_vec()
    }

    const EPS: f64 = 1e-9;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-6
    }

    fn feature(text: &str, name: &str) -> f64 {
        let idx = FEATURE_NAMES
            .iter()
            .position(|n| *n == name)
            .unwrap_or_else(|| panic!("unknown feature {name}"));
        extract_features(text)[idx]
    }

    #[test]
    fn feature_count_matches_names() {
        assert_eq!(N_FEATURES, 68);
        assert_eq!(FEATURE_NAMES.len(), N_FEATURES);
        assert_eq!(extract_features("").len(), FEATURE_NAMES.len());
        // New features are appended, so the existing order stays stable.
        assert_eq!(FEATURE_NAMES[23], "log_words");
        assert_eq!(FEATURE_NAMES[24], "aorist_ratio");
        assert_eq!(FEATURE_NAMES[67], "atesman_readability");
        // The function-word block sits right after aorist_ratio, in FUNCTION_WORD_FEATURES order.
        assert_eq!(N_FUNCTION_WORD_FEATURES, 32);
        for (i, (name, _)) in FUNCTION_WORD_FEATURES.iter().enumerate() {
            assert_eq!(FEATURE_NAMES[25 + i], *name);
            assert!(name.is_ascii(), "{name}");
        }
        assert_eq!(extract_features("Bir cümle.").len(), FEATURE_NAMES.len());
        let named = content_features("Bir cümle.").named();
        assert_eq!(named.len(), FEATURE_NAMES.len());
        for ((n, _), expected) in named.iter().zip(FEATURE_NAMES) {
            assert_eq!(n, expected);
        }
        // Names are unique.
        let mut sorted = FEATURE_NAMES.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), FEATURE_NAMES.len());
    }

    #[test]
    fn degenerate_inputs_are_finite_and_zero() {
        let inputs = [
            "",
            "   \n\n  ",
            "kelime",
            "🚀🔥😀",
            "<ÖZEL> <SAYI> <YIL>'da <URL> <EPOSTA>.",
            "...!?;:,",
            "\n",
        ];
        for text in inputs {
            let v = extract_features(text);
            assert_eq!(v.len(), N_FEATURES, "{text:?}");
            assert!(v.iter().all(|x| x.is_finite()), "{text:?}: {v:?}");
        }
        for text in ["", "🚀🔥😀", "<ÖZEL> <SAYI> <YIL>'da <URL> <EPOSTA>."] {
            let f = content_features(text);
            assert_eq!(f.log_words, 0.0, "{text:?}");
            assert_eq!(f.mattr_50, 0.0);
            assert_eq!(f.mean_word_len, 0.0);
            assert_eq!(f.oov_rate, 0.0);
        }
        let one = content_features("kelime");
        assert!(close(one.mattr_50, 1.0));
        assert!(close(one.hapax_ratio, 1.0));
        assert!(close(one.yules_k, 0.0));
        assert!(close(one.sent_len_mean, 1.0));
        assert_eq!(one.sent_len_cv, 0.0);
        assert!(close(one.log_words, 2f64.ln()));
    }

    #[test]
    fn comma_rate_and_sentence_cv() {
        // 2 sentences of 4 and 2 words; 2 commas.
        let f = content_features("elma, armut, kiraz geldi. hava güzel.");
        assert!(close(f.commas_per_sentence, 1.0), "{f:?}");
        assert!(close(f.sent_len_mean, 3.0));
        // population std of [4, 2] = 1, mean 3.
        assert!(close(f.sent_len_cv, 1.0 / 3.0));
        assert_eq!(f.long_sent_share, 0.0);
        assert!(close(f.log_words, 7f64.ln()));
    }

    #[test]
    fn punctuation_rates() {
        let f = content_features("Geldin mi? Evet geldim! Şunu bil: hayat kısa; dikkat et.");
        // Sentences: "Geldin mi?", "Evet geldim!", "Şunu bil: hayat kısa; dikkat et."
        assert!(close(f.questions_per_100s, 100.0 / 3.0), "{f:?}");
        assert!(close(f.exclamations_per_100s, 100.0 / 3.0));
        // 10 words.
        assert!(close(f.semicolons_per_100w, 10.0));
        assert!(close(f.colons_per_100w, 10.0));
        // A colon inside a masked time (no following space) is not counted.
        let g = content_features("Saat <SAYI>:<SAYI> oldu.");
        assert_eq!(g.colons_per_100w, 0.0);
    }

    #[test]
    fn heading_lines_are_sentences() {
        // A heading line without final punctuation is its own sentence: 2 + 3 + 3 tokens.
        let f = content_features("Başlık burada\n\nbir iki üç. dört beş altı.");
        assert!(close(f.sent_len_mean, 8.0 / 3.0));
    }

    #[test]
    fn vocabulary_statistics() {
        // a a b c: types 3, hapax 2 (b, c), sum f^2 = 4 + 1 + 1 = 6, N = 4.
        let f = content_features("kedi kedi köpek kuş");
        assert!(close(f.hapax_ratio, 2.0 / 3.0));
        assert!(close(f.yules_k, 1e4 * (6.0 - 4.0) / 16.0));
        assert!(close(f.mattr_50, 3.0 / 4.0));
        assert!(close(f.mean_word_len, (4.0 + 4.0 + 5.0 + 3.0) / 4.0));
        // "ve" and "bu" are function words.
        let g = content_features("kedi ve bu köpek");
        assert!(close(g.lexical_density, 0.5));
        assert!(close(g.ve_per_100w, 25.0));
    }

    #[test]
    fn mattr_uses_window_of_50() {
        // 100 distinct words: every window is all-distinct.
        let words: Vec<String> = (0..100)
            .map(|i| format!("kelime{}", "a".repeat(i + 1)))
            .collect();
        let f = content_features(&words.join(" "));
        assert!(close(f.mattr_50, 1.0));
        // Two alternating words: every window of 50 holds 2 types.
        let g = content_features(&"elma armut ".repeat(60));
        assert!(close(g.mattr_50, 2.0 / 50.0));
    }

    #[test]
    fn mask_tokens_are_excluded_from_lexical_statistics() {
        let plain = content_features("kitap okundu ve ayrıca beğenildi.");
        let masked = content_features(
            "<ÖZEL> kitap <SAYI> okundu <YIL>'da ve <URL> ayrıca <EPOSTA> beğenildi <ÖZEL>'nin.",
        );
        assert!(close(plain.mattr_50, masked.mattr_50));
        assert!(close(plain.hapax_ratio, masked.hapax_ratio));
        assert!(close(plain.yules_k, masked.yules_k));
        assert!(close(plain.mean_word_len, masked.mean_word_len));
        assert!(close(plain.lexical_density, masked.lexical_density));
        assert!(close(plain.ai_lex_per_100w, masked.ai_lex_per_100w));
        assert!(close(plain.ve_per_100w, masked.ve_per_100w));
        assert!(close(plain.oov_rate, masked.oov_rate));
        assert!(close(plain.log_words, masked.log_words));
        // ...but they still count as structural tokens for sentence length.
        assert!(close(plain.sent_len_mean, 5.0));
        assert!(close(masked.sent_len_mean, 11.0));
        // The apostrophe suffix of a mask is not a word either.
        assert_eq!(content_features("<ÖZEL>'da").log_words, 0.0);
    }

    #[test]
    fn turkish_casing_is_respected() {
        // I -> ı and İ -> i before matching.
        let f = content_features("AYRICA DİYE IŞIK");
        assert!(close(f.ai_lex_per_100w, 100.0 / 3.0), "{f:?}");
        assert!(close(f.human_lex_per_100w, 100.0 / 3.0));
        // "IŞIK" and "ışık" are one type; "ISIK" would not be.
        let g = content_features("IŞIK ışık");
        assert!(close(g.hapax_ratio, 0.0));
        assert!(close(g.mattr_50, 0.5));
        let h = content_features("İNSAN insan");
        assert!(close(h.mattr_50, 0.5));
        // Decomposed İ (i + U+0307) from a non-Turkish lower-casing is folded too.
        let d = content_features("i\u{307}nsan insan");
        assert!(close(d.mattr_50, 0.5));
    }

    #[test]
    fn word_list_prefix_matching() {
        // artır matches inflected and derived forms; short exact entries do not overmatch.
        let f = content_features("artırılması artırmak güçlendirildi vurguladı");
        assert!(close(f.ai_lex_per_100w, 100.0), "{f:?}");
        let g = content_features("şubat şunu hiçbir hiç yinelemek yine");
        // hiçbir (prefix entry), hiç, yine -> 3 of 6; "şubat", "yinelemek" do not match "şu"/"yine".
        assert!(close(g.human_lex_per_100w, 50.0), "{g:?}");
    }

    #[test]
    fn question_clitic_matching() {
        let f = content_features("geldin mi? gelecek misin? müdür geldi mü?");
        // mi, misin, mü -> 3 of 7 words; "müdür" is not the clitic.
        assert!(close(f.human_lex_per_100w, 300.0 / 7.0), "{f:?}");
    }

    #[test]
    fn suffix_heuristics() {
        let f = content_features("gelmiş yapılmaktadır olduğunu söylendi kullanan yetmiş");
        // gelmiş (yetmiş is a numeral)
        assert!(close(f.mis_per_100w, 100.0 / 6.0), "{f:?}");
        // yapılmaktadır
        assert!(close(f.makta_per_100w, 100.0 / 6.0));
        // olduğunu
        assert!(close(f.dik_per_100w, 100.0 / 6.0));
        // yapılmaktadır, söylendi (kullanan is active)
        assert!(close(f.passive_per_100w, 200.0 / 6.0));
        // 1st person plural past is not a -DIk participle.
        assert_eq!(content_features("geldik gittik").dik_per_100w, 0.0);
        assert!(content_features("geldikten sonra").dik_per_100w > 0.0);
    }

    #[test]
    fn cliches_are_counted() {
        let f = content_features("Bu çalışma kritik bir rol oynamaktadır.");
        assert!(f.cliche_per_100w > 0.0, "{f:?}");
        assert_eq!(content_features("Bugün hava güzel.").cliche_per_100w, 0.0);
    }

    #[test]
    fn oov_proxy() {
        let known = content_features("kitaplarımızdan geliyorum evlerde ağacı");
        assert!(known.oov_rate < EPS, "{known:?}");
        let f = content_features("qwxzt kitap gelmis yapmıs");
        // qwxzt and the missing-diacritic forms gelmis / yapmıs are OOV.
        assert!(close(f.oov_rate, 0.75), "{f:?}");
    }

    #[test]
    fn extraction_is_deterministic() {
        let docs: Vec<String> = [
            "Ayrıca, bu süreç kritik bir rol oynamaktadır. Özellikle verimlilik artırılmalıdır.",
            "Ben de dün gittim ama hiç beğenmedim yani. Şu işe bak!",
            "",
            "<ÖZEL> <YIL>'da geldi.",
            "Başlık\n\nParagraf bir. Paragraf iki uzun, çok uzun bir cümle.",
        ]
        .repeat(20)
        .iter()
        .map(|s| s.to_string())
        .collect();
        let single: Vec<Vec<f64>> = docs.iter().map(|d| extract_features(d)).collect();
        let again: Vec<Vec<f64>> = docs.iter().map(|d| extract_features(d)).collect();
        assert_eq!(single, again);
    }

    #[test]
    fn works_on_preprocessed_markdown() {
        let raw = "Elbette! İşte **önemli** noktalar:\n\n- Ayrıca, 2023 yılında artış oldu.\n- İstanbul'da 5 kişi geldi.";
        let clean = preprocess(raw);
        let v = extract_features(&clean);
        assert!(v.iter().all(|x| x.is_finite()));
        assert!(feature(&clean, "ai_lex_per_100w") > 0.0);
    }

    #[test]
    fn cliche_gate_is_a_necessary_condition() {
        // One sentence per pattern of the three categories, plus near misses.
        let texts = [
            "Bu çalışma kritik bir rol oynamaktadır.",
            "Eğitim hayati bir önem taşımaktadır.",
            "Bu konu büyük bir öneme sahiptir.",
            "Teknoloji vazgeçilmez bir unsur oldu.",
            "Bu sadece bir araç değil, aynı zamanda bir yaşam biçimidir.",
            "Yalnızca ekonomi değil, bununla birlikte kültür de etkilendi.",
            "Etkisi turizm ile sınırlı kalmayıp tarımı da kapsadı.",
            "Proje yerel olmakla birlikte, ulusal bir niteliğini de taşımaktadır.",
            "Ekonominin ötesinde, konunun sosyal boyutu da var.",
            "Bu durum göz ardı edilmemelidir.",
            "Unutulmamalıdır ki her şey değişir.",
            "Özetlemek gerekirse sonuç iyi.",
            "Bu önemli bir adım oldu ve temel bir dönüm noktası sayıldı.",
            "KRITIK BIR ROL OYNAMAKTADIR.",
            "Bu değil, aynı zamanda o.",
            "Rol büyük. Bir önem yok. Bir çalışma yaptık.",
            "Bugün hava güzel.",
        ];
        for text in texts {
            let words = text
                .split(|c: char| !c.is_alphabetic())
                .filter(|w| !w.is_empty())
                .count() as f64;
            let direct = super::super::cliches::count(text) as f64;
            let got = content_features(text).cliche_per_100w;
            assert!(
                close(got, 100.0 * direct / words),
                "{text:?}: {got} vs {direct}"
            );
        }
        assert!(content_features(texts[0]).cliche_per_100w > 0.0);
    }

    // ------------------------------------------------------------------- labels and pruning

    /// Words of a cleaned text, as `log_words` counts them.
    fn n_words(clean: &str) -> f64 {
        content_features(clean).log_words.exp_m1().round()
    }

    #[test]
    fn label_colons_are_not_content_colons() {
        // A line-leading label is list residue, not a content colon.
        let text = "Adım 1: Malzemeleri hazırlayın.";
        assert_eq!(content_features(text).colons_per_100w, 0.0);

        // A mid-sentence colon still counts: 1 colon over 3 words.
        let mid = "Şunu söyledi: gel";
        assert!(close(content_features(mid).colons_per_100w, 100.0 / 3.0));
        // A finite verb before the colon makes it a clause, not a label.
        assert!(close(
            content_features("Ali dedi: gel").colons_per_100w,
            100.0 / 3.0
        ));
        assert!(
            content_features("Dosya incelendi, gereği düşünüldü: onandı.").colons_per_100w > 0.0
        );
        // So does a lead-in longer than four words.
        assert!(
            content_features("Bu konuda dikkat edilmesi gereken noktalar: zaman ve para.")
                .colons_per_100w
                > 0.0
        );
        // A label after a finished sentence counts as a label too.
        assert_eq!(
            content_features("Her şey bitti. Sonuç: başarılı bir deneme.").colons_per_100w,
            0.0
        );
        // Only the first colon of a sentence can end a label: 1 of 2 colons over 4 words.
        assert!(close(
            content_features("Ad: Ali, soyad: Veli.").colons_per_100w,
            25.0
        ));
        // Possessive nouns ending in -DI are not past-tense verbs.
        assert_eq!(
            content_features("Ürün fiyatı: yüz lira.").colons_per_100w,
            0.0
        );
    }

    #[test]
    fn label_colons_are_left_out_of_content() {
        let raw = "Elbette! İşte tarif:\n\n- **Adım 1:** Malzemeleri hazırlayın.\n\
               - **Adım 2:** Fırını 180 dereceye ısıtın.\n1. Hazırlık: Kabı yağlayın.\n\n\
               Bir not daha: şunu unutmayın: sabır gerekir.\n\n**Sonuç:** Afiyet olsun.";
        let clean = preprocess(raw);
        let total = clean
            .match_indices(':')
            .filter(|&(i, _)| {
                clean[i + 1..]
                    .chars()
                    .next()
                    .is_none_or(char::is_whitespace)
            })
            .count() as f64;
        let content = content_features(&clean).colons_per_100w * n_words(&clean) / 100.0;
        // 6 colons; the labels Adım 1, Adım 2, Hazırlık, Bir not daha, Sonuç are left out.
        assert!(close(total, 6.0), "{clean:?}");
        assert!(close(content, 1.0), "{clean:?}");
    }

    #[test]
    fn pruned_juzek_words_are_not_matched() {
        // Pruned: reversed (ters), unsupported (tutmuyor) and too-rare (yetersiz) Juzek lemmas.
        const PRUNED: [&str; 13] = [
            "detaylı",
            "takma",
            "dolayısıyla",
            "ilişkin",
            "itibaren",
            "zedele",
            "titizlik",
            "aksaklık",
            "uyul",
            "çaplı",
            "dir",
            "sürdürülebilik",
            "sıkılaş",
        ];
        for w in PRUNED {
            assert!(
                !AI_LEXICON.iter().chain(HUMAN_LEXICON).any(|e| e.word == w),
                "{w} is still listed"
            );
        }
        assert_eq!(AI_LEXICON.len(), 31);
        assert_eq!(HUMAN_LEXICON.len(), 36);
        let f = content_features(
            "detaylı takma dolayısıyla ilişkin itibaren zedeledi titizlikle aksaklıklar \
         uyulmalı çaplı dir sürdürülebilik sıkılaştı",
        );
        assert_eq!(f.ai_lex_per_100w, 0.0, "{f:?}");
        assert_eq!(f.human_lex_per_100w, 0.0, "{f:?}");
        // Kept entries still match; `dolayı` is exact now, so `dolayısıyla` above did not.
        let g = content_features("ayrıca dolayı vurguladı");
        assert!(close(g.ai_lex_per_100w, 200.0 / 3.0), "{g:?}");
        assert!(close(g.human_lex_per_100w, 100.0 / 3.0), "{g:?}");
    }

    #[test]
    fn aorist_ratio_on_sentence_final_predicates() {
        // gelir, gider: aorist; bakıyor: progressive.
        assert!(close(
            content_features("Gelir. Gider. Bakıyor.").aorist_ratio,
            2.0 / 3.0
        ));
        // Person endings, the -mAz negative, a -lA verb and the potential are aorist.
        let f = content_features(
            "Her gün erken kalkarım. Bunu sevmez. Bu yöntem verimliliği sağlar. \
         Sonuç değişebilir. Çocuklar oynuyorlar.",
        );
        assert!(close(f.aorist_ratio, 0.8), "{f:?}");
        // The predicate before a question clitic counts.
        assert!(close(
            content_features("Gelir misin? Geliyor musun?").aorist_ratio,
            0.5
        ));
        // -tIr after a vowel is the aorist of a t-final stem, not the copula.
        assert!(close(
            content_features("Kuş uçar. Balık yüzer. Öğretmen anlatır.").aorist_ratio,
            1.0
        ));
        // Not aorist: copula -DIr, past, -mAktAdIr, plural nouns, listed non-verbs, masks.
        for text in [
            "Bu çok önemlidir.",
            "Bu ürüne sahiptir.",
            "Dün geldi.",
            "Süreç devam etmektedir.",
            "Kitaplar.",
            "Hepsi bu kadar.",
            "Çok teşekkürler.",
            "Tamam.",
            "Toplam <SAYI>.",
            "Yağmur yağıyor.",
            "",
        ] {
            assert_eq!(content_features(text).aorist_ratio, 0.0, "{text:?}");
        }
    }

    // ------------------------------------------------------------------- stylometry groups

    #[test]
    fn function_word_profile() {
        // 12 words: ben, de x2, için, çünkü, sen, mi; "bunu" and "senin" are not exact forms.
        let text = "Ben de bunu senin için yaptım çünkü sen de istedin mi bilmem.";
        for (name, count) in [
            ("fw_ben", 1.0),
            ("fw_de", 2.0),
            ("fw_icin", 1.0),
            ("fw_cunku", 1.0),
            ("fw_sen", 1.0),
            ("fw_mi", 1.0),
            ("fw_bu", 0.0),
            ("fw_kendi", 0.0),
        ] {
            assert!(close(feature(text, name), count * 100.0 / 12.0), "{name}");
        }
        // `da` is `fw_de`; inflected question clitics count, `müdür` does not; `kendini` is
        // not `kendi`.
        let g = "o da gelir mi misiniz müdür kendini kendi";
        assert!(close(feature(g, "fw_o"), 100.0 / 8.0));
        assert!(close(feature(g, "fw_de"), 100.0 / 8.0));
        assert!(close(feature(g, "fw_mi"), 200.0 / 8.0));
        assert!(close(feature(g, "fw_kendi"), 100.0 / 8.0));
        // Turkish casing: İÇİN -> için, DEĞİL -> değil.
        let h = "İÇİN ÇÜNKÜ DEĞİL";
        for name in ["fw_icin", "fw_cunku", "fw_degil"] {
            assert!(close(feature(h, name), 100.0 / 3.0), "{name}");
        }
        // ve / ile are their own features, not part of the profile.
        for (name, m) in FUNCTION_WORD_FEATURES {
            if let FwMatch::Forms(forms) = m {
                assert!(!forms.contains(&"ve") && !forms.contains(&"ile"), "{name}");
            }
        }
        let f = content_features("");
        assert!(f.function_words_per_100w.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn sentence_start_case() {
        // Sentences: "bugün geldim." (lower), "Yarın giderim" (upper), "nasılsın?" (lower).
        let f = content_features("bugün geldim. Yarın giderim\nnasılsın?");
        assert!(close(f.lower_start_share, 2.0 / 3.0), "{f:?}");
        // A mask start is not lower-case.
        let g = content_features("<ÖZEL> geldi. Ali \"tamam.\" Dedi ki \"evet!\"");
        assert_eq!(g.lower_start_share, 0.0, "{g:?}");
        assert_eq!(content_features("").lower_start_share, 0.0);
    }

    #[test]
    fn word_length_shares() {
        // ev, bir, gel: <= 3 letters; kitaplarım: 10 letters; evde (4) and kelimeler (9): neither.
        let f = content_features("ev bir gel evde kelimeler kitaplarım <ÖZEL>");
        assert!(close(f.short_word_share, 3.0 / 6.0), "{f:?}");
        assert!(close(f.long_word_share, 1.0 / 6.0), "{f:?}");
        assert_eq!(content_features("<SAYI>").short_word_share, 0.0);
    }

    #[test]
    fn punctuation_combinations_per_100_sentences() {
        // 5 sentences, one run of each kind.
        let f = content_features("Gel!! Ne?! Hmm... Nasıl?? Tamam.");
        assert!(close(f.multi_excl, 20.0), "{f:?}");
        assert!(close(f.interrobang, 20.0), "{f:?}");
        assert!(close(f.ellipsis, 20.0), "{f:?}");
        assert!(close(f.multi_q, 20.0), "{f:?}");
        // Mixed runs are interrobangs only; single marks and `..` count nowhere; `…` is an
        // ellipsis.
        let g = content_features("Ne!? Gel!!? Tamam! Nasıl? Bitti.. Peki…");
        assert!(
            g.interrobang > 0.0 && g.multi_excl == 0.0 && g.multi_q == 0.0,
            "{g:?}"
        );
        assert!(close(g.interrobang, 2.0 * g.ellipsis), "{g:?}");
        assert_eq!(content_features("Tamam! Nasıl? Bitti.").ellipsis, 0.0);
    }

    #[test]
    fn word_entropy_is_length_normalised() {
        // Two types, two tokens each: ln 2 / ln 4.
        assert!(close(content_features("ev göl ev göl").word_entropy, 0.5));
        // All tokens distinct: maximal.
        assert!(close(content_features("ev göl dağ yol").word_entropy, 1.0));
        assert_eq!(content_features("ev").word_entropy, 0.0);
        assert_eq!(content_features("").word_entropy, 0.0);
    }

    #[test]
    fn heaps_exponent_tracks_vocabulary_growth() {
        let distinct = "bir iki üç dört beş altı yedi sekiz dokuz on onbir oniki";
        assert!(close(content_features(distinct).heaps_exponent, 1.0));
        let repeated = "ev ev ev ev ev ev ev ev ev ev ev ev";
        assert!(close(content_features(repeated).heaps_exponent, 0.0));
        // Below 10 words the fit is not attempted.
        assert_eq!(content_features("ev göl dağ").heaps_exponent, 0.0);
    }

    #[test]
    fn compression_ratio_is_lower_for_repetitive_text() {
        let repetitive = content_features(&"Bu çok güzel bir gün. ".repeat(20)).compression_ratio;
        let varied = content_features(
            "Sabah erkenden kalktık, kahvaltıdan sonra sahile indik. Rüzgâr sertti ama deniz \
         sakindi. Öğleden sonra müzeyi gezdik; akşam da eski bir lokantada balık yedik.",
        )
        .compression_ratio;
        assert!(
            repetitive > 0.0 && repetitive < varied,
            "{repetitive} vs {varied}"
        );
        assert_eq!(content_features("").compression_ratio, 0.0);
    }

    #[test]
    fn atesman_readability_formula() {
        // "ev" 1 syllable, "güzel" 2: 198.825 - 40.175 * 1.5 - 2.610 * 2.
        assert!(close(
            content_features("Ev güzel.").atesman_readability,
            198.825 - 40.175 * 1.5 - 2.610 * 2.0
        ));
        assert_eq!(content_features("").atesman_readability, 0.0);
    }
}
