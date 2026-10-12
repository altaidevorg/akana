//! Content features for the Turkish AI-text detector.
//!
//! [`extract_features`] maps **already preprocessed** text (the output of
//! [`super::preprocess()`]: markup stripped, NFKC, chatbot opener dropped, masks such as
//! `<YIL>`, `<SAYI>`, `<ÖZEL>`, `<URL>`, `<EPOSTA>`) to a fixed-length vector in the order
//! of [`FEATURE_NAMES`]. Training (through PyO3) and inference call the same function.
//!
//! Design rules:
//! - No formatting features: markup, emoji and typography are stripped before.
//! - Every feature is length-normalised (per 100 words, per sentence, per 100 sentences,
//!   per paragraph or a share), except `log_words`, which is the length itself.
//! - Mask tokens (and their apostrophe suffix, `<ÖZEL>'da`) are excluded from every
//!   lexical statistic and from the word count used as the per-100-words denominator. They
//!   only count as structural tokens for sentence and paragraph length.
//! - The morphology analyzer is **not** used (it overgenerates, see
//!   `research/2026-10-08-project-review.md` C2). Suffix features are surface heuristics.
//! - Degenerate input (empty, one word, emoji only, masks only) yields 0 instead of NaN or
//!   infinity. The output is deterministic: no hash-map iteration order reaches a float.
//!
//! Words are maximal runs of letters (combining marks included) in the Turkish-lower-cased
//! text; an apostrophe suffix after a word is skipped, digits and punctuation separate
//! words. Sentences come from [`SentenceSegmenter`] run on each line separately, so a
//! heading or list item without final punctuation is its own sentence; paragraphs are
//! separated by a blank line (`\n\n`, as produced by [`super::preprocess::normalize`]).

use std::sync::LazyLock;

use ahash::{AHashMap, AHashSet};
use smallvec::SmallVec;

use super::preprocess::is_letter;
use super::preprocess::{MASK_EMAIL, MASK_NUMBER, MASK_PROPER_NOUN, MASK_URL, MASK_YEAR};
use crate::phonology::alphabet::to_turkish_lower;
use crate::style::{ClicheCategory, ClicheMatcher};
use crate::tokenization::SentenceSegmenter;

/// Number of content features.
pub const N_FEATURES: usize = 71;

/// Features before the function-word profile (D-002 set + D-005 `aorist_ratio`).
const N_BASE_FEATURES: usize = 26;
/// Stylometry features after the function-word profile.
const N_STYLE_TAIL_FEATURES: usize = 13;
const _: () =
    assert!(N_BASE_FEATURES + N_FUNCTION_WORD_FEATURES + N_STYLE_TAIL_FEATURES == N_FEATURES);

/// Feature names, in the order of [`extract_features`].
///
/// | name | definition |
/// |---|---|
/// | `commas_per_sentence` | `,` count / sentences |
/// | `semicolons_per_100w` | `;` count per 100 words |
/// | `questions_per_100s` | sentences containing `?`, per 100 sentences |
/// | `exclamations_per_100s` | sentences containing `!`, per 100 sentences |
/// | `colons_per_100w` | `:` followed by whitespace or end of text, per 100 words, **without** colons that end a line- or sentence-leading label (`Adım 1:`, `Örnek:`); those are the format signal `label_lines` (D-005, `label_colon_counts`) |
/// | `sent_len_mean` | mean sentence length in tokens (words + masks) |
/// | `sent_len_cv` | population std / mean of sentence length (0 with < 2 sentences) |
/// | `para_len_cv` | std / mean of paragraph length in tokens (0 with < 2 paragraphs) |
/// | `long_sent_share` | share of sentences longer than 30 tokens |
/// | `mattr_50` | moving-average type/token ratio, window 50 (plain TTR below 50 words) |
/// | `hapax_ratio` | types occurring once / types |
/// | `yules_k` | Yule's K = 10⁴ · (Σ f² − N) / N² over word frequencies f |
/// | `mean_word_len` | mean word length in characters |
/// | `lexical_density` | share of words not in [`FUNCTION_WORDS`] |
/// | `ai_lex_per_100w` | [`AI_LEXICON`] hits per 100 words |
/// | `human_lex_per_100w` | [`HUMAN_LEXICON`] hits per 100 words |
/// | `ve_per_100w` | `ve` per 100 words |
/// | `ile_per_100w` | `ile` per 100 words |
/// | `cliche_per_100w` | `ConclusionHedge` + `RhetoricalCalque` + `IndefiniteArticleInflation` matches of [`ClicheMatcher`] per 100 words |
/// | `mis_per_100w` | words with `-mIş` (+ person/copula ending) per 100 words |
/// | `makta_per_100w` | words with `-mAktA(dIr)` per 100 words |
/// | `dik_per_100w` | words with a `-DIk` participle (`-DIğ`+vowel, `-DIk`+`çA/tAn/lAr`) per 100 words |
/// | `passive_per_100w` | words with a passive `-(I)l-` / `-(I)n-` + verbal suffix per 100 words |
/// | `oov_rate` | share of word tokens not explained by the Zemberek root lexicon + surface suffix chain |
/// | `log_words` | ln(1 + words) |
/// | `aorist_ratio` | among sentences whose last word is a present-tense predicate, the share in the aorist (`-Ir/-Ar/-r`, `-mAz`, + person) rather than the progressive `-Iyor`; 0 without such a sentence (`predicate_tense`) |
/// | `fw_<word>` (32) | [`FUNCTION_WORD_FEATURES`] hits per 100 words, one feature per closed-class word |
/// | `lower_start_share` | share of sentences whose first token is a word starting with a lower-case letter |
/// | `no_terminal_punct_share` | share of sentences (or line ends) not ending in `.` `!` `?` (closing quotes/brackets skipped) |
/// | `short_word_share` | share of words with at most 3 letters |
/// | `long_word_share` | share of words with at least 10 letters |
/// | `multi_excl` | runs of 2+ `!` (no `?`), per 100 sentences |
/// | `interrobang` | runs of `!`/`?` that mix both (`?!`, `!?`, `?!!`), per 100 sentences |
/// | `ellipsis` | runs of 3+ `.` (or `…`), per 100 sentences |
/// | `multi_q` | runs of 2+ `?` (no `!`), per 100 sentences |
/// | `sents_per_para_mean` | sentences / paragraphs |
/// | `word_entropy` | Shannon entropy of the word distribution / ln(words) (0 below 2 words) |
/// | `heaps_exponent` | least-squares slope of ln(types seen) on ln(words read), over every prefix of the text (0 below 10 words) |
/// | `compression_ratio` | DEFLATE-compressed size / raw size of the lower-cased text (0 for empty text) |
/// | `atesman_readability` | Ateşman (1997) score 198.825 − 40.175 · syllables/word − 2.610 · words/sentence; syllables = vowels (at least 1 per word); 0 without words or sentences |
pub const FEATURE_NAMES: &[&str] = &[
    "commas_per_sentence",
    "semicolons_per_100w",
    "questions_per_100s",
    "exclamations_per_100s",
    "colons_per_100w",
    "sent_len_mean",
    "sent_len_cv",
    "para_len_cv",
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
    "no_terminal_punct_share",
    "short_word_share",
    "long_word_share",
    "multi_excl",
    "interrobang",
    "ellipsis",
    "multi_q",
    "sents_per_para_mean",
    "word_entropy",
    "heaps_exponent",
    "compression_ratio",
    "atesman_readability",
];

/// Content features of one document; field order equals [`FEATURE_NAMES`].
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ContentFeatures {
    pub commas_per_sentence: f64,
    pub semicolons_per_100w: f64,
    pub questions_per_100s: f64,
    pub exclamations_per_100s: f64,
    pub colons_per_100w: f64,
    pub sent_len_mean: f64,
    pub sent_len_cv: f64,
    pub para_len_cv: f64,
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
    /// `fw_*` values in [`FUNCTION_WORD_FEATURES`] order, per 100 words.
    pub function_words_per_100w: [f64; N_FUNCTION_WORD_FEATURES],
    pub lower_start_share: f64,
    pub no_terminal_punct_share: f64,
    pub short_word_share: f64,
    pub long_word_share: f64,
    pub multi_excl: f64,
    pub interrobang: f64,
    pub ellipsis: f64,
    pub multi_q: f64,
    pub sents_per_para_mean: f64,
    pub word_entropy: f64,
    pub heaps_exponent: f64,
    pub compression_ratio: f64,
    pub atesman_readability: f64,
}

impl ContentFeatures {
    /// Values in [`FEATURE_NAMES`] order; non-finite values are replaced by 0.
    pub fn to_array(&self) -> [f64; N_FEATURES] {
        let base: [f64; N_BASE_FEATURES] = [
            self.commas_per_sentence,
            self.semicolons_per_100w,
            self.questions_per_100s,
            self.exclamations_per_100s,
            self.colons_per_100w,
            self.sent_len_mean,
            self.sent_len_cv,
            self.para_len_cv,
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
            self.no_terminal_punct_share,
            self.short_word_share,
            self.long_word_share,
            self.multi_excl,
            self.interrobang,
            self.ellipsis,
            self.multi_q,
            self.sents_per_para_mean,
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
    pub fn named(&self) -> Vec<(&'static str, f64)> {
        FEATURE_NAMES.iter().copied().zip(self.to_array()).collect()
    }
}

// ------------------------------------------------------------------- word lists

/// How a lexicon entry matches a Turkish-lower-cased word.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Match {
    /// The whole word equals the entry.
    Exact,
    /// The word starts with the entry (`artır` matches `artırılması`). Used for stems of at
    /// least 5 characters and verb stems of at least 4, so short stems do not overmatch,
    /// unless the stem starts a lemma pruned by D-005 (`dolayı` / `dolayısıyla`).
    Prefix,
    /// The question clitic: `mı mi mu mü` alone or with a person/past/evidential ending
    /// (`mısın`, `miyiz`, `mıydı`, `miymiş`). `-DIr` forms are not matched because
    /// `müdür` ("director") is spelled like `mü-dür`.
    QuestionClitic,
}

/// One word-list entry: a Turkish-lower-cased lemma or stem and its match mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LexEntry {
    pub word: &'static str,
    pub mode: Match,
}

const fn lex(word: &'static str, mode: Match) -> LexEntry {
    LexEntry { word, mode }
}

// Source: Juzek, "AI-Associated Lexical Shifts Across 34 Languages" (arXiv:2605.25358),
// Turkish news continued by GPT-4.1-mini vs the human continuation,
// https://aiwordexplorer.com/data/tr_news_gpt4.1-mini.json (fetched 2026-10-08; Stanza
// lemmas, counts averaged over K=12 resamples, ~62k documents per side).
// Generated by `scripts/aidetect/derive_wordlists.py`; do not edit by hand. Criterion:
// - AI side: POS ∈ closed classes + ADJ + VERB, NOUN only with the abstract suffix -lIk;
//   human side: closed classes only (ADV ADP CCONJ SCONJ DET PRON AUX PART INTJ). Other
//   nouns/verbs in this news data are topic or reporting markers, not style.
// - letters only, AI lemmas >= 3 chars, `ve`/`ile` excluded (own features);
// - counts summed per lemma across POS; support a + h >= 80;
// - lr = ln((a + 1) / (h + 1)); AI: lr >= ln 3, human: lr <= -ln 3;
// - top 40 per side by |lr|;
// - D-005 pruning: 13 of the 80 lemmas are removed (script constant `PRUNED`) because the
//   word-shift check on our own train data (research/2026-10-08-turkish-ai-word-shifts.md)
//   found them reversed (`detaylı`, `takma`, `dolayısıyla`, `ilişkin`, `itibaren`), not
//   supported (`zedele`, `titizlik`, `aksaklık`, `uyul`, `çaplı`, `dir`) or too rare
//   (`sürdürülebilik`, `sıkılaş`). No lower-ranked lemma replaces them, and a kept lemma
//   that starts a pruned one is matched exactly (`dolayı`, not `dolayısıyla`).
// `r` below is the smoothed AI/human ratio. Known noise: the Stanza lemma artefact `moder`
// and news-register items (`yetkili`, `satın`, `ayına`).

/// Words GPT-4.1-mini overuses in Turkish news (see the source comment above).
pub const AI_LEXICON: &[LexEntry] = &[
    lex("ayrıca", Match::Prefix),         // r=26.9 (a=5456, h=202)
    lex("yetkili", Match::Prefix),        // r=19.3 (a=3749, h=193)
    lex("disiplinli", Match::Prefix),     // r=16.4 (a=111, h=6)
    lex("özellikle", Match::Prefix),      // r=16.1 (a=5530, h=342)
    lex("derinleş", Match::Prefix),       // r=15.4 (a=246, h=15)
    lex("yenilikçi", Match::Prefix),      // r=12.5 (a=438, h=34)
    lex("katılımcı", Match::Prefix),      // r=12.3 (a=207, h=16)
    lex("taşır", Match::Prefix),          // r=11.9 (a=80, h=6)
    lex("artır", Match::Prefix),          // r=11.8 (a=4383, h=371)
    lex("hızlan", Match::Prefix),         // r=10.9 (a=753, h=68)
    lex("şeffaflık", Match::Prefix),      // r=10.2 (a=162, h=15)
    lex("güçlen", Match::Prefix),         // r=10.1 (a=1802, h=177)
    lex("kültürel", Match::Prefix),       // r=9.3 (a=982, h=105)
    lex("zorluk", Match::Prefix),         // r=9.2 (a=886, h=95)
    lex("moder", Match::Prefix),          // r=9.2 (a=737, h=79)
    lex("belirsizlik", Match::Prefix),    // r=9.1 (a=325, h=35)
    lex("gözlemle", Match::Prefix),       // r=8.8 (a=318, h=35)
    lex("interaktif", Match::Prefix),     // r=8.7 (a=88, h=9)
    lex("dayanıklılık", Match::Prefix),   // r=8.7 (a=138, h=15)
    lex("verimlilik", Match::Prefix),     // r=8.5 (a=371, h=42)
    lex("yenilenebilir", Match::Prefix),  // r=8.4 (a=84, h=9)
    lex("sürdürülebilir", Match::Prefix), // r=8.4 (a=578, h=68)
    lex("çevresel", Match::Prefix),       // r=8.2 (a=227, h=27)
    lex("genellikle", Match::Prefix),     // r=7.8 (a=266, h=33)
    lex("dalgalan", Match::Prefix),       // r=7.5 (a=75, h=9)
    lex("duygusal", Match::Prefix),       // r=7.2 (a=149, h=20)
    lex("kritik", Match::Prefix),         // r=7.1 (a=686, h=96)
    lex("hedefle", Match::Prefix),        // r=7.1 (a=1482, h=208)
    lex("deneyimli", Match::Prefix),      // r=7.1 (a=136, h=18)
    lex("vurgula", Match::Prefix),        // r=6.9 (a=3046, h=438)
    lex("şeffaf", Match::Prefix),         // r=6.9 (a=206, h=29)
];

/// Words the human side of the same data favours (see the source comment above).
pub const HUMAN_LEXICON: &[LexEntry] = &[
    lex("şöyle", Match::Prefix),      // r=0.0 (a=1, h=342)
    lex("şunları", Match::Prefix),    // r=0.0 (a=2, h=338)
    lex("beraber", Match::Prefix),    // r=0.0 (a=2, h=177)
    lex("diye", Match::Exact),        // r=0.0 (a=23, h=924)
    lex("fakat", Match::Prefix),      // r=0.0 (a=2, h=91)
    lex("yani", Match::Exact),        // r=0.0 (a=7, h=168)
    lex("hepsi", Match::Prefix),      // r=0.0 (a=3, h=92)
    lex("ayına", Match::Prefix),      // r=0.1 (a=3, h=78)
    lex("yine", Match::Exact),        // r=0.1 (a=18, h=308)
    lex("mı", Match::QuestionClitic), // r=0.1 (a=32, h=431)
    lex("hiç", Match::Exact),         // r=0.1 (a=22, h=235)
    lex("bunun", Match::Prefix),      // r=0.1 (a=161, h=1588)
    lex("hatta", Match::Prefix),      // r=0.1 (a=16, h=162)
    lex("belki", Match::Prefix),      // r=0.1 (a=8, h=80)
    lex("şu", Match::Exact),          // r=0.1 (a=56, h=532)
    lex("zaten", Match::Prefix),      // r=0.1 (a=11, h=100)
    lex("beri", Match::Exact),        // r=0.1 (a=13, h=114)
    lex("dolayı", Match::Exact),      // r=0.1 (a=61, h=457)
    lex("ya", Match::Exact),          // r=0.2 (a=92, h=613)
    lex("ilk", Match::Exact),         // r=0.2 (a=16, h=100)
    lex("sana", Match::Exact),        // r=0.2 (a=26, h=152)
    lex("aslında", Match::Prefix),    // r=0.2 (a=22, h=122)
    lex("yok", Match::Exact),         // r=0.2 (a=28, h=150)
    lex("üzere", Match::Prefix),      // r=0.2 (a=275, h=1430)
    lex("satın", Match::Prefix),      // r=0.2 (a=32, h=171)
    lex("sizlerin", Match::Prefix),   // r=0.2 (a=58, h=283)
    lex("önce", Match::Exact),        // r=0.2 (a=179, h=861)
    lex("nerede", Match::Prefix),     // r=0.2 (a=35, h=169)
    lex("hiçbir", Match::Prefix),     // r=0.2 (a=68, h=312)
    lex("sonra", Match::Prefix),      // r=0.2 (a=360, h=1590)
    lex("ama", Match::Exact),         // r=0.2 (a=138, h=592)
    lex("bile", Match::Exact),        // r=0.2 (a=73, h=307)
    lex("az", Match::Exact),          // r=0.3 (a=68, h=271)
    lex("biraz", Match::Prefix),      // r=0.3 (a=30, h=113)
    lex("kadar", Match::Prefix),      // r=0.3 (a=608, h=2094)
    lex("kim", Match::Exact),         // r=0.3 (a=31, h=98)
];

/// Number of function-word profile features (`fw_*`).
pub const N_FUNCTION_WORD_FEATURES: usize = 32;

/// How a function-word profile feature matches a Turkish-lower-cased word token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FwMatch {
    /// The token equals one of the forms (no inflected forms: `kendini` is not `kendi`).
    Forms(&'static [&'static str]),
    /// The question clitic `mI` with its endings, as [`Match::QuestionClitic`].
    QuestionClitic,
}

/// Function-word profile, the classic stylometry feature set: `(feature name, match)`, one
/// feature per word, counted per 100 words. A fixed closed-class selection (conjunctions,
/// postpositions, the existentials `var`/`yok`, the negator `değil`, demonstratives,
/// determiners and quantifiers, personal and reflexive pronouns, the clitics `de/da` and
/// `mI`), following the closed word classes of Göksel & Kerslake, *Turkish: A Comprehensive
/// Grammar* (Routledge, 2005). It was fixed without looking at any corpus (D-005: no
/// data-derived lists, which would leak into the leave-one-family-out folds). `ve` and
/// `ile` are separate features. Names are ASCII-folded (`fw_icin`, `fw_cunku`).
pub const FUNCTION_WORD_FEATURES: [(&str, FwMatch); N_FUNCTION_WORD_FEATURES] = [
    ("fw_icin", FwMatch::Forms(&["için"])),
    ("fw_gore", FwMatch::Forms(&["göre"])),
    ("fw_gibi", FwMatch::Forms(&["gibi"])),
    ("fw_kadar", FwMatch::Forms(&["kadar"])),
    ("fw_ama", FwMatch::Forms(&["ama"])),
    ("fw_fakat", FwMatch::Forms(&["fakat"])),
    ("fw_cunku", FwMatch::Forms(&["çünkü"])),
    ("fw_ancak", FwMatch::Forms(&["ancak"])),
    ("fw_ki", FwMatch::Forms(&["ki"])),
    ("fw_ise", FwMatch::Forms(&["ise"])),
    ("fw_ya", FwMatch::Forms(&["ya"])),
    ("fw_veya", FwMatch::Forms(&["veya"])),
    ("fw_hem", FwMatch::Forms(&["hem"])),
    ("fw_ne", FwMatch::Forms(&["ne"])),
    ("fw_degil", FwMatch::Forms(&["değil"])),
    ("fw_var", FwMatch::Forms(&["var"])),
    ("fw_yok", FwMatch::Forms(&["yok"])),
    ("fw_bu", FwMatch::Forms(&["bu"])),
    ("fw_su", FwMatch::Forms(&["şu"])),
    ("fw_o", FwMatch::Forms(&["o"])),
    ("fw_bir", FwMatch::Forms(&["bir"])),
    ("fw_her", FwMatch::Forms(&["her"])),
    ("fw_daha", FwMatch::Forms(&["daha"])),
    ("fw_en", FwMatch::Forms(&["en"])),
    ("fw_cok", FwMatch::Forms(&["çok"])),
    ("fw_ben", FwMatch::Forms(&["ben"])),
    ("fw_biz", FwMatch::Forms(&["biz"])),
    ("fw_sen", FwMatch::Forms(&["sen"])),
    ("fw_siz", FwMatch::Forms(&["siz"])),
    ("fw_kendi", FwMatch::Forms(&["kendi"])),
    ("fw_de", FwMatch::Forms(&["de", "da"])),
    ("fw_mi", FwMatch::QuestionClitic),
];

/// Form → index into [`FUNCTION_WORD_FEATURES`].
static FUNCTION_WORD_INDEX: LazyLock<AHashMap<&'static str, usize>> = LazyLock::new(|| {
    let mut map = AHashMap::new();
    for (i, (_, m)) in FUNCTION_WORD_FEATURES.iter().enumerate() {
        if let FwMatch::Forms(forms) = m {
            for f in *forms {
                map.insert(*f, i);
            }
        }
    }
    map
});

/// Index of the question-clitic feature in [`FUNCTION_WORD_FEATURES`].
const FW_QUESTION_CLITIC: usize = N_FUNCTION_WORD_FEATURES - 1;
const _: () = assert!(matches!(
    FUNCTION_WORD_FEATURES[FW_QUESTION_CLITIC].1,
    FwMatch::QuestionClitic
));

/// [`FUNCTION_WORD_FEATURES`] index of a lower-cased word token, if any.
fn function_word_index(w: &str) -> Option<usize> {
    FUNCTION_WORD_INDEX
        .get(w)
        .copied()
        .or_else(|| is_question_clitic(w).then_some(FW_QUESTION_CLITIC))
}

/// Turkish function words (conjunctions, postpositions, pronouns, determiners, particles,
/// discourse adverbs) for `lexical_density`. Inflected pronoun forms are listed
/// explicitly; everything else counts as a content word.
pub const FUNCTION_WORDS: &[&str] = &[
    // conjunctions and discourse connectives
    "ve",
    "veya",
    "ya",
    "yahut",
    "ile",
    "ama",
    "fakat",
    "lakin",
    "ancak",
    "oysa",
    "oysaki",
    "halbuki",
    "çünkü",
    "zira",
    "eğer",
    "şayet",
    "madem",
    "ki",
    "de",
    "da",
    "dahi",
    "bile",
    "hem",
    "ne",
    "ise",
    "iken",
    "yoksa",
    "hatta",
    "üstelik",
    "ayrıca",
    "yani",
    "nitekim",
    "dolayısıyla",
    "böylece",
    "sonuçta",
    "ister",
    "gerek",
    "sanki",
    "meğer",
    // postpositions
    "gibi",
    "kadar",
    "için",
    "göre",
    "karşı",
    "rağmen",
    "dolayı",
    "beri",
    "önce",
    "sonra",
    "üzere",
    "doğru",
    "ait",
    "dair",
    "ilişkin",
    "boyunca",
    "hakkında",
    "yerine",
    "diye",
    "itibaren",
    "başka",
    "ötürü",
    "değin",
    "dek",
    "denli",
    "değil",
    "olarak",
    // question particle and copular words
    "mi",
    "mı",
    "mu",
    "mü",
    "var",
    "yok",
    "evet",
    "hayır",
    // pronouns and demonstratives
    "bu",
    "şu",
    "o",
    "bunlar",
    "şunlar",
    "onlar",
    "bunu",
    "şunu",
    "onu",
    "bunun",
    "şunun",
    "onun",
    "buna",
    "şuna",
    "ona",
    "bunda",
    "şunda",
    "onda",
    "bundan",
    "şundan",
    "ondan",
    "bununla",
    "şununla",
    "onunla",
    "bunları",
    "onları",
    "bunların",
    "onların",
    "ben",
    "sen",
    "biz",
    "siz",
    "beni",
    "seni",
    "bizi",
    "sizi",
    "bana",
    "sana",
    "bize",
    "size",
    "benim",
    "senin",
    "bizim",
    "sizin",
    "bende",
    "sende",
    "bizde",
    "sizde",
    "benden",
    "senden",
    "bizden",
    "sizden",
    "kendi",
    "kendisi",
    "kendini",
    "kendine",
    "kendisini",
    "kendilerini",
    "birbiri",
    "birbirine",
    "birbirini",
    "kim",
    "kimi",
    "kime",
    "kimse",
    "ne",
    "neyi",
    "neden",
    "niçin",
    "niye",
    "nasıl",
    "nerede",
    "nereye",
    "nereden",
    "hangi",
    "kaç",
    "biri",
    "birisi",
    "hepsi",
    "herkes",
    // determiners and quantifiers
    "bir",
    "her",
    "hiç",
    "bazı",
    "birçok",
    "birkaç",
    "çok",
    "az",
    "daha",
    "en",
    "pek",
    "gayet",
    "tüm",
    "bütün",
    "hep",
    "şey",
    "kadar",
    // adverbial particles
    "artık",
    "yine",
    "gene",
    "sadece",
    "yalnızca",
    "yalnız",
    "zaten",
    "belki",
    "acaba",
    "böyle",
    "şöyle",
    "öyle",
    "işte",
    "tabii",
    "hâlâ",
    "henüz",
    "bile",
    "ancak",
    "aslında",
];

/// The cliché categories the Turkish measurement found to be high-precision
/// (`research/2026-10-08-turkish-mgt-detection-2025.md` §5: ≥ 5% of AI texts vs ≤ 0.4%
/// of human texts).
const CLICHE_CATEGORIES: &[ClicheCategory] = &[
    ClicheCategory::ConclusionHedge,
    ClicheCategory::RhetoricalCalque,
    ClicheCategory::IndefiniteArticleInflation,
];

/// Cheap gate in front of [`ClicheMatcher::find_all`], which scans the whole cliché
/// catalogue (~25 literals, ~14 Unicode regexes) and would otherwise cost about two thirds
/// of the extraction time. The gate is a *necessary* condition for any match in
/// [`CLICHE_CATEGORIES`], checked on the word-token stream, so the count is unchanged:
/// - adjective + `bir` + noun trigram (`rol_oynamaktadir`, `onem_tasimaktadir`,
///   `oneme_sahiptir`, `vazgecilmez`, `sifat_bir_isim_enflasyonu`);
/// - a single anchor word (`göz ardı edilmemelidir`, `unutulmamalıdır ki`,
///   `özetlemek gerekirse`, `… ile sınırlı kalmayıp`);
/// - co-occurring anchors (`sadece … değil, aynı zamanda / bununla birlikte`,
///   `… olmakla birlikte, … niteliğini taşımaktadır`, `… ötesinde, … boyutu`).
///
/// Anchors are written with `ı` folded to `i` and compared the same way, because the
/// regexes' `(?i)` treats ASCII `I` as `i`. **Keep in sync with `style/cliches.rs`**; the
/// test `cliche_gate_matches_cliche_matcher` checks every pattern.
#[derive(Default)]
struct ClicheGate<'a> {
    prev2: &'a str,
    prev1: &'a str,
    hit: bool,
    flags: u8,
}

const GATE_ADJ: &[&str] = &[
    "kritik",
    "önemli",
    "büyük",
    "belirleyici",
    "temel",
    "hayati",
    "özel",
    "merkezi",
    "stratejik",
    "kapsamli",
    "başarili",
    "etkili",
    "güçlü",
    "vazgeçilmez",
];
const GATE_NOUN: &[&str] = &[
    "adim",
    "rol",
    "aşama",
    "süreç",
    "unsur",
    "faktör",
    "örnek",
    "çalişma",
    "katki",
    "dönüm",
    "önem",
    "öneme",
    "parça",
    "hale",
    "parçasidir",
    "öğe",
];
const GATE_SINGLE: &[&str] = &["edilmemelidir", "unutulmamalidir", "özetlemek", "kalmayip"];
/// `(anchor, flag bit)` for anchors that only count together.
const GATE_FLAGS: &[(&str, u8)] = &[
    ("sadece", 1),
    ("yalnizca", 1),
    ("salt", 1),
    ("değil", 2),
    ("zamanda", 4),
    ("bununla", 4),
    ("olmakla", 8),
    ("niteliğini", 16),
    ("ötesinde", 32),
    ("boyutu", 64),
];

/// `word == anchor` with `ı` read as `i`.
fn eq_fold_i(word: &str, anchor: &str) -> bool {
    word.chars()
        .map(|c| if c == 'ı' { 'i' } else { c })
        .eq(anchor.chars())
}

fn any_fold_i(word: &str, anchors: &[&str]) -> bool {
    anchors.iter().any(|a| eq_fold_i(word, a))
}

impl<'a> ClicheGate<'a> {
    fn push(&mut self, w: &'a str) {
        if !self.hit {
            let trigram = eq_fold_i(self.prev1, "bir")
                && any_fold_i(w, GATE_NOUN)
                && any_fold_i(self.prev2, GATE_ADJ);
            self.hit = trigram || any_fold_i(w, GATE_SINGLE);
            for (anchor, bit) in GATE_FLAGS {
                if eq_fold_i(w, anchor) {
                    self.flags |= bit;
                }
            }
        }
        self.prev2 = self.prev1;
        self.prev1 = w;
    }

    fn open(&self) -> bool {
        let all = |m: u8| self.flags & m == m;
        self.hit || all(1 | 2 | 4) || all(8 | 16) || all(32 | 64)
    }
}

static FUNCTION_SET: LazyLock<AHashSet<&'static str>> =
    LazyLock::new(|| FUNCTION_WORDS.iter().copied().collect());

/// Placeholders in original and Turkish-lower-cased spelling.
static MASKS: LazyLock<Vec<String>> = LazyLock::new(|| {
    let mut v = Vec::new();
    for m in [
        MASK_YEAR,
        MASK_NUMBER,
        MASK_PROPER_NOUN,
        MASK_URL,
        MASK_EMAIL,
    ] {
        v.push(m.to_string());
        let lower = to_turkish_lower(m);
        if lower != m {
            v.push(lower);
        }
    }
    v
});

// ------------------------------------------------------------------- tokens

#[derive(Clone, Copy)]
enum Tok<'a> {
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
fn scan_tokens<'a>(text: &'a str, mut f: impl FnMut(Tok<'a>)) {
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

/// Closing quotes and brackets the sentence segmenter keeps after a sentence end.
const SENTENCE_CLOSERS: &[char] = &['"', '\'', ')', ']', '»', '”', '’'];

/// Whether a sentence ends in `.`, `!`, `?` or `…`, after closing quotes and brackets.
fn ends_with_terminal(sentence: &str) -> bool {
    sentence
        .trim_end()
        .trim_end_matches(SENTENCE_CLOSERS)
        .ends_with(['.', '!', '?', '…'])
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
fn lower_text(text: &str) -> String {
    let lower = to_turkish_lower(text);
    if lower.contains('\u{0307}') {
        lower.replace('\u{0307}', "")
    } else {
        lower
    }
}

// ------------------------------------------------------------------- matchers

fn lex_hit(word: &str, entries: &[LexEntry]) -> bool {
    entries.iter().any(|e| match e.mode {
        Match::Exact => word == e.word,
        Match::Prefix => word.starts_with(e.word),
        Match::QuestionClitic => is_question_clitic(word),
    })
}

const Q_TAILS: &[&str] = &[
    "", "sın", "sin", "sun", "sün", "sınız", "siniz", "sunuz", "sünüz", "yım", "yim", "yum", "yüm",
    "yız", "yiz", "yuz", "yüz", "ydı", "ydi", "ydu", "ydü", "ydım", "ydim", "ydum", "ydüm", "ydın",
    "ydin", "ydun", "ydün", "ydık", "ydik", "yduk", "ydük", "ymış", "ymiş", "ymuş", "ymüş", "yken",
];

fn is_question_clitic(word: &str) -> bool {
    let Some(rest) = word.strip_prefix('m') else {
        return false;
    };
    let mut it = rest.chars();
    match it.next() {
        Some('ı' | 'i' | 'u' | 'ü') => Q_TAILS.contains(&it.as_str()),
        _ => false,
    }
}

const HIGH_VOWELS: [char; 4] = ['ı', 'i', 'u', 'ü'];

fn is_vowel(c: char) -> bool {
    matches!(
        c,
        'a' | 'e' | 'ı' | 'i' | 'o' | 'ö' | 'u' | 'ü' | 'â' | 'î' | 'û'
    )
}

const MIS: [&str; 4] = ["mış", "miş", "muş", "müş"];
const MIS_TAILS: &[&str] = &[
    "", "tır", "tir", "tur", "tür", "tı", "ti", "tu", "tü", "sın", "sin", "sun", "sün", "ım", "im",
    "um", "üm", "ız", "iz", "uz", "üz", "sınız", "siniz", "sunuz", "sünüz", "lar", "ler", "lardır",
    "lerdir", "sa", "se", "ken", "çasına", "çesine",
];
/// Words ending in `mIş` that are not the suffix.
const MIS_EXCEPTIONS: &[&str] = &["yetmiş", "altmış", "kamış", "yemiş", "gümüş", "dümüş"];

/// `-mIş` (evidential / perfect participle) with an optional person, copula, past,
/// conditional or `-ken` ending, after a stem of at least 2 characters.
///
/// Precision limits: lexicalised `-mIş` words not in [`MIS_EXCEPTIONS`] count
/// (`kıymış`), participles followed by case endings (`gelmişi`) do not.
fn is_mis(w: &str) -> bool {
    if !w.contains('ş') {
        return false;
    }
    MIS_TAILS.iter().any(|tail| {
        w.strip_suffix(tail).is_some_and(|base| {
            MIS.iter().any(|m| {
                base.strip_suffix(m).is_some_and(|stem| {
                    stem.chars().count() >= 2 && !MIS_EXCEPTIONS.contains(&base)
                })
            })
        })
    })
}

const MAKTA: [&str; 2] = ["makta", "mekte"];
const MAKTA_TAILS: &[&str] = &[
    "", "dır", "dir", "yım", "yim", "sın", "sin", "yız", "yiz", "sınız", "siniz", "lar", "ler",
    "dırlar", "dirler", "ydı", "ydi", "ymış", "ymiş", "yken", "ysa", "yse",
];

/// `-mAktA` progressive, usually `-mAktAdIr` (`görülmektedir`).
///
/// Precision limits: practically none known in running text; recall misses `-mAktAn`
/// ("from …ing", which is an ablative, not the progressive) on purpose.
fn is_makta(w: &str) -> bool {
    if !(w.contains("makta") || w.contains("mekte")) {
        return false;
    }
    MAKTA_TAILS.iter().any(|tail| {
        w.strip_suffix(tail).is_some_and(|base| {
            MAKTA.iter().any(|m| {
                base.strip_suffix(m)
                    .is_some_and(|stem| stem.chars().count() >= 2)
            })
        })
    })
}

type Chars = SmallVec<[char; 32]>;

fn starts_with_at(chars: &[char], at: usize, s: &str) -> bool {
    (at..).zip(s.chars()).all(|(k, c)| chars.get(k) == Some(&c))
}

/// `-DIk` participle / nominaliser: `D I ğ` + vowel (`geldiği`, `olduğunu`) or
/// `D I k` + `çA / tAn / lAr` (`geldikçe`, `geldikten`, `yaptıkları`), with at least two
/// characters of stem before it.
///
/// Precision limits: a bare `-DIk` (`geldik`) is the 1st person plural past and is not
/// counted; stems that happen to contain `dığ`/`tuk` + vowel (`tutukla-`) count; the
/// future participle `-AcAğ-` is not included.
fn is_dik(chars: &[char]) -> bool {
    let n = chars.len();
    (2..n.saturating_sub(2)).any(|p| {
        if !matches!(chars[p], 'd' | 't') || !HIGH_VOWELS.contains(&chars[p + 1]) {
            return false;
        }
        match chars[p + 2] {
            'ğ' => chars.get(p + 3).is_some_and(|&c| is_vowel(c)),
            'k' => ["ça", "çe", "tan", "ten", "lar", "ler"]
                .iter()
                .any(|t| starts_with_at(chars, p + 3, t)),
            _ => false,
        }
    })
}

/// Verbal suffixes that may follow a passive marker.
const PASSIVE_TAILS: &[&str] = &[
    "mış", "miş", "muş", "müş", "dı", "di", "du", "dü", "ır", "ir", "ur", "ür", "acak", "ecek",
    "acağ", "eceğ", "makta", "mekte", "mak", "mek", "ma", "me", "abil", "ebil", "dığ", "diğ",
    "duğ", "düğ", "dık", "dik", "duk", "dük", "ıp", "ip", "up", "üp", "arak", "erek", "ınca",
    "ince", "unca", "ünce", "malı", "meli", "sa", "se",
];
/// Endings allowed after the participle `-An` (`yapılan`, `yapılanlar`, `verilene`).
const AN_TAILS: &[&str] = &[
    "", "lar", "ler", "ı", "i", "a", "e", "da", "de", "dan", "den", "ın", "in", "la", "le",
];
/// Verb stems ending in `-lA` whose following `n` is part of the root (`kullan-`).
const N_STEM_EXCEPTIONS: &[&str] = &["kulla"];

fn passive_tail_at(chars: &[char], at: usize) -> bool {
    if PASSIVE_TAILS.iter().any(|t| starts_with_at(chars, at, t)) {
        return true;
    }
    for an in ["an", "en"] {
        if starts_with_at(chars, at, an) {
            let rest: String = chars[at + 2..].iter().collect();
            if AN_TAILS.contains(&rest.as_str()) {
                return true;
            }
        }
    }
    false
}

/// Passive verb form, surface heuristic: stem (≥ 2 chars) + passive marker + verbal
/// suffix from [`PASSIVE_TAILS`] or the participle `-An` (+ plural/case). Markers:
/// `-Il-` after a consonant (`yap-ıl-mış`), `-In-` after `l` (`bil-in-en`), and `-n-`
/// after a `-lA` stem (`söyle-n-di`, `düzenle-n-en`).
///
/// Precision limits: reflexive `-lAn-`/`-In-` verbs (`dinlenen`, `bulunan`,
/// `yararlanan`) are counted as passive; roots ending in consonant + `Il` (`çekil-`,
/// `yayıl-`) also count; impersonal passives are not distinguished from personal ones;
/// passives of vowel-final stems other than `-lA` (`oku-n-an`) are missed.
fn is_passive(chars: &[char]) -> bool {
    let n = chars.len();
    for p in 2..n {
        let c = chars[p];
        let next = chars.get(p + 1).copied();
        // -Il- after a consonant
        if HIGH_VOWELS.contains(&c)
            && next == Some('l')
            && !is_vowel(chars[p - 1])
            && passive_tail_at(chars, p + 2)
        {
            return true;
        }
        // -In- after l
        if HIGH_VOWELS.contains(&c)
            && next == Some('n')
            && chars[p - 1] == 'l'
            && passive_tail_at(chars, p + 2)
        {
            return true;
        }
        // -n- after a -lA stem
        if c == 'n'
            && p >= 3
            && matches!(chars[p - 1], 'a' | 'e')
            && chars[p - 2] == 'l'
            && passive_tail_at(chars, p + 1)
        {
            let stem: String = chars[..p].iter().collect();
            if !N_STEM_EXCEPTIONS.iter().any(|s| stem.ends_with(s)) {
                return true;
            }
        }
    }
    false
}

// ------------------------------------------------------------------- OOV proxy

/// Root lexicon: column 1 (surface lemma) and column 2 (stem) of the Zemberek lexicon,
/// lower-cased with Turkish rules. Loaded once.
static ROOTS: LazyLock<AHashSet<String>> = LazyLock::new(|| {
    let lex = include_str!("../morphology/zemberek_lexicon.txt");
    let mut set = AHashSet::with_capacity(120_000);
    for line in lex.lines() {
        let mut cols = line.split_whitespace();
        for w in [cols.next(), cols.next()].into_iter().flatten() {
            set.insert(to_turkish_lower(w));
        }
    }
    set
});

/// Verb stems of the Zemberek lexicon (column 2 of the `Verb` rows), lower-cased with
/// Turkish rules. Only used to tell `-lAr` plurals from aorists (see [`is_aorist`]).
static VERBS: LazyLock<AHashSet<String>> = LazyLock::new(|| {
    let lex = include_str!("../morphology/zemberek_lexicon.txt");
    let mut set = AHashSet::with_capacity(8_000);
    for line in lex.lines() {
        let mut cols = line.split_whitespace();
        if let (Some(_), Some(stem), Some("Verb")) = (cols.next(), cols.next(), cols.next()) {
            set.insert(to_turkish_lower(stem));
        }
    }
    set
});

/// Surface suffix templates. Archiphonemes: `I` = ı/i/u/ü, `A` = a/e, `D` = d/t,
/// `C` = c/ç, `K` = k/ğ. Harmony between morphs is not enforced.
const SUFFIX_TEMPLATES: &[&str] = &[
    // number, possession, case
    "lAr", "lArI", "Im", "m", "In", "n", "I", "sI", "ImIz", "mIz", "InIz", "nIz", "yI", "nI", "A",
    "yA", "nA", "DA", "nDA", "DAn", "nDAn", "nIn", "lA", "ylA", "CA", "ki", "ken",
    "yken", // copula and person
    "DIr", "yIm", "sIn", "Iz", "yIz", "sInIz", "k", "lAr",
    // tense, aspect, mood, negation
    "DI", "mIş", "Ar", "Ir", "r", "AcAK", "yAcAK", "Iyor", "yor", "mAktA", "mAlI", "sA", "ysA",
    "mA", "mAz", "mAm", "mI", "yAbil", "Abil", "yA", // voice
    "Il", "l", "Iş", "t", "It", "DIr", "Ar", // verbal nouns, participles, converbs
    "mAK", "yIş", "An", "yAn", "DIK", "Ip", "yIp", "ArAk", "yArAk", "IncA", "yIncA", "mAdAn",
    "DIkçA", "CAsInA", // derivation
    "lIK", "lI", "sIz", "CI", "sAl", "lAş", "lAn", "CIK", "gI", "gIn", "IcI", "yIcI", "IntI", "Im",
    "ImsI", "DAş", "lAt",
];

fn expand_template(t: &str, out: &mut AHashSet<String>) {
    let mut acc: Vec<String> = vec![String::new()];
    for c in t.chars() {
        let options: &[char] = match c {
            'I' => &['ı', 'i', 'u', 'ü'],
            'A' => &['a', 'e'],
            'D' => &['d', 't'],
            'C' => &['c', 'ç'],
            'K' => &['k', 'ğ'],
            _ => &[],
        };
        if options.is_empty() {
            for s in &mut acc {
                s.push(c);
            }
        } else {
            acc = acc
                .iter()
                .flat_map(|s| {
                    options.iter().map(move |&o| {
                        let mut s2 = s.clone();
                        s2.push(o);
                        s2
                    })
                })
                .collect();
        }
    }
    out.extend(acc);
}

static SUFFIXES: LazyLock<(AHashSet<String>, usize)> = LazyLock::new(|| {
    let mut set = AHashSet::new();
    for t in SUFFIX_TEMPLATES {
        expand_template(t, &mut set);
    }
    let max_chars = set.iter().map(|s| s.chars().count()).max().unwrap_or(0);
    (set, max_chars)
});

/// Reverses final-consonant softening of a stem: `kitab` → `kitap`, `ağac` → `ağaç`,
/// `renğ`/`çocuğ` → `-k`.
fn unsoften(stem: &str) -> Option<String> {
    let last = stem.chars().next_back()?;
    let hard = match last {
        'b' => 'p',
        'c' => 'ç',
        'd' => 't',
        'ğ' | 'g' => 'k',
        _ => return None,
    };
    let mut s = stem[..stem.len() - last.len_utf8()].to_string();
    s.push(hard);
    Some(s)
}

/// Whether a lower-cased word is explained by the root lexicon: either it is a root, or a
/// prefix of at least 2 characters is a root (directly or after reversing consonant
/// softening) and the rest splits into suffixes from [`SUFFIX_TEMPLATES`].
///
/// Precision limits: vowel dropping (`ağzı`, `burnu`), consonant doubling (`hakkı`) and
/// unlisted derivations count as OOV; the suffix chain ignores ordering and harmony, so
/// many misspellings that look like root + suffixes (`tabiki` = `tabi` + `ki`) pass.
fn is_known(word: &str) -> bool {
    let roots = &*ROOTS;
    if roots.contains(word) {
        return true;
    }
    let (suffixes, max_chars) = &*SUFFIXES;
    // Char boundaries of `word`, including the end.
    let bounds: SmallVec<[usize; 32]> = word
        .char_indices()
        .map(|(i, _)| i)
        .chain(std::iter::once(word.len()))
        .collect();
    let n = bounds.len() - 1;
    if n < 3 {
        return false;
    }
    // ok[k]: word[bounds[k]..] is a (possibly empty) chain of suffixes.
    let mut ok: SmallVec<[bool; 32]> = SmallVec::from_elem(false, n + 1);
    ok[n] = true;
    for k in (2..n).rev() {
        ok[k] = (1..=*max_chars).any(|len| {
            k + len <= n && ok[k + len] && suffixes.contains(&word[bounds[k]..bounds[k + len]])
        });
    }
    (2..n).rev().any(|k| {
        if !ok[k] {
            return false;
        }
        let stem = &word[..bounds[k]];
        roots.contains(stem) || unsoften(stem).is_some_and(|s| roots.contains(&s))
    })
}

// ------------------------------------------------------------------- tense

/// Person/number endings after a positive aorist (`gelir-im`, `gelir-sin`, `gelir-iz`,
/// `gelir-siniz`, `gelir-ler`).
const AORIST_TAILS: &[&str] = &[
    "", "ım", "im", "um", "üm", "sın", "sin", "sun", "sün", "ız", "iz", "uz", "üz", "sınız",
    "siniz", "sunuz", "sünüz", "lar", "ler",
];
/// Endings after the negative aorist `-mAz` (`gelmez`, `gelmezsin`, `gelmezsiniz`,
/// `gelmezler`).
const NEG_AORIST_TAILS: &[&str] = &["", "sın", "sin", "sınız", "siniz", "lar", "ler"];
/// Negative aorist persons without `-mAz`: 1sg `-mAm` (`gelmem`), 1pl `-mAyIz` (`gelmeyiz`).
const NEG_AORIST_FORMS: &[&str] = &["mam", "mem", "mayız", "meyiz"];
/// Person endings after the progressive `-Iyor`.
const IYOR_TAILS: &[&str] = &["", "um", "sun", "uz", "sunuz", "lar"];
/// The potential `-(y)AbIl`, removed before the verb-stem lookup (`ol-abil-ir`).
const POTENTIAL: &[&str] = &["yabil", "yebil", "abil", "ebil"];
/// Common nouns, adjectives, interjections and postpositions that end like an aorist.
/// Compared with the whole word and with the base before a person ending (`hazır-ım`).
/// Caveat (D-005): the entries were picked by looking at false positives of the surface
/// rule among sentence-final words of the pooled train split (both labels together, not
/// ranked by label); they are linguistic non-verbs, not label-associated words.
const AORIST_EXCEPTIONS: &[&str] = &[
    "ağır",
    "anahtar",
    "bahar",
    "beraber",
    "biber",
    "diğer",
    "fakir",
    "fikir",
    "haber",
    "hamam",
    "hayir",
    "hayır",
    "hazır",
    "huzur",
    "kadar",
    "karar",
    "misafir",
    "miktar",
    "naber",
    "namaz",
    "pazar",
    "pekmez",
    "peynir",
    "rehber",
    "sabır",
    "sefer",
    "selamlar",
    "sıfır",
    "şair",
    "şehir",
    "şeker",
    "şiir",
    "şükür",
    "tamam",
    "tasar",
    "tekrar",
    "tesekkur",
    "teşekkür",
    "zafer",
    "zarar",
];

/// Tense of a sentence-final predicate, for `aorist_ratio`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tense {
    Aorist,
    Progressive,
}

fn is_voiceless(c: char) -> bool {
    matches!(c, 'p' | 'ç' | 't' | 'k' | 's' | 'ş' | 'h' | 'f')
}

/// Present progressive `-Iyor` with an optional person ending (`bakıyor`, `diyor`,
/// `geliyorum`, `gelmiyorlar`): at least two characters ending in a high vowel before
/// `yor`, so `yorum` does not count.
///
/// Precision limits: practically none known; the past/evidential/conditional compounds
/// (`geliyordu`, `geliyorsa`) and `-Iyor` + copula (`geliyordur`) are not counted.
fn is_iyor(w: &str) -> bool {
    if !w.contains("yor") {
        return false;
    }
    IYOR_TAILS.iter().any(|tail| {
        w.strip_suffix(tail)
            .and_then(|b| b.strip_suffix("yor"))
            .is_some_and(|stem| {
                stem.chars().count() >= 2
                    && stem
                        .chars()
                        .next_back()
                        .is_some_and(|c| HIGH_VOWELS.contains(&c))
            })
    })
}

/// Whether the positive-aorist base `b` (… vowel + `r`) has a verb stem in [`VERBS`]: `b`
/// minus `r` (`sağla-r`, `etkile-r`) or minus vowel + `r` (`gel-ir`, `böl-er`), optionally
/// after removing the potential `-(y)AbIl` (`ol-abil-ir`) and reversing final-consonant
/// softening (`ed-er` → `et`).
fn has_verb_stem(b: &str) -> bool {
    let verbs = &*VERBS;
    let known = |s: &str| verbs.contains(s) || unsoften(s).is_some_and(|h| verbs.contains(&h));
    let Some(no_r) = b.strip_suffix('r') else {
        return false;
    };
    let no_vr = no_r
        .char_indices()
        .next_back()
        .map_or("", |(i, _)| &no_r[..i]);
    [no_r, no_vr].into_iter().any(|stem| {
        known(stem)
            || POTENTIAL
                .iter()
                .any(|p| stem.strip_suffix(p).is_some_and(known))
    })
}

/// Positive aorist base (the word without its person ending): at least 4 characters ending
/// in `a/e/ı/i/u/ü` + `r`, not the copula `-DIr`, not in [`AORIST_EXCEPTIONS`]. A base that
/// ends in `-lAr` (plural or a `-lA` verb: `kitaplar` / `sağlar`) or carries the 3pl ending
/// (`gelir-ler` / `fikir-ler`) must have a verb stem ([`has_verb_stem`]).
fn is_positive_aorist_base(b: &str, tail: &str) -> bool {
    let c: Chars = b.chars().collect();
    let n = c.len();
    if n < 4 || c[n - 1] != 'r' || !matches!(c[n - 2], 'a' | 'e' | 'ı' | 'i' | 'u' | 'ü') {
        return false;
    }
    if AORIST_EXCEPTIONS.contains(&b) {
        return false;
    }
    // Copula -DIr: d + high vowel (önemli-dir, var-dır), or t after a voiceless consonant
    // (sahip-tir, yok-tur, gelmiş-tir). -tIr after a vowel or a voiced consonant is the
    // aorist of a t-final stem (anlat-ır, belirt-ir).
    if HIGH_VOWELS.contains(&c[n - 2])
        && (c[n - 3] == 'd' || (c[n - 3] == 't' && is_voiceless(c[n - 4])))
    {
        return false;
    }
    let plural_like = c[n - 3] == 'l' && matches!(c[n - 2], 'a' | 'e');
    if plural_like || tail == "lar" || tail == "ler" {
        return has_verb_stem(b);
    }
    true
}

/// Aorist predicate, surface heuristic: positive `-Ir/-Ar/-r` with an optional person
/// ending (`gelir`, `bakar`, `okur`, `olabilir`, `ederim`, `gelirler`, `sağlar`), negative
/// `-mAz` (`gelmez`, `alamazsın`, `gelmezler`), `-mAm` (`gelmem`) and `-mAyIz`.
///
/// Precision limits: nouns and adjectives ending in vowel + `r` count unless they are in
/// [`AORIST_EXCEPTIONS`] (`gelir` "income" and `gider` "expense" count); a bare verb stem
/// ending in `r` plus a possessive (`oturum` "session") counts; plurals whose stem + `lA`
/// is a verb (`işler`, `taşlar`) count; `-tIr` after a voiceless consonant is read as the
/// copula, so `çıkart-ır` is missed; 3pl aorists whose stem is not in the lexicon are
/// missed; bases shorter than 4 characters (`der`, `yer`, `var`, `bir`) and the
/// past/conditional compounds (`gelirdi`, `gelirse`) never count.
fn is_aorist(w: &str) -> bool {
    if AORIST_EXCEPTIONS.contains(&w) || is_iyor(w) {
        return false;
    }
    let stem_ok = |stem: &str| stem.chars().count() >= 2;
    let negative = NEG_AORIST_TAILS.iter().any(|tail| {
        w.strip_suffix(tail).is_some_and(|b| {
            ["maz", "mez"]
                .iter()
                .any(|m| b.strip_suffix(m).is_some_and(stem_ok))
        })
    }) || NEG_AORIST_FORMS
        .iter()
        .any(|m| w.strip_suffix(m).is_some_and(stem_ok));
    negative
        || AORIST_TAILS.iter().any(|tail| {
            w.strip_suffix(tail)
                .is_some_and(|b| is_positive_aorist_base(b, tail))
        })
}

/// Tense of a sentence's final predicate, given its last two tokens: the last token, or
/// the word before a final
/// question clitic (`Gelir misin?` → `gelir`). `None` when that token is a mask or neither
/// an aorist nor a progressive form (past, future, `-mIş`, copula `-DIr`, `-mAktA`, nominal
/// sentences, headings, …).
fn predicate_tense(last: Option<Tok>, prev: Option<Tok>) -> Option<Tense> {
    let word = |t: Option<Tok>| match t {
        Some(Tok::Word(w)) => Some(lower_text(w)),
        _ => None,
    };
    let mut w = word(last)?;
    if is_question_clitic(&w) {
        w = word(prev)?;
    }
    if is_aorist(&w) {
        Some(Tense::Aorist)
    } else if is_iyor(&w) {
        Some(Tense::Progressive)
    } else {
        None
    }
}

// ------------------------------------------------------------------- label colons

/// Most words a label may have (`Sonuç`, `Adım 1`, `Gereç ve yöntem`).
const MAX_LABEL_WORDS: usize = 4;

/// Cataphoric words: a colon after them introduces the clause or list they point to
/// (`Şunu söyledi:`, `Sorun şu:`, `Dedi ki:`, `İşte bazı öneriler:`), so it stays content
/// punctuation even after a short span.
const CATAPHORIC: &[&str] = &[
    "aşağıdaki",
    "aşağıdakiler",
    "aşağıdakileri",
    "aşağıdakilerdir",
    "işte",
    "ki",
    "şu",
    "şudur",
    "şuna",
    "şunda",
    "şundan",
    "şunlar",
    "şunlara",
    "şunları",
    "şunlardır",
    "şunu",
    "şunun",
    "şuydu",
    "şöyle",
    "şöyledir",
    "şöyleydi",
];

/// Person endings after the past `-DI` (`geldi-m`, `geldi-niz`, `geldi-ler`).
const PAST_TAILS: &[&str] = &["", "m", "n", "k", "nız", "niz", "nuz", "nüz", "lar", "ler"];

/// Past-tense verb `-DI` (+ person), told apart from a noun with a possessive by the root
/// lexicon: the word is not a root, and the word without its final vowel is not a non-verb
/// root, directly or after reversing softening (`fiyat-ı`, `ad-ı`, `adım` are nouns;
/// `dedi`, `söyledi`, `düşünüldü` are verbs, even though `söylet` is a verb root).
///
/// Precision limits: possessive nouns whose stem is not a root count as verbs (`kaydı`,
/// from `kayıt`), and so do nouns whose stem is also a verb (`tadı`, from `tat`); verbs
/// whose form minus the vowel is a noun root do not (`oldu`: `old`, `aldı`: `alt`).
fn is_past(w: &str) -> bool {
    let roots = &*ROOTS;
    if roots.contains(w) {
        return false;
    }
    let verbs = &*VERBS;
    let noun_root = |s: &str| roots.contains(s) && !verbs.contains(s);
    PAST_TAILS.iter().any(|tail| {
        w.strip_suffix(tail).is_some_and(|b| {
            let mut rev = b.chars().rev();
            let (Some(v), Some(d)) = (rev.next(), rev.next()) else {
                return false;
            };
            if !HIGH_VOWELS.contains(&v) || !matches!(d, 'd' | 't') || b.chars().count() < 4 {
                return false;
            }
            let noun = &b[..b.len() - v.len_utf8()];
            !(noun_root(noun) || unsoften(noun).is_some_and(|s| noun_root(&s)))
        })
    })
}

/// Whether the text before a colon, from the start of its line or sentence, is a label:
/// 1 to [`MAX_LABEL_WORDS`] whitespace-separated tokens with at least one letter or digit,
/// no [`CATAPHORIC`] word, and a last word that is not a finite past (`-DI`), progressive
/// (`-Iyor`) or `-mAktA` verb (`Ali dedi:`, `Gereği düşünüldü:`). A mask as the last token
/// (`Adım <SAYI>`) is fine.
fn is_label(span: &str) -> bool {
    let n = span.split_whitespace().count();
    if n == 0 || n > MAX_LABEL_WORDS || !span.chars().any(char::is_alphanumeric) {
        return false;
    }
    let lower = lower_text(span);
    let mut cataphoric = false;
    let mut last_word: Option<&str> = None;
    scan_tokens(&lower, |tok| match tok {
        Tok::Word(w) => {
            cataphoric |= CATAPHORIC.contains(&w);
            last_word = Some(w);
        }
        Tok::Mask => last_word = None,
    });
    !cataphoric && !last_word.is_some_and(|w| is_past(w) || is_iyor(w) || is_makta(w))
}

/// Counts the colons that `colons_per_100w` looks at (`:` followed by whitespace or the end
/// of the text) and how many of them end a line- or sentence-leading label
/// (`Adım 1: …`, `Örnek: …`, `… bitti. Sonuç: …`). Returns `(colons, label_colons)`.
///
/// A colon is a label colon when it is the first such colon of its sentence
/// ([`SentenceSegmenter`] run on each line; the line start counts as a sentence start) and
/// the text from the sentence start to the colon passes [`is_label`]. Features run on
/// preprocessed text, where list markers and emphasis are already stripped, so
/// `- **Adım 1:** …` arrives here as `Adım <SAYI>: …` (D-005).
///
/// Precision limits: short imperative or aorist clauses (`Bakın:`, `Hoca der:`) and short
/// noun-phrase lead-ins (`Cevap basit:`) count as labels; see [`is_past`] for the past-tense
/// check; a label glued to the previous sentence without a space (`bitti.Not:`) counts only
/// if the whole span is at most [`MAX_LABEL_WORDS`] tokens.
pub(crate) fn label_colon_counts(text: &str) -> (usize, usize) {
    let (mut colons, mut labels) = (0usize, 0usize);
    for line in text.split('\n') {
        let found: SmallVec<[usize; 8]> = line
            .match_indices(':')
            .map(|(i, _)| i)
            .filter(|&i| line[i + 1..].chars().next().is_none_or(char::is_whitespace))
            .collect();
        if found.is_empty() {
            continue;
        }
        colons += found.len();
        let starts: SmallVec<[usize; 8]> = SentenceSegmenter::segment(line)
            .iter()
            .map(|s| s.start)
            .filter(|&s| line.is_char_boundary(s))
            .collect();
        let mut prev_unit = None;
        for i in found {
            let unit = starts
                .iter()
                .copied()
                .filter(|&s| s <= i)
                .max()
                .unwrap_or(0);
            if prev_unit == Some(unit) {
                continue;
            }
            prev_unit = Some(unit);
            labels += usize::from(is_label(&line[unit..i]));
        }
    }
    (colons, labels)
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

/// Ateşman (1997) readability: 198.825 − 40.175 · syllables/word − 2.610 · words/sentence.
fn atesman(syllables: usize, n_words: usize, n_sent: usize) -> f64 {
    if n_words == 0 || n_sent == 0 {
        return 0.0;
    }
    198.825
        - 40.175 * (syllables as f64 / n_words as f64)
        - 2.610 * (n_words as f64 / n_sent as f64)
}

// ------------------------------------------------------------------- extraction

/// Computes all content features of one preprocessed document.
pub fn content_features(clean: &str) -> ContentFeatures {
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
        syllables += chars.iter().filter(|&&c| is_vowel(c)).count().max(1);
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

    // Sentence and paragraph structure, on the original text.
    let mut sent_lens: Vec<usize> = Vec::new();
    let mut para_lens: Vec<usize> = Vec::new();
    let (mut questions, mut exclamations) = (0usize, 0usize);
    let (mut aorist, mut progressive) = (0usize, 0usize);
    let (mut lower_starts, mut no_terminal) = (0usize, 0usize);
    for para in clean.split("\n\n") {
        let mut para_len = 0usize;
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
                para_len += len;
                questions += usize::from(s.text.contains('?'));
                exclamations += usize::from(s.text.contains('!'));
                lower_starts += usize::from(scan.starts_lower());
                no_terminal += usize::from(!ends_with_terminal(s.text));
                match predicate_tense(scan.last, scan.prev) {
                    Some(Tense::Aorist) => aorist += 1,
                    Some(Tense::Progressive) => progressive += 1,
                    None => {}
                }
            }
        }
        if para_len > 0 {
            para_lens.push(para_len);
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
        ClicheMatcher::find_all(clean)
            .iter()
            .filter(|m| CLICHE_CATEGORIES.contains(&m.category))
            .count()
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
        para_len_cv: cv(&para_lens),
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
        aorist_ratio: rate(aorist, aorist + progressive, 1.0),
        function_words_per_100w: fw.map(per100),
        lower_start_share: rate(lower_starts, n_sent, 1.0),
        no_terminal_punct_share: rate(no_terminal, n_sent, 1.0),
        short_word_share: rate(short, n_words, 1.0),
        long_word_share: rate(long, n_words, 1.0),
        multi_excl: rate(multi_excl, n_sent, 100.0),
        interrobang: rate(interrobang, n_sent, 100.0),
        ellipsis: rate(ellipsis, n_sent, 100.0),
        multi_q: rate(multi_q, n_sent, 100.0),
        sents_per_para_mean: rate(n_sent, para_lens.len(), 1.0),
        word_entropy: word_entropy(&freq, n_words),
        heaps_exponent: heaps_exponent(&seq, n_types),
        compression_ratio: compression_ratio(&lower),
        atesman_readability: atesman(syllables, n_words, n_sent),
    }
}

/// Feature vector of one preprocessed document, in [`FEATURE_NAMES`] order. All values
/// are finite.
pub fn extract_features(clean: &str) -> Vec<f64> {
    content_features(clean).to_array().to_vec()
}

/// [`extract_features`] over many documents, spread over the available CPU cores.
/// Output order matches input order.
pub fn extract_features_batch<S: AsRef<str> + Sync>(cleans: &[S]) -> Vec<Vec<f64>> {
    let run = |chunk: &[S]| -> Vec<Vec<f64>> {
        chunk.iter().map(|t| extract_features(t.as_ref())).collect()
    };
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(cleans.len().div_ceil(16).max(1));
    if threads <= 1 {
        return run(cleans);
    }
    // Force the shared tables once instead of racing on them from every thread.
    LazyLock::force(&ROOTS);
    LazyLock::force(&VERBS);
    LazyLock::force(&SUFFIXES);
    LazyLock::force(&FUNCTION_WORD_INDEX);
    let chunk_size = cleans.len().div_ceil(threads);
    std::thread::scope(|scope| {
        let handles: Vec<_> = cleans
            .chunks(chunk_size)
            .map(|chunk| {
                let handle = std::thread::Builder::new().spawn_scoped(scope, move || run(chunk));
                (chunk, handle)
            })
            .collect();
        let mut out = Vec::with_capacity(cleans.len());
        for (chunk, handle) in handles {
            match handle
                .map_err(|_| ())
                .and_then(|h| h.join().map_err(|_| ()))
            {
                Ok(v) => out.extend(v),
                Err(()) => out.extend(run(chunk)),
            }
        }
        out
    })
}
