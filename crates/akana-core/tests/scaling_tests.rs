//! Public entry points stay linear (or n log n) in the size of their input. Each of these paths
//! once did quadratic or cubic work, so a few hundred KB of crafted or ordinary text took tens
//! of seconds.
//!
//! The checks compare timings instead of bounding them, so they hold in debug and release
//! builds on a slow or busy machine: the input grows 4×, and the time may grow at most
//! [`MAX_RATIO`]× (linear work gives about 4×, quadratic work 16×). Each size is timed
//! best-of-three, interleaved with the other, and the tests in this file run one at a time.

use std::sync::{Arc, LazyLock, Mutex};
use std::time::Instant;

use akana_core::chunking::{SDPMChunker, SemanticChunker, SentenceChunker, ThresholdMode};
use akana_core::embeddings::TurkishEmbeddings;
use akana_core::morphology::{morphotactics_graph, TurkishMorphology};
use akana_core::{TurkishNER, TurkishStyleAuditor, TurkishSummarizer};

/// Most the time may grow when the input grows 4×.
const MAX_RATIO: f64 = 9.0;

/// Timings are only meaningful when the tests of this file do not run at the same time.
static SERIAL: Mutex<()> = Mutex::new(());

static AUDITOR: LazyLock<TurkishStyleAuditor> = LazyLock::new(TurkishStyleAuditor::new);
static EMBEDDINGS: LazyLock<Arc<TurkishEmbeddings>> =
    LazyLock::new(|| Arc::new(TurkishEmbeddings::new()));

fn seconds(run: impl FnOnce()) -> f64 {
    let t = Instant::now();
    run();
    t.elapsed().as_secs_f64()
}

/// Asserts that `run(large)` takes at most `max_ratio` times as long as `run(small)`.
fn assert_growth(what: &str, small: &str, large: &str, max_ratio: f64, run: impl Fn(&str)) {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    run(small); // warm-up: lazily built tables
    let (mut t_small, mut t_large) = (f64::INFINITY, f64::INFINITY);
    for _ in 0..3 {
        t_small = t_small.min(seconds(|| run(small)));
        t_large = t_large.min(seconds(|| run(large)));
    }
    let ratio = t_large / t_small.max(1e-6);
    let report = format!(
        "{what}: {} → {} bytes took {t_small:.4} s → {t_large:.4} s ({ratio:.1}×)",
        small.len(),
        large.len()
    );
    eprintln!("{report}");
    assert!(ratio <= max_ratio, "{report}");
}

/// Asserts that `run` on `input(4 * n)` takes at most [`MAX_RATIO`] times as long as on
/// `input(n)`.
fn assert_linear(what: &str, n: usize, input: impl Fn(usize) -> String, run: impl Fn(&str)) {
    assert_growth(what, &input(n), &input(4 * n), MAX_RATIO, run);
}

fn audit(text: &str) {
    AUDITOR.audit(text);
}

// ------------------------------------------------------------------- AI-text detector

#[test]
fn stacked_chatbot_openers() {
    // Each opener re-segmented the whole first line: 210 KB of "Elbette! " took 37 s.
    assert_linear("Elbette!", 1_500, |n| "Elbette! ".repeat(n), audit);
    // Openers cut at a colon, and a URL that runs over the colon's space.
    assert_linear("Elbette:", 1_500, |n| "Elbette: ".repeat(n), audit);
    assert_linear(
        "URL opener",
        1_000,
        |n| "Elbette http://: ".repeat(n),
        audit,
    );
}

#[test]
fn think_blocks() {
    // The text was searched for <think> again after every </think>: 200 KB took 5 s.
    assert_linear("</think>", 2_000, |n| "</think> ".repeat(n), audit);
    assert_linear(
        "<think> pairs",
        1_000,
        |n| "<think>a</think>b ".repeat(n),
        audit,
    );
}

#[test]
fn unclosed_latex() {
    // Each unclosed \[ or \( searched to the end of the text: 200 KB took 2 s.
    assert_linear("\\[", 3_000, |n| "\\[ a ".repeat(n), audit);
    assert_linear("\\(", 3_000, |n| "\\( a ".repeat(n), audit);
    assert_linear("$$", 3_000, |n| "$$ a ".repeat(n), audit);
    // Each `$` looked for the end of its line.
    assert_linear("$ on one line", 3_000, |n| "$5 ".repeat(n), audit);
}

#[test]
fn long_words_in_detector_morphology() {
    // One 200 KB word took 3.3 s: the passive check copied the rest of the word at every
    // -Il-An, and the OOV check hashed every stem prefix.
    assert_linear("kılan…", 4_000, |n| "kılan".repeat(n), audit);
    assert_linear("kullanmış…", 2_000, |n| "kullanmış".repeat(n), audit);
    assert_linear(
        "evdede…",
        10_000,
        |n| format!("ev{}", "de".repeat(n)),
        audit,
    );
}

// ------------------------------------------------------------------- summarizer

/// `n` sentences over a small vocabulary, so most pairs of sentences share a stem.
fn sentences(n: usize) -> String {
    let words: Vec<&str> = "kitap okul öğrenci öğretmen ders sınav proje rapor şehir yol araba \
                            deniz güneş yağmur orman bilgi bilim program veri model karar \
                            mahkeme dava kanun"
        .split_whitespace()
        .collect();
    let mut x: u64 = 0x2545_f491_4f6c_dd1d;
    let mut s = String::new();
    for i in 0..n {
        s.push_str("Bu");
        for _ in 0..8 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            s.push(' ');
            s.push_str(words[(x % words.len() as u64) as usize]);
        }
        s.push_str(&format!(" {i}. kez oldu. "));
    }
    s
}

#[test]
fn textrank_is_quadratic_not_cubic() {
    // The row sum was recomputed inside the i×j loop: 1,000 sentences took 20 s. The graph is
    // dense here, so TextRank stays quadratic in the number of sentences: 4× the sentences
    // may take 16× as long, where the cubic loop took 64×.
    let summarizer = TurkishSummarizer::new();
    let (small, large) = (sentences(150), sentences(600));
    assert_growth("TextRank", &small, &large, 32.0, |t| {
        summarizer.summarize(t, 3);
    });
}

// ------------------------------------------------------------------- chunkers

/// Short sentences with multi-byte letters, spaced out: the spaces count for the char offsets
/// but cost the tokenizer and the embeddings next to nothing, so the byte → char conversion
/// dominates.
fn short_sentences(n: usize) -> String {
    format!("Şu iş çok güzel.{}", " ".repeat(100)).repeat(n)
}

#[test]
fn sentence_chunker() {
    // Every sentence counted the chars of the whole text before it: 8 M chars took 53 s.
    let chunker = SentenceChunker::with_embeddings(64, 8, 1, Arc::clone(&EMBEDDINGS));
    assert_linear("SentenceChunker", 2_000, short_sentences, |t| {
        chunker.chunk(t);
    });
}

#[test]
fn semantic_and_sdpm_chunkers() {
    let semantic = SemanticChunker::with_embeddings(
        64,
        ThresholdMode::Percentile(0.75),
        Arc::clone(&EMBEDDINGS),
    );
    assert_linear("SemanticChunker", 2_000, short_sentences, |t| {
        semantic.chunk(t);
    });
    let sdpm = SDPMChunker::with_embeddings(
        64,
        ThresholdMode::Percentile(0.75),
        0.5,
        Arc::clone(&EMBEDDINGS),
    );
    assert_linear("SDPMChunker", 2_000, short_sentences, |t| {
        sdpm.chunk(t);
    });
}

// ------------------------------------------------------------------- NER

#[test]
fn ner_capitalised_runs() {
    // The ORG scan walked to the end of the capitalised run from every token: 8,000
    // capitalised words took 2.5 s.
    assert_linear(
        "capitalised run",
        2_000,
        |n| "Zxqy ".repeat(n),
        |t| {
            TurkishNER::extract_entities(t);
        },
    );
    assert_linear(
        "capitalised run with ve",
        1_000,
        |n| "Zxqy ve Qwer ".repeat(n),
        |t| {
            TurkishNER::extract_entities(t);
        },
    );
}

// ------------------------------------------------------------------- morphology

#[test]
fn morphotactics_list_each_edge_once() {
    // A transition listed twice doubles the search below it: every passive or causative
    // suffix of a word doubled the work, and the duplicates had to be deduplicated after.
    let graph = morphotactics_graph();
    let mut seen = std::collections::HashSet::new();
    for t in &graph.transitions {
        assert!(
            seen.insert((t.from, t.to, t.template, format!("{:?}", t.condition))),
            "{} -> {} '{}' is listed twice",
            graph.states[t.from.0 as usize].name,
            graph.states[t.to.0 as usize].name,
            t.template
        );
    }
}

#[test]
fn voice_suffix_chains() {
    // Each passive or causative suffix doubled the search: yap-tır-t-ıl-dır-t-ıl-… took
    // 15–26 ms per word.
    let morph = TurkishMorphology::new();
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let (short, long) = ("yaptırtıldı", "yaptırtıldırtıldırtıldı");
    let time = |w: &str| {
        (0..3)
            .map(|_| {
                seconds(|| {
                    for _ in 0..20 {
                        morph.analyze(w);
                    }
                })
            })
            .fold(f64::INFINITY, f64::min)
    };
    time(short);
    let (t_short, t_long) = (time(short), time(long));
    let ratio = t_long / t_short.max(1e-6);
    assert!(
        ratio <= 16.0,
        "{short} → {long} took {t_short:.4} s → {t_long:.4} s ({ratio:.1}×)"
    );
}

#[test]
fn backoff_of_long_unknown_words() {
    // Every split of an unknown word was tried as a stem: a 40 KB token took seconds in the
    // sparse encoder's backoff (review 3 MORPH3-01). Only the last letters can be inflections.
    let morph = TurkishMorphology::new();
    assert_linear(
        "backoff stem",
        2_000,
        |n| format!("{}lerinden", "zorbant".repeat(n)),
        |w| {
            morph.backoff_stem(w);
        },
    );
}

// ------------------------------------------------------------------- tokenization

#[test]
fn dotted_abbreviation_runs() {
    // Single capitals with periods (A. B. C. …), initials (A.Ş.) and abbreviations before a
    // suffix (HMK.'ya) are each matched from the token where they start.
    let tokenize = |t: &str| {
        akana_core::TurkishTokenizer::tokenize_words(t);
    };
    let segment = |t: &str| {
        akana_core::SentenceSegmenter::segment(t);
    };
    assert_linear("A. B. C.", 8_000, |n| "A. ".repeat(n), tokenize);
    assert_linear("A.Ş.", 2_000, |n| "A.Ş.'nin ".repeat(n), tokenize);
    assert_linear(
        "A.B.C…",
        2_000,
        |n| format!("{}x", "A.".repeat(n)),
        tokenize,
    );
    assert_linear("HMK.'ya", 2_000, |n| "HMK.'ya ".repeat(n), tokenize);
    assert_linear("A. sentences", 8_000, |n| "A. ".repeat(n), segment);
    assert_linear("ellipsis", 8_000, |n| "... 'nin ".repeat(n), segment);
}

#[test]
fn disambiguator_on_adjective_runs() {
    // Every word that can be an adjective looked ahead over the whole run for a noun head, and
    // every word for the sentence's end: "Ali " × 16,000 took 27 s in readability (review 3
    // STYLE3-07).
    let disambiguator = akana_core::morphology::MorphologicalDisambiguator::new();
    assert_linear(
        "Ali Ali …",
        1_000,
        |n| "Ali ".repeat(n),
        |t| {
            let tokens: Vec<&str> = t.split_whitespace().collect();
            disambiguator.disambiguate(&tokens);
        },
    );
    assert_linear(
        "readability of Ali Ali …",
        1_000,
        |n| "Ali ".repeat(n),
        |t| {
            akana_core::readability::analyze_readability(t);
        },
    );
}
