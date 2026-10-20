//! Tests of the detector.

use super::samples::{ACADEMIC_CUES_TEXT, AI_TEXT, HUMAN_TEXT, POEM};
use super::*;
use crate::embeddings::TOKENIZER;

#[test]
fn model_loads() {
    assert_eq!(MODEL.features.len(), N_INPUTS);
    assert!(MODEL.threshold > 0.5 && MODEL.threshold < 1.0);
    assert!(!MODEL.patterns.is_empty());
}

#[test]
fn transition_tokens_use_akanas_tokenizer_on_flattened_text() {
    // Markdown, list markers and digits are flattened and the text is Turkish-lowercased
    // before Akana's embedding tokenizer cuts it.
    let t = tokens::transition_tokens("- **Merhaba** Dünya, 2024'te İstanbul'a gittim.");
    let want = TOKENIZER.encode("merhaba dünya, 0'te istanbul'a gittim.");
    assert!(!t.is_empty());
    assert_eq!(t, want);
}

#[test]
fn contributions_add_up_to_the_raw_score() {
    for text in [AI_TEXT, HUMAN_TEXT, ACADEMIC_CUES_TEXT] {
        let z = MODEL.standardize(&inputs(text).x);
        let sum: f64 = MODEL.contributions(&z).iter().sum();
        assert!((MODEL.bias() + sum - MODEL.raw_score(&z)).abs() < 1e-9);
    }
}

#[test]
fn ai_answer_is_ai_with_reasons_and_patterns() {
    let d = detect(AI_TEXT);
    assert_eq!(d.verdict, AiVerdict::Ai);
    assert!(d.ai_probability.is_some_and(|p| p >= d.threshold));
    assert!(d.patterns.len() >= 3);
    assert_eq!(d.signals.len(), MAX_SIGNALS);
    assert!(d.report.starts_with("# AI tespit raporu"));
    assert!(d.report.contains("## Neden AI'a benziyor?"));
}

#[test]
fn human_story_is_human() {
    let d = detect(HUMAN_TEXT);
    assert_eq!(d.verdict, AiVerdict::Human, "{d:?}");
    assert!(d.ai_probability.is_some_and(|p| p < d.uncertain_low));
}

#[test]
fn no_decision_means_no_probability_and_no_reasons() {
    for text in ["", "Elbette, yardımcı olabilirim.", POEM] {
        let d = detect(text);
        assert!(!d.verdict.is_decided(), "{text:?}");
        assert_eq!(d.ai_probability, None);
        assert!(d.signals.is_empty() && d.patterns.is_empty());
        assert!(d.report.contains("karar verilmedi"));
        assert!(!d.report.contains("Neden AI"));
    }
    assert_eq!(detect(POEM).verdict, AiVerdict::OutOfScope);
}

#[test]
fn masked_numbers_are_not_words() {
    let d = detect(&"123456 ".repeat(300));
    assert_eq!(d.verdict, AiVerdict::TooShort);
    assert_eq!(d.n_words, 0);
}

#[test]
fn repetitive_text_is_not_judged() {
    for text in [
        "bir ".repeat(100),
        "\0\u{1}\u{7f}\u{1c}\u{1d} a".repeat(100),
    ] {
        let d = detect(&text);
        assert_eq!(d.verdict, AiVerdict::TooShort, "{text:?}");
        assert!(d.n_distinct_words < MIN_DISTINCT_WORDS);
    }
}

#[test]
fn academic_cues_only_warn() {
    let d = detect(ACADEMIC_CUES_TEXT);
    assert_eq!(d.scope.map(|s| s.kind), Some(ScopeKind::Academic));
    // The warning never blocks the decision (only the length gate can).
    assert_ne!(d.verdict, AiVerdict::OutOfScope);
}

#[test]
fn sections_cover_long_texts_only() {
    assert!(sections(AI_TEXT).is_empty());
    let long = [AI_TEXT, HUMAN_TEXT].join("\n\n");
    let s = sections(&long);
    assert!(!s.is_empty());
    assert_eq!(s[0].start_word, 1);
    assert_eq!(
        s.last().map(|x| x.end_word),
        Some(long.split_whitespace().count())
    );
}

#[test]
fn label_colons_are_linear_in_a_line() {
    // Each colon looked through every sentence start of its line: colons × sentences. Four
    // times the line may take at most 9× as long (linear 4×, the old loop 16×); best of three.
    let time = |text: &str| {
        (0..3)
            .map(|_| {
                let t = std::time::Instant::now();
                assert!(super::morph::label_colon_counts(text).0 > 0);
                t.elapsed().as_secs_f64()
            })
            .fold(f64::INFINITY, f64::min)
    };
    let (small, large) = ("a: b. ".repeat(4_000), "a: b. ".repeat(16_000));
    time(&small);
    let (t_small, t_large) = (time(&small), time(&large));
    assert!(
        t_large <= 9.0 * t_small.max(1e-6),
        "{t_small:.4} s → {t_large:.4} s"
    );
}

/// One row of a reference file: a text and the training code's inputs and probability.
#[derive(Deserialize)]
struct Reference {
    text: String,
    x: Vec<f64>,
    p: f64,
}

fn check_references(rows: &[Reference]) {
    for (i, r) in rows.iter().enumerate() {
        let x = inputs(&r.text).x;
        assert_eq!(x.len(), r.x.len(), "row {i}: number of inputs");
        for (j, (&got, &want)) in x.iter().zip(&r.x).enumerate() {
            assert!(
                (got - want).abs() <= 1e-6 * (1.0 + want.abs()),
                "row {i} input {}: {got} vs {want}",
                MODEL.features[j].key
            );
        }
        let p = probability(&r.text);
        assert!((p - r.p).abs() < 1e-9, "row {i}: {p} vs {}", r.p);
    }
}

/// Pins the model inputs and probabilities to the training code's on a committed sample, so
/// a change in shared code (sentence segmenter, Turkish lowercasing, cliché matcher,
/// embedding tokenizer) that would shift the model's inputs fails here.
#[test]
fn outputs_match_training_reference() {
    #[derive(Deserialize)]
    struct File {
        rows: Vec<Reference>,
    }
    let file: File = serde_json::from_str(include_str!(
        "../../../tests/data/ai_detector_reference.json"
    ))
    .expect("valid reference file");
    // Every input is non-zero in at least two rows, so drift in any of them shows up.
    for k in 0..N_INPUTS {
        let nonzero = file.rows.iter().filter(|r| r.x[k] != 0.0).count();
        assert!(
            nonzero >= 2,
            "input {} is zero in the reference",
            MODEL.features[k].key
        );
    }
    check_references(&file.rows);
}

/// The same check on the 400-text evaluation fixture written by the training code (not in
/// this repository): `AKANA_AI_DETECTOR_FIXTURE=<parity_fixture.json> cargo test -p
/// akana-core --release -- --ignored outputs_match_training_fixture`.
#[test]
#[ignore = "needs AKANA_AI_DETECTOR_FIXTURE"]
fn outputs_match_training_fixture() {
    let path = std::env::var("AKANA_AI_DETECTOR_FIXTURE").expect("set AKANA_AI_DETECTOR_FIXTURE");
    let rows: Vec<Reference> =
        serde_json::from_str(&std::fs::read_to_string(path).expect("readable fixture"))
            .expect("valid fixture");
    check_references(&rows);
}

/// Re-baselines `tests/data/ai_detector_reference.json` after a deliberate change to the
/// model's inputs, when retraining is not done at the same time:
/// `AKANA_AI_DETECTOR_REBASELINE="<why>" cargo test -p akana-core --lib -- --ignored
/// rebaseline_training_reference`. Rows whose inputs and probability still match keep the
/// training code's values byte for byte; the others get this implementation's inputs and
/// probability and a `rebaselined` note. Retraining the model replaces the whole file.
#[test]
#[ignore = "rewrites tests/data/ai_detector_reference.json; set AKANA_AI_DETECTOR_REBASELINE"]
fn rebaseline_training_reference() {
    let note = std::env::var("AKANA_AI_DETECTOR_REBASELINE")
        .expect("set AKANA_AI_DETECTOR_REBASELINE to the reason for the re-baseline");
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/ai_detector_reference.json"
    );
    let original = std::fs::read_to_string(path).expect("readable reference");
    let mut changed = 0;
    let mut lines = Vec::new();
    // The file holds one row object per line, each but the last followed by a comma.
    for line in original.lines() {
        let (body, comma) = match line.strip_suffix(',') {
            Some(body) => (body, ","),
            None => (line, ""),
        };
        let Ok(row) = serde_json::from_str::<serde_json::Value>(body) else {
            lines.push(line.to_string());
            continue;
        };
        let text = row["text"].as_str().expect("row text");
        let stored: Vec<f64> = serde_json::from_value(row["x"].clone()).expect("row x");
        let x = inputs(text).x;
        let p = probability(text);
        let same = x.len() == stored.len()
            && x.iter()
                .zip(&stored)
                .all(|(&got, &want)| (got - want).abs() <= 1e-6 * (1.0 + want.abs()))
            && (p - row["p"].as_f64().expect("row p")).abs() < 1e-9;
        if same {
            lines.push(line.to_string());
            continue;
        }
        changed += 1;
        let field = |v: &serde_json::Value| serde_json::to_string(v).expect("serialisable");
        lines.push(format!(
            "{{\"source\":{},\"id\":{},\"text\":{},\"x\":{},\"p\":{},\"rebaselined\":{}}}{comma}",
            field(&row["source"]),
            field(&row["id"]),
            field(&row["text"]),
            serde_json::to_string(&x).expect("serialisable"),
            serde_json::to_string(&p).expect("serialisable"),
            serde_json::to_string(&note).expect("serialisable"),
        ));
    }
    let mut out = lines.join("\n");
    if original.ends_with('\n') {
        out.push('\n');
    }
    std::fs::write(path, out).expect("writable reference");
    println!("re-baselined {changed} rows");
}
