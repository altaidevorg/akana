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
/// probability and a `rebaselined` note, appended to any earlier one. Retraining the model
/// replaces the whole file.
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
        // A row re-baselined before keeps its earlier notes (review 3 STYLE3-08).
        let notes = match row["rebaselined"].as_str() {
            Some(earlier) => format!("{earlier} | {note}"),
            None => note.clone(),
        };
        let field = |v: &serde_json::Value| serde_json::to_string(v).expect("serialisable");
        lines.push(format!(
            "{{\"source\":{},\"id\":{},\"text\":{},\"x\":{},\"p\":{},\"rebaselined\":{}}}{comma}",
            field(&row["source"]),
            field(&row["id"]),
            field(&row["text"]),
            serde_json::to_string(&x).expect("serialisable"),
            serde_json::to_string(&p).expect("serialisable"),
            serde_json::to_string(&notes).expect("serialisable"),
        ));
    }
    let mut out = lines.join("\n");
    if original.ends_with('\n') {
        out.push('\n');
    }
    std::fs::write(path, out).expect("writable reference");
    println!("re-baselined {changed} rows");
}

/// The drift canary's source: paragraphs of [`CANARY_SENTENCES`] consecutive sentences of the
/// grammar gate's court corpus (CC0), so the canary needs no text of its own.
const CANARY_SOURCE: &str = include_str!("../../../tests/data/grammar_clean_court.txt");
const CANARY_SENTENCES: usize = 15;
const CANARY_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/data/ai_detector_canary.jsonl"
);

fn canary_texts() -> Vec<String> {
    let lines: Vec<&str> = CANARY_SOURCE
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect();
    lines
        .chunks(CANARY_SENTENCES)
        .map(|c| c.join(" "))
        .collect()
}

/// A fingerprint of the word sets the detector reads from the shared lexicon ([`super::morph`]'s
/// ROOTS and VERBS): any lexicon edit that changes them changes `oov_rate`, `aorist_ratio` or
/// `colons_per_100w` on some text, whether or not a canary text shows it.
fn lexicon_sets_fingerprint() -> String {
    let sorted = |set: &std::collections::HashSet<&str>| {
        let mut v: Vec<&str> = set.iter().copied().collect();
        v.sort_unstable();
        v.join("\n")
    };
    let roots = super::morph::ROOTS.iter().map(String::as_str).collect();
    let verbs = super::morph::VERBS.iter().map(String::as_str).collect();
    crate::fingerprint::Fingerprint::new()
        .section("roots", sorted(&roots).as_bytes())
        .section("verbs", sorted(&verbs).as_bytes())
        .hex()
}

#[derive(Serialize, Deserialize)]
struct CanaryHead {
    about: String,
    lexicon_sets: String,
    history: Vec<String>,
}

#[derive(Serialize, Deserialize)]
struct CanaryRow {
    id: usize,
    verdict: AiVerdict,
    p: f64,
    x: Vec<f64>,
}

fn canary_now() -> Vec<CanaryRow> {
    canary_texts()
        .iter()
        .enumerate()
        .map(|(id, text)| {
            let x = inputs(text).x;
            let p = MODEL.probability(MODEL.raw_score(&MODEL.standardize(&x)));
            CanaryRow {
                id,
                verdict: detect(text).verdict,
                p,
                x,
            }
        })
        .collect()
}

fn read_canary(text: &str) -> (CanaryHead, Vec<CanaryRow>) {
    let mut lines = text.lines();
    let head = serde_json::from_str(lines.next().expect("canary head")).expect("valid head");
    let rows = lines
        .filter(|l| !l.is_empty())
        .map(|l| serde_json::from_str(l).expect("valid canary row"))
        .collect();
    (head, rows)
}

/// What moved between two canary runs: texts whose inputs or probability moved (with the
/// training reference's tolerances), verdicts that changed, and the largest probability move.
fn canary_drift(old: &[CanaryRow], new: &[CanaryRow]) -> (usize, usize, f64) {
    let mut moved = old.len().abs_diff(new.len());
    let (mut verdicts, mut max_dp) = (0, 0.0f64);
    for (a, b) in old.iter().zip(new) {
        let inputs_moved = a.x.len() != b.x.len()
            || a.x
                .iter()
                .zip(&b.x)
                .any(|(&want, &got)| (got - want).abs() > 1e-6 * (1.0 + want.abs()));
        if inputs_moved || (a.p - b.p).abs() >= 1e-9 {
            moved += 1;
        }
        verdicts += usize::from(a.verdict != b.verdict);
        max_dp = max_dp.max((a.p - b.p).abs());
    }
    (moved, verdicts, max_dp)
}

/// Pins the detector's inputs on 100 committed court paragraphs, and the lexicon word sets it
/// reads (review 3 STYLE3-01). The model was trained on fixed inputs, and the 45-row training
/// reference above sees few of the shared-code changes that move them: since review 2 the
/// inputs moved on 55 % of a 1,928-text sample while it noticed 4 rows. This test fails on any
/// such change, so it is recorded on purpose with
/// `AKANA_AI_DETECTOR_CANARY="<why>" cargo test -p akana-core --lib -- --ignored
/// rewrite_drift_canary`, and the model's retrain is planned with it.
#[test]
fn inputs_match_drift_canary() {
    let (head, old) = read_canary(include_str!("../../../tests/data/ai_detector_canary.jsonl"));
    let sets = lexicon_sets_fingerprint();
    let new = canary_now();
    let (moved, verdicts, max_dp) = canary_drift(&old, &new);
    assert!(
        moved == 0 && head.lexicon_sets == sets,
        "the AI detector's inputs moved: {moved} of {} canary texts (max |Δp| {max_dp:.2e}, \
         {verdicts} verdicts changed); lexicon word sets changed: {}. The model was trained on \
         fixed inputs: if the change is deliberate, record it with \
         AKANA_AI_DETECTOR_CANARY=\"<why>\" cargo test -p akana-core --lib -- --ignored \
         rewrite_drift_canary, and plan the retrain.",
        new.len(),
        head.lexicon_sets != sets
    );
}

/// Rewrites `tests/data/ai_detector_canary.jsonl` after a deliberate change of the detector's
/// inputs, appending the reason and what moved to its history.
#[test]
#[ignore = "rewrites tests/data/ai_detector_canary.jsonl; set AKANA_AI_DETECTOR_CANARY"]
fn rewrite_drift_canary() {
    let note = std::env::var("AKANA_AI_DETECTOR_CANARY")
        .expect("set AKANA_AI_DETECTOR_CANARY to the reason for the rewrite");
    let new = canary_now();
    let sets = lexicon_sets_fingerprint();
    let (mut history, entry) = match std::fs::read_to_string(CANARY_PATH) {
        Ok(text) if !read_canary(&text).1.is_empty() => {
            let (head, old) = read_canary(&text);
            let (moved, verdicts, max_dp) = canary_drift(&old, &new);
            let entry = format!(
                "{note} (inputs moved on {moved} of {} texts, max |Δp| {max_dp:.2e}, {verdicts} \
                 verdicts changed; lexicon word sets changed: {})",
                new.len(),
                head.lexicon_sets != sets
            );
            (head.history, entry)
        }
        _ => (Vec::new(), note),
    };
    history.push(entry);
    let head = CanaryHead {
        about: "The AI-text detector's 75 inputs (x, model input order), probability (p) and \
                verdict on paragraphs of 15 consecutive sentences of grammar_clean_court.txt \
                (CC0), computed by the Rust implementation, and a fingerprint of the lexicon word \
                sets it reads. A drift canary (style::ai_detector::tests::inputs_match_drift_canary), \
                not training-code parity: history records each deliberate input change."
            .to_string(),
        lexicon_sets: sets,
        history,
    };
    let mut out = serde_json::to_string(&head).expect("serialisable") + "\n";
    for row in &new {
        out += &(serde_json::to_string(row).expect("serialisable") + "\n");
    }
    std::fs::write(CANARY_PATH, out).expect("writable canary");
}
