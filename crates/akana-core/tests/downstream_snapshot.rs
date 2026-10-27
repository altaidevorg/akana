//! What PII masking and the dependency parser make of fixed synthetic sentences
//! (`tests/data/downstream/`), as snapshots: both read the analyzer's readings, so a change of
//! readings moved masked names and parse trees with no test noticing (review 2 PII2-03,
//! PII2-04, PARSER2-01). A change here must be deliberate and listed in its commit.
//!
//! After a deliberate change: `AKANA_GOLD_WRITE=1 cargo test -p akana-core --test
//! downstream_snapshot`, and give the diff of the snapshots in the commit message.

use akana_core::parser::TurkishDependencyParser;
use akana_core::pii::{PiiMode, TurkishPiiEngine};
use std::path::PathBuf;

fn data(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/downstream")
        .join(name)
}

fn sentences(name: &str) -> Vec<String> {
    std::fs::read_to_string(data(name))
        .unwrap_or_else(|e| panic!("read {name}: {e}"))
        .lines()
        .map(str::trim_end)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(str::to_string)
        .collect()
}

/// Compares `now` with the snapshot `name`, or writes it under `AKANA_GOLD_WRITE`.
fn check(name: &str, now: &str) {
    let path = data(name);
    if std::env::var_os("AKANA_GOLD_WRITE").is_some() {
        std::fs::write(&path, now).expect("write snapshot");
        return;
    }
    let before = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {name}: {e}"));
    let diffs: Vec<String> = before
        .lines()
        .zip(now.lines())
        .filter(|(a, b)| a != b)
        .map(|(a, b)| format!("  {a}\n  -> {b}"))
        .collect();
    assert!(
        diffs.is_empty() && before.lines().count() == now.lines().count(),
        "{name} changed ({} lines differ, {} lines before, {} now):\n{}\n\
         If the change is deliberate, rewrite it with AKANA_GOLD_WRITE=1 and list it in the commit.",
        diffs.len(),
        before.lines().count(),
        now.lines().count(),
        diffs.join("\n")
    );
}

#[test]
fn pii_masks_stay_as_recorded() {
    let engine = TurkishPiiEngine::new();
    let mut now = String::new();
    for sentence in sentences("pii_sentences.txt") {
        let result = engine.mask(&sentence, PiiMode::Placeholder);
        now.push_str(&format!("> {sentence}\n{}\n", result.masked_text));
        for e in &result.entities {
            now.push_str(&format!("\t{}\t{}\n", e.label, e.text));
        }
    }
    check("pii_snapshot.txt", &now);
}

#[test]
fn parse_trees_stay_as_recorded() {
    let parser = TurkishDependencyParser::new();
    let mut now = String::new();
    for sentence in sentences("parser_sentences.txt") {
        let tokens: Vec<&str> = sentence.split(' ').collect();
        let tree = parser.parse(&tokens);
        now.push_str(&format!("# {sentence}\n"));
        for n in &tree.nodes {
            now.push_str(&format!(
                "{}\t{}\t{}\t{}\t{}\t{}\n",
                n.id, n.form, n.lemma, n.upos, n.head, n.deprel
            ));
        }
        now.push('\n');
    }
    check("parser_snapshot.conllu", &now);
}
