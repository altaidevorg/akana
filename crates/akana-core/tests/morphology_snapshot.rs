//! The best analysis of the most frequent word types of court decisions and RagTurk
//! (`tests/data/morphology/top_types.txt`), as a snapshot (`top_types_analyses.tsv`): lemma,
//! root POS, secondary POS, tags, the POS after derivation and the Zemberek analysis. Sparse
//! terms are lemmas, so a lemma that moves changes every index that holds the word. A change
//! here must be deliberate and listed in its commit.
//!
//! After a deliberate change: `AKANA_GOLD_WRITE=1 cargo test -p akana-core --test
//! morphology_snapshot`, and give the diff of the snapshot in the commit message.

use akana_core::morphology::TurkishMorphology;
use std::path::PathBuf;

fn data(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/morphology")
        .join(name)
}

#[test]
fn frequent_types_keep_their_best_analysis() {
    let types = std::fs::read_to_string(data("top_types.txt")).expect("read top_types.txt");
    let morphology = TurkishMorphology::new();
    let mut now = String::new();
    for word in types.lines().filter(|w| !w.is_empty()) {
        let best = morphology.analyze(word).into_iter().next().map_or_else(
            || "-\t-\t-\t-\t-\t-".to_string(),
            |p| {
                format!(
                    "{}\t{}\t{}\t{}\t{}\t{}",
                    p.lemma,
                    p.primary_pos.as_str(),
                    p.secondary_pos.as_str(),
                    p.morpheme_tags.join("+"),
                    p.pos.as_str(),
                    p.analysis
                )
            },
        );
        now.push_str(&format!("{word}\t{best}\n"));
    }
    let path = data("top_types_analyses.tsv");
    if std::env::var_os("AKANA_GOLD_WRITE").is_some() {
        std::fs::write(&path, &now).expect("write snapshot");
        return;
    }
    let before = std::fs::read_to_string(&path).expect("read top_types_analyses.tsv");
    let diffs: Vec<String> = before
        .lines()
        .zip(now.lines())
        .filter(|(a, b)| a != b)
        .map(|(a, b)| format!("{a}  ->  {b}"))
        .collect();
    let lemma_diffs = before
        .lines()
        .zip(now.lines())
        .filter(|(a, b)| a.split('\t').nth(1) != b.split('\t').nth(1))
        .count();
    assert!(
        diffs.is_empty() && before.lines().count() == now.lines().count(),
        "{} of {} frequent types changed their best analysis ({lemma_diffs} their lemma):\n{}\n\
         If deliberate, rewrite with AKANA_GOLD_WRITE=1 and list the changes in the commit.",
        diffs.len(),
        now.lines().count(),
        diffs
            .iter()
            .take(30)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}
