//! The encoders' golden vectors: the exact sparse vector (ids, value bits, terms) and hybrid
//! bytes of fixed texts (`tests/data/embeddings/golden.jsonl`), and the embedded resources they
//! were made with (the encoder fingerprint also covers the source code, which changes with
//! every edit). Any change to them fails here, on every platform CI runs. That is the point: a
//! change to the vectors or the resources must be deliberate.
//!
//! After a deliberate change, measure it with `scripts/encoder_churn.py` (the bump class of
//! `embeddings/spec.rs`), record the class in the commit, and rewrite the file:
//! `UPDATE_GOLDEN=1 cargo test -p akana-core --test golden_vectors`.

use akana_core::embeddings::{encoder_info, MorphologicalSparseEncoder, TurkishHybridEmbeddings};
use serde_json::{json, Value};
use std::path::PathBuf;

fn golden_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/embeddings/golden.jsonl")
}

/// The vectors of the release that froze sparse spec 1, never rewritten after it.
fn release_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/embeddings/golden-1.0.jsonl")
}

/// A text's vectors as the golden file holds them.
fn vectors(
    sparse: &MorphologicalSparseEncoder,
    hybrid: &TurkishHybridEmbeddings,
    text: &str,
) -> Value {
    let v = sparse.encode_sparse(text);
    let terms: Vec<Value> = v
        .indices
        .iter()
        .zip(&v.values)
        .zip(&v.terms)
        .map(|((id, value), term)| json!([id, format!("{:08x}", value.to_bits()), term]))
        .collect();
    let bytes = hybrid.embed_hybrid(text);
    let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    json!({ "sparse": terms, "hybrid": hex })
}

/// The embedded resources' fingerprints, without the analyses' (which covers source code).
fn resources() -> Value {
    encoder_info()
        .resources
        .iter()
        .filter(|(name, _)| *name != "analysis")
        .map(|(name, hash)| (name.to_string(), json!(hash)))
        .collect::<serde_json::Map<String, Value>>()
        .into()
}

/// Review 3 MSTAB3-03: a build may move at most 5 % of the golden texts' vectors away from the
/// release that froze its sparse spec major; more is a major change, which needs a new
/// `SPARSE_SPEC` major (and that release's own fixture). `UPDATE_GOLDEN` does not touch the
/// release fixture, so rewriting the golden file cannot hide such a change.
#[test]
fn the_spec_release_bounds_the_change() {
    let content = std::fs::read_to_string(release_path()).expect("read golden-1.0.jsonl");
    let mut lines = content.lines();
    let header: Value = serde_json::from_str(lines.next().expect("header")).expect("header json");
    let major = |spec: &str| spec.split('.').next().map(str::to_string);
    let stored = header["sparse_spec"].as_str().expect("sparse_spec");
    if major(stored) != major(&encoder_info().sparse_spec) {
        return;
    }
    let sparse = MorphologicalSparseEncoder::new();
    let hybrid = TurkishHybridEmbeddings::new();
    let rows: Vec<Value> = lines
        .map(|l| serde_json::from_str(l).expect("row json"))
        .collect();
    let changed = rows
        .iter()
        .filter(|row| {
            let now = vectors(&sparse, &hybrid, row["text"].as_str().expect("text"));
            now["sparse"] != row["sparse"] || now["hybrid"] != row["hybrid"]
        })
        .count();
    assert!(
        changed * 20 <= rows.len(),
        "{changed} of {} golden texts have other vectors than the release of spec {stored}: more \
         than 5 % is a major change (embeddings/spec.rs), which needs a new SPARSE_SPEC major",
        rows.len()
    );
}

#[test]
fn vectors_equal_the_golden_ones() {
    let path = golden_path();
    let content = std::fs::read_to_string(&path).expect("read golden.jsonl");
    let mut lines = content.lines();
    let header: Value = serde_json::from_str(lines.next().expect("header")).expect("header json");
    let rows: Vec<Value> = lines
        .map(|l| serde_json::from_str(l).expect("row json"))
        .collect();
    let info = encoder_info();
    let sparse = MorphologicalSparseEncoder::new();
    let hybrid = TurkishHybridEmbeddings::new();

    if std::env::var_os("UPDATE_GOLDEN").is_some() {
        let mut out = String::new();
        out.push_str(
            &json!({ "sparse_spec": info.sparse_spec, "resources": resources() }).to_string(),
        );
        out.push('\n');
        for row in &rows {
            let text = row["text"].as_str().expect("text");
            let mut full = vectors(&sparse, &hybrid, text);
            full["text"] = json!(text);
            out.push_str(&full.to_string());
            out.push('\n');
        }
        std::fs::write(&path, out).expect("write golden.jsonl");
        return;
    }

    assert_eq!(
        header["sparse_spec"], info.sparse_spec,
        "golden.jsonl was made for another sparse spec"
    );
    let mut changed = Vec::new();
    for row in &rows {
        let text = row["text"].as_str().expect("text");
        let now = vectors(&sparse, &hybrid, text);
        if now["sparse"] != row["sparse"] || now["hybrid"] != row["hybrid"] {
            changed.push(text.chars().take(60).collect::<String>());
        }
    }
    assert!(
        changed.is_empty(),
        "{} of {} golden texts changed their vectors (first: {:?}). If the change is deliberate, \
         measure it with scripts/encoder_churn.py and rewrite with UPDATE_GOLDEN=1.",
        changed.len(),
        rows.len(),
        &changed[..changed.len().min(5)]
    );
    assert_eq!(
        header["resources"],
        resources(),
        "an embedded resource changed although no golden vector did: rewrite with \
         UPDATE_GOLDEN=1 once that is deliberate"
    );
}
