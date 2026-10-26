//! The encoders' golden vectors: the exact sparse vector (ids, value bits, terms) and hybrid
//! bytes of fixed texts (`tests/data/embeddings/golden.jsonl`), and the encoder fingerprint they
//! were made with. Any change to them fails here, on every platform CI runs. That is the point:
//! a change to the vectors must be deliberate.
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
            &json!({ "sparse_spec": info.sparse_spec, "fingerprint": info.fingerprint })
                .to_string(),
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
        header["fingerprint"], info.fingerprint,
        "the encoder fingerprint changed although no golden vector did (a resource, a constant or \
         an analysis of the probe set): rewrite with UPDATE_GOLDEN=1 once that is deliberate"
    );
}
