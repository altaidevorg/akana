//! The encoders' golden vectors: the exact sparse vector (ids, value bits, terms) and hybrid
//! bytes of fixed texts (`tests/data/embeddings/golden.jsonl`), and the embedded resources they
//! were made with (the encoder fingerprint also covers the source code, which changes with
//! every edit). Any change to them fails here, on every platform CI runs. That is the point: a
//! change to the vectors or the resources must be deliberate.
//!
//! After a deliberate change, measure it with `scripts/encoder_churn.py` (the bump class of
//! `embeddings/spec.rs`), record the class in the commit, and rewrite the file:
//! `UPDATE_GOLDEN=1 cargo test -p akana-core --test golden_vectors`.

#[path = "../src/fingerprint.rs"]
#[allow(dead_code)]
mod fingerprint;

use akana_core::embeddings::spec::SPARSE_SPEC;
use akana_core::embeddings::{encoder_info, MorphologicalSparseEncoder, TurkishHybridEmbeddings};
use serde_json::{json, Value};
use std::path::PathBuf;

/// Each sparse spec major's release fixture (`golden-<major>.0.jsonl`: the vectors of the
/// release that froze it) and the fingerprint of the file as that release wrote it, so that it
/// is never rewritten (review 4 CONTRACT4-02). The release that bumps the major adds its file
/// and its line.
const RELEASE_FIXTURES: &[(u16, &str)] = &[(1, "162471033997f13fea15b87549c82f98")];

/// A release fixture holds at least this many texts: a truncated one bounds nothing.
const RELEASE_MIN_ROWS: usize = 150;

fn golden_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/embeddings/golden.jsonl")
}

/// The vectors of the release that froze sparse spec `major`, never rewritten after it.
fn release_path(major: u16) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(format!("tests/data/embeddings/golden-{major}.0.jsonl"))
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
/// release fixture, and the fixture's own fingerprint is pinned, so rewriting either golden
/// file cannot hide such a change (review 4 CONTRACT4-02).
#[test]
fn the_spec_release_bounds_the_change() {
    let major = SPARSE_SPEC.0;
    let path = release_path(major);
    let pinned = RELEASE_FIXTURES
        .iter()
        .find(|(m, _)| *m == major)
        .map(|(_, f)| *f)
        .unwrap_or_else(|| {
            panic!(
                "no release fixture for sparse spec {major}: the release that sets the major \
                 writes {} and pins it in RELEASE_FIXTURES",
                path.display()
            )
        });
    let content =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert_eq!(
        fingerprint::of(content.as_bytes()),
        pinned,
        "{} differs from the file the release of spec {major} wrote: it is never rewritten",
        path.display()
    );
    let mut lines = content.lines();
    let header: Value = serde_json::from_str(lines.next().expect("header")).expect("header json");
    let stored = header["sparse_spec"].as_str().expect("sparse_spec");
    assert_eq!(
        stored.split('.').next(),
        Some(major.to_string().as_str()),
        "{} was written for spec {stored}",
        path.display()
    );
    let sparse = MorphologicalSparseEncoder::new();
    let hybrid = TurkishHybridEmbeddings::new();
    let rows: Vec<Value> = lines
        .map(|l| serde_json::from_str(l).expect("row json"))
        .collect();
    assert!(
        rows.len() >= RELEASE_MIN_ROWS,
        "{} holds {} texts, fewer than {RELEASE_MIN_ROWS}",
        path.display(),
        rows.len()
    );
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
