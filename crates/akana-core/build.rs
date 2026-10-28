//! Hashes the source code the analyses and the encoders' vectors depend on, for their
//! fingerprints (`morphology::analysis_fingerprint`, `embeddings::encoder_info`). A probe text
//! sees a change of code only if it exercises it; with the sources hashed, any change of the
//! code on the path changes the fingerprint, so two builds report "identical" only when they
//! were built from the same inputs (review 3 MSTAB3-02). Line endings are normalised, so a
//! checkout's conversion does not change the hash.

#[path = "src/fingerprint.rs"]
#[allow(dead_code)]
mod fingerprint;

use std::path::{Path, PathBuf};

/// The source the analyses depend on: the analyzer and disambiguator and their phonology.
const ANALYSIS: &[&str] = &["src/morphology", "src/phonology"];
/// The source the sparse vectors depend on: the analyses, the tokenizer and sentence segmenter,
/// the sparse encoder and the hash it uses.
const SPARSE: &[&str] = &[
    "src/morphology",
    "src/phonology",
    "src/tokenization",
    "src/embeddings/sparse.rs",
    "src/fingerprint.rs",
];
/// The source the dense vectors depend on: the model's tokenizer, the weights reader, the
/// pooling and the 2-bit and 1-bit quantisers (review 4 CONTRACT4-01).
const DENSE: &[&str] = &[
    "src/embeddings/mod.rs",
    "src/embeddings/tokenizer.rs",
    "src/embeddings/weights.rs",
    "src/fingerprint.rs",
];
/// The source the hybrid vectors add: the packer.
const HYBRID: &[&str] = &["src/embeddings/hybrid.rs"];

fn rust_files(path: &Path, out: &mut Vec<PathBuf>) {
    if path.is_dir() {
        let mut entries: Vec<PathBuf> = std::fs::read_dir(path)
            .expect("readable source directory")
            .map(|e| e.expect("readable directory entry").path())
            .collect();
        entries.sort();
        for entry in entries {
            rust_files(&entry, out);
        }
    } else if path.extension().is_some_and(|e| e == "rs") {
        out.push(path.to_path_buf());
    }
}

/// The fingerprint of the Rust files under `roots`, each by its path from the crate root.
fn sources(crate_dir: &Path, roots: &[&str]) -> String {
    let mut files = Vec::new();
    for root in roots {
        rust_files(&crate_dir.join(root), &mut files);
    }
    let mut fp = fingerprint::Fingerprint::new();
    for file in files {
        let name = file
            .strip_prefix(crate_dir)
            .expect("a file under the crate")
            .to_string_lossy()
            .replace('\\', "/");
        let text = std::fs::read(&file).expect("readable source file");
        let mut lf = Vec::with_capacity(text.len());
        for (i, &b) in text.iter().enumerate() {
            if !(b == b'\r' && text.get(i + 1) == Some(&b'\n')) {
                lf.push(b);
            }
        }
        fp = fp.section(&name, &lf);
    }
    fp.hex()
}

/// The hash of akana's own `Cargo.lock`, or, for a build that has none of its own (from the
/// published crate, as a git or vendored dependency of another project), a value of this build
/// alone: the dependencies it resolved are unknown, so two such builds never read "identical"
/// (review 4 CONTRACT4-04).
fn lock(crate_dir: &Path) -> String {
    let workspace = crate_dir.join("../../Cargo.toml");
    let own = std::fs::read_to_string(&workspace)
        .is_ok_and(|toml| toml.contains("[workspace]") && toml.contains("\"crates/akana-core\""));
    if own {
        if let Ok(bytes) = std::fs::read(crate_dir.join("../../Cargo.lock")) {
            println!("cargo:rerun-if-changed=../../Cargo.lock");
            println!("cargo:rerun-if-changed=../../Cargo.toml");
            let text: Vec<u8> = bytes.into_iter().filter(|&b| b != b'\r').collect();
            return fingerprint::Fingerprint::new()
                .section("Cargo.lock", &text)
                .hex();
        }
    }
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    format!("unlocked-{now}")
}

fn main() {
    let crate_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("set by cargo"));
    for root in SPARSE.iter().chain(DENSE).chain(HYBRID) {
        println!("cargo:rerun-if-changed={root}");
    }
    for (name, roots) in [
        ("AKANA_ANALYSIS_SOURCES", ANALYSIS),
        ("AKANA_SPARSE_SOURCES", SPARSE),
        ("AKANA_DENSE_SOURCES", DENSE),
        ("AKANA_HYBRID_SOURCES", HYBRID),
    ] {
        println!("cargo:rustc-env={name}={}", sources(&crate_dir, roots));
    }
    // The resolved dependencies (regex, unicode-normalization) move vectors too.
    println!("cargo:rustc-env=AKANA_CARGO_LOCK={}", lock(&crate_dir));
}
