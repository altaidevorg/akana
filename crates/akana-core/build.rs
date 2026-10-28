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
/// The source the sparse and hybrid vectors depend on: the analyses, the tokenizer and
/// sentence segmenter, the encoders themselves and the hash they use.
const ENCODERS: &[&str] = &[
    "src/morphology",
    "src/phonology",
    "src/tokenization",
    "src/embeddings",
    "src/fingerprint.rs",
];

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

fn main() {
    let crate_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("set by cargo"));
    for root in ENCODERS {
        println!("cargo:rerun-if-changed={root}");
    }
    println!(
        "cargo:rustc-env=AKANA_ANALYSIS_SOURCES={}",
        sources(&crate_dir, ANALYSIS)
    );
    println!(
        "cargo:rustc-env=AKANA_ENCODER_SOURCES={}",
        sources(&crate_dir, ENCODERS)
    );
    // The resolved dependencies (regex, unicode-normalization) move vectors too. A build from
    // the published crate has no workspace lock file, and says so.
    let lock = crate_dir.join("../../Cargo.lock");
    let locked = match std::fs::read(&lock) {
        Ok(bytes) => {
            println!("cargo:rerun-if-changed=../../Cargo.lock");
            let text: Vec<u8> = bytes.into_iter().filter(|&b| b != b'\r').collect();
            fingerprint::Fingerprint::new()
                .section("Cargo.lock", &text)
                .hex()
        }
        Err(_) => "none".to_string(),
    };
    println!("cargo:rustc-env=AKANA_CARGO_LOCK={locked}");
}
