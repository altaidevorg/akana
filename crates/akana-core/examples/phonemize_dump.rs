//! Writes the phonemiser's whole output for a corpus, to compare two builds byte for byte (the
//! default features against `--no-default-features --features speech`, for example).
//!
//! ```text
//! cargo run --release -p akana-core --example phonemize_dump -- --corpus FILE --out FILE
//! ```
//!
//! The corpus is a JSON file with a `clips` array of objects with a `text` (EchoCraft's corpora),
//! or JSONL with a `text` per line. For every text and both profiles it writes one line: the
//! text's index, the profile, the tokens as `akana.phonemize` returns them, and the whole
//! utterance (words with their provenance, tokens, warnings) that `akana.phonemize_detailed`
//! is built from.

use akana_core::speech::{Phonemizer, Profile};
use std::io::Write;

fn texts(path: &str) -> Result<Vec<String>, String> {
    let raw = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    if let Ok(serde_json::Value::Object(doc)) = serde_json::from_str::<serde_json::Value>(&raw) {
        let clips = doc
            .get("clips")
            .and_then(|c| c.as_array())
            .ok_or_else(|| format!("{path}: no `clips` array"))?;
        return Ok(clips
            .iter()
            .filter_map(|c| c.get("text").and_then(|t| t.as_str()).map(str::to_string))
            .collect());
    }
    Ok(raw
        .lines()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .filter_map(|v| v.get("text").and_then(|t| t.as_str()).map(str::to_string))
        .collect())
}

fn run() -> Result<(), String> {
    let mut corpus = None;
    let mut out = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--corpus" => corpus = args.next(),
            "--out" => out = args.next(),
            _ => return Err(format!("unknown argument {a}")),
        }
    }
    let corpus = corpus.ok_or("--corpus FILE is required")?;
    let out = out.ok_or("--out FILE is required")?;
    let texts = texts(&corpus)?;
    let file = std::fs::File::create(&out).map_err(|e| format!("{out}: {e}"))?;
    let mut w = std::io::BufWriter::new(file);
    let json = |e: serde_json::Error| e.to_string();
    for profile in [Profile::TtsV1, Profile::TtsV2] {
        let phonemizer = Phonemizer::new(profile);
        for (i, text) in texts.iter().enumerate() {
            let u = phonemizer.phonemize(text);
            let tokens = serde_json::to_string(&u.tokens).map_err(json)?;
            let whole = serde_json::to_string(&u).map_err(json)?;
            writeln!(w, "{i}\t{}\t{tokens}\t{whole}", profile.name()).map_err(|e| e.to_string())?;
        }
    }
    w.flush().map_err(|e| e.to_string())?;
    eprintln!("{} texts, 2 profiles -> {out}", texts.len());
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("phonemize_dump: {e}");
        std::process::exit(2);
    }
}
