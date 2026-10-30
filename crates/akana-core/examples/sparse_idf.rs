//! Finds the sparse encoder's function lemmas and writes their frozen weights.
//!
//! ```text
//! python scripts/morph_corpus_sample.py --root ... --out .cache/morph_sample --train-text
//! python scripts/fetch_morph_sources.py
//! cargo run --release -p akana-core --example sparse_idf -- \
//!     --court .cache/morph_sample/train --ud .cache/morph_sources/ud \
//!     --out crates/akana-core/src/embeddings/function_lemmas.tsv
//! ```
//!
//! The table is part of sparse spec 1 (`embeddings/spec.rs`). Regenerating it changes the value
//! of a function lemma in nearly every vector, so it is rebuilt only for a new spec major.
//!
//! Both samples are cut into the chunks a retrieval index holds: `SentenceChunker` with 256
//! tokens, no overlap.
//! - The court sample is the decisions of each `*.txt` file of `--court`, separated by a line
//!   holding only `\x1e`.
//! - The general sample is the sentences of the train and dev splits of the UD Turkish BOUN
//!   (news, essays, biographies, non-fiction) and Kenet (dictionary examples) treebanks, in file
//!   order. The test splits are never used. Penn is left out: it is translated Wall Street
//!   Journal text, and with it şirket "company" counted as a function lemma.
//!
//! Each chunk is encoded with [`MorphologicalSparseEncoder::term_occurrences`], and a term's
//! frequency in a sample is the share of the sample's chunks it occurs in. A term common in both,
//! in more than [`FUNCTION_LEMMA_DF`] of the chunks of each, is a function lemma. Its weight is
//! `ln p / ln FUNCTION_LEMMA_DF`, with `p` the smaller of the two shares: 1 at the threshold, 0
//! for a term in every chunk. The weight is written in 64ths, the value the encoder uses.
//!
//! The UD treebanks are under CC BY-SA 4.0, so the table is marked in THIRD_PARTY_NOTICES as
//! derived from CC BY-SA data.

use akana_core::chunking::SentenceChunker;
use akana_core::embeddings::sparse::FUNCTION_LEMMA_DF;
use akana_core::embeddings::{MorphologicalSparseEncoder, SPARSE_SPEC};
use std::collections::{HashMap, HashSet};
use std::io::Write;

/// The general sample: these UD treebanks' train and dev splits.
const UD_GENERAL: &[&str] = &["BOUN", "Kenet"];

/// Chunk size in tokens, as the retrieval benchmarks and the README index.
const CHUNK_TOKENS: usize = 256;

fn main() {
    let mut court_dir = String::from(".cache/morph_sample/train");
    let mut ud_dir = String::from(".cache/morph_sources/ud");
    let mut out = String::from("crates/akana-core/src/embeddings/function_lemmas.tsv");
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--court" => court_dir = args.next().expect("--court DIR"),
            "--ud" => ud_dir = args.next().expect("--ud DIR"),
            "--out" => out = args.next().expect("--out FILE"),
            other => panic!("unknown argument {other}"),
        }
    }

    let chunker = SentenceChunker::new(CHUNK_TOKENS, 0, 1);
    let court_docs = court_documents(&court_dir);
    let general_docs = ud_documents(&ud_dir);
    let court_chunks = chunks(&chunker, &court_docs);
    let general_chunks = chunks(&chunker, &general_docs);

    let encoder = MorphologicalSparseEncoder::new();
    let court = chunk_frequencies(&encoder, &court_chunks, "court");
    let general = chunk_frequencies(&encoder, &general_chunks, "general (UD)");

    let share = |n: u32, total: usize| f64::from(n) / total as f64;
    let mut rows: Vec<(&String, u32, u32, f64)> = court
        .iter()
        .filter_map(|(term, &c)| {
            let g = general.get(term).copied()?;
            let p = share(c, court_chunks.len()).min(share(g, general_chunks.len()));
            (p > FUNCTION_LEMMA_DF).then_some((term, c, g, p))
        })
        .collect();
    rows.sort_by(|a, b| b.3.total_cmp(&a.3).then_with(|| a.0.cmp(b.0)));

    let mut file = std::io::BufWriter::new(std::fs::File::create(&out).expect("create --out"));
    let header = [
        "# Function lemmas of the sparse encoder, frozen with sparse spec 1 (examples/sparse_idf.rs).",
        "# term, weight in 64ths, then the number of chunks of the court and the general sample it",
        "# occurs in. DERIVED FROM CC BY-SA DATA: the general sample is UD Turkish BOUN and Kenet",
        "# (CC BY-SA 4.0); see THIRD_PARTY_NOTICES.",
    ];
    for line in header {
        writeln!(file, "{line}").expect("write");
    }
    writeln!(file, "#spec\t{}", SPARSE_SPEC.0).expect("write");
    writeln!(
        file,
        "#chunks\t{}\t{}",
        court_chunks.len(),
        general_chunks.len()
    )
    .expect("write");
    for (term, c, g, p) in &rows {
        let weight = p.ln() / FUNCTION_LEMMA_DF.ln();
        writeln!(file, "{term}\t{}\t{c}\t{g}", (weight * 64.0).round() as u32).expect("write");
    }
    file.flush().expect("flush");
    eprintln!("wrote {} function lemmas to {out}", rows.len());
    for (term, c, g, p) in &rows {
        let weight = p.ln() / FUNCTION_LEMMA_DF.ln();
        eprintln!("  {term}\t{c}\t{g}\tshare {p:.3}\tweight {weight:.3}");
    }
}

/// The documents of a directory's `*.txt` files, separated by a line holding only `\x1e`.
fn court_documents(dir: &str) -> Vec<String> {
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read {dir}: {e}"))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "txt"))
        .collect();
    files.sort();
    let mut docs = Vec::new();
    for file in &files {
        let text = std::fs::read_to_string(file).expect("read document file");
        docs.extend(
            text.split("\n\x1e\n")
                .filter(|d| !d.trim().is_empty())
                .map(str::to_string),
        );
    }
    assert!(!docs.is_empty(), "no documents in {dir}/*.txt");
    docs
}

/// One document per UD file: the `# text = ` sentences of the general treebanks' train and dev
/// splits, in file order.
fn ud_documents(dir: &str) -> Vec<String> {
    let mut docs = Vec::new();
    for treebank in UD_GENERAL {
        let lower = treebank.to_lowercase();
        for split in ["train", "dev"] {
            let path = format!("{dir}/UD_Turkish-{treebank}/tr_{lower}-ud-{split}.conllu");
            let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
                panic!("read {path}: {e} (run scripts/fetch_morph_sources.py)")
            });
            let sentences: Vec<&str> = text
                .lines()
                .filter_map(|l| l.strip_prefix("# text = "))
                .collect();
            docs.push(sentences.join("\n"));
        }
    }
    docs
}

fn chunks(chunker: &SentenceChunker, docs: &[String]) -> Vec<String> {
    docs.iter()
        .flat_map(|d| chunker.chunk(d).into_iter().map(|c| c.text))
        .collect()
}

/// Each term's number of chunks.
fn chunk_frequencies(
    encoder: &MorphologicalSparseEncoder,
    chunks: &[String],
    name: &str,
) -> HashMap<String, u32> {
    let start = std::time::Instant::now();
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let part_len = chunks.len().div_ceil(threads).max(1);
    let partial: Vec<HashMap<String, u32>> = std::thread::scope(|s| {
        let handles: Vec<_> = chunks
            .chunks(part_len)
            .map(|part| {
                s.spawn(move || {
                    let mut df: HashMap<String, u32> = HashMap::new();
                    for chunk in part {
                        let terms: HashSet<String> = encoder
                            .term_occurrences(chunk)
                            .into_iter()
                            .map(|(term, _)| term)
                            .collect();
                        for term in terms {
                            *df.entry(term).or_insert(0) += 1;
                        }
                    }
                    df
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("worker"))
            .collect()
    });
    let mut df: HashMap<String, u32> = HashMap::new();
    for part in partial {
        for (term, n) in part {
            *df.entry(term).or_insert(0) += n;
        }
    }
    eprintln!(
        "{name}: {} chunks, {} terms in {:.1} s ({threads} threads)",
        chunks.len(),
        df.len(),
        start.elapsed().as_secs_f64()
    );
    df
}
