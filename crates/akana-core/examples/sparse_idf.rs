//! Finds the sparse encoder's function lemmas from two corpus samples.
//!
//! ```text
//! python scripts/morph_corpus_sample.py --root ... --out .cache/morph_sample --train-text
//! python scripts/sparse_idf_general.py --out .cache/sparse_idf/general
//! cargo run --release -p akana-core --example sparse_idf -- \
//!     --court .cache/morph_sample/train --general .cache/sparse_idf/general \
//!     --out crates/akana-core/src/embeddings/function_lemmas.tsv
//! ```
//!
//! Every document of each directory's `*.txt` files (separated by a line holding only `\x1e`)
//! is encoded with [`MorphologicalSparseEncoder::term_occurrences`], and a term's document
//! frequency is the number of documents it occurs in. The output lists the sizes of the two
//! samples and every term found in more than [`FUNCTION_LEMMA_DF`] of the documents of both,
//! with its two document frequencies, for `sparse.rs` to embed.
//!
//! It also prints how many distinct terms of the court sample share their 32-bit id, or would
//! share a 24-bit one.

use akana_core::embeddings::sparse::FUNCTION_LEMMA_DF;
use akana_core::embeddings::{hash_term_32bit, term_id_24bit, MorphologicalSparseEncoder};
use std::collections::{HashMap, HashSet};
use std::io::Write;

fn main() {
    let mut court_dir = String::from(".cache/morph_sample/train");
    let mut general_dir = String::from(".cache/sparse_idf/general");
    let mut out = String::from("crates/akana-core/src/embeddings/function_lemmas.tsv");
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--court" => court_dir = args.next().expect("--court DIR"),
            "--general" => general_dir = args.next().expect("--general DIR"),
            "--out" => out = args.next().expect("--out FILE"),
            other => panic!("unknown argument {other}"),
        }
    }

    let encoder = MorphologicalSparseEncoder::new();
    let (court_docs, court) = document_frequencies(&encoder, &court_dir);
    let (general_docs, general) = document_frequencies(&encoder, &general_dir);

    let shared = |ids: Vec<u32>| {
        let mut seen: HashMap<u32, u32> = HashMap::new();
        for id in ids {
            *seen.entry(id).or_insert(0) += 1;
        }
        seen.values().filter(|&&n| n > 1).sum::<u32>()
    };
    let ids: Vec<u32> = court.keys().map(|t| hash_term_32bit(t)).collect();
    eprintln!(
        "court terms sharing an id: {} of {} with 32 bits, {} with 24 bits",
        shared(ids.clone()),
        court.len(),
        shared(ids.into_iter().map(term_id_24bit).collect())
    );

    let fraction = |df: Option<&u32>, docs: usize| df.map_or(0.0, |&n| n as f64 / docs as f64);
    let mut rows: Vec<(&String, u32, u32, f64)> = court
        .iter()
        .map(|(term, &c)| {
            let g = general.get(term).copied().unwrap_or(0);
            let p = fraction(Some(&c), court_docs).min(fraction(Some(&g), general_docs));
            (term, c, g, p)
        })
        .filter(|row| row.3 > FUNCTION_LEMMA_DF)
        .collect();
    rows.sort_by(|a, b| b.3.total_cmp(&a.3).then_with(|| a.0.cmp(b.0)));

    let mut file = std::io::BufWriter::new(std::fs::File::create(&out).expect("create --out"));
    writeln!(
        file,
        "# Terms in more than {FUNCTION_LEMMA_DF} of the documents of both samples \
         (examples/sparse_idf.rs): term, court df, general df."
    )
    .expect("write");
    writeln!(file, "#docs\t{court_docs}\t{general_docs}").expect("write");
    for (term, c, g, _) in &rows {
        writeln!(file, "{term}\t{c}\t{g}").expect("write");
    }
    file.flush().expect("flush");
    eprintln!("wrote {} function lemmas to {out}", rows.len());
    for (term, c, g, p) in &rows {
        let weight = p.ln() / FUNCTION_LEMMA_DF.ln();
        eprintln!("  {term}\t{c}\t{g}\tdf {p:.3}\tweight {weight:.3}");
    }
}

/// The number of documents of a directory's `*.txt` files and each term's document frequency.
fn document_frequencies(
    encoder: &MorphologicalSparseEncoder,
    dir: &str,
) -> (usize, HashMap<String, u32>) {
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read {dir}: {e}"))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "txt"))
        .collect();
    files.sort();
    let mut docs: Vec<String> = Vec::new();
    for file in &files {
        let text = std::fs::read_to_string(file).expect("read document file");
        docs.extend(
            text.split("\n\x1e\n")
                .filter(|d| !d.trim().is_empty())
                .map(str::to_string),
        );
    }
    assert!(!docs.is_empty(), "no documents in {dir}/*.txt");

    let start = std::time::Instant::now();
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let chunk = docs.len().div_ceil(threads);
    let partial: Vec<HashMap<String, u32>> = std::thread::scope(|s| {
        let handles: Vec<_> = docs
            .chunks(chunk)
            .map(|part| {
                s.spawn(move || {
                    let mut df: HashMap<String, u32> = HashMap::new();
                    for doc in part {
                        let terms: HashSet<String> = encoder
                            .term_occurrences(doc)
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
        "{dir}: {} documents, {} terms in {:.1} s ({threads} threads)",
        docs.len(),
        df.len(),
        start.elapsed().as_secs_f64()
    );
    (docs.len(), df)
}
