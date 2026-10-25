//! Estimates the analyzer's frequency prior from a corpus sample.
//!
//! ```text
//! python scripts/morph_corpus_sample.py ...   # writes .cache/morph_sample/{train,eval}
//! cargo run --release -p akana-core --example morph_prior -- \
//!     --types .cache/morph_sample/train --out crates/akana-core/src/morphology/prior.tsv.gz
//! ```
//!
//! Every word type of the training sample is analysed without the prior. A reading's
//! probability is P(item) * P(shape | POS): how often its dictionary item (lemma, root, POS,
//! secondary POS) is meant, times how often a word of that POS takes its shape (the tag
//! sequence, Noun+P3sg+Loc). EM estimates both: each word's count is split among its readings
//! in proportion to their probability, and the split counts give the next estimates.
//!
//! Two fixed factors join them, each estimated from the words whose readings all agree:
//! - the case of the word as written, P(capitalised | name) and P(capitalised | other word):
//!   "Ali" is capitalised 1,142 times in 1,143, "yani" 200 times in 1,571, so Ali is the name
//!   and Yani the conjunction at the start of a sentence;
//! - a register factor for the second person (P2sg, A2pl, ...), which almost never stands alone
//!   in the sample but is everywhere an alternative (karar-ın-ın is P3sg+Gen or P2sg+Gen,
//!   mahkeme-nin Gen or P2sg+Gen), so EM would credit it from every word it could explain: the
//!   count of words that can only be second person over those that can only be a third-person
//!   possessive.
//!
//! The output lists the factors and both kinds of counts, gzipped, for `analyzer.rs` to embed.

use akana_core::morphology::{
    prior_capitalised, prior_is_name, prior_item_key, prior_shape_key, TurkishMorphology,
    SECOND_PERSON,
};
use std::collections::HashMap;
use std::io::Write;

/// One reading of a word type: item id, shape id, whether it names something, whether it is
/// second person.
#[derive(Clone, Copy, PartialEq)]
struct Reading {
    item: usize,
    shape: usize,
    name: bool,
    second: bool,
}

/// One word type: its count, whether it is capitalised, its readings.
struct Word {
    count: f64,
    capitalised: bool,
    readings: Vec<Reading>,
}

fn main() {
    let mut types_dir = String::from(".cache/morph_sample/train");
    let mut out = String::from("crates/akana-core/src/morphology/prior.tsv.gz");
    let mut iterations = 10usize;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--types" => types_dir = args.next().expect("--types DIR"),
            "--out" => out = args.next().expect("--out FILE"),
            "--iterations" => {
                iterations = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .expect("--iterations N")
            }
            other => panic!("unknown argument {other}"),
        }
    }
    let morphology = TurkishMorphology::new().with_prior(false);

    // word -> count over every *.types.tsv file of the directory
    let mut counts: HashMap<String, f64> = HashMap::new();
    let mut files: Vec<_> = std::fs::read_dir(&types_dir)
        .expect("types dir")
        .map(|e| e.expect("entry").path())
        .filter(|p| p.to_string_lossy().ends_with(".types.tsv"))
        .collect();
    files.sort();
    for path in &files {
        for line in std::fs::read_to_string(path).expect("read").lines() {
            if let Some((w, c)) = line.split_once('\t') {
                *counts.entry(w.to_string()).or_default() += c.parse::<f64>().unwrap_or(0.0);
            }
        }
    }
    let mut types: Vec<(&String, f64)> = counts.iter().map(|(w, &c)| (w, c)).collect();
    types.sort_by(|a, b| a.0.cmp(b.0));

    // Each shape's POS (its first tag) as an id, for P(shape | POS).
    let mut items: HashMap<String, usize> = HashMap::new();
    let mut shapes: HashMap<String, usize> = HashMap::new();
    let mut pos_ids: HashMap<String, usize> = HashMap::new();
    let mut shape_pos: Vec<usize> = Vec::new();
    let mut third_shape: Vec<bool> = Vec::new();
    let mut words: Vec<Word> = Vec::new();
    for (w, count) in types {
        let parses = morphology.analyze(w);
        if parses.is_empty() {
            continue;
        }
        let mut readings = Vec::new();
        for p in &parses {
            let n = items.len();
            let item = *items.entry(prior_item_key(p)).or_insert(n);
            let key = prior_shape_key(p);
            let n = shapes.len();
            let shape = *shapes.entry(key.clone()).or_insert(n);
            if shape == shape_pos.len() {
                let pos = key.split('+').next().unwrap_or_default().to_string();
                let n = pos_ids.len();
                shape_pos.push(*pos_ids.entry(pos).or_insert(n));
                third_shape.push(key.split('+').any(|t| matches!(t, "P3sg" | "P3pl")));
            }
            let reading = Reading {
                item,
                shape,
                name: prior_is_name(p),
                second: p
                    .morpheme_tags
                    .iter()
                    .any(|t| SECOND_PERSON.contains(&t.as_str())),
            };
            if !readings.contains(&reading) {
                readings.push(reading);
            }
        }
        words.push(Word {
            count,
            capitalised: prior_capitalised(w),
            readings,
        });
    }

    // The fixed factors, from words whose readings all agree.
    let share = |of: &dyn Fn(&Word) -> bool, among: &dyn Fn(&Word) -> bool| -> f64 {
        let (hit, total) = words
            .iter()
            .filter(|w| among(w))
            .fold((0.0, 0.0), |(h, t), w| {
                (h + if of(w) { w.count } else { 0.0 }, t + w.count)
            });
        ((hit + 0.5) / (total + 1.0)).clamp(0.001, 0.999)
    };
    let capitalised_name = share(&|w| w.capitalised, &|w| w.readings.iter().all(|r| r.name));
    let capitalised_other = share(&|w| w.capitalised, &|w| w.readings.iter().all(|r| !r.name));
    let (only_second, only_third) = words.iter().fold((0.0, 0.0), |(a, b), w| {
        let second = w.readings.iter().all(|r| r.second);
        let third = w.readings.iter().all(|r| third_shape[r.shape]);
        (
            a + if second { w.count } else { 0.0 },
            b + if third { w.count } else { 0.0 },
        )
    });
    let register = ((only_second + 0.5) / (only_third + 0.5)).min(1.0);
    let factor = |w: &Word, r: &Reading| -> f64 {
        let case = match (w.capitalised, r.name) {
            (true, true) => capitalised_name,
            (true, false) => capitalised_other,
            (false, true) => 1.0 - capitalised_name,
            (false, false) => 1.0 - capitalised_other,
        };
        case * if r.second { register } else { 1.0 }
    };

    let item_probs = |counts: &[f64]| -> Vec<f64> {
        let total: f64 = counts.iter().sum::<f64>() + 0.5 * counts.len() as f64;
        counts.iter().map(|c| (c + 0.5) / total).collect()
    };
    let shape_probs = |counts: &[f64]| -> Vec<f64> {
        let mut totals = vec![0.0f64; pos_ids.len()];
        for (s, c) in counts.iter().enumerate() {
            totals[shape_pos[s]] += c + 0.5;
        }
        counts
            .iter()
            .enumerate()
            .map(|(s, c)| (c + 0.5) / totals[shape_pos[s]])
            .collect()
    };
    let mut item_c = vec![0.0f64; items.len()];
    let mut shape_c = vec![0.0f64; shapes.len()];
    // Uniform start: readings that always compete in the same words (A3pl+Gen and A3pl+P2sg
    // in -lerin, ActN and Neg+Imp+A2sg in -me) keep equal probabilities, and the analyzer's
    // tie-break decides between them.
    let mut item_p = item_probs(&item_c);
    let mut shape_p = shape_probs(&shape_c);
    for _ in 0..iterations {
        item_c.iter_mut().for_each(|x| *x = 0.0);
        shape_c.iter_mut().for_each(|x| *x = 0.0);
        for w in &words {
            let weight = |r: &Reading| item_p[r.item] * shape_p[r.shape] * factor(w, r);
            let z: f64 = w.readings.iter().map(weight).sum();
            for r in &w.readings {
                let share = weight(r) / z;
                item_c[r.item] += w.count * share;
                shape_c[r.shape] += w.count * share;
            }
        }
        item_p = item_probs(&item_c);
        shape_p = shape_probs(&shape_c);
    }

    let mut rows: Vec<String> = vec![
        format!("R\tcapitalised-name\t{capitalised_name:.6}"),
        format!("R\tcapitalised-other\t{capitalised_other:.6}"),
        format!("R\tsecond-person\t{register:.6}"),
    ];
    for (prefix, keys, counts) in [("I", &items, &item_c), ("S", &shapes, &shape_c)] {
        let mut kind: Vec<(&String, f64)> = keys.iter().map(|(k, &i)| (k, counts[i])).collect();
        kind.retain(|r| r.1 >= 1.0);
        kind.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(b.0)));
        rows.extend(kind.iter().map(|(k, c)| format!("{prefix}\t{k}\t{c:.0}")));
    }
    let file = std::fs::File::create(&out).expect("create");
    let mut gz = flate2::GzBuilder::new().write(file, flate2::Compression::best());
    for row in &rows {
        writeln!(gz, "{row}").expect("write");
    }
    gz.finish().expect("finish");
    println!(
        "{} word types, {} items, {} shapes; capitalised: names {capitalised_name:.3}, \
         other words {capitalised_other:.3}; second-person factor {register:.6} \
         ({only_second:.0} / {only_third:.0} tokens); {} rows written to {out}",
        words.len(),
        items.len(),
        shapes.len(),
        rows.len()
    );
}
