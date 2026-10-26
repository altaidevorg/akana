//! Estimates the analyzer's frequency prior from a corpus sample.
//!
//! ```text
//! python scripts/morph_corpus_sample.py ...   # writes .cache/morph_sample/{train,eval}
//! cargo run --release -p akana-core --example morph_prior -- \
//!     --types .cache/morph_sample/train --out crates/akana-core/src/morphology/prior.tsv.gz
//! ```
//!
//! Every word type of the training sample is analysed without the prior. A reading's
//! probability is P(item) * P(shape | group): how often its dictionary item (lemma, root, POS,
//! secondary POS) is meant, times how often a word of that group takes its shape (the tag
//! sequence, Noun+P3sg+Loc). The group is the POS, except that the personal pronouns share one
//! without their person (bana and ona are both Pers+Dat; see `prior_shape_key`). EM estimates
//! both: each word's count is split among its readings in proportion to their probability, and
//! the split counts give the next estimates.
//!
//! Two fixed factors join them, each estimated from the words whose readings all agree:
//! - the case of the word as written, P(capitalised | name) and P(capitalised | other word):
//!   "Ali" is capitalised 1,142 times in 1,143, "yani" 200 times in 1,571, so Ali is the name
//!   and Yani the conjunction at the start of a sentence;
//! - a register factor for the second person (`prior_second_person`: P2sg, A2pl, the bare
//!   imperative), which almost never stands alone in the sample but is everywhere an
//!   alternative (karar-ın-ın is P3sg+Gen or P2sg+Gen, gel-me the verbal noun or "don't come",
//!   et the noun or "do!"), so EM would credit it from every word it could explain: the count
//!   of words that can only be second person over those that can only be a third-person
//!   possessive.
//!
//! One factor is the lexicon's: where two readings are in every word that has one (kat-ıl-dı and
//! katıl-dı, bula-ma-dığı and bul-ama-dığı), the counts cannot separate them; the lexicon's
//! attributes decide, and the other reading takes `LEXICAL_PREFERENCE` (`prior_yields`).
//!
//! The output is the model as the analyzer uses it, gzipped for `analyzer.rs` to embed: every
//! item's and shape's log probability in integer rank units (`rank_units`), with its count, the
//! backoffs for unseen items, shapes and tags, and the fixed factors. The analyzer only adds
//! these integers, so it ranks alike on every platform, and nothing it reads was rounded or
//! renormalised after estimation. Every sum here runs in a fixed order, so the file reproduces
//! bit for bit.

use akana_core::morphology::{
    prior_capitalised, prior_is_name, prior_item_key, prior_second_person, prior_shape_group,
    prior_shape_key, prior_yields, rank_units, TurkishMorphology, LEXICAL_PREFERENCE,
};
use std::collections::HashMap;
use std::io::Write;

/// One reading of a word type: item id, shape id, whether it names something, whether the
/// register factor applies to it, whether the lexicon ranks it below its twin.
#[derive(Clone, Copy, PartialEq)]
struct Reading {
    item: usize,
    shape: usize,
    name: bool,
    second: bool,
    yields: bool,
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
                let pos = prior_shape_group(&key).to_string();
                let n = pos_ids.len();
                shape_pos.push(*pos_ids.entry(pos).or_insert(n));
                third_shape.push(key.split('+').any(|t| matches!(t, "P3sg" | "P3pl")));
            }
            let reading = Reading {
                item,
                shape,
                name: prior_is_name(p),
                second: prior_second_person(p),
                yields: prior_yields(p),
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
        let lexical = if r.yields {
            f64::from(LEXICAL_PREFERENCE)
        } else {
            1.0
        };
        case * lexical * if r.second { register } else { 1.0 }
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

    let units = |p: f64| rank_units(p.ln());
    let item_total: f64 = item_c.iter().sum::<f64>() + 0.5 * item_c.len() as f64;
    let mut group_totals = vec![0.0f64; pos_ids.len()];
    for (s, c) in shape_c.iter().enumerate() {
        group_totals[shape_pos[s]] += c + 0.5;
    }
    // P(tag | POS) over the overt morphemes of the POS's shapes: the backoff for a shape the
    // file does not have.
    let mut shape_list: Vec<(&String, usize)> = shapes.iter().map(|(k, &s)| (k, s)).collect();
    shape_list.sort();
    let mut tag_counts: Vec<(String, f64)> = Vec::new();
    let mut tag_totals: Vec<(String, f64)> = Vec::new();
    let add = |list: &mut Vec<(String, f64)>, key: String, start: f64, c: f64| match list
        .iter_mut()
        .find(|(k, _)| *k == key)
    {
        Some((_, total)) => *total += c,
        None => list.push((key, start + c)),
    };
    for &(key, s) in &shape_list {
        let tags_only = key.rsplit_once('/').map_or(key.as_str(), |(_, tags)| tags);
        let mut tags = tags_only.split('+');
        let pos = tags.next().unwrap_or_default();
        for tag in tags.filter(|t| !matches!(*t, "A3sg" | "Pnon" | "Nom")) {
            add(&mut tag_counts, format!("{pos}+{tag}"), 0.0, shape_c[s]);
            add(&mut tag_totals, pos.to_string(), 0.5, shape_c[s]);
        }
    }
    tag_counts.sort_by(|a, b| a.0.cmp(&b.0));
    let tag_total = |pos: &str| {
        tag_totals
            .iter()
            .find(|(p, _)| p == pos)
            .map_or(0.5, |(_, t)| *t)
    };
    let max_tag_total = tag_totals.iter().map(|(_, t)| *t).fold(0.5, f64::max);
    let mut groups: Vec<(&String, usize)> = pos_ids.iter().map(|(g, &i)| (g, i)).collect();
    groups.sort();
    let unseen_shape = groups
        .iter()
        .map(|&(_, i)| units(0.5 / group_totals[i]))
        .fold(units(0.5), i32::min);

    let mut rows: Vec<String> = vec![
        "#akana-prior\t2\tlog probabilities in rank units of 1.02^(1/16) (examples/morph_prior.rs)"
            .to_string(),
    ];
    for (key, p) in [
        ("case-name-capitalised", capitalised_name),
        ("case-name-lower", 1.0 - capitalised_name),
        ("case-other-capitalised", capitalised_other),
        ("case-other-lower", 1.0 - capitalised_other),
        ("second-person", register),
        ("lexical-preference", f64::from(LEXICAL_PREFERENCE)),
    ] {
        rows.push(format!("R\t{key}\t{}\t{p:.6}", units(p)));
    }
    rows.push(format!("U\titem\t{}", units(0.5 / item_total)));
    rows.push(format!("U\tshape\t{unseen_shape}"));
    for &(group, i) in &groups {
        rows.push(format!(
            "U\tshape:{group}\t{}",
            units(0.5 / group_totals[i])
        ));
    }
    rows.push(format!("U\ttag\t{}", units(0.5 / max_tag_total)));
    for (key, c) in &tag_counts {
        let pos = key.split('+').next().unwrap_or_default();
        rows.push(format!(
            "T\t{key}\t{}\t{c:.3}",
            units((c + 0.5) / tag_total(pos))
        ));
    }
    let mut item_rows: Vec<(&String, f64)> = items.iter().map(|(k, &i)| (k, item_c[i])).collect();
    item_rows.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(b.0)));
    for (key, c) in item_rows {
        rows.push(format!(
            "I\t{key}\t{}\t{c:.3}",
            units((c + 0.5) / item_total)
        ));
    }
    let mut shape_rows: Vec<(&String, usize)> = shape_list.clone();
    shape_rows.sort_by(|a, b| shape_c[b.1].total_cmp(&shape_c[a.1]).then(a.0.cmp(b.0)));
    for (key, s) in shape_rows {
        let c = shape_c[s];
        let p = (c + 0.5) / group_totals[shape_pos[s]];
        rows.push(format!("S\t{key}\t{}\t{c:.3}", units(p)));
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
