//! A tagger for persons, places and organisations: a structured averaged perceptron over BIO tags
//! (PER, LOC, ORG) with Viterbi decoding. It is trained on Turkish WikiNER (Turkish NLP Suite, CC
//! BY-SA 4.0; the weights in `model.tsv.gz` are data derived from it, see THIRD_PARTY_NOTICES) by
//! the ignored test `train_ner_model`, whose input `scripts/prepare_wikiner.py` writes. The
//! features read the word, its shape and affixes, the suffix after its apostrophe, its place in
//! the sentence, its neighbours, akana's name and place lists, and the analyzer.

use super::{LOC_SUFFIXES, ORG_SUFFIXES, TITLE_TRIGGERS};
use crate::morphology::{
    readings_log_prior, standard_reading, MorphParse, PrimaryPos, SecondaryPos, TurkishMorphology,
    APOSTROPHES,
};
use crate::phonology::to_turkish_lower;
use crate::pii::{is_demonym, is_given_name, is_person_name, is_place_name, is_surname};
use lazy_static::lazy_static;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::Write as _;
use std::io::Read;
use std::rc::Rc;

/// The tags, `O` first; B and I of PER, LOC and ORG.
pub(crate) const TAGS: [&str; 7] = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG"];
const T: usize = TAGS.len();
/// The transition row of a sentence's start.
const START: usize = T;

lazy_static! {
    static ref MORPHOLOGY: TurkishMorphology = TurkishMorphology::new();
    /// The trained model.
    pub(crate) static ref MODEL: Model = Model::parse(&gunzip(include_bytes!("model.tsv.gz")));
}

/// Words' facts per thread, so a word that recurs is analysed once (a few MB at most); emptied
/// when it fills.
const FACTS_CACHE: usize = 1 << 15;

thread_local! {
    static FACTS: RefCell<HashMap<String, Rc<Facts>>> = RefCell::new(HashMap::new());
}

fn gunzip(bytes: &[u8]) -> String {
    let mut text = String::new();
    if !bytes.is_empty() {
        let _ = flate2::read::GzDecoder::new(bytes).read_to_string(&mut text);
    }
    text
}

/// Feature weights per tag, and the transition weights from a tag (or a sentence's start).
pub(crate) struct Model {
    weights: HashMap<String, [f32; T]>,
    transitions: [[f32; T]; T + 1],
}

impl Model {
    /// Lines `feature TAB w_O ... w_I-ORG`, and `#t TAB from TAB weights` for transitions.
    fn parse(text: &str) -> Self {
        let mut weights = HashMap::new();
        let mut transitions = [[0.0; T]; T + 1];
        for line in text.lines() {
            let mut cells = line.split('\t');
            let Some(key) = cells.next() else { continue };
            if key == "#t" {
                let Some(from) = cells.next().and_then(|c| c.parse::<usize>().ok()) else {
                    continue;
                };
                if let (Some(row), Some(w)) = (transitions.get_mut(from), row_of(cells)) {
                    *row = w;
                }
            } else if let Some(w) = row_of(cells) {
                weights.insert(key.to_string(), w);
            }
        }
        Self {
            weights,
            transitions,
        }
    }

    /// The tag (an index into [`TAGS`]) of each word of a sentence.
    pub(crate) fn tag(&self, words: &[&str]) -> Vec<usize> {
        let facts: Vec<Rc<Facts>> = FACTS.with(|cache| {
            let mut cache = cache.borrow_mut();
            if cache.len() > FACTS_CACHE {
                cache.clear();
            }
            words.iter().map(|w| facts_of(w, &mut cache)).collect()
        });
        let mut buffer = String::new();
        let first_word = first_word(&facts);
        let scores: Vec<[f32; T]> = (0..words.len())
            .map(|i| {
                let mut s = [0.0; T];
                each_feature(&facts, i, first_word, &mut buffer, |f| {
                    if let Some(w) = self.weights.get(f) {
                        for (a, b) in s.iter_mut().zip(w) {
                            *a += b;
                        }
                    }
                });
                s
            })
            .collect();
        viterbi(&scores, &self.transitions)
    }
}

fn row_of<'a>(cells: impl Iterator<Item = &'a str>) -> Option<[f32; T]> {
    let values: Vec<f32> = cells.filter_map(|c| c.parse().ok()).collect();
    values.try_into().ok()
}

/// Whether tag `cur` may follow `prev` (or a sentence's start): an I- tag only after its B- or I-.
fn allowed(prev: usize, cur: usize) -> bool {
    match cur {
        2 => matches!(prev, 1 | 2),
        4 => matches!(prev, 3 | 4),
        6 => matches!(prev, 5 | 6),
        _ => true,
    }
}

fn viterbi(scores: &[[f32; T]], transitions: &[[f32; T]; T + 1]) -> Vec<usize> {
    let n = scores.len();
    if n == 0 {
        return Vec::new();
    }
    let mut best = vec![[f32::NEG_INFINITY; T]; n];
    let mut back = vec![[0usize; T]; n];
    for t in 0..T {
        if allowed(START, t) {
            best[0][t] = transitions[START][t] + scores[0][t];
        }
    }
    for i in 1..n {
        for t in 0..T {
            for p in 0..T {
                if best[i - 1][p] == f32::NEG_INFINITY || !allowed(p, t) {
                    continue;
                }
                let s = best[i - 1][p] + transitions[p][t] + scores[i][t];
                if s > best[i][t] {
                    best[i][t] = s;
                    back[i][t] = p;
                }
            }
        }
    }
    let mut tag = (0..T)
        .max_by(|&a, &b| best[n - 1][a].total_cmp(&best[n - 1][b]).then(b.cmp(&a)))
        .unwrap_or(0);
    let mut tags = vec![0; n];
    for i in (0..n).rev() {
        tags[i] = tag;
        tag = back[i][tag];
    }
    tags
}

/// What the features read of a word form.
struct Facts {
    /// The word before its apostrophe, lower-cased.
    lower: String,
    /// The suffix after its apostrophe, lower-cased ("" when there is none).
    suffix: String,
    apostrophe: bool,
    shape: String,
    alpha: bool,
    flags: Vec<&'static str>,
    pos: &'static str,
    freq: &'static str,
}

fn facts_of(word: &str, cache: &mut HashMap<String, Rc<Facts>>) -> Rc<Facts> {
    if let Some(f) = cache.get(word) {
        return Rc::clone(f);
    }
    let (base, suffix, apostrophe) = match word.find(APOSTROPHES) {
        Some(i) => {
            let rest = &word[i..];
            let after = rest.char_indices().nth(1).map_or("", |(j, _)| &rest[j..]);
            (&word[..i], after, true)
        }
        None => (word, "", false),
    };
    let lower = to_turkish_lower(base);
    let alpha = base.chars().any(char::is_alphabetic);
    let mut flags = Vec::new();
    let mut pos = "none";
    let mut freq = "none";
    if alpha {
        let lists: [(&'static str, bool); 8] = [
            ("gn", is_given_name(&lower)),
            ("pn", is_person_name(&lower)),
            ("sn", is_surname(&lower)),
            ("pl", is_place_name(&lower)),
            ("dm", is_demonym(&lower)),
            ("ti", TITLE_TRIGGERS.contains(lower.as_str())),
            ("os", ORG_SUFFIXES.contains(lower.as_str())),
            ("ls", LOC_SUFFIXES.contains(lower.as_str())),
        ];
        flags.extend(lists.iter().filter(|(_, on)| *on).map(|(f, _)| *f));
        let common = |p: &MorphParse| {
            !matches!(
                p.secondary_pos,
                SecondaryPos::ProperNoun | SecondaryPos::Abbreviation
            )
        };
        // One analysis of the lower-cased word serves its standard readings, its likeliest part
        // of speech and its prior; the word as written is analysed again only when it differs.
        let readings = MORPHOLOGY.analyze(&lower);
        let standard = standard_reading(&lower);
        let kept = || readings.iter().filter(|p| standard(p) && common(p));
        if kept().next().is_some() {
            flags.push("cw");
        }
        let proper = |ps: &[MorphParse]| {
            ps.iter()
                .any(|p| p.secondary_pos == SecondaryPos::ProperNoun)
        };
        if if base == lower {
            proper(&readings)
        } else {
            proper(&MORPHOLOGY.analyze(base))
        } {
            flags.push("pr");
        }
        pos = readings.first().map_or("unk", |p| pos_name(p.pos));
        freq = match readings_log_prior(&lower, kept()) {
            None => "none",
            Some(u) if u > -8 * 808 => "high",
            Some(u) if u > -11 * 808 => "mid",
            Some(u) if u > -14 * 808 => "low",
            Some(_) => "rare",
        };
    }
    let f = Rc::new(Facts {
        suffix: to_turkish_lower(suffix),
        lower,
        apostrophe,
        shape: shape_of(base),
        alpha,
        flags,
        pos,
        freq,
    });
    cache.insert(word.to_string(), Rc::clone(&f));
    f
}

fn pos_name(pos: PrimaryPos) -> &'static str {
    match pos {
        PrimaryPos::Noun => "noun",
        PrimaryPos::Adj => "adj",
        PrimaryPos::Verb => "verb",
        PrimaryPos::Adv => "adv",
        PrimaryPos::Num => "num",
        PrimaryPos::Pron => "pron",
        PrimaryPos::Postp => "postp",
        PrimaryPos::Conj => "conj",
        _ => "other",
    }
}

/// A word's letters as X (capital), x (small) and d (digit), runs merged, others as they are,
/// at most 5 (Ankara: Xx, TBMM: X, İİK.: X., 1923: d).
fn shape_of(word: &str) -> String {
    let mut out = String::new();
    for c in word.chars() {
        let k = if c.is_uppercase() {
            'X'
        } else if c.is_lowercase() {
            'x'
        } else if c.is_ascii_digit() {
            'd'
        } else {
            c
        };
        if !out.ends_with(k) {
            out.push(k);
        }
        if out.chars().count() >= 5 {
            break;
        }
    }
    out
}

/// Where a sentence's first word (with a letter) is, or its length when it has none.
fn first_word(facts: &[Rc<Facts>]) -> usize {
    facts.iter().position(|f| f.alpha).unwrap_or(facts.len())
}

/// Calls `emit` with each feature of word `i` of a sentence whose first word is `first_word`,
/// written in `buffer`.
fn each_feature(
    facts: &[Rc<Facts>],
    i: usize,
    first_word: usize,
    buffer: &mut String,
    mut emit: impl FnMut(&str),
) {
    let mut put = |args: std::fmt::Arguments| {
        buffer.clear();
        let _ = buffer.write_fmt(args);
        emit(buffer);
    };
    let f = &facts[i];
    put(format_args!("b"));
    put(format_args!("w={}", f.lower));
    if f.alpha {
        let n = f.lower.chars().count();
        let at = |k: usize| {
            f.lower
                .char_indices()
                .nth(k)
                .map_or(f.lower.len(), |(j, _)| j)
        };
        put(format_args!("p3={}", &f.lower[..at(3)]));
        put(format_args!("s3={}", &f.lower[at(n.saturating_sub(3))..]));
        put(format_args!("s4={}", &f.lower[at(n.saturating_sub(4))..]));
    }
    put(format_args!("sh={}", f.shape));
    if f.apostrophe {
        put(format_args!("ap"));
        put(format_args!("as={}", f.suffix));
    }
    // No word before it: the sentence's start, numbers and punctuation aside.
    let start = i <= first_word;
    if start {
        put(format_args!("st"));
        put(format_args!("st&sh={}", f.shape));
    }
    for flag in &f.flags {
        put(format_args!("{flag}"));
        if start {
            put(format_args!("st&{flag}"));
        }
    }
    put(format_args!("pos={}", f.pos));
    put(format_args!("fq={}", f.freq));
    if start {
        put(format_args!("st&fq={}", f.freq));
    }
    for d in [-2isize, -1, 1, 2] {
        let j = i as isize + d;
        let near = usize::try_from(j).ok().and_then(|j| facts.get(j));
        let (w, sh) = match near {
            Some(x) => (x.lower.as_str(), x.shape.as_str()),
            None if j < 0 => ("<s>", "<s>"),
            None => ("</s>", "</s>"),
        };
        put(format_args!("w{d}={w}"));
        if d.abs() == 1 {
            put(format_args!("sh{d}={sh}"));
            if let Some(x) = near {
                for flag in &x.flags {
                    put(format_args!("{flag}{d}"));
                }
                if x.apostrophe {
                    put(format_args!("ap{d}"));
                }
            }
        }
    }
}

/// The entity spans of a tag sequence: (kind, first word, word after the last).
pub(crate) fn spans(tags: &[usize]) -> Vec<(&'static str, usize, usize)> {
    let mut out = Vec::new();
    let mut open: Option<(&'static str, usize)> = None;
    for (i, &t) in tags.iter().chain([0usize].iter()).enumerate() {
        let kind = match t {
            1 | 2 => Some("PER"),
            3 | 4 => Some("LOC"),
            5 | 6 => Some("ORG"),
            _ => None,
        };
        let begins = matches!(t, 1 | 3 | 5);
        if let Some((k, s)) = open {
            if begins || kind != Some(k) {
                out.push((k, s, i));
                open = None;
            }
        }
        if begins {
            open = kind.map(|k| (k, i));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_model_is_embedded_and_reads() {
        assert!(
            MODEL.weights.len() > 10_000,
            "{} features",
            MODEL.weights.len()
        );
        assert!(MODEL.transitions.iter().flatten().any(|w| *w != 0.0));
        let tags = MODEL.tag(&["Ahmet", "Yılmaz", "Ankara'ya", "gitti", "."]);
        assert_eq!(spans(&tags), [("PER", 0, 2), ("LOC", 2, 3)]);
    }

    #[test]
    fn decoding_keeps_the_bio_order() {
        // An I- tag scores best alone; Viterbi may not start a sentence with it, nor follow O.
        let mut scores = vec![[0.0; T]; 2];
        scores[0][2] = 5.0;
        scores[1][4] = 5.0;
        let tags = viterbi(&scores, &[[0.0; T]; T + 1]);
        assert!(tags
            .iter()
            .zip(std::iter::once(START).chain(tags.iter().copied()))
            .all(|(&cur, prev)| allowed(prev, cur)));
        assert_eq!(
            spans(&[1, 2, 0, 3, 5, 6]),
            [("PER", 0, 2), ("LOC", 3, 4), ("ORG", 4, 6)]
        );
    }
}

#[cfg(test)]
mod training {
    use super::*;
    use std::io::Write;

    fn sentences(path: &std::path::Path) -> Vec<(Vec<String>, Vec<usize>)> {
        let text = std::fs::read_to_string(path).expect("NER training data");
        let mut out = Vec::new();
        let (mut words, mut tags) = (Vec::new(), Vec::new());
        for line in text.lines() {
            if line.is_empty() {
                if !words.is_empty() {
                    out.push((std::mem::take(&mut words), std::mem::take(&mut tags)));
                }
                continue;
            }
            let (w, t) = line.split_once('\t').expect("token TAB tag");
            words.push(w.to_string());
            tags.push(TAGS.iter().position(|x| *x == t).expect("known tag"));
        }
        if !words.is_empty() {
            out.push((words, tags));
        }
        out
    }

    fn micro_f1(model: &Model, data: &[(Vec<String>, Vec<usize>)]) -> f64 {
        let (mut tp, mut fp, mut fn_) = (0usize, 0usize, 0usize);
        for (words, gold) in data {
            let refs: Vec<&str> = words.iter().map(String::as_str).collect();
            let pred = spans(&model.tag(&refs));
            let gold = spans(gold);
            tp += pred.iter().filter(|s| gold.contains(s)).count();
            fp += pred.iter().filter(|s| !gold.contains(s)).count();
            fn_ += gold.iter().filter(|s| !pred.contains(s)).count();
        }
        2.0 * tp as f64 / ((2 * tp + fp + fn_).max(1)) as f64
    }

    /// Trains the model (AKANA_NER_TRAIN: the folder of train.tsv and validation.tsv; at most
    /// AKANA_NER_EPOCHS epochs, 12 by default; the epoch with the best validation F1 is kept) and
    /// writes it to AKANA_NER_MODEL_OUT (gzip). Features seen once in training are left out.
    #[test]
    #[ignore]
    fn train_ner_model() {
        let (Ok(dir), Ok(out)) = (
            std::env::var("AKANA_NER_TRAIN"),
            std::env::var("AKANA_NER_MODEL_OUT"),
        ) else {
            return;
        };
        let epochs: usize = std::env::var("AKANA_NER_EPOCHS")
            .ok()
            .and_then(|e| e.parse().ok())
            .unwrap_or(12);
        let dir = std::path::Path::new(&dir);
        let train = sentences(&dir.join("train.tsv"));
        let validation = sentences(&dir.join("validation.tsv"));
        let mut cache = HashMap::new();
        // Two passes over the sentences, so that only one sentence's feature strings are held at
        // a time: count the features, then index those seen more than once.
        let features_of = |words: &[String], cache: &mut HashMap<String, Rc<Facts>>| {
            let facts: Vec<Rc<Facts>> = words.iter().map(|w| facts_of(w, cache)).collect();
            let mut buffer = String::new();
            let first_word = first_word(&facts);
            (0..words.len())
                .map(|i| {
                    let mut fs = Vec::new();
                    each_feature(&facts, i, first_word, &mut buffer, |f| {
                        fs.push(f.to_string())
                    });
                    fs
                })
                .collect::<Vec<_>>()
        };
        let mut counts: HashMap<String, u32> = HashMap::new();
        for (words, _) in &train {
            for fs in features_of(words, &mut cache) {
                for f in fs {
                    *counts.entry(f).or_default() += 1;
                }
            }
        }
        let mut ids: HashMap<String, usize> = HashMap::new();
        let indexed: Vec<Vec<Vec<usize>>> = train
            .iter()
            .map(|(words, _)| {
                features_of(words, &mut cache)
                    .into_iter()
                    .map(|fs| {
                        fs.into_iter()
                            .filter(|f| counts[f] > 1)
                            .map(|f| {
                                let next = ids.len();
                                *ids.entry(f).or_insert(next)
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();
        drop(counts);
        let names: Vec<String> = {
            let mut v = vec![String::new(); ids.len()];
            for (f, &i) in &ids {
                v[i] = f.clone();
            }
            v
        };
        let mut w = vec![[0.0f32; T]; ids.len()];
        let mut acc = vec![[0.0f32; T]; ids.len()];
        let mut wt = [[0.0f32; T]; T + 1];
        let mut acct = [[0.0f32; T]; T + 1];
        let mut c = 1.0f32;
        let averaged = |w: &[[f32; T]],
                        acc: &[[f32; T]],
                        wt: &[[f32; T]; T + 1],
                        acct: &[[f32; T]; T + 1],
                        c: f32| {
            let mut weights = HashMap::new();
            for (i, (a, b)) in w.iter().zip(acc).enumerate() {
                let row: [f32; T] = std::array::from_fn(|t| a[t] - b[t] / c);
                if row.iter().any(|x| x.abs() > 1e-3) {
                    weights.insert(names[i].clone(), row);
                }
            }
            let mut transitions = [[0.0; T]; T + 1];
            for p in 0..=T {
                for t in 0..T {
                    transitions[p][t] = wt[p][t] - acct[p][t] / c;
                }
            }
            Model {
                weights,
                transitions,
            }
        };
        let mut order: Vec<usize> = (0..train.len()).collect();
        let mut seed = 20261030u64;
        let mut best: Option<(f64, usize, Model)> = None;
        for epoch in 1..=epochs {
            for k in (1..order.len()).rev() {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                order.swap(k, (seed >> 33) as usize % (k + 1));
            }
            for &s in &order {
                let gold = &train[s].1;
                let scores: Vec<[f32; T]> = indexed[s]
                    .iter()
                    .map(|fs| {
                        let mut v = [0.0; T];
                        for &f in fs {
                            for t in 0..T {
                                v[t] += w[f][t];
                            }
                        }
                        v
                    })
                    .collect();
                let pred = viterbi(&scores, &wt);
                if pred != *gold {
                    for i in 0..gold.len() {
                        if pred[i] != gold[i] {
                            for &f in &indexed[s][i] {
                                w[f][gold[i]] += 1.0;
                                acc[f][gold[i]] += c;
                                w[f][pred[i]] -= 1.0;
                                acc[f][pred[i]] -= c;
                            }
                        }
                        let pg = if i == 0 { START } else { gold[i - 1] };
                        let pp = if i == 0 { START } else { pred[i - 1] };
                        if (pg, gold[i]) != (pp, pred[i]) {
                            wt[pg][gold[i]] += 1.0;
                            acct[pg][gold[i]] += c;
                            wt[pp][pred[i]] -= 1.0;
                            acct[pp][pred[i]] -= c;
                        }
                    }
                }
                c += 1.0;
            }
            let model = averaged(&w, &acc, &wt, &acct, c);
            let f1 = micro_f1(&model, &validation);
            println!(
                "epoch {epoch}: validation micro F1 {f1:.4} ({} features)",
                model.weights.len()
            );
            if best.as_ref().is_none_or(|(b, _, _)| f1 > *b) {
                best = Some((f1, epoch, model));
            }
        }
        let (f1, epoch, model) = best.expect("at least one epoch");
        println!("kept epoch {epoch} (validation micro F1 {f1:.4})");
        let mut keys: Vec<&String> = model.weights.keys().collect();
        keys.sort();
        let mut text = String::new();
        for p in 0..=T {
            let row: Vec<String> = model.transitions[p]
                .iter()
                .map(|x| format!("{x:.3}"))
                .collect();
            text.push_str(&format!("#t\t{p}\t{}\n", row.join("\t")));
        }
        for k in keys {
            let row: Vec<String> = model.weights[k].iter().map(|x| format!("{x:.3}")).collect();
            text.push_str(&format!("{k}\t{}\n", row.join("\t")));
        }
        let file = std::fs::File::create(out).expect("model output");
        let mut gz = flate2::write::GzEncoder::new(file, flate2::Compression::best());
        gz.write_all(text.as_bytes()).expect("model write");
        gz.finish().expect("model write");
    }
}
