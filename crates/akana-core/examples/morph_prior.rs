//! Estimates the analyzer's frequency prior from a corpus sample.
//!
//! ```text
//! cargo run --release -p akana-core --example morph_prior -- \
//!     --types data/morphology/prior_train_types.tsv.gz \
//!     --out crates/akana-core/src/morphology/prior.tsv.gz
//! ```
//!
//! The training counts are committed (`data/morphology/prior_train_types.tsv.gz`: word and
//! count, from the train split of `scripts/morph_corpus_sample.py`), so re-estimating needs no
//! corpus, and CI checks that the committed prior is what they give. `--types` also takes a
//! directory of `*.types.tsv` files, as the sampler writes them.
//!
//! The court sample is mixed with general text (decision 3: one prior for legal and general
//! text): `--general` (`data/morphology/prior_general_types.tsv.gz`, the Universal Dependencies
//! treebanks' sentences; scripts/prior_general_types.py) is weighed up to `GENERAL_SHARE` of the
//! mixed counts (`--general-share`), so that general usage (a question, an address, a story)
//! counts beside court style.
//!
//! Every word type of the training sample is analysed without the prior. A reading's
//! probability is P(item) * P(shape | group): how often its dictionary item (lemma, root, POS,
//! secondary POS) is meant, times how often a word of that group takes its shape (the tag
//! sequence, Noun+P3sg+Loc). An item is one dictionary entry: homographs (kalp "heart" and kalp
//! "counterfeit") have keys of their own (`prior_item_key`). The group is the POS, except that
//! the personal pronouns share one
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
//!   possessive. Court text all but never addresses its reader, which put the factor near
//!   1/100 and made a bare imperative unreadable in a sentence that does (yıka, bekle); it is
//!   floored at `SECOND_PERSON_FLOOR`. The negative imperative, which every verbal noun in -mA
//!   spells (`prior_negative_imperative`), takes none: the factor had put "Unutma !" at 1e-16,
//!   past any context, and the treebanks' tags now tell gel-me's readings apart.
//!
//! Where the words are written alike, EM cannot tell a word's POS twins apart: the one shape
//! that explains every bare token wins, so ilk, tek and yeni went to the adverb and insan to
//! the adjective. The treebanks tagged them by hand: for a word written in lower case that
//! `--twins` lists (data/morphology/pos_twins.tsv, how often the Universal Dependencies train
//! and dev splits tag each form as each class, a verb by its VerbForm; scripts/pos_twins.py),
//! the count of its readings that are not names is split among their classes as tagged there,
//! smoothed toward the model's own split by `TWIN_WEIGHT` tokens, and within a class as the
//! model splits it (`twin_factors`). That also separates readings that are in every word that
//! has one, which EM leaves where it started (git-tik the past 1pl or the participle). Items
//! and shapes cannot always carry such a split back to the word: a
//! verb's or a pronoun's bare shape is rare among the shapes of its POS, a conjunction's is
//! its only one, so değil came out the conjunction and ne the adverb however the counts were
//! split. The file therefore also holds, for each such word of the sample, what each class of
//! its readings adds in the analyzer to take its treebank share (`W` rows). The table is
//! DERIVED FROM CC BY-SA DATA, and so is the prior (THIRD_PARTY_NOTICES).
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
    prior_capitalised, prior_class, prior_inventory, prior_is_name, prior_item_key,
    prior_negative_imperative, prior_second_person, prior_shape_group, prior_shape_key,
    prior_yields, rank_units, TurkishMorphology, LEXICAL_PREFERENCE, PRIOR_CLASSES,
    RANK_UNITS_PER_NAT, RANK_UNITS_PER_STEP,
};
use std::collections::HashMap;
use std::io::Write;

/// The reading classes ([`prior_class`]).
const CLASSES: usize = PRIOR_CLASSES.len();

/// How many tokens' weight the model's own split of a tagged word's readings has against the
/// treebanks' counts (`--twin-weight`).
const TWIN_WEIGHT: f64 = 5.0;

/// The general text's share of the mixed counts (`--general-share`).
const GENERAL_SHARE: f64 = 0.1;

/// The least register factor of a second-person reading other than a negative imperative.
const SECOND_PERSON_FLOOR: f64 = 0.05;

/// How many of a word's tokens the treebanks must tag in the classes of its readings before
/// they split it: fewer say little (sözlü: four adjectives, two nouns).
const MIN_TWIN_TOKENS: f64 = 10.0;

/// The classes the twin table counts the treebanks' tags in (scripts/pos_twins.py), and the
/// reading classes each one can be: a participle (VerbPart) is the analyzer's participle, as an
/// adjective or a noun; a word the treebank takes as lexicalised (koruma, geçen) is a NOUN, ADJ
/// or ADV, which the analyzer may read as a verb's.
const TAGGED: [(&str, &[&str]); 14] = [
    ("Noun", &["Noun", "VerbNoun"]),
    ("Adj", &["Adj", "VerbAdj"]),
    ("Adv", &["Adv", "VerbAdv"]),
    ("Verb", &["Verb"]),
    ("VerbPart", &["VerbAdj", "VerbNoun"]),
    ("VerbNoun", &["VerbNoun"]),
    ("VerbAdv", &["VerbAdv"]),
    ("Pron", &["Pron"]),
    ("Det", &["Det"]),
    ("Num", &["Num"]),
    ("Conj", &["Conj"]),
    ("Postp", &["Postp"]),
    ("Interj", &["Interj"]),
    ("Part", &["Part"]),
];

/// Whether a reading can be what the treebanks tag as `TAGGED[tag]`.
fn tagged_as(tag: usize, r: &Reading) -> bool {
    (0..CLASSES).any(|c| r.classes & (1 << c) != 0 && TAGGED[tag].1.contains(&PRIOR_CLASSES[c]))
}

/// What each of a tagged word's readings is multiplied by, given their weights. The treebanks
/// split a word among the classes of one lemma (değil the verb or the conjunction, ilk the
/// adjective or the adverb); which lemma it is (alacak "receivable" or al-acak "will take")
/// depends on the text, and the model, which counts the lemma's every form in the sample, keeps
/// it. So each lemma's readings that are not names keep their weight together; each token the
/// treebanks tag the word with (`seen`, per `TAGGED` class) goes to the lemma's readings it can
/// be in proportion to their weight, and `weight` more tokens go to all of them so (the
/// model's own split, which a word tagged a few times keeps close to). The names' weights, and
/// those of a lemma whose classes the treebanks tag fewer than `MIN_TWIN_TOKENS` times, stand.
fn twin_factors(
    readings: &[Reading],
    weights: &[f64],
    seen: &[f64; TAGGED.len()],
    weight: f64,
) -> Vec<f64> {
    let mut factors = vec![1.0f64; readings.len()];
    let mut lemmas: Vec<usize> = readings
        .iter()
        .filter(|r| !r.name)
        .map(|r| r.lemma)
        .collect();
    lemmas.sort_unstable();
    lemmas.dedup();
    for lemma in lemmas {
        let member = |r: &Reading| !r.name && r.lemma == lemma;
        let common: f64 = readings
            .iter()
            .zip(weights)
            .filter(|(r, _)| member(r))
            .map(|(_, w)| w)
            .sum();
        let tag_mass: Vec<f64> = (0..TAGGED.len())
            .map(|t| {
                readings
                    .iter()
                    .zip(weights)
                    .filter(|(r, _)| member(r) && tagged_as(t, r))
                    .map(|(_, w)| w)
                    .sum()
            })
            .collect();
        let read = |t: &usize| tag_mass[*t] > 0.0;
        let tagged: f64 = (0..TAGGED.len()).filter(read).map(|t| seen[t]).sum();
        if tagged < MIN_TWIN_TOKENS {
            continue;
        }
        for (r, f) in readings.iter().zip(factors.iter_mut()) {
            if !member(r) {
                continue;
            }
            let per_weight: f64 = (0..TAGGED.len())
                .filter(read)
                .filter(|&t| tagged_as(t, r))
                .map(|t| seen[t] / tag_mass[t])
                .sum();
            *f = (per_weight * common + weight) / (tagged + weight);
        }
    }
    factors
}

/// One reading of a word type: item id, shape id, lemma id, whether it names something, whether
/// the register factor applies to it (and whether it is a negative imperative), whether the
/// lexicon ranks it below its twin; and the classes ([`prior_class`]) of the analyzer's
/// readings it stands for, as bits (a participle's adjective and noun twins have one item and
/// shape, and are one reading).
#[derive(Clone, Copy)]
struct Reading {
    item: usize,
    shape: usize,
    lemma: usize,
    name: bool,
    second: bool,
    negative: bool,
    yields: bool,
    classes: u32,
}

impl Reading {
    /// The same reading but for its classes.
    fn same(&self, other: &Reading) -> bool {
        (
            self.item,
            self.shape,
            self.name,
            self.second,
            self.negative,
            self.yields,
        ) == (
            other.item,
            other.shape,
            other.name,
            other.second,
            other.negative,
            other.yields,
        )
    }
}

/// Word counts from a gzipped `word<TAB>count` list, or from every `*.types.tsv` file of a
/// directory, in a fixed order. Lines without a tab (a header) are skipped.
fn read_types(path: &str) -> Vec<(String, f64)> {
    let texts: Vec<String> = if path.ends_with(".gz") {
        use std::io::Read;
        let mut text = String::new();
        flate2::read::GzDecoder::new(std::fs::File::open(path).expect("types file"))
            .read_to_string(&mut text)
            .expect("read types file");
        vec![text]
    } else {
        let mut files: Vec<_> = std::fs::read_dir(path)
            .expect("types dir")
            .map(|e| e.expect("entry").path())
            .filter(|p| p.to_string_lossy().ends_with(".types.tsv"))
            .collect();
        files.sort();
        files
            .iter()
            .map(|path| std::fs::read_to_string(path).expect("read"))
            .collect()
    };
    texts
        .iter()
        .flat_map(|text| text.lines())
        .filter_map(|line| line.split_once('\t'))
        .map(|(w, c)| (w.to_string(), c.parse::<f64>().unwrap_or(0.0)))
        .collect()
}

/// One word type: its count, whether it is capitalised, its readings (and how many of the
/// analyzer's readings each stands for), and how often the treebanks tag it as each class, if
/// they do.
struct Word {
    word: String,
    count: f64,
    capitalised: bool,
    readings: Vec<Reading>,
    copies: Vec<f64>,
    twins: Option<[f64; TAGGED.len()]>,
}

fn main() {
    let mut types_dir = String::from(".cache/morph_sample/train");
    let mut out = String::from("crates/akana-core/src/morphology/prior.tsv.gz");
    let mut twins_file = String::from("data/morphology/pos_twins.tsv");
    let mut general_file = String::from("data/morphology/prior_general_types.tsv.gz");
    let mut general_share = GENERAL_SHARE;
    let mut iterations = 10usize;
    let mut twin_weight = TWIN_WEIGHT;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--types" => types_dir = args.next().expect("--types DIR|FILE.gz"),
            "--out" => out = args.next().expect("--out FILE"),
            "--twins" => twins_file = args.next().expect("--twins FILE"),
            "--general" => general_file = args.next().expect("--general FILE.gz"),
            "--general-share" => {
                general_share = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .expect("--general-share SHARE")
            }
            "--twin-weight" => {
                twin_weight = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .expect("--twin-weight TOKENS")
            }
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

    // word -> count over every *.types.tsv file of the directory, or over the committed list
    // (data/morphology/prior_train_types.tsv.gz, the same counts in one gzipped file); then the
    // general text's counts, weighed up to their share.
    let mut counts: HashMap<String, f64> = HashMap::new();
    let court = read_types(&types_dir);
    let court_tokens: f64 = court.iter().map(|(_, c)| c).sum();
    for (w, c) in court {
        *counts.entry(w).or_default() += c;
    }
    let general = if general_share > 0.0 {
        read_types(&general_file)
    } else {
        Vec::new()
    };
    let general_tokens: f64 = general.iter().map(|(_, c)| c).sum();
    let scale = if general_tokens > 0.0 {
        general_share / (1.0 - general_share) * court_tokens / general_tokens
    } else {
        0.0
    };
    for (w, c) in general {
        *counts.entry(w).or_default() += scale * c;
    }
    // form -> tokens per class, as the treebanks tag it.
    let mut twins: HashMap<String, [f64; TAGGED.len()]> = HashMap::new();
    let twin_text = std::fs::read_to_string(&twins_file).expect("twins file");
    for line in twin_text.lines().filter(|l| !l.starts_with('#')) {
        let (form, seen) = line.split_once('\t').expect("form<TAB>counts");
        let mut row = [0.0; TAGGED.len()];
        for pair in seen.split(' ') {
            let (class, n) = pair.split_once(':').expect("Class:count");
            let c = TAGGED.iter().position(|(t, _)| *t == class).expect("class");
            row[c] = n.parse().expect("count");
        }
        twins.insert(form.to_string(), row);
    }

    let mut types: Vec<(&String, f64)> = counts.iter().map(|(w, &c)| (w, c)).collect();
    types.sort_by(|a, b| a.0.cmp(b.0));

    // Each shape's POS (its first tag) as an id, for P(shape | POS).
    let mut items: HashMap<String, usize> = HashMap::new();
    let mut lemmas: HashMap<String, usize> = HashMap::new();
    let mut lemma_names: Vec<String> = Vec::new();
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
        let mut copies = Vec::new();
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
            let lemma = *lemmas.entry(p.lemma.clone()).or_insert_with(|| {
                lemma_names.push(p.lemma.clone());
                lemma_names.len() - 1
            });
            let reading = Reading {
                item,
                shape,
                lemma,
                name: prior_is_name(p),
                second: prior_second_person(p),
                negative: prior_negative_imperative(p),
                yields: prior_yields(p),
                classes: 1 << prior_class(p),
            };
            match readings.iter().position(|r: &Reading| r.same(&reading)) {
                Some(i) => {
                    readings[i].classes |= reading.classes;
                    copies[i] += 1.0;
                }
                None => {
                    readings.push(reading);
                    copies.push(1.0);
                }
            }
        }
        let lower = !w.chars().any(char::is_uppercase);
        words.push(Word {
            word: w.to_string(),
            count,
            capitalised: prior_capitalised(w),
            readings,
            copies,
            twins: if lower {
                twins.get(w.as_str()).copied()
            } else {
                None
            },
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
    let register_floored = register.max(SECOND_PERSON_FLOOR);
    let register_of = |r: &Reading| match (r.second, r.negative) {
        (true, false) => register_floored,
        _ => 1.0,
    };
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
        case * lexical * register_of(r)
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
            let mut weights: Vec<f64> = w.readings.iter().map(weight).collect();
            if let Some(seen) = &w.twins {
                let factors = twin_factors(&w.readings, &weights, seen, twin_weight);
                for (x, f) in weights.iter_mut().zip(factors) {
                    *x *= f;
                }
            }
            let z: f64 = weights.iter().sum();
            for (r, x) in w.readings.iter().zip(&weights) {
                let share = x / z;
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
        format!("#inventory\t{}", prior_inventory()),
        "#licence\tDERIVED FROM CC BY-SA DATA: the Universal Dependencies Turkish treebanks' \
         tags (CC BY-SA 4.0); see THIRD_PARTY_NOTICES"
            .to_string(),
    ];
    for (key, p) in [
        ("case-name-capitalised", capitalised_name),
        ("case-name-lower", 1.0 - capitalised_name),
        ("case-other-capitalised", capitalised_other),
        ("case-other-lower", 1.0 - capitalised_other),
        ("second-person", register_floored),
        ("second-person-negative-imperative", 1.0),
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
    // The word-form corrections: what the analyzer adds to the readings of each lemma and class
    // of a tagged word (not to its names) so that they take the treebanks' shares, from the
    // logs it will compute for them out of the rows above. A form whose readings move by less
    // than one 2% step is left out.
    let case_units = |name: bool| {
        units(if name {
            1.0 - capitalised_name
        } else {
            1.0 - capitalised_other
        })
    };
    let mut form_rows = 0usize;
    for w in &words {
        let Some(seen) = &w.twins else { continue };
        let logs: Vec<i32> = w
            .readings
            .iter()
            .map(|r| {
                units(item_p[r.item])
                    + units(shape_p[r.shape])
                    + case_units(r.name)
                    + units(register_of(r))
                    + if r.yields {
                        units(f64::from(LEXICAL_PREFERENCE))
                    } else {
                        0
                    }
            })
            .collect();
        let best = logs.iter().copied().max().unwrap_or(0);
        let weights: Vec<f64> = logs
            .iter()
            .zip(&w.copies)
            .map(|(&log, copies)| copies * (f64::from(log - best) / RANK_UNITS_PER_NAT).exp())
            .collect();
        let factors = twin_factors(&w.readings, &weights, seen, twin_weight);
        // Each lemma and class once: the first of its readings decides.
        let mut adds: Vec<(&str, usize, i32)> = Vec::new();
        for (r, f) in w.readings.iter().zip(factors) {
            for c in (0..CLASSES).filter(|c| r.classes & (1 << c) != 0) {
                let lemma = lemma_names[r.lemma].as_str();
                if !r.name && !adds.iter().any(|&(l, k, _)| l == lemma && k == c) {
                    adds.push((lemma, c, units(f)));
                }
            }
        }
        adds.sort_unstable();
        if adds.iter().any(|&(_, _, a)| a.abs() >= RANK_UNITS_PER_STEP) {
            for (lemma, c, add) in adds {
                rows.push(format!("W\t{}/{lemma}/{}\t{add}", w.word, PRIOR_CLASSES[c]));
            }
            form_rows += 1;
        }
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
    let tagged = words.iter().filter(|w| w.twins.is_some());
    println!(
        "{} word types ({} tagged in the treebanks, {form_rows} corrected), {} items, {} \
         shapes; capitalised: names {capitalised_name:.3}, other words {capitalised_other:.3}; \
         second-person factor {register:.6} ({only_second:.0} / {only_third:.0} tokens, \
         floored {register_floored:.3}); general text {general_tokens:.0} tokens x {scale:.2}; {} \
         rows written to {out}",
        words.len(),
        tagged.count(),
        items.len(),
        shapes.len(),
        rows.len()
    );
}
