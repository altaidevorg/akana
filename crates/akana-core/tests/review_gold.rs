//! Review 2's gold sets as ratchets (`tests/data/morphology/review/`):
//!
//! - `sentences_court.tsv` and `sentences_colloquial.tsv`: hand-checked readings of 90 sentences
//!   in context. RagTurk's 60 Wikipedia sentences are CC BY-SA and stay out of git: they are
//!   scored the same way when `.cache/morph_gold/sentences_ragturk.tsv` exists, against the
//!   failure list beside it.
//! - `zemberek_tests.tsv`: the checks of zemberek-nlp's own morphology tests, each run on the
//!   test's tiny lexicon (as Zemberek runs them) and on the built-in one.
//! - `paradigm_lists.tsv`: forms to accept with a lemma and analysis, and forms to reject.
//!
//! Each set has a `*_failures.txt` beside it listing every item the analyzer gets wrong today,
//! with a summary in its comment lines. A test fails when the current failures differ from the
//! list in either direction, so a change shows which items it fixes and which it breaks. After
//! a deliberate change, `AKANA_GOLD_WRITE=1 cargo test -p akana-core --test review_gold`
//! rewrites the lists, and the commit gives their diff.

use akana_core::morphology::{
    MorphParse, MorphologicalDisambiguator, RootLexicon, TurkishMorphology,
};
use akana_core::phonology::to_turkish_lower;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::{Path, PathBuf};

fn review(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/morphology/review")
        .join(name)
}

fn percent(part: usize, total: usize) -> String {
    format!("{:.1} %", 100.0 * part as f64 / total.max(1) as f64)
}

/// Compares `failures` with the list in `path` (its non-comment lines), or rewrites the list
/// with `summary` as comments when `AKANA_GOLD_WRITE` is set.
fn ratchet(path: &Path, summary: &[String], failures: &BTreeSet<String>) {
    let name = path.file_name().unwrap_or_default().to_string_lossy();
    if std::env::var_os("AKANA_GOLD_WRITE").is_some() {
        let mut text =
            String::from("# Gold items the analyzer gets wrong today (tests/review_gold.rs).\n");
        for line in summary {
            text.push_str(&format!("# {line}\n"));
        }
        for f in failures {
            text.push_str(f);
            text.push('\n');
        }
        std::fs::write(path, text).unwrap_or_else(|e| panic!("write {name}: {e}"));
        return;
    }
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {name}: {e}"));
    let known: BTreeSet<String> = text
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(str::to_string)
        .collect();
    let broken: Vec<&String> = failures.difference(&known).collect();
    let fixed: Vec<&String> = known.difference(failures).collect();
    let mut report = String::new();
    for f in broken.iter().take(40) {
        report.push_str(&format!("  newly failing: {f}\n"));
    }
    for f in fixed.iter().take(40) {
        report.push_str(&format!("  now passing:   {f}\n"));
    }
    assert!(
        broken.is_empty() && fixed.is_empty(),
        "{name}: {} newly failing, {} now passing ({} known, {} now; now: {}):\n{report}\
         If deliberate, rewrite with AKANA_GOLD_WRITE=1 and give the diff in the commit.",
        broken.len(),
        fixed.len(),
        known.len(),
        failures.len(),
        summary.join("; ")
    );
}

/// A reading as the gold writes it: lemma, root POS, tags, and P for a proper noun or A for an
/// abbreviation. Taken from `formatted` (`[lemma:Pos] root:Tag+Tag`), as review 2 scored it.
fn key_of(p: &MorphParse) -> [String; 4] {
    let (head, body) = p.formatted.split_once("] ").unwrap_or((&p.formatted, ""));
    let (lemma, pos) = head
        .trim_start_matches('[')
        .rsplit_once(':')
        .unwrap_or((head, ""));
    let tags = body.split_once(':').map_or(body, |(_, t)| t);
    let flag = match p.secondary_pos.as_str() {
        "Prop" => "P",
        "Abbrv" => "A",
        _ => "",
    };
    [lemma, pos, tags, flag].map(str::to_string)
}

/// Scores one sentence gold file: the disambiguated reading of each scored token must be one of
/// its gold readings. Failures are `sid:token-index<TAB>token<TAB>what is wrong<TAB>reading`
/// (what: tags, when lemma and POS are right; pos; lemma; lemma+pos), and
/// `missing<TAB>sid:token-index<TAB>token` when no analysis of the token is a gold reading.
fn score_sentences(path: &Path) -> (Vec<String>, BTreeSet<String>) {
    let name = path.file_name().unwrap_or_default().to_string_lossy();
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {name}: {e}"));
    let morphology = TurkishMorphology::new();
    let disambiguator = MorphologicalDisambiguator::new();
    let mut failures = BTreeSet::new();
    let (mut tokens, mut exact, mut lemma_ok, mut pos_ok, mut oracle) = (0, 0, 0, 0, 0);
    let mut sentence: Vec<&str> = Vec::new();
    let mut chosen: Vec<MorphParse> = Vec::new();
    let mut sid = 0;
    for line in text.lines() {
        if line.is_empty() || line.starts_with('#') || line.starts_with("@ ") {
            continue;
        }
        if let Some(s) = line.strip_prefix("> ") {
            sentence = s.split(' ').collect();
            chosen = disambiguator.disambiguate(&sentence);
            sid += 1;
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        let [tid, token, kind, gold] = fields[..] else {
            panic!("malformed {name} line: {line:?}");
        };
        let tid: usize = tid.parse().expect("token index");
        assert_eq!(sentence.get(tid), Some(&token), "{name}: {line:?}");
        if kind == "S" || kind == "T" {
            continue;
        }
        let alts: Vec<Vec<&str>> = gold
            .split(" || ")
            .map(|a| {
                let mut f: Vec<&str> = a.split('|').collect();
                f.resize(4, "");
                f
            })
            .collect();
        let same_lemma = |k: &[String; 4]| {
            alts.iter()
                .any(|a| to_turkish_lower(&k[0]) == to_turkish_lower(a[0]))
        };
        let same_pos = |k: &[String; 4]| alts.iter().any(|a| k[1] == a[1]);
        let same = |k: &[String; 4]| {
            alts.iter().any(|a| {
                to_turkish_lower(&k[0]) == to_turkish_lower(a[0])
                    && k[1] == a[1]
                    && k[2] == a[2]
                    && k[3] == a[3]
            })
        };
        tokens += 1;
        let readings = morphology.analyze(token);
        if readings.iter().any(|p| same(&key_of(p))) {
            oracle += 1;
        } else {
            failures.insert(format!("missing\t{sid}:{tid}\t{token}"));
        }
        let parse = &chosen[tid];
        let key = if parse.primary_pos.as_str() == "Unk" {
            [String::new(), "Unk".into(), String::new(), String::new()]
        } else {
            key_of(parse)
        };
        let (l, p) = (same_lemma(&key), same_pos(&key));
        lemma_ok += usize::from(l);
        pos_ok += usize::from(p);
        if same(&key) {
            exact += 1;
        } else {
            let what = match (l, p) {
                (true, true) => "tags",
                (true, false) => "pos",
                (false, true) => "lemma",
                (false, false) => "lemma+pos",
            };
            failures.insert(format!("{sid}:{tid}\t{token}\t{what}\t{}", key.join("|")));
        }
    }
    let summary = vec![
        format!(
            "{tokens} tokens in context: exact {}, lemma {}, POS {}",
            percent(exact, tokens),
            percent(lemma_ok, tokens),
            percent(pos_ok, tokens)
        ),
        format!(
            "a gold reading is among the analyses for {}",
            percent(oracle, tokens)
        ),
    ];
    (summary, failures)
}

#[test]
fn court_sentences_keep_their_known_failures() {
    let (summary, failures) = score_sentences(&review("sentences_court.tsv"));
    ratchet(&review("sentences_court_failures.txt"), &summary, &failures);
}

#[test]
fn colloquial_sentences_keep_their_known_failures() {
    let (summary, failures) = score_sentences(&review("sentences_colloquial.tsv"));
    ratchet(
        &review("sentences_colloquial_failures.txt"),
        &summary,
        &failures,
    );
}

/// RagTurk's Wikipedia sentences, local only (see the module comment).
#[test]
fn ragturk_sentences_keep_their_known_failures() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.cache/morph_gold");
    let gold = dir.join("sentences_ragturk.tsv");
    if !gold.exists() {
        eprintln!("skipped: {} is not there", gold.display());
        return;
    }
    let (summary, failures) = score_sentences(&gold);
    ratchet(
        &dir.join("sentences_ragturk_failures.txt"),
        &summary,
        &failures,
    );
}

/// A Zemberek analysis (`MorphParse::analysis`) as its morpheme ids, `Noun + A3sg + P1sg`.
fn lexical(analysis: &str) -> String {
    let body = analysis.split_once("] ").map_or(analysis, |x| x.1);
    body.split(['+', '|', '→'])
        .filter(|t| !t.is_empty())
        .map(|t| t.rsplit_once(':').map_or(t, |x| x.1))
        .collect::<Vec<_>>()
        .join(" + ")
}

fn tiny_morphology(dict: &[String]) -> TurkishMorphology {
    if dict.iter().any(|l| l == "DEFAULT") {
        return TurkishMorphology::new();
    }
    let mut lexicon = RootLexicon::empty();
    let text: Vec<&str> = dict
        .iter()
        .filter(|l| *l != "EMPTY")
        .map(String::as_str)
        .collect();
    lexicon.load_from_str(&text.join("\n"));
    TurkishMorphology::with_lexicon(lexicon)
}

/// Zemberek's checks, each on the test's tiny lexicon (`tiny`) and on the built-in one
/// (`full`, where only readings of the test's own lemmas count, a SINGLE is an ANY and a COUNT
/// asks for one reading at least). Failures are `mode<TAB>file/test<TAB>kind<TAB>word<TAB>arg`.
#[test]
fn zemberek_tests_keep_their_known_failures() {
    let path = review("zemberek_tests.tsv");
    let text = std::fs::read_to_string(&path).expect("read zemberek_tests.tsv");
    let full = TurkishMorphology::new();
    let (mut file, mut test) = (String::new(), String::new());
    let mut dict: Vec<String> = Vec::new();
    let mut dict_fresh = false;
    let mut tiny: Option<TurkishMorphology> = None;
    let mut tallies: BTreeMap<(String, &str), (usize, usize)> = BTreeMap::new();
    let mut failures = BTreeSet::new();
    for line in text.lines() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        match cols[0] {
            "FILE" => file = cols[1].to_string(),
            "TEST" | "RESET" => {
                if cols[0] == "TEST" {
                    test = cols[1].to_string();
                }
                dict.clear();
                dict_fresh = true;
                tiny = None;
            }
            "DICT" => {
                if !dict_fresh {
                    dict.clear();
                }
                dict_fresh = true;
                dict.push(cols[1].to_string());
                tiny = None;
            }
            kind => {
                dict_fresh = false;
                let tiny = tiny.get_or_insert_with(|| tiny_morphology(&dict));
                let word = cols[1];
                let arg = cols.get(2).copied().unwrap_or("");
                let lemmas: HashSet<String> = dict
                    .iter()
                    .filter_map(|l| l.split_whitespace().next())
                    .map(|s| to_turkish_lower(s).replace('â', "a"))
                    .collect();
                for (mode, m) in [("tiny", &*tiny), ("full", &full)] {
                    if mode == "full" && dict.iter().any(|d| d == "DEFAULT" || d == "EMPTY") {
                        continue;
                    }
                    let parses = m.analyze(word);
                    let own: Vec<&MorphParse> = parses
                        .iter()
                        .filter(|p| mode == "tiny" || lemmas.contains(&p.lemma))
                        .collect();
                    let any_tail = |t: &str| own.iter().any(|p| lexical(&p.analysis).ends_with(t));
                    let ok = match kind {
                        "ANY" | "TAILANY" => any_tail(arg),
                        "SINGLE" => any_tail(arg) && (mode == "full" || own.len() == 1),
                        "NOTANY" => !any_tail(arg),
                        "SUCCESS" => !own.is_empty(),
                        "FAIL" => own.is_empty(),
                        "COUNT" => {
                            let n: usize = arg.parse().expect("COUNT n");
                            if mode == "tiny" {
                                own.len() == n
                            } else {
                                !own.is_empty()
                            }
                        }
                        "LEMMA" => {
                            !own.is_empty()
                                && own
                                    .iter()
                                    .all(|p| p.lemma == arg || p.display_lemma() == arg)
                        }
                        "NOTCONTAINS" => {
                            !own.is_empty()
                                && !own
                                    .iter()
                                    .any(|p| lexical(&p.analysis).split(" + ").any(|x| x == arg))
                        }
                        _ => panic!("unknown check {kind:?} in zemberek_tests.tsv"),
                    };
                    let tally = tallies.entry((file.clone(), mode)).or_default();
                    tally.1 += 1;
                    if ok {
                        tally.0 += 1;
                    } else {
                        failures.insert(format!("{mode}\t{file}/{test}\t{kind}\t{word}\t{arg}"));
                    }
                }
            }
        }
    }
    let mut summary = Vec::new();
    for mode in ["tiny", "full"] {
        let (pass, total) = tallies
            .iter()
            .filter(|((_, m), _)| *m == mode)
            .fold((0, 0), |(p, t), (_, (a, b))| (p + a, t + b));
        summary.push(format!(
            "{mode} lexicons: {pass} of {total} checks pass ({})",
            percent(pass, total)
        ));
    }
    for ((f, mode), (pass, total)) in &tallies {
        summary.push(format!("  {f} {mode} {pass}/{total}"));
    }
    ratchet(&review("zemberek_failures.txt"), &summary, &failures);
}

/// Review 2's paradigm lists on the built-in lexicon. A lemma matches `lemma` or, as Zemberek
/// spells it, `display_lemma` (kâğıt). Failures are `section<TAB>line`.
#[test]
fn paradigm_lists_keep_their_known_failures() {
    let text =
        std::fs::read_to_string(review("paradigm_lists.tsv")).expect("read paradigm_lists.tsv");
    let morphology = TurkishMorphology::new();
    let mut section = String::from("-");
    let mut tallies: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let mut failures = BTreeSet::new();
    for line in text.lines() {
        if let Some(s) = line.strip_prefix("##") {
            section = s.trim().to_string();
            continue;
        }
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        let (kind, word) = (cols[0], cols[1]);
        let parses = morphology.analyze(word);
        let has = |lemma: &str, tail: &str| {
            parses.iter().any(|p| {
                (lemma == "*" || p.lemma == lemma || p.display_lemma() == lemma)
                    && (tail == "*" || lexical(&p.analysis).ends_with(tail))
            })
        };
        let ok = match kind {
            "ACC" => has(cols[2], cols.get(3).copied().unwrap_or("*")),
            "REJ" => parses.is_empty(),
            "REJL" => !has(cols[2], cols.get(3).copied().unwrap_or("*")),
            _ => panic!("unknown check {kind:?} in paradigm_lists.tsv"),
        };
        let tally = tallies.entry(kind.to_string()).or_default();
        tally.1 += 1;
        if ok {
            tally.0 += 1;
        } else {
            failures.insert(format!("{section}\t{line}"));
        }
    }
    let (pass, total) = tallies
        .values()
        .fold((0, 0), |(p, t), (a, b)| (p + a, t + b));
    let mut summary = vec![format!(
        "{pass} of {total} checks pass ({})",
        percent(pass, total)
    )];
    for (kind, (pass, total)) in &tallies {
        summary.push(format!("  {kind} {pass}/{total}"));
    }
    ratchet(&review("paradigm_failures.txt"), &summary, &failures);
}
