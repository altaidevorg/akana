//! The `espeak-compat` profile's data, derived from eSpeak NG's observed outputs only (the
//! reference fixtures and a CC0 word list run through the same eSpeak build; see
//! `espeak_compat.tsv.gz`'s header and `build_espeak_compat_data`).
//!
//! - **Lexicon:** the words of those outputs that the suffix model and the letter rules read
//!   otherwise than eSpeak (269 of 33,501), with their stress (per syllable: primary, unstressed
//!   or secondary) and, when the letter rules give other phonemes, eSpeak's phonemes. The builder
//!   drops every word the tables already read as eSpeak does, so the output is the same as with
//!   all of them.
//! - **Suffix model:** for a word outside the lexicon, the stress position (syllables from the
//!   end; 0 = unstressed) predicted from the longest ending it shares with lexicon words. The
//!   table stores, for each ending, the position a word with that ending and an unseen letter
//!   before it gets: the mean of the ending's continuations' position distributions,
//!   interpolated Witten-Bell style with the next shorter ending's (pruned where an ending
//!   predicts what its parent does).
//! - **Apostrophe endings:** after an apostrophe eSpeak stresses the last syllable except before
//!   a few endings (`'la`, `'le`, `'dir`, ...).

use super::letters::Seg;
use super::Phoneme;
use lazy_static::lazy_static;
use std::collections::HashMap;
use std::io::Read;

/// A syllable's lexical stress mark.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mark {
    /// The word's primary stress.
    Primary,
    /// Unstressed.
    Unstressed,
    /// eSpeak's secondary stress 3 (`ile`, `yoksa`, `kime`, spelled letters).
    Secondary,
}

impl Mark {
    pub fn from_char(c: char) -> Option<Mark> {
        match c {
            'P' => Some(Mark::Primary),
            'U' => Some(Mark::Unstressed),
            '3' => Some(Mark::Secondary),
            _ => None,
        }
    }
}

/// A lexicon entry.
pub(crate) struct Entry {
    pub marks: Vec<Mark>,
    /// eSpeak's phonemes when the letter rules differ; `None` when they agree.
    pub phonemes: Option<Vec<Seg>>,
}

pub(crate) struct Model {
    pub lexicon: HashMap<String, Entry>,
    /// Ending → predicted stress position, pruned.
    pub suffixes: HashMap<String, u8>,
    /// Ending after an apostrophe → stress position, for the endings that are not final.
    pub apostrophe: HashMap<String, u8>,
    /// The data file's header lines (its provenance).
    pub header: Vec<String>,
}

pub(crate) const DATA_GZ: &[u8] = include_bytes!("espeak_compat.tsv.gz");

lazy_static! {
    pub(crate) static ref MODEL: Model = Model::parse(&gunzip(DATA_GZ));
}

fn gunzip(bytes: &[u8]) -> String {
    let mut text = String::new();
    if !bytes.is_empty() {
        let _ = flate2::read::GzDecoder::new(bytes).read_to_string(&mut text);
    }
    text
}

/// Phonemes written `a b: dZ` (a colon marks length).
pub(crate) fn parse_phonemes(s: &str) -> Option<Vec<Seg>> {
    s.split(' ')
        .filter(|t| !t.is_empty())
        .map(|t| {
            let (sym, long) = match t.strip_suffix(':') {
                Some(sym) if !sym.is_empty() && sym != "_" => (sym, true),
                _ => (t, false),
            };
            Phoneme::from_symbol(sym).map(|ph| Seg::new(ph, long))
        })
        .collect()
}

#[cfg(test)]
pub(crate) fn format_phonemes(segs: &[Seg]) -> String {
    segs.iter()
        .map(|s| format!("{}{}", s.ph.symbol(), if s.long { ":" } else { "" }))
        .collect::<Vec<_>>()
        .join(" ")
}

impl Model {
    /// Lines: `L key marks phonemes`, `S ending position`, `A ending position`, tab-separated;
    /// `#` lines are the header. Malformed lines are skipped.
    pub fn parse(text: &str) -> Model {
        let mut lexicon = HashMap::new();
        let mut suffixes = HashMap::new();
        let mut apostrophe = HashMap::new();
        let mut header = Vec::new();
        for line in text.lines() {
            if let Some(h) = line.strip_prefix('#') {
                header.push(h.trim().to_string());
                continue;
            }
            let mut f = line.split('\t');
            match f.next() {
                Some("L") => {
                    let (Some(key), Some(marks)) = (f.next(), f.next()) else {
                        continue;
                    };
                    let Some(marks) = marks.chars().map(Mark::from_char).collect() else {
                        continue;
                    };
                    let phonemes = match f.next() {
                        Some(p) if !p.is_empty() => match parse_phonemes(p) {
                            Some(p) => Some(p),
                            None => continue,
                        },
                        _ => None,
                    };
                    lexicon.insert(key.to_string(), Entry { marks, phonemes });
                }
                Some(kind @ ("S" | "A")) => {
                    let (Some(end), Some(pos)) = (f.next(), f.next()) else {
                        continue;
                    };
                    let Ok(pos) = pos.parse::<u8>() else {
                        continue;
                    };
                    if kind == "S" {
                        suffixes.insert(end.to_string(), pos);
                    } else {
                        apostrophe.insert(end.to_string(), pos);
                    }
                }
                _ => {}
            }
        }
        Model {
            lexicon,
            suffixes,
            apostrophe,
            header,
        }
    }

    /// The stress position the suffix model gives a word outside the lexicon, and the ending
    /// it used (empty for the default, final stress).
    pub fn suffix_position<'a>(&self, key: &'a str) -> (u8, &'a str) {
        let mut best: Option<(u8, &str)> = None;
        // endings of the key, shortest first; the search stops at the first ending the table
        // lacks (an ending is kept only with its shorter endings)
        let mut starts: Vec<usize> = key.char_indices().map(|(i, _)| i).collect();
        starts.reverse();
        for start in starts {
            let end = &key[start..];
            match self.suffixes.get(end) {
                Some(&pos) => best = Some((pos, end)),
                None => break,
            }
        }
        best.unwrap_or((1, ""))
    }
}

#[cfg(test)]
pub(crate) mod build {
    //! The builder of `espeak_compat.tsv.gz` (run on demand, it needs the fixtures):
    //!
    //! `AKANA_ESPEAK_FIXTURES=DIR [AKANA_ESPEAK_EXTRA=FILE.jsonl] cargo test --release -p
    //! akana-core --lib -- --ignored build_espeak_compat_data`
    //!
    //! It reads the `train` rows of `DIR/espeak-plan.jsonl` and the extra rows (a word list run
    //! through the same eSpeak build), aligns each text's words with eSpeak's, and writes the
    //! lexicon, the suffix table and the apostrophe table.

    use super::super::letters::{has_vowel, key_to_phonemes, Seg};
    use super::super::text::{segment, TextWord};
    use super::super::Phoneme;
    use super::format_phonemes;
    use std::collections::{BTreeMap, HashMap, HashSet};
    use std::io::{BufRead, Write};

    /// A word's observation: eSpeak's phonemes and per-vowel stress, and whether it ended its
    /// clause.
    pub(crate) struct Observed {
        pub key: String,
        pub segs: Vec<Seg>,
        pub stress: Vec<u8>,
        pub at_end: bool,
    }

    /// The words of a fixture row aligned with eSpeak's, or `None` when the counts differ.
    pub(crate) fn align_row(row: &serde_json::Value) -> Option<Vec<Observed>> {
        let text = row.get("text")?.as_str()?;
        let tokens = row.get("tokens")?.as_array()?;
        let mut words: Vec<TextWord> = Vec::new();
        segment(text, &mut words, false);
        let mut groups: Vec<(Vec<Seg>, Vec<u8>)> = Vec::new();
        for t in tokens {
            if t.get("type")?.as_str()? == "pause" {
                continue;
            }
            let ph = Phoneme::from_symbol(t.get("ph")?.as_str()?)?;
            let long = t.get("lengthen")?.as_u64()? == 1;
            if t.get("word")?.as_u64()? == 1 || groups.is_empty() {
                groups.push((Vec::new(), Vec::new()));
            }
            let g = groups.last_mut()?;
            g.0.push(Seg::new(ph, long));
            if ph.is_vowel() {
                g.1.push(t.get("stress")?.as_u64()? as u8);
            }
        }
        if groups.len() != words.len() {
            return None;
        }
        let n = words.len();
        Some(
            groups
                .into_iter()
                .enumerate()
                .map(|(i, (segs, stress))| Observed {
                    key: words[i].key.clone(),
                    segs,
                    stress,
                    at_end: i + 1 == n || words[i + 1].boundary,
                })
                .collect(),
        )
    }

    fn marks_of(stress: &[u8]) -> String {
        stress
            .iter()
            .map(|&s| {
                if s >= 4 {
                    'P'
                } else if s == 3 {
                    '3'
                } else {
                    'U'
                }
            })
            .collect()
    }

    fn most_common<T: Ord + Clone>(c: &BTreeMap<T, usize>) -> Option<T> {
        // the largest count; ties to the smallest key
        let mut best: Option<(&T, usize)> = None;
        for (k, &n) in c {
            if best.is_none_or(|(_, b)| n > b) {
                best = Some((k, n));
            }
        }
        best.map(|(k, _)| k.clone())
    }

    fn position(marks: &str) -> u8 {
        let chars: Vec<char> = marks.chars().collect();
        match chars.iter().rposition(|&c| c == 'P') {
            Some(i) => (chars.len() - i) as u8,
            None => 0,
        }
    }

    pub(crate) struct Built {
        pub lexicon: BTreeMap<String, (String, Option<String>)>,
        pub suffixes: BTreeMap<String, u8>,
        pub apostrophe: BTreeMap<String, u8>,
        pub main_rows: usize,
        pub extra_rows: usize,
    }

    /// Builds the tables from aligned rows: `main` first; `extra` only adds words `main` lacks.
    pub(crate) fn build(main: &[Vec<Observed>], extra: &[Vec<Observed>]) -> Built {
        type Counts = HashMap<String, BTreeMap<String, usize>>;
        let mut seg: Counts = HashMap::new();
        let mut st_mid: Counts = HashMap::new();
        let mut st_end: Counts = HashMap::new();
        let mut main_keys: HashSet<String> = HashSet::new();
        for (rows, is_extra) in [(main, false), (extra, true)] {
            for row in rows {
                for o in row {
                    if is_extra && main_keys.contains(&o.key) {
                        continue;
                    }
                    if !is_extra {
                        main_keys.insert(o.key.clone());
                    }
                    *seg.entry(o.key.clone())
                        .or_default()
                        .entry(format_phonemes(&o.segs))
                        .or_default() += 1;
                    let target = if o.at_end { &mut st_end } else { &mut st_mid };
                    *target
                        .entry(o.key.clone())
                        .or_default()
                        .entry(marks_of(&o.stress))
                        .or_default() += 1;
                }
            }
        }
        let mut lexicon = BTreeMap::new();
        let mut rule = Vec::new();
        for (key, c) in &seg {
            let letters: String = key.chars().filter(|&c| c != '\'').collect();
            if !has_vowel(&letters) {
                continue;
            }
            let Some(best) = most_common(c) else { continue };
            // canonical form: a word-final l closing its syllable (eSpeak links it to a
            // following vowel, which the phonemiser does again)
            let mut segs = super::parse_phonemes(&best).unwrap_or_default();
            if let Some(last) = segs.last_mut() {
                last.ph = match last.ph {
                    Phoneme::L => Phoneme::LCoda,
                    Phoneme::LDark => Phoneme::LDarkCoda,
                    p => p,
                };
            }
            let marks = match st_mid.get(key).and_then(most_common) {
                Some(m) => m,
                None => match st_end.get(key).and_then(most_common) {
                    Some(m) => m,
                    None => continue,
                },
            };
            key_to_phonemes(key, &mut rule);
            let same = rule.len() == segs.len()
                && rule
                    .iter()
                    .zip(&segs)
                    .all(|(a, b)| a.ph == b.ph && a.long == b.long);
            let phonemes = if same {
                None
            } else {
                Some(format_phonemes(&segs))
            };
            lexicon.insert(key.clone(), (marks, phonemes));
        }
        // the suffix trie over '^' + key
        let mut counts: BTreeMap<String, BTreeMap<u8, usize>> = BTreeMap::new();
        for (key, (marks, _)) in &lexicon {
            let pos = position(marks);
            let w: Vec<char> = std::iter::once('^').chain(key.chars()).collect();
            for j in 1..=w.len() {
                let s: String = w[w.len() - j..].iter().collect();
                *counts.entry(s).or_default().entry(pos).or_default() += 1;
            }
        }
        let mut kids: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for s in counts.keys() {
            let mut cs = s.chars();
            if s.chars().count() > 1 {
                cs.next();
                kids.entry(cs.as_str().to_string())
                    .or_default()
                    .push(s.clone());
            }
        }
        let beta = 0.5_f64;
        let mut memo: HashMap<String, BTreeMap<u8, f64>> = HashMap::new();
        // the root's distribution: all length-1 endings' counts
        let mut root: BTreeMap<u8, f64> = BTreeMap::new();
        for (s, c) in &counts {
            if s.chars().count() == 1 {
                for (&p, &n) in c {
                    *root.entry(p).or_default() += n as f64;
                }
            }
        }
        let z: f64 = root.values().sum();
        for v in root.values_mut() {
            *v /= z;
        }
        fn novel(
            x: &str,
            counts: &BTreeMap<String, BTreeMap<u8, usize>>,
            kids: &BTreeMap<String, Vec<String>>,
            root: &BTreeMap<u8, f64>,
            beta: f64,
            memo: &mut HashMap<String, BTreeMap<u8, f64>>,
        ) -> BTreeMap<u8, f64> {
            if let Some(d) = memo.get(x) {
                return d.clone();
            }
            if x.is_empty() {
                return root.clone();
            }
            let mut cs = x.chars();
            cs.next();
            let par = novel(cs.as_str(), counts, kids, root, beta, memo);
            let mut d: BTreeMap<u8, f64> = BTreeMap::new();
            let ks = kids.get(x).map(Vec::as_slice).unwrap_or(&[]);
            for c in ks {
                let cc = &counts[c];
                let t: usize = cc.values().sum();
                for (&p, &n) in cc {
                    *d.entry(p).or_default() += n as f64 / t as f64;
                }
            }
            let k = ks.len() as f64;
            let lam = if ks.is_empty() { 0.0 } else { k / (k + beta) };
            let z: f64 = d.values().sum();
            let mut out: BTreeMap<u8, f64> = BTreeMap::new();
            for p in d.keys().chain(par.keys()) {
                let a = if z > 0.0 {
                    d.get(p).copied().unwrap_or(0.0) / z
                } else {
                    0.0
                };
                out.insert(
                    *p,
                    lam * a + (1.0 - lam) * par.get(p).copied().unwrap_or(0.0),
                );
            }
            memo.insert(x.to_string(), out.clone());
            out
        }
        fn argmax(d: &BTreeMap<u8, f64>) -> u8 {
            let mut best: Option<(u8, f64)> = None;
            for (&p, &v) in d {
                if best.is_none_or(|(_, b)| v > b) {
                    best = Some((p, v));
                }
            }
            best.map(|(p, _)| p).unwrap_or(1)
        }
        // predictions for endings (not whole words, which the lexicon answers)
        let mut pred: BTreeMap<String, u8> = BTreeMap::new();
        for s in counts.keys() {
            if s.starts_with('^') {
                continue;
            }
            let d = novel(s, &counts, &kids, &root, beta, &mut memo);
            pred.insert(s.clone(), argmax(&d));
        }
        // prune: keep an ending when it predicts other than its parent (the empty ending
        // predicts 1, final stress) or a longer ending below it is kept
        let mut keep: HashSet<String> = HashSet::new();
        let mut by_len: Vec<&String> = pred.keys().collect();
        by_len.sort_by_key(|s| std::cmp::Reverse(s.chars().count()));
        for s in by_len {
            let mut cs = s.chars();
            cs.next();
            let parent = cs.as_str();
            let parent_pred = if parent.is_empty() { 1 } else { pred[parent] };
            let child_kept = kids
                .get(s.as_str())
                .is_some_and(|ks| ks.iter().any(|k| keep.contains(k)));
            if pred[s] != parent_pred || child_kept {
                keep.insert(s.clone());
            }
        }
        let suffixes: BTreeMap<String, u8> =
            pred.into_iter().filter(|(s, _)| keep.contains(s)).collect();
        // apostrophe endings
        let mut apo: BTreeMap<String, BTreeMap<u8, usize>> = BTreeMap::new();
        for (key, (marks, _)) in &lexicon {
            if let Some((_, tail)) = key.rsplit_once('\'') {
                let pos = position(marks);
                if pos > 0 {
                    *apo.entry(tail.to_string())
                        .or_default()
                        .entry(pos)
                        .or_default() += 1;
                }
            }
        }
        let apostrophe: BTreeMap<String, u8> = apo
            .into_iter()
            .filter_map(|(t, c)| most_common(&c).filter(|&p| p != 1).map(|p| (t, p)))
            .collect();
        // Keep only the words the tables and the letter rules would get wrong: the rules' phonemes
        // (with the stress-dependent choices made under the word's own stress) must equal
        // eSpeak's, and the suffix model's stress the word's. The others answer the same without
        // an entry, so the output does not change.
        let rules_only = super::Model {
            lexicon: HashMap::new(),
            suffixes: suffixes.iter().map(|(s, p)| (s.clone(), *p)).collect(),
            apostrophe: apostrophe.iter().map(|(s, p)| (s.clone(), *p)).collect(),
            header: Vec::new(),
        };
        let lexicon = lexicon
            .into_iter()
            .filter_map(|(key, (marks, phonemes))| {
                let observed = match &phonemes {
                    Some(p) => super::parse_phonemes(p).unwrap_or_default(),
                    None => {
                        let mut s = Vec::new();
                        key_to_phonemes(&key, &mut s);
                        s
                    }
                };
                let mut rule = Vec::new();
                key_to_phonemes(&key, &mut rule);
                let levels: Vec<u8> = marks
                    .chars()
                    .map(|c| match c {
                        'P' => 6,
                        '3' => 3,
                        _ => 1,
                    })
                    .collect();
                super::super::letters::stress_dependent_segments(&mut rule, &levels);
                let same_segments = rule.len() == observed.len()
                    && rule
                        .iter()
                        .zip(&observed)
                        .all(|(a, b)| a.ph == b.ph && a.long == b.long);
                let nsyl = observed.iter().filter(|s| s.ph.is_vowel()).count();
                let (predicted, _) = super::super::predict_marks(&rules_only, &key, nsyl);
                let predicted: String = predicted
                    .iter()
                    .map(|m| match m {
                        super::Mark::Primary => 'P',
                        super::Mark::Secondary => '3',
                        super::Mark::Unstressed => 'U',
                    })
                    .collect();
                if same_segments && predicted == marks {
                    return None;
                }
                let phonemes = if same_segments {
                    phonemes
                } else {
                    Some(format_phonemes(&observed))
                };
                Some((key, (marks, phonemes)))
            })
            .collect();
        Built {
            lexicon,
            suffixes,
            apostrophe,
            main_rows: main.len(),
            extra_rows: extra.len(),
        }
    }

    pub(crate) fn read_rows(
        path: &std::path::Path,
        split: Option<&str>,
    ) -> (Vec<Vec<Observed>>, usize) {
        let file = std::fs::File::open(path).expect("rows file");
        let mut rows = Vec::new();
        let mut skipped = 0;
        for line in std::io::BufReader::new(file).lines() {
            let line = line.expect("line");
            let v: serde_json::Value = serde_json::from_str(&line).expect("json row");
            if let Some(s) = split {
                if v.get("split").and_then(|s| s.as_str()) != Some(s) {
                    continue;
                }
            }
            match align_row(&v) {
                Some(r) => rows.push(r),
                None => skipped += 1,
            }
        }
        (rows, skipped)
    }

    #[test]
    #[ignore = "needs AKANA_ESPEAK_FIXTURES; rewrites src/speech/espeak_compat.tsv.gz"]
    fn build_espeak_compat_data() {
        let dir = std::env::var("AKANA_ESPEAK_FIXTURES").expect("AKANA_ESPEAK_FIXTURES");
        let plan = std::path::Path::new(&dir).join("espeak-plan.jsonl");
        let (main, skipped) = read_rows(&plan, Some("train"));
        let (extra, extra_skipped) = match std::env::var("AKANA_ESPEAK_EXTRA") {
            Ok(p) => read_rows(std::path::Path::new(&p), None),
            Err(_) => (Vec::new(), 0),
        };
        let b = build(&main, &extra);
        let sha = |p: &std::path::Path| -> String {
            use sha2_free::hex_digest;
            hex_digest(&std::fs::read(p).expect("read"))
        };
        let mut out = String::new();
        out.push_str(
            "# akana espeak-compat profile data v1: derived from eSpeak NG 1.52.0 outputs only\n",
        );
        // file names only (with their parent directory): no local paths in the shipped file
        let name = |p: &std::path::Path| -> String {
            let parent = p
                .parent()
                .and_then(|d| d.file_name())
                .map(|f| f.to_string_lossy().to_string())
                .unwrap_or_default();
            let file = p
                .file_name()
                .map(|f| f.to_string_lossy().to_string())
                .unwrap_or_default();
            format!("{parent}/{file}")
        };
        out.push_str(&format!(
            "# train rows of {} (sha256 {}): {} aligned, {} skipped\n",
            name(&plan),
            sha(&plan),
            b.main_rows,
            skipped
        ));
        if let Ok(p) = std::env::var("AKANA_ESPEAK_EXTRA") {
            let p = std::path::Path::new(&p);
            out.push_str(&format!(
                "# extra: eSpeak's plan for scripts/speech/espeak_compat_wordlist.json, {} (sha256 {}): {} rows aligned, {} skipped\n",
                name(p),
                sha(p),
                b.extra_rows,
                extra_skipped
            ));
        }
        out.push_str("# L key marks phonemes | S ending position | A apostrophe-ending position\n");
        out.push_str(&data_text(&b));
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/speech/espeak_compat.tsv.gz");
        let file = std::fs::File::create(&path).expect("create");
        let mut gz = flate2::write::GzEncoder::new(file, flate2::Compression::best());
        gz.write_all(out.as_bytes()).expect("write");
        gz.finish().expect("finish");
        eprintln!(
            "lexicon {} (phonemes {}), suffixes {}, apostrophe {}",
            b.lexicon.len(),
            b.lexicon.values().filter(|(_, p)| p.is_some()).count(),
            b.suffixes.len(),
            b.apostrophe.len()
        );
    }

    mod sha2_free {
        //! A small SHA-256 for the data file's provenance header (the `sha2` crate is an
        //! optional dependency of another feature).
        pub fn hex_digest(data: &[u8]) -> String {
            const K: [u32; 64] = [
                0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
                0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
                0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
                0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
                0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
                0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
                0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
                0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
                0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
                0xc67178f2,
            ];
            let mut h: [u32; 8] = [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ];
            let mut msg = data.to_vec();
            let bits = (data.len() as u64).wrapping_mul(8);
            msg.push(0x80);
            while msg.len() % 64 != 56 {
                msg.push(0);
            }
            msg.extend_from_slice(&bits.to_be_bytes());
            for chunk in msg.chunks(64) {
                let mut w = [0u32; 64];
                for i in 0..16 {
                    w[i] = u32::from_be_bytes([
                        chunk[4 * i],
                        chunk[4 * i + 1],
                        chunk[4 * i + 2],
                        chunk[4 * i + 3],
                    ]);
                }
                for i in 16..64 {
                    let s0 =
                        w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
                    let s1 =
                        w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
                    w[i] = w[i - 16]
                        .wrapping_add(s0)
                        .wrapping_add(w[i - 7])
                        .wrapping_add(s1);
                }
                let mut a = h;
                for i in 0..64 {
                    let s1 = a[4].rotate_right(6) ^ a[4].rotate_right(11) ^ a[4].rotate_right(25);
                    let ch = (a[4] & a[5]) ^ (!a[4] & a[6]);
                    let t1 = a[7]
                        .wrapping_add(s1)
                        .wrapping_add(ch)
                        .wrapping_add(K[i])
                        .wrapping_add(w[i]);
                    let s0 = a[0].rotate_right(2) ^ a[0].rotate_right(13) ^ a[0].rotate_right(22);
                    let maj = (a[0] & a[1]) ^ (a[0] & a[2]) ^ (a[1] & a[2]);
                    let t2 = s0.wrapping_add(maj);
                    a = [
                        t1.wrapping_add(t2),
                        a[0],
                        a[1],
                        a[2],
                        a[3].wrapping_add(t1),
                        a[4],
                        a[5],
                        a[6],
                    ];
                }
                for i in 0..8 {
                    h[i] = h[i].wrapping_add(a[i]);
                }
            }
            h.iter().map(|x| format!("{x:08x}")).collect()
        }
    }

    /// The data file's text for built tables.
    pub(crate) fn data_text(b: &Built) -> String {
        let mut out = String::new();
        for (k, (m, p)) in &b.lexicon {
            out.push_str(&format!("L\t{k}\t{m}\t{}\n", p.as_deref().unwrap_or("")));
        }
        for (s, p) in &b.suffixes {
            out.push_str(&format!("S\t{s}\t{p}\n"));
        }
        for (s, p) in &b.apostrophe {
            out.push_str(&format!("A\t{s}\t{p}\n"));
        }
        out
    }

    /// Five-fold cross-validation over the train split's sentences: the tables are built from
    /// four folds (and the extra word list, which holds no train word) and the fifth is read.
    /// The held-out fold's words outside the four folds and the word list are read by the
    /// letter rules and the suffix model alone, so this measures them on unseen words. Per-token
    /// agreement pairs tokens along a longest common subsequence of the phoneme symbols.
    #[test]
    #[ignore = "needs AKANA_ESPEAK_FIXTURES"]
    fn cross_validate_espeak_compat() {
        use super::super::{phonemize_with, Profile, Utterance};
        let dir = std::env::var("AKANA_ESPEAK_FIXTURES").expect("AKANA_ESPEAK_FIXTURES");
        let plan = std::path::Path::new(&dir).join("espeak-plan.jsonl");
        let file = std::fs::File::open(&plan).expect("plan");
        let rows: Vec<serde_json::Value> = std::io::BufReader::new(file)
            .lines()
            .map_while(Result::ok)
            .filter_map(|l| serde_json::from_str(&l).ok())
            .filter(|v: &serde_json::Value| v["split"] == "train")
            .collect();
        let (extra, _) = match std::env::var("AKANA_ESPEAK_EXTRA") {
            Ok(p) => read_rows(std::path::Path::new(&p), None),
            Err(_) => (Vec::new(), 0),
        };
        let fold = |id: &str| -> usize {
            // FNV-1a of the id: a fixed, documented fold assignment
            let mut h: u32 = 0x811c9dc5;
            for b in id.bytes() {
                h ^= u32::from(b);
                h = h.wrapping_mul(0x01000193);
            }
            (h % 5) as usize
        };
        let fields = ["ph", "stress", "word", "lengthen", "mark"];
        let (mut n, mut exact, mut ph_exact, mut tokens) = (0usize, 0usize, 0usize, 0usize);
        let mut agree = [0usize; 5];
        let mut u = Utterance::default();
        for f in 0..5 {
            let main: Vec<Vec<Observed>> = rows
                .iter()
                .filter(|r| fold(r["id"].as_str().unwrap_or("")) != f)
                .filter_map(align_row)
                .collect();
            let model = super::Model::parse(&data_text(&build(&main, &extra)));
            for r in rows
                .iter()
                .filter(|r| fold(r["id"].as_str().unwrap_or("")) == f)
            {
                phonemize_with(
                    &model,
                    Profile::EspeakCompat,
                    r["text"].as_str().unwrap_or(""),
                    &mut u,
                )
                .expect("phonemize");
                let obs = r["tokens"].as_array().cloned().unwrap_or_default();
                n += 1;
                tokens += obs.len();
                let same_ph = obs.len() == u.tokens.len()
                    && obs
                        .iter()
                        .zip(&u.tokens)
                        .all(|(o, t)| o["ph"].as_str() == Some(t.ph.symbol()));
                ph_exact += usize::from(same_ph);
                // tokens paired along a longest common subsequence of the phoneme symbols
                let a: Vec<&str> = obs.iter().map(|o| o["ph"].as_str().unwrap_or("")).collect();
                let b: Vec<&str> = u.tokens.iter().map(|t| t.ph.symbol()).collect();
                let mut dp = vec![vec![0u32; b.len() + 1]; a.len() + 1];
                for i in (0..a.len()).rev() {
                    for j in (0..b.len()).rev() {
                        dp[i][j] = if a[i] == b[j] {
                            dp[i + 1][j + 1] + 1
                        } else {
                            dp[i + 1][j].max(dp[i][j + 1])
                        };
                    }
                }
                let (mut i, mut j) = (0, 0);
                let mut pairs = Vec::new();
                while i < a.len() && j < b.len() {
                    if a[i] == b[j] {
                        pairs.push((i, j));
                        i += 1;
                        j += 1;
                    } else if dp[i + 1][j] >= dp[i][j + 1] {
                        i += 1;
                    } else {
                        j += 1;
                    }
                }
                let mut all = same_ph;
                for (o, t) in pairs.iter().map(|&(i, j)| (&obs[i], &u.tokens[j])) {
                    let ok = [
                        true,
                        o["stress"].as_u64() == Some(u64::from(t.stress)),
                        o["word"].as_u64() == Some(u64::from(t.word)),
                        o["lengthen"].as_u64() == Some(u64::from(t.lengthen)),
                        o["mark"].as_str() == Some(t.mark.to_string().as_str()),
                    ];
                    for (k, x) in ok.iter().enumerate() {
                        agree[k] += usize::from(*x);
                    }
                    all &= ok.iter().all(|x| *x);
                }
                exact += usize::from(all);
            }
        }
        eprintln!(
            "5-fold CV on train: {n} sentences, exact {:.4}, ph sequence exact {:.4}",
            exact as f64 / n as f64,
            ph_exact as f64 / n as f64
        );
        for (k, f) in fields.iter().enumerate() {
            eprintln!("  {f} {:.5}", agree[k] as f64 / tokens as f64);
        }
    }

    #[test]
    fn sha256_of_abc() {
        assert_eq!(
            sha2_free::hex_digest(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
