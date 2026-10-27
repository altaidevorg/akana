//! Prints every analysis of the words given on the command line (or one per stdin line): akana's
//! format, then Zemberek's on the line below.
//!
//! ```text
//! cargo run --release -p akana-core --example morph_probe -- kitabı gelmem bana
//! ```
//!
//! With `--backoff`, the stem search indexes an unknown word by (`backoff_stem`) instead. With
//! `--time FILE`, the words analysed per second over FILE's words (one per line), best of three.
//! With `--generate WORD TAGS ...`, the word each root or lemma and `+`-joined tags generate.
//! With `--split`, the compound split of the root of each word's best analysis (one per stdin
//! line, `word<TAB>count` allowed), for the words that split.

use akana_core::morphology::{CompoundDecomposer, TurkishMorphology};
use std::io::BufRead;

fn main() {
    let morphology = TurkishMorphology::new();
    let mut words: Vec<String> = std::env::args().skip(1).collect();
    if words.first().is_some_and(|w| w == "--stats") {
        let graph = akana_core::morphology::morphotactics_graph();
        println!(
            "graph: {} states, {} transitions; lexicon: {} items",
            graph.states.len(),
            graph.transitions.len(),
            morphology.lexicon().count()
        );
        for word in &words[1..] {
            let start = std::time::Instant::now();
            let n = (0..20)
                .map(|_| morphology.analyze(word).len())
                .sum::<usize>()
                / 20;
            println!(
                "{word}: {n} analyses, {:.3} ms",
                start.elapsed().as_secs_f64() * 1000.0 / 20.0
            );
        }
        return;
    }
    if words.first().is_some_and(|w| w == "--generate") {
        for pair in words[1..].chunks(2) {
            let [word, tags] = pair else {
                panic!("--generate takes word and Tag+Tag pairs");
            };
            let tags: Vec<&str> = tags.split('+').collect();
            println!(
                "{word} {}: {:?}",
                tags.join("+"),
                morphology.generate(word, &tags)
            );
        }
        return;
    }
    if words.first().is_some_and(|w| w == "--split") {
        for line in std::io::stdin().lock().lines().map_while(Result::ok) {
            let word = line.split('\t').next().unwrap_or_default().trim();
            let Some(top) = morphology.analyze(word).into_iter().next() else {
                continue;
            };
            if let Some((first, head)) =
                CompoundDecomposer::split_lexical_compound(&morphology, &top.root)
            {
                println!("{line}\t{}\t{first}+{head}", top.root);
            }
        }
        return;
    }
    if words.first().is_some_and(|w| w == "--time") {
        let text = std::fs::read_to_string(&words[1]).expect("a word list");
        let list: Vec<&str> = text.lines().filter(|l| !l.is_empty()).collect();
        let best = (0..3)
            .map(|_| {
                let start = std::time::Instant::now();
                let n: usize = list.iter().map(|w| morphology.analyze(w).len()).sum();
                std::hint::black_box(n);
                start.elapsed().as_secs_f64()
            })
            .fold(f64::INFINITY, f64::min);
        println!("{:.0} words/s", list.len() as f64 / best);
        return;
    }
    if words.first().is_some_and(|w| w == "--backoff-candidates") {
        for word in &words[1..] {
            let candidates: Vec<String> = morphology
                .backoff_candidates(word)
                .into_iter()
                .map(|(stem, n)| format!("{stem}/{n}"))
                .collect();
            println!("{word}	{}", candidates.join(" "));
        }
        return;
    }
    if words.first().is_some_and(|w| w == "--backoff") {
        for word in &words[1..] {
            let stem = morphology.backoff_stem(word);
            println!("{word}: {}", stem.as_deref().unwrap_or("-"));
        }
        return;
    }
    if words.is_empty() {
        words = std::io::stdin()
            .lock()
            .lines()
            .map_while(Result::ok)
            .collect();
    }
    for word in words {
        let parses = morphology.analyze(&word);
        if parses.is_empty() {
            println!("{word}: -");
        }
        for p in parses {
            let secondary = p.secondary_pos.as_str();
            println!(
                "{word}: {} pos={} {}{:.3e}\n    {}",
                p.formatted,
                p.pos.as_str(),
                if secondary.is_empty() {
                    String::new()
                } else {
                    format!("{secondary} ")
                },
                p.score,
                p.analysis
            );
        }
    }
}
