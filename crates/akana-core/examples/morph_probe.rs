//! Prints every analysis of the words given on the command line (or one per stdin line).
//!
//! ```text
//! cargo run --release -p akana-core --example morph_probe -- kitabı gelmem bana
//! ```

use akana_core::morphology::TurkishMorphology;
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
                "{word}: {} pos={} {}{:.3e}",
                p.formatted,
                p.pos.as_str(),
                if secondary.is_empty() {
                    String::new()
                } else {
                    format!("{secondary} ")
                },
                p.score
            );
        }
    }
}
