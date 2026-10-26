//! Prints every syntactic parse (inflectional groups) of the words given on the command line,
//! with the morphological reading each comes from.
//!
//! ```text
//! cargo run --release -p akana-core --example syntactic_probe -- geldiğimizde evdeyim
//! ```

use akana_core::morphology::TurkishMorphology;
use akana_core::syntactic_morphology::TurkishSyntacticMorphology;

fn main() {
    let morphology = TurkishMorphology::new();
    for word in std::env::args().skip(1) {
        let parses = morphology.analyze(&word);
        if parses.is_empty() {
            println!("{word}: -");
        }
        for p in &parses {
            let s = TurkishSyntacticMorphology::from_parse(&word, p);
            println!("{word}: {}\n    {}", p.formatted, s.formatted);
        }
    }
}
