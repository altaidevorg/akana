use clap::{Parser, Subcommand};
use akana_core::*;

#[derive(Parser)]
#[command(name = "akana")]
#[command(about = "Akana: Modern and blazingly fast Turkish NLP CLI", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Tokenize Turkish text into words
    Tokenize {
        /// Text to tokenize
        text: String,
    },
    /// Perform morphological analysis on a word
    Analyze {
        /// Word to analyze
        word: String,
    },
    /// Generate a surface word from lemma and tags
    Generate {
        /// Base lemma
        lemma: String,
        /// Morpheme tags (e.g. A3pl, P1sg, Dat)
        #[arg(short, long, value_delimiter = ',')]
        tags: Vec<String>,
    },
    /// De-asciify Turkish text
    Deasciify {
        /// Text to restore diacritics
        text: String,
    },
    /// Asciify Turkish text
    Asciify {
        /// Text to convert to ASCII
        text: String,
    },
    /// Normalize informal text
    Normalize {
        /// Text to normalize
        text: String,
    },
    /// Check spelling and suggest corrections
    Spellcheck {
        /// Word to check
        word: String,
    },
    /// Parse sentence dependency tree
    Parse {
        /// Sentence to parse
        sentence: String,
    },
    /// Calculate modern and classical Turkish readability metrics
    Readability {
        /// Text to analyze
        text: String,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Tokenize { text } => {
            let tokens = tokenization::TurkishTokenizer::tokenize_words(&text);
            println!("{}", serde_json::to_string_pretty(&tokens).unwrap());
        }
        Commands::Analyze { word } => {
            let morph = morphology::TurkishMorphology::new();
            let parses = morph.analyze(&word);
            println!("{}", serde_json::to_string_pretty(&parses).unwrap());
        }
        Commands::Generate { lemma, tags } => {
            let gen = morphology::TurkishGenerator::new();
            let tag_refs: Vec<&str> = tags.iter().map(|s| s.as_str()).collect();
            if let Some(surface) = gen.generate(&lemma, &tag_refs) {
                println!("Generated surface: {}", surface);
            } else {
                println!("Failed to generate surface form.");
            }
        }
        Commands::Deasciify { text } => {
            let res = normalization::TurkishDeasciifier::deasciify(&text);
            println!("{}", res);
        }
        Commands::Asciify { text } => {
            let res = normalization::TurkishAsciifier::asciify(&text);
            println!("{}", res);
        }
        Commands::Normalize { text } => {
            let res = normalization::TurkishInformalNormalizer::normalize_text(&text);
            println!("{}", res);
        }
        Commands::Spellcheck { word } => {
            let checker = normalization::TurkishSpellChecker::new();
            let suggestions = checker.suggest(&word, 2, 5);
            println!("{}", serde_json::to_string_pretty(&suggestions).unwrap());
        }
        Commands::Parse { sentence } => {
            let tokens = tokenization::TurkishTokenizer::tokenize_words(&sentence);
            let parser = parser::TurkishDependencyParser::new();
            let tree = parser.parse(&tokens);
            println!("{}", tree.to_conllu());
        }
        Commands::Readability { text } => {
            let report = readability::analyze_readability(&text);
            println!("{}", serde_json::to_string_pretty(&report).unwrap());
        }
    }
}
