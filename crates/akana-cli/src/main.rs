use clap::{Parser, Subcommand};
use std::path::PathBuf;
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
        text: Option<String>,
        /// Read text from file
        #[arg(short, long)]
        file: Option<PathBuf>,
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
        text: Option<String>,
        /// Read text from file
        #[arg(short, long)]
        file: Option<PathBuf>,
    },
    /// Asciify Turkish text
    Asciify {
        /// Text to convert to ASCII
        text: Option<String>,
        /// Read text from file
        #[arg(short, long)]
        file: Option<PathBuf>,
    },
    /// Normalize informal text
    Normalize {
        /// Text to normalize
        text: Option<String>,
        /// Read text from file
        #[arg(short, long)]
        file: Option<PathBuf>,
    },
    /// Check spelling and suggest corrections
    Spellcheck {
        /// Word to check
        word: String,
    },
    /// Parse sentence dependency tree
    Parse {
        /// Sentence to parse
        sentence: Option<String>,
        /// Read text from file
        #[arg(short, long)]
        file: Option<PathBuf>,
    },
    /// Calculate modern and classical Turkish readability metrics
    Readability {
        /// Text to analyze
        text: Option<String>,
        /// Read text from file
        #[arg(short, long)]
        file: Option<PathBuf>,
    },
    /// Syllabify and hyphenate a Turkish word
    Syllabify {
        /// Word to syllabify
        word: String,
    },
    /// Convert number to Turkish written words
    Number {
        /// Number to convert
        num: i64,
    },
    /// Extract Named Entities (PER, LOC, ORG, DATE, MONEY, PERCENT)
    Ner {
        /// Text to analyze
        text: Option<String>,
        /// Read text from file
        #[arg(short, long)]
        file: Option<PathBuf>,
    },
    /// Extract top keywords from Turkish text
    Keywords {
        /// Text to extract keywords from
        text: Option<String>,
        /// Read text from file
        #[arg(short, long)]
        file: Option<PathBuf>,
        /// Top K keywords
        #[arg(short, long, default_value_t = 10)]
        top_k: usize,
    },
    /// Extractive text summarization
    Summarize {
        /// Text to summarize
        text: Option<String>,
        /// Read text from file
        #[arg(short, long)]
        file: Option<PathBuf>,
        /// Maximum number of sentences
        #[arg(short, long, default_value_t = 3)]
        sentences: usize,
    },
    /// Find morphological root / stem of a word
    Stem {
        /// Word to stem
        word: String,
    },
    /// Audit text for AI writing signatures, clichés, and punctuation anomalies
    AiAudit {
        /// Text to audit
        text: Option<String>,
        /// Read text from file
        #[arg(short, long)]
        file: Option<PathBuf>,
    },
    /// Generate an actionable LLM rewrite prompt to humanize Turkish text
    HumanizePrompt {
        /// Text to humanize
        text: Option<String>,
        /// Read text from file
        #[arg(short, long)]
        file: Option<PathBuf>,
        /// Target register (e.g. blog, akademik, haber, edebi, hukuki)
        #[arg(short, long, default_value = "blog")]
        register: String,
    },
}

fn resolve_input_text(text: Option<String>, file: Option<PathBuf>) -> Result<String, String> {
    if let Some(f) = file {
        std::fs::read_to_string(&f).map_err(|e| format!("Failed to read file '{}': {}", f.display(), e))
    } else if let Some(t) = text {
        Ok(t)
    } else {
        Err("Please provide input text as argument or use -f/--file <PATH>".to_string())
    }
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Tokenize { text, file } => {
            let input = match resolve_input_text(text, file) {
                Ok(t) => t,
                Err(e) => { eprintln!("Error: {}", e); std::process::exit(1); }
            };
            let tokens = tokenization::TurkishTokenizer::tokenize_words(&input);
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
        Commands::Deasciify { text, file } => {
            let input = match resolve_input_text(text, file) {
                Ok(t) => t,
                Err(e) => { eprintln!("Error: {}", e); std::process::exit(1); }
            };
            let res = normalization::TurkishDeasciifier::deasciify(&input);
            println!("{}", res);
        }
        Commands::Asciify { text, file } => {
            let input = match resolve_input_text(text, file) {
                Ok(t) => t,
                Err(e) => { eprintln!("Error: {}", e); std::process::exit(1); }
            };
            let res = normalization::TurkishAsciifier::asciify(&input);
            println!("{}", res);
        }
        Commands::Normalize { text, file } => {
            let input = match resolve_input_text(text, file) {
                Ok(t) => t,
                Err(e) => { eprintln!("Error: {}", e); std::process::exit(1); }
            };
            let res = normalization::TurkishInformalNormalizer::normalize_text(&input);
            println!("{}", res);
        }
        Commands::Spellcheck { word } => {
            let checker = normalization::TurkishSpellChecker::new();
            let suggestions = checker.suggest(&word, 2, 5);
            println!("{}", serde_json::to_string_pretty(&suggestions).unwrap());
        }
        Commands::Parse { sentence, file } => {
            let input = match resolve_input_text(sentence, file) {
                Ok(t) => t,
                Err(e) => { eprintln!("Error: {}", e); std::process::exit(1); }
            };
            let tokens = tokenization::TurkishTokenizer::tokenize_words(&input);
            let parser = parser::TurkishDependencyParser::new();
            let tree = parser.parse(&tokens);
            println!("{}", tree.to_conllu());
        }
        Commands::Readability { text, file } => {
            let input = match resolve_input_text(text, file) {
                Ok(t) => t,
                Err(e) => { eprintln!("Error: {}", e); std::process::exit(1); }
            };
            let report = readability::analyze_readability(&input);
            println!("{}", serde_json::to_string_pretty(&report).unwrap());
        }
        Commands::Syllabify { word } => {
            let syllables = phonology::TurkishSyllabifier::syllabify(&word);
            let hyphenated = phonology::TurkishSyllabifier::hyphenate(&word, "-");
            println!("Syllables: {:?} (Count: {}, Hyphenated: {})", syllables, syllables.len(), hyphenated);
        }
        Commands::Number { num } => {
            let words = normalization::TurkishNumberConverter::number_to_words(num);
            let ordinal = normalization::TurkishNumberConverter::ordinal_to_words(num);
            println!("Cardinal: {}\nOrdinal:  {}", words, ordinal);
        }
        Commands::Ner { text, file } => {
            let input = match resolve_input_text(text, file) {
                Ok(t) => t,
                Err(e) => { eprintln!("Error: {}", e); std::process::exit(1); }
            };
            let entities = ner::TurkishNER::extract_entities(&input);
            println!("{}", serde_json::to_string_pretty(&entities).unwrap());
        }
        Commands::Keywords { text, file, top_k } => {
            let input = match resolve_input_text(text, file) {
                Ok(t) => t,
                Err(e) => { eprintln!("Error: {}", e); std::process::exit(1); }
            };
            let extractor = analysis::TurkishKeywordExtractor::new();
            let kw = extractor.extract_keywords(&input, top_k);
            println!("{}", serde_json::to_string_pretty(&kw).unwrap());
        }
        Commands::Summarize { text, file, sentences } => {
            let input = match resolve_input_text(text, file) {
                Ok(t) => t,
                Err(e) => { eprintln!("Error: {}", e); std::process::exit(1); }
            };
            let summarizer = analysis::TurkishSummarizer::new();
            let summary = summarizer.summarize(&input, sentences);
            for (idx, sent) in summary.iter().enumerate() {
                println!("{}. {}", idx + 1, sent);
            }
        }
        Commands::Stem { word } => {
            let stemmer = morphology::TurkishStemmer::new();
            println!("Stem: {}", stemmer.stem(&word));
        }
        Commands::AiAudit { text, file } => {
            let input = match resolve_input_text(text, file) {
                Ok(t) => t,
                Err(e) => { eprintln!("Error: {}", e); std::process::exit(1); }
            };
            let auditor = style::TurkishStyleAuditor::new();
            let report = auditor.audit(&input);
            println!("{}", serde_json::to_string_pretty(&report).unwrap());
        }
        Commands::HumanizePrompt { text, file, register } => {
            let input = match resolve_input_text(text, file) {
                Ok(t) => t,
                Err(e) => { eprintln!("Error: {}", e); std::process::exit(1); }
            };
            let prompt = style::TurkishHumanizer::generate_prompt(&input, &register);
            println!("{}", prompt);
        }
    }
}
