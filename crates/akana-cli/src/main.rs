use akana_core::*;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

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
    /// Perform morphological analysis on a word (Zemberek-compatible format)
    Analyze {
        /// Word to analyze
        word: String,
    },
    /// Perform Google-style syntactic morphological analysis with Inflectional Groups (FSMNLP 2019)
    SyntacticAnalyze {
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
    /// Check and correct grammar and spelling rules (the default rules unless told otherwise)
    Grammar {
        /// Text to check
        text: Option<String>,
        /// Read text from file
        #[arg(short, long)]
        file: Option<PathBuf>,
        /// Run only these rules (comma-separated ids; see --list-rules)
        #[arg(long, value_delimiter = ',')]
        rules: Option<Vec<String>>,
        /// Also run these rules
        #[arg(long, value_delimiter = ',')]
        enable: Vec<String>,
        /// Do not run these rules
        #[arg(long, value_delimiter = ',')]
        disable: Vec<String>,
        /// Also run every experimental rule (off by default: not precise enough on correct text)
        #[arg(long)]
        experimental: bool,
        /// Print only the corrected text
        #[arg(short, long)]
        correct: bool,
        /// Output the full result as JSON
        #[arg(short, long)]
        json: bool,
        /// List the rules, whether each is on by default, and its confidence (measured precision)
        #[arg(long)]
        list_rules: bool,
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
    /// Generate 256-dimensional Turkish sentence embedding vector
    Embed {
        /// Text to embed
        text: Option<String>,
        /// Read text from file
        #[arg(short, long)]
        file: Option<PathBuf>,
    },
    /// Calculate semantic cosine similarity between two Turkish texts
    Similarity {
        /// First text
        text_a: String,
        /// Second text
        text_b: String,
    },
    /// Split Turkish text into chunks (semantic, sentence, or sdpm)
    Chunk {
        /// Text to chunk
        text: Option<String>,
        /// Read text from file
        #[arg(short, long)]
        file: Option<PathBuf>,
        /// Chunking strategy: semantic, sentence, sdpm
        #[arg(short, long, default_value = "semantic")]
        strategy: String,
        /// Maximum tokens per chunk
        #[arg(short = 'c', long, default_value_t = 512)]
        chunk_size: usize,
        /// Overlap tokens between chunks (for sentence strategy)
        #[arg(short, long, default_value_t = 0)]
        overlap: usize,
        /// Minimum tokens before allowing a semantic split
        #[arg(short = 'k', long, default_value_t = 20)]
        min_chunk_size: usize,
        /// Similarity threshold (0.0 to 1.0) or cutoff value
        #[arg(short, long)]
        threshold: Option<f32>,
        /// Threshold calculation mode: similarity, percentile, stdev, iqr, auto
        #[arg(short = 'm', long, default_value = "percentile")]
        mode: String,
        /// Output formatted JSON
        #[arg(short, long)]
        json: bool,
    },
    /// Mask Turkish PII entities in text (KVKK aligned)
    PiiMask {
        /// Text to mask
        text: Option<String>,
        /// Read text from file
        #[arg(short, long)]
        file: Option<PathBuf>,
        /// Masking mode: placeholder, surrogate, tag, anonymize
        #[arg(short, long, default_value = "placeholder")]
        mode: String,
        /// Output formatted JSON including entities and mapping
        #[arg(short, long)]
        json: bool,
    },
    /// Restore masked Turkish PII text using a JSON mapping dictionary
    PiiRestore {
        /// Masked text
        text: Option<String>,
        /// Read text from file
        #[arg(short, long)]
        file: Option<PathBuf>,
        /// Path to JSON mapping file
        #[arg(short, long)]
        mapping: PathBuf,
    },
    /// Find personal data with the learned PII tagger (spans as byte offsets). Its models are
    /// downloaded from the Hugging Face Hub on first use and cached (AKANA_CACHE_DIR; AKANA_OFFLINE=1
    /// turns downloads off), unless --model names exported ones (scripts/pii_train/export_onnx.py)
    #[cfg(feature = "pii-tagger")]
    PiiTag {
        /// Text to tag
        text: Option<String>,
        /// Read text from file
        #[arg(short, long)]
        file: Option<PathBuf>,
        /// Tag every line of a JSON-lines file ({"text": ...}), writing one {"spans": ...} line each
        #[arg(long)]
        jsonl: Option<PathBuf>,
        /// The accurate mode: BERTurk averaged with the small model (about five times slower)
        #[arg(long)]
        accurate: bool,
        /// The strict operating point of the mode: more person, username and date tags, so that
        /// fewer personal values are left in clear
        #[arg(long, conflicts_with = "bias")]
        strict: bool,
        /// A small model's directory (model.onnx and tokenizer.json) instead of the published one
        #[arg(long, conflicts_with = "accurate")]
        model: Option<PathBuf>,
        /// A bigger model to average with --model on every window; repeatable
        #[arg(long, requires = "model")]
        big: Vec<PathBuf>,
        /// Operating point (0 is the trained one); with --labels it raises only those labels
        #[arg(long)]
        bias: Option<f32>,
        /// Labels the bias raises (default: every label, by lowering O)
        #[arg(long, value_delimiter = ',')]
        labels: Vec<String>,
        /// Inference threads (default: the machine's physical cores)
        #[arg(long)]
        threads: Option<usize>,
        /// Also report call-centre and service numbers and role mailboxes
        #[arg(long)]
        no_policy: bool,
    },
    /// Download the PII tagger's models into the cache if they are not there yet, and print
    /// their directories
    #[cfg(feature = "pii-tagger")]
    PiiModels {
        /// Also the accurate mode's BERTurk (about 440 MB)
        #[arg(long)]
        accurate: bool,
    },
}

/// What `pii-tag` runs: the published models of a mode, or model directories given on the command line.
#[cfg(feature = "pii-tagger")]
struct TaggerChoice {
    accurate: bool,
    strict: bool,
    model: Option<PathBuf>,
    big: Vec<PathBuf>,
    bias: Option<f32>,
    labels: Vec<String>,
    threads: Option<usize>,
    no_policy: bool,
}

#[cfg(feature = "pii-tagger")]
fn pii_mode(accurate: bool) -> pii::tagger::Mode {
    if accurate {
        pii::tagger::Mode::Accurate
    } else {
        pii::tagger::Mode::Fast
    }
}

/// The published models' directories, downloading what the cache lacks (reported on stderr).
#[cfg(feature = "pii-tagger")]
fn pii_model_dirs(mode: pii::tagger::Mode) -> Result<Vec<PathBuf>, String> {
    let mut report = |path: &str, size: u64| {
        eprintln!(
            "downloading {path} ({:.1} MB) from {}",
            size as f64 / 1e6,
            pii::tagger::hub::REPO
        );
    };
    pii::tagger::hub::model_dirs(mode, &mut report).map_err(|e| e.to_string())
}

#[cfg(feature = "pii-tagger")]
fn pii_tag(
    text: Option<String>,
    file: Option<PathBuf>,
    jsonl: Option<PathBuf>,
    choice: TaggerChoice,
) -> Result<(), String> {
    use std::io::{BufRead, Write};
    let mode = pii_mode(choice.accurate || !choice.big.is_empty());
    let mut options = if choice.strict {
        pii::tagger::TaggerOptions::strict(mode.strict_bias())
    } else {
        pii::tagger::TaggerOptions {
            bias: choice.bias.unwrap_or(0.0),
            bias_labels: choice.labels,
            ..Default::default()
        }
    };
    options.public_policy = !choice.no_policy;
    options.threads = choice.threads;
    let dirs: Vec<PathBuf> = match choice.model {
        Some(model) => std::iter::once(model).chain(choice.big).collect(),
        None => pii_model_dirs(mode)?,
    };
    let tagger = pii::tagger::PiiTagger::load(&dirs, options).map_err(|e| e.to_string())?;
    let Some(path) = jsonl else {
        let input = resolve_input_text(text, file)?;
        let spans = tagger.tag(&input).map_err(|e| e.to_string())?;
        let out: Vec<_> = spans
            .iter()
            .map(|s| {
                serde_json::json!({"start": s.start, "end": s.end, "label": s.label,
                                        "text": &input[s.start..s.end]})
            })
            .collect();
        println!(
            "{}",
            serde_json::to_string_pretty(&out).map_err(|e| e.to_string())?
        );
        return Ok(());
    };
    let reader = std::io::BufReader::new(
        std::fs::File::open(&path).map_err(|e| format!("{}: {e}", path.display()))?,
    );
    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    for line in reader.lines() {
        let line = line.map_err(|e| e.to_string())?;
        if line.trim().is_empty() {
            continue;
        }
        let row: serde_json::Value = serde_json::from_str(&line).map_err(|e| e.to_string())?;
        let input = row["text"]
            .as_str()
            .ok_or("a line without a \"text\" field")?;
        let spans = tagger.tag(input).map_err(|e| e.to_string())?;
        let spans: Vec<_> = spans
            .iter()
            .map(|s| serde_json::json!([s.start, s.end, s.label]))
            .collect();
        writeln!(out, "{}", serde_json::json!({"spans": spans})).map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn resolve_input_text(text: Option<String>, file: Option<PathBuf>) -> Result<String, String> {
    if let Some(f) = file {
        std::fs::read_to_string(&f)
            .map_err(|e| format!("Failed to read file '{}': {}", f.display(), e))
    } else if let Some(t) = text {
        Ok(t)
    } else {
        Err("Please provide input text as argument or use -f/--file <PATH>".to_string())
    }
}

/// The grammar rules the `grammar` command runs.
fn grammar_rules(
    rules: Option<Vec<String>>,
    enable: Vec<String>,
    disable: Vec<String>,
    experimental: bool,
) -> Result<grammar::RuleSet, grammar::UnknownRule> {
    let mut set = match rules {
        Some(ids) => grammar::RuleSet::from_ids(ids)?,
        None => grammar::RuleSet::default(),
    };
    if experimental {
        set = set.with_experimental();
    }
    for rule in grammar::RuleSet::from_ids(enable)?.iter() {
        set.enable(rule);
    }
    for rule in grammar::RuleSet::from_ids(disable)?.iter() {
        set.disable(rule);
    }
    Ok(set)
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Tokenize { text, file } => {
            let input = match resolve_input_text(text, file) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("Error: {e}");
                    std::process::exit(1);
                }
            };
            let tokens = tokenization::TurkishTokenizer::tokenize_words(&input);
            println!("{}", serde_json::to_string_pretty(&tokens).unwrap());
        }
        Commands::Analyze { word } => {
            let morph = morphology::TurkishMorphology::new();
            let parses = morph.analyze(&word);
            println!("{}", serde_json::to_string_pretty(&parses).unwrap());
        }
        Commands::SyntacticAnalyze { word } => {
            let morph = syntactic_morphology::TurkishSyntacticMorphology::new();
            let parses = morph.analyze(&word);
            println!("{}", serde_json::to_string_pretty(&parses).unwrap());
        }
        Commands::Generate { lemma, tags } => {
            let gen = morphology::TurkishGenerator::new();
            let tag_refs: Vec<&str> = tags.iter().map(|s| s.as_str()).collect();
            if let Some(surface) = gen.generate(&lemma, &tag_refs) {
                println!("Generated surface: {surface}");
            } else {
                println!("Failed to generate surface form.");
            }
        }
        Commands::Deasciify { text, file } => {
            let input = match resolve_input_text(text, file) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("Error: {e}");
                    std::process::exit(1);
                }
            };
            let res = normalization::TurkishDeasciifier::deasciify(&input);
            println!("{res}");
        }
        Commands::Asciify { text, file } => {
            let input = match resolve_input_text(text, file) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("Error: {e}");
                    std::process::exit(1);
                }
            };
            let res = normalization::TurkishAsciifier::asciify(&input);
            println!("{res}");
        }
        Commands::Normalize { text, file } => {
            let input = match resolve_input_text(text, file) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("Error: {e}");
                    std::process::exit(1);
                }
            };
            let res = normalization::TurkishInformalNormalizer::normalize_text(&input);
            println!("{res}");
        }
        Commands::Grammar {
            text,
            file,
            rules,
            enable,
            disable,
            experimental,
            correct,
            json,
            list_rules,
        } => {
            if list_rules {
                for rule in grammar::GrammarRule::ALL {
                    println!(
                        "{:<24} {:<12} confidence {:.2}  {}",
                        rule.id(),
                        if rule.is_default() {
                            "default"
                        } else {
                            "experimental"
                        },
                        rule.confidence(),
                        rule.summary()
                    );
                }
                return;
            }
            let input = match resolve_input_text(text, file) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("Error: {e}");
                    std::process::exit(1);
                }
            };
            let set = match grammar_rules(rules, enable, disable, experimental) {
                Ok(set) => set,
                Err(e) => {
                    eprintln!("Error: {e} (see --list-rules)");
                    std::process::exit(1);
                }
            };
            let checker = grammar::TurkishGrammarChecker::with_rules(set);
            let result = checker.check(&input);
            if json {
                println!("{}", serde_json::to_string_pretty(&result).unwrap());
            } else if correct {
                println!("{}", result.corrected);
            } else {
                for f in &result.findings {
                    println!(
                        "[{}] {:?} -> {:?}: {}",
                        f.rule, f.original_text, f.replacement, f.message_tr
                    );
                }
                println!("{}", result.corrected);
            }
        }
        Commands::Spellcheck { word } => {
            let checker = normalization::TurkishSpellChecker::new();
            let suggestions = checker.suggest(&word, 2, 5);
            println!("{}", serde_json::to_string_pretty(&suggestions).unwrap());
        }
        Commands::Parse { sentence, file } => {
            let input = match resolve_input_text(sentence, file) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("Error: {e}");
                    std::process::exit(1);
                }
            };
            let tokens = tokenization::TurkishTokenizer::tokenize_words(&input);
            let parser = parser::TurkishDependencyParser::new();
            let tree = parser.parse(&tokens);
            println!("{}", tree.to_conllu());
        }
        Commands::Readability { text, file } => {
            let input = match resolve_input_text(text, file) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("Error: {e}");
                    std::process::exit(1);
                }
            };
            let report = readability::analyze_readability(&input);
            println!("{}", serde_json::to_string_pretty(&report).unwrap());
        }
        Commands::Syllabify { word } => {
            let syllables = phonology::TurkishSyllabifier::syllabify(&word);
            let hyphenated = phonology::TurkishSyllabifier::hyphenate(&word, "-");
            println!(
                "Syllables: {:?} (Count: {}, Hyphenated: {})",
                syllables,
                syllables.len(),
                hyphenated
            );
        }
        Commands::Number { num } => {
            let words = normalization::TurkishNumberConverter::number_to_words(num);
            let ordinal = normalization::TurkishNumberConverter::ordinal_to_words(num);
            println!("Cardinal: {words}\nOrdinal:  {ordinal}");
        }
        Commands::Ner { text, file } => {
            let input = match resolve_input_text(text, file) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("Error: {e}");
                    std::process::exit(1);
                }
            };
            let entities = ner::TurkishNER::extract_entities(&input);
            println!("{}", serde_json::to_string_pretty(&entities).unwrap());
        }
        Commands::Keywords { text, file, top_k } => {
            let input = match resolve_input_text(text, file) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("Error: {e}");
                    std::process::exit(1);
                }
            };
            let extractor = analysis::TurkishKeywordExtractor::new();
            let kw = extractor.extract_keywords(&input, top_k);
            println!("{}", serde_json::to_string_pretty(&kw).unwrap());
        }
        Commands::Summarize {
            text,
            file,
            sentences,
        } => {
            let input = match resolve_input_text(text, file) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("Error: {e}");
                    std::process::exit(1);
                }
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
                Err(e) => {
                    eprintln!("Error: {e}");
                    std::process::exit(1);
                }
            };
            let auditor = style::TurkishStyleAuditor::new();
            let report = auditor.audit(&input);
            println!("{}", serde_json::to_string_pretty(&report).unwrap());
        }
        Commands::HumanizePrompt {
            text,
            file,
            register,
        } => {
            let input = match resolve_input_text(text, file) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("Error: {e}");
                    std::process::exit(1);
                }
            };
            let prompt = style::TurkishHumanizer::generate_prompt(&input, &register);
            println!("{prompt}");
        }
        Commands::Embed { text, file } => {
            let input = match resolve_input_text(text, file) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("Error: {e}");
                    std::process::exit(1);
                }
            };
            let embeddings = embeddings::TurkishEmbeddings::new();
            let vec = embeddings.embed(&input);
            println!("{}", serde_json::to_string(&vec).unwrap());
        }
        Commands::Similarity { text_a, text_b } => {
            let embeddings = embeddings::TurkishEmbeddings::new();
            let score = embeddings.similarity(&text_a, &text_b);
            println!("{score:.4}");
        }
        Commands::Chunk {
            text,
            file,
            strategy,
            chunk_size,
            overlap,
            min_chunk_size,
            threshold,
            mode,
            json,
        } => {
            let input = match resolve_input_text(text, file) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("Error: {e}");
                    std::process::exit(1);
                }
            };

            let thresh_mode = match mode.to_lowercase().as_str() {
                "similarity" | "sim" => {
                    chunking::ThresholdMode::Similarity(threshold.unwrap_or(0.32))
                }
                "percentile" | "perc" => {
                    chunking::ThresholdMode::Percentile(threshold.unwrap_or(0.75))
                }
                "stdev" | "std" => {
                    chunking::ThresholdMode::StandardDeviation(threshold.unwrap_or(0.80))
                }
                "iqr" => chunking::ThresholdMode::Interquartile(threshold.unwrap_or(1.0)),
                "auto" => chunking::ThresholdMode::Auto,
                _ => chunking::ThresholdMode::Percentile(threshold.unwrap_or(0.75)),
            };

            let chunks = match strategy.to_lowercase().as_str() {
                "sentence" => {
                    let chunker = chunking::SentenceChunker::new(chunk_size, overlap, 1);
                    chunker.chunk(&input)
                }
                "sdpm" => {
                    let chunker = chunking::SDPMChunker::new(
                        chunk_size,
                        thresh_mode,
                        threshold.unwrap_or(0.32),
                    )
                    .with_min_chunk_size(min_chunk_size);
                    chunker.chunk(&input)
                }
                _ => {
                    let chunker = chunking::SemanticChunker::new(chunk_size, thresh_mode)
                        .with_min_chunk_size(min_chunk_size);
                    chunker.chunk(&input)
                }
            };

            if json {
                println!("{}", serde_json::to_string_pretty(&chunks).unwrap());
            } else {
                println!("Generated {} chunk(s):\n", chunks.len());
                for (i, chunk) in chunks.iter().enumerate() {
                    println!(
                        "--- Chunk #{} (tokens: {}, chars: {}-{}) ---",
                        i + 1,
                        chunk.token_count,
                        chunk.start_index,
                        chunk.end_index
                    );
                    println!("{}\n", chunk.text);
                }
            }
        }
        Commands::PiiMask {
            text,
            file,
            mode,
            json,
        } => {
            let input = match resolve_input_text(text, file) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("Error: {e}");
                    std::process::exit(1);
                }
            };
            let engine = pii::TurkishPiiEngine::new();
            let pii_mode = match mode.to_lowercase().as_str() {
                "surrogate" => pii::PiiMode::SyntheticSurrogate,
                "tag" => pii::PiiMode::Tag,
                "anonymize" => pii::PiiMode::Anonymize,
                _ => pii::PiiMode::Placeholder,
            };
            let result = engine.mask(&input, pii_mode);
            if json {
                println!("{}", serde_json::to_string_pretty(&result).unwrap());
            } else {
                println!("{}", result.masked_text);
            }
        }
        Commands::PiiRestore {
            text,
            file,
            mapping,
        } => {
            let input = match resolve_input_text(text, file) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("Error: {e}");
                    std::process::exit(1);
                }
            };
            let map_str = match std::fs::read_to_string(&mapping) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("Failed to read mapping file: {e}");
                    std::process::exit(1);
                }
            };
            let map: std::collections::HashMap<String, String> =
                match serde_json::from_str(&map_str) {
                    Ok(m) => m,
                    Err(e) => {
                        eprintln!("Failed to parse mapping JSON: {e}");
                        std::process::exit(1);
                    }
                };
            let engine = pii::TurkishPiiEngine::new();
            let restored = engine.restore_with_mapping(&input, &map);
            println!("{}", restored);
        }
        #[cfg(feature = "pii-tagger")]
        Commands::PiiTag {
            text,
            file,
            jsonl,
            accurate,
            strict,
            model,
            big,
            bias,
            labels,
            threads,
            no_policy,
        } => {
            let choice = TaggerChoice {
                accurate,
                strict,
                model,
                big,
                bias,
                labels,
                threads,
                no_policy,
            };
            if let Err(e) = pii_tag(text, file, jsonl, choice) {
                eprintln!("Error: {e}");
                std::process::exit(1);
            }
        }
        #[cfg(feature = "pii-tagger")]
        Commands::PiiModels { accurate } => match pii_model_dirs(pii_mode(accurate)) {
            Ok(dirs) => {
                for dir in dirs {
                    println!("{}", dir.display());
                }
            }
            Err(e) => {
                eprintln!("Error: {e}");
                std::process::exit(1);
            }
        },
    }
}
