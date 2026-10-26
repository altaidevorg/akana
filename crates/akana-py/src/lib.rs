//! PyO3 Python bindings for the Akana Turkish NLP toolkit.
//!
//! Conventions every binding follows:
//! - Core calls run inside [`guard`], or [`release`] when they do real work. A Rust panic
//!   becomes a Python `RuntimeError` carrying the panic message: PyO3 would raise
//!   `PanicException`, which derives from `BaseException`, so `except Exception` would not
//!   catch it. [`release`] also releases the GIL, so Python threads run calls in parallel.
//! - Spans returned to Python (`start`/`end`, `start_offset`/`end_offset`) are character
//!   offsets, so `text[start:end]` is the span; the core reports UTF-8 byte offsets.
//! - Module-level functions use engines built once per process.
#![allow(clippy::useless_conversion)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::redundant_closure)]

use pyo3::exceptions::{PyIOError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use std::any::Any;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, LazyLock};

/// A `RuntimeError` for a Rust panic, with the panic message.
fn panic_error(payload: Box<dyn Any + Send>) -> PyErr {
    let message = payload
        .downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic".to_string());
    PyRuntimeError::new_err(format!("internal error in akana: {message}"))
}

/// Runs `f` with the GIL held, turning a panic into a `RuntimeError`. For cheap calls, where
/// releasing the GIL would cost more than the call.
fn guard<T>(f: impl FnOnce() -> T) -> PyResult<T> {
    catch_unwind(AssertUnwindSafe(f)).map_err(panic_error)
}

/// Runs `f` with the GIL released, turning a panic into a `RuntimeError`. Other Python threads
/// run meanwhile, so `f` must not touch Python objects.
fn release<T: Send>(py: Python<'_>, f: impl FnOnce() -> T + Send) -> PyResult<T> {
    py.allow_threads(|| catch_unwind(AssertUnwindSafe(f)))
        .map_err(panic_error)
}

fn json_error(e: serde_json::Error) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Character offsets (Python string indices) of byte offsets into `text`, in one pass over
/// the text: counting from the start of the text for every span was quadratic on span-dense
/// input (a log with an IP address on every line).
fn char_offsets(text: &str, bytes: &[usize]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..bytes.len()).collect();
    order.sort_by_key(|&i| bytes[i]);
    let mut chars_at = vec![0usize; bytes.len()];
    let (mut pos, mut chars) = (0usize, 0usize);
    let mut it = text.char_indices();
    for i in order {
        while pos < bytes[i] {
            match it.next() {
                Some((b, c)) => {
                    pos = b + c.len_utf8();
                    chars += 1;
                }
                None => break,
            }
        }
        chars_at[i] = chars;
    }
    chars_at
}

/// Rewrites `(start, end)` byte offsets into `text` as character offsets, in place.
fn spans_to_chars<'a>(text: &str, spans: impl IntoIterator<Item = (&'a mut usize, &'a mut usize)>) {
    let mut spans: Vec<_> = spans.into_iter().collect();
    let bytes: Vec<usize> = spans.iter().flat_map(|(s, e)| [**s, **e]).collect();
    let chars = char_offsets(text, &bytes);
    for ((start, end), c) in spans.iter_mut().zip(chars.chunks(2)) {
        **start = c[0];
        **end = c[1];
    }
}

/// Character offsets of the PII entities' byte spans.
fn entity_char_offsets(text: &str, entities: &[akana_core::pii::PiiEntity]) -> Vec<(usize, usize)> {
    let bytes: Vec<usize> = entities.iter().flat_map(|e| [e.start, e.end]).collect();
    char_offsets(text, &bytes)
        .chunks(2)
        .map(|p| (p[0], p[1]))
        .collect()
}

/// A grammar check of `text` with `rules`, its findings' offsets in characters.
fn grammar_check_json(
    checker: &akana_core::grammar::TurkishGrammarChecker,
    text: &str,
    rules: akana_core::grammar::RuleSet,
) -> Result<String, serde_json::Error> {
    let mut res = checker.check_with_rules(text, rules);
    spans_to_chars(
        text,
        res.findings
            .iter_mut()
            .map(|f| (&mut f.start_offset, &mut f.end_offset)),
    );
    serde_json::to_string(&res)
}

static ENGINE: LazyLock<akana_core::AkanaEngine> = LazyLock::new(akana_core::AkanaEngine::new);
static STEMMER: LazyLock<akana_core::morphology::TurkishStemmer> =
    LazyLock::new(akana_core::morphology::TurkishStemmer::new);
static STOPWORDS: LazyLock<akana_core::morphology::TurkishStopwords> =
    LazyLock::new(akana_core::morphology::TurkishStopwords::new);
static KEYWORD_EXTRACTOR: LazyLock<akana_core::analysis::TurkishKeywordExtractor> =
    LazyLock::new(akana_core::analysis::TurkishKeywordExtractor::new);
static SUMMARIZER: LazyLock<akana_core::analysis::TurkishSummarizer> =
    LazyLock::new(akana_core::analysis::TurkishSummarizer::new);
static STYLE_AUDITOR: LazyLock<akana_core::style::TurkishStyleAuditor> =
    LazyLock::new(akana_core::style::TurkishStyleAuditor::new);
static GRAMMAR_CHECKER: LazyLock<akana_core::grammar::TurkishGrammarChecker> =
    LazyLock::new(akana_core::grammar::TurkishGrammarChecker::new);
static EMBEDDINGS: LazyLock<akana_core::TurkishEmbeddings> =
    LazyLock::new(akana_core::TurkishEmbeddings::new);
static HYBRID_EMBEDDINGS: LazyLock<akana_core::TurkishHybridEmbeddings> =
    LazyLock::new(akana_core::TurkishHybridEmbeddings::new);
static SPARSE_ENCODER: LazyLock<akana_core::MorphologicalSparseEncoder> =
    LazyLock::new(akana_core::MorphologicalSparseEncoder::new);
/// A PII engine with the embedding scorer, whose centroids are computed once; engines with
/// `use_embeddings=True` start as a copy of it.
static PII_ENGINE_WITH_EMBEDDINGS: LazyLock<akana_core::pii::TurkishPiiEngine> =
    LazyLock::new(|| {
        akana_core::pii::TurkishPiiEngine::with_embeddings(Arc::new(
            akana_core::embeddings::TurkishEmbeddings::new(),
        ))
    });

#[pyfunction]
fn to_turkish_lower(text: &str) -> PyResult<String> {
    guard(|| akana_core::phonology::to_turkish_lower(text))
}

#[pyfunction]
fn to_turkish_upper(text: &str) -> PyResult<String> {
    guard(|| akana_core::phonology::to_turkish_upper(text))
}

#[pyfunction]
fn to_turkish_title(text: &str) -> PyResult<String> {
    guard(|| akana_core::phonology::to_turkish_title(text))
}

#[pyfunction]
fn check_major_vowel_harmony(word: &str) -> PyResult<bool> {
    guard(|| akana_core::phonology::check_major_vowel_harmony(word))
}

#[pyfunction]
fn check_minor_vowel_harmony(word: &str) -> PyResult<bool> {
    guard(|| akana_core::phonology::check_minor_vowel_harmony(word))
}

#[pyfunction]
fn asciify(py: Python<'_>, text: &str) -> PyResult<String> {
    release(py, || {
        akana_core::normalization::TurkishAsciifier::asciify(text)
    })
}

#[pyfunction]
fn deasciify(py: Python<'_>, text: &str) -> PyResult<String> {
    release(py, || {
        akana_core::normalization::TurkishDeasciifier::deasciify(text)
    })
}

#[pyfunction]
fn normalize_informal(py: Python<'_>, text: &str) -> PyResult<String> {
    release(py, || {
        akana_core::normalization::TurkishInformalNormalizer::normalize_text(text)
    })
}

#[pyfunction]
fn tokenize_words(py: Python<'_>, text: &str) -> PyResult<Vec<String>> {
    release(py, || {
        akana_core::tokenization::TurkishTokenizer::tokenize_words(text)
            .into_iter()
            .map(|s| s.to_string())
            .collect()
    })
}

#[pyclass(name = "SpellChecker")]
struct PySpellChecker {
    inner: akana_core::normalization::TurkishSpellChecker,
}

#[pymethods]
impl PySpellChecker {
    #[new]
    fn new(py: Python<'_>) -> PyResult<Self> {
        release(py, || Self {
            inner: akana_core::normalization::TurkishSpellChecker::new(),
        })
    }

    fn add_word(&mut self, word: &str) -> PyResult<()> {
        guard(|| self.inner.add_word(word))
    }

    fn is_correct(&self, py: Python<'_>, word: &str) -> PyResult<bool> {
        release(py, || self.inner.is_correct(word))
    }

    #[pyo3(signature = (word, max_distance=None, max_suggestions=None))]
    fn suggest<'py>(
        &self,
        py: Python<'py>,
        word: &str,
        max_distance: Option<usize>,
        max_suggestions: Option<usize>,
    ) -> PyResult<Bound<'py, PyList>> {
        let dist = max_distance.unwrap_or(2);
        let count = max_suggestions.unwrap_or(5);
        let suggestions = release(py, || self.inner.suggest(word, dist, count))?;

        let list = PyList::empty_bound(py);
        for s in suggestions {
            let dict = PyDict::new_bound(py);
            dict.set_item("word", s.word)?;
            dict.set_item("distance", s.distance)?;
            dict.set_item("score", s.score)?;
            list.append(dict)?;
        }
        Ok(list)
    }
}

#[pyclass(name = "Morphology")]
struct PyMorphology {
    inner: akana_core::morphology::TurkishMorphology,
}

#[pymethods]
impl PyMorphology {
    #[new]
    fn new(py: Python<'_>) -> PyResult<Self> {
        release(py, || Self {
            inner: akana_core::morphology::TurkishMorphology::new(),
        })
    }

    fn analyze<'py>(&self, py: Python<'py>, word: &str) -> PyResult<Bound<'py, PyList>> {
        let parses = release(py, || self.inner.analyze(word))?;
        let list = PyList::empty_bound(py);

        for p in parses {
            let dict = PyDict::new_bound(py);
            dict.set_item("surface", p.surface)?;
            dict.set_item("lemma", p.lemma)?;
            dict.set_item("root", p.root)?;
            dict.set_item("primary_pos", p.primary_pos.as_str())?;
            dict.set_item("secondary_pos", p.secondary_pos.as_str())?;
            dict.set_item("morphemes", p.morpheme_tags)?;
            dict.set_item("formatted", p.formatted)?;
            dict.set_item("analysis", p.analysis)?;
            dict.set_item("score", p.score)?;
            list.append(dict)?;
        }
        Ok(list)
    }

    /// Generates from this analyzer's lexicon, with any dictionary it has loaded.
    fn generate(&self, py: Python<'_>, lemma: &str, tags: Vec<String>) -> PyResult<Option<String>> {
        release(py, || {
            let tag_refs: Vec<&str> = tags.iter().map(|s| s.as_str()).collect();
            self.inner.generate(lemma, &tag_refs)
        })
    }

    fn load_dictionary_file(&mut self, py: Python<'_>, path: &str) -> PyResult<()> {
        let inner = &mut self.inner;
        release(py, || inner.load_dictionary_file(path))?
            .map_err(|e| PyIOError::new_err(e.to_string()))
    }

    fn load_dictionary_str(&mut self, py: Python<'_>, text: &str) -> PyResult<()> {
        let inner = &mut self.inner;
        release(py, || inner.load_dictionary_str(text))
    }
}

#[pyclass(name = "SyntacticMorphology")]
struct PySyntacticMorphology {
    inner: akana_core::syntactic_morphology::TurkishSyntacticMorphology,
}

#[pymethods]
impl PySyntacticMorphology {
    #[new]
    fn new(py: Python<'_>) -> PyResult<Self> {
        release(py, || Self {
            inner: akana_core::syntactic_morphology::TurkishSyntacticMorphology::new(),
        })
    }

    fn analyze<'py>(&self, py: Python<'py>, word: &str) -> PyResult<Bound<'py, PyList>> {
        let parses = release(py, || self.inner.analyze(word))?;
        let list = PyList::empty_bound(py);

        for p in parses {
            let dict = PyDict::new_bound(py);
            dict.set_item("surface", p.surface)?;
            dict.set_item("root", p.root)?;
            dict.set_item("root_pos", p.root_pos)?;
            dict.set_item("is_proper", p.is_proper)?;
            dict.set_item("formatted", p.formatted)?;

            let ig_list = PyList::empty_bound(py);
            for ig in p.inflectional_groups {
                let ig_dict = PyDict::new_bound(py);
                ig_dict.set_item("pos", ig.pos)?;
                ig_dict.set_item("derivation", ig.derivation)?;
                ig_dict.set_item("features", ig.features)?;
                ig_list.append(ig_dict)?;
            }
            dict.set_item("inflectional_groups", ig_list)?;

            list.append(dict)?;
        }
        Ok(list)
    }
}

#[pyclass(name = "CompoundDecomposer")]
struct PyCompoundDecomposer {
    inner: akana_core::morphology::CompoundDecomposer,
}

#[pymethods]
impl PyCompoundDecomposer {
    #[new]
    fn new(py: Python<'_>) -> PyResult<Self> {
        release(py, || Self {
            inner: akana_core::morphology::CompoundDecomposer::new(),
        })
    }

    fn decompose<'py>(&self, py: Python<'py>, word: &str) -> PyResult<Bound<'py, PyList>> {
        let analyses = release(py, || self.inner.decompose(word))?;
        let list = PyList::empty_bound(py);

        for a in analyses {
            let dict = PyDict::new_bound(py);
            dict.set_item("surface", a.surface)?;
            dict.set_item("part1", a.part1)?;
            dict.set_item("part2", a.part2)?;

            let parse1 = PyDict::new_bound(py);
            parse1.set_item("root", a.parse1.root)?;
            parse1.set_item("pos", a.parse1.primary_pos.as_str())?;
            dict.set_item("parse1", parse1)?;

            let parse2 = PyDict::new_bound(py);
            parse2.set_item("root", a.parse2.root)?;
            parse2.set_item("pos", a.parse2.primary_pos.as_str())?;
            dict.set_item("parse2", parse2)?;

            list.append(dict)?;
        }
        Ok(list)
    }
}

#[pyclass(name = "Disambiguator")]
struct PyDisambiguator {
    inner: akana_core::morphology::MorphologicalDisambiguator,
}

#[pymethods]
impl PyDisambiguator {
    #[new]
    fn new(py: Python<'_>) -> PyResult<Self> {
        release(py, || Self {
            inner: akana_core::morphology::MorphologicalDisambiguator::new(),
        })
    }

    fn disambiguate<'py>(
        &self,
        py: Python<'py>,
        tokens: Vec<String>,
    ) -> PyResult<Bound<'py, PyList>> {
        let parses = release(py, || {
            let token_refs: Vec<&str> = tokens.iter().map(|s| s.as_str()).collect();
            self.inner.disambiguate(&token_refs)
        })?;
        let list = PyList::empty_bound(py);

        for p in parses {
            let dict = PyDict::new_bound(py);
            dict.set_item("surface", p.surface)?;
            dict.set_item("lemma", p.lemma)?;
            dict.set_item("root", p.root)?;
            dict.set_item("primary_pos", p.primary_pos.as_str())?;
            dict.set_item("secondary_pos", p.secondary_pos.as_str())?;
            dict.set_item("morphemes", p.morpheme_tags)?;
            dict.set_item("formatted", p.formatted)?;
            dict.set_item("analysis", p.analysis)?;
            list.append(dict)?;
        }
        Ok(list)
    }
}

#[pyclass(name = "DependencyParser")]
struct PyDependencyParser {
    inner: akana_core::parser::TurkishDependencyParser,
}

#[pymethods]
impl PyDependencyParser {
    #[new]
    fn new(py: Python<'_>) -> PyResult<Self> {
        release(py, || Self {
            inner: akana_core::parser::TurkishDependencyParser::new(),
        })
    }

    fn parse_conllu(&self, py: Python<'_>, tokens: Vec<String>) -> PyResult<String> {
        release(py, || {
            let token_refs: Vec<&str> = tokens.iter().map(|s| s.as_str()).collect();
            self.inner.parse(&token_refs).to_conllu()
        })
    }
}

#[pyfunction]
fn analyze_document_json(py: Python<'_>, text: &str) -> PyResult<String> {
    release(py, || serde_json::to_string(&ENGINE.analyze_document(text)))?.map_err(json_error)
}

#[pyfunction]
fn analyze_readability_json(py: Python<'_>, text: &str) -> PyResult<String> {
    release(py, || {
        serde_json::to_string(&akana_core::readability::analyze_readability(text))
    })?
    .map_err(json_error)
}

#[pyfunction]
fn syllabify(word: &str) -> PyResult<Vec<String>> {
    guard(|| akana_core::phonology::TurkishSyllabifier::syllabify(word))
}

#[pyfunction]
#[pyo3(signature = (word, delimiter=None))]
fn hyphenate(word: &str, delimiter: Option<&str>) -> PyResult<String> {
    let delim = delimiter.unwrap_or("-");
    guard(|| akana_core::phonology::TurkishSyllabifier::hyphenate(word, delim))
}

#[pyfunction]
fn count_syllables(word: &str) -> PyResult<usize> {
    guard(|| akana_core::phonology::TurkishSyllabifier::count_syllables(word))
}

#[pyfunction]
fn number_to_words(n: i64) -> PyResult<String> {
    guard(|| akana_core::normalization::TurkishNumberConverter::number_to_words(n))
}

#[pyfunction]
fn ordinal_to_words(n: i64) -> PyResult<String> {
    guard(|| akana_core::normalization::TurkishNumberConverter::ordinal_to_words(n))
}

#[pyfunction]
#[pyo3(signature = (amount, currency=None))]
fn currency_to_words(amount: f64, currency: Option<&str>) -> PyResult<String> {
    let curr = currency.unwrap_or("TL");
    guard(|| akana_core::normalization::TurkishNumberConverter::currency_to_words(amount, curr))
}

#[pyfunction]
fn words_to_number(text: &str) -> PyResult<i64> {
    guard(|| akana_core::normalization::TurkishNumberConverter::words_to_number(text))?
        .map_err(|e| PyValueError::new_err(e))
}

#[pyfunction]
fn stem(py: Python<'_>, word: &str) -> PyResult<String> {
    release(py, || STEMMER.stem(word))
}

#[pyfunction]
fn is_stopword(word: &str) -> PyResult<bool> {
    guard(|| STOPWORDS.is_stopword(word))
}

#[pyfunction]
fn remove_stopwords(tokens: Vec<String>) -> PyResult<Vec<String>> {
    guard(|| {
        let token_refs: Vec<&str> = tokens.iter().map(|s| s.as_str()).collect();
        STOPWORDS
            .filter_tokens(&token_refs)
            .into_iter()
            .map(|s| s.to_string())
            .collect()
    })
}

/// Named entities as JSON; `start`/`end` are character offsets into `text`.
#[pyfunction]
fn extract_entities_json(py: Python<'_>, text: &str) -> PyResult<String> {
    release(py, || {
        let mut entities = akana_core::ner::TurkishNER::extract_entities(text);
        spans_to_chars(
            text,
            entities.iter_mut().map(|e| (&mut e.start, &mut e.end)),
        );
        serde_json::to_string(&entities)
    })?
    .map_err(json_error)
}

#[pyfunction]
#[pyo3(signature = (text, top_k=None))]
fn extract_keywords_json(py: Python<'_>, text: &str, top_k: Option<usize>) -> PyResult<String> {
    release(py, || {
        serde_json::to_string(&KEYWORD_EXTRACTOR.extract_keywords(text, top_k.unwrap_or(10)))
    })?
    .map_err(json_error)
}

#[pyfunction]
#[pyo3(signature = (text, max_sentences=None))]
fn summarize(py: Python<'_>, text: &str, max_sentences: Option<usize>) -> PyResult<Vec<String>> {
    release(py, || {
        SUMMARIZER.summarize(text, max_sentences.unwrap_or(3))
    })
}

/// The style audit as JSON; findings' `start`/`end` are character offsets into `text`.
#[pyfunction]
fn audit_ai_style_json(py: Python<'_>, text: &str) -> PyResult<String> {
    release(py, || {
        let mut report = STYLE_AUDITOR.audit(text);
        spans_to_chars(
            text,
            report
                .findings
                .iter_mut()
                .map(|f| (&mut f.start, &mut f.end)),
        );
        serde_json::to_string(&report)
    })?
    .map_err(json_error)
}

#[pyfunction]
#[pyo3(signature = (text, register=None))]
fn generate_humanizer_prompt(
    py: Python<'_>,
    text: &str,
    register: Option<&str>,
) -> PyResult<String> {
    let reg = register.unwrap_or("blog");
    release(py, || {
        akana_core::style::TurkishHumanizer::generate_prompt(text, reg)
    })
}

fn grammar_rule(id: &str) -> PyResult<akana_core::grammar::GrammarRule> {
    id.parse()
        .map_err(|e: akana_core::grammar::UnknownRule| PyValueError::new_err(e.to_string()))
}

/// The grammar rules to run: `rules` (ids) or else the default set, plus every experimental
/// rule with `experimental`, then `enable` added and `disable` removed. An unknown id raises
/// `ValueError`.
fn grammar_rule_set(
    rules: Option<Vec<String>>,
    enable: Option<Vec<String>>,
    disable: Option<Vec<String>>,
    experimental: bool,
) -> PyResult<akana_core::grammar::RuleSet> {
    let mut set = match rules {
        Some(ids) => ids
            .iter()
            .map(|id| grammar_rule(id))
            .collect::<PyResult<_>>()?,
        None => akana_core::grammar::RuleSet::default(),
    };
    if experimental {
        set = set.with_experimental();
    }
    for id in enable.iter().flatten() {
        set.enable(grammar_rule(id)?);
    }
    for id in disable.iter().flatten() {
        set.disable(grammar_rule(id)?);
    }
    Ok(set)
}

/// A grammar checker with its own rule selection; checks run on the process's shared engine.
/// `GrammarChecker()` runs the default rules. `rules` replaces them with a list of rule ids,
/// `experimental=True` adds every experimental rule, and `enable`/`disable` add or remove ids.
#[pyclass(name = "GrammarChecker")]
struct PyGrammarChecker {
    rules: akana_core::grammar::RuleSet,
}

#[pymethods]
impl PyGrammarChecker {
    #[new]
    #[pyo3(signature = (rules=None, enable=None, disable=None, experimental=false))]
    fn new(
        rules: Option<Vec<String>>,
        enable: Option<Vec<String>>,
        disable: Option<Vec<String>>,
        experimental: bool,
    ) -> PyResult<Self> {
        Ok(Self {
            rules: grammar_rule_set(rules, enable, disable, experimental)?,
        })
    }

    /// The check as JSON; findings' `start_offset`/`end_offset` are character offsets into
    /// `text`.
    fn check_json(&self, py: Python<'_>, text: &str) -> PyResult<String> {
        let rules = self.rules;
        release(py, || grammar_check_json(&GRAMMAR_CHECKER, text, rules))?.map_err(json_error)
    }

    fn correct(&self, py: Python<'_>, text: &str) -> PyResult<String> {
        let rules = self.rules;
        release(py, || GRAMMAR_CHECKER.correct_with_rules(text, rules))
    }

    fn enable_rule(&mut self, rule: &str) -> PyResult<()> {
        self.rules.enable(grammar_rule(rule)?);
        Ok(())
    }

    fn disable_rule(&mut self, rule: &str) -> PyResult<()> {
        self.rules.disable(grammar_rule(rule)?);
        Ok(())
    }

    /// Also runs every experimental rule.
    fn enable_experimental(&mut self) {
        self.rules = self.rules.with_experimental();
    }

    /// The ids of the rules this checker runs.
    fn enabled_rules(&self) -> Vec<&'static str> {
        self.rules.iter().map(|r| r.id()).collect()
    }
}

/// The grammar check as JSON; findings' `start_offset`/`end_offset` are character offsets
/// into `text`. The rule arguments are those of `GrammarChecker`.
#[pyfunction]
#[pyo3(signature = (text, rules=None, enable=None, disable=None, experimental=false))]
fn check_grammar_json(
    py: Python<'_>,
    text: &str,
    rules: Option<Vec<String>>,
    enable: Option<Vec<String>>,
    disable: Option<Vec<String>>,
    experimental: bool,
) -> PyResult<String> {
    let rules = grammar_rule_set(rules, enable, disable, experimental)?;
    release(py, || grammar_check_json(&GRAMMAR_CHECKER, text, rules))?.map_err(json_error)
}

/// `text` with the grammar corrections applied. The rule arguments are those of
/// `GrammarChecker`.
#[pyfunction]
#[pyo3(signature = (text, rules=None, enable=None, disable=None, experimental=false))]
fn correct_grammar(
    py: Python<'_>,
    text: &str,
    rules: Option<Vec<String>>,
    enable: Option<Vec<String>>,
    disable: Option<Vec<String>>,
    experimental: bool,
) -> PyResult<String> {
    let rules = grammar_rule_set(rules, enable, disable, experimental)?;
    release(py, || GRAMMAR_CHECKER.correct_with_rules(text, rules))
}

/// Every grammar rule as JSON: `id`, `category`, `default` (in the default set),
/// `confidence` (its measured precision) and `summary`.
#[pyfunction]
fn grammar_rules_json() -> PyResult<String> {
    let rules: Vec<serde_json::Value> = akana_core::grammar::GrammarRule::ALL
        .iter()
        .map(|r| {
            serde_json::json!({
                "id": r.id(),
                "category": r.category().as_str(),
                "default": r.is_default(),
                "confidence": r.confidence(),
                "summary": r.summary(),
            })
        })
        .collect();
    serde_json::to_string(&rules).map_err(json_error)
}

fn packed_2bit<'a>(v: &'a [u8], what: &str) -> PyResult<&'a [u8; 64]> {
    v.try_into().map_err(|_| {
        PyValueError::new_err(format!("{what} packed vectors must be exactly 64 bytes"))
    })
}

#[pyclass(name = "Embeddings")]
struct PyEmbeddings {
    inner: akana_core::TurkishEmbeddings,
}

#[pymethods]
impl PyEmbeddings {
    #[new]
    fn new(py: Python<'_>) -> PyResult<Self> {
        release(py, || Self {
            inner: akana_core::TurkishEmbeddings::new(),
        })
    }

    fn embed(&self, py: Python<'_>, text: &str) -> PyResult<Vec<f32>> {
        release(py, || self.inner.embed(text))
    }

    fn embed_packed_2bit<'py>(&self, py: Python<'py>, text: &str) -> PyResult<Bound<'py, PyBytes>> {
        let packed = release(py, || self.inner.embed_packed_2bit(text))?;
        Ok(PyBytes::new_bound(py, &packed))
    }

    fn embed_packed_1bit<'py>(&self, py: Python<'py>, text: &str) -> PyResult<Bound<'py, PyBytes>> {
        let packed = release(py, || self.inner.embed_packed_1bit(text))?;
        Ok(PyBytes::new_bound(py, &packed))
    }

    fn embed_batch_packed_2bit<'py>(
        &self,
        py: Python<'py>,
        texts: Vec<String>,
    ) -> PyResult<Bound<'py, PyList>> {
        let packed_vecs = release(py, || {
            let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            self.inner.embed_batch_packed_2bit(&text_refs)
        })?;
        Ok(PyList::new_bound(
            py,
            packed_vecs.iter().map(|p| PyBytes::new_bound(py, p)),
        ))
    }

    fn embed_batch_packed_1bit<'py>(
        &self,
        py: Python<'py>,
        texts: Vec<String>,
    ) -> PyResult<Bound<'py, PyList>> {
        let packed_vecs = release(py, || {
            let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            self.inner.embed_batch_packed_1bit(&text_refs)
        })?;
        Ok(PyList::new_bound(
            py,
            packed_vecs.iter().map(|p| PyBytes::new_bound(py, p)),
        ))
    }

    fn sparse_embed<'py>(&self, py: Python<'py>, text: &str) -> PyResult<Bound<'py, PyDict>> {
        sparse_embed(py, text)
    }

    fn hybrid_embed<'py>(&self, py: Python<'py>, text: &str) -> PyResult<Bound<'py, PyBytes>> {
        hybrid_embed(py, text)
    }

    fn embed_hybrid<'py>(&self, py: Python<'py>, text: &str) -> PyResult<Bound<'py, PyBytes>> {
        hybrid_embed(py, text)
    }

    fn embed_batch_hybrid<'py>(
        &self,
        py: Python<'py>,
        texts: Vec<String>,
    ) -> PyResult<Bound<'py, PyList>> {
        embed_batch_hybrid(py, texts)
    }

    fn similarity_packed_2bit(&self, a: &[u8], b: &[u8]) -> PyResult<f32> {
        let arr_a = packed_2bit(a, "2-bit")?;
        let arr_b = packed_2bit(b, "2-bit")?;
        guard(|| akana_core::dot_product_2bit_64(arr_a, arr_b))
    }

    fn distance_packed_2bit(&self, a: &[u8], b: &[u8]) -> PyResult<f32> {
        let arr_a = packed_2bit(a, "2-bit")?;
        let arr_b = packed_2bit(b, "2-bit")?;
        guard(|| akana_core::cosine_distance_2bit_64(arr_a, arr_b))
    }

    fn distance_packed_1bit(&self, a: &[u8], b: &[u8]) -> PyResult<f32> {
        let arr_a: &[u8; 32] = a
            .try_into()
            .map_err(|_| PyValueError::new_err("1-bit packed vectors must be exactly 32 bytes"))?;
        let arr_b: &[u8; 32] = b
            .try_into()
            .map_err(|_| PyValueError::new_err("1-bit packed vectors must be exactly 32 bytes"))?;
        guard(|| akana_core::hamming_distance_1bit_32(arr_a, arr_b))
    }

    #[pyo3(signature = (a, b, alpha=None))]
    fn distance_hybrid(&self, a: &[u8], b: &[u8], alpha: Option<f32>) -> PyResult<f32> {
        let arr_a: &[u8; 128] = a
            .try_into()
            .map_err(|_| PyValueError::new_err("hybrid vectors must be exactly 128 bytes"))?;
        let arr_b: &[u8; 128] = b
            .try_into()
            .map_err(|_| PyValueError::new_err("hybrid vectors must be exactly 128 bytes"))?;
        let a_val = alpha.unwrap_or(0.5);
        guard(|| akana_core::hybrid_distance_128(arr_a, arr_b, a_val))
    }

    fn tokenize(&self, py: Python<'_>, text: &str) -> PyResult<Vec<usize>> {
        release(py, || self.inner.tokenize(text))
    }

    fn embed_batch(&self, py: Python<'_>, texts: Vec<String>) -> PyResult<Vec<Vec<f32>>> {
        release(py, || {
            let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            self.inner.embed_batch(&text_refs)
        })
    }

    fn similarity(&self, py: Python<'_>, text_a: &str, text_b: &str) -> PyResult<f32> {
        release(py, || self.inner.similarity(text_a, text_b))
    }

    #[staticmethod]
    fn similarity_vectors(vec_a: Vec<f32>, vec_b: Vec<f32>) -> PyResult<f32> {
        guard(|| akana_core::cosine_similarity(&vec_a, &vec_b))
    }

    #[getter]
    fn dimension(&self) -> usize {
        self.inner.embedding_dim()
    }
}

#[pyfunction]
fn embed(py: Python<'_>, text: &str) -> PyResult<Vec<f32>> {
    release(py, || EMBEDDINGS.embed(text))
}

#[pyfunction]
fn embed_packed_2bit<'py>(py: Python<'py>, text: &str) -> PyResult<Bound<'py, PyBytes>> {
    let packed = release(py, || EMBEDDINGS.embed_packed_2bit(text))?;
    Ok(PyBytes::new_bound(py, &packed))
}

#[pyfunction]
fn embed_packed_1bit<'py>(py: Python<'py>, text: &str) -> PyResult<Bound<'py, PyBytes>> {
    let packed = release(py, || EMBEDDINGS.embed_packed_1bit(text))?;
    Ok(PyBytes::new_bound(py, &packed))
}

#[pyfunction]
fn embed_batch_packed_2bit<'py>(
    py: Python<'py>,
    texts: Vec<String>,
) -> PyResult<Bound<'py, PyList>> {
    let packed_vecs = release(py, || {
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        EMBEDDINGS.embed_batch_packed_2bit(&text_refs)
    })?;
    Ok(PyList::new_bound(
        py,
        packed_vecs.iter().map(|p| PyBytes::new_bound(py, p)),
    ))
}

#[pyfunction]
fn embed_batch_packed_1bit<'py>(
    py: Python<'py>,
    texts: Vec<String>,
) -> PyResult<Bound<'py, PyList>> {
    let packed_vecs = release(py, || {
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        EMBEDDINGS.embed_batch_packed_1bit(&text_refs)
    })?;
    Ok(PyList::new_bound(
        py,
        packed_vecs.iter().map(|p| PyBytes::new_bound(py, p)),
    ))
}

#[pyfunction]
fn sparse_embed<'py>(py: Python<'py>, text: &str) -> PyResult<Bound<'py, PyDict>> {
    let sparse = release(py, || SPARSE_ENCODER.encode_sparse(text))?;
    let dict = PyDict::new_bound(py);
    dict.set_item("indices", sparse.indices)?;
    dict.set_item("values", sparse.values)?;
    dict.set_item("terms", sparse.terms)?;
    Ok(dict)
}

#[pyfunction]
fn hybrid_embed<'py>(py: Python<'py>, text: &str) -> PyResult<Bound<'py, PyBytes>> {
    let hybrid = release(py, || HYBRID_EMBEDDINGS.embed_hybrid(text))?;
    Ok(PyBytes::new_bound(py, &hybrid))
}

#[pyfunction]
fn embed_hybrid<'py>(py: Python<'py>, text: &str) -> PyResult<Bound<'py, PyBytes>> {
    hybrid_embed(py, text)
}

#[pyfunction]
fn embed_batch_hybrid<'py>(py: Python<'py>, texts: Vec<String>) -> PyResult<Bound<'py, PyList>> {
    let hybrid_vecs = release(py, || {
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        HYBRID_EMBEDDINGS.embed_hybrid_batch(&text_refs)
    })?;
    Ok(PyList::new_bound(
        py,
        hybrid_vecs.iter().map(|h| PyBytes::new_bound(py, h)),
    ))
}

#[pyfunction]
fn get_usearch_metric_pointer_2bit() -> PyResult<usize> {
    guard(akana_core::get_usearch_metric_pointer_2bit)
}

#[pyfunction]
fn get_usearch_metric_pointer_1bit() -> PyResult<usize> {
    guard(akana_core::get_usearch_metric_pointer_1bit)
}

#[pyfunction]
#[pyo3(signature = (alpha=None))]
fn get_usearch_metric_pointer_hybrid(alpha: Option<f32>) -> PyResult<usize> {
    guard(|| akana_core::get_usearch_metric_pointer_hybrid(alpha))
}

#[pyfunction]
fn set_hybrid_metric_alpha(alpha: f32) -> PyResult<()> {
    guard(|| akana_core::set_hybrid_metric_alpha(alpha))
}

#[pyfunction]
fn get_hybrid_metric_alpha() -> PyResult<f32> {
    guard(akana_core::embeddings::hybrid::get_hybrid_metric_alpha)
}

#[pyfunction]
#[pyo3(signature = (dense_results, sparse_results, k=None, alpha=None))]
fn reciprocal_rank_fusion(
    py: Python<'_>,
    dense_results: Vec<(u64, f32)>,
    sparse_results: Vec<(u64, f32)>,
    k: Option<f32>,
    alpha: Option<f32>,
) -> PyResult<Vec<(u64, f32)>> {
    let k_val = k.unwrap_or(60.0);
    let alpha_val = alpha.unwrap_or(1.0);
    release(py, || {
        akana_core::reciprocal_rank_fusion(&dense_results, &sparse_results, k_val, alpha_val)
            .into_iter()
            .map(|f| (f.id, f.score))
            .collect()
    })
}

#[pyfunction]
fn tokenize_embedding_text(py: Python<'_>, text: &str) -> PyResult<Vec<usize>> {
    release(py, || EMBEDDINGS.tokenize(text))
}

#[pyfunction]
fn embed_batch(py: Python<'_>, texts: Vec<String>) -> PyResult<Vec<Vec<f32>>> {
    release(py, || {
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        EMBEDDINGS.embed_batch(&text_refs)
    })
}

#[pyfunction]
fn similarity(py: Python<'_>, text_a: &str, text_b: &str) -> PyResult<f32> {
    release(py, || EMBEDDINGS.similarity(text_a, text_b))
}

#[pyfunction]
fn similarity_vectors(vec_a: Vec<f32>, vec_b: Vec<f32>) -> PyResult<f32> {
    guard(|| akana_core::cosine_similarity(&vec_a, &vec_b))
}

fn parse_threshold_mode(
    threshold: Option<f32>,
    threshold_mode: Option<&str>,
    threshold_value: Option<f32>,
) -> akana_core::chunking::ThresholdMode {
    use akana_core::chunking::ThresholdMode;

    if let Some(mode_str) = threshold_mode {
        match mode_str.to_lowercase().as_str() {
            "similarity" | "sim" => {
                let val = threshold_value.or(threshold).unwrap_or(0.70);
                ThresholdMode::Similarity(val)
            }
            "percentile" | "perc" => {
                let val = threshold_value.or(threshold).unwrap_or(0.75);
                ThresholdMode::Percentile(val)
            }
            "stdev" | "std" | "standard_deviation" => {
                let val = threshold_value.or(threshold).unwrap_or(0.80);
                ThresholdMode::StandardDeviation(val)
            }
            "iqr" | "interquartile" => {
                let val = threshold_value.or(threshold).unwrap_or(1.0);
                ThresholdMode::Interquartile(val)
            }
            "auto" => ThresholdMode::Auto,
            _ => ThresholdMode::Percentile(threshold.unwrap_or(0.75)),
        }
    } else if let Some(t) = threshold {
        // If 0 < t <= 1 and caller passed threshold as float, treat as Similarity if >= 0.1, or Percentile
        ThresholdMode::Similarity(t)
    } else {
        ThresholdMode::Percentile(0.75)
    }
}

#[pyclass(name = "Chunk")]
#[derive(Clone)]
struct PyChunk {
    inner: akana_core::chunking::Chunk,
}

#[pymethods]
impl PyChunk {
    #[getter]
    fn text(&self) -> String {
        self.inner.text.clone()
    }

    #[getter]
    fn start_index(&self) -> usize {
        self.inner.start_index
    }

    #[getter]
    fn end_index(&self) -> usize {
        self.inner.end_index
    }

    #[getter]
    fn token_count(&self) -> usize {
        self.inner.token_count
    }

    #[getter]
    fn sentences(&self) -> Vec<String> {
        self.inner.sentences.clone()
    }

    fn __repr__(&self) -> String {
        let text_chars: Vec<char> = self.inner.text.chars().collect();
        let preview = if text_chars.len() > 40 {
            let s: String = text_chars.into_iter().take(37).collect();
            format!("{}...", s)
        } else {
            self.inner.text.clone()
        };
        format!(
            "<Chunk tokens={} ({}:{}) text='{}'>",
            self.inner.token_count, self.inner.start_index, self.inner.end_index, preview
        )
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new_bound(py);
        dict.set_item("text", &self.inner.text)?;
        dict.set_item("start_index", self.inner.start_index)?;
        dict.set_item("end_index", self.inner.end_index)?;
        dict.set_item("token_count", self.inner.token_count)?;
        dict.set_item("sentences", &self.inner.sentences)?;
        Ok(dict)
    }
}

fn py_chunks(chunks: Vec<akana_core::chunking::Chunk>) -> Vec<PyChunk> {
    chunks.into_iter().map(|c| PyChunk { inner: c }).collect()
}

fn py_chunk_batches(batches: Vec<Vec<akana_core::chunking::Chunk>>) -> Vec<Vec<PyChunk>> {
    batches.into_iter().map(py_chunks).collect()
}

#[pyclass(name = "SemanticChunker")]
struct PySemanticChunker {
    inner: akana_core::chunking::SemanticChunker,
}

impl PySemanticChunker {
    fn build(
        chunk_size: Option<usize>,
        threshold: Option<f32>,
        threshold_mode: Option<&str>,
        threshold_value: Option<f32>,
        similarity_window: Option<usize>,
        min_chunk_size: Option<usize>,
        min_sentences_per_chunk: Option<usize>,
        use_smoothing: Option<bool>,
    ) -> akana_core::chunking::SemanticChunker {
        let size = chunk_size.unwrap_or(512);
        let mode = parse_threshold_mode(threshold, threshold_mode, threshold_value);
        let mut chunker = akana_core::chunking::SemanticChunker::new(size, mode);

        if let Some(w) = similarity_window {
            chunker = chunker.with_similarity_window(w);
        }
        if let Some(min_s) = min_chunk_size {
            chunker = chunker.with_min_chunk_size(min_s);
        }
        if let Some(min_sent) = min_sentences_per_chunk {
            chunker = chunker.with_min_sentences_per_chunk(min_sent);
        }
        if let Some(smooth) = use_smoothing {
            chunker = chunker.with_smoothing(smooth);
        }
        chunker
    }
}

#[pymethods]
impl PySemanticChunker {
    #[new]
    #[pyo3(signature = (
        chunk_size=None,
        threshold=None,
        threshold_mode=None,
        threshold_value=None,
        similarity_window=None,
        min_chunk_size=None,
        min_sentences_per_chunk=None,
        use_smoothing=None
    ))]
    fn new(
        py: Python<'_>,
        chunk_size: Option<usize>,
        threshold: Option<f32>,
        threshold_mode: Option<&str>,
        threshold_value: Option<f32>,
        similarity_window: Option<usize>,
        min_chunk_size: Option<usize>,
        min_sentences_per_chunk: Option<usize>,
        use_smoothing: Option<bool>,
    ) -> PyResult<Self> {
        release(py, || Self {
            inner: Self::build(
                chunk_size,
                threshold,
                threshold_mode,
                threshold_value,
                similarity_window,
                min_chunk_size,
                min_sentences_per_chunk,
                use_smoothing,
            ),
        })
    }

    fn chunk(&self, py: Python<'_>, text: &str) -> PyResult<Vec<PyChunk>> {
        release(py, || py_chunks(self.inner.chunk(text)))
    }

    fn chunk_batch(&self, py: Python<'_>, texts: Vec<String>) -> PyResult<Vec<Vec<PyChunk>>> {
        release(py, || {
            let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            py_chunk_batches(self.inner.chunk_batch(&text_refs))
        })
    }

    fn __call__(&self, py: Python<'_>, text: &str) -> PyResult<Vec<PyChunk>> {
        self.chunk(py, text)
    }
}

#[pyclass(name = "SentenceChunker")]
struct PySentenceChunker {
    inner: akana_core::chunking::SentenceChunker,
}

impl PySentenceChunker {
    fn build(
        chunk_size: Option<usize>,
        chunk_overlap: Option<usize>,
        min_sentences_per_chunk: Option<usize>,
    ) -> akana_core::chunking::SentenceChunker {
        let size = chunk_size.unwrap_or(512);
        let overlap = chunk_overlap.unwrap_or(0);
        let min_sent = min_sentences_per_chunk.unwrap_or(1);
        akana_core::chunking::SentenceChunker::new(size, overlap, min_sent)
    }
}

#[pymethods]
impl PySentenceChunker {
    #[new]
    #[pyo3(signature = (chunk_size=None, chunk_overlap=None, min_sentences_per_chunk=None))]
    fn new(
        py: Python<'_>,
        chunk_size: Option<usize>,
        chunk_overlap: Option<usize>,
        min_sentences_per_chunk: Option<usize>,
    ) -> PyResult<Self> {
        release(py, || Self {
            inner: Self::build(chunk_size, chunk_overlap, min_sentences_per_chunk),
        })
    }

    fn chunk(&self, py: Python<'_>, text: &str) -> PyResult<Vec<PyChunk>> {
        release(py, || py_chunks(self.inner.chunk(text)))
    }

    fn chunk_batch(&self, py: Python<'_>, texts: Vec<String>) -> PyResult<Vec<Vec<PyChunk>>> {
        release(py, || {
            let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            py_chunk_batches(self.inner.chunk_batch(&text_refs))
        })
    }

    fn __call__(&self, py: Python<'_>, text: &str) -> PyResult<Vec<PyChunk>> {
        self.chunk(py, text)
    }
}

#[pyclass(name = "SDPMChunker")]
struct PySDPMChunker {
    inner: akana_core::chunking::SDPMChunker,
}

impl PySDPMChunker {
    fn build(
        chunk_size: Option<usize>,
        threshold: Option<f32>,
        threshold_mode: Option<&str>,
        threshold_value: Option<f32>,
        merge_threshold: Option<f32>,
        similarity_window: Option<usize>,
        min_chunk_size: Option<usize>,
    ) -> akana_core::chunking::SDPMChunker {
        let size = chunk_size.unwrap_or(512);
        let mode = parse_threshold_mode(threshold, threshold_mode, threshold_value);
        let merge_thresh = merge_threshold.unwrap_or(0.32);
        let mut chunker = akana_core::chunking::SDPMChunker::new(size, mode, merge_thresh);

        if let Some(w) = similarity_window {
            chunker = chunker.with_similarity_window(w);
        }
        if let Some(min_s) = min_chunk_size {
            chunker = chunker.with_min_chunk_size(min_s);
        }
        chunker
    }
}

#[pymethods]
impl PySDPMChunker {
    #[new]
    #[pyo3(signature = (
        chunk_size=None,
        threshold=None,
        threshold_mode=None,
        threshold_value=None,
        merge_threshold=None,
        similarity_window=None,
        min_chunk_size=None
    ))]
    fn new(
        py: Python<'_>,
        chunk_size: Option<usize>,
        threshold: Option<f32>,
        threshold_mode: Option<&str>,
        threshold_value: Option<f32>,
        merge_threshold: Option<f32>,
        similarity_window: Option<usize>,
        min_chunk_size: Option<usize>,
    ) -> PyResult<Self> {
        release(py, || Self {
            inner: Self::build(
                chunk_size,
                threshold,
                threshold_mode,
                threshold_value,
                merge_threshold,
                similarity_window,
                min_chunk_size,
            ),
        })
    }

    fn chunk(&self, py: Python<'_>, text: &str) -> PyResult<Vec<PyChunk>> {
        release(py, || py_chunks(self.inner.chunk(text)))
    }

    fn chunk_batch(&self, py: Python<'_>, texts: Vec<String>) -> PyResult<Vec<Vec<PyChunk>>> {
        release(py, || {
            let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            py_chunk_batches(self.inner.chunk_batch(&text_refs))
        })
    }

    fn __call__(&self, py: Python<'_>, text: &str) -> PyResult<Vec<PyChunk>> {
        self.chunk(py, text)
    }
}

#[pyfunction]
#[pyo3(signature = (
    text,
    chunk_size=None,
    threshold=None,
    threshold_mode=None,
    threshold_value=None,
    similarity_window=None,
    min_chunk_size=None,
    min_sentences_per_chunk=None,
    use_smoothing=None
))]
fn chunk_semantic(
    py: Python<'_>,
    text: &str,
    chunk_size: Option<usize>,
    threshold: Option<f32>,
    threshold_mode: Option<&str>,
    threshold_value: Option<f32>,
    similarity_window: Option<usize>,
    min_chunk_size: Option<usize>,
    min_sentences_per_chunk: Option<usize>,
    use_smoothing: Option<bool>,
) -> PyResult<Vec<PyChunk>> {
    release(py, || {
        let chunker = PySemanticChunker::build(
            chunk_size,
            threshold,
            threshold_mode,
            threshold_value,
            similarity_window,
            min_chunk_size,
            min_sentences_per_chunk,
            use_smoothing,
        );
        py_chunks(chunker.chunk(text))
    })
}

#[pyfunction]
#[pyo3(signature = (text, chunk_size=None, chunk_overlap=None, min_sentences_per_chunk=None))]
fn chunk_sentences(
    py: Python<'_>,
    text: &str,
    chunk_size: Option<usize>,
    chunk_overlap: Option<usize>,
    min_sentences_per_chunk: Option<usize>,
) -> PyResult<Vec<PyChunk>> {
    release(py, || {
        let chunker = PySentenceChunker::build(chunk_size, chunk_overlap, min_sentences_per_chunk);
        py_chunks(chunker.chunk(text))
    })
}

#[pyfunction]
#[pyo3(signature = (
    text,
    chunk_size=None,
    threshold=None,
    threshold_mode=None,
    threshold_value=None,
    merge_threshold=None,
    similarity_window=None,
    min_chunk_size=None
))]
fn chunk_sdpm(
    py: Python<'_>,
    text: &str,
    chunk_size: Option<usize>,
    threshold: Option<f32>,
    threshold_mode: Option<&str>,
    threshold_value: Option<f32>,
    merge_threshold: Option<f32>,
    similarity_window: Option<usize>,
    min_chunk_size: Option<usize>,
) -> PyResult<Vec<PyChunk>> {
    release(py, || {
        let chunker = PySDPMChunker::build(
            chunk_size,
            threshold,
            threshold_mode,
            threshold_value,
            merge_threshold,
            similarity_window,
            min_chunk_size,
        );
        py_chunks(chunker.chunk(text))
    })
}

#[pyfunction]
fn validate_tckn(s: &str) -> PyResult<bool> {
    guard(|| akana_core::pii::validate_tckn(s))
}

#[pyfunction]
fn validate_vkn(s: &str) -> PyResult<bool> {
    guard(|| akana_core::pii::validate_vkn(s))
}

#[pyfunction]
fn validate_iban(s: &str) -> PyResult<bool> {
    guard(|| akana_core::pii::validate_iban(s))
}

#[pyfunction]
fn validate_credit_card(s: &str) -> PyResult<Option<String>> {
    guard(|| akana_core::pii::validate_credit_card(s).map(|b| format!("{:?}", b)))
}

#[pyfunction]
fn validate_plate(s: &str) -> PyResult<bool> {
    guard(|| akana_core::pii::validate_plate(s))
}

#[pyfunction]
fn validate_vin(s: &str) -> PyResult<bool> {
    guard(|| akana_core::pii::validate_vin(s))
}

#[pyfunction]
fn validate_imei(s: &str) -> PyResult<bool> {
    guard(|| akana_core::pii::validate_imei(s))
}

#[pyfunction]
fn validate_ssn(s: &str) -> PyResult<bool> {
    guard(|| akana_core::pii::validate_ssn(s))
}

#[pyfunction]
fn harmonize_suffix(stem: &str, case: &str) -> PyResult<String> {
    let turkish_case = match case.to_lowercase().as_str() {
        "dative" | "dat" | "yonelme" => akana_core::pii::TurkishCase::Dative,
        "genitive" | "gen" | "ilgi" | "tamlayan" => akana_core::pii::TurkishCase::Genitive,
        "accusative" | "acc" | "belirtme" => akana_core::pii::TurkishCase::Accusative,
        "locative" | "loc" | "bulunma" => akana_core::pii::TurkishCase::Locative,
        "ablative" | "abl" | "ayrilma" => akana_core::pii::TurkishCase::Ablative,
        "instrumental" | "ins" | "vasita" => akana_core::pii::TurkishCase::Instrumental,
        "plural" | "plu" | "cogul" => akana_core::pii::TurkishCase::Plural,
        _ => return Err(PyValueError::new_err(format!("Unknown case: {}", case))),
    };
    guard(|| akana_core::pii::harmonize_suffix(stem, turkish_case))
}

/// Panics with `message`; the tests use it to check that a panic raises `RuntimeError`.
#[pyfunction]
fn _panic_for_tests(py: Python<'_>, message: &str) -> PyResult<()> {
    release(py, || panic!("{message}"))
}

/// The `{placeholder: original}` map of a plain or detailed mapping dict.
fn mapping_from_dict(
    mapping: &Bound<'_, PyDict>,
) -> PyResult<std::collections::HashMap<String, String>> {
    let mut map = std::collections::HashMap::new();
    for (k, v) in mapping.iter() {
        let key: String = k.extract()?;
        if let Ok(val_str) = v.extract::<String>() {
            map.insert(key, val_str);
        } else if let Ok(sub_dict) = v.downcast::<PyDict>() {
            if let Some(orig) = sub_dict.get_item("original_text")? {
                let val_str: String = orig.extract()?;
                map.insert(key, val_str);
            }
        }
    }
    Ok(map)
}

#[pyclass(name = "PiiVault")]
pub struct PyPiiVault {
    inner: akana_core::pii::PiiVault,
}

#[pymethods]
impl PyPiiVault {
    #[new]
    fn new() -> Self {
        Self {
            inner: akana_core::pii::PiiVault::new(),
        }
    }

    fn restore(&self, py: Python<'_>, llm_response: &str) -> PyResult<String> {
        release(py, || self.inner.restore(llm_response))
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(json_error)
    }

    #[staticmethod]
    fn from_json(json_str: &str) -> PyResult<Self> {
        let vault: akana_core::pii::PiiVault =
            serde_json::from_str(json_str).map_err(json_error)?;
        Ok(Self { inner: vault })
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new_bound(py);
        for (k, v) in &self.inner.placeholder_to_entry {
            let entry_dict = PyDict::new_bound(py);
            entry_dict.set_item("placeholder", &v.placeholder)?;
            entry_dict.set_item("original_text", &v.original_text)?;
            entry_dict.set_item("pii_label", &v.pii_label)?;
            entry_dict.set_item("original_stem", &v.original_stem)?;
            entry_dict.set_item("original_suffix", &v.original_suffix)?;
            dict.set_item(k, entry_dict)?;
        }
        Ok(dict)
    }

    fn get_mapping<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new_bound(py);
        for (k, v) in &self.inner.placeholder_to_entry {
            dict.set_item(k, &v.original_text)?;
        }
        Ok(dict)
    }

    #[staticmethod]
    fn restore_with_mapping(
        py: Python<'_>,
        llm_response: &str,
        mapping: &Bound<'_, PyDict>,
    ) -> PyResult<String> {
        let map = mapping_from_dict(mapping)?;
        release(py, || {
            akana_core::pii::PiiVault::restore_with_mapping(llm_response, &map)
        })
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }
}

#[pyclass(name = "TurkishPiiEngine")]
pub struct PyTurkishPiiEngine {
    inner: akana_core::pii::TurkishPiiEngine,
}

#[pymethods]
impl PyTurkishPiiEngine {
    #[new]
    #[pyo3(signature = (use_embeddings=false, preserve_corporate_emails=true, disabled_types=None, preserve_public_figures=false))]
    fn new(
        py: Python<'_>,
        use_embeddings: bool,
        preserve_corporate_emails: bool,
        disabled_types: Option<Vec<String>>,
        preserve_public_figures: bool,
    ) -> PyResult<Self> {
        let mut engine = release(py, || {
            if use_embeddings {
                PII_ENGINE_WITH_EMBEDDINGS.clone()
            } else {
                akana_core::pii::TurkishPiiEngine::new()
            }
        })?;
        engine.set_preserve_corporate_emails(preserve_corporate_emails);
        engine.set_preserve_public_figures(preserve_public_figures);
        if let Some(types) = disabled_types {
            for t in types {
                if let Some(pii_type) = akana_core::pii::PiiType::from_name(&t) {
                    engine.disable_type(pii_type);
                }
            }
        }
        Ok(Self { inner: engine })
    }

    fn disable_type(&mut self, pii_type: &str) -> PyResult<()> {
        if let Some(pt) = akana_core::pii::PiiType::from_name(pii_type) {
            self.inner.disable_type(pt);
            Ok(())
        } else {
            Err(PyValueError::new_err(format!(
                "Unknown PiiType: {}",
                pii_type
            )))
        }
    }

    fn enable_type(&mut self, pii_type: &str) -> PyResult<()> {
        if let Some(pt) = akana_core::pii::PiiType::from_name(pii_type) {
            self.inner.enable_type(pt);
            Ok(())
        } else {
            Err(PyValueError::new_err(format!(
                "Unknown PiiType: {}",
                pii_type
            )))
        }
    }

    fn set_type_enabled(&mut self, pii_type: &str, enabled: bool) -> PyResult<()> {
        if let Some(pt) = akana_core::pii::PiiType::from_name(pii_type) {
            self.inner.set_type_enabled(pt, enabled);
            Ok(())
        } else {
            Err(PyValueError::new_err(format!(
                "Unknown PiiType: {}",
                pii_type
            )))
        }
    }

    fn is_type_enabled(&self, pii_type: &str) -> bool {
        if let Some(pt) = akana_core::pii::PiiType::from_name(pii_type) {
            self.inner.is_type_enabled(pt)
        } else {
            false
        }
    }

    fn detect<'py>(&self, py: Python<'py>, text: &str) -> PyResult<Bound<'py, PyList>> {
        let (entities, offsets) = release(py, || {
            let entities = self.inner.detect(text);
            let offsets = entity_char_offsets(text, &entities);
            (entities, offsets)
        })?;
        let list = PyList::empty_bound(py);
        for (e, (char_start, char_end)) in entities.into_iter().zip(offsets) {
            let dict = PyDict::new_bound(py);
            dict.set_item("text", e.text)?;
            dict.set_item("label", e.label)?;
            dict.set_item("pii_type", e.pii_type.as_str())?;
            dict.set_item("start", char_start)?;
            dict.set_item("end", char_end)?;
            dict.set_item("confidence", e.confidence)?;
            dict.set_item("stem", e.stem)?;
            dict.set_item("suffix", e.suffix)?;
            list.append(dict)?;
        }
        Ok(list)
    }

    #[pyo3(signature = (text, mode="placeholder", vault=None))]
    fn mask<'py>(
        &self,
        py: Python<'py>,
        text: &str,
        mode: &str,
        vault: Option<PyRefMut<'py, PyPiiVault>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let pii_mode = match mode.to_lowercase().as_str() {
            "placeholder" => akana_core::pii::PiiMode::Placeholder,
            "tag" => akana_core::pii::PiiMode::Tag,
            "anonymize" => akana_core::pii::PiiMode::Anonymize,
            "surrogate" | "syntheticsurrogate" => akana_core::pii::PiiMode::SyntheticSurrogate,
            _ => return Err(PyValueError::new_err(format!("Invalid mode: {}", mode))),
        };

        let mut vault = vault;
        let vault_inner = vault.as_mut().map(|v| &mut v.inner);
        let (result, offsets) = release(py, || {
            let result = match vault_inner {
                Some(v) => self.inner.mask_with_vault(text, pii_mode, v),
                None => self.inner.mask(text, pii_mode),
            };
            let offsets = entity_char_offsets(&result.original_text, &result.entities);
            (result, offsets)
        })?;
        let dict = PyDict::new_bound(py);
        dict.set_item("original_text", &result.original_text)?;
        dict.set_item("masked_text", &result.masked_text)?;

        // Plain string mapping: { placeholder_or_surrogate: original_text }
        let mapping_dict = PyDict::new_bound(py);
        for (k, v) in &result.mapping {
            mapping_dict.set_item(k, v)?;
        }
        dict.set_item("mapping", mapping_dict)?;

        // Detailed mapping: { placeholder: { original_text, pii_label, stem, suffix } }
        let detailed_dict = PyDict::new_bound(py);
        for (k, v) in &result.vault.placeholder_to_entry {
            let entry_dict = PyDict::new_bound(py);
            entry_dict.set_item("placeholder", &v.placeholder)?;
            entry_dict.set_item("original_text", &v.original_text)?;
            entry_dict.set_item("pii_label", &v.pii_label)?;
            entry_dict.set_item("original_stem", &v.original_stem)?;
            entry_dict.set_item("original_suffix", &v.original_suffix)?;
            detailed_dict.set_item(k, entry_dict)?;
        }
        dict.set_item("detailed_mapping", detailed_dict)?;

        let ent_list = PyList::empty_bound(py);
        for (e, (char_start, char_end)) in result.entities.into_iter().zip(offsets) {
            let ent_dict = PyDict::new_bound(py);
            ent_dict.set_item("text", e.text)?;
            ent_dict.set_item("label", e.label)?;
            ent_dict.set_item("pii_type", e.pii_type.as_str())?;
            ent_dict.set_item("start", char_start)?;
            ent_dict.set_item("end", char_end)?;
            ent_dict.set_item("confidence", e.confidence)?;
            ent_list.append(ent_dict)?;
        }
        dict.set_item("entities", ent_list)?;
        dict.set_item(
            "vault",
            Py::new(
                py,
                PyPiiVault {
                    inner: result.vault,
                },
            )?,
        )?;

        Ok(dict)
    }

    /// Intercepts outbound prompt and masks text using an existing PiiVault session instance.
    #[pyo3(signature = (text, vault, mode="placeholder"))]
    fn mask_with_vault<'py>(
        &self,
        py: Python<'py>,
        text: &str,
        vault: PyRefMut<'py, PyPiiVault>,
        mode: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        self.mask(py, text, mode, Some(vault))
    }

    /// Flexible restore function: restores LLM response using either a PiiVault instance,
    /// a plain dictionary mapping (e.g. `{"{{AD_1}}": "Ahmet"}`), a detailed mapping, or a JSON string.
    #[pyo3(signature = (llm_response, mapping_or_vault))]
    fn restore(
        &self,
        py: Python<'_>,
        llm_response: &str,
        mapping_or_vault: &Bound<'_, PyAny>,
    ) -> PyResult<String> {
        if let Ok(vault) = mapping_or_vault.extract::<PyRef<PyPiiVault>>() {
            let vault = &vault.inner;
            release(py, || self.inner.restore_response(llm_response, vault))
        } else if let Ok(dict) = mapping_or_vault.downcast::<PyDict>() {
            let map = mapping_from_dict(dict)?;
            release(py, || {
                akana_core::pii::PiiVault::restore_with_mapping(llm_response, &map)
            })
        } else if let Ok(json_str) = mapping_or_vault.extract::<&str>() {
            release(py, || {
                if let Ok(vault) = serde_json::from_str::<akana_core::pii::PiiVault>(json_str) {
                    Some(self.inner.restore_response(llm_response, &vault))
                } else if let Ok(map) =
                    serde_json::from_str::<std::collections::HashMap<String, String>>(json_str)
                {
                    Some(akana_core::pii::PiiVault::restore_with_mapping(
                        llm_response,
                        &map,
                    ))
                } else {
                    None
                }
            })?
            .ok_or_else(|| PyValueError::new_err("Invalid JSON format for mapping or vault"))
        } else {
            Err(PyTypeError::new_err(
                "Expected PiiVault, dict mapping, or JSON string for restoration",
            ))
        }
    }

    fn restore_response(
        &self,
        py: Python<'_>,
        llm_response: &str,
        vault: &PyPiiVault,
    ) -> PyResult<String> {
        release(py, || {
            self.inner.restore_response(llm_response, &vault.inner)
        })
    }
}

#[pymodule]
fn _core(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(to_turkish_lower, m)?)?;
    m.add_function(wrap_pyfunction!(to_turkish_upper, m)?)?;
    m.add_function(wrap_pyfunction!(to_turkish_title, m)?)?;
    m.add_function(wrap_pyfunction!(check_major_vowel_harmony, m)?)?;
    m.add_function(wrap_pyfunction!(check_minor_vowel_harmony, m)?)?;
    m.add_function(wrap_pyfunction!(asciify, m)?)?;
    m.add_function(wrap_pyfunction!(deasciify, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_informal, m)?)?;
    m.add_function(wrap_pyfunction!(tokenize_words, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_document_json, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_readability_json, m)?)?;
    m.add_function(wrap_pyfunction!(syllabify, m)?)?;
    m.add_function(wrap_pyfunction!(hyphenate, m)?)?;
    m.add_function(wrap_pyfunction!(count_syllables, m)?)?;
    m.add_function(wrap_pyfunction!(number_to_words, m)?)?;
    m.add_function(wrap_pyfunction!(ordinal_to_words, m)?)?;
    m.add_function(wrap_pyfunction!(currency_to_words, m)?)?;
    m.add_function(wrap_pyfunction!(words_to_number, m)?)?;
    m.add_function(wrap_pyfunction!(stem, m)?)?;
    m.add_function(wrap_pyfunction!(is_stopword, m)?)?;
    m.add_function(wrap_pyfunction!(remove_stopwords, m)?)?;
    m.add_function(wrap_pyfunction!(extract_entities_json, m)?)?;
    m.add_function(wrap_pyfunction!(extract_keywords_json, m)?)?;
    m.add_function(wrap_pyfunction!(summarize, m)?)?;
    m.add_function(wrap_pyfunction!(audit_ai_style_json, m)?)?;
    m.add_function(wrap_pyfunction!(generate_humanizer_prompt, m)?)?;
    m.add_function(wrap_pyfunction!(check_grammar_json, m)?)?;
    m.add_function(wrap_pyfunction!(correct_grammar, m)?)?;
    m.add_function(wrap_pyfunction!(grammar_rules_json, m)?)?;
    m.add_function(wrap_pyfunction!(embed, m)?)?;
    m.add_function(wrap_pyfunction!(embed_packed_2bit, m)?)?;
    m.add_function(wrap_pyfunction!(embed_packed_1bit, m)?)?;
    m.add_function(wrap_pyfunction!(embed_batch_packed_2bit, m)?)?;
    m.add_function(wrap_pyfunction!(embed_batch_packed_1bit, m)?)?;
    m.add_function(wrap_pyfunction!(sparse_embed, m)?)?;
    m.add_function(wrap_pyfunction!(hybrid_embed, m)?)?;
    m.add_function(wrap_pyfunction!(embed_hybrid, m)?)?;
    m.add_function(wrap_pyfunction!(embed_batch_hybrid, m)?)?;
    m.add_function(wrap_pyfunction!(get_usearch_metric_pointer_2bit, m)?)?;
    m.add_function(wrap_pyfunction!(get_usearch_metric_pointer_1bit, m)?)?;
    m.add_function(wrap_pyfunction!(get_usearch_metric_pointer_hybrid, m)?)?;
    m.add_function(wrap_pyfunction!(set_hybrid_metric_alpha, m)?)?;
    m.add_function(wrap_pyfunction!(get_hybrid_metric_alpha, m)?)?;
    m.add_function(wrap_pyfunction!(reciprocal_rank_fusion, m)?)?;
    m.add("EMBEDDING_SCALE_2BIT", akana_core::EMBEDDING_SCALE_2BIT)?;
    m.add("HYBRID_VECTOR_BYTES", akana_core::HYBRID_VECTOR_BYTES)?;
    m.add("PACKED_2BIT_BYTES", akana_core::PACKED_2BIT_BYTES)?;
    m.add("PACKED_1BIT_BYTES", akana_core::PACKED_1BIT_BYTES)?;
    m.add_function(wrap_pyfunction!(tokenize_embedding_text, m)?)?;
    m.add_function(wrap_pyfunction!(embed_batch, m)?)?;
    m.add_function(wrap_pyfunction!(similarity, m)?)?;
    m.add_function(wrap_pyfunction!(similarity_vectors, m)?)?;
    m.add_function(wrap_pyfunction!(chunk_semantic, m)?)?;
    m.add_function(wrap_pyfunction!(chunk_sentences, m)?)?;
    m.add_function(wrap_pyfunction!(chunk_sdpm, m)?)?;
    m.add_function(wrap_pyfunction!(validate_tckn, m)?)?;
    m.add_function(wrap_pyfunction!(validate_vkn, m)?)?;
    m.add_function(wrap_pyfunction!(validate_iban, m)?)?;
    m.add_function(wrap_pyfunction!(validate_credit_card, m)?)?;
    m.add_function(wrap_pyfunction!(validate_plate, m)?)?;
    m.add_function(wrap_pyfunction!(validate_vin, m)?)?;
    m.add_function(wrap_pyfunction!(validate_imei, m)?)?;
    m.add_function(wrap_pyfunction!(validate_ssn, m)?)?;
    m.add_function(wrap_pyfunction!(harmonize_suffix, m)?)?;
    m.add_function(wrap_pyfunction!(_panic_for_tests, m)?)?;
    m.add_class::<PySpellChecker>()?;
    m.add_class::<PyMorphology>()?;
    m.add_class::<PySyntacticMorphology>()?;
    m.add_class::<PyCompoundDecomposer>()?;
    m.add_class::<PyDisambiguator>()?;
    m.add_class::<PyDependencyParser>()?;
    m.add_class::<PyGrammarChecker>()?;
    m.add_class::<PyEmbeddings>()?;
    m.add_class::<PyChunk>()?;
    m.add_class::<PySemanticChunker>()?;
    m.add_class::<PySentenceChunker>()?;
    m.add_class::<PySDPMChunker>()?;
    m.add_class::<PyPiiVault>()?;
    m.add_class::<PyTurkishPiiEngine>()?;
    Ok(())
}
