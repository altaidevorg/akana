//! PyO3 Python bindings for the Akana Turkish NLP toolkit.
#![allow(clippy::useless_conversion)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::redundant_closure)]

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

#[pyfunction]
fn to_turkish_lower(text: &str) -> String {
    akana_core::phonology::to_turkish_lower(text)
}

#[pyfunction]
fn to_turkish_upper(text: &str) -> String {
    akana_core::phonology::to_turkish_upper(text)
}

#[pyfunction]
fn to_turkish_title(text: &str) -> String {
    akana_core::phonology::to_turkish_title(text)
}

#[pyfunction]
fn check_major_vowel_harmony(word: &str) -> bool {
    akana_core::phonology::check_major_vowel_harmony(word)
}

#[pyfunction]
fn check_minor_vowel_harmony(word: &str) -> bool {
    akana_core::phonology::check_minor_vowel_harmony(word)
}

#[pyfunction]
fn asciify(text: &str) -> String {
    akana_core::normalization::TurkishAsciifier::asciify(text)
}

#[pyfunction]
fn deasciify(text: &str) -> String {
    akana_core::normalization::TurkishDeasciifier::deasciify(text)
}

#[pyfunction]
fn normalize_informal(text: &str) -> String {
    akana_core::normalization::TurkishInformalNormalizer::normalize_text(text)
}

#[pyfunction]
fn tokenize_words(text: &str) -> Vec<String> {
    akana_core::tokenization::TurkishTokenizer::tokenize_words(text)
        .into_iter()
        .map(|s| s.to_string())
        .collect()
}

#[pyclass(name = "SpellChecker")]
struct PySpellChecker {
    inner: akana_core::normalization::TurkishSpellChecker,
}

#[pymethods]
impl PySpellChecker {
    #[new]
    fn new() -> Self {
        Self {
            inner: akana_core::normalization::TurkishSpellChecker::new(),
        }
    }

    fn add_word(&mut self, word: &str) {
        self.inner.add_word(word);
    }

    fn is_correct(&self, word: &str) -> bool {
        self.inner.is_correct(word)
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
        let suggestions = self.inner.suggest(word, dist, count);

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
    fn new() -> Self {
        Self {
            inner: akana_core::morphology::TurkishMorphology::new(),
        }
    }

    fn analyze<'py>(&self, py: Python<'py>, word: &str) -> PyResult<Bound<'py, PyList>> {
        let parses = self.inner.analyze(word);
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
            dict.set_item("score", p.score)?;
            list.append(dict)?;
        }
        Ok(list)
    }

    fn generate(&self, lemma: &str, tags: Vec<String>) -> Option<String> {
        let generator = akana_core::morphology::TurkishGenerator::new();
        let tag_refs: Vec<&str> = tags.iter().map(|s| s.as_str()).collect();
        generator.generate(lemma, &tag_refs)
    }

    fn load_dictionary_file(&mut self, path: &str) -> PyResult<()> {
        self.inner
            .load_dictionary_file(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    fn load_dictionary_str(&mut self, text: &str) {
        self.inner.load_dictionary_str(text);
    }
}

#[pyclass(name = "SyntacticMorphology")]
struct PySyntacticMorphology {
    inner: akana_core::syntactic_morphology::TurkishSyntacticMorphology,
}

#[pymethods]
impl PySyntacticMorphology {
    #[new]
    fn new() -> Self {
        Self {
            inner: akana_core::syntactic_morphology::TurkishSyntacticMorphology::new(),
        }
    }

    fn analyze<'py>(&self, py: Python<'py>, word: &str) -> PyResult<Bound<'py, PyList>> {
        let parses = self.inner.analyze(word);
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
    fn new() -> Self {
        Self {
            inner: akana_core::morphology::CompoundDecomposer::new(),
        }
    }

    fn decompose<'py>(&self, py: Python<'py>, word: &str) -> PyResult<Bound<'py, PyList>> {
        let analyses = self.inner.decompose(word);
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
    fn new() -> Self {
        Self {
            inner: akana_core::morphology::MorphologicalDisambiguator::new(),
        }
    }

    fn disambiguate<'py>(
        &self,
        py: Python<'py>,
        tokens: Vec<String>,
    ) -> PyResult<Bound<'py, PyList>> {
        let token_refs: Vec<&str> = tokens.iter().map(|s| s.as_str()).collect();
        let parses = self.inner.disambiguate(&token_refs);
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
    fn new() -> Self {
        Self {
            inner: akana_core::parser::TurkishDependencyParser::new(),
        }
    }

    fn parse_conllu(&self, tokens: Vec<String>) -> String {
        let token_refs: Vec<&str> = tokens.iter().map(|s| s.as_str()).collect();
        let tree = self.inner.parse(&token_refs);
        tree.to_conllu()
    }
}

#[pyfunction]
fn analyze_document_json(text: &str) -> PyResult<String> {
    let engine = akana_core::AkanaEngine::new();
    let doc = engine.analyze_document(text);
    serde_json::to_string(&doc).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn analyze_readability_json(text: &str) -> PyResult<String> {
    let report = akana_core::readability::analyze_readability(text);
    serde_json::to_string(&report)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn syllabify(word: &str) -> Vec<String> {
    akana_core::phonology::TurkishSyllabifier::syllabify(word)
}

#[pyfunction]
#[pyo3(signature = (word, delimiter=None))]
fn hyphenate(word: &str, delimiter: Option<&str>) -> String {
    let delim = delimiter.unwrap_or("-");
    akana_core::phonology::TurkishSyllabifier::hyphenate(word, delim)
}

#[pyfunction]
fn count_syllables(word: &str) -> usize {
    akana_core::phonology::TurkishSyllabifier::count_syllables(word)
}

#[pyfunction]
fn number_to_words(n: i64) -> String {
    akana_core::normalization::TurkishNumberConverter::number_to_words(n)
}

#[pyfunction]
fn ordinal_to_words(n: i64) -> String {
    akana_core::normalization::TurkishNumberConverter::ordinal_to_words(n)
}

#[pyfunction]
#[pyo3(signature = (amount, currency=None))]
fn currency_to_words(amount: f64, currency: Option<&str>) -> String {
    let curr = currency.unwrap_or("TL");
    akana_core::normalization::TurkishNumberConverter::currency_to_words(amount, curr)
}

#[pyfunction]
fn words_to_number(text: &str) -> PyResult<i64> {
    akana_core::normalization::TurkishNumberConverter::words_to_number(text)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
}

#[pyfunction]
fn stem(word: &str) -> String {
    let stemmer = akana_core::morphology::TurkishStemmer::new();
    stemmer.stem(word)
}

#[pyfunction]
fn is_stopword(word: &str) -> bool {
    let sw = akana_core::morphology::TurkishStopwords::new();
    sw.is_stopword(word)
}

#[pyfunction]
fn remove_stopwords(tokens: Vec<String>) -> Vec<String> {
    let sw = akana_core::morphology::TurkishStopwords::new();
    let token_refs: Vec<&str> = tokens.iter().map(|s| s.as_str()).collect();
    sw.filter_tokens(&token_refs)
        .into_iter()
        .map(|s| s.to_string())
        .collect()
}

#[pyfunction]
fn extract_entities_json(text: &str) -> PyResult<String> {
    let entities = akana_core::ner::TurkishNER::extract_entities(text);
    serde_json::to_string(&entities)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (text, top_k=None))]
fn extract_keywords_json(text: &str, top_k: Option<usize>) -> PyResult<String> {
    let extractor = akana_core::analysis::TurkishKeywordExtractor::new();
    let keywords = extractor.extract_keywords(text, top_k.unwrap_or(10));
    serde_json::to_string(&keywords)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (text, max_sentences=None))]
fn summarize(text: &str, max_sentences: Option<usize>) -> Vec<String> {
    let summarizer = akana_core::analysis::TurkishSummarizer::new();
    summarizer.summarize(text, max_sentences.unwrap_or(3))
}

#[pyfunction]
fn audit_ai_style_json(text: &str) -> PyResult<String> {
    let auditor = akana_core::style::TurkishStyleAuditor::new();
    let report = auditor.audit(text);
    serde_json::to_string(&report)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (text, register=None))]
fn generate_humanizer_prompt(text: &str, register: Option<&str>) -> String {
    let reg = register.unwrap_or("blog");
    akana_core::style::TurkishHumanizer::generate_prompt(text, reg)
}

#[pyclass(name = "GrammarChecker")]
struct PyGrammarChecker {
    inner: akana_core::grammar::TurkishGrammarChecker,
}

#[pymethods]
impl PyGrammarChecker {
    #[new]
    fn new() -> Self {
        Self {
            inner: akana_core::grammar::TurkishGrammarChecker::new(),
        }
    }

    fn check_json(&self, text: &str) -> PyResult<String> {
        let res = self.inner.check(text);
        serde_json::to_string(&res)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn correct(&self, text: &str) -> String {
        self.inner.correct(text)
    }
}

lazy_static::lazy_static! {
    static ref GLOBAL_GRAMMAR_CHECKER: akana_core::grammar::TurkishGrammarChecker = akana_core::grammar::TurkishGrammarChecker::new();
    static ref GLOBAL_EMBEDDINGS: akana_core::TurkishEmbeddings = akana_core::TurkishEmbeddings::new();
}

#[pyfunction]
fn check_grammar_json(text: &str) -> PyResult<String> {
    let res = GLOBAL_GRAMMAR_CHECKER.check(text);
    serde_json::to_string(&res).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn correct_grammar(text: &str) -> String {
    GLOBAL_GRAMMAR_CHECKER.correct(text)
}

#[pyclass(name = "Embeddings")]
struct PyEmbeddings {
    inner: akana_core::TurkishEmbeddings,
}

#[pymethods]
impl PyEmbeddings {
    #[new]
    fn new() -> Self {
        Self {
            inner: akana_core::TurkishEmbeddings::new(),
        }
    }

    fn embed(&self, text: &str) -> Vec<f32> {
        self.inner.embed(text)
    }

    fn tokenize(&self, text: &str) -> Vec<usize> {
        self.inner.tokenize(text)
    }

    fn embed_batch(&self, texts: Vec<String>) -> Vec<Vec<f32>> {
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        self.inner.embed_batch(&text_refs)
    }

    fn similarity(&self, text_a: &str, text_b: &str) -> f32 {
        self.inner.similarity(text_a, text_b)
    }

    #[staticmethod]
    fn similarity_vectors(vec_a: Vec<f32>, vec_b: Vec<f32>) -> f32 {
        akana_core::cosine_similarity(&vec_a, &vec_b)
    }

    #[getter]
    fn dimension(&self) -> usize {
        self.inner.embedding_dim()
    }
}

#[pyfunction]
fn embed(text: &str) -> Vec<f32> {
    GLOBAL_EMBEDDINGS.embed(text)
}

#[pyfunction]
fn tokenize_embedding_text(text: &str) -> Vec<usize> {
    GLOBAL_EMBEDDINGS.tokenize(text)
}

#[pyfunction]
fn embed_batch(texts: Vec<String>) -> Vec<Vec<f32>> {
    let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
    GLOBAL_EMBEDDINGS.embed_batch(&text_refs)
}

#[pyfunction]
fn similarity(text_a: &str, text_b: &str) -> f32 {
    GLOBAL_EMBEDDINGS.similarity(text_a, text_b)
}

#[pyfunction]
fn similarity_vectors(vec_a: Vec<f32>, vec_b: Vec<f32>) -> f32 {
    akana_core::cosine_similarity(&vec_a, &vec_b)
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

#[pyclass(name = "SemanticChunker")]
struct PySemanticChunker {
    inner: akana_core::chunking::SemanticChunker,
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
        chunk_size: Option<usize>,
        threshold: Option<f32>,
        threshold_mode: Option<&str>,
        threshold_value: Option<f32>,
        similarity_window: Option<usize>,
        min_chunk_size: Option<usize>,
        min_sentences_per_chunk: Option<usize>,
        use_smoothing: Option<bool>,
    ) -> Self {
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

        Self { inner: chunker }
    }

    fn chunk(&self, text: &str) -> Vec<PyChunk> {
        self.inner
            .chunk(text)
            .into_iter()
            .map(|c| PyChunk { inner: c })
            .collect()
    }

    fn chunk_batch(&self, texts: Vec<String>) -> Vec<Vec<PyChunk>> {
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        self.inner
            .chunk_batch(&text_refs)
            .into_iter()
            .map(|batch| batch.into_iter().map(|c| PyChunk { inner: c }).collect())
            .collect()
    }

    fn __call__(&self, text: &str) -> Vec<PyChunk> {
        self.chunk(text)
    }
}

#[pyclass(name = "SentenceChunker")]
struct PySentenceChunker {
    inner: akana_core::chunking::SentenceChunker,
}

#[pymethods]
impl PySentenceChunker {
    #[new]
    #[pyo3(signature = (chunk_size=None, chunk_overlap=None, min_sentences_per_chunk=None))]
    fn new(
        chunk_size: Option<usize>,
        chunk_overlap: Option<usize>,
        min_sentences_per_chunk: Option<usize>,
    ) -> Self {
        let size = chunk_size.unwrap_or(512);
        let overlap = chunk_overlap.unwrap_or(0);
        let min_sent = min_sentences_per_chunk.unwrap_or(1);
        Self {
            inner: akana_core::chunking::SentenceChunker::new(size, overlap, min_sent),
        }
    }

    fn chunk(&self, text: &str) -> Vec<PyChunk> {
        self.inner
            .chunk(text)
            .into_iter()
            .map(|c| PyChunk { inner: c })
            .collect()
    }

    fn chunk_batch(&self, texts: Vec<String>) -> Vec<Vec<PyChunk>> {
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        self.inner
            .chunk_batch(&text_refs)
            .into_iter()
            .map(|batch| batch.into_iter().map(|c| PyChunk { inner: c }).collect())
            .collect()
    }

    fn __call__(&self, text: &str) -> Vec<PyChunk> {
        self.chunk(text)
    }
}

#[pyclass(name = "SDPMChunker")]
struct PySDPMChunker {
    inner: akana_core::chunking::SDPMChunker,
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
        chunk_size: Option<usize>,
        threshold: Option<f32>,
        threshold_mode: Option<&str>,
        threshold_value: Option<f32>,
        merge_threshold: Option<f32>,
        similarity_window: Option<usize>,
        min_chunk_size: Option<usize>,
    ) -> Self {
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

        Self { inner: chunker }
    }

    fn chunk(&self, text: &str) -> Vec<PyChunk> {
        self.inner
            .chunk(text)
            .into_iter()
            .map(|c| PyChunk { inner: c })
            .collect()
    }

    fn chunk_batch(&self, texts: Vec<String>) -> Vec<Vec<PyChunk>> {
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        self.inner
            .chunk_batch(&text_refs)
            .into_iter()
            .map(|batch| batch.into_iter().map(|c| PyChunk { inner: c }).collect())
            .collect()
    }

    fn __call__(&self, text: &str) -> Vec<PyChunk> {
        self.chunk(text)
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
    text: &str,
    chunk_size: Option<usize>,
    threshold: Option<f32>,
    threshold_mode: Option<&str>,
    threshold_value: Option<f32>,
    similarity_window: Option<usize>,
    min_chunk_size: Option<usize>,
    min_sentences_per_chunk: Option<usize>,
    use_smoothing: Option<bool>,
) -> Vec<PyChunk> {
    let chunker = PySemanticChunker::new(
        chunk_size,
        threshold,
        threshold_mode,
        threshold_value,
        similarity_window,
        min_chunk_size,
        min_sentences_per_chunk,
        use_smoothing,
    );
    chunker.chunk(text)
}

#[pyfunction]
#[pyo3(signature = (text, chunk_size=None, chunk_overlap=None, min_sentences_per_chunk=None))]
fn chunk_sentences(
    text: &str,
    chunk_size: Option<usize>,
    chunk_overlap: Option<usize>,
    min_sentences_per_chunk: Option<usize>,
) -> Vec<PyChunk> {
    let chunker = PySentenceChunker::new(chunk_size, chunk_overlap, min_sentences_per_chunk);
    chunker.chunk(text)
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
    text: &str,
    chunk_size: Option<usize>,
    threshold: Option<f32>,
    threshold_mode: Option<&str>,
    threshold_value: Option<f32>,
    merge_threshold: Option<f32>,
    similarity_window: Option<usize>,
    min_chunk_size: Option<usize>,
) -> Vec<PyChunk> {
    let chunker = PySDPMChunker::new(
        chunk_size,
        threshold,
        threshold_mode,
        threshold_value,
        merge_threshold,
        similarity_window,
        min_chunk_size,
    );
    chunker.chunk(text)
}

#[pyfunction]
fn validate_tckn(s: &str) -> bool {
    akana_core::pii::validate_tckn(s)
}

#[pyfunction]
fn validate_vkn(s: &str) -> bool {
    akana_core::pii::validate_vkn(s)
}

#[pyfunction]
fn validate_iban(s: &str) -> bool {
    akana_core::pii::validate_iban(s)
}

#[pyfunction]
fn validate_credit_card(s: &str) -> Option<String> {
    akana_core::pii::validate_credit_card(s).map(|b| format!("{:?}", b))
}

#[pyfunction]
fn validate_plate(s: &str) -> bool {
    akana_core::pii::validate_plate(s)
}

#[pyfunction]
fn validate_vin(s: &str) -> bool {
    akana_core::pii::validate_vin(s)
}

#[pyfunction]
fn validate_imei(s: &str) -> bool {
    akana_core::pii::validate_imei(s)
}

#[pyfunction]
fn validate_ssn(s: &str) -> bool {
    akana_core::pii::validate_ssn(s)
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
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown case: {}",
                case
            )))
        }
    };
    Ok(akana_core::pii::harmonize_suffix(stem, turkish_case))
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

    fn restore(&self, llm_response: &str) -> String {
        self.inner.restore(llm_response)
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    #[staticmethod]
    fn from_json(json_str: &str) -> PyResult<Self> {
        let vault: akana_core::pii::PiiVault = serde_json::from_str(json_str)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
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
    fn restore_with_mapping(llm_response: &str, mapping: &Bound<'_, PyDict>) -> PyResult<String> {
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
        Ok(akana_core::pii::PiiVault::restore_with_mapping(
            llm_response,
            &map,
        ))
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
    #[pyo3(signature = (use_embeddings=false, preserve_corporate_emails=true, disabled_types=None))]
    fn new(
        use_embeddings: bool,
        preserve_corporate_emails: bool,
        disabled_types: Option<Vec<String>>,
    ) -> Self {
        let mut engine = if use_embeddings {
            let emb = std::sync::Arc::new(akana_core::embeddings::TurkishEmbeddings::new());
            akana_core::pii::TurkishPiiEngine::with_embeddings(emb)
        } else {
            akana_core::pii::TurkishPiiEngine::new()
        };
        engine.set_preserve_corporate_emails(preserve_corporate_emails);
        if let Some(types) = disabled_types {
            for t in types {
                if let Some(pii_type) = akana_core::pii::PiiType::from_name(&t) {
                    engine.disable_type(pii_type);
                }
            }
        }
        Self { inner: engine }
    }

    fn disable_type(&mut self, pii_type: &str) -> PyResult<()> {
        if let Some(pt) = akana_core::pii::PiiType::from_name(pii_type) {
            self.inner.disable_type(pt);
            Ok(())
        } else {
            Err(pyo3::exceptions::PyValueError::new_err(format!(
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
            Err(pyo3::exceptions::PyValueError::new_err(format!(
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
            Err(pyo3::exceptions::PyValueError::new_err(format!(
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
        let entities = self.inner.detect(text);
        let list = PyList::empty_bound(py);
        for e in entities {
            let dict = PyDict::new_bound(py);
            dict.set_item("text", e.text)?;
            dict.set_item("label", e.label)?;
            dict.set_item("pii_type", e.pii_type.as_str())?;
            let char_start = text
                .get(..e.start)
                .map(|s| s.chars().count())
                .unwrap_or(e.start);
            let char_end = text
                .get(..e.end)
                .map(|s| s.chars().count())
                .unwrap_or(e.end);
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
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid mode: {}",
                    mode
                )))
            }
        };

        let result = if let Some(mut v) = vault {
            self.inner.mask_with_vault(text, pii_mode, &mut v.inner)
        } else {
            self.inner.mask(text, pii_mode)
        };
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

        let orig_text = &result.original_text;
        let ent_list = PyList::empty_bound(py);
        for e in result.entities {
            let ent_dict = PyDict::new_bound(py);
            ent_dict.set_item("text", e.text)?;
            ent_dict.set_item("label", e.label)?;
            ent_dict.set_item("pii_type", e.pii_type.as_str())?;
            let char_start = orig_text
                .get(..e.start)
                .map(|s| s.chars().count())
                .unwrap_or(e.start);
            let char_end = orig_text
                .get(..e.end)
                .map(|s| s.chars().count())
                .unwrap_or(e.end);
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
        _py: Python<'_>,
        llm_response: &str,
        mapping_or_vault: &Bound<'_, PyAny>,
    ) -> PyResult<String> {
        if let Ok(vault) = mapping_or_vault.extract::<PyRef<PyPiiVault>>() {
            Ok(self.inner.restore_response(llm_response, &vault.inner))
        } else if let Ok(dict) = mapping_or_vault.downcast::<PyDict>() {
            let mut map = std::collections::HashMap::new();
            for (k, v) in dict.iter() {
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
            Ok(akana_core::pii::PiiVault::restore_with_mapping(
                llm_response,
                &map,
            ))
        } else if let Ok(json_str) = mapping_or_vault.extract::<&str>() {
            if let Ok(vault) = serde_json::from_str::<akana_core::pii::PiiVault>(json_str) {
                Ok(self.inner.restore_response(llm_response, &vault))
            } else if let Ok(map) =
                serde_json::from_str::<std::collections::HashMap<String, String>>(json_str)
            {
                Ok(akana_core::pii::PiiVault::restore_with_mapping(
                    llm_response,
                    &map,
                ))
            } else {
                Err(pyo3::exceptions::PyValueError::new_err(
                    "Invalid JSON format for mapping or vault",
                ))
            }
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "Expected PiiVault, dict mapping, or JSON string for restoration",
            ))
        }
    }

    fn restore_response(&self, llm_response: &str, vault: &PyPiiVault) -> String {
        self.inner.restore_response(llm_response, &vault.inner)
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
    m.add_function(wrap_pyfunction!(embed, m)?)?;
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
