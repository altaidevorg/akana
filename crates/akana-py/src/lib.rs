//! PyO3 Python bindings for the Akana Turkish NLP toolkit.

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
        self.inner.load_dictionary_file(path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(e.to_string())
        })
    }

    fn load_dictionary_str(&mut self, text: &str) {
        self.inner.load_dictionary_str(text);
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

    fn disambiguate<'py>(&self, py: Python<'py>, tokens: Vec<String>) -> PyResult<Bound<'py, PyList>> {
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
    serde_json::to_string(&report).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
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
    m.add_class::<PySpellChecker>()?;
    m.add_class::<PyMorphology>()?;
    m.add_class::<PyCompoundDecomposer>()?;
    m.add_class::<PyDisambiguator>()?;
    m.add_class::<PyDependencyParser>()?;
    Ok(())
}
