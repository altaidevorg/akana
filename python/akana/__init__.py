"""
Akana: Modern and blazingly fast Turkish NLP toolkit in Rust & Python (PyO3).
"""

from typing import List, Optional, Dict, Any
import json

try:
    from akana._core import (
        to_turkish_lower,
        to_turkish_upper,
        to_turkish_title,
        check_major_vowel_harmony,
        check_minor_vowel_harmony,
        asciify,
        deasciify,
        normalize_informal,
        tokenize_words,
        SpellChecker,
        Morphology,
        CompoundDecomposer,
        Disambiguator,
        DependencyParser,
        analyze_document_json,
        analyze_readability_json,
        syllabify,
        hyphenate,
        count_syllables,
        number_to_words,
        ordinal_to_words,
        currency_to_words,
        words_to_number,
        stem,
        is_stopword,
        remove_stopwords,
        extract_entities_json,
        extract_keywords_json,
        summarize,
    )
except ImportError:
    pass

class NamedEntity:
    def __init__(self, raw: Dict[str, Any]):
        self.text: str = raw.get("text", "")
        self.label: str = raw.get("label", "")
        self.start: int = raw.get("start", 0)
        self.end: int = raw.get("end", 0)

    def __repr__(self) -> str:
        return f"<NamedEntity: '{self.text}' [{self.label}] ({self.start}:{self.end})>"

class Document:
    def __init__(self, raw_data: Dict[str, Any]):
        self.text: str = raw_data.get("text", "")
        self.sentences: List[Sentence] = [Sentence(s) for s in raw_data.get("sentences", [])]

    def __repr__(self) -> str:
        return f"<Document: {len(self.sentences)} sentences, {sum(len(s.tokens) for s in self.sentences)} tokens>"

class Sentence:
    def __init__(self, raw_data: Dict[str, Any]):
        self.text: str = raw_data.get("text", "")
        self.tokens: List[str] = raw_data.get("tokens", [])
        self.parses: List[Dict[str, Any]] = raw_data.get("parses", [])
        self.dependency_tree: Dict[str, Any] = raw_data.get("dependency_tree", {})

    def __repr__(self) -> str:
        return f"<Sentence: '{self.text}'>"

class FormulaResult:
    def __init__(self, raw: Dict[str, Any]):
        self.score: float = raw.get("score", 0.0)
        self.grade_level: str = raw.get("grade_level", "")
        self.description: str = raw.get("description", "")

    def __repr__(self) -> str:
        return f"<FormulaResult: score={self.score}, grade='{self.grade_level}'>"

class TextStatistics:
    def __init__(self, raw: Dict[str, Any]):
        self.total_sentences: int = raw.get("total_sentences", 0)
        self.total_words: int = raw.get("total_words", 0)
        self.total_syllables: int = raw.get("total_syllables", 0)
        self.words_per_sentence: float = raw.get("words_per_sentence", 0.0)  # OCU
        self.syllables_per_word: float = raw.get("syllables_per_word", 0.0)  # OSU
        self.unfamiliar_words: int = raw.get("unfamiliar_words", 0)
        self.unfamiliar_word_ratio: float = raw.get("unfamiliar_word_ratio", 0.0)  # KL (%)
        self.single_occurrence_words: int = raw.get("single_occurrence_words", 0)
        self.type_token_ratio: float = raw.get("type_token_ratio", 0.0)  # TTR (%)
        self.complex_sentences: int = raw.get("complex_sentences", 0)
        self.complex_sentence_ratio: float = raw.get("complex_sentence_ratio", 0.0)  # KCO (%)
        self.total_fiilimsiler: int = raw.get("total_fiilimsiler", 0)
        self.clause_ratio: float = raw.get("clause_ratio", 0.0)  # YCO
        self.polysyllabic_words: int = raw.get("polysyllabic_words", 0)
        self.polysyllabic_word_ratio: float = raw.get("polysyllabic_word_ratio", 0.0)

    def __repr__(self) -> str:
        return f"<TextStatistics: sentences={self.total_sentences}, words={self.total_words}, KL={self.unfamiliar_word_ratio:.1f}%, TTR={self.type_token_ratio:.1f}%>"

class ReadabilityReport:
    def __init__(self, raw: Dict[str, Any]):
        self.statistics = TextStatistics(raw.get("statistics", {}))
        
        k_raw = raw.get("kalyoncu", {})
        self.kalyoncu_formula1 = FormulaResult(k_raw.get("formula1", {}))
        self.kalyoncu_formula2 = FormulaResult(k_raw.get("formula2", {}))
        self.kalyoncu_formula3 = FormulaResult(k_raw.get("formula3", {}))
        self.kalyoncu_formula4 = FormulaResult(k_raw.get("formula4", {}))

        l_raw = raw.get("legacy", {})
        self.atesman = FormulaResult(l_raw.get("atesman", {}))
        self.cetinkaya_uzun = FormulaResult(l_raw.get("cetinkaya_uzun", {}))
        self.bezirci_yilmaz = FormulaResult(l_raw.get("bezirci_yilmaz", {}))

    def __repr__(self) -> str:
        return (
            f"<ReadabilityReport: Kalyoncu-F1={self.kalyoncu_formula1.score} ({self.kalyoncu_formula1.grade_level}), "
            f"Ateşman={self.atesman.score} ({self.atesman.grade_level})>"
        )

def analyze(text: str) -> Document:
    """Performs full end-to-end NLP analysis on Turkish text."""
    json_str = analyze_document_json(text)
    data = json.loads(json_str)
    return Document(data)

def analyze_readability(text: str) -> ReadabilityReport:
    """Calculates modern (Kalyoncu 2025) and classic Turkish readability metrics."""
    json_str = analyze_readability_json(text)
    data = json.loads(json_str)
    return ReadabilityReport(data)

def decompose_compound(word: str) -> List[Dict[str, Any]]:
    """Decomposes a Turkish compound word into its constituents."""
    decomposer = CompoundDecomposer()
    return decomposer.decompose(word)

def extract_entities(text: str) -> List[NamedEntity]:
    """Extracts Named Entities (PER, LOC, ORG, DATE, MONEY, PERCENT) from Turkish text."""
    json_str = extract_entities_json(text)
    data = json.loads(json_str)
    return [NamedEntity(e) for e in data]

def extract_keywords(text: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """Extracts top keywords and keyphrases using Turkish RAKE and morphology."""
    json_str = extract_keywords_json(text, top_k)
    return json.loads(json_str)

__version__ = "0.2.0"
__all__ = [
    "to_turkish_lower",
    "to_turkish_upper",
    "to_turkish_title",
    "check_major_vowel_harmony",
    "check_minor_vowel_harmony",
    "asciify",
    "deasciify",
    "normalize_informal",
    "tokenize_words",
    "SpellChecker",
    "Morphology",
    "CompoundDecomposer",
    "decompose_compound",
    "Disambiguator",
    "DependencyParser",
    "analyze",
    "analyze_readability",
    "syllabify",
    "hyphenate",
    "count_syllables",
    "number_to_words",
    "ordinal_to_words",
    "currency_to_words",
    "words_to_number",
    "stem",
    "is_stopword",
    "remove_stopwords",
    "extract_entities",
    "extract_keywords",
    "summarize",
    "Document",
    "Sentence",
    "NamedEntity",
    "ReadabilityReport",
    "TextStatistics",
    "FormulaResult",
]
