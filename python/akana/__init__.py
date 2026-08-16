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
    )
except ImportError:
    # Development fallback
    pass

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

def analyze(text: str) -> Document:
    """Performs full end-to-end NLP analysis on Turkish text."""
    json_str = analyze_document_json(text)
    data = json.loads(json_str)
    return Document(data)

def decompose_compound(word: str) -> List[Dict[str, Any]]:
    """Decomposes a Turkish compound word into its constituents."""
    decomposer = CompoundDecomposer()
    return decomposer.decompose(word)

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
    "Document",
    "Sentence",
]
