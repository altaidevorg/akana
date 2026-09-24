"""
Akana: Modern and blazingly fast Turkish NLP toolkit in Rust & Python (PyO3).
"""

import json
from typing import Any

try:
    from akana._core import (
        Chunk,
        CompoundDecomposer,
        DependencyParser,
        Disambiguator,
        Embeddings,
        GrammarChecker,
        Morphology,
        PiiVault,
        SDPMChunker,
        SemanticChunker,
        SentenceChunker,
        SpellChecker,
        SyntacticMorphology,
        TurkishPiiEngine,
        analyze_document_json,
        analyze_readability_json,
        asciify,
        audit_ai_style_json,
        check_grammar_json,
        check_major_vowel_harmony,
        check_minor_vowel_harmony,
        chunk_sdpm,
        chunk_semantic,
        chunk_sentences,
        correct_grammar,
        count_syllables,
        currency_to_words,
        deasciify,
        embed,
        embed_batch,
        extract_entities_json,
        extract_keywords_json,
        generate_humanizer_prompt,
        harmonize_suffix,
        hyphenate,
        is_stopword,
        normalize_informal,
        number_to_words,
        ordinal_to_words,
        remove_stopwords,
        similarity,
        similarity_vectors,
        stem,
        summarize,
        syllabify,
        to_turkish_lower,
        to_turkish_title,
        to_turkish_upper,
        tokenize_embedding_text,
        tokenize_words,
        validate_credit_card,
        validate_iban,
        validate_imei,
        validate_plate,
        validate_ssn,
        validate_tckn,
        validate_vin,
        validate_vkn,
        words_to_number,
    )
except ImportError:
    pass


class DiagnosticFinding:
    def __init__(self, raw: dict[str, Any]):
        self.category: str = raw.get("category", "")
        self.severity: str = raw.get("severity", "")
        self.message: str = raw.get("message", "")
        self.suggestion: str | None = raw.get("suggestion")
        self.start: int = raw.get("start", 0)
        self.end: int = raw.get("end", 0)

    def __repr__(self) -> str:
        return (
            f"<DiagnosticFinding [{self.severity}]: {self.category} - '{self.message}'>"
        )


class StyleAuditReport:
    def __init__(self, raw: dict[str, Any]):
        self.ai_score: float = raw.get("ai_score", 0.0)
        self.verdict: str = raw.get("verdict", "")
        self.findings: list[DiagnosticFinding] = [
            DiagnosticFinding(f) for f in raw.get("findings", [])
        ]
        self.metrics: dict[str, Any] = raw.get("metrics", {})

    def __repr__(self) -> str:
        return f"<StyleAuditReport: Score={self.ai_score}/100 ({self.verdict}), {len(self.findings)} findings>"


class NamedEntity:
    def __init__(self, raw: dict[str, Any]):
        self.text: str = raw.get("text", "")
        self.label: str = raw.get("label", "")
        self.start: int = raw.get("start", 0)
        self.end: int = raw.get("end", 0)

    def __repr__(self) -> str:
        return f"<NamedEntity: '{self.text}' [{self.label}] ({self.start}:{self.end})>"


class Document:
    def __init__(self, raw_data: dict[str, Any]):
        self.text: str = raw_data.get("text", "")
        self.sentences: list[Sentence] = [
            Sentence(s) for s in raw_data.get("sentences", [])
        ]

    def __repr__(self) -> str:
        return f"<Document: {len(self.sentences)} sentences, {sum(len(s.tokens) for s in self.sentences)} tokens>"


class Sentence:
    def __init__(self, raw_data: dict[str, Any]):
        self.text: str = raw_data.get("text", "")
        self.tokens: list[str] = raw_data.get("tokens", [])
        self.parses: list[dict[str, Any]] = raw_data.get("parses", [])
        self.dependency_tree: dict[str, Any] = raw_data.get("dependency_tree", {})

    def __repr__(self) -> str:
        return f"<Sentence: '{self.text}'>"


class FormulaResult:
    def __init__(self, raw: dict[str, Any]):
        self.score: float = raw.get("score", 0.0)
        self.grade_level: str = raw.get("grade_level", "")
        self.description: str = raw.get("description", "")

    def __repr__(self) -> str:
        return f"<FormulaResult: score={self.score}, grade='{self.grade_level}'>"


class TextStatistics:
    def __init__(self, raw: dict[str, Any]):
        self.total_sentences: int = raw.get("total_sentences", 0)
        self.total_words: int = raw.get("total_words", 0)
        self.total_syllables: int = raw.get("total_syllables", 0)
        self.words_per_sentence: float = raw.get("words_per_sentence", 0.0)  # OCU
        self.syllables_per_word: float = raw.get("syllables_per_word", 0.0)  # OSU
        self.unfamiliar_words: int = raw.get("unfamiliar_words", 0)
        self.unfamiliar_word_ratio: float = raw.get(
            "unfamiliar_word_ratio", 0.0
        )  # KL (%)
        self.single_occurrence_words: int = raw.get("single_occurrence_words", 0)
        self.type_token_ratio: float = raw.get("type_token_ratio", 0.0)  # TTR (%)
        self.complex_sentences: int = raw.get("complex_sentences", 0)
        self.complex_sentence_ratio: float = raw.get(
            "complex_sentence_ratio", 0.0
        )  # KCO (%)
        self.total_fiilimsiler: int = raw.get("total_fiilimsiler", 0)
        self.clause_ratio: float = raw.get("clause_ratio", 0.0)  # YCO
        self.polysyllabic_words: int = raw.get("polysyllabic_words", 0)
        self.polysyllabic_word_ratio: float = raw.get("polysyllabic_word_ratio", 0.0)

    def __repr__(self) -> str:
        return f"<TextStatistics: sentences={self.total_sentences}, words={self.total_words}, KL={self.unfamiliar_word_ratio:.1f}%, TTR={self.type_token_ratio:.1f}%>"


class ReadabilityReport:
    def __init__(self, raw: dict[str, Any]):
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


def decompose_compound(word: str) -> list[dict[str, Any]]:
    """Decomposes a Turkish compound word into its constituents."""
    decomposer = CompoundDecomposer()
    return decomposer.decompose(word)


def extract_entities(text: str) -> list[NamedEntity]:
    """Extracts Named Entities (PER, LOC, ORG, DATE, MONEY, PERCENT) from Turkish text."""
    json_str = extract_entities_json(text)
    data = json.loads(json_str)
    return [NamedEntity(e) for e in data]


def extract_keywords(text: str, top_k: int = 10) -> list[dict[str, Any]]:
    """Extracts top keywords and keyphrases using Turkish RAKE and morphology."""
    json_str = extract_keywords_json(text, top_k)
    return json.loads(json_str)


def audit_ai_style(text: str) -> StyleAuditReport:
    """Audits Turkish text for AI writing signatures, punctuation inflation, and clichés."""
    json_str = audit_ai_style_json(text)
    data = json.loads(json_str)
    return StyleAuditReport(data)


def humanize_prompt(text: str, register: str = "blog") -> str:
    """Generates an actionable LLM rewrite prompt to humanize Turkish text."""
    return generate_humanizer_prompt(text, register)


class InflectionalGroup:
    def __init__(self, raw: dict[str, Any]):
        self.pos: str = raw.get("pos", "")
        self.derivation: str | None = raw.get("derivation")
        self.features: dict[str, str] = raw.get("features", {})

    def __repr__(self) -> str:
        deriv = f"-{self.derivation}" if self.derivation else ""
        return f"<IG [{self.pos}{deriv}] {self.features}>"


class SyntacticParse:
    def __init__(self, raw: dict[str, Any]):
        self.surface: str = raw.get("surface", "")
        self.root: str = raw.get("root", "")
        self.root_pos: str = raw.get("root_pos", "")
        self.is_proper: bool = raw.get("is_proper", False)
        self.formatted: str = raw.get("formatted", "")
        self.inflectional_groups: list[InflectionalGroup] = [
            InflectionalGroup(ig) for ig in raw.get("inflectional_groups", [])
        ]

    def __repr__(self) -> str:
        return f"<SyntacticParse: {self.formatted}>"


def syntactic_analyze(word: str) -> list[SyntacticParse]:
    """Performs two-level Google-style syntactic morphological analysis (FSMNLP 2019) with Inflectional Groups."""
    analyzer = SyntacticMorphology()
    raw_parses = analyzer.analyze(word)
    return [SyntacticParse(p) for p in raw_parses]


class GrammarFinding:
    def __init__(self, raw: dict[str, Any]):
        self.category: str = raw.get("category", "")
        self.start_offset: int = raw.get("start_offset", 0)
        self.end_offset: int = raw.get("end_offset", 0)
        self.original_text: str = raw.get("original_text", "")
        self.replacement: str = raw.get("replacement", "")
        self.message_tr: str = raw.get("message_tr", "")
        self.message_en: str = raw.get("message_en", "")
        self.confidence: float = raw.get("confidence", 1.0)

    def __repr__(self) -> str:
        return f"<GrammarFinding [{self.category}]: '{self.original_text}' -> '{self.replacement}' ({self.message_tr})>"


class GrammarCheckResult:
    def __init__(self, raw: dict[str, Any]):
        self.original: str = raw.get("original", "")
        self.corrected: str = raw.get("corrected", "")
        self.findings: list[GrammarFinding] = [
            GrammarFinding(f) for f in raw.get("findings", [])
        ]
        self.processing_time_us: int = raw.get("processing_time_us", 0)

    def __repr__(self) -> str:
        return f"<GrammarCheckResult: {len(self.findings)} findings, time={self.processing_time_us}µs>"


def check_grammar(text: str) -> GrammarCheckResult:
    """Performs full grammatical error detection and correction on Turkish text with detailed explanations."""
    raw_json = check_grammar_json(text)
    return GrammarCheckResult(json.loads(raw_json))


__version__ = "0.5.3"
__all__ = [
    "Chunk",
    "CompoundDecomposer",
    "DependencyParser",
    "DiagnosticFinding",
    "Disambiguator",
    "Document",
    "Embeddings",
    "FormulaResult",
    "GrammarCheckResult",
    "GrammarChecker",
    "GrammarFinding",
    "InflectionalGroup",
    "Morphology",
    "NamedEntity",
    "PiiVault",
    "ReadabilityReport",
    "SDPMChunker",
    "SemanticChunker",
    "Sentence",
    "SentenceChunker",
    "SpellChecker",
    "StyleAuditReport",
    "SyntacticMorphology",
    "SyntacticParse",
    "TextStatistics",
    "TurkishPiiEngine",
    "analyze",
    "analyze_readability",
    "asciify",
    "audit_ai_style",
    "check_grammar",
    "check_major_vowel_harmony",
    "check_minor_vowel_harmony",
    "chunk_sdpm",
    "chunk_semantic",
    "chunk_sentences",
    "correct_grammar",
    "count_syllables",
    "currency_to_words",
    "deasciify",
    "decompose_compound",
    "embed",
    "embed_batch",
    "extract_entities",
    "extract_keywords",
    "harmonize_suffix",
    "humanize_prompt",
    "hyphenate",
    "is_stopword",
    "normalize_informal",
    "number_to_words",
    "ordinal_to_words",
    "pii_mask",
    "pii_restore",
    "remove_stopwords",
    "similarity",
    "similarity_vectors",
    "stem",
    "summarize",
    "syllabify",
    "syntactic_analyze",
    "to_turkish_lower",
    "to_turkish_title",
    "to_turkish_upper",
    "tokenize_embedding_text",
    "tokenize_words",
    "validate_credit_card",
    "validate_iban",
    "validate_imei",
    "validate_plate",
    "validate_ssn",
    "validate_tckn",
    "validate_vin",
    "validate_vkn",
    "words_to_number",
]


def pii_mask(
    text: str,
    mode: str = "placeholder",
    use_embeddings: bool = False,
    preserve_corporate_emails: bool = True,
    disabled_types: list[str] | set[str] | None = None,
    vault: Any | None = None,
) -> dict[str, Any]:
    """
    Detects and masks/pseudonymizes Turkish PII entities (KVKK aligned).

    Args:
        text: Raw outbound prompt containing Turkish PII.
        mode: Masking strategy:
            - 'placeholder': structured tokens e.g. {{AD_1}}, {{TCKN_1}}, {{IBAN_1}}, {{OZEL_URL_1}}
            - 'surrogate': realistic fake Turkish synthetic data e.g. Can Demir, 51980838902
            - 'tag': bracketed classification tags e.g. [AD], [TCKN], [OZEL_URL]
            - 'anonymize': category label redaction e.g. [AD], [TCKN], [SIFRE], [IBAN]
        use_embeddings: If True, uses Akana's bundled 256-dim Model2Vec embedding prototype scorer.
        preserve_corporate_emails: If True (default), institutional support/functional emails
            (info@, destek@, satis@, etc.) are preserved and not masked as personal data.
        disabled_types: Optional list or set of PiiTypes to disable (e.g. ['Name', 'Person', 'Age']).
            Spans of disabled types are left completely verbatim.
        vault: Optional persistent PiiVault session instance to continue conversational context.

    Returns:
        Dict containing:
            - 'masked_text': Masked/pseudonymized prompt ready for LLM gateway.
            - 'mapping': Dict[str, str] mapping placeholders/surrogates to original values.
            - 'detailed_mapping': Dict[str, dict] with full stems, suffixes, and labels.
            - 'entities': List of detected entity spans and classifications.
            - 'vault': PiiVault instance with bi-directional session state.
    """
    engine = TurkishPiiEngine(
        use_embeddings=use_embeddings,
        preserve_corporate_emails=preserve_corporate_emails,
        disabled_types=list(disabled_types) if disabled_types else None,
    )
    if vault is not None:
        return engine.mask_with_vault(text, vault, mode=mode)
    return engine.mask(text, mode=mode)


def pii_restore(llm_response: str, mapping_or_vault: Any) -> str:
    """
    Restores masked placeholders or surrogates in an LLM response.

    Args:
        llm_response: Inbound LLM response text containing placeholders/surrogates.
        mapping_or_vault: Either:
            - Plain Python dict mapping (e.g. {"{{AD_1}}": "Ahmet Yılmaz"})
            - Detailed mapping dict (from result['detailed_mapping'])
            - PiiVault session instance (from result['vault'])
            - Serialized JSON string of the mapping or vault

    Returns:
        De-pseudonymized text with original sensitive values and restored Turkish vowel harmony.
    """
    engine = TurkishPiiEngine()
    return engine.restore(llm_response, mapping_or_vault)
