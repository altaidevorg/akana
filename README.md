# Akana (Turkish NLP Toolkit)

[![PyPI Version](https://img.shields.io/pypi/v/akana.svg)](https://pypi.org/project/akana/)
[![License: MIT / Apache-2.0](https://img.shields.io/badge/license-MIT%20%2F%20Apache--2.0-blue.svg)](LICENSE-MIT)
[![Rust: >= 1.75](https://img.shields.io/badge/rust-%3E%3D1.75-orange.svg)](https://www.rust-lang.org)
[![Python: >= 3.10](https://img.shields.io/badge/python-%3E%3D3.10-blue.svg)](https://www.python.org)

**Akana** (named after *Ak Ana*, the primordial creator goddess in Turkic mythology) is a modern, blazingly fast Turkish Natural Language Processing toolkit written in **Rust** with seamless **Python bindings via PyO3** and native hardware SIMD acceleration via **StringZilla**.

Repository: [https://github.com/altaidevorg/akana](https://github.com/altaidevorg/akana)

---

## Key Features

- **Phonology & Orthography Engine**:
  - Turkish alphabet characteristics and locale-aware casing (`ı` $\leftrightarrow$ `I`, `i` $\leftrightarrow$ `İ`).
  - Major (2-way `A/E`) and Minor (4-way `I/İ/U/Ü`) Vowel Harmony validation.
  - Consonant softening / mutation ($p \rightarrow b$, $ç \rightarrow c$, $t \rightarrow d$, $k \rightarrow \check{g}/g$).
  - Epenthetic vowel drop (*burun* $\rightarrow$ *burnu*, *akıl* $\rightarrow$ *aklı*).
  - Consonant doubling / gemination (*hak* $\rightarrow$ *hakkı*, *his* $\rightarrow$ *hissi*).
- **Tokenization & Sentence Segmentation**:
  - Zero-allocation, rule-based Turkish tokenizer handling proper nouns with apostrophes (`İstanbul'da`), abbreviations (`Prof.`, `Dr.`, `vb.`), currencies, URLs, emails, hashtags, dates, and times.
  - Sentence Boundary Detector with Turkish quotation and abbreviation lookahead.
- **Normalization & Spell Checking**:
  - **Asciifier** & **De-asciifier** for Turkish diacritics restoration.
  - **SIMD Spell Checker**: Accelerated with **StringZilla** hardware instructions for ultra-fast Levenshtein / edit distance candidate scoring.
  - **Informal Text Normalizer**: Spoken Turkish colloquialisms reduction (`yapcam` $\rightarrow$ `yapacağım`, `geliyom` $\rightarrow$ `geliyorum`, `noldu` $\rightarrow$ `ne oldu`) and letter elongation deduping (`çooook` $\rightarrow$ `çok`).
- **Dual-Engine Morphology Suite**:
  1. **Standard Morphology (`akana.Morphology`)**:
     - **93,167 Root Lexicon**: Broad-coverage Turkish vocabulary ingested from Zemberek, TDK, location gazetteers, and modern corpus lexicons.
     - Multi-tier morphotactic graph: nominal cases, plurals, possessives, verbal tenses, compound copulas, voices (passive/causative), participles, diminutives, relative `-ki` chains, and derivations.
     - **Compound Word Decomposer**: Deconstructs compound nouns (`denizaltı` $\rightarrow$ `deniz + altı`, `akbaba` $\rightarrow$ `ak + baba`).
     - Morphological Generator (`generate("kitap", ["Noun", "A3sg", "P1sg", "Dat"])` $\rightarrow$ `"kitabıma"`).
     - Context-aware Disambiguator for best-parse selection.
  2. **Syntactic Expressive Morphology (`akana.SyntacticMorphology`)** *(Google FSMNLP 2019 Architecture)*:
     - **Inflectional Groups (IG)**: Hierarchical derivational tiers with Universal Dependencies (UD) category-value feature maps.
     - **Zero-Derivation Elimination**: Cross-categorized lexical entries (e.g. *güzel*, *hızlı*, *soğuk*) eliminating phantom `+^DB` morphemes.
     - Dedicated, isolated 47,202 gold-standard root lexicon.
- **Modern & Classic Turkish Readability Suite**:
  - **Kalyoncu (2025) Formula Suite**: Multi-regression equations (Formulas 1–4, $R^2$ up to 0.99) with embedded 4,600-word familiarity lexicon and exact grade-level mapping (*3. Sınıf Öncesi* to *Lisansüstü*).
  - **Classical Formulas**: Ateşman (1997), Çetinkaya-Uzun (2010), and Bezirci-Yılmaz (2010).
- **Turkish AI Writing Style Auditor & Humanizer Engine**:
  - Detects LLM writing signatures: punctuation anomalies (excessive em-dashes, semicolons with conjunctions, colons), predicate tense repetition (`-mektedir` ratio), rhythm monotony ($CV = \sigma / \mu$), bureaucratic connectors, translationese calques, tricolon lists, and hypophora questions.
  - Actionable prompt generation across 5 registers (*hukuki-idari, akademik-kurumsal, analitik-gazetecilik, deneme-blog, edebi-yaratıcı*).
- **Syntax & Universal Dependencies**:
  - Transition-based parser outputting Universal Dependencies (UD) format and CoNLL-U trees.
- **High-Level NLP Primitives**:
  - Turkish Syllabification & Hyphenation.
  - Number to Words Converter (Cardinals, Ordinals, Currency).
  - Named Entity Recognition (PER, LOC, ORG, DATE, MONEY, PERCENT).
  - Keyword Extraction (Turkish RAKE) & Extractive Summarization (TextRank).
- **Grammatical Error Correction & Detection (GEC/GED) Engine**:
  - **Full GECTurk 25-Category Coverage**: High-precision rule-based grammar and orthography checker covering clitic separations (`de/da`, `ki`, `mi`), consonant assimilation (*kitapda* $\rightarrow$ *kitapta*), vowel syncope (*akılı* $\rightarrow$ *aklı*), consonant softening (*kitapı* $\rightarrow$ *kitabı*), over-narrowing (*başlıyan* $\rightarrow$ *başlayan*), proper noun / numeric apostrophes (*Ahmetler'in* $\rightarrow$ *Ahmetlerin*, *1923'de* $\rightarrow$ *1923'te*), compound modal verbs (*ola bilir* $\rightarrow$ *olabilir*), indefinite determiners (*bir çok* $\rightarrow$ *birçok*), reduplications (*elele* $\rightarrow$ *el ele*), and tautologies.
  - **Hardware SIMD Acceleration**: Accelerated with **StringZilla** for zero-regex, full-text substring and edit-distance scanning reaching **>1,470 sentences/sec** (>16,000 tokens/sec) on a single CPU core.
  - **Linguistic Diagnostics**: Detailed Turkish and English explanations with character-level finding offsets and confidence scores.
- **High-Performance Architecture**:
  - Pure Rust core with zero JVM dependency.
  - Python package via `pyo3` and `maturin` (compatible with `uv`).
  - Command Line Interface (CLI) for shell workflows.

---

## Performance Benchmarks (Akana vs Zeyrek / Zemberek & StringZilla SIMD)

Tested on real Turkish text corpora and 10,500 morphological queries (`benchmarks/`):

| Benchmark Metric | Zeyrek (Python Zemberek Port) | Akana (Rust + StringZilla SIMD) | Performance / Throughput |
| :--- | :--- | :--- | :--- |
| **Active Root Lexicon** | ~90,000 roots | **93,167 roots** | **Full Coverage** |
| **Startup / Lexicon Init** | `2,733.8 ms` (~2.7s) | **`200.2 ms`** | **13.6x faster** |
| **Morphological Parse (10.5k words)** | `55,349.7 ms` (55.3s) | **`989.7 ms`** (0.98s) | **55.9x faster** (`10,609 words/sec`) |
| **Tokenization (Zero-Allocation)** | ~230 words/sec | **`949,991 tokens/sec`** | **>4,000x faster** (<21 ms for 19.5k tokens) |
| **Informal Normalization** | N/A | **`36,222 words/sec`** | **High Throughput** (Zero-Regex Suffix Matching) |
| **AI Writing Style Audit** | N/A | **`27,179 words/sec`** | **StringZilla SIMD** (10.2k words in 375 ms) |
| **Named Entity Recognition (NER)** | N/A | **`1.14 MB/sec`** | **Linear Token Stream** (1,500 entities in 37 ms) |
| **Grammar Correction (GEC)** | ~22–55 sent/s (Neural) | **`1,471 sent/s`** | **~27x–67x faster** (20.7k sentences in 14.1s) |
| **Hardware Acceleration** | Pure Python loops | **StringZilla AVX-512 / AVX2 / NEON** | **Native SIMD Instructions** |

### Grammatical Error Correction Benchmark (GECTurk - arXiv:2309.11346)

Evaluated across the full 25-category **HuggingFace `GGLab/GECTurk`** benchmark test sets:

| Model Architecture | Execution Device | Throughput | Latency (ms) | Out-of-Domain $F_{0.5}$ (Human Movie Reviews) | Full In-Domain $F_{0.5}$ (20,769 Sents) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **mT5-base (NMT)** | GPU (NVIDIA T4) | ~22 sent/s | 45.0 ms | 42.1% | 70.4% |
| **mGPT (Prefix-Tuning)** | GPU (NVIDIA T4) | ~15 sent/s | 65.0 ms | 41.8% | 66.5% |
| **SeqTag (BERTurk)** | CPU (8-core) | ~55 sent/s | 18.0 ms | 52.8% | 86.2% |
| **Akana (Rust Engine)** | **CPU (1-core)** | **`1,471 sent/s`** | **`0.68 ms`** | **`75.3%`** | **`77.8%`** |

* 🚀 **Throughput Speedup:** **26.7x faster** than BERTurk on CPU and **66.8x faster** than mT5 on GPU.
* 🎯 **Out-of-Domain Superiority:** Akana achieves **75.3% $F_{0.5}$** on real-world human movie reviews (outperforming BERTurk at 52.8% and mT5 at 42.1%) with zero neural generative hallucinations.

---

## Python Quickstart

### Installation

```bash
# Using uv
uv pip install akana

# Using pip
pip install akana
```

### Usage in Python

```python
import akana

# 1. Advanced Grammatical Error Correction (GEC) & Diagnostics
text = "Ali de geldi, Veli te geldi. Evi terketmek zorunda kaldı ve 1923'de kurulan cumhuriyeti andık."

# Direct correction
corrected = akana.correct_grammar(text)
print("Corrected:", corrected)
# -> "Ali de geldi, Veli de geldi. Evi terk etmek zorunda kaldı ve 1923'te kurulan cumhuriyeti andık."

# Detailed diagnostic findings
res = akana.check_grammar(text)
for f in res.findings:
    print(f"[{f.category}] '{f.original_text}' -> '{f.replacement}' | {f.message_tr}")

# 2. Standard Morphological Analysis (Zemberek-Compatible Format)
morph = akana.Morphology()
parses = morph.analyze("kitabıma")
for parse in parses:
    print(parse["lemma"], parse["primary_pos"], parse["morphemes"])
# -> kitap Noun ['Noun', 'P1sg', 'Dat']

# 3. Google-Style Syntactic Expressive Morphology (Inflectional Groups & UD)
syn_parses = akana.syntactic_analyze("geldiğimizde")
for p in syn_parses:
    print(p.formatted)
    # Output: (gel[VB]+[Polarity=Pos])([NOMP]-PastNom+[Case=Loc]+[PersonNumber=A3sg]+[Possessive=P1pl])+[Proper=False]
    for ig in p.inflectional_groups:
        print(f"  • IG [{ig.pos}] Deriv: {ig.derivation} -> {ig.features}")

# 4. Morphological Generation
surface = morph.generate("kitap", ["Noun", "A3sg", "P1sg", "Dat"])
print(surface)  # -> kitabıma

# 5. Spell Checking with StringZilla SIMD
spell = akana.SpellChecker()
print("Is 'kitap' correct?", spell.is_correct("kitap"))
suggestions = spell.suggest("ktap", max_distance=2, max_suggestions=3)
print("Suggestions for 'ktap':", [s["word"] for s in suggestions])

# 6. De-asciification & Normalization
print(akana.deasciify("turkce nlp cok hizli calisiyor"))
# -> türkçe nlp çok hızlı çalışıyor

print(akana.normalize_informal("nooldu ya yapcam dedim"))
# -> ne oldu ya yapacağım dedim

# 7. Compound Word Decomposition
compounds = akana.decompose_compound("denizaltı")
print(compounds)
# -> [{'surface': 'denizaltı', 'part1': 'deniz', 'part2': 'altı', ...}]

# 8. Modern Turkish Readability Analysis (Kalyoncu 2025 & Classic)
report = akana.analyze_readability("Küçük çocuk bahçede neşeyle koşuyordu.")
print(f"Kalyoncu F1: {report.kalyoncu_formula1.score} ({report.kalyoncu_formula1.grade_level})")
print(f"Ateşman: {report.atesman.score} ({report.atesman.grade_level})")

# 9. Turkish AI Writing Style Auditor & Actionable Humanizer Prompt
audit = akana.audit_ai_style("Yapay zeka teknolojileri, modern dünyada kritik bir rol oynamaktadır. Bu bağlamda —özellikle veri alanında— hayati önem taşımaktadır.")
print(f"AI Score: {audit.ai_score}/100 ({audit.verdict})")

prompt = akana.humanize_prompt("Bu doğrultuda hayati önem taşımaktadır.", register="blog")
print(prompt)

# 10. High-Level Turkish NLP Suite
# Syllabification & Hyphenation
print(akana.syllabify("Türkçe"))     # -> ['Türk', 'çe']
print(akana.hyphenate("bilgisayar")) # -> 'bil-gi-sa-yar'

# Number to Words Converter
print(akana.number_to_words(1923))            # -> 'bin dokuz yüz yirmi üç'
print(akana.currency_to_words(1250.50, "TL"))  # -> 'bin iki yüz elli lira elli kuruş'

# Named Entity Recognition (NER)
entities = akana.extract_entities("Prof. Dr. Ahmet Yılmaz 16 Ağustos 2026 tarihinde 500 TL ödeme yaptı.")
for e in entities:
    print(f"[{e.label}] {e.text}")

# Keyword Extraction (Turkish RAKE) & Extractive Summarization (TextRank)
keywords = akana.extract_keywords("Doğal dil işleme ve morfolojik analiz...", top_k=5)
summary = akana.summarize("Uzun metin...", max_sentences=2)
```

---

## CLI Usage

The `akana` CLI supports direct text arguments or reading from file via `-f, --file`:

```bash
# AI style auditing
akana ai-audit "Bu bağlamda kritik bir rol oynamaktadır."
akana ai-audit -f article.txt

# Generate humanizer rewrite prompt
akana humanize-prompt "Bu doğrultuda hayati önem taşımaktadır." --register blog

# Syntactic morphological analysis (Google FSMNLP format)
akana syntactic-analyze "geldiğimizde"

# Standard morphological analysis
akana analyze "evlerimizde"

# Readability analysis
akana readability "Küçük çocuk bahçede neşeyle koşuyordu."

# Syllabification & Number conversion
akana syllabify "bilgisayar"
akana number 1923

# De-asciification & Normalization
akana deasciify "turkce nlp"
akana normalize "yapcam"

# Universal Dependencies Parsing
akana parse "Ali güzel kitabı okudu."
```

---

## Rust Crate Usage (`akana-core`)

Add to `Cargo.toml`:
```toml
[dependencies]
akana-core = { version = "0.1", default-features = true }
```

```rust
use akana_core::grammar::TurkishGrammarChecker;
use akana_core::morphology::TurkishMorphology;
use akana_core::syntactic_morphology::TurkishSyntacticMorphology;
use akana_core::phonology::to_turkish_lower;

fn main() {
    let lower = to_turkish_lower("İSTANBUL");
    println!("Lower: {}", lower);

    // 1. Grammatical Error Correction & Diagnostics
    let grammar_checker = TurkishGrammarChecker::new();
    let res = grammar_checker.check("Ali de geldi, Veli te geldi. Pazardan üç elmalar aldık.");
    println!("Corrected: {}", res.corrected);
    for f in &res.findings {
        println!("[{:?}] '{}' -> '{}'", f.category, f.original_text, f.replacement);
    }

    // 2. Standard Morphology
    let morph = TurkishMorphology::new();
    let parses = morph.analyze("kitabım");
    for p in parses {
        println!("{}", p.formatted);
    }

    // 3. Syntactic Expressive Morphology (Inflectional Groups)
    let syn_morph = TurkishSyntacticMorphology::new();
    let syn_parses = syn_morph.analyze("geldiğimizde");
    for p in syn_parses {
        println!("{}", p.formatted);
    }
}
```

---

## Developer Guide & Publishing

For local development setup, testing, running benchmarks, compiling native wheels, and publishing releases to PyPI & crates.io, see the [Developer & Maintainer Guide](developer-guide.md).

---

## Acknowledgements & Academic Citations

Akana builds upon decades of pioneering linguistic and natural language processing research in Turkish. We gratefully acknowledge and credit:

- **Kemal Oflazer**: Foundational two-level Turkish morphological analysis (1994) and the Inflectional Group (IG) representation (2003) for Turkish dependency syntax.
- **Ahmet A. Akın & The Zemberek Team**: Open-source Turkish morphology, phonotactics, and extensive root vocabulary database.
- **Oğuzhan Güngör & Zeyrek Contributors**: The pure-Python Zemberek port that inspired modern open Turkish NLP tooling.
- **Koç University GGLab (Duygu Ataman & Co-authors)**: *"GECTurk: Grammatical Error Correction and Detection Dataset for Turkish"* (arXiv:2309.11346), providing the 25-category Turkish grammatical error taxonomy and benchmark datasets.
- **Google Research (Adnan Öztürel, Tolga Kayadelen, Işın Demirşahin)**: *"A Syntactically Expressive Morphological Analyzer for Turkish"* (FSMNLP 2019), introducing zero-derivation elimination and two-level inflectional group FSTs.
- **Mustafa Kalyoncu & Co-authors (2025)**: Development of modern Turkish readability formulas (Formulas 1–4) and the empirical 4,600-word grade-level familiarity corpus.
- **Ender Ateşman (1997), Çetinkaya-Uzun (2010), Bezirci-Yılmaz (2010)**: Classical readability research for the Turkish education system.
- **Ash Vardanian & Unum Cloud**: **StringZilla**, providing hardware-accelerated SIMD vector search and edit distance algorithms.
- **Bushra Beg (Turkce-Humanizer)**: Research into Turkish AI writing style signatures, calques, and stylistic heuristics.

---

## License

Licensed under either of:
- [Apache License, Version 2.0](LICENSE-APACHE)
- [MIT License](LICENSE-MIT)

at your option.
