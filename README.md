# Akana (Turkish NLP Toolkit)

**Akana** (named after *Ak Ana*, the primordial creator goddess in Turkic mythology) is a modern, blazingly fast Turkish Natural Language Processing toolkit written in **Rust** with seamless **Python bindings via PyO3** and SIMD acceleration via **StringZilla**.

---

## Key Features

- **Phonology & Orthography Engine**:
  - Turkish alphabet characteristics and locale-aware casing (`ı` $\leftrightarrow$ `I`, `i` $\leftrightarrow$ `İ`).
  - Major (2-way `A/E`) and Minor (4-way `I/İ/U/Ü`) Vowel Harmony checks.
  - Consonant softening / mutation ($p \rightarrow b$, $ç \rightarrow c$, $t \rightarrow d$, $k \rightarrow \check{g}/g$).
  - Vowel drop (*burun* $\rightarrow$ *burnu*, *akıl* $\rightarrow$ *aklı*).
  - Consonant doubling (*hak* $\rightarrow$ *hakkı*, *his* $\rightarrow$ *hissi*).
- **Tokenization & Sentence Segmentation**:
  - Zero-copy, rule-based Turkish tokenizer handling proper nouns with apostrophes (`İstanbul'da`), abbreviations (`Prof.`, `Dr.`, `vb.`), currencies, URLs, emails, hashtags, dates, and times.
  - Sentence Boundary Detector with Turkish quotation and abbreviation lookahead.
- **Normalization & Spell Checking**:
  - **Asciifier** & **De-asciifier** for Turkish diacritics restoration.
  - **SIMD Spell Checker**: Accelerated with **StringZilla** hardware instructions for fast Levenshtein / edit distance candidate scoring.
  - **Informal Text Normalizer**: Spoken Turkish colloquialisms reduction (`yapcam` $\rightarrow$ `yapacağım`, `geliyom` $\rightarrow$ `geliyorum`, `noldu` $\rightarrow$ `ne oldu`) and letter elongation deduping (`çooook` $\rightarrow$ `çok`).
- **Morphology (Morphological Analyzer, Generator & Compound Decomposer)**:
  - **93,000+ Root Lexicon**: Complete Turkish vocabulary ingested from Zemberek, TDK, location gazetteers, and modern corpus lexicons.
  - Finite-State Morphotactics Graph covering nominal cases, plurals, possessives, verbal tenses, compound tenses, voices (passive/causative), participles, diminutives, relative `-ki` chains, and derivations.
  - **Compound Word Decomposer**: Deconstructs compound nouns (`denizaltı` $\rightarrow$ `deniz + altı`, `akbaba` $\rightarrow$ `ak + baba`).
  - Morphological Generator for synthesizing words from lemmas and tags (`generate("kitap", ["Noun", "A3sg", "P1sg", "Dat"])` $\rightarrow$ `"kitabıma"`).
  - Morphological Disambiguator for context-aware best-parse selection.
- **Modern Turkish Readability Suite**:
  - **Kalyoncu (2025) Formula Suite**: Multi-regression equations (Formulas 1–4, $R^2$ up to 0.99) with embedded 4,600-word familiarity lexicon and exact grade-level mapping (*3. Sınıf Öncesi* to *Lisansüstü*).
  - **Classical Formulas**: Ateşman (1997), Çetinkaya-Uzun (2010), and Bezirci-Yılmaz (2010).
- **Syntax & Dependency Parsing**:
  - Transition-based parser outputting Universal Dependencies (UD) format and CoNLL-U trees.
- **High-Performance Architecture**:
  - Pure Rust core with zero JVM dependency.
  - Python package via `pyo3` and `maturin` (compatible with `uv`).
  - Command Line Interface (CLI) for shell workflows.

---

## Performance Benchmarks (Akana vs Zeyrek / Zemberek)

Tested on 10,500 real morphological queries using the complete 93,000-word Turkish root lexicon (`benchmarks/benchmark_vs_zeyrek.py`):

| Benchmark Metric | Zeyrek (Python Zemberek Port) | Akana (Rust + PyO3) | Improvement |
| :--- | :--- | :--- | :--- |
| **Active Vocabulary** | ~90,000 roots | **93,167 roots** | **Full Coverage** |
| **Startup / Lexicon Init** | `2,733.8 ms` (~2.7s) | **`200.2 ms`** | **13.6x faster** |
| **Full Morphological Parse (10.5k queries)** | `55,349.7 ms` (55.3s) | **`989.7 ms`** (0.98s) | **55.9x faster** |
| **Parsing Throughput** | `189.7 words/sec` | **`10,609.0 words/sec`** | **55.9x higher** |
| **Tokenization & Normalization** | ~230 words/sec | **>84,000 words/sec** | **>360x faster** |
| **String Acceleration** | Pure Python loops / Regex | **StringZilla SIMD (AVX/NEON)** | **Hardware Accelerated** |

---

## Python Quickstart

### Installation via `uv` / `maturin`

```bash
# Build and install locally with uv & maturin
uv pip install maturin
uv run maturin develop --release
```

### Usage in Python

```python
import akana

# 1. Morphological Analysis
morph = akana.Morphology()
parses = morph.analyze("kitabıma")
for parse in parses:
    print(parse["lemma"], parse["primary_pos"], parse["morphemes"])
# -> kitap Noun ['Noun', 'P1sg', 'Dat']

# 2. Morphological Generation
surface = morph.generate("kitap", ["Noun", "A3sg", "P1sg", "Dat"])
print(surface)
# -> kitabıma

# 3. Spell Checking with StringZilla SIMD
spell = akana.SpellChecker()
print("Is 'kitap' correct?", spell.is_correct("kitap"))
suggestions = spell.suggest("ktap", max_distance=2, max_suggestions=3)
print("Suggestions for 'ktap':", [s["word"] for s in suggestions])

# 4. De-asciification & Normalization
print(akana.deasciify("turkce nlp cok hizli calisiyor"))
# -> türkçe nlp çok hızlı çalışıyor

print(akana.normalize_informal("nooldu ya yapcam dedim"))
# -> ne oldu ya yapacağım dedim

# 5. Compound Word Decomposition
compounds = akana.decompose_compound("denizaltı")
print(compounds)
# -> [{'surface': 'denizaltı', 'part1': 'deniz', 'part2': 'altı', ...}]

# 6. Modern Turkish Readability Analysis (Kalyoncu 2025 & Classic)
report = akana.analyze_readability("Küçük çocuk bahçede neşeyle koşuyordu.")
print(f"Kalyoncu F1: {report.kalyoncu_formula1.score} ({report.kalyoncu_formula1.grade_level})")
print(f"Ateşman: {report.atesman.score} ({report.atesman.grade_level})")

# 7. Full End-to-End Document Analysis
doc = akana.analyze("Ak Ana, Türk mitolojisinde deniz tanrıçasıdır. Prof. Dr. Ayşe Hanım geldi.")
for sentence in doc.sentences:
    print("Sentence:", sentence.text)
    print("Tokens:", sentence.tokens)
```

---

## CLI Usage

```bash
# Readability analysis
cargo run -p akana-cli -- readability "Küçük çocuk bahçede neşeyle koşuyordu."

# Tokenization
cargo run -p akana-cli -- tokenize "Prof. Dr. Ahmet İstanbul'a gitti."

# Morphological analysis
cargo run -p akana-cli -- analyze "evlerimizde"

# Morphological generation
cargo run -p akana-cli -- generate kitap --tags A3pl,P1sg,Loc

# De-asciification
cargo run -p akana-cli -- deasciify "turkce nlp"

# Dependency parsing
cargo run -p akana-cli -- parse "Ali güzel kitabı okudu."
```

---

## Rust Crate Usage (`akana-core`)

Add to `Cargo.toml`:
```toml
[dependencies]
akana-core = { path = "crates/akana-core" }
```

```rust
use akana_core::morphology::TurkishMorphology;
use akana_core::phonology::to_turkish_lower;

fn main() {
    let lower = to_turkish_lower("İSTANBUL");
    println!("Lower: {}", lower);

    let morph = TurkishMorphology::new();
    let parses = morph.analyze("kitabım");
    for p in parses {
        println!("{}", p.formatted);
    }
}
```

---

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache 2.0](LICENSE-APACHE).
