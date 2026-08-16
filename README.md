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
- **Morphology (Morphological Analyzer & Generator)**:
  - Finite-State Morphotactics Graph covering nominal cases, plurals, possessives, verbal tenses, moods, aspects, person agreements, copulas, and derivational suffixes.
  - Morphological Generator for synthesizing words from lemmas and tags (`generate("kitap", ["Noun", "A3sg", "P1sg", "Dat"])` $\rightarrow$ `"kitabıma"`).
  - Morphological Disambiguator for context-aware best-parse selection.
- **Syntax & Dependency Parsing**:
  - Transition-based parser outputting Universal Dependencies (UD) format and CoNLL-U trees.
- **High-Performance Architecture**:
  - Pure Rust core with zero JVM dependency.
  - Python package via `pyo3` and `maturin` (compatible with `uv`).
  - Command Line Interface (CLI) for shell workflows.

---

## Performance Benchmarks (Akana vs Zeyrek / Zemberek)

Tested on 10,000 Turkish words on identical hardware:

| Benchmark Metric | Zeyrek (Python Zemberek Port) | Akana (Rust + PyO3) | Improvement |
| :--- | :--- | :--- | :--- |
| **Startup / Init Time** | `2,082.95 ms` (~2.1s) | **`0.14 ms`** | **15,259x faster** |
| **Throughput (10k words)** | `229 words/sec` (43.7s total) | **`84,203 words/sec`** (0.11s total) | **367.9x faster** |
| **Memory Footprint** | ~150 – 250 MB | **< 15 MB** | **>10x lighter** |
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
