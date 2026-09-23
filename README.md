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
- **Turkish Text Chunking Suite for RAG & LLMs (Chonkie-Inspired)**:
  - **`SemanticChunker`**: Segments long Turkish documents at natural topic shifts using embedded 2-bit TurboQuant vector similarity drops.
  - **Adaptive Threshold Modes**: Direct `similarity`, `percentile` (default: 0.75), `standard_deviation` ($\mu - k\sigma$), `interquartile` ($Q_1 - k \cdot IQR$), and `auto`.
  - **Similarity Smoothing**: Built-in 5-point Savitzky-Golay and Moving Average digital filters to suppress high-frequency local noise in sentence transitions.
  - **`SDPMChunker` (Semantic Double-Pass Merge)**: Two-pass hierarchical merge algorithm creating dense, coherent chunks with high information packing for vector databases.
  - **`SentenceChunker`**: Rule-based sentence chunking respecting Turkish grammatical boundaries with configurable token counts and overlaps.
  - **High Performance**: **>50,000–1,000,000 words/sec** on CPU with exact UTF-8 character and byte offset preservation.
- **Embedded Turkish Sentence Embeddings (TurboQuant 2-Bit Model2Vec)**:
  - **Ultra-lightweight (2.5 MB)**: Built-in 2-bit quantized static sentence embedding model distilled from BGE-M3 (`altaidevorg/turkish-bge-m3-model2vec-turboquant-2bit`).
  - **256-dimensional** vector embeddings with **92.19%** STSb-TR benchmark accuracy.
  - **High Throughput**: **>20,000 sentences/sec** on CPU with zero deep learning runtime or PyTorch dependencies.
  - Built-in cosine similarity and batch embedding support.
- **Grammatical Error Correction & Detection (GEC/GED) Engine**:
  - **Full GECTurk 25-Category Coverage**: High-precision rule-based grammar and orthography checker covering clitic separations (`de/da`, `ki`, `mi`), consonant assimilation (*kitapda* $\rightarrow$ *kitapta*), vowel syncope (*akılı* $\rightarrow$ *aklı*), consonant softening (*kitapı* $\rightarrow$ *kitabı*), over-narrowing (*başlıyan* $\rightarrow$ *başlayan*), proper noun / numeric apostrophes (*Ahmetler'in* $\rightarrow$ *Ahmetlerin*, *1923'de* $\rightarrow$ *1923'te*), compound modal verbs (*ola bilir* $\rightarrow$ *olabilir*), indefinite determiners (*bir çok* $\rightarrow$ *birçok*), reduplications (*elele* $\rightarrow$ *el ele*), and tautologies.
  - **Hardware SIMD Acceleration**: Accelerated with **StringZilla** for zero-regex, full-text substring and edit-distance scanning reaching **>1,470 sentences/sec** (>16,000 tokens/sec) on a single CPU core.
  - **Linguistic Diagnostics**: Detailed Turkish and English explanations with character-level finding offsets and confidence scores.
- **Turkish PII Masking & Bi-Directional Restoration (KVKK-Aligned)**:
  - **50+ Turkish PII Categories**: Comprehensive recognition of personal identity identifiers: TCKN (Turkish Republic ID with Modulo-10/11 algorithmic validation), VKN (Tax ID), IBAN (MOD-97 checksum), Credit Card (Luhn check), Turkish person names and surnames (with polysemy disambiguation), GSM and landline phones, Addresses, Passports, Driving Licenses, Vehicle Plates, VIN, IMEI, MAC, IP Addresses, Port numbers, Age/Age ranges, Blood types, Health conditions, Passwords, API tokens, and OTP codes.
  - **Positive Private Date & URL Detection**: Accurately isolates private personal dates (birth dates, appointment schedules, billing due dates) without false positives on calendar holidays or campaign end dates. Recognizes private URLs containing sensitive authentication and session tokens.
  - **Bi-Directional Restoration with Morphological Vowel Harmony**: Automatically tracks an in-memory or serializable mapping dictionary to restore masked entities losslessly, adapting case and possessive suffixes in downstream text according to Turkish 2-way and 4-way vowel harmony (e.g. `{{AD_1}}'a` $\rightarrow$ `Ahmet Yılmaz'a`, `{{AD_1}}'e` $\rightarrow$ `Mehmet Demir'e`).
  - **Multiple Masking Modes**:
    - `placeholder`: Standard structured tokens (`{{AD_1}}`, `{{TCKN_1}}`, `{{IBAN_1}}`) designed for clean downstream processing and 100% reversible restoration.
    - `surrogate`: Statistically and morphologically valid Turkish synthetic replacements (e.g., valid surrogate TCKNs, realistic Turkish names, valid surrogate IBANs).
    - `tag`: Standard semantic entity tags (`[AD]`, `[TCKN]`, `[IBAN]`).
    - `anonymize`: Irreversible fixed-length masking (`***` or `[GİZLENDİ]`).
  - **Smart Non-PII Filtering**: Preserves corporate/functional emails (`info@`, `destek@`, `satis@`) and public dates while filtering individual personal data.
  - **Zero-Dependency Model2Vec Disambiguation**: Employs Akana's built-in 256-dim Turkish embeddings to contextually distinguish ambiguous names (*Deniz*, *Barış*, *Gül*, *Kaya*) from common nouns and verbs based on contextual semantic similarity.
  - **High Performance**: **>3,500 docs/sec** (< 280 µs per document) on CPU.
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
| **Turkish Embeddings** | 79 sent/s (BGE-M3) | **`20,013 sent/s`** | **253x faster** (2.5 MB 2-bit TurboQuant) |
| **Sentence Chunking** | N/A | **`1,120,000 words/sec`** | **Zero-Allocation Slicing** (<3 ms / doc) |
| **Semantic Chunking (TurboQuant)** | ~80 words/sec (Neural) | **`62,500 words/sec`** | **~780x faster** (~40 ms for 200 sents) |
| **Turkish PII Masking & Restoration** | N/A (Standard Regex / Spacy) | **`3,570 docs/sec`** | **Ultra-Fast KVKK Pipeline** (<0.28 ms / doc, 50+ entity types) |
| **Hardware Acceleration** | Pure Python loops | **StringZilla AVX-512 / AVX2 / NEON** | **Native SIMD Instructions** |

### Turkish Text Chunking Performance Benchmark (`scripts/benchmark_chunking.py`)

Evaluated on realistic multi-domain Turkish documents (50 paragraphs, ~200 sentences, ~2,500 words):

| Strategy | Algorithm | Latency per Doc | Throughput (Words/Sec) | Throughput (Docs/Sec) | Key Benefit |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`SentenceChunker`** | Rule-Based Sentence Packing | **2.23 ms** | **1,121,076 words/s** | **448.4 docs/s** | Ultra-low latency, token overlaps |
| **`SemanticChunker`** | 2-Bit Embedding Cosine Valleys | **39.85 ms** | **62,735 words/s** | **25.1 docs/s** | Natural topic boundary detection |
| **`SDPMChunker`** | Semantic Double-Pass Merge | **44.10 ms** | **56,689 words/s** | **22.7 docs/s** | Optimal chunk density & cohesion |

* 🚀 **High Throughput on CPU**: `SemanticChunker` processes **>62,000 words per second** on a single CPU core without PyTorch, CUDA, or heavy ONNX dependencies.
* 📦 **Zero External Infrastructure**: Built-in 2.5 MB TurboQuant embeddings execute entirely in-process with minimal memory footprint.

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

### Turkish Sentence Embedding Benchmark (Model2Vec TurboQuant 2-Bit)

Evaluated on standard Turkish Semantic Textual Similarity Benchmark (STSb) test sets:

| Model | Vocab | Size | Compression | STSb (gorkem) | STSb (emrecan) | Speed | Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| BGE-M3 (Teacher) | 250,002 | ~2,200 MB | 1x | **96.35%** | **79.57%** | 79 s/s | 1.0x |
| Our Model | **39,655** | **2.50 MB** | **880x** | 92.19% | 63.53% | **20,013 s/s** | **253x** |

* ⚡ **880x Model Compression:** Compressed from **~2,200 MB** to **2.50 MB** embedded directly into the binary with zero runtime dependencies.
* 🚀 **253x Speedup:** Delivers **20,013 sentences/sec** on CPU with high retention of semantic quality against the teacher model.

### Turkish PII Detection & Masking Benchmark (Saturday Labs & Belgin Datasets)

Evaluated across established Turkish PII and privacy benchmarks:

| Benchmark / Dataset | Evaluated Dimension | Akana Performance / Accuracy | Key Strength |
| :--- | :--- | :---: | :--- |
| **Saturday Labs Benchmark** (`cagrigungor/turkish-pii-masking-benchmark`) | **Recall (200 diverse samples)** | **94.1%** | **100% recall across 34 core categories** (TCKN, VKN, IBAN, Credit Card, Passports, Plates, Phone, Email, Coordinates, MAC, IMEI, Passwords) |
| **Belgin Dataset** (`negentropi/belgin-pii-dataset`) | **Private Date Recall** | **100.0%** (270/270) | Specific detection of birth dates, medical appointments, and billing due dates |
| **Belgin Dataset** (`negentropi/belgin-pii-dataset`) | **Public Date Preservation** | **100.0%** (300/300) | Zero false positives on calendar holidays and campaign deadlines |
| **Throughput & Latency** | **CPU Throughput (Single-Core)** | **`3,570 docs/sec`** | **0.28 ms** average latency per document with zero neural network overhead |

* 🛡️ **KVKK Compliance**: Algorithmic validation for Turkish IDs (TCKN, VKN, IBAN, Driving License, Plate) prevents false positives while capturing authentic identifiers.
* 🔄 **100% Reversible Restoration**: Restores anonymized text losslessly with Turkish morphological vowel harmony adjustment on inflected suffixes.

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

# 11. TurboQuant 2-Bit Turkish Sentence Embeddings & Semantic Similarity
vec = akana.embed("Türkiye'nin başkenti Ankara'dır.")
print(f"Vector dim: {len(vec)}")  # -> 256

# Semantic cosine similarity
score = akana.similarity("ev", "evler")
print(f"Similarity: {score:.4f}")  # -> ~0.9130

# Batch embedding
vecs = akana.embed_batch(["Merhaba dünya", "Hava bugün çok güzel"])
print(f"Batch size: {len(vecs)}")  # -> 2

# 12. Turkish Text Chunking for RAG & LLMs (Chonkie-Inspired)
# A. SemanticChunker: Splits at natural topic shifts using TurboQuant embeddings
semantic_chunker = akana.SemanticChunker(
    chunk_size=512,
    threshold_mode="percentile",  # "percentile", "similarity", "stdev", "iqr", "auto"
    threshold_value=0.75,
    min_chunk_size=20,
    min_sentences_per_chunk=1,
    use_smoothing=True,
)
chunks = semantic_chunker(
    "Kuantum mekaniği atom altı parçacıkları inceler... "
    "Fenerbahçe dün akşam derbide galip geldi... "
    "Geleneksel Türk mutfağında baklava çok meşhurdur..."
)
for chunk in chunks:
    print(f"Chunk ({chunk.token_count} tokens, chars {chunk.start_index}:{chunk.end_index}): {chunk.text}")
    print("Sentences:", chunk.sentences)

# B. SentenceChunker: Fast sentence packing with token constraints and overlaps
sentence_chunker = akana.SentenceChunker(chunk_size=256, chunk_overlap=30)
sentence_chunks = sentence_chunker("Metin 1... Metin 2...")

# C. SDPMChunker: Semantic Double-Pass Merge for dense, cohesive chunks
sdpm_chunker = akana.SDPMChunker(
    chunk_size=512,
    threshold_mode="percentile",
    threshold_value=0.75,
    merge_threshold=0.65,
)
sdpm_chunks = sdpm_chunker("Uzun doküman...")

# D. Dict / JSON serialization
chunk_dict = chunks[0].to_dict()
print(chunk_dict["text"], chunk_dict["start_index"], chunk_dict["end_index"])

# 13. Turkish PII Masking & Bi-Directional Restoration (KVKK-Aligned)
raw_text = (
    "Müşterimiz Ahmet Yılmaz, 10000000146 nolu TCKN ve TR33 0006 1005 1978 6457 8413 26 "
    "nolu IBAN ile 12.05.1988 doğumludur. Şifresi: Limon004_."
)

# A. Reversible Structured Placeholder Masking
masked = akana.pii_mask(raw_text, mode="placeholder")
print("Masked:", masked["masked_text"])
# -> "Müşterimiz {{AD_1}}, {{TCKN_1}} nolu TCKN ve {{IBAN_1}} nolu IBAN ile {{OZEL_TARIH_1}} doğumludur. Şifresi: {{SIFRE_1}}."
print("Mapping:", masked["mapping"])

# B. Lossless Restoration with Turkish Vowel Harmony
restored = akana.pii_restore(masked["masked_text"], masked["mapping"])
assert restored == raw_text

# Downstream inflected text restoration (vowel harmony preserved automatically):
inflected_text = "Ödeme {{AD_1}}'a yapılmış ve {{AD_1}}'ın dosyası onaylanmıştır."
print(akana.pii_restore(inflected_text, masked["mapping"]))
# -> "Ödeme Ahmet Yılmaz'a yapılmış ve Ahmet Yılmaz'ın dosyası onaylanmıştır."

# C. Realistic Turkish Synthetic Surrogates Mode
surrogate = akana.pii_mask(raw_text, mode="surrogate")
print("Surrogate:", surrogate["masked_text"])
# -> "Müşterimiz Can Demir, 52381940562 nolu TCKN ve TR92 0006 1000 ... doğumludur..."
```

---

## CLI Usage

The `akana` CLI supports direct text arguments or reading from file via `-f, --file`:

```bash
# Turkish PII Masking & Restoration (KVKK-Aligned)
akana pii-mask "Ahmet Yılmaz 10000000146 nolu TCKN ile başvurdu." --mode placeholder
akana pii-mask "Ali Kaya Kadıköy şubesine geldi." --mode surrogate
akana pii-mask -f confidential.txt --mode placeholder --json
akana pii-restore "Ödeme {{AD_1}}'a yapıldı." --mapping mapping.json

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

# Turkish Sentence Embeddings & Similarity
akana embed "Türkiye'nin başkenti Ankara'dır."
akana similarity "ev" "evler"

# Text Chunking (Semantic, Sentence, SDPM)
akana chunk --strategy semantic --chunk-size 512 --mode percentile -t 0.75 -f document.txt
akana chunk --strategy sentence --chunk-size 256 --overlap 30 -f document.txt
akana chunk --strategy sdpm --chunk-size 512 --json -f document.txt
```

---

## Rust Crate Usage (`akana-core`)

Add to `Cargo.toml`:
```toml
[dependencies]
akana-core = { version = "0.5", default-features = true }
```

```rust
use akana_core::chunking::{SemanticChunker, SentenceChunker, SDPMChunker, ThresholdMode};
use akana_core::grammar::TurkishGrammarChecker;
use akana_core::morphology::TurkishMorphology;
use akana_core::syntactic_morphology::TurkishSyntacticMorphology;
use akana_core::embeddings::TurkishEmbeddings;
use akana_core::pii::{TurkishPiiEngine, PiiMode};
use akana_core::phonology::to_turkish_lower;

fn main() {
    let lower = to_turkish_lower("İSTANBUL");
    println!("Lower: {}", lower);

    // 1. Semantic Chunking with TurboQuant Embeddings
    let chunker = SemanticChunker::new(512, ThresholdMode::Percentile(0.75));
    let chunks = chunker.chunk("Kuantum mekaniği... Fenerbahçe dün akşam...");
    for chunk in chunks {
        println!("Chunk ({} tokens, {}-{}): {}", chunk.token_count, chunk.start_index, chunk.end_index, chunk.text);
    }

    // 2. Turkish Sentence Embeddings (TurboQuant 2-Bit)
    let embeddings = TurkishEmbeddings::new();
    let vec = embeddings.embed("Türkiye'nin başkenti Ankara'dır.");
    println!("Embedding dim: {}", vec.len()); // 256
    let sim = embeddings.similarity("ev", "evler");
    println!("Similarity: {:.4}", sim); // 0.9130

    // 3. Grammatical Error Correction & Diagnostics
    let grammar_checker = TurkishGrammarChecker::new();
    let res = grammar_checker.check("Ali de geldi, Veli te geldi. Pazardan üç elmalar aldık.");
    println!("Corrected: {}", res.corrected);
    for f in &res.findings {
        println!("[{:?}] '{}' -> '{}'", f.category, f.original_text, f.replacement);
    }

    // 4. Standard Morphology
    let morph = TurkishMorphology::new();
    let parses = morph.analyze("kitabım");
    for p in parses {
        println!("{}", p.formatted);
    }

    // 5. Syntactic Expressive Morphology (Inflectional Groups)
    let syn_morph = TurkishSyntacticMorphology::new();
    let syn_parses = syn_morph.analyze("geldiğimizde");
    for p in syn_parses {
        println!("{}", p.formatted);
    }

    // 6. Turkish PII Masking & Bi-Directional Restoration (KVKK-Aligned)
    let pii_engine = TurkishPiiEngine::new();
    let pii_res = pii_engine.mask(
        "Ahmet Yılmaz 10000000146 nolu TCKN ile TR33 0006 1005 1978 6457 8413 26 nolu hesaptan işlem yaptı.",
        PiiMode::Placeholder,
    );
    println!("Masked: {}", pii_res.masked_text);

    // Lossless restoration with automatic vowel harmony adjustment
    let restored = pii_engine.restore_with_mapping(
        &format!("İşlem {}'a ait hesaba başarıyla yansıtıldı.", "{{AD_1}}"),
        &pii_res.mapping,
    );
    println!("Restored: {}", restored);
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
