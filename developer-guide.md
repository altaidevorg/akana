# Akana Developer & Maintainer Guide

This guide covers local development, testing, the procedure for changing the analyzer or the encoders, building distribution wheels, and publishing releases to PyPI and crates.io.

---

## 1. Prerequisites

- **Rust**: the toolchain pinned in `rust-toolchain.toml` (1.99.0), which `rustup` picks up in this directory. Analyses and vectors depend on the Unicode tables of the standard library, so builds use that version.
- **Python**: Version $\ge 3.10$
- **uv**: Modern, fast Python package and project manager (`curl -LsSf https://astral.sh/uv/install.sh` or `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`)
- **maturin**: PyO3 build backend (`uv pip install maturin`)

---

## 2. Local Development & Testing

### A. Run Rust Unit & Integration Tests
```bash
# Core and CLI tests (the Python crate is tested through pytest)
cargo test --locked -p akana-core -p akana-cli

# The release profile, which wheels are built with
cargo test --release --locked -p akana-core -p akana-cli
```

### B. Compile Python Extension in Editable Mode
```bash
# Compiles the Rust PyO3 crate and installs into local virtualenv
uv run maturin develop --release
```

### C. Run Python Pytest Suite
```bash
# Run all Python API tests (tests/test_golden_vectors.py pins the Python vectors)
uv run pytest
```

### D. Run Benchmarks
```bash
# Benchmark vs Zeyrek / Zemberek morphology
uv run python benchmarks/benchmark_vs_zeyrek.py

# Benchmark StringZilla SIMD & Zero-Regex Optimizations
uv run python benchmarks/benchmark_stringzilla_optimizations.py

# Retrieval on RagTurk and Turkuaz, and the README's table (fails if a dataset is missing)
uv run python scripts/benchmark_rag_real_data.py --dataset all --readme

# The analyzer and the dependency parser on held-out UD Turkish (after fetch_morph_sources.py)
uv run python scripts/ud_eval.py
uv run python scripts/parser_ud_eval.py
```

---

## 3. Changing the Lexicon, the Analyzer or the Encoders

Sparse and hybrid indexes store `akana.encoder_info()` and are checked against it
(`embeddings/spec.rs`), so a change that moves analyses or vectors is made on purpose and
measured. Several committed files are derived from the analyzer and must be regenerated after
any change to the lexicon, the morphotactics or the ranking, in this order:

1. **Lexicon**: edit `scripts/lexicon/akana-*.dict`, then `python scripts/build_full_lexicon.py`
   (it fails on a removal or respelling line that matches nothing). For abbreviations, run
   `python scripts/audit_abbreviations.py` before and after.
2. **UD inputs of the prior**: `python scripts/fetch_morph_sources.py` once, then, with the new
   build importable, `python scripts/pos_twins.py` (and `scripts/prior_general_types.py` if the
   word cutting changed). CI checks that both reproduce.
3. **Prior**: `cargo run --release -p akana-core --example morph_prior -- --types
   data/morphology/prior_train_types.tsv.gz --out crates/akana-core/src/morphology/prior.tsv.gz`.
   CI checks that it reproduces.
4. **Sparse term tables**: `AKANA_GOLD_WRITE=sparse_tables cargo test -p akana-core --lib
   committed_tables` (stop lemmas and lemma weights).
5. **Fixtures**: the tests that record analyses fail on any change by design. After reviewing
   the diff, rewrite them with `AKANA_GOLD_WRITE=<suites>` (`morphology_snapshot`,
   `review_gold`, `morphology_gold`, `downstream`, `pii_view`, comma-separated) and the golden
   vectors with `UPDATE_GOLDEN=1 cargo test -p akana-core --test golden_vectors`. Give the
   diffs in the commit.
6. **Bump class**: dump the vectors of the old and the new build with
   `python scripts/encoder_churn.py dump` (it needs the court sample of
   `scripts/morph_corpus_sample.py` and the cached RagTurk and Turkuaz sets) and run
   `encoder_churn.py compare OLD NEW --declared <class>`. Patch: no vector moved; minor: at most
   5 % of each corpus's documents and 0.005 stale MRR@10; major otherwise, which needs a new
   `SPARSE_SPEC` major. `tests/data/embeddings/golden-1.0.jsonl` holds the vectors of the
   release that froze spec 1.0 and is never rewritten after that release: CI fails when more
   than 5 % of them move within the spec.
7. **AI detector**: its drift canary fails when its inputs move. Rewrite it once per commit with
   `AKANA_AI_DETECTOR_CANARY="<why>" cargo test -p akana-core --lib -- --ignored
   rewrite_drift_canary`.

The encoder fingerprints also hash the Rust source of the analysis and encoder path and
`Cargo.lock` (`crates/akana-core/build.rs`), so any build from other inputs reports
`"compatible"` rather than `"identical"`.

---

## 4. Building Distribution Packages

### A. Python Binary Wheels & Source Distribution (sdist)
```bash
# Build abi3 wheels into dist/
uv run maturin build --release --locked --out dist/

# Build source distribution (.tar.gz), which carries rust-toolchain.toml
uv run maturin sdist --out dist/
```

---

## 5. Publishing Releases

### A. Publishing to PyPI via GitHub Actions (Recommended)

The release workflow (`.github/workflows/release.yml`) runs on a tag `v<version>`:

1. it checks that the tag names the version of `Cargo.toml` and `pyproject.toml` (bump both together, with `Cargo.lock`);
2. it runs the Rust tests in the release profile;
3. it builds wheels for Linux (`x86_64`), Windows (`x86_64`) and macOS (`aarch64`), and the sdist;
4. it installs each wheel and runs the golden-vector and Python API tests against it;
5. only then, and only for a tag, it publishes to PyPI.

A manual run builds and tests but publishes nothing. Configure the `PYPI_API_TOKEN` secret in the repository settings, then tag and push:
```bash
git tag v0.9.0
git push origin v0.9.0
```

### B. Publishing to PyPI Manually

#### Option 1: Using `maturin publish`
```bash
# Set your PyPI token
export MATURIN_PYPI_TOKEN="pypi-..."

# Publish wheels from dist/
uv run maturin publish --out dist/
```

#### Option 2: Using `uv publish`
```bash
export UV_PUBLISH_TOKEN="pypi-..."
uv publish dist/*
```

### C. Publishing Rust Crates to crates.io
```bash
# Login to crates.io
cargo login <your-crates-io-token>

# Publish core library
cargo publish -p akana-core

# Publish CLI binary
cargo publish -p akana-cli
```

A build from the published crate has no workspace `Cargo.lock` and resolves its own dependencies: its fingerprints differ from the wheels', and indexes built with the wheels read as `"compatible"` there.
