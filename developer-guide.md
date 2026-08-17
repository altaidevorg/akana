# Akana Developer & Maintainer Guide

This guide covers local development, testing, compiling native extensions, building distribution wheels, and publishing releases to PyPI and crates.io.

---

## 1. Prerequisites

- **Rust**: Version $\ge 1.75$ (`rustup update stable`)
- **Python**: Version $\ge 3.10$
- **uv**: Modern, fast Python package and project manager (`curl -LsSf https://astral.sh/uv/install.sh` or `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`)
- **maturin**: PyO3 build backend (`uv pip install maturin`)

---

## 2. Local Development & Testing

### A. Run Rust Unit & Integration Tests
```bash
# Run all workspace tests
cargo test --workspace

# Run core and CLI package tests
cargo test -p akana-core -p akana-cli
```

### B. Compile Python Extension in Editable Mode
```bash
# Compiles the Rust PyO3 crate and installs into local virtualenv
uv run maturin develop --release
```

### C. Run Python Pytest Suite
```bash
# Run all Python API tests
uv run pytest
```

### D. Run Benchmarks
```bash
# Benchmark vs Zeyrek / Zemberek morphology
uv run python benchmarks/benchmark_vs_zeyrek.py

# Benchmark StringZilla SIMD & Zero-Regex Optimizations
uv run python benchmarks/benchmark_stringzilla_optimizations.py
```

---

## 3. Building Distribution Packages

### A. Python Binary Wheels & Source Distribution (sdist)
```bash
# Build multi-platform abi3 wheels into dist/
uv run maturin build --release --out dist/

# Build source distribution (.tar.gz)
uv run maturin build --sdist --out dist/
```

---

## 4. Publishing Releases

### A. Publishing to PyPI via GitHub Actions (Recommended)

Akana includes an automated GitHub Actions release workflow (`.github/workflows/release.yml`) that builds binary wheels for Linux (`x86_64`, `aarch64`), Windows (`x86_64`), macOS (`x86_64`, `aarch64`), and source distribution, then publishes to PyPI automatically.

1. Configure `PYPI_API_TOKEN` secret in your GitHub repository settings:
   - Go to `https://github.com/altaidevorg/akana/settings/secrets/actions`
   - Add secret named `PYPI_API_TOKEN` with your PyPI token.
2. Tag and push a release:
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
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
