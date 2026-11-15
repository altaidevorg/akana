# The TTS runtime moves into Akana: record (2026-11-14)

The owner decided on 2026-11-14 that the Turkish student TTS runtime moves from EchoCraft
(`crates/echocraft-student`) into Akana. The plan and its analysis are in EchoCraft's
`experiments/2026-11-14-inference-runtimes-plan.md` ("Akana analysis", "Owner proposal").
Branch `claude/tts-features-21ed1f`, from `05ba965`. Part A (this record) ships in 0.12.0;
Part B, the runtime itself, was built and then cancelled by the owner, and is kept unmerged on
branch `claude/tts-runtime`.

## Part A: the feature split

### Findings, re-checked on this branch

- **StringZilla** (C, built through `cc`) is used in `grammar/orthography.rs`,
  `normalization/spellcheck.rs`, `pii/{mod,morphology,normalize,secrets,spelled_numbers}.rs` and
  `style/cliches.rs`, as in 0.11.0.
- **`speech`** reads `normalization::TurkishNumberConverter` (`numbers.rs`, no StringZilla),
  `pii::gazetteer::PROVINCES` and `pii::is_place_name`.
- **`parser`** reads `pii::is_given_name`. **`ner`** also reads `pii`: `is_demonym`,
  `is_given_name`, `is_person_name`, `is_place_name`, `is_surname` (not in the 0.11.0 analysis).
- Nothing depends on `grammar` or `style`. `pii` depends on `embeddings` (`embedding_scorer`),
  `style` on `embeddings` and `readability`, `grammar` on `parser`, `chunking` on `embeddings`.
  `morphology` names `embeddings` only in doc links.
- **getrandom 0.3.4** comes only from ahash's default `runtime-rng` feature.
- `build.rs` hashes `src/morphology`, `src/phonology`, `src/tokenization`, parts of
  `src/embeddings`, `src/fingerprint.rs` and `Cargo.lock` into the encoder fingerprints. None
  of those files changed in Part A, so the fingerprints of a default build were those of
  `05ba965`, and Part A leaves `Cargo.lock` as it was. The 0.12.0 version bump changes the
  lock's hash, and with it the sparse, dense and hybrid fingerprints (`embeddings/spec.rs`; the
  analysis fingerprint does not hash the lock), as every release's bump does: indexes built
  before read `"compatible"`, not `"identical"`.

### Features

| Feature | Adds | Pulls in | Why a feature |
|---|---|---|---|
| *(always)* | `phonology`, `tokenization`, `morphology`, `syntactic_morphology`, `normalization` without the spell checker, `lexicon`, `analysis`, `readability` | pure Rust | what `speech` needs, and small modules with no C, no model and no expected growth |
| `speech` | `speech` | – | the selection the TTS runtime and the browser build name |
| `parser` | `parser`, `AkanaEngine`, `TurkishAnalysisDoc` | – | a capability downstream will select (phrase breaks); gating it later would break `default-features = false` users |
| `ner` | `ner` | – | a trained model (1 MB) |
| `embeddings` | `embeddings`, `chunking` | – | a model (5.3 MB); `chunking` cannot work without it |
| `spellcheck` | `normalization::spellcheck` | StringZilla | C |
| `pii` | `pii` | StringZilla, `embeddings` | C |
| `grammar` | `grammar` | StringZilla, `parser` | C |
| `style` | `style` | StringZilla, `embeddings` | C; a model (3.4 MB) |
| `pii-tagger` | `pii::tagger` | `pii`, RTen | opt-in, unchanged |
| `pii-download` | `pii::tagger::hub` | `pii-tagger`, ureq, sha2 | opt-in, unchanged |
| `full` (default) | every module feature above | | |

- `full` is "everything a build without features had in 0.11.0", not every feature: with
  `pii-tagger` and `pii-download` in it, the default build would gain RTen, ureq and a network
  path. They stay opt-in, as before.
- Finer gates (one per always-built module) were not taken: `analysis` (283 lines),
  `readability` (566 lines and a 37 KB list) and `syntactic_morphology` (445 lines) cost little to
  build, and each extra feature is another combination to keep warning-free.

### API compatibility

- Default features: every public path of 0.11.0 is unchanged. `pii::gazetteer` re-exports
  `GIVEN_NAMES`, `SURNAMES`, `POLYSEMOUS_NAMES`, `PROVINCES` and `DISTRICTS` from `lexicon`, so
  `akana_core::pii::gazetteer::PROVINCES`, `akana_core::pii::PROVINCES` and
  `akana_core::PROVINCES` all resolve.
- New public API: the `lexicon` module with the five sets and `is_place_name`, `is_given_name`,
  `is_person_name`, `is_surname`, `is_demonym` (crate-private in 0.11.0). It is not glob-exported
  at the crate root, so the root's names are unchanged.
- Reduced builds lose the root re-exports of the modules they leave out (`akana_core::TurkishPiiEngine`
  needs `pii`, and so on).
- `pii`'s name detector builds its given-name kinds from the same lists; a test
  (`pii::names::tests::lexicon_forms_equal_the_detectors_keys`) checks that its keys and
  surnames equal `lexicon`'s, which NER now reads without building the PII lexicon.
- On `wasm32-unknown-unknown` ahash keeps its fixed seeds (no `runtime-rng`), so the graph has no
  getrandom and needs no `wasm_js` backend or `RUSTFLAGS`. Native builds keep `runtime-rng`, as
  before. A browser page has no hash-flooding adversary that matters here.
- The PII data scripts (`scripts/names_from_pii.py`, `scripts/pii_data/{values,sample_real}.py`)
  read the moved files; their lists have the same sizes before and after (given 5,032, common
  given 526, surnames 197, foreign given 292, ambiguous 73, public figures 177; the
  name-dictionary writer 613 / 258 / 26).

### Checks (this laptop, rustc 1.99.0, `jobs = 1`)

| Check | Command | Result |
|---|---|---|
| No StringZilla | `cargo tree -p akana-core --no-default-features --features speech -i stringzilla` | not in the graph ("did not match any packages"); `cc` likewise, `--target all` |
| No getrandom in the browser | `cargo tree ... --features speech --target wasm32-unknown-unknown -e normal,build` | absent (natively it stays, from ahash's `runtime-rng`) |
| Browser build | `cargo check -p akana-core --no-default-features --features speech --target wasm32-unknown-unknown` | passes, no warnings, no `RUSTFLAGS` or getrandom feature; 2 min 01 s cold |
| Each feature alone | `cargo check -p akana-core --no-default-features [--features F] --lib` for none and each of the 10 features | all pass, no warnings |
| All targets | `cargo check --all-targets` with no features, `speech`, default, `--all-features` | all pass, no warnings |
| Phonemiser parity | `examples/phonemize_dump.rs` on EchoCraft's `corpus-w1.json` (40,914 texts), both profiles, release builds | default and speech-only builds byte-identical: 81,828 lines, SHA-256 `51a923bb…33f2cd`. A build of `05ba965` (HEAD, before the change) gives the same file. |
| Format | `cargo fmt --all -- --check` | clean (after rustfmt on the new code) |
| Clippy | `cargo clippy -p akana-core -p akana-cli -p akana-py --all-targets`; `-p akana-core --no-default-features --features speech --all-targets` | no warnings (akana-py needs `PYO3_PYTHON` set to the venv's Python) |
| Rust tests, default | `cargo test --locked -p akana-core -p akana-cli` | 548 passed, 0 failed, 6 ignored; every gated test file still runs its tests |
| Rust tests, speech only | `cargo test --locked -p akana-core --no-default-features --features speech` | 154 passed, 0 failed, 1 ignored (the golden rewriter); the speech goldens included |
| Golden vectors, release | `cargo test --release --locked -p akana-core --test golden_vectors` | 3 passed |
| Python | `maturin develop --release --locked`, `pytest` (venv in the worktree, Python 3.12) | 173 passed, 9 skipped |
| Fingerprint inputs | `git diff HEAD -- Cargo.lock src/morphology src/phonology src/tokenization src/embeddings src/fingerprint.rs build.rs` | unchanged |

- The speech-only example binary is 5.84 MB against 5.90 MB for the default build: the linker
  already drops modules the speech path does not reach, so the gain is the absence of C (and
  shorter builds), not size.
- CI (`.github/workflows/ci.yml`, job `features`) runs the graph check, the wasm32 check, each
  feature alone and the speech-only tests, with `RUSTFLAGS=-D warnings`. It has not run on
  GitHub (nothing is pushed); its shell steps were run locally.
- A first draft of its graph step used `cargo tree -i`, which exits 0 with "nothing to print"
  when a target filters the crate out; the step now searches the printed graph.

### Owner decisions after Part A (relayed 2026-11-14)

- A `tts` feature would be opt-in, not in `full` (as `pii-tagger`).
- The worktree's debug build output was deleted before Part B (3.4 GB).
- Part B was then built and cancelled (2026-11-15); its code, tests and record are on
  `claude/tts-runtime`, not in 0.12.0.
- 2026-11-15: Part A merges with the service-text verbaliser (`docs/tts-normaliser-2026-11-15.md`)
  and ships as 0.12.0.
