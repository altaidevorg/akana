# The verbaliser for service text: record (2026-11-15)

The brief (from the EchoCraft session, at the owner's request) asked for the TTS front end,
`akana.verbalize`, to read service text as Turkish readers do: behaviours B1–B11, 65 cases (53
`must`, 10 `guard`, 2 open decisions), precision first. EchoCraft's service voice is trained on
`tts-v2` tokens. Branch `claude/tts-normaliser`, from `main` at `05ba965`; nothing committed.

## What changed

All new readings apply with the word rules only: `verbalize` (Python `akana.verbalize`) and
profile `tts-v2`, now **`tts-v2.1`**. `tts-v1` (`verbalize_with(text, false)`) keeps every
earlier reading.

| | Reading | Guards (stay as before) |
|---|---|---|
| B1 telephone | groups with a comma between them; also `5321234567` (mobile, 10 figures from 5), `0850 222 0 600` (11 figures from 0 in any groups), `444 0 333`, and 10 figures from 2, 3, 4 or 8 next to `telefon`, `tel`, `cep`, `gsm`, `faks`, `hat(t…)` | a quantity or count noun after it; six or more trailing zeros (`5000000000 kişi`); `hatlarında` |
| B2 identifier | five figures or more, no thousands dot, one figure at a time, when `kod…`, `numara…`, `nolu`, `şifre`, `parola`, `iban` stands before it, or one of them with a suffix after it, or a record word (`sipariş`, `takip`, `doğrulama`, `dosya`, `poliçe`, `referans`, `müşteri`, `abone`, `posta`, `rezervasyon`, `kimlik`, `kargo` …) before it or before such a word after it (`76950 posta kodlu`) | a currency, unit, percent sign or scale word after it; a count noun after it (`12500 başvuru`, `12500 puan`); a bare `kod` after it (`12345 kod satırı`); `hesap` and `fatura` with a case ending (`hesabınıza 15000`); a plural (`siparişleri`) |
| B3 letters | an initialism the lexicon spells: one Turkish letter name each, K *ka*, Q *kû*; the written suffix picks *ka*/*ke* and *he*/*ha*; a single capital consonant used as a label (`G bloğu`, `Plan B.`, `C'den`); `SMS` esemes | initialisms read as words (`NATO`, `ODTÜ`, `UYAP`, `CNN` sienen); a word in capitals; vowels (`O`, `A blok`); initials (`M. Kemal`); elisions (`N'aber`, `M'Vila`) |
| B4 codes | capitals and figures interleaved (`X7K9QZ`), or letters and five figures or more (`TR78240865`): letters by name, figures one by one; plate-shaped codes `34 ABC 123` (four figures in two pairs) | model names (`MP3`, `G20`, `COVID-19`, `A4`); units, currencies, words and months in capitals (`16 GB 2400`, `20 YIL 10 AY`, `12 OCA 2026`) |
| B5 IBAN | `TR` + check figures as a number, groups of four one figure at a time, last two as a number, commas between | |
| B6 minus | (already right) | |
| B7 slash | `7/24` yedi yirmi dört | file numbers keep *bölü*; fractions and dates as before |
| B8 dose | `2x1`, `2x500 mg` after `günde` or before `tablet`, `kapsül`, `şeklinde` … : iki çarpı bir | dimensions (`3x4 metre` üçe dört) |
| B9 decimals | `,5` after a whole number *buçuk* (`buçuğu` before a vowel suffix); a dot and one or two figures is a decimal, after `%` too | `0,5` sıfır virgül beş; a version's or section's name before it (`Python 3.12`, `HDMI 2.1`, `madde 12.5`); two dots (`1.2.24`) |
| B10 money | no leading zero (`039,50 TL`); `lira'lık` liralık (a short list of common nouns) | |
| B11 | `16.10 tarihinde` on altı ekim | `16.10'da`, `saat 16.10` (times) |

- **The phonemiser** (`tts-v2` only): a word inside a spelled initialism, a letter label or an
  identifier is a content word, never the clitic `de` or the conjunction `ve`, and gets no
  optional pause. Without it `KDV` read *ka de _! ve* with *de* and *ve* unstressed. The brief
  expected no phonemiser change; this one is needed for the readings to sound right.
- **Fixed:** `3×4` (the multiplication sign, two bytes) made the verbaliser panic in both
  profiles (`main`: `RuntimeError ... not a char boundary`).
- **Version:** `tts-v2.0` → `tts-v2.1`. `tts_v2_golden.jsonl` and `verbalize_golden.jsonl`
  were rewritten. 9 of the 100 `tts-v2` rows changed tokens, all from `verbalize_inputs.txt`;
  the 60 plain-text sentences did not. `tts_v1_golden.jsonl` is untouched.
- **Fixtures:** the brief's 65 cases are `tests/data/speech/service_cases.jsonl`, checked by
  `speech_tests.rs::service_cases_read_as_expected`. The guards found here are in
  `verbalize.rs::service_conventions_and_their_guards`.

## Checks

| Check | Result |
|---|---|
| `score_cases.py`, `main` (05ba965, built here) | must **10/53**, guards 10/10, median 22 µs |
| `score_cases.py`, branch | must **53/53**, guards **10/10**, median 20 µs |
| `cargo fmt --all -- --check`; `cargo clippy -p akana-core -p akana-cli -p akana-py --all-targets` | clean, no warnings |
| `cargo test --locked -p akana-core -p akana-cli` | 549 passed (546 + 3 new), 0 failed, 6 ignored |
| `cargo test --release --test golden_vectors` | 3 passed |
| `pytest` (`maturin develop --release`, Python 3.12) | 173 passed, 9 skipped |
| `tts-v1` unchanged | verbaliser output and tokens identical to `main` on 17,916 service lines and 40,914 corpus sentences |
| Word-rule readings that changed | 9,040 of 17,916 service lines; 252 of 40,914 corpus sentences (initialisms and letter labels: the corpus is already spoken text) |

- **Precision review.**
  - The changed readings were read: every corpus change, the service changes in random samples,
    and every identifier reading next to a money or count word (1,158).
  - Rules were narrowed where they misread: plates, record words with a case ending, a bare `kod`
    after a number, a ten-figure quantity, and elisions.
  - The service lines are EchoCraft's pool of a third-party dataset, used for local evaluation
    only and not committed.
- **Latency** (`bench_verbalize.py`): eight alternating processes, `main` and branch (ABBA
  twice); each timed the 65 cases ×20 and the 17,916 service lines.
  - Branch ÷ `main` per adjacent pair, cases: 1.00, 0.99, 0.91, 0.97.
  - Service lines: 1.04, 1.17, 0.99, 1.26 (the last pair noisy: p90 39 µs).
  - The cases are no slower, and service lines about +10% at the median (within the +20%
    target). Per-sentence medians: 13–17 µs on the cases, 19–25 µs on the service lines.

## The owner's decisions (2026-11-15, relayed)

- **PDF and QR:** Turkish letter names, for consistency: *pe de fe*, *kû re*, never the word
  *küre*. My recommendation had been the mixed readings (*pe de ef*, *kü ar*); the owner chose
  consistency ("pe de fe is more common").
- **H in initialisms:** *he*, never *ha* ("ha is considered to be inappropriate for professional
  settings"): THY *te he ye*, HGS *he ge se*, HGS'ye *he ge seye*.
  - The initialism rule no longer turns a last *he* into *ha* before a back-vowel suffix. The
    suffix follows *he*. No golden reading changed.
- **K** stays *ka* in initialisms, as in the brief.
- **Fixture:** the addendum's five `must` cases (`cases-addendum-1.jsonl`) replace the two `info`
  cases in `service_cases.jsonl`, which now has 68 cases (58 must, 10 guards).
  - `score_cases.py`: the addendum alone, must 5/5.
  - The two files concatenated: must 58/58, guards 10/10.

## For the owner
- **tts-v1:** it keeps every earlier reading (digits included), so its voice still reads phone
  numbers without pauses and identifiers as numbers. Extending the conventions to it would change
  `tts-v1`'s output on text with digits; its frozen plain-text output would stay.
- **Provenance note:** `verbalize.rs`'s header and `docs/phonemizer-tts-v2.md` name another
  normalisation project as the black box used in stage 2. The brief asks that new code not name
  such projects; nothing new does, and the existing note was left as it was.
- **A trap found on the way (Part A record):** `maturin build` with another checkout's sources
  and a shared `CARGO_TARGET_DIR` reuses that target's `akana-core`. Cargo hashes path packages
  relative to the workspace, and the extracted files looked older. The Part A claim "a build of
  05ba965 gives the same phonemiser output" was therefore re-checked with `main` built in its own
  target. Both profiles' `phonemize_detailed` and `verbalize` are identical between `main` and the
  feature-split build, on 40,914 corpus sentences and 17,916 service lines.
