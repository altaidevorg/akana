# Turkish phonemiser, stage 1: the `espeak-compat` profile (2026-11-07)

Findings and handoff for `akana_core::speech` (Python: `akana.phonemize`, `akana.phonemize_detailed`).
Brief: EchoCraft `experiments/2026-11-07-akana-phonemiser-brief.md`. Reference fixtures: EchoCraft
`output/diagnostics/20261107-akana-phonemiser-fixtures-v3` (eSpeak NG 1.52.0, research-hook runtime v4,
rate 175).

## Summary

- **What it does:** Turkish text in, the token stream eSpeak NG gives EchoCraft's student TTS out:
  `ph` (eSpeak mnemonic), `stress` (0–7), `word`, `lengthen`, `mark`, plus `type` and `ipa`. Every
  word carries the provenance of its stress and of its segments.
- **Parity, test split (300 sentences, reported once, design frozen):** phoneme sequence exact on
  **99.67%** of sentences (target ≥ 98%); per token `ph` **99.993%** (≥ 99.8%), `stress` **99.83%**
  (≥ 99.5%), `word` 99.987%, `lengthen` and `mark` 99.993% (≥ 99.9%). 98.67% of sentences are exact
  in every field. All acceptance targets are met.
- **Words outside the profile's lexicon** (letter rules and suffix model alone): 98.9% of the 268 such
  test words are exact; in 5-fold cross-validation on train they get stress right per token 99.64% of
  the time.
- **Latency:** median **7–10 µs** per sentence of 50–70 characters, 99th percentile 14–20 µs (budget:
  2 ms and 10 ms), release build, warm, one thread of this laptop; the data loads in 11 ms. That is
  about **280,000× faster than real time** (speech at eSpeak's rate of 175 words per minute), against
  0.034–0.095 of real time for the eSpeak subprocess today.
- **Not covered:** symbols (`$ € = + &`), letter-name spelling beyond simple cases, and eSpeak's own
  number pronunciations and compounds; the catalogue below lists them for stage 3.
- **Next step: stage 2a, lexical stress** (last section).

## Interface

```rust
use akana_core::speech::{Phonemizer, Profile, Utterance};
let p = Phonemizer::new(Profile::EspeakCompat);   // data loaded once per process
let mut u = Utterance::default();
p.phonemize_into("Bir ağaçta gül de biter, diken de.", &mut u)?;   // reuses u's buffers
for t in u.espeak_tokens() { /* t.ph.symbol(), t.stress, t.word, t.lengthen, t.mark, t.ipa() */ }
for w in &u.words { /* w.span (bytes), w.reading, w.phones, w.stress_source.label(), w.segment_source.label() */ }
```

```python
import akana
akana.phonemize("Bir ağaçta gül de biter, diken de.")            # fixture schema: list of dicts
akana.phonemize_detailed("...")                                     # words, provenance, warnings; char offsets
```

- **Provenance labels.** Stress: `lexicon:espeak-compat`, `espeak-compat-rule:suffix-model(-ending)`,
  `espeak-compat-rule:apostrophe-ending('la)`, `espeak-compat-rule:apostrophe-final`, `default-final`,
  `espeak-compat-rule:spelled`; plus `promoted` (stress added at a clause's end) and `tonic`.
  Segments: `lexicon:espeak-compat`, `espeak-compat-rule:letters`, `espeak-compat-rule:letters+stress`,
  `espeak-compat-rule:spelled`; `normalized: "number"` for the words of a number.
- **Errors:** never panics; texts over 1 MiB return `PhonemizeError::TooLong`; characters the profile
  does not read (symbols, emoji) are skipped and reported as warnings with their spans.
- **Determinism:** no randomness, clocks or locale; the same text and build give the same tokens.
  `Profile::version()` is `espeak-compat-1`; the profile's output changes only with a version bump.

## How it works

1. **Text → words, clauses, pauses** (`speech/text.rs`). Words are letter runs with apostrophe groups
   and the izafet `-i`; a small letter before a capital splits a word (`nasıI` → `nası I`); `ister
   misin(iz)` and `ne yaptın` are one word. Clause punctuation gives `_: _`; quotes and brackets `_: _:`;
   a word with an apostrophe is followed by `_: _:` (not at a clause end, not `Allah'a`); `ve veya
   yoksa veda ama ki kime` take `_!` unless they end their clause; a full stop before a small letter
   is no boundary. Marks follow EchoCraft's `clause_marks` regex, including its quirks (a decimal
   comma counts; with an ellipsis in the last two clauses the last clause keeps the previous number).
   Digits are read as Turkish numbers (`20:00` yirmi sıfır sıfır, `07.30` sıfır yedi nokta otuz,
   `%25` yüzde yirmi beş, `1923'te` … te `_: _:`).
2. **Words → phonemes** (`speech/letters.rs`, `speech/model.rs`). The lexicon (33,501 words) has eSpeak's
   stress for every word it saw and eSpeak's phonemes for the 269 where the letter rules differ. The
   letter rules: lax vowels in closed and word-final syllables, `&` before a tautosyllabic m n l, `*`
   between vowels, clear/dark l by the following vowel (by the preceding one in a coda and before an
   apostrophe), `J` before front vowels, `c` in `-ken`, ğ as length after back vowels and `j` after
   e/i, VğV as one long vowel, doubled stops as one long stop, â î û as front for l and g. Two choices
   depend on stress and are made after it (g/J in a stressed final syllable; lax e after c/ç).
3. **Stress position.** The lexicon; for other words a suffix model: the longest ending shared with
   lexicon words decides, through a table of 1,508 endings, each holding the position a word with that
   ending and an unseen letter before it gets (the mean of the ending's continuations, interpolated
   Witten–Bell style with the shorter ending's; pruned where an ending predicts what its parent does).
   After an apostrophe the last syllable is stressed except before `'la 'le 'dir`-type endings.
4. **Levels** (`speech/mod.rs`, all observed): 6 on a word's primary, 7 on the clause's last primary; 3
   for `ile yoksa kime` and spelled letters; a clause-final `ben sen biz siz bu o şu bir` is stressed;
   a clause with no stress gets one. Unstressed syllables: before the stress, the first is 0 after a
   level-1 syllable and 1 otherwise, the rest 0; after it, `[1]`, `[1, 1]` (`[1, 0]` when the word ends
   in a consonant, before `_!`, or in a clause that ends in an unstressed word), `[0, 0, 1]`, `[0, 0, 1,
   0]`; open function words (`bu o şu de da mi ki ta ve`) stay 1 after a 1 mid-clause. Consonants and
   pauses take the level of the next vowel in their clause; a word-final l before a vowel-initial word
   loses its coda mark.

**Data** (`speech/espeak_compat.tsv.gz`, 132 KB, built by the ignored test `build_espeak_compat_data`):
the train split's words (14,828 sentences) and 15,817 more words of the Common Voice sentence
collector, from sentences in none of the fixture splits, run through the same eSpeak build as a word list
(`scripts/speech/espeak_compat_wordlist.json`, made by `scripts/speech/make_espeak_compat_wordlist.py`;
88 s of eSpeak). The texts are CC0; only words ship, with eSpeak's outputs for them.

**Clean room.** Built only from eSpeak NG's observed outputs (the fixtures, the word-list run), the
brief's behavioural description and the phonology literature it cites. No eSpeak NG source, rule or
data file was read, copied or translated. Every rule above names its evidence in the code.

## Parity report

**Method.**

- Rules and tables were tuned on `train`; `validation` was the check during development.
- `test` was scored **once**, after the design was frozen. One later fix is reported separately and
  labelled post hoc: e stays tense before the j of ğ in the c/ç rule. Its evidence was in train
  (122 tense against 10 lax), and it was confirmed by cross-validation.
- Per-field agreement pairs tokens along a longest common subsequence of the phoneme symbols
  (`examples/espeak_compat_parity.rs`); tokens on unpaired stretches count as disagreeing.
- **Unseen words** are words whose reading is not in the train split. **Outside the lexicon** are words
  the profile's lexicon lacks, so the letter rules and the suffix model read them.
- **Cross-validation** (ignored test `cross_validate_espeak_compat`): 5 folds of train sentences. The
  tables are rebuilt from four folds plus the word list (which holds no train word), and the fifth fold
  is read. Its new words therefore test the rules alone.

| | sentences | all fields exact | ph sequence exact | ph | stress | word | lengthen | mark |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| acceptance (brief §6) | | | ≥ 98% | ≥ 99.8% | ≥ 99.5% | ≥ 99.9% | ≥ 99.9% | ≥ 99.9% |
| train (in the lexicon) | 14,828 | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| validation | 200 | 99.50% | 100% | 100% | 99.949% | 100% | 100% | 100% |
| **test, reported once** | 300 | **98.67%** | **99.67%** | **99.993%** | **99.829%** | **99.987%** | **99.993%** | **99.993%** |
| test, post hoc (ğ fix) | 300 | 99.00% | 100% | 100% | 99.836% | 99.993% | 100% | 100% |
| 5-fold CV on train (rules on unseen words) | 14,828 | 98.30% | 99.68% | 99.987% | 99.644% | 99.987% | 99.986% | 99.987% |

Before the ğ fix the cross-validation gave 97.98% exact, 99.34% ph sequence and 99.632% stress.

**Words unseen in train** (the brief's subset):

| | words | words exact | tokens | ph | stress | word | lengthen | mark |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | 295 | 99.66% | 2,395 | 100% | 99.79% | 100% | 100% | 100% |
| test, reported once | 442 | 99.10% | 3,843 | 99.974% | 99.688% | 99.974% | 99.974% | 99.974% |
| test, post hoc | 442 | 99.32% | 3,843 | 100% | 99.714% | 100% | 100% | 100% |

Many of these words are in the word-list part of the lexicon. Of the test words **outside the whole
lexicon**, 98.88% are exact (268 words; 99.25% post hoc). The cross-validation row above is the clean
measure of the rules on words they never saw.

**Confusions (test, reported once).**

- `ph`: one, `e` → `E` (*diyebileceğim*; fixed post hoc).
- `stress` (eSpeak → Akana): 1→6 ×6, 6→0 ×6, 0→1 ×5, 1→0 ×3, 6→1 ×3, 0→6 ×2.
- They come from the four differing sentences: stress positions the suffix model got wrong (class 1)
  with their knock-on effect on the next word (class 2), and the *insan gibi* join (class 3).

**Probes** (51 hand-written screen-reader strings; for behaviour, not scored): 33 (64.7%) exact. The
18 that differ fall in the catalogue's classes 5–8: numbers, symbols, spelling and brackets.

## Divergence catalogue

| # | class | examples | where | decision |
|---|---|---|---|---|
| 1 | Stress position of words outside the lexicon: eSpeak's letter rules mark syllables unstressed in ways the suffix model only approximates (misfires such as *-tım*, *-la-*, *-sa*, *-dır* inside a word) | *gözümle* (eSpeak *göz*), *tokat* (*to*), *sağaltır* (final; Akana *-tır* pre-stress), *dayım* | test 3 words, validation 2; CV 0.36% of stress tokens | **Keep.** It is a parity gap only. eSpeak's misfires contradict the literature (EchoCraft audit), and stage 2 replaces this stress model. More eSpeak word lists would shrink the gap. |
| 2 | Knock-on 0/1 levels: a wrong stress changes the next word's first unstressed syllable (0 after a 1, else 1) | *Güğümün içindeki* | follows class 1 | Keep (follows 1). |
| 3 | *insan gibi* joined into one word by eSpeak (seen only in test) | sr0109 | test 1 | **Keep.** One example, outside train; not tuned on test. |
| 4 | e before the j of ğ after c/ç | *diyebileceğim* | test 1 | **Fixed post hoc** (train evidence 122:10; CV +0.3 points exact). |
| 5 | Numbers: eSpeak reads digits with its own number words: tense vowels (*yirmi* `j i R m i`, *iki* `i c i`, *bin* `b i n`), compounds with secondary stress (*yirmibeş* `j i R3 m3 i3 b6 E6 S`, *dokuzyüz*), `_! _ _|` pauses inside long numbers, *virgül* at level 2 | probes num-01…12 | probes | **Keep the plain reading.** The words are eSpeak's, with text pronunciations, as separate words. The student never trained on eSpeak's digit readings (the corpus has no digits). Stage 3 designs number reading for the screen reader. |
| 6 | Symbols read by eSpeak: `&` *ve*, `$` *dolar*, `€` *avro*, `=` *eşit*, `+` *artı*, `www` *dabılyu…* | sym-01…03 | probes | **Stage 3.** Skipped with a warning today. |
| 7 | Spelling details: `TBMM` has a tense *be*; `ğ` alone is two words *yumuşak ge*; `II.` has 3 on its first vowel | abbr-01, sr-02, num-08 | probes | Stage 3 (spelling mode, Roman numerals). |
| 8 | Post-stress pattern before brackets: eSpeak demotes the last syllable before `(…)`, as before `_!` | sym-04 *sessizce* | probes | Keep until there is train evidence. |
| 9 | A clause with no stressed word: which word eSpeak promotes (*Ben mi?* → *ben*, *Bu o mu?* → *mu*) | 10 train sentences | rule from few cases | Keep; revisit with more eSpeak output if it matters. |
| 10 | Lexical exceptions that letters cannot predict (dark l in *zalim*, lax e in *esen*, tense *iyi*, *idi*) | train | in the lexicon | Exact for seen words; rules otherwise. |

Reproduced quirks that are not divergences: the decimal comma counting as a mark; the ellipsis clause
numbering; the `_: _:` after apostrophe words replacing `_!`; `Allah'a` without a pause; the joins and
case splits; dark l in all back-vowel loans (*kabul*); the circumflex dropped except for l and g.

## Latency

`cargo run --release -p akana-core --example espeak_compat_bench`, warm, 20 repetitions of each of the
15,328 corpus texts, one thread:

| | median | p90 | p99 | max |
|---|---:|---:|---:|---:|
| 50–70 characters (1,641 texts; 8.2 words, 2.8 s of speech at 175 wpm) | 7.3–10.1 µs | 9–14 µs | 14–20 µs | 0.1–0.4 ms |
| all texts | 2.9–3.1 µs | 9–13 µs | 13–18 µs | 0.14–1.1 ms |

The ranges span three runs; the rare maximums are scheduling outliers on a busy laptop. Loading the data
takes 11–12 ms at `Phonemizer::new`. The Python binding adds a
JSON round trip (tens of µs).

## Reproduce

```bash
# data (needs the fixtures and the word list's eSpeak run)
AKANA_ESPEAK_FIXTURES=DIR AKANA_ESPEAK_EXTRA=WL/espeak-plan.jsonl cargo test --release -p akana-core --lib -- --ignored build_espeak_compat_data
# cross-validation
AKANA_ESPEAK_FIXTURES=DIR AKANA_ESPEAK_EXTRA=WL/espeak-plan.jsonl cargo test --release -p akana-core --lib -- --ignored cross_validate_espeak_compat --nocapture
# parity report and latency
cargo run --release -p akana-core --example espeak_compat_parity -- --fixtures DIR --splits validation,test --out report.json
cargo run --release -p akana-core --example espeak_compat_bench -- --fixtures DIR
# tests (fixture tests skip without AKANA_ESPEAK_FIXTURES)
cargo test -p akana-core --test speech_tests
pytest tests/test_speech.py
```

The word list's eSpeak run (`espeak-plan.jsonl`, sha256 `5a335a3a…c11f0`) is reproducible with
EchoCraft's `espeak_fixtures.py` on `scripts/speech/espeak_compat_wordlist.json`. A copy was left in the
EchoCraft session notes folder (`akana-phonemiser-wordlist-v1/`).

## For the owner

- **Shipped data:** the profile ships a table of 33,501 CC0 words with eSpeak's stress and phonemes,
  derived from eSpeak's outputs only (THIRD_PARTY_NOTICES §4). The brief allowed a word-level table;
  please confirm you are happy to ship it in the library.
- **Golden tests:** 60 CC0 sentences with eSpeak's tokens are committed as golden tests
  (`tests/data/speech/`).
- **EchoCraft integration:** the `--frontend akana` switch, set-R CER (the budget is 0.3 points over
  0.51%), front-end latency and a labelled old/new listening page are EchoCraft's.
  `akana.phonemize(text)` returns exactly the token dicts `synth.plan` builds.

## Next step: stage 2a, lexical stress

eSpeak's stress, as the fixtures show it, is a set of letter rules: final stress unless a suffix-like
letter pattern marks syllables unstressed, which includes misfires (*başladı*, *ettim*, *dostum*,
*gözümle*). The teacher's F0 follows the literature instead for initial-accent adverbs (67% first-syllable
peaks against 38% for controls) and kinship terms (EchoCraft audit, Part 2).

- **Profile:** `literature-v1`, one mechanism, a new profile version and fixtures.
  1. An exceptional-stress lexicon with a source per entry: *nasıl, hangi, yarın, sonra, şimdi, yine*;
     *anne, amca, teyze, dayı, hala*; loans such as *lokanta, gazete, pencere, sinema*; names.
  2. Leave everything else as `espeak-compat` gives it, so the change is measurable alone.
- **Evaluation:**
  - EchoCraft reruns the teacher-F0 stress test with Akana's marks (`espeak_word_analyses.py stress`).
  - Pre-registered target: words marked non-final show the teacher's peak shift, with no loss on
    control words.
  - Then retrain the student and check the set-R CER guard.
- **Later stages of 2:**
  - Morphological stress from Akana's analyses: pre-accenting *-mA*, *-(y)lA*, *-cA*, copulas after
    nominals, *-iyor* before *-lAr*, *-ArAk*, *-ki*.
  - Sezer stress for place names.
  - Check eSpeak's suspected misfires (*başladı, söyledi, ettim, baktım*) against the teacher first.
- **Where it plugs in:** `predict_marks` in `speech/mod.rs` gets a profile-specific stress source ahead
  of the lexicon, and `StressSource` gains `lexicon:adverb`, `morphology:pre-accenting(-mA)` and
  `sezer`.
