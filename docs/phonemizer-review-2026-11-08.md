# Review of the phonemiser and verbaliser branch (2026-11-08)

**What was reviewed.** An independent reviewer agent read the whole branch diff against `main`
at `f764c3c`: `speech/*`, the Python binding, tests, docs and notices. It was read-only and ran
behaviour probes in Python, including a differential test against the stage-2 wheel (`0c9d167`)
and a fuzzer. Each finding was then re-run by hand. Every one reproduced, and each fix has a
regression test: `speech::verbalize::tests::review_regressions` and
`long_input_maps_spans_in_linear_time`, `speech::text::tests::decomposed_text_keeps_its_words_with_marks`,
and `tests/test_speech.py::test_review_regressions`.

| # | finding | verdict | fix |
|---|---|---|---|
| 1 | Digits that are not ASCII (`1١11`, `1２３４`, `12.03.٢٠٢٤`) panicked: the regexes' `\d` matches every Unicode digit, but the code slices by ASCII byte | confirmed, crash | the regexes take `[0-9]` only |
| 2 | `tts-v1` changed text of letters and punctuation: initialisms (`TBMM`), Roman numerals (`II. Dünya`), capitals spelled with hyphens, `.net`, URLs in letters alone (1,054 of 3,000 random letter texts differed from stage 2) | confirmed, contract | without the word rules, only stretches with a digit or a symbol are verbalised; the same 3,000 texts now match stage 2 exactly |
| 3 | An initial with a dot was read as a Roman ordinal (`M. Kemal` bininci, `C. Ronaldo`); `CV`, `CD`, `MD` after a name too; `XIV'ün` fell to the initialism rule | confirmed | a single letter is an ordinal only for I or X not after a name, or V before a small letter; after a name only I, V and X count; `CD`, `CV`, `MD` and similar are never Roman; a Roman numeral takes its suffix |
| 4 | Telephone numbers: `+905321234567` lost two digits; `(0532) 123 45 67` was never read as one; `0850`, `0123` were grouped in threes | confirmed | `phone_at` checks the digit count for `+90`, `(` starts a telephone number, long groups split 3-3-2-2 or 4-3-2-2, and a number from a zero is a code only from seven digits |
| 5 | File and case numbers (`2012/3121`) were read as fractions | confirmed | a fraction only when the numerator is smaller and the denominator at most 100; otherwise *bölü* |
| 6 | `3.14` and `Python 3.12` were read as times; a dotted decimal lost its unit (`2.5 kg`) and currency (`3.5 TL`) | confirmed | a time with a dot needs a two-digit hour, *saat* before it or a suffix after it; a dotted decimal takes its unit; one or two decimals before a currency are money |
| 7 | Sentence boundaries: `T.C. Kimlik`, `bkz. Ek` gained a full stop; `A.Ş. Bugün` lost one | confirmed | the full stop ends a sentence only after a post-positioned abbreviation (`vb.`) or a company suffix (`A.Ş.`) before a capital, and at the end of the text |
| 8 | Suffixes after foreign words and symbols kept the written buffer: `Wi-Fi'ye` vayfayya, `Nike'ı` naykii, `1/2'si` ikide birsi, `¼'ü` çeyrek'ü | confirmed | the suffix is re-buffered to the reading (`vayfaya`, `naykiyi`, `biri`, `çeyreği`) |
| 9 | The span map scanned every segment for every word: 1 MiB of `x 5` took 691 s; a 1 MiB word took 33 s | confirmed, performance | binary search in `Verbalized::source`, and linear scans in `syllable_levels` and the letter rules: now 8 s and 3.4 s |
| 10 | Decomposed (NFD) text split words at every combining mark | confirmed | `tts-v2` keeps combining marks inside their word and analyses the composed form; `tts-v1` stays as frozen |
| 11 | Smaller readings: `1990-95` (no *ila*), `24:00`, `x=-5` (minus lost), `09:05:07` (seconds), `Kaç? 3. Ben` (ordinal), `1.2.24` (a date), `gittim.dev` (a URL), `5. Sok.`, Cyrillic capitals as initialisms | confirmed | each fixed and asserted in `review_regressions`; initialisms only in Latin capitals |
| 12 | Docs against behaviour: the `verbalize` docstring, the module header, `tts-v2.md` on what `tts-v1` reads, the speed figures; an unused regex; `Seg.at` as `u16` | confirmed | docs corrected; regex removed; `Seg.at` is `u32` |
| 13 | Clean room: no eSpeak NG source or data and no normalizer-tr code in the tree; but the branch's history (`7ecbf78`, `96fe19a`) holds the eSpeak-derived table `espeak_compat.tsv.gz` | confirmed, history only | **for the owner:** squash or rewrite the branch before it is pushed or merged, if that history should not reach the repository |

**Checked clean by the reviewer:**
- **tts-v1 freeze:** all 15,775 fixture and word-list rows match the stage-2 reference, apart from the probes.
- **Token contract:** in both profiles, on the fixtures and on 6,000 random and adversarial strings.
- **Span mapping:** correct.
- **tts-v2 rules:** fire as documented.
- **Suffix harmony:** correct for numbers, money and units.
- **Python binding:** errors and offsets correct.

**After the fixes:**
- **tts-v1 freeze:** identical to `0c9d167` on all 15,328 fixture clips and the 15,817-word list.
  Ten probe rows with digits or symbols differ, as intended.
- **tts-v2 contract:** every `ipa` is in the aligner vocabulary.
- **Full gate:** passes, as recorded in the commit.

## Review of the consensus update (`c058cad`)

An independent reviewer agent read the delta `7b07dd5..c058cad`: `tts-v2` updated to the
literature consensus. It was read-only and probed in Python: all 15,379 fixture texts, the
fixtures in capitals, 6,000 fuzz texts, and a differential `tts-v1` test against the stage-2 wheel.
Every finding was re-run by hand and reproduced. Each fix is asserted in
`speech::tts::tests::tts_v2_follows_the_literature_consensus`.

| # | finding | verdict | fix |
|---|---|---|---|
| 1 | The intensive detector (a (C)V, then p/m/r/s, then a known word) fired on about 287 ordinary words in the fixtures: *araba*, *arasında*, *durdurdu*, *serseri*, *ürünler*, *Amasya*. Its vowel copy also wrote [æ] into open syllables (*emellerine*) | confirmed, wrong output | a closed list of the 52 standard intensives, and words built on them (*kapkaranlık*) |
| 2 | Common words were treated as names. *-iye* fired on *Bu Belediye*, after a colon or line break, and in text in capitals (*saniye*); *-CIlAr* and Sezer's rule too (*Başlık: Satıcılar*, *BU SATICILAR*, *TOKAT*) | confirmed, wrong output | a sentence start also follows a colon, line break, quote, bracket or dash; a word in capitals throughout is not a capitalised name; *-iye* and *-CIlAr* need a proper noun, a gazetteer place or an apostrophe |
| 3 | r̝̊ and the phrase-final /i u/ lowering applied before the optional `_!` of a conjunction (*Kar ve yağmur*), where L13 finds /ɾ/ voiced | confirmed, wrong output | phrase-final means before `_:` or at the end of the text |
| 4 | A bare *Doğancılar* (the brief's probe) keeps final stress, but the doc did not list this as a departure | confirmed, doc | listed under the departures from the brief, with the reason |
| 5 | The *hem*/*elbise*/*Belçika* rule un-lowered an /e/ a suffix closes (*elbisem*) | confirmed | only where the root itself closes the syllable |
| 6 | Lexicalised *-mAz* nouns (*tükenmez*) are not lowered | confirmed, nit | documented as an open point: the morphology has them as single roots |
| 7 | A code comment shared with `tts-v2` cited teacher durations for ğ; the doc's ğ row cited L16 for a hiatus it does not state | confirmed, doc | the comment cites the literature; the doc row cites the consensus's "hiatus or long vowel" (row 14) |

**Checked clean by the reviewer:**
- **Brief outputs:** every expected output is reproduced, except those above.
- **Derived forms:** correct (*bugünkü*, *başbakanlık*, *Denizli'ye*, *gelmezsen*, *kısmen de*).
- **Unaffected words:** *hemen* and *hemşire* are unaffected.
- **NFD input and stress mapping:** both correct.
- **Token contract:** holds on all fixtures, the fuzz texts and adversarial input; no `r̝`.
- **`tts-v1`:** unchanged (0 differences on the fixtures, the word list and 3,000 random letter
  texts); every new path is gated on `tts-v2`.
- **Doc counts:** exact.

## PR review comment (checked against `1f947cd`)

A review comment on the pull request listed five findings. Several cited the code of the first
stage (`model.rs`, the `espeak-compat` profile), which later commits removed. Each was checked
against the branch head.

| # | finding | verdict at `1f947cd` | evidence or fix |
|---|---|---|---|
| 1 | `model.rs`'s `gunzip` discards I/O errors, so a corrupt `espeak_compat.tsv.gz` gives an empty model | not applicable | `model.rs` and `espeak_compat.tsv.gz` were removed with the `espeak-compat` profile in `ecacaba`; `speech/` loads no data file (`git ls-tree HEAD crates/akana-core/src/speech/`) |
| 2 | the Python bindings ignore `profile` and use a global `PHONEMIZER` fixed to `EspeakCompat`; the detailed JSON names the wrong profile | not applicable | the global was removed in `0c9d167`. `phonemize_json` and `phonemize_detailed_json` build `Phonemizer::new(profile)` on every call, and the detailed JSON writes `profile.name()` and `profile.version()` (`crates/akana-py/src/lib.rs`). `test_default_profile_is_tts_v1` and the `tts-v2` tests (IPA `ph`) show the profile is honoured |
| 3 | an unchecked `out.tokens[k]` in the consonant-stress copy loop could panic | not applicable | that loop was part of the `espeak-compat` prosody pass removed in `ecacaba`; no `tokens[` index remains in `speech/`. The token contract was fuzzed in both profiles (6,000 texts and adversarial input) without a panic |
| 4 | `read_digits` overflows `i64` for 16–18 digits and reads them as 0 | disputed | the guard sends every run of more than 15 digits to the digit-by-digit reading, so `i64::parse` sees at most 15 digits (below 9.2 × 10¹⁸), and `scan_number` collects ASCII digits only. Regression test `long_digit_runs_never_overflow`: 16, 18, 19 and 25 digits are read digit by digit, and 15 nines as a number |
| 5 | `Phonemizer::phonemize` drops the over-limit error, giving an empty utterance | confirmed, fixed | an over-limit text now gives its two edge pauses (the token contract) and one warning over the whole text, with the error's message; `phonemize_into` still returns the error and Python raises `ValueError`. Regression test `oversized_text_is_reported` |
