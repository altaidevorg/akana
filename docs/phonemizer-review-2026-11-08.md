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
