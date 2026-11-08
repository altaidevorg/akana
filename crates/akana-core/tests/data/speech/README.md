# Phonemiser and verbaliser golden files

Checked by `crates/akana-core/tests/speech_tests.rs` and `tests/test_speech.py`. All tokens and
readings are Akana's own output.

- **`tts_v1_golden.jsonl`:** profile `tts-v1`'s tokens (`[ph, stress, word, lengthen, mark,
  type, ipa]`, edge pauses included) for 60 sentences of the Common Voice Turkish sentence
  collector (CC0, `server/data/tr/sentence-collector.txt`, common-voice commit `b12cde65`).
  They were chosen to cover apostrophes, quotes, ellipses, the short pause before conjunctions,
  izafet, the question particle, ğ, doubled stops, circumflex vowels, spelled words and
  clause-final pronouns. The output is frozen as of stage 2 (commit `0c9d167`): EchoCraft's
  shipped voice was trained on this profile. Never rewrite this file.
- **`tts_v2_golden.jsonl`:** profile `tts-v2`'s tokens for the same 60 sentences and the 40 of
  `verbalize_inputs.txt`, frozen with `tts-v2.0`.
- **`verbalize_inputs.txt`:** 40 sentences written for the verbaliser (numbers, dates, money,
  units, telephone numbers, abbreviations, initialisms, symbols, foreign words and the stress
  rules of `tts-v2`).
- **`verbalize_golden.jsonl`:** the verbaliser's reading of each, with its segments (`[source,
  spoken, kind]`).

After a deliberate change to `tts-v2` or the verbaliser, bump the profile's version and rewrite
the last two with `cargo test -p akana-core --test speech_tests write_goldens -- --ignored`.
