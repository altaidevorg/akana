# `espeak_compat_golden.jsonl`

Golden token streams for the `espeak-compat` phonemiser profile (`akana_core::speech`), checked by
`crates/akana-core/tests/speech_tests.rs`.

- **Texts:** 60 sentences of the Common Voice Turkish sentence collector, CC0 public domain
  (`server/data/tr/sentence-collector.txt`, common-voice commit `b12cde65`). They were chosen to cover
  apostrophes, quotes, ellipses, the short pause before *ve*/*ki*/*ama*, izafet, the joined pairs, the
  question particle, ğ, doubled stops, *-ken*, circumflex vowels, *ile*/*yoksa*, spelled words and
  clause-final pronouns.
- **Tokens:** eSpeak NG 1.52.0's output for each text as EchoCraft's student TTS reads it
  (`[ph, stress, word, lengthen, mark]`, edge pauses included), copied from EchoCraft's reference
  fixtures `20261107-akana-phonemiser-fixtures-v3/espeak-plan.jsonl` (rows `st*` of its train split and
  `sv*` of its validation split). Only eSpeak's observed outputs are used; nothing from eSpeak's rule
  files or source.

# `tts_v1_golden.jsonl`

The `tts-v1` profile's own output for the same 60 texts, frozen with the profile's version
(`tts-v1.0`): `[ph, stress, word, lengthen, mark, type, ipa]` per token, edge pauses included.
`tts_v1_golden_is_frozen` fails on any change. After a deliberate change, bump
`Profile::TtsV1.version()` and rewrite the file with
`cargo test -p akana-core --test speech_tests write_tts_v1_golden -- --ignored`.
