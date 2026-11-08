# Turkish phonemiser and verbaliser, stage 3: `tts-v2` (2026-11-08)

`tts-v2` is Akana's TTS front end for the next model. It is [`tts-v1`](phonemizer-tts-v1.md), which
stays frozen for EchoCraft's shipped voice, with three groups of changes:

- the segments of the Turkish phonology literature instead of eSpeak NG's allophones;
- the stress rules `tts-v1` lacked;
- every rule of the verbaliser.

Its `ph` symbols are the phonemes' IPA. Brief: EchoCraft
`experiments/2026-11-07-akana-phonemiser-brief.md`, stage 3, the owner's decisions of 2026-11-07
and 2026-11-08, and EchoCraft's review of `tts-v1`.

## Summary

- **Interface:** `akana.phonemize(text, profile="tts-v2")` gives the same token schema as
  `tts-v1` (`ph`, `stress`, `word`, `lengthen`, `mark`, `type`, `ipa`). The default profile is
  still `tts-v1`; Rust uses `Profile::TtsV2`. The version string is `tts-v2.0`.
- **Contract:**
  - `ph` is the IPA without a length mark; pauses are `_`, `_:` and `_!`, and every pause `ph`
    and no other starts with `_`.
  - `ipa` is exactly one entry of the aligner vocabulary: all 436,958 tokens of the 15,379
    fixture texts are, with none outside.
  - `stress` keeps `tts-v1`'s prominence levels (7 the nucleus down to 0 a clitic), and so do
    the pauses and marks.
- **Differences from `tts-v1` on the fixture texts:** 17.9% of tokens change their `ipa` and
  0.14% their stress; 3 of the 15,328 corpus clips read other words.
- **Stress changes against the teacher's F0** (tuning split `st`; `sr` not read): on the 110
  words where `tts-v2` moves the stress, the teacher's peak is on `tts-v2`'s syllable 54% of the
  time and on `tts-v1`'s 11%. *nerede* (66 words) carries almost all of it: 76% against 3%.
- **Latency:** median 86 µs per 50–70-character sentence, p99 177 µs. That is about 33,000×
  faster than real time.

## Segments

`tts-v1`'s letter rules, then these changes (`speech::letters::tts_v2_segments`).

| rule | source | `tts-v1` → `tts-v2` (tokens on the fixtures) |
|---|---|---|
| vowels are tense inside the word; /i u e/ are lowered only in the open final syllable of a phrase (before a pause or the end) | Zimmer & Orgun 1999: 155 (phrase-final; Göksel & Kerslake 2005: 10–11 say word-final) | ɪ → i 19,867; ɛ → e 12,658; ʊ → u 8,227 |
| no lax o | Zimmer & Orgun 1999: 155 (/o/ is not lowered) | ɔ → o 6,309 |
| ü is [y]; its lowered [ʏ] is not in the aligner's vocabulary | Zimmer & Orgun 1999: 155 | ø → y 4,138 |
| /e/ is [æ] before a coda m, n, l or r | Göksel & Kerslake 2005: 10 (*kendi*, *hem*); lowering "before coda /m n l r/" in most speakers | ɛ → æ 3,028 |
| /ɾ/ is a tap only between vowels; fricated [r̝] word-initially, after a consonant and before a voiced one; devoiced [r̝̊] word-finally and before a voiceless consonant; no trill | Zimmer & Orgun 1999: 155; Yavuz & Balcı 2011: 25 | r → r̝̊ 10,662; r → r̝ 6,786 |
| /k g/ are palatal [c ɟ] next to a front vowel | Zimmer & Orgun 1999: 154–155; Göksel & Kerslake 2005: 7–9 | k → c 6,347; kː → cː 67; ɡ → ɟ 10 |
| a circumflex vowel is long only in an open syllable | the teacher (below); Lewis 2001: 3–4 has the circumflex mark palatal k g l and, in some words, length | aː → a 49; uː → u 11 |

The tap is `type` `liquid` in `tts-v2`; `tts-v1` keeps `vstop`. The fricated r is `vfricative`,
the devoiced one `fricative`.

**Teacher durations for the circumflex** (eSpeak tokens aligned to the teacher's audio, 80 fps, splits `st` and `sv`):

| context | median frames | short vowel |
|---|---|---|
| open syllable, after k g l (`hâlâ`, `pekâlâ`, `şikâyet`) | 10–12 (n = 94) | 6 |
| open syllable, elsewhere (`hâkim`, `âciz`, `mâni`) | 11 (n = 54) | 6 |
| closed syllable (`imkân`, `rüzgâr`, `mahkûm`) | 6–8 (n = 48) | 6–7 |

**Not changed, and why:**
- /e o ø/ are mid [e̞ o̞ ø̞] (Zimmer & Orgun 1999: 155), but `tts-v2` writes the broad `e`, `o`,
  `œ` that the aligner learned for Turkish.
- ı stays `ɯ`; Zimmer & Orgun describe it as [ɨ].
- Final /h/ as [x ç] and /v/ as [β] next to rounded vowels vary by speaker (Zimmer & Orgun
  1999: 155; Göksel & Kerslake 2005: 6), so they are left out.
- The aligner has no voiceless [l̥], so final and preconsonantal /l/ stays voiced.

These segments have no acoustic check yet. The F0 index measures pitch, not vowel quality or
frication. EchoCraft's CTC alignment score on `tts-v2` tokens, set against `tts-v1`'s on the
same clips, is the check to run before training on them.

## Stress

`tts-v2` keeps `tts-v1`'s rules and adds these.

| rule | source | teacher (`st`): peak on `tts-v2`'s syllable / `tts-v1`'s |
|---|---|---|
| *nerede*, *nereden*, *neresi* initial (`tts-v1` read *nerede* finally, missing its lexicon stem) | Dursunoğlu 2006: 272; Göksel & Kerslake (*-rA* unstressable: *néreye*, *órada*) | 76% / 3% (n = 66) |
| *-(y)Iver* stressed on its first syllable (*yaPIver*) | Inkelas & Orgun 2003: 142; Özçelik 2014: 232 | 18% / 9% (n = 22); `sv` 1 / 0 (n = 1) |
| place names in *-istan*, *-hane*, *-lAr*, *-mAz* final (*Hindistán*) | Inkelas & Orgun 2003: 142–152; Dursunoğlu 2006: 272 | 1 / 0 (n = 1) |
| *-ki* after a genitive pre-accenting (*seNİNki*, *beNİMki*) | Kabak & Vogel 2001: 328 | 11% / 22% (n = 9) and 40% / 40% (n = 5): **the teacher leans final** |
| *-leyin* pre-accenting (*saBAHleyin*) | Özçelik 2014: 251 | 17% / 67% (n = 6): **the teacher reads it finally** |

Where the literature and the teacher disagree (*-ki* after a genitive, *-leyin*), `tts-v2` follows
the literature, as the owner decided, and records the disagreement here. Both rest on few words.

**-ki elsewhere** (`evdeki`, `yarınki`): stressable, final, as in `tts-v1`. Göksel & Kerslake
(2nd ed. 2025, as summarised by halbuki.org) count the nominal *-ki* among stressable suffixes
(*kitaplarımdakí*). Kabak & Vogel's (2001: 328) pre-accenting example is the pronominal *benimki*,
which the row above covers. The teacher shows no shift on the locative *-ki*.

## Verbaliser

`tts-v2` uses every rule of `speech::verbalize` (Python `akana.verbalize`). `tts-v1` verbalises
only stretches with a digit or a symbol: numbers, money, units, telephone numbers, e-mail,
symbols, `A4`, `C++`, and `No:`/`s.` before a number. Abbreviations, initialisms, Roman numerals,
foreign words, spelled letters and URLs written in letters alone are kept as written, so its
output on text of letters and punctuation is unchanged. Each reading keeps the byte span of the text it reads, and the phonemiser's words
point back at it.

| written | spoken | rule |
|---|---|---|
| `3'te`, `1990'larda`, `4'ü` | üçte, bin dokuz yüz doksanlarda, dördü | cardinal and the written suffix, re-harmonised (`dört` → `dörd-`) |
| `2. sınıf`, `5'inci`, `1. Lig'de` | ikinci sınıf, beşinci, birinci Lig'de | ordinal: a dot before a small letter or at a sentence's start; the `-(I)ncI` suffix |
| `XIV. Louis`, `Mehmet II` | on dördüncü Louis, Mehmet ikinci | Roman numeral with a dot, or after a name |
| `3,5`, `0,05`, `2.1` | üç virgül beş, sıfır virgül sıfır beş, iki nokta bir | decimals |
| `2/3'ü`, `24/7`, `10-15`, `3-1`, `3x4` | üçte ikisi, yirmi dört yedi, on ila on beş, üç bir, üçe dört | fraction, range, score, dimensions |
| `12.03.2024`, `2024-03-12'de` | on iki mart iki bin yirmi dört(te) | dates |
| `14:30'da`, `09.45'te`, `14:00-16:00` | on dört otuzda, dokuz kırk beşte, on dört ila on altı | times |
| `1.299,90 TL`, `₺850'ye`, `3.50 TL`, `1,5 milyon TL` | … lira doksan kuruş, sekiz yüz elli liraya, üç lira elli kuruş, bir virgül beş milyon lira | money: the suffix moves to the currency word |
| `%25'lik`, `50%`, `‰5` | yüzde yirmi beşlik, yüzde elli, binde beş | percent |
| `12 km'den`, `5 kg'ye`, `90 km/sa`, `-5°C`, `3 m²` | on iki kilometreden, beş kilograma, saatte doksan kilometre, eksi beş derece, üç metrekare | units and rates |
| `0532 123 45 67`, `+90 212 555 12 34` | sıfır beş yüz otuz iki yüz yirmi üç … | telephone numbers in their groups |
| `Dr. Ahmet`, `Atatürk Cad.`, `vb.` | doktor Ahmet, Atatürk caddesi, ve benzeri | abbreviations by place: a title before a name, an address word after one, the rest anywhere |
| `TBMM'de`, `TDK'ye`, `CNN'e`, `NATO` | tebememede, tedekeye, sienene, NATO | initialisms as the lexicon reads them; the suffix decides *ka* or *ke*; a word in capitals is read as the word |
| `ali@ornek.com`, `www.ornek.com.tr` | ali et ornek nokta kom, ve ve ve nokta ornek nokta kom nokta te re | electronic |
| `&`, `+`, `=`, `#`, `@`, `°`, `½` | ve, artı, eşittir, diyez, et, derece, yarım | symbols |
| `Ç-A-Y`, `A4`, `COVID-19`, `2x` | çe a ye, a dört, kovid on dokuz, iki iks | letters |
| `iPhone'u`, `Facebook'ta`, `Wikipedia`, `café` | ayfonu, feysbukta, Vikipedia, cafe | foreign words (a table of Akana's own transcriptions) and letters (w v, q k, x ks, accents dropped) |

Sources and checks:
- **Rule sources.** Ordinals with a dot, suffixes after numbers and abbreviations with an
  apostrophe, and suffixes after an abbreviation following its reading all come from TDK's
  spelling rules. Suffix harmony comes from Göksel & Kerslake 2005. Initialism readings come
  from Akana's lexicon (Zemberek's `pr=` attributes, Apache-2.0).
- **normalizer-tr.** Its 0.7.0 release (Apache-2.0) was run as a black box on 17 test sentences
  to compare readings. No code or data of it is used.
- **Where Akana reads more.** It also reads ordinals, Roman numerals with a dot, `KDV'si`,
  e-mail addresses, `₺850'ye`, fractions, `3x4`, spelled letters and `2023-2024`.
- **A deliberate difference.** Akana says *lira* where normalizer-tr says *Türk lirası*.

## Freeze and tests

- **`tts-v1`:** identical to commit `0c9d167` on all 15,328 fixture clips and the 15,817-word
  list, and on 3,000 random texts of letters and punctuation (the review's differential test).
  Only 10 probe rows with digits or symbols read differently, because they are now verbalised.
- **Review:** an independent review of the whole branch, its findings and their fixes:
  [phonemizer-review-2026-11-08.md](phonemizer-review-2026-11-08.md).
- **Golden files** (Akana's own output):
  - `tts_v1_golden.jsonl`: 60 CC0 sentences, never rewritten;
  - `tts_v2_golden.jsonl`: the 60 sentences plus the 40 verbaliser sentences of
    `verbalize_inputs.txt`;
  - `verbalize_golden.jsonl`: readings, spans and kinds.

  Rust (`tests/speech_tests.rs`) and Python (`tests/test_speech.py`) check them.
- **Unit tests:** the letter rules of both profiles, the verbaliser (cardinals, ordinals,
  decimals, dates, money, units, telephone numbers, abbreviations, initialisms, symbols, context
  rules, span mapping, plain text unchanged, odd input), stress, prominence and pauses.

## Open points

- **Acoustic checks of the segments:** EchoCraft's alignment score on `tts-v2` against `tts-v1`.
- **Stress of adverbs in *-en/-an*** (*kısmen*, *şahsen*): penultimate in descriptions of Turkish
  stress, but without a citation found yet, so they stay final.
- **Foreign words:** a table and letter rules, not a full loanword reader.

## References

The page numbers are those the secondary sources give. The rules were read in the Wikipedia
article *Turkish phonology* (consulted 2026-11-08, sections on vowels, consonants and stress) and
in halbuki.org's summary of Göksel & Kerslake (2nd ed.), not in the originals. Check the originals
before citing them further.

- Dursunoğlu, H. (2006): 272. Initial stress of *nerede*, *nereye*, *nasıl*, *yarın*; final
  stress of place names in *-istan*.
- Göksel, A. & Kerslake, C. (2005). *Turkish: A Comprehensive Grammar*. Routledge (2nd ed. 2025):
  5–11 (segments), unstressable *-rA*, stressable nominal *-ki*.
- Inkelas, S. & Orgun, C. O. (2003). Turkish stress: a review. *Phonology* 20: 142–152 (place
  names, stressed suffixes).
- Kabak, B. & Vogel, I. (2001). The phonological word and stress assignment in Turkish.
  *Phonology* 18: 328 (pre-accenting *-ki*, *benimki*).
- Lewis, G. (2001). *Turkish Grammar*, 2nd ed. Oxford University Press: 3–4 (the circumflex).
- Özçelik, Ö. (2014). *Phonology* 31: 232 (stressed *-iver*), 251 (pre-accenting *-leyin*).
- Yavuz, H. & Balcı, A. (2011): 25. Allophones of /ɾ/.
- Zimmer, K. & Orgun, O. (1999). Turkish. In *Handbook of the International Phonetic
  Association*. Cambridge University Press: 154–158.
