# Turkish phonemiser, stage 2: the `tts-v1` profile (2026-11-07)

`tts-v1` is Akana's own TTS front end for Turkish: phonemes, a prominence hierarchy and pauses,
from the phonology literature and Akana's morphology, checked against the teacher TTS's pitch. It
uses no eSpeak-derived data. Brief: EchoCraft `experiments/2026-11-07-akana-phonemiser-brief.md`,
stage 2 and the token contract of 2026-11-07.

**Frozen.** EchoCraft's shipped voice (student 0.3) was trained on this profile's tokens, so its
output on text of letters and punctuation never changes: the 0.11.0 build gives the same tokens as
the stage-2 freeze (commit `0c9d167`) on all 15,328 fixture clips and the 15,817-word list. Since
0.11.0 it verbalises digits, symbols, initialisms, Roman numerals, URLs and addresses first
(`verbalize_with(text, false)`); words written in letters stay as written. Every improvement goes
to [`tts-v2`](phonemizer-tts-v2.md).

## Summary

- **Token schema:** one token per phoneme or pause with `ph`, `stress`, `word`,
  `lengthen`, `mark`, `type` and `ipa`, a `_` at each edge. Each word carries the provenance of its
  stress and its segments, and its `prominence`.
- **Stress from the literature:** default final stress; 87 lexical exceptions (adverbs,
  postpositions, spatial stems, kinship terms, loans); Sezer's rule for place names; compounds on
  their first part; pre-accenting suffixes from Akana's morphological analysis; stressed suffixes.
- **A prominence hierarchy, not one Boolean focus:** 7 the nucleus, 6 phrase accents, 4 deaccented
  phrase-internal heads, 3 function words, 2 post-nuclear words, 1 unstressed syllables, 0 clitics.
- **Against the teacher's pitch:** where `tts-v1` and eSpeak put a word's stress on different
  syllables, the teacher's F0 peak is on `tts-v1`'s syllable 50.2% of the time and on eSpeak's
  30.5% (tuning split `st`, 6,000 words); on the check split `sv` it is 54.9% against 28.8% (184
  words). The held-out split `sr` was not looked at.
- **Pauses where the teacher pauses:** no apostrophe or quote pause pairs (the teacher pauses at
  0–5% of them), short pauses after adverbial clauses, at brackets and before conjunctions. The
  teacher paused at 266 places in `st` where eSpeak planned no pause; `tts-v1` plans a pause at 132
  of them (50%).
- **Long vowels and geminates:** circumflex vowels, two equal vowels and a vowel before `ğ` and a
  consonant are long; `ğ` between back vowels is silent and between front vowels a glide (checked
  against the teacher's durations); geminate stops and affricates and a doubled clear l are one
  long segment. Every `ipa` is one entry of the
  aligner's vocabulary (437,136 tokens on 15,379 fixture texts, none outside it).
- **Latency:** median 126 µs per sentence of 50–70 characters, 99th percentile 216 µs (budget: 2 ms
  and 10 ms); about **22,000× faster than real time**.

## Interface

```python
import akana
akana.phonemize("Serkan mı bavulu hazırladı?", profile="tts-v1")       # tokens
akana.phonemize_detailed("Serkan mı bavulu hazırladı?", profile="tts-v1")  # words, provenance
```

```rust
use akana_core::speech::{Phonemizer, Profile, Utterance};
let p = Phonemizer::new(Profile::TtsV1);   // loads Akana's morphology once per process
let mut u = Utterance::default();
p.phonemize_into("Serkan mı bavulu hazırladı?", &mut u)?;
for t in &u.tokens { /* t.ph_symbol(), t.stress, t.word, t.lengthen, t.mark, t.ipa() */ }
for w in &u.words { /* w.prominence, w.stress_source.label(), w.segment_source.label() */ }
```

Profile names: `"tts-v1"` (also `"tts_v1"`, `"TtsV1"`); version string `tts-v1.0`. The output is
deterministic: the same text gives the same tokens on every call and in every process.

## Symbol inventory

`ph` keeps the stage-1 mnemonics (eSpeak's notation), so that tokenisers built then read it. `ipa` is the aligner's
own symbol (wav2vec2-xlsr-53-espeak-cv-ft vocabulary). Counts are the tokens of the 15,379 fixture
texts.

| `ph` | `ipa` | long `ipa` | `type` | letter, context | tokens |
|---|---|---|---|---|---|
| `a` | a | aː | vowel | a, â | 46,756 |
| `e` | e | eː | vowel | e, tense (open syllable, long) | 15,222 |
| `E` | ɛ | | vowel | e, lax (closed or final syllable) | 16,406 |
| `&` | æ | | vowel | e before a tautosyllabic m, n or l | 5,099 |
| `i` | i | iː | vowel | i, tense; î | 11,617 |
| `I` | ɪ | | vowel | i, lax | 21,799 |
| `@` | ɯ | ɯɯ | vowel | ı (the aligner writes a long ı as `ɯɯ`; it has no `ɯː`) | 19,326 |
| `o` | o | oː | vowel | o, tense | 4,487 |
| `O` | ɔ | | vowel | o, lax | 6,304 |
| `u` | u | uː | vowel | u, tense; û | 5,178 |
| `U` | ʊ | | vowel | u, lax | 9,589 |
| `y` | y | yː | vowel | ü, tense | 3,529 |
| `Y` | ø | | vowel | ü, lax | 3,970 |
| `W` | œ | œː | vowel | ö | 3,364 |
| `b` | b | bː | vstop | b, bb | 13,098 |
| `d` | d | dː | vstop | d, dd | 19,379 |
| `g` | ɡ | | vstop | g before a back vowel | 456 |
| `J` | ɟ | | vstop | g before a front vowel | 4,724 |
| `dZ` | dʒ | dʒː | vstop | c, cc | 3,270 |
| `*` | ɾ | | vstop | r between vowels (a tap; eSpeak's class kept) | 10,028 |
| `p` | p | pː | stop | p, pp | 2,994 |
| `t` | t | tː | stop | t, tt | 12,239 |
| `k` | k | kː | stop | k, kk | 17,912 |
| `c` | c | cː | stop | k before a long front or back circumflex vowel (`kâr`, `hikâye`) | 97 |
| `tS` | tʃ | | stop | ç | 4,511 |
| `f` | f | | fricative | f | 1,965 |
| `s` | s | | fricative | s (ss stays two: no `sː`) | 12,045 |
| `S` | ʃ | | fricative | ş | 7,221 |
| `h` | h | | fricative | h | 5,174 |
| `v` | v | | vfricative | v | 4,043 |
| `z` | z | | vfricative | z | 6,669 |
| `Z` | ʒ | | vfricative | j | 40 |
| `Q` | ɣ | | vfricative | ğ after a consonant (typos) | 2 |
| `m` | m | | nasal | m | 18,301 |
| `n` | n | | nasal | n | 28,772 |
| `l` | l | lː | liquid | clear l before a front vowel; ll of a front word | 7,374 |
| `l/` | l | | liquid | clear l closing a syllable | 2,236 |
| `L` | ɫ | | liquid | dark l before a back vowel | 7,941 |
| `L/` | ɫ | | liquid | dark l closing a syllable | 3,242 |
| `R` | r | | liquid | r | 17,453 |
| `j` | j | | liquid | y; ğ between front vowels | 16,474 |
| `w` | w | | liquid | w (foreign spellings) | 2 |
| `_` | | | pause | the two edges of the text | 30,758 |
| `_:` | | | pause | clause boundary (`,` `;` `:` `.` inside the text) | 3,744 |
| `_!` | | | pause | short pause (see Pauses) | 2,326 |

The aligner has long forms only for the listed segments (`tː` but no `sː`, `nː`, `rː`, `ʃː`): a
doubled s, n, m, r, ş, l of a back word and the others stay two tokens.

## Stress levels

`stress` is per token. A vowel carries its syllable's level; a consonant takes the level of its
syllable (a single consonant before a vowel is that vowel's onset, otherwise it closes the
syllable before). A word's level is the level of its stressed syllable (`prominence` in
`phonemize_detailed`).

| level | meaning |
|---|---|
| 7 | **Nucleus**, one per clause (a clause ends at `_:` or the text's end): the host of a question clitic `mI`; else, when the clause ends in a finite verb, the nearest accented word before that verb within the last intermediate phrase (after the last `_!`), or the verb if there is none; else the clause's last accented word. |
| 6 | **Phrase accent**: the stressed syllable of an accented content word before the nucleus. Several words of a clause carry it. |
| 5 | unused, kept for emphasis. |
| 4 | **Deaccented head**: the stressed syllable of a noun inside a phrase whose accent is on its first word: a noun after an adjective, a numeral other than `bir` or a genitive (`kırmızı çanta`, `evin kapısı`), and a possessed noun after a bare noun (`kapı kolu`). |
| 3 | **Function word**: pronouns, postpositions, conjunctions, determiners, interjections, `bir`, `şey`. |
| 2 | **Post-nuclear**: the stressed syllable of any word after the nucleus. |
| 1 | an unstressed syllable. |
| 0 | a **clitic**, and pauses: `de`, `da`, `mi` (all forms), `ki`, `ve`, `ya`. |

Teacher check of the levels (`st`, 14,549 sentences): the stressed syllable's mean F0 against the
sentence's mean, in semitones, by third of the sentence.

| third | L7 | L6 | L4 | L3 | L2 | L0 |
|---|---|---|---|---|---|---|
| first | +2.6 | +3.8 | +1.9 | +4.8 | −0.6 | +2.2 |
| middle | −0.5 | +1.6 | +0.8 | +2.0 | −2.5 | −0.5 |
| last | −3.5 | +0.2 | −0.6 | +0.7 | −5.5 | −4.2 |

Post-nuclear words (2) are the lowest in each third and phrase accents (6) sit above deaccented
heads (4). The nucleus (7) is not the highest pitch: Turkish marks it with the fall that follows it
and pitch declines through the clause, so a level is a prominence rank, not a pitch target. Level 3
is high early in the sentence because sentence-initial pronouns and demonstratives (`Bu`, `O`) carry
the onset's high pitch; the levels do not encode pitch shapes.

## Rules and evidence

The teacher's F0 is EchoCraft's index of the teacher's 14,549 training sentences (per-token mean
log F0). Tuning used `st*` only, `sv*` checks it, and `sr*` (EchoCraft's held-out test set) was not
read. "Peak" is the syllable with the word's highest vowel F0; "control" is the same syllable
position in words of the same length without the suffix.

| rule | source | teacher evidence (`st`) |
|---|---|---|
| final stress by default | Lewis 2001; Kabak & Vogel 2001 | `başladı`, `ettim`, `baktım` peak finally: eSpeak's non-final readings of these are not followed |
| lexical exceptions (87: adverbs, postpositions, spatial stems, kinship terms, loans) | Lewis 2001; Göksel & Kerslake 2005 | adverbs 67% initial peaks, kinship terms 79%, against 38–42% for controls; each `T` entry has ≥ 6 mid-clause tokens above the control rate |
| Sezer's rule for place names | Sezer 1981 | literature only (place names are rare mid-clause) |
| compounds on the first part | Kabak & Vogel 2001 | literature only |
| negative `-mA` pre-accenting (not aorist `-mAz`) | Kabak & Vogel 2001 | 3-syllable words: antepenult peak 68% against 41% |
| copula `-DIr`, `-(y)DI`, `-(y)mIş`, `-(y)sA` after a nominal | Kornfilt 1997 | 2-syllable words: initial peak 82% against 42% |
| comitative/instrumental `-(y)lA` | Göksel & Kerslake 2005 | 3-syllable words: penult 55% against 17% |
| conditional `-sA` | Kabak & Vogel 2001 | penult 37% against 17% |
| `-cA`, `-(y)ken`, `-mAdAn` pre-accenting; language names in `-cA` (`TÜRKçe`) | Göksel & Kerslake 2005 | literature (too few tokens) |
| z-paradigm person endings after `-AcAk`, `-Ir`, `-mIş`, `-mAlI`, `-Iyor` and the copula (`gelMİŞsin`, `yorGUNum`) | Göksel & Kerslake 2005 | literature |
| `-Iyor`, `-ArAk`, `-IncA`, `-(y)AmA` stressed on their first syllable | Lewis 2001 | `-IncA` penult 57% against 17%; `-ArAk` (4 syllables) penult 35% against 10% |
| `-ki` is not pre-accenting | — | no shift in the teacher (eSpeak's reading not followed) |
| ğ: long vowel before a consonant, short in hiatus | Göksel & Kerslake 2005 | teacher's vowel duration (80 fps frames, eSpeak tokens aligned to its audio): a 6.0 frames in an open syllable, 6.0 before ğ and another vowel (553), 12.0 before ğ and a consonant (297), 13.0 for a ğ a (155); o 6.0, 7.0, 10.0; e + ğ + consonant 11 frames against 9 for e + y + consonant |
| proper names (people) final | Lewis 2001 | strongly final |
| modifier before head: head deaccented (4) | Kabak & Vogel 2001; Kan 2009 | the modifier's peak is above the head's in 79–86% of pairs, against 71–73% for other adjacent pairs |
| function words below content words (3) | Kan 2009 | a postposition's host is above it in 90% of pairs, by 3.1 st on average |
| the host of `mI` is the nucleus, `mI` a clitic (0) | Göksel & Kerslake 2005 | the host is 7.9 st above `mI` on average |
| post-nuclear deaccenting (2) | Kan 2009; Özge & Bozşahin 2010 | lowest level in every third of the sentence (table above) |

## Pauses

| pause | where | teacher |
|---|---|---|
| `_` | each edge of the text | — |
| `_:` | one at a clause boundary (`,` `;` `:` `.` `!` `?` `…` inside the text) | pauses at 96% of them |
| `_!` | at quotes and brackets (one, not eSpeak's `_: _:` pair) | eSpeak's pairs at quotes and apostrophes: 0–5% |
| `_!` | before `ve`, `veya`, `yahut`, `yoksa`, `ama`, `fakat`, `ancak`, `lakin`, `çünkü`, `ki`, `oysa`, `halbuki`, `hatta` unless the clause ends there | 30% of planned short pauses get a pause |
| `_!` | after a converb or a conditional (`gelince`, `giderek`, `gelip`, `gelirken`, `gelirse`) | the teacher paused after 79 of them where eSpeak had no pause |
| `_!` | after `için`, `gibi`, `sonra`, `önce`, `kadar`, `halde`, `zaman` and other clause closers after a participle or verbal noun (`geldiği için`, `gittikten sonra`) | 97 |

No pause is planned after an apostrophe, and words are not joined into one (`ister misiniz` is two
words). Clips the teacher reads with a pause where none is planned are dropped by EchoCraft (eSpeak
lost 262 clips); of the 266 such places in `st`, `tts-v1` now plans 132. The other 134 have no
cue in the text that a rule here found reliable, so no pause is invented for them.

`mark` is the punctuation that actually ends each clause (the last of `, . ; : ! ? …` before the
boundary, `…` read as `.`, `.` when there is none), on every token of the clause.

## Segments

- The letter rules of stage 1 (`speech::letters`), which follow eSpeak's allophones for the
  aligner's sake (`tts-v2` replaces them with the literature's): lax `E I U Y O` in closed and final syllables, `&` before a tautosyllabic m, n,
  l; palatal `J` before front vowels; clear and dark l by vowel; tap `*` between vowels.
- Circumflex vowels are long (`hâlâ` h aː l aː), and k before a circumflex vowel is `c` (`kâr`).
- Two equal vowels are one long vowel, next to each other or across ğ (`saat` s aː t, `ağaç`
  aː tʃ).
- ğ before a consonant or the end lengthens the vowel before it (`dağ` d aː, `doğru` d oː ɾ ʊ,
  `eğlence` eː l æ n dʒ ɛ, `öğretmen` œː ɾ ɛ t m æ n).
- ğ between back vowels is silent and the first vowel stays short (`soğuk` s o ʊ k, `sağır`
  s a ɯ r); between front vowels it is a glide (`değil` d e j ɪ l, `düğün` d y j ø n).
  eSpeak lengthens the first vowel of `soğuk`; the teacher does not.
- Long vowels are tense.
- Geminate b, c, d, k, p, t, ç, g are one long segment (`dikkat` d ɪ kː a t), and close the
  syllable before them; a doubled clear l is `lː` (`elli` æ lː ɪ); a doubled dark l stays two.
- Roots the lexicon marks InverseHarmony (`saat`, `kabul`, `hal`) take a clear l
  (`segment_source = lexicon:akana(InverseHarmony)`).
- No palatal `c` in `-ken` (eSpeak's reading, not attested in the literature).

## Provenance

`stress_source` is one of `lexicon:{class}:{sources}` (e.g. `lexicon:adverb:L,A,T`),
`morphology:{rule}` (e.g. `morphology:pre-accenting(-mA)`), `sezer`, `compound:first-part`,
`clitic`, `default-final` or `rule:spelled`. `segment_source` is `tts-v1-rule:letters` or
`lexicon:akana(InverseHarmony)`.

## Tests and gates

- Rust unit tests in `speech::tts` (word stress, prominence, pauses, segments, every `ipa` in the
  aligner vocabulary, determinism and odd input) and Python tests in `tests/test_speech.py`.
- `tts_v1_golden_is_frozen` (Rust and Python) checks the frozen tokens of 60 CC0 sentences.
- Latency: `cargo run --release -p akana-core --example phonemizer_bench -- --profile tts-v1`.

## References

- Göksel, A. & Kerslake, C. (2005). *Turkish: A Comprehensive Grammar*. Routledge.
- Kabak, B. & Vogel, I. (2001). The phonological word and stress assignment in Turkish.
  *Phonology* 18.
- Kan, S. (2009). *Prosodic domains and the syntax-prosody mapping in Turkish*. MA thesis,
  Boğaziçi University.
- Kornfilt, J. (1997). *Turkish*. Routledge.
- Lewis, G. (2001). *Turkish Grammar*, 2nd ed. Oxford University Press.
- Özge, U. & Bozşahin, C. (2010). Intonation in the grammar of Turkish. *Lingua* 120.
- Sezer, E. (1981). On non-final stress in Turkish. *Journal of Turkish Studies* 5.
- Zimmer, K. & Orgun, O. (1999). Turkish. In *Handbook of the IPA*. Cambridge University Press.
