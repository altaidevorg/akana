# Turkish phonemiser and verbaliser: `tts-v2`, the literature consensus (2026-11-08)

`tts-v2` is Akana's TTS front end for the next model: the consensus of the Turkish phonology
literature. [`tts-v1`](phonemizer-tts-v1.md) stays frozen for EchoCraft's shipped voice and is
largely eSpeak-compatible.

`tts-v2` adds three things:

- the segments the literature describes instead of eSpeak NG's allophones;
- its stress rules;
- every rule of the verbaliser.

Its `ph` symbols are the phonemes' IPA.

**Briefs:** EchoCraft `experiments/2026-11-07-akana-phonemiser-brief.md` (stage 3), and
`experiments/2026-11-08-tts-v2-consensus-update-brief.md` (the consensus update).

**Owner's decisions (2026-11-08):**

- `tts-v2` is updated in place, because it never shipped.
- Rules follow the literature consensus. Modern pronunciation is adopted where the consensus
  clearly backs it.
- The teacher TTS's measurements are context, never the reason for a rule.

**Sources:** the ids L1–L23 refer to the source table at the end, taken from EchoCraft's
`experiments/2026-11-08-turkish-phonology-consensus.md`.

## Summary

- **Interface:**
  - Python: `akana.phonemize(text, profile="tts-v2")`; Rust: `Profile::TtsV2`; version string
    `tts-v2.0`.
  - The token schema is `tts-v1`'s: `ph`, `stress`, `word`, `lengthen`, `mark`, `type`, `ipa`.
  - The default profile is still `tts-v1`.
- **Contract:**
  - `ph` is the IPA without a length mark.
  - Pauses are `_`, `_:` and `_!`. Every pause `ph`, and no other, starts with `_`.
  - Every `ipa` is exactly one entry of the aligner vocabulary: true for all 436,958 tokens of
    the 15,379 fixture texts.
  - `stress` keeps `tts-v1`'s prominence levels (7 the nucleus down to 0 a clitic), and so do
    the pauses and marks.
- **Differences from `tts-v1` on the fixture texts:** 16.4% of tokens change their `ipa` and
  0.32% their stress; 3 of the 15,328 corpus clips read other words (`vb.`, `online`,
  `Shakespeare'i`).
- **Latency:** median about 85 µs per 50–70-character sentence, about 33,000× faster than real
  time.

## Segments

`tts-v2` starts from `tts-v1`'s letter rules. `speech::letters::tts_v2_segments` changes them, and
`speech::tts` applies the rules that need the word's analysis or its place in the phrase.

| rule | sources (strength) | `tts-v1` → `tts-v2` on the fixtures |
|---|---|---|
| vowels are tense inside the word: no closed-syllable laxing, which is the older system and receding; no lax o | L7, L6 (medium) | ɪ → i 19,971; ʊ → u 8,254; ɔ → o 6,296; ɛ → e 5,893 |
| a word-final open /e/ is [ɛ] in every word; high /i u/ are lowered only in the open final syllable of a phrase (before a clause pause `_:` or at the end of the text, not before the optional `_!` of a conjunction) | L1 word-final (Göksel & Kerslake 2005, 2025), L6; L3 phrase-final (medium) | *dede geldi* [dedɛ ɟældɪ] |
| ü is [y]; its lowered [ʏ] is not in the aligner's vocabulary | L3 | ø → y 4,135 |
| /e/ is [æ] before a tautosyllabic coda m, n, l or r, syllabified within the word (*gel* [ɟæl], *gel-ir* [ɟeliɾ]); not before /j/ | L6 (9 speakers), L7, L8, L4 (strong) | ɛ → æ 3,164 |
| the negative aorist *-mAz* and *pekmez* also take [æ] before their coda z: *gelmez* [ɟælmæz]; *merkez* varies and is left | L6 (all 9 speakers), L7 (strong) | |
| *hem*, *elbise* and *Belçika* keep [e] in the modern standard, where the root itself closes the syllable (*elbisem* keeps [æ] in *-sem*); *kendi* and *bel* vary and follow the rule | L6, L1 (medium) | æ → e 57 |
| /ɾ/ is a tap everywhere: word-initially, after and before consonants, and word-finally before the next word. It is devoiced and fricated [r̝̊] only phrase-finally: before a clause pause `_:` or at the end of the text, not before the optional `_!` of a conjunction | L13 (10 speakers: utterance-final devoicing 0.63–0.82, word-final before a voiced sound 0.01, before a voiceless consonant about 0.24 and optional), Nichols 2016 via L13, L3, L7 (strong) | r → ɾ 14,857; r → r̝̊ 2,586 |
| no fricated [r̝]: no acoustic source supports it | L13; Yavuz & Balcı 2011 (via L23) is unsupported | |
| /k g/ are palatal [c ɟ] when tautosyllabic with a front vowel: before one in the onset, after one in the coda (*ekmek* [ecmec], *teklif* [teclif], *gök* [ɟœc]); velar before a back vowel (*ekonomi*, *dekor*, *ekol*, *dekan*). Lexical palatals in loans: *kâr*, *rüzgâr* (and clear l in *hal*, *rol*, *kalp*) | Clements & Sezer 1982 via L15 (medium) | k → c 6,341; kː → cː 67; ɡ → ɟ 10 |
| a circumflex vowel (an underlying long vowel of a loan) is long only in an open syllable: long vowels shorten in closed syllables | L15 (van der Hulst & van de Weijer 1991, after Underhill) (medium) | aː → a 49; uː → u 11 |
| ğ lengthens the vowel before it; between vowels a hiatus (the first vowel short), one long vowel between equal vowels; [j] between front vowels | consensus row 14: L16, L1 (medium; the hiatus is one of the realisations the consensus allows) | as `tts-v1` |

**Types:** the tap is `type` `liquid` in `tts-v2` (`tts-v1` keeps `vstop`); the devoiced r̝̊ is
`fricative`.

**Documented, not marked:**

- High-vowel devoicing between voiceless consonants is real and gradient (L10: 13.8% of high
  vowels in read words; L11; L12). It is phonetic and predictable from context, so no token or
  field marks it.
- /ø/ is lowered before a coda sonorant (*göl*; L4, L7). `tts-v2` writes every ö as `œ`, the
  aligner's Turkish symbol, so the split cannot be shown.
- Final /h/ as [x ç] and /v/ as [β] next to rounded vowels vary by speaker (L3, L1).

**Not changed, and why:**

- /e o ø/ are phonetically mid [e̞ o̞ ø̞] (L3), but `tts-v2` writes the broad `e`, `o`, `œ` that
  the aligner learned for Turkish.
- ı stays `ɯ`, though L3 describes it as [ɨ].
- The aligner has no voiceless [l̥], so final and preconsonantal /l/ stays voiced.

**Context, not evidence:** EchoCraft measured two synthetic voices (yazgi-tts-1 and Trendyol-TTS)
on these rules. They agree with the literature on /ɾ/. They keep the older closed-syllable
laxing, which the literature finds receding (L7), so `tts-v2` does not follow them there. Run
EchoCraft's CTC alignment score on `tts-v2` tokens against `tts-v1` before training on them.

## Stress

`tts-v2` keeps `tts-v1`'s rules: final by default, lexical exceptions, pre-accenting and stressed
suffixes, clitics, and the prominence hierarchy. It adds the rules below; the stressed syllable is
in capitals.

| rule | examples | sources (strength) |
|---|---|---|
| question words and *-rA* forms are initial | NErede, NEreden, NEresi (`tts-v1` read *nerede* finally) | Dursunoğlu 2006 via L23; L2 (strong) |
| adverbs in Arabic *-en/-an* are pre-stressing | KISmen, ŞAHsen, ASlen, RESmen, nisPEten, FIIlen, taMAmen, AYnen, eSAsen, mecBUren, takRIben, tahMInen. Not *hakikaten*: no source gives its stress, and the rule and its long vowel point to different syllables | L2 (Göksel & Kerslake 2025: *tamámen*), Banguoğlu via L17 (medium) |
| equative and adverbial *-CA* are pre-stressing (diminutive *-CA* is stressed) | çoCUKça, ayRIca, kaDINca, TÜRKçe | L2; Gencan via L17 (medium) |
| *-(y)Iver* is stressed on its first syllable | yaPIver | L2, L17, L18 (strong) |
| *-leyin* is pre-stressing | saBAHleyin, geCEleyin | L18, L17 (medium) |
| place names follow Sezer's rule: the antepenult if it is heavy and the penult light, else the penult | ANkara, iSTANbul, eDIRne, deNIZli, ERzurum, BELçika | L20, L17, L2 (medium) |
| *-iye* names are pre-stressing; *-istan* is final | HARbiye, süleyMAniye, TÜRkiye; hindisTAN | L17, L2 (medium) |
| names in *-CIlAr* follow Sezer's rule; final *-lAr* is dropped | doĞANcılar, deMIRciler | L17 (Tekin), L20 (medium) |
| closed compounds are stressed on the first element; listed final exceptions | BUgün, BAŞbakan, BAŞkent; final: bilgisaYAR, alışveRİŞ, kabakuLAK | L2; Kabak & Vogel 2001 via L23 (medium) |
| intensive reduplication stresses the prefix; a closed list of the standard intensives (52) and the words built on them (*kapkaranlık*), since the pattern alone also matches ordinary words (*araba*, *durdurdu*, *serseri*). The prefix vowel follows the ordinary lowering (TERtemiz [tæɾ]), except where the prefix and base vowels match in İleri's data: BEMbeyaz [bem], PESpembe [pæs] | KAPkara, BEMbeyaz, SAPsarı, TERtemiz, MASmavi, YApayalnız, Apaçık, PESpembe | stress: L2, L17 (strong); prefix vowels: L8 (İleri 2025, slide 2: *bem-be.jaz*, *pæs-pæm.be*, *tær-te.miz*; a single source) |
| relativiser *-ki* stressable; clitic *ki* pre-stressing | evdeKİ; gördÜM ki | L2, L18 (medium) |
| pronominal *-ki* after a genitive is pre-accenting | seNİNki, beNİMki | Kabak & Vogel 2001 only (single source, weak) |
| *yeniden* is penultimate | yeNİden | Banguoğlu via L17 only (single source) |

**When a word counts as a name.** Only its letters before the apostrophe count.

- **Place names (Sezer's rule and *-istan*):**
  - a gazetteer place the morphology reads as a proper noun;
  - a place written with a capital, not at the start of a sentence, line, list item or
    quotation;
  - a place with an apostrophe;
  - a province with a capital even at the start of a sentence (*Denizli*), unless the lexicon also
    has it as a common word.
- ***-iye* and *-CIlAr*:** only in a proper noun, a gazetteer place, or a word with an
  apostrophe.

These words are therefore not names:

- a capitalised common word at the start of a sentence, after a colon or line break, or written
  in capitals throughout: *Satıcılar indirim yaptı*, *Başlık: Satıcılar*, *BU SATICILAR*,
  *Bu Belediye*;
- a bare *Doğancılar*. *Doğancılar'a* gets Sezer's rule.

**Departure from the brief:**

- **A bare *Doğancılar*:** the brief expected [doˈandʒɯɫaɾ]. The morphology reads it as the
  common noun *doğancı-lar* ("falconers"). At the start of a text it cannot be told from
  *Satıcılar indirim yaptı*, so it keeps final stress. With an apostrophe (*Doğancılar'a*) it
  takes Sezer's rule.

**Context, not evidence:** the teacher's F0 peaks fall on `tts-v2`'s syllable of *nerede* in 76%
of 66 words. They lean final on *-ki* after a genitive and on *-leyin*, where the literature
decides.

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
- **Diminutive *-CA*** (*büyükçe*) should be stressed, but the morphology analyses it as the
  pre-stressing *-CA*.
- **Lexicalised *-mAz* nouns** (*tükenmez*) are single roots in the morphology, so their /e/
  follows the general rule rather than the *-mAz* lowering.
- **Variable lowering:** the words the literature calls variable (*kendi*, *bel*, *merkez*)
  follow the rule. A screen reader's neutral register could keep them at [e]; that is a style
  choice.
- **Foreign words:** a table and letter rules, not a full loanword reader.

## Sources

How they were read is EchoCraft's: "full text", "summary", or "via" another source. Page
numbers are those the sources give. Check the originals before citing their claims further.

| id | source | read | used for |
|---|---|---|---|
| L1 | Göksel & Kerslake 2005, *Turkish: A Comprehensive Grammar*, Routledge | via L6, L7, L10 | /e/ allophones, word-final lowering, [β] |
| L2 | Göksel & Kerslake 2025, 2nd ed., summarised at halbuki.org/turkish-word-stress | full summary page | stress classes, pre-stressing and stressed suffixes, adverbs, compounds, place names, intensives |
| L3 | Zimmer & Orgun 1999, Turkish, *Handbook of the IPA*: 154–158 | via L23 | phrase-final lowering, the tap, final [x ç] |
| L4 | Kornfilt 1997, *Turkish*: 512 | via L7 | /e ø/ lowered before coda sonorants |
| L6 | Dadan, Erken & Canalis 2024, /e/ lowering in Turkish, *Proc. TU+9*: 121–134 | full text | [æ] before coda sonorants, *-mAz*, *pekmez*, *hem*, *elbise*, *Belçika* |
| L7 | Gopal & Nichols 2022, apparent-time change in Turkish front mid vowels (LAGB slides) | full text | receding closed-syllable laxing, /ø/, *bir* [biɾ̥] |
| L8 | İleri 2025, ICTL 22 and CLS 62 slides | full text | the [æ] rule, vowel identity in intensive prefixes |
| L10 | Sarmış & Canalis 2021, high vowel shortening, *Proc. TU+6* | full text | devoicing of high vowels |
| L11 | Jannedy 1995, *OSU WPL* 45: 56–84 | summary | gradient devoicing |
| L12 | Ay & Bekâr 2015, *Ankara Papers in Turkish and Turkic Linguistics* | full text | devoicing of ı |
| L13 | Sanıyar & Tunçer 2022, continuant devoicing in Turkish, *Proc. TU+7* | full text | /ɾ/ devoicing by context; Nichols 2016 |
| L15 | van der Hulst & van de Weijer 1991, Topics in Turkish phonology | full text (sections) | Clements & Sezer 1982 on palatals; long vowels shortened in closed syllables |
| L16 | Ünal-Logacev, Żygis & Fuchs 2019, soft g in Turkish, *JIPA* | summary | ğ as lengthening |
| L17 | Tekin 1991, Türkçede kelime vurgusu kuralları, *Dilbilim Araştırmaları* | full text | adverbs, place names, *-iye*, *-CIlAr*, compounds, intensives; quotes Gencan, Banguoğlu, Ergin |
| L18 | Özçelik 2014, Prosodic faithfulness to foot edges, *Phonology* 31: 229–269 | full text (sections) | stressed and pre-stressing suffixes, *-leyin* |
| L20 | Sezer 1981/1983, as stated in secondary sources | summary | place-name stress |
| L23 | Wikipedia, *Turkish phonology*, accessed 2026-11-08 | full text | a map to sources (Dursunoğlu 2006, Kabak & Vogel 2001, Yavuz & Balcı 2011) |

**Single-source items:**

- pronominal *-ki* (Kabak & Vogel 2001);
- *yeniden* (Banguoğlu);
- the prefix vowels of *bembeyaz* and *pespembe* (L8).
