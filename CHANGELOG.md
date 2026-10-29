# Changelog

## 0.9.0 (unreleased)

The morphology engine is rebuilt on Zemberek's design, the sparse and hybrid encoders now have
a version and a compatibility check, and the bindings report character offsets. Several
results change, and indexes built with 0.8.0 must be rebuilt: see *Migrating from 0.8.0*.

### Migrating from 0.8.0

**Indexes.**
- Rebuild every sparse or hybrid index built with 0.8.0. Term ids are 32-bit now, the terms
  are disambiguated lemmas, and their weights changed.
- Store `akana.encoder_info()` (Rust: `embeddings::encoder_info_json()`) with each new index.
  When the index is opened, pass the stored value and what the index holds (`"sparse"`,
  `"dense"` or `"hybrid"`, the default) to `akana.check_encoder()`
  (`embeddings::compatibility_of`). It returns one of three values:
  - `"identical"`: built from the same inputs as that kind of vector needs (resources, the
    source of its encoders, akana's own `Cargo.lock`, the Unicode version): the vectors are
    the same.
  - `"compatible"`: the same sparse spec major (sparse), the same dense model, tokenizer and
    weights (dense), both and the hybrid layout (hybrid); the vectors may differ within the
    bump policy of `embeddings/spec.rs`, and no rebuild is needed.
  - `"rebuild"`, also for a record `encoder_info()` did not write.
  The stored value may be the dict or its JSON text, with fields of the index's own and
  numpy numbers; fingerprints compare without regard to case.
- An index without stored info should be treated as `"rebuild"`.
- What the rebuild is worth, measured with `scripts/encoder_churn.py` (0.8.0 against 0.9.0,
  the same texts; exact search unless HNSW is named): every sparse and hybrid vector changes
  (court chunks, RagTurk and Turkuaz alike). A 0.8.0 hybrid index queried with 0.9.0 loses
  MRR@10 (RagTurk 0.738 against 0.790 rebuilt, Turkuaz 0.372 against 0.439), and a 0.8.0
  sparse index (its `sparse_embed` vectors, or a BM25 index of their terms) finds nothing:
  MRR@10 0.000 on both, as every term id changed. Rebuilt, retrieval improves; MRR@10 per
  question, paired, with 95 % intervals: BM25 +0.034 [+0.017, +0.052] on RagTurk and +0.024
  [+0.009, +0.038] on Turkuaz; the 128-byte hybrid +0.035 [+0.015, +0.055] on RagTurk and
  +0.008 [-0.007, +0.023] on Turkuaz, within noise. In the benchmark's HNSW index the hybrid
  gains +0.031 [+0.010, +0.051] and +0.037 [+0.013, +0.060]: on Turkuaz the index finds the
  new vectors' neighbours better than 0.8.0's (0.8.0 scores 0.431 by exact search and 0.407 in
  HNSW). Dense vectors are unchanged and RRF moves within noise (0.8.0's RRF was not
  deterministic).
- Sparse spec 1.0 and analysis version 1.0 are frozen with this release: later releases
  report their vector changes against them under the bump policy.

**Offsets (Python).** NER `start`/`end`, grammar `start_offset`/`end_offset` and style
`start`/`end` are character offsets now, where 0.8.0 gave byte offsets. Remove any byte-to-
character conversion. The CLI's JSON still reports byte offsets.

**Verb lemmas.** A verb's lemma is its infinitive, as in Zemberek: `okudu` gives `okumak`,
where it gave `oku`. This holds in `Morphology.analyze`, `Disambiguator`, `analyze()` and
`parse_conllu`, the CoNLL-U LEMMA column included. `root` keeps the stem (`oku`), and
`formatted` shows the infinitive (`[okumak:Verb] oku:Verb+Past`). Code that matched verb
lemmas against stems should match `root`.

**Readings.** `Morphology.analyze` returns other parse sets than 0.8.0:
- Duplicates are gone: `geliyorum` gives 1 parse where it gave 2, and `gideceğim` 3 where it
  gave 4 (the future, and the participle as a noun and as an adjective).
- Misspellings are rejected (`kitapda`, `gelti`).
- Missing cells are there (`gelmem`, `bana`, `suyunu`).
- Tags follow the morphotactics: `burnum` also reads as `CopPres1sg`, and `çalışkan` lost its
  `Verb+DevGen` readings.

- Parses are sorted by `score`, which is now the probability of the reading given the word,
  times a spelling factor; a word's scores sum to at most 1.
- The parse dicts of `Morphology.analyze` and `Disambiguator.disambiguate` have new keys:
  `analysis`, the analysis in Zemberek's format; `pos`, the POS after derivation; and
  `display_lemma`.
- Participles and -IcI forms read both as a noun and as an adjective, which share
  `formatted` but differ in `pos` and `analysis`: `olduğu` is `PastPart→Noun` "that it is" or
  `PastPart→Adj` "which it is", and `bağlayıcı` is `Agt→Noun` or `Agt→Adj`. The noun comes
  first. The disambiguator picks the adjective before the noun the form describes. Code that
  expects one parse per `formatted` string should key on `analysis`.
- `display_lemma` gives the dictionary's spelling of the lemma: `Ankara`, `kâğıt`, `Hakkâri`,
  `e-posta`. `analysis` uses it, as Zemberek does (`[Ankara:Noun,Prop] ankara:Noun+A3sg`).
  `lemma` stays the folded, lower-case key (`ankara`, `kağıt`) that sparse terms use.
- Spellings that usage has but TDK does not, such as old orthography (`karariyle`,
  `olmıyan`, `vermeğe`), read with the modern lemma (`karar`, `olmak`, `vermek`), at half the
  score, with the new key `nonstandard` set. The spell checker, the grammar rules and the
  deasciifier ignore such readings: `kitapı` and `olmıyan` are still misspellings to them.
  `Morphology.analyze` returns them; code that judges spellings should skip them, as
  `TurkishMorphology::analyze_standard` does in Rust. `analyze_standard` also leaves out what
  the analyzer reads only as text is found (a name without its apostrophe, `Türkiyede`; a capital
  I typed for İ, `KITAPLAR`, but not a foreign acronym's, `FBI`; a numeral written as one word,
  `onbeş`) and a few rare productive paths that mostly read misspellings of frequent words
  (`kedilen`, `maddemsi`, `tarafınsan`).
- Misspellings Zemberek lists as words read the same way, with the TDK lemma (`şöför`:
  `şoför`, `döküman`: `doküman`, `orjinal`: `orijinal`, `ünvan`: `unvan`; `herşey` keeps its
  lemma), and so do the words Zemberek refers to another spelling (`motorsiklet`:
  `motosiklet`). A spelling TDK refers to the other takes its lemma without being
  nonstandard (`hükümet`: `hükumet`, `rızk`: `rızık`, `kast`: `kasıt`). Inflected forms listed
  as words read as the word's (`mevkii`, `bayii`, `zayii`: P3sg).
- Corpus "names" that were common words without their Turkish letters (`egitim`, `ayrica`,
  `Dogan`) or ASCII-I twins of a name (`ıstanbul`) are gone: `Istanbul` reads `istanbul`, and
  the spell checker no longer accepts `egitim`.
- Abbreviations read as the court inflects them, by the last word of what they stand for as
  well as letter by letter: `SGK'nun`, `ATK'nun` (Kurumu), `SPK'nun` (Kurulu), `TFF'nun`,
  `TK'nun`, `İİK'un`, `TCMB'nın`, `YTL'nın`, `TCY'nın`, `CBS'nca`, `MR'da`, `SMS'te`; `TİKA`,
  `psikomotor`, `ayrıksı` and `acenta` (lemma `acente`) are new. Of the apostrophe forms of
  lexicon abbreviations in the court samples, the analyzer rejects 128 tokens, mostly typos,
  where it rejected 306 (`scripts/audit_abbreviations.py`).

**User dictionaries.** `Morphology.load_dictionary_str` and `load_dictionary_file` (Rust:
`TurkishMorphology::load_dictionary_str`/`_file`, `RootLexicon::load_from_str`/`_file`) return
the lines they could not read in full, where they returned nothing, and such a line adds
nothing:
- an unknown POS (0.8.0 read it as a noun), attribute, secondary POS or field (a misspelt
  `A:Voicng` was dropped silently), a missing column, an unclosed `[`, text after `]` other
  than a `# comment`, a voice attribute on a word that is neither a verb nor a pronoun,
  attributes that contradict each other (`Voicing` with `NoVoicing`, `AoristA` with
  `AoristI`), or a root with characters no Turkish word has (a ligature, a control character,
  nothing);
- a UTF-8 byte-order mark is skipped, and a line is read as Unicode NFC (a decomposed `ş`
  finds the word).

They also take Zemberek's own syntax (`kriptopara [P:Noun; A:CompoundP3sg;
Roots:kripto-para]`, a verb's infinitive alone), which 0.8.0 misread, and read its lines as
akana's lexicon build does: a capitalised word is a proper noun and never a verb
(`Kızılırmak`), a postposition takes its case (`ait [P:Postp,PCDat]`), the first secondary POS
counts, an acronym's Latin I is i (`IBAN`: `iban`), `Ref:` gives the lemma and spelling of
the word a variant spells while the root keeps the variant (`şöför [A:Nonstandard;
Ref:şoför]`), `Recip` and `Reflex` on a verb are its voice (not a secondary POS), and
Zemberek's attributes that akana does not use (`Ext`, `NoQuote`) are allowed. Every line of
Zemberek's dictionaries that the build reads as it is loads as the same item.

**Syntactic morphology.** `SyntacticMorphology` and `syntactic_analyze` report UD features:
`Number`, `Person`, `Number[psor]`, `Person[psor]` and `VerbForm`, where 0.8.0 had
`PersonNumber` and `Possessive`. `Possessive=Pnon` is gone. The inflectional groups come from
the analyzer's own morphotactic path, and `TurkishSyntacticMorphology` has no lexicon of its
own (`google_lexicon.txt` was removed).

**Grammar.**
- `check_grammar` and `correct_grammar` run the 26 rules measured precise on correct text.
  The 7 experimental rules are off: enable them with `experimental=True`, or pick rules with
  `rules=`, `enable=` and `disable=`. `grammar_rules()` lists every rule with its id,
  category, default state and confidence.
- Findings carry a `rule` id, and `GrammarFinding.__repr__` shows it instead of the category.
- In Rust, `GrammarFinding` has a `rule` field. Findings serialized by 0.8.0 still
  deserialize: they get the first rule of their category.

**Generation.** `Morphology.generate` uses the instance's own lexicon. It returns `None` for
an unknown tag or a form the morphotactics cannot build, where 0.8.0 echoed the lemma.
- Tags match in any case (`noun`, `P1SG`).
- A name or an abbreviation comes back as the dictionary spells it, with the apostrophe
  before its case, possessive and copula, and suffixes that follow its pronunciation
  (`Türkiye'de`, `Ali'nin`, `ABD'nin`, `TBMM'de`, but `Ankaralı`), or as the request does when
  that differs only in capitals (`NATO'nun`, not `Nato'nun`). A request written with a
  capital prefers the name (`ABD` the abbreviation, `abd` the common noun); a common word
  asked for with a capital keeps it (`Yönetmeliğin`), in capitals all of it (`KİTABA`).
- The lemma may be spelt as `display_lemma` gives it or folded (`kâğıt` or `kağıt`,
  `e-posta`, `Fin-Ugor`, `FIFA` with a Latin I); the word comes back folded (`kağıtta`), a
  name as the dictionary spells it (`Hakkâri'de`). Among homographs the one spelt as asked
  comes first (`dâhil`: `dahilinde`).
- A case the word may leave implicit is shown (`karşı` with `Dat` is `karşıya`), and a
  pronoun's instrumental is the full form (`bununla`, `benimle`, not `bunla`, `benle`).
- A reading's `generation_tags` (a new key of the parse dicts; Rust:
  `MorphParse::generation_tags`) are its `morphemes` with the progressive named, `Prog1` or
  `Prog2`, so `generate(lemma, generation_tags)` gives the word back (`bakmakta`, which its
  `morphemes` gave back as `bakıyor`).
- `Prog` is the -Iyor progressive, also on a stem that changes (`gidiyorum`, `ediyor`,
  `diyor`), where it gave -mAktA (`gitmekteyim`, `demekte`), except before the copula -DIr
  (`CopPres3sg`), where written Turkish has -mAktA (`gelmektedir`, not `geliyordur`). The
  request tags `Prog1` and `Prog2` ask for one of the two (`Prog2`: `bulunmaktadır`); both
  still analyse as `Prog`. After a verb the copula is `CopPres3sg`, `CopPast`, `CopEvid` or
  `CopCond`, as `analyze` shows it.
- A CompoundP3sg noun's plural keeps its own possessive (`cumhurbaşkanı` with `A3pl+Dat` is
  `cumhurbaşkanlarına`).

**Errors (Python).**
- A panic in the native code raises `RuntimeError`, where it raised
  `pyo3_runtime.PanicException`.
- If the native extension fails to load, `import akana` raises `ImportError`, where 0.8.0
  imported and failed later.

**Rust API.**
- `AkanaEngine` no longer has the public `spell_checker` and `grammar_checker` fields;
  construct `TurkishSpellChecker` and `TurkishGrammarChecker` directly, which share the
  built-in lexicon.
- The `morphology::graph` and `morphology::suffixes` modules (`MorphState`,
  `SuffixTransition`, `SuffixType`, `TurkishMorphotactics::transitions()`) are replaced by the
  new morphotactic graph.
- `MorphParse` has the new fields `analysis` and `nonstandard` (both `#[serde(default)]`)
  and the method `display_lemma()`; `TurkishMorphology::analyze_standard` returns the readings
  without the nonstandard ones.

**Other outputs that changed with the analyzer.** `stem`, `extract_keywords`, the
readability metrics (Kalyoncu word lists and grade), and PII masks (leak fixes) give
different results.

**Dependency trees.** `parse_conllu`, `analyze()` and `TurkishDependencyParser` give other
trees, closer to UD Turkish: on the UD Turkish test splits (`scripts/parser_ud_eval.py`) UAS
rose from 48.5 to 55.9 and LAS from 34.8 to 41.9.
- `değil` negates the nominal before it, which is the predicate (`AUX`, `aux`, as the UD
  Turkish test splits label it 189 times against 27 `cop`): `Ali öğretmen değil .` is rooted
  in `öğretmen`, and in each conjunct so (`Sanık suçlu değil , masumdur .` is rooted in
  `suçlu`).
- Clauses joined by a comma or a conjunction are conjuncts: the first predicate is the root
  and heads the others (`conj`), and each word attaches to its own clause's predicate.
- New relations: `compound` for a bare noun before a light verb (`karar verdi`, `tebliğ
  edildi`, also passive, as the UD treebanks have it), `nummod`, and `flat` for the
  later words of a name (`Ahmet Yılmaz'ın`, `Mustafa Kemal Atatürk`): names side by side
  join up to a word with a case, which joins only a person's name, so `Ankara Türkiye'nin
  başkentidir` has two. A
  participle the dictionary lists as an adjective (`gelen`) is a `VERB` `acl`, and a modifier
  reaches its noun past determiners and adjectives (`Bilinmeyen bir sebeple`).

**AI-text detector.** Its model is unchanged. It reads the shared lexicon and the sentence
segmenter, so this release moves some of its inputs on most texts: `oov_rate` and the
sentence statistics. On a 1,928-text sample no verdict changed and the probability moved
by at most 0.034. The model will be retrained on this release's inputs. A drift canary now
records the inputs on 160 texts (court paragraphs, and questions, exclamations, ellipses,
assistant-style answers and other shapes court text lacks) and the lexicon word sets, and
fails when one moves, so each such change is recorded on purpose
(`tests/data/ai_detector_canary.jsonl`); a change that moves inputs only on texts unlike all of
them passes it.

### Added

- Every public class, function and method of the Python module has a docstring (116 of 148
  had none), and a test keeps it so.
- `akana.encoder_info()` and `akana.check_encoder(stored, kind)`; in Rust, `embeddings::spec`
  (`SPARSE_SPEC`, `HYBRID_LAYOUT`, `DENSE_MODEL`, `ANALYSIS_VERSION`, `encoder_info`,
  `encoder_info_json`, `IndexKind`, `compatibility_of`). The record carries the akana and
  Unicode versions and three fingerprints: sparse (the analyses, the term tables, stopwords
  and compound rules, the source of the sparse path, akana's own `Cargo.lock`, the Unicode
  version and the probe set's sparse vectors), dense (the model, its tokenizer and weights,
  the source of the dense path, the lock, the Unicode version, the probe set's dense vectors)
  and the hybrid one over both and the packer. A build script hashes each path's source, so
  code a probe text does not exercise still changes the fingerprint of that kind only, and a
  build without akana's own lock file (another project's dependency) gets a fingerprint of
  its own. The probe set includes characters the dense tokenizer normalises (`…`, `ﬁ`, `½`,
  fullwidth letters).
- `MorphParse::analysis`, `MorphParse::nonstandard`, `MorphParse::display_lemma()`,
  `TurkishMorphology::analyze_standard` and `TurkishMorphology::backoff_stem`, and the Python
  keys `analysis`, `pos`, `display_lemma` and `nonstandard`.
- Grammar rule selection (`rules`, `enable`, `disable`, `experimental`) and `grammar_rules()`.
- `morphology::verb_form` and `verb_form_of` (Rust): a reading's non-finite verb form and its
  UD `VerbForm`.
- Measurement scripts: `scripts/ud_eval.py` (the analyzer on held-out UD Turkish data, by
  treebank and genre), `scripts/benchmark_embedding_cost.py`, and `--readme` in
  `scripts/benchmark_rag_real_data.py`, which prints the README's retrieval table. The
  retrieval benchmark caches Turkuaz on disk after the first run and has an exact FP32 row.
  The README's retrieval, morphology and embedding-cost figures are regenerated with them.
  The benchmark reads the 300 RagTurk articles pinned in `scripts/ragturk_articles.tsv` (name
  and SHA-256) and fails with `--readme` instead of falling back to its synthetic corpus.
  `scripts/encoder_churn.py` measures sparse and hybrid churn and stale-index MRR@10 per
  corpus on pinned texts (it refuses other or missing data unless told `--unpinned`), classes
  a change of the dense model, tokenizer, weights, stopwords or compound rules major, exits 3
  above a `--declared CLASS` and 4 when a kind of vector moved under the same fingerprint.
  `scripts/parser_ud_eval.py` measures the dependency parser on the UD Turkish test splits
  (UAS and LAS per treebank and per relation, beside attaching each word to the next word
  that is not punctuation; a sentence it cannot score counts as wrong; `--min-las` fails
  below a floor).
- THIRD_PARTY_NOTICES in the repository, the crates and the Python package. It covers the
  Zemberek dictionaries and morphotactics (Apache 2.0), and the UD-derived function-lemma
  table, morphology prior and the parser's light verbs (CC BY-SA 4.0).

### Changed

- **Morphology.**
  - The analyzer follows Zemberek's morphotactics and stem transitions.
  - Its lexicon is built from Zemberek's dictionaries with their pronunciations, aorist
    classes and compounds, plus akana's legal supplement.
  - Analyses are ranked by a corpus prior, in integer arithmetic, so that every platform
    ranks alike. The prior is estimated on court decisions mixed with general text (the UD
    Turkish treebanks' sentences), and a word with readings in more than one part of speech
    (`değil`, `ile`, `ilk`, `insan`) is split among them as the treebanks tag it. Imperatives
    and other second-person readings are no longer ruled out by court style; the
    disambiguator prefers them in a sentence that addresses someone (`!`, a question
    particle, a second-person verb). A capital at the start of a sentence or in a word
    written in capitals no longer makes a common word a name (`Kanun , ...`), nor does a
    title make a derived word one (`İnfaz Hâkimliğinin`). A circumflex in a word is
    evidence for the reading spelled with it (`Resmî`, `şeklî`, `hâkimlik`).
  - The prior records the lexicon and morphotactic graph it was estimated for, and a test
    fails when they no longer match.
  - Real-world input reads: a name written with its capital but without the apostrophe
    (`Türkiyede`, at a lower score), dotted abbreviations (`Dr.`, `TCK.nun`), a capital I typed
    for İ (`KITAPLAR`) and foreign acronyms with `i` lemmas (`fbi`, `covid`), times on the hour
    (`17.00'ye`), distributives on digits (`3'er`), numerals written as one word (`onbeş`),
    lower-case Roman numerals and list letters, old orthography and usage variants of
    alternations (`temerrüte`, as `nonstandard` readings).
  - Forms Zemberek's morphotactics lacked or the port got wrong now read: `deyip`, `deyince`,
    `yiyince` (not `diyip`), a participle's bare dative and instrumental (`olacağa`, `olacakla`),
    `gelmişlercesine`, `okuduklarımızdansınız`, `dolayısı`. A bare -DIk takes no dative
    (`olduğa` is rejected; `ettiklere` and `olacağa` read), and -(y)ken after a verb form makes
    a converb (`gelmişken`).
  - Input as word processors leave it reads: soft hyphens and zero-width characters inside a
    word, ligatures and fullwidth letters (`tedbirine` with a soft hyphen, `ﬁkir` with the ﬁ ligature, `ＫＩＴＡＰ`). Initials read as
    their abbreviation (`A.Ş.`, `T.C.`), and `C.`, `D.`, `L.`, `M.`, `CD.`, `MD.` as
    abbreviations rather than Roman numerals when the lexicon has one, and a single `C`, `D`,
    `L` or `M` is never one; a dotted suffix keeps the abbreviation's own reading (`Şti.nin`).
    An abbreviation of a phrase that ends in a possessive takes the phrase's cases after a
    silent P3sg (`HMK'na`, `CBS'nca`, `Şti'nden`, which read as P2sg). Readings that differ
    only in a silent morpheme are one (`milletvekili`), and so are `-mIşken`'s evidential and
    participle readings (`gelmişken`: Evid+While).
  - Lexicon: court and general words added (`kesinleşmek`, `gerçekleştirmek`,
    `kamulaştırmak`, `satın`, `giderim`, `uyarlık`, `ruhsal`, `evrensel`, `haksız`,
    `istisnaen`, `tevhiden`, `mündericat`, `Katar`, `Yüksel` …); `kasıt` makes `kastı`; the
    readings of `OHAL'in`, `HUMK'nın`, `HGK'nun`, `HSK'nun`, `VUK'nun` and `YİDK'in`; `sakın`
    and `yaşasın` are interjections again; rare items no longer take a frequent word's forms
    (`ekleri` is `ek`, `eleme` is `elemek`). Homographs that shared a prior key have their own
    (`dahil` and `dâhil`). Words the lexicon had only as names read as words (`rücuan`,
    `rızai`, `karz`, `meskeniyet`, `talasemi`, `rapçi`), and acronyms as abbreviations rather
    than corpus names typed with ı (`IBAN`, `BİST`; the build drops such a name). The given
    names and surnames of akana's PII lists that Zemberek lacks as names are names (227:
    `Serpil` read as `serpmek`, `Gökçen` and `Özgen` with a P2sg, `Halide` as `hali-de`; `Hakan`,
    `Metin`, `Kaya` had no name reading, nor did the court decisions' `Basri BAĞCI` (`bağ`),
    `Mehmet ERTEN` (`erte`) or `Kâni Vrana` (`kani`); `scripts/names_from_pii.py` writes
    `akana-names.dict`, leaving out a surname whose capital is mostly a heading's, as in `Temel
    Hak`), and six company words among the corpus names are dropped (`Mimarlık`,
    `Yayıncılık`, `Denizcilik` ...). An abbreviation of one word reads as that word (`md.`,
    `mad.`: `madde`; `Av.`: `avukat`; `Sok.`, `Mah.`, `Cad.`: `sokak`, `mahalle`, `cadde`;
    `San.`, `Tic.`, `Ltd.`, `Şti.`, `A.Ş.`: `sanayi`, `ticaret`, `limited`, `şirket`; `Dr.`,
    `Prof.`, `Doç.`, `Yrd.`, `Müh.`: `doktor`, `profesör`, `doçent`, `yardımcı`, `mühendis`;
    `par.`: `paragraf`), so it is one term with it and no other word's (`av`, `san`, `aş`).
  - Ranking: the treebanks' tags choose between lemmas for `varsa`, `vardı`, `vardır`,
    `doğan`, `örneğin`, `altın`, `bilir` and `çıktı`; the time adverbs `önce`, `sonra`,
    `şimdi`, `bugün` and `daha` keep their time noun apart; an adjective's or a present
    participle's -lArI comes in a noun's order (`diğerleri` A3pl+P3sg). In context, `var` and
    `yok` are their clause's predicate (`Bir sorun vardı .`) unless a dative precedes
    (`eve vardı`), `ile` is the postposition before `ilgili`, `birlikte`, `beraber` and after
    a verbal noun, and a noun's -(s)I after `bu`, `şu`, `o`, `her` or `hangi` is the
    accusative (`Bu filmi beğendim`). A word written in lower case is a name only where it
    reads as nothing else (`mimarlık`, `güler`, `kadri` read as words, where a corpus name
    ranked first; `ankara` stays Ankara), as `analyze_standard` already judged it.
- **Tokenizer.** A number, date or time keeps its suffixes after an apostrophe in one token
  (`1924'te`, `17.00'ye`), as a name does, and an abbreviation in capitals keeps the suffixes
  court decisions write after its period (`TCK.nun`, `HMK.'ya`; not `A.Ş.vekili`). NER
  entities still end with the number. Amounts are numbers, not times (`12.500`, `3.200,00`).
  The apostrophe may be any of ' ’ ‘ ` ´ ′ ＇, and a soft hyphen or zero-width character
  inside a word keeps it one token. A single letter with a period is an abbreviation (`A.
  Yılmaz`, `2019/12 E.`, `(d. 1950)`), except the pronoun `o`, and so are `md.`, `örn.`,
  `HD.`, `CD.`, `Bşk.` and `Müd.`. A byte order mark is no token. A dash right after a word
  or a number is no sign: `2005-2009` is `2005`, `-`, `2009` (it gave `-2009`), and
  `COVID-19` has the number `19`. Initials are one token, bare or with a suffix after their
  last period (`H.M.K.`, `T.C.`, `K.A.nın`); a combining mark stays with its letter, so
  decomposed text is not cut inside a word (`s` + U+0327, a decomposed `ş`); a date ends where
  no digit follows (`15.30-17.00` is two times and a dash).
- **Sentences.** An ellipsis inside a sentence no longer ends it (`davacı ...'nin vekili`,
  `davacı ... vekili`), nor does a court's or paragraph's number before a capital
  (`Ankara 3. Asliye Hukuk Mahkemesi`), a period right before a comma, or a period before an
  apostrophe and its suffix (`2.'si`), which is no closing quote. Nor does a redaction's
  ellipsis before a capital (`Tire ... Asliye Ceza Mahkemesine`) or before a comma
  (`E:..., K:...`), a period and a closing quote before a lower-case word (`belirler.''
  şeklinde`), a Roman ordinal before a capital (`II. Dünya Savaşı`), an acronym's period
  before a lower-case word (`TTK. nun`), or the dash after an amount (`12.500.-TL`).
- **Sparse encoder.**
  - One disambiguated lemma per word occurrence, with 32-bit term ids.
  - The function-lemma weights come from a table frozen with the sparse spec.
  - A term weighs by its lemma's lexicon class, not by the reading the disambiguator picked,
    so a change of ranking no longer moves weights.
  - A word the analyzer cannot read is indexed by its backoff stem (`hiperbariğe` and
    `hiperbarik` are one term; a vowel-final word and its forms are one, `zorbanta` and
    `zorbantanın`), an abbreviation has one term however it is written (`Av.`, `Av`), and a
    suffix written apart (`1924 'te`, `TCK. nun`, `...'nin`, `Sözleşmesi"nin`, `TTK nın`,
    `65 inci`, `%20 si`, `... ın`, `Şti.nin`) is no term. A word the lexicon lacks, written
    with a capital inside a sentence or in capitals, keeps all its letters (`Larnaca`,
    `BOZBOĞA`); at a sentence's start, where a capital says nothing, it gives its letters and
    its backoff stem (`Makrofajlar`: `makrofajlar` and `makrofaj`, which the word gives in
    lower case). An unknown word's suffixes after its apostrophe are no part of its term
    (`Trump'ın` is `trump`, where it was `trump'ın`), initials are their letters (`H.M.K.`
    and `H.M.K.'nın`: `hmk`; `K.A.nın`: `ka`), and a backoff stem that is a function word is
    no term. A numeral's term has no period (`XIX.` and `XIX` are one term), and a numeral of
    one letter is none (`I.`, `V.`), as no one-character term is. Decomposed text gives the
    terms of composed text.
  - A derived word the dictionary lists is its own lemma (`başvurucu`, `uyuşmazlık`,
    `yüklenici`, `gözlük`, `deyiş`), so that its lemma no longer turns with a re-estimated
    prior.
  - A word the dictionary lists as a postposition or an adverb has a fixed term, whichever
    reading its context picks: the word itself (`tarafından`, `içinde`, `yerine`, `göre`), or
    the verb an adverb is a form of (`bilerek`: `bilmek`).
  - Compound parts: a compound that has become a word of its own is no longer split into
    parts that mislead as terms (`anayasa` is no `ana` + `yasa`; `gözaltı`, `olağanüstü`,
    `cephane`, `elbirliği`), and common compounds the lexicon does not mark are
    (`bilgisayar`, `kahverengi`, `ayakkabı`, `bilirkişi`, `kötüniyet`). `decompose` follows
    the same lists.
  - Function words are no terms, decided by the lemma: a lemma whose dictionary items are
    all pronouns, conjunctions, postpositions, determiners or question particles (`ve`,
    `için`, `bu`, `kendi`, `uyarınca`), and `bir`, `değil`, `diye`, `önce`, `sonra`,
    `tarafından` and `üzere`. Before, the stopword list was a list of surface words, and
    whether `tarafından` was a term depended on the reading the disambiguator picked.
- **Version.** Every version string is 0.9.0, and `akana.__version__` is the compiled
  library's. A release tag must name it; the release workflow runs the tests in the release
  profile and the Python tests against each built wheel (on Python 3.10 and 3.12) and against
  the sdist built into a library, and publishes only from a pushed tag (a manual run on a
  tag publishes nothing).
- **Builds.** Cargo.lock is committed, the toolchain is pinned (rust-toolchain.toml), and
  embedded data keeps its line endings on every platform. Golden vectors, a frequent-type
  analysis snapshot and review gold sets fail CI on any unintended change of the encoders'
  output, and downstream snapshots on any change of the parser's trees (with XPOS and FEATS)
  and of `analyze_document` on court texts. `AKANA_GOLD_WRITE` names the fixtures it rewrites
  (`downstream`, `morphology_gold`, `morphology_snapshot`, `review_gold`, `pii_view`,
  `sparse_tables`, comma-separated), so a run of the whole crate rewrites only those. The
  stop lemmas and the lemma weights of the sparse encoder are committed tables
  (`stop_lemmas.txt`, `lemma_weights.tsv`) that a test derives from the lexicon, so a lexicon
  edit cannot move them unnoticed. `golden-1.0.jsonl` keeps the vectors of the release that
  froze spec 1.0 (its fingerprint pinned, at least 150 texts; each spec major gets its own
  file), and CI fails when more than 5 % of them change within the spec. CI also
  regenerates the prior's UD-derived inputs from the pinned treebanks, runs the golden
  vectors in the release profile and from Python, and holds the parser's UD LAS to a floor.
- **Performance.** The quadratic and cubic public paths are linear. The lexicon and the
  embedding weights are built once per process, and the Python bindings release the GIL.

### Fixed

- Crashes on statute headings and non-ASCII plates (PII), upper-case suffixes, non-ASCII
  spaces and overlapping edits (grammar), and cycles or extra roots in dependency trees.
- PII leaks: upper-case Turkish triggers, and text that `restore` would turn into an earlier
  value. Names whose only common readings are copula forms are names again (`İzzettin
  Demir`, `Cindi Yılmaz`), and a role plural mid-sentence no longer joins the name after it
  (`Tanıklar`); the official form keeps its second and its short surnames (`Şebnem NEBİOĞLU
  ÖNER`, `Mustafa BAL`) and ends at the line's end (two people on two lines are two names);
  an initial is masked with its surname (`A. Yılmaz`); common given names no list had are
  added (`Bahadır`, `Cumali`, `Nejdet` …), and a capitalised role before a name no longer hides
  it (`Davacı Kudret Demir`).
- PII over-masks: company names after `Davalı`/`Davacı` (`Hayat Sigorta A.Ş.`, `Axa Sigorta`,
  `Doğan Holding`, `Cihan Medya A.Ş.`) and a company's legal form in a decision title (`LTD.
  ŞTİ. BAŞVURUSU`) are no names; a person's business keeps its name masked (`Ahmet Yılmaz
  İnşaat`). A company word counts only in capitals or as initials (A.Ş.): `Umut Kaya şirketi
  zarara uğrattı` is masked, and a role before a name is no company unless a company word
  follows on its line (`DAVACI : SEDA ARSLANTÜRK`, `Davalı Seda'nın vekili`). An acronym the
  lexicon has is no surname and no name (`Ancak AİHM`, `Ayrıca OHAL`, `ZMMS`), a heading
  letter opens a sentence (`A. Adil yargılanma hakkı`), and the official form's second surname
  stops at a title or a word such as `TARAFINDAN` or `KARAR` (`Mehmet KAYA KATİP Ayşe DEMİR`).
- `PiiVault` serialises in one order on every run: its maps are ordered (`BTreeMap`;
  Rust callers that name the field types must follow), and `from_mapping` reads its pairs in
  placeholder order.
- A plural role noun before a name (`Davacılar Ahmet Yılmaz`) is no longer masked as part of
  the name; a sentence without a verb (`Hava çok soğuk .`, `Evde kimse yok .`) has its predicate
  as the dependency root, never the punctuation, and a postposition attaches to its noun
  (`case`); the de/da rule no longer splits a misspelt locative as a genitive (`aranında`).
- A CompoundP3sg noun derives on its stem without the possessive only: `cumhurbaşkanılık` and
  `cumhurbaşkanılığı` no longer analyse (`cumhurbaşkanlık` does; -(I)msI takes either,
  `zeytinyağımsı`).
- The 52 lexicon verbs Zemberek marks Reciprocal or Reflexive (`atışmak`, `paylaşmak`,
  `karşılaşmak`, `savaşmak`, `yıkanmak`) carry the voice as Zemberek does, an implicit
  `Recip`/`Reflex` in `analysis` (`atış:Verb|Recip→Verb+tı:Past`), where they had it as their
  `secondary_pos`, which belongs to pronouns (`kendi`, `birbiri`).
- The lexicon build fails on a removal or respelling line that matches no entry (three
  removals had silently matched nothing), and a supplement entry with `Ref:` replaces the
  Zemberek entry of its spelling instead of standing beside it.
- `reciprocal_rank_fusion` is deterministic: equal scores are ordered by best rank, then id
  (they came in hash order, so the same inputs gave different rankings), an id listed twice in
  one list counts once, and a `k` that is not positive and finite, or an `alpha` that is not
  finite, is rejected (`ValueError` in Python) where it returned infinite scores.
- Spell checker: its vocabulary holds common words only, so a lower-cased name or abbreviation
  is no spelling (`türkiye`, `varmı`, `cbk`) and no suggestion; `Türkiye` and `TBMM` as written
  still are.
- Deasciifier: a word is judged in lower case first, so a capital at the start of a sentence
  no longer turns it into a name or an abbreviation (`Bu` stayed `Bu`, it gave `Bü`; `Cocuk`
  gives `Çocuk`, not `Çoçuk`); a capitalised word that is a name stays as it is (`Ali`,
  `Michael`); a capital I may be İ (`KITAP`: `KİTAP`, `Istanbul`: `İstanbul`); a word that
  already has Turkish letters and reads, or has them in capitals, is left alone, İ among them
  (`AİHM`, `İSKİ`, `İnsan`, which gave `AIHM`, `IŞKI`, `İnşan`), so the function is
  idempotent; so is a word with a capital after a small letter (`iPhone`, `McDonald`).
- Deasciifier: a spelling is judged by how likely the corpus prior makes it as a word, not by
  how many Turkish letters it has, and the word as written wins a tie, so correct text stays
  (`konusu`, `son`, `olup` gave `konuşu`, `sön`, `ölüp`) and `kisi`, `oldugu` give `kişi`,
  `olduğu`, not `kışı`, `öldüğü`. A suffix written apart from its word takes the spelling the
  word reads with: after an apostrophe (`Kanun'un` gave `Kanun'ün`), after an abbreviation's
  period (`İİK.nun`) and, where it reads so, after a space that follows a numeral or an
  abbreviation (`10 uncu`, `TTK nun`). The question particle takes the vowel of the word
  before it (`değil mi`, `var mı`; `bu mudur` gave `bu müdür`). A letter alone, a Roman
  numeral in capitals (`II.` gave `İİ.`) and the words of an address (`Orbat.com`,
  `ali@gmail.com`) stay as written. Measured with `scripts/benchmark_deasciifier.py` on the
  1,500 committed clean court sentences: of the words of correct text it changed 2.98 % (40.2 %
  of the sentences), now 0.03 % (0.5 %); of the words typed without their Turkish letters it
  restored 94.07 %, now 97.93 %. On 3,000 RagTurk sentences (held out): 3.16 % (32.7 %) now
  0.72 % (8.4 %), restored 85.80 % now 90.04 %. What it still changes is a word with a likelier
  twin (`su`: `şu`, `ayni`: `aynı`, `Koyu`: `Köyü`) and foreign names (`Minor`: `Minör`).
- Grammar: the de/da rule writes `onu da` for `onu'da` (it kept the apostrophe), cases the
  clitic like the word (`KİTABI DA`) and checks words written in capitals; a genitive with -da
  written on it is split again unless a frequent word of another root is one letter away
  (`ülkeninde`, `seninde`, `dosyanında` are split; `aranında`, `alınında` are not, nor
  vowel-dropped to `alnında`), which also removes the rule's cost on long tokens; a merged
  question particle is split after a name (`Ahmetmi`, `Türkiye'denmi`) and at the start of a
  sentence (`Varmı?`). A lower-case word with the clitic after an apostrophe is left alone
  as part of a foreign name only when an unknown capitalised word stands within three words
  before it (`Rossii po futbolu'da`), not after a known name (`Dün Ayşe bunu'da`: `bunu
  da`). `tests/data/grammar_recall.tsv` pins the exact corrections.
- Readability: a paragraph's number or letter alone is no sentence (court decisions number
  their paragraphs); the familiarity list is read without circumflexes (`hikaye` and `hikâye`
  are familiar alike), and a postposition or adverb is familiar by the noun it is spelled
  from (`tarafından`, `içinde`); a plural verbal noun is no fiilimsi (`ödemelerin`); a text
  with no word gets "Yetersiz metin (sözcük yok)", 0 instead of confident grades and no
  sentence; a letter with its period (a heading's `A.`, `s.`) is no word; the reported
  Ateşman score stays within 0-100, and a Kalyoncu score is never below 0 (an 8,000-word
  sentence gave K1 -5,236; the four are linear regressions, so K1 exceeds 100 on ordinary
  court text and is reported so). The disambiguator, and readability with it, is
  linear on long runs of adjectives without a noun or a sentence end (`Ali` × 16,000 took
  27 s).
- The hybrid metric's alpha: NaN made every hybrid distance NaN. In Python a NaN or infinite
  alpha raises `ValueError`; in Rust `set_hybrid_metric_alpha(NaN)` leaves the alpha as it was
  and `hybrid_distance_128` reads NaN as the default 0.5.
- Python objects can be configured while other threads use them, where the configuration
  call raised `RuntimeError: Already borrowed`. `Morphology.load_dictionary_*` and
  `SpellChecker.add_word` wait for the calls in progress on the object, and the calls after
  them wait for them. `GrammarChecker.enable_rule`, `disable_rule` and `enable_experimental`
  and `TurkishPiiEngine.enable_type`, `disable_type` and `set_type_enabled` make no call wait:
  the checker's calls copy its rules, and the PII engine's configuration is copy-on-write
  (`is_type_enabled` held a lock with the GIL and could stall the process for a second). A
  `PiiVault` that threads share serves one `mask` at a time, and only while the mask
  registers its values: the text is read before the vault is taken, and `mask(vault=v)`
  returns `v` itself as its `vault`. A mask's `mapping` and `detailed_mapping` hold the
  placeholders of its own text (one reused from an earlier turn included), where they held
  every entry of the session (20,002 for a call that masked 2); the vault holds the session
  (Rust: `PiiResult::mapping`, and `TurkishPiiEngine::mask_into`, which copies no vault).
  Finding a session's placeholders in a text no longer runs the shape scan once per key.
- `number_to_words` gives words for -2^63 (it overflowed); `currency_to_words` rounds to
  whole kuruş once (`2.999` is `üç lira`, it gave `iki lira yüz kuruş`; `-1.5` keeps its
  `elli kuruş`) and raises `ValueError` for an amount that is not finite or too large (Rust:
  `try_currency_to_words` returns `None`, `currency_to_words` an empty string);
  `similarity_vectors` raises `ValueError` for vectors of two lengths, which it compared over
  their common part; a grammar finding's `original_text` is the text its offsets cover, also
  across a newline (`hiç\nbir`).
- `extract_keywords` returns the same keywords on every call (ties are in text order before
  `top_k` cuts), and a number's suffixes are no keyword (`1923'te`).
- The dependency parser, the readability metrics and the syntactic analyzer share one
  inventory of non-finite verb forms (`morphology::verb_form`), the full set the analyzer
  emits: -DIkçA, -cAsInA, -AmAdAn, -AsIyA converbs and -mIş, -Ar/-mAz, -AsI participles were
  missing from the parser (`Güldükçe` was `ADV`/`advmod`, now `VERB`/`advcl`), and readability
  missed -ken, -AlI and -mAdAn and still listed tags the analyzer no longer emits. Readability
  reads each word in its sentence (the disambiguator's reading), not its first analysis. The
  syntactic analyzer no longer gives `VerbForm=Vnoun` to -mAzlIk and -mAcA nouns, which are
  ordinary nouns.
- `stem`, keyword extraction and the summarizer analyse a word as written, where they
  lower-cased it first: a capital keeps a name (`Miller`, not `mil`), and a Roman numeral its
  root (`XIX.`, `II'nci` → `II`, not `xıx.`, `ıı'nci`).
