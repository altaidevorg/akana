# Changelog

## 0.9.0 (unreleased)

The morphology engine is rebuilt on Zemberek's design, the sparse and hybrid encoders now have
a version and a compatibility check, and the bindings report character offsets. Several
results change, and indexes built with 0.8.0 must be rebuilt: see *Migrating from 0.8.0*.

### Migrating from 0.8.0

**Indexes.**
- Rebuild every sparse or hybrid index built with 0.8.0. Term ids are 32-bit now, the terms
  are disambiguated lemmas, and their weights changed.
- Store `akana.encoder_info()` (Rust: `embeddings::encoder_info()`) with each new index. When
  the index is opened, pass the stored value to `akana.check_encoder()`
  (`embeddings::compatibility`). It returns one of three values:
  - `"identical"`: the vectors are the same.
  - `"compatible"`: same sparse spec major and hybrid layout; the vectors may differ within
    the bump policy of `embeddings/spec.rs`, and no rebuild is needed.
  - `"rebuild"`.
- An index without stored info should be treated as `"rebuild"`.

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

**User dictionaries.** `Morphology.load_dictionary_str` and `load_dictionary_file` (Rust:
`TurkishMorphology::load_dictionary_str`/`_file`, `RootLexicon::load_from_str`/`_file`) return
the lines they could not read, where they returned nothing; a line with an unknown POS is one of
them (0.8.0 read it as a noun). They also take Zemberek's own syntax (`kriptopara [P:Noun;
A:CompoundP3sg; Roots:kripto-para]`, a verb's infinitive alone), which 0.8.0 misread.
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
  `TurkishMorphology::analyze_standard` does in Rust.

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
  (`Türkiye'de`, `Ali'nin`, `ABD'nin`, `TBMM'de`, but `Ankaralı`). A request written with a
  capital prefers the name (`ABD` the abbreviation, `abd` the common noun).
- `Prog` is the -Iyor progressive, also on a stem that changes (`gidiyorum`, `ediyor`), where
  it gave -mAktA (`gitmekteyim`). The request tags `Prog1` and `Prog2` ask for one of the two
  (`Prog2`: `bulunmaktadır`); both still analyse as `Prog`.
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

### Added

- `akana.encoder_info()` and `akana.check_encoder()`; in Rust, `embeddings::spec`
  (`SPARSE_SPEC`, `HYBRID_LAYOUT`, `DENSE_MODEL`, `ANALYSIS_VERSION`, `encoder_info`,
  `compatibility`). Their fingerprints hash the embedded resources, the encoders' constants
  and the analyses of a probe set.
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
- THIRD_PARTY_NOTICES in the repository, the crates and the Python package. It covers the
  Zemberek dictionaries and morphotactics (Apache 2.0), and the UD-derived function-lemma table
  and morphology prior (CC BY-SA 4.0).

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
    `gelmişlercesine`, `okuduklarımızdansınız`, `dolayısı`.
- **Tokenizer.** A number, date or time keeps its suffixes after an apostrophe in one token
  (`1924'te`, `17.00'ye`), as a name does, and an abbreviation in capitals keeps the suffixes
  court decisions write after its period (`TCK.nun`). NER entities still end with the number.
- **Sparse encoder.**
  - One disambiguated lemma per word occurrence, with 32-bit term ids.
  - The function-lemma weights come from a table frozen with the sparse spec.
  - A term weighs by its lemma's lexicon class, not by the reading the disambiguator picked,
    so a change of ranking no longer moves weights.
  - A word the analyzer cannot read is indexed by its backoff stem (`hiperbariğe` and
    `hiperbarik` are one term), an abbreviation has one term however it is written (`Av.`,
    `Av`), and a suffix written apart (`1924 'te`, `TCK. nun`) is no term.
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
- **Builds.** Cargo.lock is committed, the toolchain is pinned (rust-toolchain.toml), and
  embedded data keeps its line endings on every platform. Golden vectors, a frequent-type
  analysis snapshot and review gold sets fail CI on any unintended change of the encoders'
  output.
- **Performance.** The quadratic and cubic public paths are linear. The lexicon and the
  embedding weights are built once per process, and the Python bindings release the GIL.

### Fixed

- Crashes on statute headings and non-ASCII plates (PII), upper-case suffixes, non-ASCII
  spaces and overlapping edits (grammar), and cycles or extra roots in dependency trees.
- PII leaks: upper-case Turkish triggers, and text that `restore` would turn into an earlier
  value.
- A plural role noun before a name (`Davacılar Ahmet Yılmaz`) is no longer masked as part of
  the name; a sentence without a verb (`Hava çok soğuk .`, `Evde kimse yok .`) has its predicate
  as the dependency root, never the punctuation, and a postposition attaches to its noun
  (`case`); the de/da rule no longer splits a misspelt locative as a genitive (`aranında`).
- A CompoundP3sg noun derives on its stem without the possessive only: `cumhurbaşkanılık` and
  `cumhurbaşkanılığı` no longer analyse (`cumhurbaşkanlık` does; -(I)msI takes either,
  `zeytinyağımsı`).
- `reciprocal_rank_fusion` is deterministic: equal scores are ordered by best rank, then id
  (they came in hash order, so the same inputs gave different rankings), an id listed twice in
  one list counts once, and a `k` that is not positive and finite, or an `alpha` that is not
  finite, is rejected (`ValueError` in Python) where it returned infinite scores.
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
