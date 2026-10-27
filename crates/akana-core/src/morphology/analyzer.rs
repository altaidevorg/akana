//! Turkish morphological analyzer: a port of Zemberek's `RuleBasedAnalyzer` over the
//! morphotactic graph of [`super::turkish_morphotactics`] and the stems of
//! [`super::stems`].

use super::dictionary::{DictionaryItem, RootLexicon};
use super::morphotactics::{Context, Graph, Morpheme as M, PathView, StateId};
use super::phonetics::{attributes_of, PhoneticAttrs as P};
use super::pos::{PrimaryPos, RootAttr, SecondaryPos};
use super::stems::{self, StemIndex, StemTransition};
use super::turkish_morphotactics::{self, RootStates};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, LazyLock};
use unicode_normalization::UnicodeNormalization;

/// The marks written for an apostrophe (Ankara’ya, Ankara`ya); input folds them to `'`.
const APOSTROPHES: [char; 7] = ['\'', '’', '‘', 'ʼ', '`', '´', '′'];

/// The spelling factor of a nonstandard reading ([`MorphParse::nonstandard`]), and its log in
/// rank units (ln 0.5 / ln 1.02 * 16).
const NONSTANDARD_FACTOR: f32 = 0.5;
const NONSTANDARD_UNITS: i32 = -560;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MorphemeData {
    pub name: String,
    pub surface: String,
    pub tag: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MorphParse {
    pub surface: String,
    /// The lemma as a folded, lower-case key (ankara, kağıt, gelmek): what sparse terms and the
    /// prior use. [`MorphParse::display_lemma`] spells it as the dictionary does.
    pub lemma: String,
    pub root: String,
    /// POS of the dictionary root (Zemberek's `DictionaryItem.primaryPos`).
    pub primary_pos: PrimaryPos,
    /// POS of the word after its last derivation (Zemberek's `SingleAnalysis.getPos`):
    /// baş-la-yınca is `Adv`, oku-duğ-u is `Noun`, an-laş-ıyor-um is `Verb`. Unlike
    /// Zemberek, the copula does not change it: mutlu-y-duk stays `Adj`.
    #[serde(default = "unknown_pos")]
    pub pos: PrimaryPos,
    pub secondary_pos: SecondaryPos,
    pub morpheme_tags: Vec<String>,
    pub formatted: String,
    /// The analysis in Zemberek's format (`SingleAnalysis.formatLong`): every morpheme by its
    /// Zemberek id, after its surface when it has one, `|` before a derivation and `→` before
    /// the POS it derives: `[kitap:Noun] kitab:Noun+A3sg+ım:P1sg+a:Dat`, `[gelmek:Verb]
    /// gel:Verb|diğ:PastPart→Noun+A3sg+imiz:P1pl+de:Loc`. The silent A3sg and Zero are shown;
    /// Pnon and Nom are not, as in Zemberek. `morpheme_tags` and `formatted` stay akana's.
    #[serde(default)]
    pub analysis: String,
    /// How likely the reading is: a spelling factor (1 for a bare root, 0.95 for a suffixed
    /// word, 0.6 for a name's suffixes written without an apostrophe) times P(reading | word)
    /// under the corpus prior, so a word's scores sum to at most 1 and an unambiguous word's is
    /// its factor. Analyses come sorted by it.
    pub score: f32,
    /// Whether the reading is of a spelling usage has but TDK does not: an alternation left
    /// out or applied against the dictionary (temerrüte, gaspı) or old orthography
    /// (karariyle, olmıyan, vermeğe). It keeps the standard lemma, so search finds the word,
    /// but it is no correct spelling: the spell checker, the grammar rules and the
    /// deasciifier count only standard readings.
    #[serde(default)]
    pub nonstandard: bool,
    /// The morphotactic path the reading took, which the syntactic analyzer splits into
    /// inflectional groups. Empty for a reading the analyzer did not make.
    #[serde(skip)]
    pub(crate) path: MorphPath,
    /// Whether the lexicon ranks this reading below its twin (see [`prior_yields`]).
    #[serde(skip)]
    pub(crate) yields: bool,
    /// `score` as a log in rank units ([`rank_units`]), relative to the word's other readings:
    /// what the disambiguator weighs, so that it ranks with integers only.
    #[serde(skip)]
    pub(crate) rank: i32,
    /// The dictionary item's homograph index ([`DictionaryItem::homograph`]).
    #[serde(skip)]
    pub(crate) homograph: u8,
}

impl MorphParse {
    /// The lemma as the dictionary spells it, as Zemberek prints it (Ankara, kâğıt, Hakkâri,
    /// TBMM; see [`DictionaryItem::display_lemma`]), where `lemma` is the folded key. It is the
    /// dictionary item of `analysis`; a reading without one (deserialized from an older
    /// version) gives `lemma`.
    pub fn display_lemma(&self) -> &str {
        self.analysis
            .strip_prefix('[')
            .and_then(|a| a.split_once("] "))
            .and_then(|(item, _)| item.rsplit_once(':'))
            .map_or(&self.lemma, |(lemma, _)| lemma)
    }
}

/// The path of an analysis through the morphotactic graph: the root's state, then the state each
/// morpheme reached (POS roots and derivations included) and whether it had a surface.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct MorphPath {
    pub root: Option<StateId>,
    pub steps: Vec<PathStep>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PathStep {
    pub state: StateId,
    pub overt: bool,
}

fn unknown_pos() -> PrimaryPos {
    PrimaryPos::Unknown
}

/// The morphotactic graph, built once per process.
pub(crate) struct Engine {
    pub graph: Graph,
    pub roots: RootStates,
}

pub(crate) static ENGINE: LazyLock<Engine> = LazyLock::new(|| {
    let (graph, roots) = turkish_morphotactics::build();
    Engine { graph, roots }
});

/// The morphotactic graph the analyzer walks (for inspection and tests).
pub fn morphotactics_graph() -> &'static Graph {
    &ENGINE.graph
}

/// Whether a tag of [`MorphParse::morpheme_tags`] is a derivation (it opens a new
/// inflectional group: Ness, With, Inf, PastPart, AdvErek, Caus, ...).
pub fn is_derivational_tag(tag: &str) -> bool {
    DERIVATIONAL_TAGS.contains(&tag)
}

/// The built-in lexicon and its stems, built once per process and shared by every analyzer
/// that has not added its own items.
static BUILTIN: LazyLock<(Arc<RootLexicon>, Arc<StemIndex>)> = LazyLock::new(|| {
    let lexicon = RootLexicon::new();
    let stems = StemIndex::new(&lexicon, &ENGINE.roots);
    (Arc::new(lexicon), Arc::new(stems))
});

/// How many common nouns and adjectives of the built-in lexicon end in each four letters (or
/// are those letters): how well a guessed stem ends like a word, for
/// [`TurkishMorphology::backoff_stem`] (dosya beside dosyat, banka beside bankas).
static ROOT_ENDINGS: LazyLock<HashMap<String, u32>> = LazyLock::new(|| {
    let mut endings = HashMap::new();
    for item in BUILTIN.0.items() {
        if matches!(item.primary_pos, PrimaryPos::Noun | PrimaryPos::Adj)
            && item.secondary_pos == SecondaryPos::None
        {
            *endings
                .entry(last_letters(&item.root, 4).to_string())
                .or_insert(0) += 1;
        }
    }
    endings
});

/// The last `n` letters of a word, or the word if it is shorter.
fn last_letters(word: &str, n: usize) -> &str {
    word.char_indices()
        .rev()
        .nth(n - 1)
        .map_or(word, |(i, _)| &word[i..])
}

/// Total search steps one word may take; a word that needs more (a crafted chain of voice
/// suffixes) keeps the analyses found so far.
const MAX_WORK: usize = 100_000;
/// Search steps after which a word's dead ends are remembered. An ordinary word takes far
/// fewer, and hashing every step would cost more than the memo saves.
const MEMO_AFTER: usize = 512;
/// How often one state may recur on a path (verbRoot_S recurs once per voice suffix).
pub(super) const MAX_STATE_REPEAT: usize = 8;

/// The corpus frequency prior, written by `examples/morph_prior.rs`: how often each dictionary
/// item and each analysis shape was the right reading in a sample of court decisions.
const PRIOR_GZ: &[u8] = include_bytes!("prior.tsv.gz");

pub(super) static PRIOR: LazyLock<Prior> = LazyLock::new(|| Prior::parse(PRIOR_GZ));

/// The version of the analyses: lexicon, morphotactics, prior and disambiguator together. It
/// follows the encoders' bump policy (`embeddings::spec`): a minor version changes some
/// analyses without changing what a lemma or tag means; a major one changes conventions.
pub const ANALYSIS_VERSION: &str = "1.0";

/// Words and sentences whose analyses [`analysis_fingerprint`] covers, so that a change in code
/// the resources do not show (a rule, a tie-break, a factor) still changes it.
const PROBE_WORDS: &[&str] = &[
    "kitap",
    "kitabı",
    "kitapları",
    "evlerimizden",
    "gelmem",
    "gelmeyecekmişsiniz",
    "bana",
    "sana",
    "ona",
    "onların",
    "katıldı",
    "katılmıyoruz",
    "alınan",
    "bulundurması",
    "kaynaklanan",
    "birleştirilmesi",
    "bulamadığı",
    "et",
    "al",
    "aç",
    "ister",
    "olduğu",
    "edildiği",
    "hakları",
    "yapılacak",
    "başkanlığı",
    "hâlâ",
    "hala",
    "kalbi",
    "ay",
    "devam",
    "ileri",
    "ama",
    "değil",
    "ile",
    "yeni",
    "ilk",
    "kamu",
    "insan",
    "Kanun",
    "Ankara'da",
    "TBMM'nin",
    "ABD'ye",
    "1924'te",
    "%50'si",
    "XIX.",
    "15.",
    "3,5",
    "zeytinyağı",
    "cumhurbaşkanına",
    "üzerlerine",
    "reddine",
    "tıbbi",
    "suyun",
    "diyor",
    "yiyecek",
    "Yani",
    "Ali",
    "ekmek",
    "geliyorum",
    "okuyorlar",
    "anlaşılamadı",
    "yapılmıştır",
    "denizaltı",
    "temyizen",
    "mahkemece",
    "dairesince",
];

const PROBE_SENTENCES: &[&[&str]] = &[
    &["Davacının", "hakları", "korunmalıdır", "."],
    &["Bana", "bir", "kitap", "ver", "."],
    &["Çoğunluk", "görüşüne", "katılmıyoruz", "."],
    &["Uzun", "yaşa", "!"],
    &[
        "Siyasi",
        "partiler",
        "ister",
        "iktidarda",
        "ister",
        "muhalefette",
        "olsunlar",
        ".",
    ],
    &["Mahkemenin", "kararı", "kesinleşti", "."],
];

/// The analyses' fingerprint: the lexicon, the morphotactic graph, the prior and the analyses
/// and disambiguations of a built-in probe set (32 hex digits). Two libraries with the same one
/// analyse alike.
pub fn analysis_fingerprint() -> &'static str {
    static FINGERPRINT: LazyLock<String> = LazyLock::new(|| {
        let morphology = TurkishMorphology::new();
        let mut probe = String::new();
        for word in PROBE_WORDS {
            for p in morphology.analyze(word) {
                probe.push_str(&format!(
                    "{}|{:?}|{}|{:08x}|{}\n",
                    p.formatted,
                    p.secondary_pos,
                    p.analysis,
                    p.score.to_bits(),
                    p.rank
                ));
            }
            probe.push('\n');
        }
        let disambiguator = super::disambiguator::MorphologicalDisambiguator::new();
        for sentence in PROBE_SENTENCES {
            for p in disambiguator.disambiguate(sentence) {
                probe.push_str(&p.formatted);
                probe.push('\n');
            }
        }
        crate::fingerprint::Fingerprint::new()
            .section("version", ANALYSIS_VERSION.as_bytes())
            .section("lexicon", super::ZEMBEREK_LEXICON.as_bytes())
            .section("graph", graph_signature().as_bytes())
            .section("prior", PRIOR_GZ)
            .section("probe", probe.as_bytes())
            .hex()
    });
    &FINGERPRINT
}

/// Every state and transition of the morphotactic graph, as text.
fn graph_signature() -> String {
    let graph = &ENGINE.graph;
    let mut s = String::new();
    for state in &graph.states {
        s.push_str(&format!("{state:?}\n"));
    }
    for t in &graph.transitions {
        s.push_str(&format!(
            "{:?} {:?} {} {:?}\n",
            t.from, t.to, t.template, t.condition
        ));
    }
    s
}

/// Whether the prior scales a reading by its register factor (see `examples/morph_prior.rs`):
/// a second-person possessive or person (P2sg, A2pl), which rides along with another reading in
/// every word that has it (karar-ın-ın beside P3sg+Gen, gel-me "don't come" beside the verbal
/// noun, dönüş-ün beside the genitive), the bare imperative included (et, al, aç beside the
/// noun or adjective: court text all but never addresses its reader). The pronouns are exempt:
/// their person is their own (sen, size). The disambiguator takes the factor back in a sentence
/// that does address someone ("Sözümü anla !").
pub fn prior_second_person(p: &MorphParse) -> bool {
    p.primary_pos != PrimaryPos::Pron
        && p.morpheme_tags
            .iter()
            .any(|t| matches!(t.as_str(), "P2sg" | "P2pl" | "A2sg" | "A2pl"))
}

/// Whether a second-person reading is a negative imperative (gel-me "don't come", which every
/// verbal noun in -mA spells): it takes no register factor, where the other second-person
/// readings take one of at least `SECOND_PERSON_FLOOR` (examples/morph_prior.rs).
pub fn prior_negative_imperative(p: &MorphParse) -> bool {
    p.morpheme_tags.iter().any(|t| t == "Neg") && p.morpheme_tags.iter().any(|t| t == "Imp")
}

/// The register factor a reading takes, in rank units: 0 unless it is [`prior_second_person`].
pub(crate) fn prior_register_units(p: &MorphParse) -> i32 {
    PRIOR.register(p)
}

/// The prior's factor for a reading the lexicon ranks below another ([`prior_yields`]), where the
/// corpus cannot tell the two apart. It is a decision, not a nudge: it outweighs a thousandfold
/// difference between the two items' frequencies (düşünmek is some 400 times more frequent than
/// düşündürmek, whose forms are also düşün-dür-), and the disambiguator's context does not
/// overturn it.
pub const LEXICAL_PREFERENCE: f32 = 0.001;

/// Whether the prior scales a reading by [`LEXICAL_PREFERENCE`]. Both readings of each pair
/// below are in every word that has one, so the corpus counts cannot separate them, and the
/// lexicon decides:
/// - a lexicon verb and a derivation that spells it (katıl-dı and kat-ıl-dı, kaynaklan-an and
///   kaynak-lan-an): a verb made from a noun or adjective wins (it is listed because it is a
///   word of its own); between a verb and the voice derivation of another verb, the verb wins
///   when it is `Lexicalized` (katılmak "join"), the derivation when it is `NotLexicalized`
///   (alınmak, whose forms are al-ın "be taken"), and with neither the prior decides;
/// - a `Rare` item (bulamak "smear") yields to any other reading of the word (bul-ama-dığı);
/// - a `Nonstandard` stem's reading yields to any standard reading of the word.
pub fn prior_yields(p: &MorphParse) -> bool {
    p.yields
}

/// Rank units in one 2% step of a reading's prior relative to its word's best reading. Ranking
/// is done in integers of 1.02^(1/16): the prior file stores them, and the analyzer and the
/// disambiguator add them, so a word's readings rank alike on every platform (no `ln`, `exp` or
/// `powf` at run time, whose last bits differ between libms).
pub const RANK_UNITS_PER_STEP: i32 = 16;

/// Rank units in one natural-log unit: 16 / ln 1.02.
pub const RANK_UNITS_PER_NAT: f64 = 807.973_596_669_502_4;

/// A natural log (ln p) in rank units, rounded half away from zero.
pub const fn rank_units(nats: f64) -> i32 {
    let units = nats * RANK_UNITS_PER_NAT;
    (if units >= 0.0 {
        units + 0.5
    } else {
        units - 0.5
    }) as i32
}

/// 1.02^-q, for a reading q 2% steps below its word's best, built by repeated division (IEEE
/// basic operations give the same bits everywhere); 0 beyond the f32 range.
static STEP_RATIOS: LazyLock<Vec<f32>> = LazyLock::new(|| {
    let mut ratio = 1.0f64;
    (0..4420)
        .map(|_| {
            let r = ratio as f32;
            ratio /= 1.02;
            r
        })
        .collect()
});

fn step_ratio(steps: u32) -> f32 {
    STEP_RATIOS.get(steps as usize).copied().unwrap_or(0.0)
}

/// The version of the prior file format this analyzer reads (`#akana-prior` header).
const PRIOR_FORMAT: &str = "2";

/// Log probabilities, in rank units, of items ([`prior_item_key`]) and of shapes
/// ([`prior_shape_key`]) given their group ([`prior_shape_group`]: the POS, or the personal
/// pronouns' `Pers`): P(item) * P(shape | group) is how likely a reading is. The file holds them
/// ready to add (`examples/morph_prior.rs`), so loading computes nothing. The keys are hashed
/// ([`KeyHash`]), so looking a reading up builds no string.
pub(super) struct Prior {
    items: KeyMap,
    shapes: KeyMap,
    unseen_item: i32,
    /// A shape not in the file, per group; [`Prior::unseen_shape`] for a group not in it either.
    unseen_shapes: HashMap<String, i32>,
    unseen_shape: i32,
    /// Per-morpheme backoff for a shape not in the file: log P(tag | POS) over the overt
    /// morphemes of the POS's shapes, `POS+tag` -> rank units. An unseen shape scores the
    /// smoothing count times the probabilities of its morphemes, so one of fewer and more
    /// common morphemes outranks one of more or rarer morphemes.
    tag_penalties: HashMap<String, i32>,
    /// The penalty of a tag a POS's shapes never show.
    unseen_tag_penalty: i32,
    /// The register factor of second-person readings, and of negative imperatives.
    second_person: i32,
    negative_imperative: i32,
    /// [`LEXICAL_PREFERENCE`].
    lexical: i32,
    /// log P(case as written | reading): name capitalised, name lower-case, other word
    /// capitalised, other word lower-case.
    case: [i32; 4],
    /// Word forms whose classes the treebanks tag in other shares than items and shapes give
    /// them (değil, ne): the form in lower case ([`KeyHash`]) -> what the readings of each
    /// lemma ([`KeyHash`]) and class ([`prior_class`]) that are not names add.
    forms: HashMap<u64, Vec<(u64, usize, i32)>, std::hash::BuildHasherDefault<IdentityHasher>>,
    /// [`prior_inventory`] of the lexicon and graph the file was estimated for.
    inventory: String,
}

impl Prior {
    /// Reads the rows `examples/morph_prior.rs` writes: `I`tems, `S`hapes, `T`ag penalties,
    /// `U`nseen values, fixed facto`R`s and `W`ord-form corrections (`form/Class`), each
    /// `kind<TAB>key<TAB>rank units[<TAB>...]`. A damaged file, or one of another format, only
    /// loses the ranking.
    fn parse(gz: &[u8]) -> Self {
        use std::io::Read;
        let mut text = String::new();
        if flate2::read::GzDecoder::new(gz)
            .read_to_string(&mut text)
            .is_err()
            || !text.starts_with(&format!("#akana-prior\t{PRIOR_FORMAT}\t"))
        {
            text.clear();
        }
        let mut prior = Prior {
            items: KeyMap::default(),
            shapes: KeyMap::default(),
            unseen_item: 0,
            unseen_shapes: HashMap::new(),
            unseen_shape: 0,
            tag_penalties: HashMap::new(),
            unseen_tag_penalty: 0,
            second_person: 0,
            negative_imperative: 0,
            lexical: 0,
            case: [0; 4],
            forms: HashMap::default(),
            inventory: String::new(),
        };
        for line in text.lines() {
            let mut fields = line.split('\t');
            let (Some(kind), Some(key), Some(units)) =
                (fields.next(), fields.next(), fields.next())
            else {
                continue;
            };
            let Ok(units) = units.parse::<i32>() else {
                continue;
            };
            match (kind, key) {
                ("I", _) => {
                    prior.items.insert(KeyHash::of(key), units);
                }
                ("S", _) => {
                    prior.shapes.insert(KeyHash::of(key), units);
                }
                ("T", _) => {
                    prior.tag_penalties.insert(key.to_string(), units);
                }
                ("U", "item") => prior.unseen_item = units,
                ("U", "shape") => prior.unseen_shape = units,
                ("U", "tag") => prior.unseen_tag_penalty = units,
                ("U", _) => {
                    if let Some(group) = key.strip_prefix("shape:") {
                        prior.unseen_shapes.insert(group.to_string(), units);
                    }
                }
                ("R", "second-person") => prior.second_person = units,
                ("R", "second-person-negative-imperative") => prior.negative_imperative = units,
                ("R", "lexical-preference") => prior.lexical = units,
                ("R", "case-name-capitalised") => prior.case[0] = units,
                ("R", "case-name-lower") => prior.case[1] = units,
                ("R", "case-other-capitalised") => prior.case[2] = units,
                ("R", "case-other-lower") => prior.case[3] = units,
                ("W", _) => {
                    // form/lemma/Class
                    let parts = key.rsplit_once('/').and_then(|(rest, class)| {
                        let (form, lemma) = rest.rsplit_once('/')?;
                        Some((form, lemma, PRIOR_CLASSES.iter().position(|c| *c == class)?))
                    });
                    if let Some((form, lemma, class)) = parts {
                        prior.forms.entry(KeyHash::of(form)).or_default().push((
                            KeyHash::of(lemma),
                            class,
                            units,
                        ));
                    }
                }
                _ => {}
            }
        }
        if let Some(line) = text.lines().find(|l| l.starts_with("#inventory\t")) {
            prior.inventory = line["#inventory\t".len()..].to_string();
        }
        prior
    }

    /// Log P(item) of a dictionary item in rank units, for choosing among homographs (hak the
    /// noun).
    pub(super) fn item_log_p(&self, item: &DictionaryItem) -> i32 {
        let secondary = match item.secondary_pos.as_str() {
            "" => "-",
            s => s,
        };
        let parts = [
            item.lemma.as_str(),
            " ",
            item.root.as_str(),
            " ",
            item.primary_pos.as_str(),
            " ",
            secondary,
            homograph_suffix(item.homograph),
        ];
        self.items
            .get(&KeyHash::of_parts(parts))
            .copied()
            .unwrap_or(self.unseen_item)
    }

    /// The register factor of a reading, in rank units ([`prior_register_units`]).
    fn register(&self, p: &MorphParse) -> i32 {
        if !prior_second_person(p) {
            0
        } else if prior_negative_imperative(p) {
            self.negative_imperative
        } else {
            self.second_person
        }
    }

    /// A reading's log probability in rank units. `capitalised`: the word as written starts
    /// with a capital.
    fn log_p(&self, p: &MorphParse, capitalised: bool) -> i32 {
        let item = self
            .items
            .get(&KeyHash::of_parts(item_key_parts(p)))
            .copied()
            .unwrap_or(self.unseen_item);
        let shape_key = shape_key_parts(p).fold(KeyHash::new(), KeyHash::feed).0;
        let shape = self.shapes.get(&shape_key).copied().unwrap_or_else(|| {
            let pos = p.morpheme_tags.first().map_or("", String::as_str);
            let group = if personal_pronoun(p) {
                PERSONAL_GROUP
            } else {
                pos
            };
            let morphemes: i32 = p.morpheme_tags[1..]
                .iter()
                .filter(|t| !matches!(t.as_str(), "A3sg" | "Pnon" | "Nom"))
                .map(|t| {
                    self.tag_penalties
                        .get(&format!("{pos}+{t}"))
                        .copied()
                        .unwrap_or(self.unseen_tag_penalty)
                })
                .sum();
            self.unseen_shapes
                .get(group)
                .copied()
                .unwrap_or(self.unseen_shape)
                + morphemes
        });
        let register = self.register(p);
        let lexical = if p.yields { self.lexical } else { 0 };
        let case = self.case[match (prior_is_name(p), capitalised) {
            (true, true) => 0,
            (true, false) => 1,
            (false, true) => 2,
            (false, false) => 3,
        }];
        item + shape + register + lexical + case
    }
}

/// The classes of readings the prior's word-form corrections apply to ([`prior_class`]): the
/// POS after derivation, with a verb's participles, verbal nouns and converbs apart
/// (`VerbAdj`, `VerbNoun`, `VerbAdv`), which the treebanks tag VERB or, lexicalised (koruma,
/// geçen), NOUN, ADJ or ADV.
pub const PRIOR_CLASSES: [&str; 15] = [
    "Noun", "Adj", "Adv", "Verb", "VerbNoun", "VerbAdj", "VerbAdv", "Pron", "Det", "Num", "Conj",
    "Postp", "Interj", "Part", "Other",
];

/// A reading's class in [`PRIOR_CLASSES`]; the question particle is Part. A bare time noun is
/// an adverb here: the treebanks tag yarın, dün and sabah NOUN where Zemberek (and akana) read
/// them first as adverbs, so their tags do not choose between the two.
pub fn prior_class(p: &MorphParse) -> usize {
    let bare_time = p.secondary_pos == SecondaryPos::Time
        && p.morpheme_tags
            .get(1..)
            .unwrap_or_default()
            .iter()
            .all(|t| matches!(t.as_str(), "A3sg" | "Pnon" | "Nom"));
    let name = match (p.primary_pos, p.pos) {
        _ if bare_time => "Adv",
        (PrimaryPos::Verb, PrimaryPos::Noun) => "VerbNoun",
        (PrimaryPos::Verb, PrimaryPos::Adj) => "VerbAdj",
        (PrimaryPos::Verb, PrimaryPos::Adv) => "VerbAdv",
        (PrimaryPos::Verb, _) => "Verb",
        (_, PrimaryPos::Q) => "Part",
        (_, pos) => pos.as_str(),
    };
    PRIOR_CLASSES
        .iter()
        .position(|c| *c == name)
        .unwrap_or(PRIOR_CLASSES.len() - 1)
}

/// Whether the prior's case model counts a reading as a name (a proper noun or an
/// abbreviation), which is written capitalised.
pub fn prior_is_name(p: &MorphParse) -> bool {
    matches!(
        p.secondary_pos,
        SecondaryPos::ProperNoun | SecondaryPos::Abbreviation
    )
}

/// Whether the prior's case model counts a word as written capitalised (Ali, KARAR).
pub fn prior_capitalised(word: &str) -> bool {
    word.trim().chars().next().is_some_and(char::is_uppercase)
}

/// The prior's key for a reading's dictionary item: lemma, root, POS and secondary POS, and the
/// homograph index after the first (`kalp kalp Noun - #1`), so each item has its own frequency.
pub fn prior_item_key(p: &MorphParse) -> String {
    item_key_parts(p).concat()
}

/// The key suffix of a homograph after the first.
fn homograph_suffix(homograph: u8) -> &'static str {
    const SUFFIXES: [&str; 8] = ["", " #1", " #2", " #3", " #4", " #5", " #6", " #7"];
    SUFFIXES
        .get(usize::from(homograph))
        .copied()
        .unwrap_or(" #8+")
}

/// A signature of what the prior's keys stand for: the built-in lexicon's bytes and every state
/// and transition of the morphotactic graph (FNV-1a, hex). `examples/morph_prior.rs` writes it
/// into the prior's header; a test fails when they differ, that is, when the lexicon or the
/// graph changed and the prior was not re-estimated for them.
pub fn prior_inventory() -> String {
    let hash = KeyHash::new()
        .feed(super::ZEMBEREK_LEXICON)
        .feed(&graph_signature());
    format!("{:016x}", hash.0)
}

/// The inventory signature the embedded prior was estimated for ([`prior_inventory`]).
pub fn embedded_prior_inventory() -> &'static str {
    &PRIOR.inventory
}

/// The prior's key for a reading's shape: its tag sequence (Noun+P3sg+Loc). The personal
/// pronouns share a group of their own, without their person, which the lemma gives
/// (`Pers/Pron+Dat` for bana, sana, ona, bize, size): pooled with the other pronouns, whose
/// shapes bu and o dominate, ben's forms looked rare beside the noun ban (bana) and the noun
/// ben "mole". A group per pronoun drew whole nouns' forms to it instead (bent-i read as
/// ben-di "it was me").
pub fn prior_shape_key(p: &MorphParse) -> String {
    shape_key_parts(p).collect()
}

/// The group a shape's probability is normalised in: the personal pronouns' (`Pers`), else the
/// POS (the shape's first tag).
pub fn prior_shape_group(shape_key: &str) -> &str {
    match shape_key.split_once('/') {
        Some((group, _)) => group,
        None => shape_key.split('+').next().unwrap_or_default(),
    }
}

/// The shape group of the personal pronouns.
const PERSONAL_GROUP: &str = "Pers";

fn item_key_parts(p: &MorphParse) -> [&str; 8] {
    let secondary = match p.secondary_pos.as_str() {
        "" => "-",
        s => s,
    };
    let pos = p.primary_pos.as_str();
    [
        &p.lemma,
        " ",
        &p.root,
        " ",
        pos,
        " ",
        secondary,
        homograph_suffix(p.homograph),
    ]
}

/// A personal pronoun (ben, sen, o, biz, siz), whose shapes form [`PERSONAL_GROUP`]. Other
/// pronouns stay pooled by POS: a group per lexical pronoun (cümlesi, kimse) drew nouns' forms
/// to it (cümlesinde).
fn personal_pronoun(p: &MorphParse) -> bool {
    p.primary_pos == PrimaryPos::Pron && p.secondary_pos == SecondaryPos::Personal
}

fn shape_key_parts(p: &MorphParse) -> impl Iterator<Item = &str> {
    let personal = personal_pronoun(p);
    let group: [&str; 2] = if personal {
        [PERSONAL_GROUP, "/"]
    } else {
        ["", ""]
    };
    group.into_iter().chain(
        p.morpheme_tags
            .iter()
            .filter(move |t| {
                !(personal
                    && matches!(
                        t.as_str(),
                        "A1sg" | "A2sg" | "A3sg" | "A1pl" | "A2pl" | "A3pl"
                    ))
            })
            .enumerate()
            .flat_map(|(i, t)| [if i == 0 { "" } else { "+" }, t.as_str()]),
    )
}

/// FNV-1a over the bytes of a prior key, fed in parts: the hash of a key's parts is the hash of
/// the key.
struct KeyHash(u64);

impl KeyHash {
    fn new() -> Self {
        KeyHash(0xcbf2_9ce4_8422_2325)
    }

    fn feed(mut self, part: &str) -> Self {
        for b in part.bytes() {
            self.0 = (self.0 ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
        }
        self
    }

    fn of(key: &str) -> u64 {
        KeyHash::new().feed(key).0
    }

    fn of_parts(parts: [&str; 8]) -> u64 {
        parts.into_iter().fold(KeyHash::new(), KeyHash::feed).0
    }
}

/// A map keyed by [`KeyHash`] values, which need no hashing again.
type KeyMap = HashMap<u64, i32, std::hash::BuildHasherDefault<IdentityHasher>>;

#[derive(Default)]
struct IdentityHasher(u64);

impl std::hash::Hasher for IdentityHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 << 8) | u64::from(b);
        }
    }

    fn write_u64(&mut self, n: u64) {
        self.0 = n;
    }
}

/// The morphological analyzer. Analyzers share the built-in lexicon, so creating or cloning
/// one is cheap; adding items or loading a dictionary gives that analyzer a private copy of
/// the lexicon (copy-on-write) and leaves the others unchanged.
#[derive(Clone)]
pub struct TurkishMorphology {
    lexicon: Arc<RootLexicon>,
    stems: Arc<StemIndex>,
    prior: bool,
}

impl Default for TurkishMorphology {
    fn default() -> Self {
        Self::new()
    }
}

impl TurkishMorphology {
    pub fn new() -> Self {
        Self {
            lexicon: Arc::clone(&BUILTIN.0),
            stems: Arc::clone(&BUILTIN.1),
            prior: true,
        }
    }

    /// Whether analyses are ranked by the corpus prior (the default). Without it they keep
    /// the spelling factor alone and come in the analyzer's tie-break order; the prior's
    /// estimator analyses that way.
    pub fn with_prior(mut self, prior: bool) -> Self {
        self.prior = prior;
        self
    }

    pub fn with_lexicon(lexicon: RootLexicon) -> Self {
        let stems = StemIndex::new(&lexicon, &ENGINE.roots);
        Self {
            lexicon: Arc::new(lexicon),
            stems: Arc::new(stems),
            prior: true,
        }
    }

    /// The built-in lexicon that [`TurkishMorphology::new`] analyzers share.
    pub fn builtin_lexicon() -> Arc<RootLexicon> {
        Arc::clone(&BUILTIN.0)
    }

    /// The lexicon this analyzer uses.
    pub fn lexicon(&self) -> &RootLexicon {
        &self.lexicon
    }

    /// Length in characters of the longest root in the lexicon.
    pub fn max_root_chars(&self) -> usize {
        self.lexicon.max_root_chars()
    }

    /// The lexicon items whose root is exactly `root` (lower-case), without analysing it.
    pub fn root_items(&self, root: &str) -> &[DictionaryItem] {
        self.lexicon.get_items(root).map_or(&[], Vec::as_slice)
    }

    /// Adds a dictionary item to the morphological analyzer at runtime.
    pub fn add_item(&mut self, item: DictionaryItem) {
        Arc::make_mut(&mut self.lexicon).add_item(item.clone());
        Arc::make_mut(&mut self.stems).add(item, &self.lexicon, &ENGINE.roots);
    }

    /// Loads custom dictionary definitions from a file at runtime.
    pub fn load_dictionary_file<P: AsRef<std::path::Path>>(
        &mut self,
        path: P,
    ) -> std::io::Result<()> {
        // Read the file before copying the shared lexicon, so a missing file costs nothing.
        let text = std::fs::read_to_string(path)?;
        self.load_dictionary_str(&text);
        Ok(())
    }

    /// Loads custom dictionary definitions from a multiline string.
    pub fn load_dictionary_str(&mut self, text: &str) {
        let mut parsed = RootLexicon::empty();
        parsed.load_from_str(text);
        let mut items: Vec<DictionaryItem> = parsed.items().cloned().collect();
        items.sort_by(|a, b| a.root.cmp(&b.root).then_with(|| a.lemma.cmp(&b.lemma)));
        for item in items {
            self.add_item(item);
        }
    }

    /// Analyzes a surface word and returns all valid morphological parses, the most likely
    /// first.
    pub fn analyze(&self, word: &str) -> Vec<MorphParse> {
        let mut parses = self.analyze_unranked(word);
        if self.prior && parses.len() > 1 {
            let prior = &*PRIOR;
            let capitalised = prior_capitalised(word);
            let mut logs: Vec<i32> = parses.iter().map(|p| prior.log_p(p, capitalised)).collect();
            // A form the treebanks tag: each lemma's readings take their classes' share there
            // (değil the verb, not the conjunction); the lemmas' shares, and the names', stay
            // the prior's (alacak "receivable" in court decisions, al-acak elsewhere).
            if !prior.forms.is_empty() {
                let lower = super::super::phonology::to_turkish_lower(word.trim());
                if let Some(adds) = prior.forms.get(&KeyHash::of(&lower)) {
                    for (log, p) in logs.iter_mut().zip(&parses) {
                        if prior_is_name(p) {
                            continue;
                        }
                        let (lemma, class) = (KeyHash::of(&p.lemma), prior_class(p));
                        if let Some(&(_, _, add)) =
                            adds.iter().find(|&&(l, c, _)| l == lemma && c == class)
                        {
                            *log += add;
                        }
                    }
                }
            }
            let best = logs.iter().copied().max().unwrap_or(0);
            // Each reading's prior relative to the best, in steps of 2% (rounded half up):
            // readings the corpus cannot tell apart (A3pl+Gen and A3pl+P2sg always compete in
            // the same words) come out of the estimator equal up to rounding, and tie, so the
            // analyzer's order decides between them. Divided by their sum, they are
            // P(reading | word).
            let half = RANK_UNITS_PER_STEP / 2;
            let mut steps: Vec<u32> = logs
                .iter()
                .map(|&log| ((best - log + half) / RANK_UNITS_PER_STEP).unsigned_abs())
                .collect();
            // Co-occurring readings come out of the estimator in whatever ratio its start
            // gave them: they share the best of their priors, and each one after the first of
            // its set gives up a millionth of its score and a rank unit per place, which puts
            // them in their fixed order.
            let mut places: Vec<u8> = Vec::new();
            if parses.iter().filter_map(co_occurring).nth(1).is_some() {
                let sets: Vec<Option<CoOccurring>> = parses.iter().map(co_occurring).collect();
                for i in 0..parses.len() {
                    for j in 0..parses.len() {
                        if let (Some(a), Some(b)) = (sets[i], sets[j]) {
                            if i != j && same_class(&parses[i], a, &parses[j], b) {
                                steps[i] = steps[i].min(steps[j]);
                            }
                        }
                    }
                }
                places = sets.iter().map(|c| c.map_or(0, |c| c.place)).collect();
            }
            let total: f32 = steps.iter().map(|&q| step_ratio(q)).sum();
            for (i, (p, q)) in parses.iter_mut().zip(steps).enumerate() {
                p.score *= step_ratio(q) / total;
                p.rank -= q as i32 * RANK_UNITS_PER_STEP;
                if let Some(&place) = places.get(i) {
                    p.score *= 1.0 - 1e-6 * f32::from(place);
                    p.rank -= i32::from(place);
                }
            }
            order(&mut parses);
        }
        parses
    }

    /// The stem of a word the lexicon cannot read, as search should index it: the word without
    /// the inflections a noun takes (plural, possessive, case, -ki, copula), read through a
    /// noun guessed from what is left, as Zemberek's analysis of unknown words guesses one
    /// (hiperbariğe: hiperbarik; istanbulda: istanbul; kampüsü: kampüs; dosyada: dosya).
    /// Of the stems that read so ([`backoff_candidates`](Self::backoff_candidates)), the one
    /// with the most inflections after it wins, then the one whose last four letters end the
    /// most nouns and adjectives of the lexicon, then the longest (plan step D6). A derivation
    /// is not taken off (kitapçı is no kitap). `None` when no ending reads so; the word's own
    /// analyses are not looked at.
    pub fn backoff_stem(&self, word: &str) -> Option<String> {
        let endings = &*ROOT_ENDINGS;
        self.backoff_candidates(word)
            .into_iter()
            .max_by_key(|(stem, n)| {
                let like = endings.get(last_letters(stem, 4)).copied().unwrap_or(0);
                (*n, like, stem.chars().count())
            })
            .map(|(stem, _)| stem)
    }

    /// The stems [`backoff_stem`](Self::backoff_stem) chooses from, each with the most
    /// inflections an ending of the word reads as after it: every split whose stem keeps three
    /// letters and a vowel and can end a word, and for a stem in b, c, d or ğ before a vowel,
    /// also the stem unvoiced (vücud-una: vücut, sonuc-una: sonuç).
    #[doc(hidden)]
    pub fn backoff_candidates(&self, word: &str) -> Vec<(String, usize)> {
        // Without circumflexes, as lexicon roots are (dâvaların: dava).
        let lower: String = super::super::phonology::to_turkish_lower(word.trim())
            .chars()
            .map(|c| match c {
                'â' => 'a',
                'î' => 'i',
                'û' => 'u',
                c => c,
            })
            .collect();
        if !lower.chars().all(char::is_alphabetic) {
            return Vec::new();
        }
        let mut out = Vec::new();
        for (at, _) in lower.char_indices().skip(3) {
            let (stem, ending) = lower.split_at(at);
            if !stem.chars().any(super::phonetics::is_vowel)
                || !word_final(stem)
                || !ending.starts_with(|c: char| "lıiuüsnydtaem".contains(c))
            {
                continue;
            }
            let mut roots = vec![(stem.to_string(), RootAttr::NONE)];
            if ending.starts_with(super::phonetics::is_vowel) {
                let last = stem.chars().next_back();
                let unvoiced = match last {
                    Some('b') => Some('p'),
                    Some('c') => Some('ç'),
                    Some('d') => Some('t'),
                    Some('ğ') => Some('k'),
                    _ => None,
                };
                if let (Some(last), Some(unvoiced)) = (last, unvoiced) {
                    let head = &stem[..stem.len() - last.len_utf8()];
                    roots.push((format!("{head}{unvoiced}"), RootAttr::VOICING));
                }
            }
            for (root, attributes) in roots {
                let mut item = DictionaryItem::new(
                    root.as_str(),
                    root.as_str(),
                    PrimaryPos::Noun,
                    SecondaryPos::None,
                    attributes,
                );
                item.pronunciation = Some(root.clone());
                let inflections = self
                    .analyze_guessed(&item, &lower, word)
                    .iter()
                    .filter(|p| {
                        p.morpheme_tags.len() > 1
                            && p.morpheme_tags[1..]
                                .iter()
                                .all(|t| INFLECTION_TAGS.contains(&t.as_str()))
                    })
                    .map(|p| p.morpheme_tags.len() - 1)
                    .max();
                if let Some(n) = inflections {
                    out.push((root, n));
                }
            }
        }
        out
    }

    /// The readings of a word as TDK spells it, in the case it is written in:
    /// [`analyze`](Self::analyze) without the [`nonstandard`](MorphParse::nonstandard) ones,
    /// and, for a word written all in lower case, without a name's reading or an
    /// abbreviation's the dictionary spells in capitals (ankara is no spelling of Ankara, bü
    /// none of BÜ; km, and vb. with its period, are). What the spell checker, the grammar rules
    /// and the deasciifier judge a spelling by, so they pass the word as it is written.
    pub fn analyze_standard(&self, word: &str) -> Vec<MorphParse> {
        let lower_case = !word.chars().any(char::is_uppercase);
        let dotted = word.contains('.');
        let mut parses = self.analyze(word);
        parses.retain(|p| {
            !p.nonstandard
                && !(lower_case
                    && match p.secondary_pos {
                        SecondaryPos::ProperNoun => true,
                        SecondaryPos::Abbreviation => {
                            !dotted && p.display_lemma().chars().any(char::is_uppercase)
                        }
                        _ => false,
                    })
        });
        parses
    }

    pub(super) fn analyze_unranked(&self, word: &str) -> Vec<MorphParse> {
        let parses = self.analyze_written(word);
        // Text typed without the Turkish keyboard writes İ as I (KITAPLAR, INSAN, Internet), and
        // so do foreign acronyms (FBI, IMF), whose lemmas have i: a word with an I that its
        // Turkish reading (ı) leaves without a lexicon word is read with İ in place of its I's
        // too, in every combination for up to four, the fewest first. The first that makes a
        // lexicon word wins; failing that, a word with no reading takes the first with any.
        let clean = word.trim();
        let lexical = |ps: &[MorphParse]| ps.iter().any(|p| !self.root_items(&p.root).is_empty());
        if !clean.contains('I') || lexical(&parses) || self.analyze_roman(clean).is_some() {
            return parses;
        }
        let positions: Vec<usize> = clean.match_indices('I').map(|(i, _)| i).take(4).collect();
        let mut subsets: Vec<u32> = (1..1u32 << positions.len()).collect();
        subsets.sort_by_key(|s| s.count_ones());
        let mut fallback: Option<Vec<MorphParse>> = None;
        for subset in subsets {
            let mut dotted = String::with_capacity(clean.len() + 4);
            let mut last = 0;
            for (bit, &at) in positions.iter().enumerate() {
                if subset & (1 << bit) != 0 {
                    dotted.push_str(&clean[last..at]);
                    dotted.push('İ');
                    last = at + 1;
                }
            }
            dotted.push_str(&clean[last..]);
            let mut readings = self.analyze_written(&dotted);
            for p in &mut readings {
                p.surface = clean.to_string();
            }
            if lexical(&readings) {
                return readings;
            }
            if fallback.is_none() && !readings.is_empty() {
                fallback = Some(readings);
            }
        }
        match fallback {
            Some(readings) if parses.is_empty() => readings,
            _ => parses,
        }
    }

    /// The analyses of a word as written: `analyze_unranked` without its reading of I as İ.
    fn analyze_written(&self, word: &str) -> Vec<MorphParse> {
        let clean = word.trim();
        if clean.is_empty() {
            return Vec::new();
        }

        // Decomposed input (I + U+0307, o + U+0308) is composed first, so İstanbul and göz
        // written that way are the same words.
        let composed: std::borrow::Cow<str> =
            match unicode_normalization::is_nfc_quick(clean.chars()) {
                unicode_normalization::IsNormalized::Yes => std::borrow::Cow::Borrowed(clean),
                _ => std::borrow::Cow::Owned(clean.nfc().collect()),
            };
        let clean: &str = &composed;
        // Lexicon roots carry no circumflex (nikâh -> nikah), as in Zemberek's
        // normalizeForAnalysis, nor the acute accent some texts write for it (hálá,
        // muhafazakár); in a word written with a capital, an acute accent is a foreign name's
        // (José, Pará, which is no para). Every apostrophe-like mark is an apostrophe.
        let acute = !clean.starts_with(char::is_uppercase);
        let lower: String = super::super::phonology::to_turkish_lower(clean)
            .chars()
            .map(|c| match c {
                'â' => 'a',
                'î' => 'i',
                'û' => 'u',
                'á' if acute => 'a',
                'í' if acute => 'i',
                'ú' if acute => 'u',
                'é' if acute => 'e',
                'ó' if acute => 'o',
                c if APOSTROPHES.contains(&c) => '\'',
                _ => c,
            })
            .collect();

        if lower.starts_with(|c: char| c.is_ascii_digit() || c == '%') {
            return self.analyze_number(clean, &lower, None);
        }
        if let Some(parses) = self.analyze_hyphenated(clean, &lower) {
            return parses;
        }
        // Court decisions write an abbreviation's suffixes after a period (TCK.nun,
        // H.U.M.K.nun): read as after an apostrophe (TCK'nun).
        if let Some((abbreviation, ending)) = dotted_suffix(clean) {
            let quoted = format!("{abbreviation}'{ending}");
            let stem = super::super::phonology::to_turkish_lower(&abbreviation);
            let ending = super::super::phonology::to_turkish_lower(ending);
            return self
                .analyze_with_apostrophe(&quoted, &stem, &ending)
                .into_iter()
                .map(|mut p| {
                    p.surface = clean.to_string();
                    p
                })
                .collect();
        }

        // The suffixes follow the last apostrophe; an earlier one belongs to the word
        // (Kur'an'ı).
        let (stem, mut parses) = match lower.rsplit_once('\'') {
            Some((stem, ending)) if !stem.is_empty() && !ending.is_empty() => {
                let stem = stem.replace('\'', "");
                let parses = self.analyze_with_apostrophe(clean, &stem, ending);
                (stem, parses)
            }
            _ => {
                let word = lower.replace('\'', "");
                let mut parses = self.analyze_normalized(clean, &word, false);
                if parses.is_empty() {
                    parses = self.analyze_name_without_apostrophe(clean, &word);
                }
                if parses.is_empty() && word.ends_with('.') && self.analyze_roman(clean).is_none() {
                    parses = self.analyze_dotted(clean, &word);
                }
                if parses.is_empty() {
                    parses = self.analyze_joined_numeral(clean, &word);
                }
                if parses.is_empty() {
                    parses = self.analyze_old_orthography(clean, &word);
                }
                // A letter, as a list's marker (i, ü, l, x, ğ), that the lexicon lacks: the
                // letter, by its name.
                if parses.is_empty()
                    && word.chars().count() == 1
                    && word.chars().all(char::is_alphabetic)
                {
                    let mut letter = DictionaryItem::new(
                        word.as_str(),
                        word.as_str(),
                        PrimaryPos::Noun,
                        SecondaryPos::Abbreviation,
                        RootAttr::NONE,
                    );
                    letter.pronunciation = Some(super::pronunciation::letter_pronunciation(&word));
                    // spelled as written: a list's letter is a spelling in lower case too
                    letter.display = Some(clean.to_string());
                    parses = self.analyze_guessed(&letter, &word, clean);
                }
                // A list's numerals in lower case (ii, iii, xiv).
                if parses.is_empty()
                    && word.chars().nth(1).is_some()
                    && word.chars().all(|c| matches!(c, 'i' | 'v' | 'x'))
                {
                    if let Some(mut roman) = self.analyze_roman(&word.to_ascii_uppercase()) {
                        for p in &mut roman {
                            p.surface = clean.to_string();
                        }
                        parses = roman;
                    }
                }
                (word, parses)
            }
        };
        if let Some(roman) = self.analyze_roman(clean) {
            // A word the lexicon knows keeps its own readings first (MI, the question particle
            // in capitals; CD; XL, the size); otherwise the numeral replaces a guessed
            // abbreviation (II, XIX., IV'üncü), so II'nın is no word.
            let stem = stem.strip_suffix('.').unwrap_or(&stem);
            if self.root_items(stem).is_empty() {
                parses.retain(|p| p.secondary_pos != SecondaryPos::Abbreviation);
                parses.splice(0..0, roman);
            } else {
                parses.extend(roman);
            }
        }
        parses
    }

    /// A Roman numeral in capitals (II, XIX., IV'üncü), read as its number; the numeral as
    /// written is its root and lemma. `None` when the word is no Roman numeral.
    fn analyze_roman(&self, clean: &str) -> Option<Vec<MorphParse>> {
        let (numeral, ending) = clean.split_once(APOSTROPHES).unwrap_or((clean, ""));
        let body = numeral.strip_suffix('.').unwrap_or(numeral);
        let value = super::pronunciation::roman_value(body)?;
        let dot = if body.len() < numeral.len() { "." } else { "" };
        let ending = super::super::phonology::to_turkish_lower(ending);
        let number = if ending.is_empty() {
            format!("{value}{dot}")
        } else {
            format!("{value}{dot}'{ending}")
        };
        Some(self.analyze_number(clean, &number, Some(numeral)))
    }

    /// A number, read through its last word (Zemberek's numeral analysis): 1923 is üç to
    /// its suffixes (1923'te, not 1923'de), 15. is beşinci, %50'si is elli-si, 3,5'e is beş-e.
    /// The analyses are those of the word, with the number (or `written`, for a Roman
    /// numeral) as root and lemma.
    fn analyze_number(&self, clean: &str, lower: &str, written: Option<&str>) -> Vec<MorphParse> {
        let (number, ending) = match lower.split_once('\'') {
            Some((n, e)) => (n, e),
            None => (lower, ""),
        };
        let (percent, digits_part) = match number.strip_prefix('%') {
            Some(rest) => (true, rest),
            None => (false, number),
        };
        let (ordinal, digits_part) = match digits_part.strip_suffix('.') {
            Some(rest) if !rest.is_empty() => (true, rest),
            _ => (false, digits_part),
        };
        if digits_part.is_empty()
            || !digits_part.starts_with(|c: char| c.is_ascii_digit())
            || !digits_part
                .chars()
                .all(|c| c.is_ascii_digit() || matches!(c, '.' | ',' | ':' | '/' | '-'))
            || digits_part.ends_with(|c: char| !c.is_ascii_digit())
        {
            return Vec::new();
        }
        // 1.000.000 is one number; in 3,5, 21.10.2026, 10:30 or 1990-1995 the last part is
        // read last.
        let thousands = {
            let groups: Vec<&str> = digits_part.split('.').collect();
            groups.len() > 1
                && groups[0].len() <= 3
                && groups[1..].iter().all(|g| g.len() == 3)
                && !digits_part.contains([',', ':', '/', '-'])
        };
        // A time on the hour is read by its hour (17.00'ye, 22:00'den: "on yedi", "yirmi iki").
        let on_the_hour = digits_part
            .rsplit_once(['.', ':'])
            .filter(|(hour, minutes)| {
                *minutes == "00" && (1..=2).contains(&hour.len()) && !hour.contains(['.', ':'])
            })
            .map(|(hour, _)| hour);
        let digits: String = if thousands {
            digits_part.replace('.', "")
        } else if let Some(hour) = on_the_hour {
            hour.to_string()
        } else {
            digits_part
                .rsplit(['.', ',', ':', '/', '-'])
                .next()
                .unwrap_or(digits_part)
                .to_string()
        };
        let last = super::pronunciation::last_number_word(&digits);
        let word = if ordinal {
            super::pronunciation::ordinal_word(last)
        } else {
            last
        };
        // dört voices before a vowel, which the digits do not show (4'e is dörde).
        let stem = if word == "dört" && ending.starts_with(super::phonetics::is_vowel) {
            "dörd"
        } else {
            word
        };
        let secondary = if percent {
            SecondaryPos::Percentage
        } else if ordinal || word != last {
            SecondaryPos::Ordinal
        } else {
            SecondaryPos::Cardinal
        };
        let joined = format!("{stem}{ending}");
        let root = written.unwrap_or(number).to_string();
        self.analyze_normalized(clean, &joined, true)
            .into_iter()
            .filter(|p| {
                p.primary_pos == PrimaryPos::Num
                    && (p.root == word
                        || p.root == super::pronunciation::ordinal_word(word)
                        // The distributive (3'er, 2'şer, %50'şer: üçer, ikişer, ellişer).
                        || (p.secondary_pos == SecondaryPos::Distributive
                            && !ordinal
                            && p.root.starts_with(stem)))
            })
            .map(|mut p| {
                let distributive = p.secondary_pos == SecondaryPos::Distributive;
                let ordinal_reading = (p.root != word && !distributive) || ordinal;
                p.formatted = format!("[{root}:Num] {root}:{}", p.morpheme_tags.join("+"));
                p.lemma = root.clone();
                p.root = root.clone();
                p.secondary_pos = if distributive {
                    SecondaryPos::Distributive
                } else if ordinal_reading && !percent {
                    SecondaryPos::Ordinal
                } else {
                    secondary
                };
                p.analysis =
                    restem_analysis(&p.analysis, &root, PrimaryPos::Num, p.secondary_pos, &root);
                p
            })
            .collect::<Vec<_>>()
            .into_iter()
            .fold(Vec::new(), |mut out: Vec<MorphParse>, p| {
                if !out.iter().any(|q| q.formatted == p.formatted) {
                    out.push(p);
                }
                out
            })
    }

    /// A hyphenated word: the lexicon's own spelling first (e-posta is eposta), else its last
    /// part (Türk-İslam, anti-demokratik), whose analyses stand for the whole word.
    fn analyze_hyphenated(&self, clean: &str, lower: &str) -> Option<Vec<MorphParse>> {
        let (head, last) = lower.rsplit_once('-')?;
        // COVID-19 and A-4 are codes, not a word and its number.
        if head.is_empty() || last.is_empty() || last.starts_with(|c: char| c.is_ascii_digit()) {
            return None;
        }
        let first = lower.split('-').next().unwrap_or(head);
        let joined = lower.replace('-', "");
        let spanning: Vec<MorphParse> = self
            .analyze_unranked(&joined)
            .into_iter()
            .filter(|p| {
                p.root.chars().count() > first.chars().count() && joined.starts_with(&p.root)
            })
            .map(|mut p| {
                p.surface = clean.to_string();
                p
            })
            .collect();
        if !spanning.is_empty() {
            return Some(spanning);
        }
        let original_last = clean.rsplit('-').next().unwrap_or(last);
        Some(
            self.analyze_unranked(original_last)
                .into_iter()
                .map(|mut p| {
                    p.surface = clean.to_string();
                    p
                })
                .collect(),
        )
    }

    /// Zemberek's analyzeWordsWithApostrophe: analyse the word without the quote and keep the
    /// noun parses whose root is the part before it (İstanbul'un, Antalya'da). Failing that, a
    /// lexicon root the apostrophe falls inside (Kur'an, re'sen, fer'i), as Zemberek's
    /// dictionary spells them. Then a name that ends in a possessive (Anayasa Mahkemesi'ne,
    /// Türk Ceza Kanunu'nun), which Zemberek guesses as a name. A name the lexicon does not know
    /// gets a proper-noun root, as Zemberek's unidentified-token analyzer does for words with an
    /// apostrophe.
    fn analyze_with_apostrophe(&self, clean: &str, stem: &str, ending: &str) -> Vec<MorphParse> {
        let joined = format!("{stem}{ending}");
        let all = self.analyze_normalized(clean, &joined, true);
        let stem_chars = stem.chars().count();
        let (standard, spanning): (Vec<MorphParse>, Vec<MorphParse>) = all
            .iter()
            .filter(|p| {
                (p.primary_pos == PrimaryPos::Noun && p.root == stem)
                    || (p.root.chars().count() > stem_chars && joined.starts_with(&p.root))
            })
            .cloned()
            .partition(|p| p.root == stem);
        if !standard.is_empty() {
            return standard;
        }
        if !spanning.is_empty() {
            return spanning;
        }
        // The quote follows the possessive of a name's head noun, so the word reads on through
        // it (mahkeme+P3sg+Dat, başkan+Ness+P3sg+Dat).
        let heads: Vec<MorphParse> = self
            .analyze_normalized(clean, stem, false)
            .into_iter()
            .filter(|s| {
                s.primary_pos == PrimaryPos::Noun
                    && s.morpheme_tags
                        .last()
                        .is_some_and(|t| t == "P3sg" || t == "P3pl")
            })
            .collect();
        let through: Vec<MorphParse> = all
            .into_iter()
            .filter(|p| {
                heads.iter().any(|s| {
                    s.root == p.root
                        && p.morpheme_tags.len() > s.morpheme_tags.len()
                        && p.morpheme_tags.starts_with(&s.morpheme_tags)
                })
            })
            .collect();
        if !through.is_empty() {
            return through;
        }
        // A name or abbreviation the lexicon knows is not guessed again: ABD'nın is not a word
        // because ABD is read "abede". A common word used as a name is (Başlangıç'ı, with no
        // voicing). Nor is an acronym the lexicon spells with the i of its Latin I (CIA'nın,
        // when CIA is read "siyayey").
        let original_stem = clean.rfind(APOSTROPHES).map_or(clean, |i| &clean[..i]);
        let named = |root: &str| {
            self.root_items(root).iter().any(|i| {
                matches!(
                    i.secondary_pos,
                    SecondaryPos::ProperNoun | SecondaryPos::Abbreviation
                )
            })
        };
        let latin = original_stem.contains('I').then(|| {
            super::super::phonology::to_turkish_lower(&original_stem.replace('I', "İ"))
                .replace(APOSTROPHES, "")
        });
        if named(stem) || latin.is_some_and(|i| named(&i)) {
            return Vec::new();
        }
        // An upper-case stem is an abbreviation, read letter by letter unless it reads as a
        // word (TCDD'nin "tecedede", NATO'nun).
        let abbreviation =
            original_stem.chars().count() > 1 && original_stem.chars().all(|c| !c.is_lowercase());
        let mut guessed = if abbreviation {
            let mut item = DictionaryItem::new(
                stem,
                stem,
                PrimaryPos::Noun,
                SecondaryPos::Abbreviation,
                RootAttr::NONE,
            );
            item.pronunciation = Some(super::pronunciation::guess_abbreviation(stem));
            item
        } else {
            DictionaryItem::new(
                stem,
                stem,
                PrimaryPos::Noun,
                SecondaryPos::ProperNoun,
                RootAttr::PROPER_NOUN,
            )
        };
        if guessed.pronunciation.is_none() && !stem.chars().any(super::phonetics::is_vowel) {
            guessed.pronunciation = Some(super::pronunciation::letter_pronunciation(stem));
        }
        self.analyze_guessed(&guessed, &joined, clean)
    }

    /// The analyses of `input` through an item the lexicon does not have, as after an
    /// apostrophe.
    fn analyze_guessed(&self, item: &DictionaryItem, input: &str, clean: &str) -> Vec<MorphParse> {
        let stems: Vec<StemTransition> = stems::generate(item, 0, &ENGINE.roots)
            .into_iter()
            .filter(|s| input.starts_with(s.surface.as_ref()))
            .collect();
        let candidates: Vec<(&StemTransition, &DictionaryItem)> =
            stems.iter().map(|s| (s, item)).collect();
        let mut results = Vec::new();
        self.search(&candidates, input, clean, true, &mut results);
        finish(results)
    }

    /// A word in the orthography of older texts, which TDK no longer writes, read as the modern
    /// spelling with its readings `nonstandard`: -iyle for -(y)lA whatever the harmony
    /// (karariyle, dolayısiyle: kararıyla, dolayısıyla); a or e narrowed before y outside
    /// -yor (olmıyan, sağlıyacak, edilmiyecek: olmayan, sağlayacak, edilmeyecek); -mAğA for
    /// -mAyA (vermeğe, açmağa, yapmağı: vermeye, açmaya, yapmayı).
    fn analyze_old_orthography(&self, clean: &str, lower: &str) -> Vec<MorphParse> {
        let mut modern: Vec<String> = Vec::new();
        if let Some(i) = lower.rfind("iyle").or_else(|| lower.rfind("iyla")) {
            for ending in ["ıyla", "uyla", "üyle", "iyle"] {
                modern.push(format!(
                    "{}{ending}{}",
                    &lower[..i],
                    &lower[i + "iyle".len()..]
                ));
            }
        }
        for (old, new) in [
            ("ıya", "aya"),
            ("iye", "eye"),
            ("mağ", "may"),
            ("meğ", "mey"),
        ] {
            if let Some(i) = lower.rfind(old) {
                let rest = &lower[i + old.len()..];
                if !old.starts_with('m') || rest.starts_with(super::phonetics::is_vowel) {
                    modern.push(format!("{}{new}{rest}", &lower[..i]));
                }
            }
        }
        let mut parses: Vec<MorphParse> = Vec::new();
        for word in modern.iter().filter(|w| w.as_str() != lower) {
            for mut p in self.analyze_normalized(clean, word, false) {
                if !parses.iter().any(|q| q.formatted == p.formatted) {
                    p.nonstandard = true;
                    p.score *= NONSTANDARD_FACTOR;
                    p.rank += NONSTANDARD_UNITS;
                    parses.push(p);
                }
            }
        }
        parses
    }

    /// A numeral written as one word (onbeş, yirmidört, beşbin, onüçüncü, onar), as court
    /// decisions often write it: the readings of its last number word, with the whole numeral
    /// as root and lemma, where the words before it and that word make a number as it is read
    /// (onbeş is 15; beşon is none).
    fn analyze_joined_numeral(&self, clean: &str, lower: &str) -> Vec<MorphParse> {
        let mut splits = Vec::new();
        let mut rest = lower;
        while let Some(word) = super::pronunciation::number_word_prefix(rest) {
            rest = &rest[word.len()..];
            if rest.is_empty() {
                break;
            }
            splits.push(lower.len() - rest.len());
        }
        for &at in splits.iter().rev() {
            let (prefix, last) = lower.split_at(at);
            let readings: Vec<MorphParse> = self
                .analyze_normalized(clean, last, false)
                .into_iter()
                .filter(|p| {
                    p.primary_pos == PrimaryPos::Num
                        && super::pronunciation::cardinal_word(&p.root).is_some_and(|c| {
                            super::pronunciation::joined_number_value(&format!("{prefix}{c}"))
                                .is_some()
                        })
                })
                .map(|mut p| {
                    let root = format!("{prefix}{}", p.root);
                    let body = p.analysis.split_once("] ").map_or("", |(_, b)| b);
                    let stem = format!("{prefix}{}", body.split(':').next().unwrap_or(""));
                    p.formatted = format!("[{root}:Num] {root}:{}", p.morpheme_tags.join("+"));
                    p.analysis = restem_analysis(
                        &p.analysis,
                        &root,
                        PrimaryPos::Num,
                        p.secondary_pos,
                        &stem,
                    );
                    p.lemma = root.clone();
                    p.root = root;
                    p
                })
                .collect();
            if !readings.is_empty() {
                return readings;
            }
        }
        Vec::new()
    }

    /// A word with the period of an abbreviation, which the tokenizer keeps (Dr., vb., No.,
    /// Av., TCK.): the abbreviation's reading, one guessed for a word in capitals the lexicon
    /// lacks, or else the word's own (madde., where the tokenizer took a sentence's period for
    /// an abbreviation's).
    fn analyze_dotted(&self, clean: &str, lower: &str) -> Vec<MorphParse> {
        let original = clean.strip_suffix('.').unwrap_or(clean);
        let stem = lower.strip_suffix('.').unwrap_or(lower);
        if original.is_empty() || original.contains(['.', '\'']) {
            return Vec::new();
        }
        let own = self.analyze_unranked(original);
        let mut parses: Vec<MorphParse> = own
            .iter()
            .filter(|p| p.secondary_pos == SecondaryPos::Abbreviation && p.root == stem)
            .cloned()
            .collect();
        if parses.is_empty()
            && original.chars().nth(1).is_some()
            && original.chars().all(char::is_uppercase)
        {
            parses = self.analyze_with_apostrophe(original, stem, "");
        }
        if parses.is_empty() {
            parses = own;
        }
        for p in &mut parses {
            p.surface = clean.to_string();
        }
        parses
    }

    /// A name written with its capital and a case or copula ending but without the apostrophe
    /// (Türkiyede, Avrupanın, Asyadaki, Ermeniyim, TBMMnin): the name's reading, at the factor
    /// of a name's suffixes without one, for a word that has no other. A left-out apostrophe is
    /// told from a name the lexicon lacks only where the ending is one lower-case case or
    /// copula ending that starts with a consonant: a vowel after a name is as often the end of
    /// another name (Saide is not Said+Dat, Türkeri not Türker+Acc, Özbakım not
    /// özbak+CopPres1sg), and so is -ye after a name in -i (Nuriye, Zuhriye). In lower case
    /// the word is more likely a misspelling read through one of the names collected from text
    /// (burnda, not Burn+Loc), so istanbulda gets no reading; the sparse encoder's backoff stem
    /// is what matches it to İstanbul. Not for a name that is also a common word (Kitapı is a
    /// misspelt kitabı, not the name Kitap with its accusative), nor for a name of fewer than
    /// three letters, an abbreviation of one letter (Ade is no A+Loc) or one not written in
    /// capitals.
    fn analyze_name_without_apostrophe(&self, clean: &str, lower: &str) -> Vec<MorphParse> {
        if !clean.starts_with(char::is_uppercase) {
            return Vec::new();
        }
        let named = |i: &DictionaryItem| {
            matches!(
                i.secondary_pos,
                SecondaryPos::ProperNoun | SecondaryPos::Abbreviation
            )
        };
        let candidates: Vec<(&StemTransition, &DictionaryItem)> = self
            .stems
            .prefix_matches(lower)
            .map(|s| (s, self.stems.item(s)))
            .filter(|(s, item)| {
                let stem_chars = s.surface.chars().count();
                let mut ending = clean.chars().skip(stem_chars).peekable();
                if ending.peek().is_none_or(|&c| super::phonetics::is_vowel(c))
                    || ending.any(char::is_uppercase)
                {
                    return false;
                }
                match item.secondary_pos {
                    SecondaryPos::ProperNoun => {
                        stem_chars >= 3 && self.root_items(&item.root).iter().all(named)
                    }
                    SecondaryPos::Abbreviation => {
                        stem_chars >= 2 && clean.chars().take(stem_chars).all(|c| !c.is_lowercase())
                    }
                    _ => false,
                }
            })
            .collect();
        if candidates.is_empty() {
            return Vec::new();
        }
        let mut results = Vec::new();
        self.search(&candidates, lower, clean, true, &mut results);
        results.retain(|p| match p.morpheme_tags.get(1..) {
            Some([dative]) if dative == "Dat" => !p.root.ends_with(['i', 'ı']),
            Some([one]) => matches!(
                one.as_str(),
                "Loc"
                    | "Abl"
                    | "Gen"
                    | "Acc"
                    | "Ins"
                    | "CopPres1sg"
                    | "CopPres2sg"
                    | "CopPres1pl"
                    | "CopPres2pl"
            ),
            Some([case, rel]) => matches!(case.as_str(), "Loc" | "Gen") && rel == "RelClitic",
            _ => false,
        });
        finish(results)
    }

    /// `apostrophe`: the word had one (Ankara'ya), so a name may take any suffix after it.
    fn analyze_normalized(&self, clean: &str, lower: &str, apostrophe: bool) -> Vec<MorphParse> {
        let candidates: Vec<(&StemTransition, &DictionaryItem)> = self
            .stems
            .prefix_matches(lower)
            .map(|s| (s, self.stems.item(s)))
            .collect();
        let mut results = Vec::new();
        self.search(&candidates, lower, clean, apostrophe, &mut results);
        finish(results)
    }

    fn search(
        &self,
        candidates: &[(&StemTransition, &DictionaryItem)],
        input: &str,
        original: &str,
        apostrophe: bool,
        results: &mut Vec<MorphParse>,
    ) {
        let engine = &*ENGINE;
        let mut search = Search {
            graph: &engine.graph,
            input,
            steps: Vec::new(),
            dead: HashSet::new(),
            work: 0,
            found: Vec::new(),
        };
        for (i, (stem, item)) in candidates.iter().enumerate() {
            let root_morpheme = engine.graph.states[stem.to.0 as usize].morpheme;
            let ctx = Context::at_root(stem.to, root_morpheme);
            let start = stem.surface.len();
            search.dfs(i, stem, item, start, ctx, stem.attrs);
        }
        let Search { found, .. } = search;
        let first = results.len();
        // Readings through a derivation that spells a lexicon verb (kat-ıl = katıl, hava-lan =
        // havalan), with that verb and whether the derivation starts from a noun or adjective.
        let mut twins: Vec<(usize, &DictionaryItem, bool)> = Vec::new();
        let mut rare: Vec<usize> = Vec::new();
        for (i, steps) in found {
            let (stem, item) = &candidates[i];
            if !apostrophe && !suffixes_without_apostrophe(&engine.graph, item, &steps) {
                continue;
            }
            let index = results.len();
            if item.attributes.contains(RootAttr::RARE) {
                rare.push(index);
            }
            let nominal = item.primary_pos != PrimaryPos::Verb;
            for end in derived_verb_stems(&engine.graph, &steps) {
                twins.extend(
                    self.root_items(&input[..end])
                        .iter()
                        .filter(|v| v.primary_pos == PrimaryPos::Verb && v.lemma != item.lemma)
                        .map(|v| (index, v, nominal)),
                );
            }
            results.push(render(engine, item, stem, &steps, input, original));
        }
        // The lexicon decides which of the two wins (see `prior_yields`): a verb made from a
        // noun or adjective is listed because it is a word of its own (kaynak-lan "arise",
        // uzak-laş "move away"), so it wins unless marked NotLexicalized; between a verb and a
        // voice derivation the verb's attribute decides, and with none the prior does.
        for (derived, verb, nominal) in twins {
            if verb.attributes.contains(RootAttr::NOT_LEXICALIZED) {
                for p in &mut results[first..] {
                    if p.lemma == verb.lemma
                        && p.root == verb.root
                        && p.primary_pos == PrimaryPos::Verb
                    {
                        p.yields = true;
                    }
                }
            } else if nominal || verb.attributes.contains(RootAttr::LEXICALIZED) {
                results[derived].yields = true;
            }
        }
        if rare.len() < results.len() - first {
            for i in rare {
                results[i].yields = true;
            }
        }
        if results[first..].iter().any(|p| !p.nonstandard) {
            for p in &mut results[first..] {
                p.yields |= p.nonstandard;
            }
        }
    }
}

/// The ends (byte offsets into the input) of the verb stems a path's leading derivations spell:
/// voice suffixes on a verb (kat-ıl-dı has katıl, bul-un-dur-ul-du has bulun and bulundur) and
/// the verb-making -lAş and -lAn on a noun or adjective (hava-lan-dır-ma has havalan and
/// havalandır).
fn derived_verb_stems<'s>(graph: &'s Graph, steps: &'s [Step]) -> impl Iterator<Item = usize> + 's {
    steps
        .iter()
        .map(|s| {
            (
                graph.states[graph.transitions[s.transition as usize].to.0 as usize].morpheme,
                s,
            )
        })
        .take_while(|(m, s)| {
            s.end == s.start
                || matches!(
                    m,
                    M::Pass | M::Caus | M::Recip | M::Reflex | M::Become | M::Acquire
                )
        })
        .filter(|(m, s)| {
            s.end > s.start
                && matches!(
                    m,
                    M::Pass | M::Caus | M::Recip | M::Reflex | M::Become | M::Acquire
                )
        })
        .map(|(_, s)| s.end)
}

/// An abbreviation in capitals with its suffixes after a period (TCK.nun; H.U.M.K.nun, whose
/// letters are dotted too): the abbreviation without its periods, and the suffixes.
fn dotted_suffix(word: &str) -> Option<(String, &str)> {
    let (abbreviation, ending) = word.rsplit_once('.')?;
    if ending.is_empty() || !ending.chars().all(char::is_lowercase) {
        return None;
    }
    let solid = !abbreviation.contains('.');
    let initials = abbreviation.split('.').all(|l| l.chars().count() == 1);
    let letters: String = abbreviation.chars().filter(|&c| c != '.').collect();
    (letters.chars().nth(1).is_some()
        && letters.chars().all(char::is_uppercase)
        && (solid || initials))
        .then_some((letters, ending))
}

/// Whether a name's suffixes may be written without an apostrophe. Turkish spelling separates a
/// proper noun's case, possessive and copula with one (Ankara'ya, Ayşe'nin, Ali'dir) but not its
/// plural or derivations (Ahmetler, Ankaralı, Atatürkçü), and every suffix of an abbreviation
/// (TBMM'de). Without the rule a misspelled common word read as a name plus a suffix: kitap-ı
/// through the proper noun Kitap.
fn suffixes_without_apostrophe(graph: &Graph, item: &DictionaryItem, steps: &[Step]) -> bool {
    match item.secondary_pos {
        SecondaryPos::Abbreviation => steps.iter().all(|s| s.end == s.start),
        // Language names are -CA derivations, which take their suffixes without an
        // apostrophe (Türkçenin, İngilizcede).
        SecondaryPos::ProperNoun
            if ["ca", "ce", "ça", "çe"]
                .iter()
                .any(|e| item.root.ends_with(e)) =>
        {
            true
        }
        SecondaryPos::ProperNoun => {
            for s in steps {
                let state = &graph.states[graph.transitions[s.transition as usize].to.0 as usize];
                let has = s.end > s.start;
                if (state.derivative && state.morpheme != M::Zero)
                    || (has && state.morpheme == M::A3pl)
                {
                    return true;
                }
                if has {
                    return false;
                }
            }
            true
        }
        _ => true,
    }
}

/// One suffix taken on a path: the transition and the byte range of its surface.
#[derive(Clone, Copy)]
pub(super) struct Step {
    pub transition: u32,
    pub start: usize,
    pub end: usize,
}

struct Search<'a> {
    graph: &'a Graph,
    input: &'a str,
    steps: Vec<Step>,
    /// Sub-searches known to find nothing: a hash of (candidate, position, phonetic attributes,
    /// context). Conditions read nothing else, so an equal key has the same continuations.
    dead: HashSet<u64>,
    work: usize,
    found: Vec<(usize, Vec<Step>)>,
}

impl Search<'_> {
    fn key(candidate: usize, pos: usize, ctx: &Context, attrs: P) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        candidate.hash(&mut h);
        pos.hash(&mut h);
        ctx.hash(&mut h);
        attrs.bits().hash(&mut h);
        h.finish()
    }

    /// Searches on from `pos`; returns whether any analysis was found below.
    fn dfs(
        &mut self,
        candidate: usize,
        stem: &StemTransition,
        item: &DictionaryItem,
        pos: usize,
        ctx: Context,
        attrs: P,
    ) -> bool {
        self.work += 1;
        if self.work > MAX_WORK {
            return false;
        }
        let tail = &self.input[pos..];
        let state = &self.graph.states[ctx.current.0 as usize];
        if tail.is_empty() && state.terminal && !attrs.contains(P::CANNOT_TERMINATE) {
            self.found.push((candidate, self.steps.clone()));
            return true;
        }
        let key = (self.work > MEMO_AFTER).then(|| Self::key(candidate, pos, &ctx, attrs));
        if key.is_some_and(|k| self.dead.contains(&k)) {
            return false;
        }
        let graph = self.graph;
        let mut found = false;
        for (id, t) in graph.outgoing(ctx.current) {
            // A NoSuffix root (a letter name, an indeclinable such as dekore) takes no suffix
            // with a surface.
            if t.has_surface() && (tail.is_empty() || item.attributes.contains(RootAttr::NO_SUFFIX))
            {
                continue;
            }
            let Some(surface) = t.surface(attrs) else {
                continue;
            };
            if !tail.starts_with(surface) {
                continue;
            }
            let view = PathView {
                ctx: &ctx,
                item,
                stem_surface: &stem.surface,
                attrs,
                tail_empty: tail.is_empty(),
            };
            if !t.condition.eval(graph, &view) {
                continue;
            }
            let repeats = self
                .steps
                .iter()
                .filter(|s| graph.transitions[s.transition as usize].to == t.to)
                .count();
            if repeats >= MAX_STATE_REPEAT {
                continue;
            }
            let next_attrs = if t.has_surface() {
                let mut a = attributes_of(surface, attrs);
                a -= P::CANNOT_TERMINATE;
                match t.ends_voiced() {
                    Some(false) => a |= P::EXPECTS_CONSONANT,
                    Some(true) => a |= P::EXPECTS_VOWEL | P::CANNOT_TERMINATE,
                    None => {}
                }
                a
            } else {
                attrs
            };
            let next_ctx = ctx.step(graph, t.to, !surface.is_empty());
            self.steps.push(Step {
                transition: id,
                start: pos,
                end: pos + surface.len(),
            });
            found |= self.dfs(
                candidate,
                stem,
                item,
                pos + surface.len(),
                next_ctx,
                next_attrs,
            );
            self.steps.pop();
        }
        if let Some(key) = key.filter(|_| !found && self.work <= MAX_WORK) {
            self.dead.insert(key);
        }
        found
    }
}

/// Akana's tag for a morpheme in its context, or `None` for one that is not shown (the
/// default A3sg, Pnon and Nom, zero derivations, POS markers).
pub(super) fn tag_of(m: M) -> Option<&'static str> {
    Some(match m {
        M::Dat => "Dat",
        M::Acc => "Acc",
        M::Abl => "Abl",
        M::Loc => "Loc",
        M::Ins => "Ins",
        M::Gen => "Gen",
        M::Equ => "Equ",
        M::Dim => "Dim",
        M::Ness => "Ness",
        M::With => "With",
        M::Without => "Without",
        M::Related => "Related",
        M::JustLike => "JustLike",
        M::Rel => "RelClitic",
        M::Agt => "Agt",
        M::Become => "BecomeVerb",
        M::Acquire => "Acquire",
        M::Ly => "Ly",
        M::Caus => "Caus",
        M::Able => "Abil",
        M::Pass => "Pass",
        M::Inf1 => "Inf",
        M::Inf2 => "ActN",
        M::Inf3 => "Manner",
        M::ActOf => "ActOf",
        M::PastPart => "PastPart",
        M::NarrPart => "NarrPart",
        M::FutPart => "FutPart",
        M::PresPart => "PresPart",
        M::AorPart => "AorPart",
        M::NotState => "NotState",
        M::FeelLike => "FeelLike",
        M::EverSince => "EverSince",
        M::Repeat => "Repeat",
        M::Almost => "Almost",
        M::Hastily => "Hastily",
        M::Stay => "Stay",
        M::Start => "Start",
        M::AsIf => "AsIf",
        M::While => "While",
        M::When => "AdvInce",
        M::SinceDoingSo => "SinceDoingSo",
        M::AsLongAs => "AsLongAs",
        M::ByDoingSo => "AdvErek",
        M::Adamantly => "Adamantly",
        M::AfterDoingSo => "AdvIp",
        M::WithoutHavingDoneSo => "WithoutHavingDoneSo",
        M::WithoutBeingAbleToHaveDoneSo => "WithoutBeingAbleToHaveDoneSo",
        M::Unable => "NegAbil",
        M::Prog1 | M::Prog2 => "Prog",
        M::Aor => "Aor",
        M::Fut => "Fut",
        M::Imp => "Imp",
        M::Opt => "Opt",
        M::Desr => "Cond",
        M::Neces => "Nec",
        _ => return None,
    })
}

const PERSONS: &[&str] = &[
    "A1sg",
    "A2sg",
    "A1pl",
    "A2pl",
    "A3pl",
    "CopPres1sg",
    "CopPres2sg",
    "CopPres1pl",
    "CopPres2pl",
    "CopPres3pl",
];

/// Renders a path as akana's tags: the root's POS, then the morphemes that are shown. A
/// nominal predicate shows its persons and tenses as the copula (evde-yim CopPres1sg,
/// güzel-di CopPast), as does a tense after a tense (geliyor-du CopPast).
pub(super) fn render(
    engine: &Engine,
    item: &DictionaryItem,
    stem: &StemTransition,
    steps: &[Step],
    input: &str,
    original: &str,
) -> MorphParse {
    let graph = &engine.graph;
    let roots = &engine.roots;
    let mut tags: Vec<&'static str> = vec![item.primary_pos.as_str()];
    let mut pos = item.primary_pos;
    let mut copula = roots.nominal_verb.contains(stem.to);
    // A tense seen in the current verbal group: a later past or evidential is the copula.
    let mut tense = false;
    let mut any_surface = false;
    let mut agreement: Option<M> = None;
    // A person shown in the current group: the copula after it is Cop, not CopPres3sg.
    let mut person = false;
    let pronoun = item.primary_pos == PrimaryPos::Pron;
    let degil = item.lemma == "değil";
    for step in steps {
        let t = &graph.transitions[step.transition as usize];
        let state = &graph.states[t.to.0 as usize];
        let m = state.morpheme;
        let has = step.end > step.start;
        any_surface |= has;
        if state.pos_root {
            if roots.nominal_verb.contains(t.to) {
                copula = true;
            } else {
                copula = false;
                if let Some(p) = m.pos() {
                    pos = p;
                }
            }
            tense = false;
            person = false;
            continue;
        }
        if state.derivative && m != M::Zero {
            copula = false;
            tense = false;
            person = false;
        }
        let tag: Option<&'static str> = match m {
            M::A3sg => {
                agreement.get_or_insert(M::A3sg);
                None
            }
            M::A3pl if !has => {
                agreement.get_or_insert(M::A3pl);
                None
            }
            M::A1sg | M::A2sg | M::A1pl | M::A2pl | M::A3pl => {
                agreement.get_or_insert(m);
                let plain = match m {
                    M::A1sg => "A1sg",
                    M::A2sg => "A2sg",
                    M::A1pl => "A1pl",
                    M::A2pl => "A2pl",
                    _ => "A3pl",
                };
                if copula && tags.last() == Some(&"Pres") {
                    tags.pop();
                    Some(match m {
                        M::A1sg => "CopPres1sg",
                        M::A2sg => "CopPres2sg",
                        M::A1pl => "CopPres1pl",
                        M::A2pl => "CopPres2pl",
                        _ => "CopPres3pl",
                    })
                } else if pronoun || copula || tense || has || m == M::A2sg {
                    Some(plain)
                } else {
                    None
                }
            }
            M::P1sg | M::P2sg | M::P3sg | M::P1pl | M::P2pl | M::P3pl => {
                let implicit = (m == M::P1sg && item.attributes.contains(RootAttr::IMPLICIT_P1SG))
                    || (m == M::P2sg && item.attributes.contains(RootAttr::IMPLICIT_P2SG));
                (has || implicit).then(|| m.id())
            }
            M::Pres => {
                // Shown only through the person that follows (CopPres1sg); a placeholder
                // until then.
                copula.then_some("Pres")
            }
            M::Past => Some(if copula || tense { "CopPast" } else { "Past" }),
            M::Narr => Some(if copula || tense { "CopEvid" } else { "Evid" }),
            M::Cond => Some("CopCond"),
            M::Cop => {
                if tags.last() == Some(&"Pres") {
                    tags.pop();
                }
                Some(if person { "Cop" } else { "CopPres3sg" })
            }
            M::Neg => (!degil).then_some("Neg"),
            M::Recip | M::Reflex if !has => None,
            M::Recip => Some("Recip"),
            M::Reflex => Some("Reflex"),
            m => tag_of(m),
        };
        if let Some(tag) = tag {
            if PERSONS.contains(&tag) {
                person = true;
            }
            tags.push(tag);
        }
        if matches!(
            m,
            M::Prog1 | M::Prog2 | M::Aor | M::Fut | M::Past | M::Narr | M::Opt | M::Desr | M::Neces
        ) {
            tense = true;
        }
    }
    // A present-tense placeholder never followed by a person is the silent 3rd person.
    tags.retain(|t| *t != "Pres");
    // A bare noun or pronoun spells out its number and case, as before: ev is Noun+A3sg+Nom.
    if !any_surface
        && matches!(item.primary_pos, PrimaryPos::Noun | PrimaryPos::Pron)
        && tags[1..].iter().all(|t| PERSONS.contains(t))
    {
        if tags.len() == 1 {
            tags.push(match agreement {
                Some(M::A3pl) => "A3pl",
                _ => "A3sg",
            });
        }
        tags.push("Nom");
    }

    // `[lemma:Pos] root:Tag+Tag`, written in place (format! with tags.join allocated twice).
    let mut formatted =
        String::with_capacity(item.lemma.len() + item.root.len() + 8 * tags.len() + 8);
    formatted.push('[');
    formatted.push_str(&item.lemma);
    formatted.push(':');
    formatted.push_str(item.primary_pos.as_str());
    formatted.push_str("] ");
    formatted.push_str(&item.root);
    formatted.push(':');
    for (i, tag) in tags.iter().enumerate() {
        if i > 0 {
            formatted.push('+');
        }
        formatted.push_str(tag);
    }
    let bare = !any_surface && input == stem.surface.as_ref();
    let named = matches!(
        item.secondary_pos,
        SecondaryPos::ProperNoun | SecondaryPos::Abbreviation
    );
    let is_upper = original.chars().next().is_some_and(|c| c.is_uppercase());
    let has_apostrophe = original.contains(APOSTROPHES);
    // Turkish spelling separates the suffixes of a proper noun or abbreviation with an
    // apostrophe (Ayşe'nin, TBMM'de), so an inflected reading of one written without it
    // (k-ı "ki", Anne-m "Annem") is unlikely; so is a proper noun written lower-case.
    // Each factor with its log in rank units (ln f / ln 1.02 * 16).
    let (score, rank) = if bare {
        if named && !is_upper {
            (0.65, -348)
        } else if item.primary_pos == PrimaryPos::Verb {
            (0.95, -41)
        } else {
            (1.0, 0)
        }
    } else if named {
        if has_apostrophe {
            (0.98, -16)
        } else {
            (0.6, -413)
        }
    } else {
        (0.95, -41)
    };
    // A spelling TDK does not write counts half.
    let nonstandard = item.attributes.contains(RootAttr::NONSTANDARD);
    let (score, rank) = if nonstandard {
        (score * NONSTANDARD_FACTOR, rank + NONSTANDARD_UNITS)
    } else {
        (score, rank)
    };
    MorphParse {
        surface: original.to_string(),
        lemma: item.lemma.clone(),
        root: item.root.clone(),
        primary_pos: item.primary_pos,
        pos,
        secondary_pos: item.secondary_pos,
        morpheme_tags: tags.into_iter().map(str::to_string).collect(),
        formatted,
        nonstandard,
        analysis: zemberek_analysis(graph, item, stem, steps, input),
        score,
        yields: false,
        rank,
        homograph: item.homograph,
        path: MorphPath {
            root: Some(stem.to),
            steps: steps
                .iter()
                .map(|s| PathStep {
                    state: graph.transitions[s.transition as usize].to,
                    overt: s.end > s.start,
                })
                .collect(),
        },
    }
}

/// A path in Zemberek's default format (`AnalysisFormatters.DEFAULT`): the dictionary item
/// with its lemma as the dictionary spells it ([`DictionaryItem::display_lemma`]), then the stem
/// with its POS morpheme and each suffix morpheme, `surface:Id` or `Id`, joined by `+`, with `|`
/// before a derivational morpheme and `→` after it. Pnon and Nom are left out, as Zemberek's
/// `SingleAnalysis.fromSearchPath` leaves them out.
fn zemberek_analysis(
    graph: &Graph,
    item: &DictionaryItem,
    stem: &StemTransition,
    steps: &[Step],
    input: &str,
) -> String {
    let mut out = String::with_capacity(item.lemma.len() + 2 * input.len() + 6 * steps.len() + 16);
    out.push('[');
    item.push_display_lemma(&mut out);
    push_zemberek_pos(&mut out, item.primary_pos, item.secondary_pos);
    out.push_str(&stem.surface);
    out.push(':');
    out.push_str(graph.states[stem.to.0 as usize].morpheme.id());
    let mut derived = false;
    for step in steps {
        let m = graph.states[graph.transitions[step.transition as usize].to.0 as usize].morpheme;
        if matches!(m, M::Pnon | M::Nom) {
            continue;
        }
        let derivation = m.is_derivational();
        if derivation {
            out.push('|');
        } else if !derived {
            out.push('+');
        }
        if step.end > step.start {
            out.push_str(&input[step.start..step.end]);
            out.push(':');
        }
        out.push_str(m.id());
        if derivation {
            out.push('→');
        }
        derived = derivation;
    }
    out
}

/// Zemberek's dictionary item part, `[lemma:Pos] ` or `[lemma:Pos,Secondary] `; its POS short
/// forms are akana's but `Ques` for the question particle.
fn push_zemberek_item(out: &mut String, lemma: &str, pos: PrimaryPos, secondary: SecondaryPos) {
    out.push('[');
    out.push_str(lemma);
    push_zemberek_pos(out, pos, secondary);
}

/// The rest of [`push_zemberek_item`] after the lemma: `:Pos] ` or `:Pos,Secondary] `.
fn push_zemberek_pos(out: &mut String, pos: PrimaryPos, secondary: SecondaryPos) {
    out.push(':');
    out.push_str(match pos {
        PrimaryPos::Q => "Ques",
        pos => pos.as_str(),
    });
    if secondary != SecondaryPos::None {
        out.push(',');
        out.push_str(secondary.as_str());
    }
    out.push_str("] ");
}

/// A Zemberek analysis with its dictionary item and stem replaced, for a word read through
/// another (1923'te through üç-te: `[1923:Num,Card] 1923:Num|Zero→Noun+A3sg+te:Loc`, as
/// Zemberek's numeral analysis writes it).
fn restem_analysis(
    analysis: &str,
    lemma: &str,
    pos: PrimaryPos,
    secondary: SecondaryPos,
    stem: &str,
) -> String {
    // The stem is the item's root or a form of it, without a ':'.
    let body = analysis.split_once("] ").map_or(analysis, |(_, body)| body);
    let rest = body.find(':').map_or("", |i| &body[i..]);
    let mut out = String::with_capacity(lemma.len() + stem.len() + rest.len() + 16);
    push_zemberek_item(&mut out, lemma, pos, secondary);
    out.push_str(stem);
    out.push_str(rest);
    out
}

/// Whether a stem can end as a Turkish word does: not in a stop followed by a consonant
/// (kitapç; ks is, in boks), nor in a voiced stop after a consonant but n or r (istanbuld;
/// android has one consonant).
fn word_final(stem: &str) -> bool {
    let mut letters = stem.chars().rev();
    let (Some(last), Some(before)) = (letters.next(), letters.next()) else {
        return true;
    };
    if super::phonetics::is_vowel(last) || super::phonetics::is_vowel(before) {
        return true;
    }
    let stop = |c: char| matches!(c, 'p' | 'ç' | 't' | 'k' | 'b' | 'c' | 'd' | 'g' | 'ğ');
    let voiced_stop = matches!(last, 'b' | 'c' | 'd' | 'g' | 'ğ');
    !(stop(before) && !(before == 'k' && last == 's'))
        && !(voiced_stop && !matches!(before, 'n' | 'r'))
}

/// The inflections [`TurkishMorphology::backoff_stem`] takes off an unknown word: plural,
/// possessive (but the singular first and second persons'), case, -ki and the third person's
/// copula. -m and -n are left on, as more often the end of a foreign word (bitcoin is no
/// bitcoi-n, instagram no instagra-m), and so is the equative (Endonezyaca is a language).
const INFLECTION_TAGS: &[&str] = &[
    "A3pl",
    "P3sg",
    "P1pl",
    "P2pl",
    "P3pl",
    "Gen",
    "Dat",
    "Acc",
    "Loc",
    "Abl",
    "Ins",
    "RelClitic",
    "Cop",
    "CopPres3sg",
    "CopPres3pl",
    "CopPast",
    "CopEvid",
    "CopCond",
];

/// Derivational tags: an analysis without them is preferred among equals.
const DERIVATIONAL_TAGS: &[&str] = &[
    "Dim",
    "Ness",
    "With",
    "Without",
    "Related",
    "JustLike",
    "RelClitic",
    "Agt",
    "BecomeVerb",
    "Acquire",
    "Ly",
    "Caus",
    "Abil",
    "Pass",
    "Inf",
    "ActN",
    "Manner",
    "ActOf",
    "PastPart",
    "NarrPart",
    "FutPart",
    "PresPart",
    "AorPart",
    "NotState",
    "FeelLike",
    "EverSince",
    "Repeat",
    "Almost",
    "Hastily",
    "Stay",
    "Start",
    "AsIf",
    "While",
    "AdvInce",
    "SinceDoingSo",
    "AsLongAs",
    "AdvErek",
    "Adamantly",
    "AdvIp",
    "WithoutHavingDoneSo",
    "WithoutBeingAbleToHaveDoneSo",
    "Recip",
    "Reflex",
];

/// Sorts (see [`order`]) and drops repeats of the same reading: the same Zemberek analysis
/// after its dictionary item, of the same lemma, root and POS. Readings that akana's tags render
/// alike stay apart when their analyses differ: a participle is a noun (ol-duğ-u "that it is",
/// PastPart→Noun) and an adjective (ol-duğ-u "which it is", PastPart→Adj), -IcI an agent noun
/// and an adjective (bağla-yıcı). The root tells apart items whose stems spell alike (irad-ı from
/// irad and from irat), which the analysis does not. Homographs that differ only in spelling
/// (adet "count" and âdet "custom", metin and metîn) stay one reading, as the order spells it.
fn finish(mut results: Vec<MorphParse>) -> Vec<MorphParse> {
    order(&mut results);
    let mut seen = HashSet::with_capacity(results.len());
    let first: Vec<bool> = results
        .iter()
        .map(|p| {
            let body = p
                .analysis
                .split_once("] ")
                .map_or(p.analysis.as_str(), |(_, b)| b);
            seen.insert((
                body,
                p.lemma.as_str(),
                p.root.as_str(),
                p.primary_pos,
                p.secondary_pos,
            ))
        })
        .collect();
    let mut first = first.into_iter();
    results.retain(|_| first.next().unwrap_or(true));
    results
}

/// Where a reading's tags are one of a set that the corpus cannot tell apart, because every
/// word that has one has the others too:
/// - the readings of -lArI after a noun: kitap-lar-ı A3pl+P3sg "his books", A3pl+Acc "the
///   books" (at the end of the word), A3pl+P3pl "their books", kitap-ları P3pl "their book";
/// - the readings of -lIğI: başkan-lığ-ı Ness+P3sg, Ness+Acc.
///
/// The estimator leaves their ratio where its start put it, so any change of the lexicon could
/// reorder them. Instead they share their best prior and come in a fixed order
/// ([`TurkishMorphology::analyze`]): the order of their frequency in UD Turkish's train and
/// dev splits (BOUN, Kenet, Penn, Tourism, Atis, FrameNet). Of 9,329 nouns ending in -lArI,
/// 76.5 % are plural with a 3sg possessor, 14.6 % plural accusatives, 2.7 % plural with a 3pl
/// possessor and 1.6 % singular with a 3pl possessor; before a case (-lArInA) the shares are
/// 77 %, 14 % and 4 % for A3pl+P3sg, A3pl+P3pl and P3pl. Of 1,166 -lIğI, 77 % are P3sg and 14 %
/// Acc. Context still chooses among them in the disambiguator. The -lArI set is a noun's only:
/// a participle's P3pl is its subject (git-tik-leri "that they went").
#[derive(Clone, Copy, PartialEq, Eq)]
struct CoOccurring {
    /// The tags `start..end` are the ambiguous part.
    start: usize,
    end: usize,
    /// Which set (-lArI 0, -lIğI 1) and the reading's place in it, first 0.
    set: u8,
    place: u8,
}

/// How many tags more than one a reading's co-occurring part has ([`CoOccurring`]):
/// A3pl+P3sg and A3pl+P3pl count as one tag, as P3pl does, wherever readings are compared by
/// their number of morphemes.
pub(crate) fn co_occurring_extra_tags(p: &MorphParse) -> usize {
    co_occurring(p).map_or(0, |c| c.end - c.start - 1)
}

fn co_occurring(p: &MorphParse) -> Option<CoOccurring> {
    let tags = &p.morpheme_tags;
    let n = tags.len();
    let nominal = matches!(p.primary_pos, PrimaryPos::Noun | PrimaryPos::Adj);
    let set = |start, end, set, place| {
        Some(CoOccurring {
            start,
            end,
            set,
            place,
        })
    };
    if nominal && n >= 2 && tags[n - 2] == "Ness" && matches!(tags[n - 1].as_str(), "P3sg" | "Acc")
    {
        return set(n - 1, n, 1, u8::from(tags[n - 1] == "Acc"));
    }
    // A noun's: a participle's P3pl is its subject (git-tik-leri "that they went").
    if p.primary_pos != PrimaryPos::Noun {
        return None;
    }
    let Some(i) = tags.iter().position(|t| t == "P3pl" || t == "P3sg") else {
        return (n >= 2 && tags[n - 2] == "A3pl" && tags[n - 1] == "Acc").then_some(CoOccurring {
            start: n - 2,
            end: n,
            set: 0,
            place: 1,
        });
    };
    let plural = i > 0 && tags[i - 1] == "A3pl";
    match (plural, tags[i].as_str()) {
        (true, "P3sg") => set(i - 1, i + 1, 0, 0),
        (true, _) => set(i - 1, i + 1, 0, 2),
        (false, "P3pl") => set(i, i + 1, 0, 3),
        _ => None,
    }
}

/// Whether two readings are members of one set of [`CoOccurring`] readings: the same item, and
/// the same tags but for the ambiguous part.
fn same_class(a: &MorphParse, pa: CoOccurring, b: &MorphParse, pb: CoOccurring) -> bool {
    pa.set == pb.set
        && a.lemma == b.lemma
        && a.root == b.root
        && a.primary_pos == b.primary_pos
        && a.secondary_pos == b.secondary_pos
        && a.homograph == b.homograph
        && a.morpheme_tags[..pa.start] == b.morpheme_tags[..pb.start]
        && a.morpheme_tags[pa.end..] == b.morpheme_tags[pb.end..]
}

/// Most likely first; ties go to fewer tags, then away from the 2nd-person possessive and the
/// copula (much rarer than the 3rd person and the bare noun), then to the string. Readings
/// that akana's tags render alike tie on all of that: the noun comes before the adjective
/// (ol-duğ-u is "that it is" unless a noun follows, which the disambiguator sees); a word
/// written with a circumflex takes the spelling that has one (âdet "custom", not adet "count":
/// a missing circumflex says nothing, as Turkish often leaves it out); and otherwise the
/// analyzer's order stands (the sort is stable): the lexicon's items in their order, which puts
/// the plainly spelled homograph first, and the compound before its inner P3sg (üzeri-nden
/// before üzer-i-nden).
fn order(results: &mut [MorphParse]) {
    let p2sg = |p: &MorphParse| p.morpheme_tags.iter().any(|t| t == "P2sg");
    let predicate = |p: &MorphParse| {
        p.primary_pos != PrimaryPos::Verb && p.morpheme_tags.iter().any(|t| t.starts_with("Cop"))
    };
    let adjective = |p: &MorphParse| p.pos == PrimaryPos::Adj;
    let circumflex = |s: &str| s.contains(['â', 'î', 'û', 'Â', 'Î', 'Û']);
    let respelled = |p: &MorphParse| circumflex(&p.surface) && !circumflex(p.display_lemma());
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.morpheme_tags.len().cmp(&b.morpheme_tags.len()))
            .then_with(|| p2sg(a).cmp(&p2sg(b)))
            .then_with(|| predicate(a).cmp(&predicate(b)))
            .then_with(|| a.formatted.cmp(&b.formatted))
            .then_with(|| adjective(a).cmp(&adjective(b)))
            .then_with(|| respelled(a).cmp(&respelled(b)))
    });
}
