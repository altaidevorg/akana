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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MorphemeData {
    pub name: String,
    pub surface: String,
    pub tag: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MorphParse {
    pub surface: String,
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
pub(super) static PRIOR: LazyLock<Prior> =
    LazyLock::new(|| Prior::parse(include_bytes!("prior.tsv.gz")));

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

/// The register factor [`prior_second_person`] readings take, in rank units.
pub(crate) fn prior_second_person_units() -> i32 {
    PRIOR.second_person
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
/// - a `Rare` item (bulamak "smear") yields to any other reading of the word (bul-ama-dığı).
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
    /// The register factor of second-person readings.
    second_person: i32,
    /// [`LEXICAL_PREFERENCE`].
    lexical: i32,
    /// log P(case as written | reading): name capitalised, name lower-case, other word
    /// capitalised, other word lower-case.
    case: [i32; 4],
}

impl Prior {
    /// Reads the rows `examples/morph_prior.rs` writes: `I`tems, `S`hapes, `T`ag penalties,
    /// `U`nseen values and fixed facto`R`s, each `kind<TAB>key<TAB>rank units[<TAB>...]`. A
    /// damaged file, or one of another format, only loses the ranking.
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
            lexical: 0,
            case: [0; 4],
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
                ("R", "lexical-preference") => prior.lexical = units,
                ("R", "case-name-capitalised") => prior.case[0] = units,
                ("R", "case-name-lower") => prior.case[1] = units,
                ("R", "case-other-capitalised") => prior.case[2] = units,
                ("R", "case-other-lower") => prior.case[3] = units,
                _ => {}
            }
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
        ];
        self.items
            .get(&KeyHash::of_parts(parts))
            .copied()
            .unwrap_or(self.unseen_item)
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
        let register = if prior_second_person(p) {
            self.second_person
        } else {
            0
        };
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

/// The prior's key for a reading's dictionary item: lemma, root, POS and secondary POS.
pub fn prior_item_key(p: &MorphParse) -> String {
    item_key_parts(p).concat()
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

fn item_key_parts(p: &MorphParse) -> [&str; 7] {
    let secondary = match p.secondary_pos.as_str() {
        "" => "-",
        s => s,
    };
    let pos = p.primary_pos.as_str();
    [&p.lemma, " ", &p.root, " ", pos, " ", secondary]
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

    fn of_parts(parts: [&str; 7]) -> u64 {
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
            let logs: Vec<i32> = parses.iter().map(|p| prior.log_p(p, capitalised)).collect();
            let best = logs.iter().copied().max().unwrap_or(0);
            // Each reading's prior relative to the best, in steps of 2% (rounded half up):
            // readings the corpus cannot tell apart (A3pl+Gen and A3pl+P2sg always compete in
            // the same words) come out of the estimator equal up to rounding, and tie, so the
            // analyzer's order decides between them. Divided by their sum, they are
            // P(reading | word).
            let half = RANK_UNITS_PER_STEP / 2;
            let steps: Vec<u32> = logs
                .iter()
                .map(|&log| ((best - log + half) / RANK_UNITS_PER_STEP).unsigned_abs())
                .collect();
            let total: f32 = steps.iter().map(|&q| step_ratio(q)).sum();
            for (p, q) in parses.iter_mut().zip(steps) {
                p.score *= step_ratio(q) / total;
                p.rank -= q as i32 * RANK_UNITS_PER_STEP;
            }
            order(&mut parses);
        }
        parses
    }

    pub(super) fn analyze_unranked(&self, word: &str) -> Vec<MorphParse> {
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
        // normalizeForAnalysis; every apostrophe-like mark is an apostrophe.
        let lower: String = super::super::phonology::to_turkish_lower(clean)
            .chars()
            .map(|c| match c {
                'â' => 'a',
                'î' => 'i',
                'û' => 'u',
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
                let parses = self.analyze_normalized(clean, &word, false);
                (word, parses)
            }
        };
        if let Some(roman) = self.analyze_roman(clean) {
            // A word the lexicon knows keeps its own readings first (MI, the question particle
            // in capitals; CD); otherwise the numeral replaces a guessed abbreviation (II,
            // XIX., IV'üncü), so II'nın is no word.
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
        let digits: String = if thousands {
            digits_part.replace('.', "")
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
                    && (p.root == word || p.root == super::pronunciation::ordinal_word(word))
            })
            .map(|mut p| {
                let ordinal_reading = p.root != word || ordinal;
                p.formatted = format!("[{root}:Num] {root}:{}", p.morpheme_tags.join("+"));
                p.lemma = root.clone();
                p.root = root.clone();
                p.secondary_pos = if ordinal_reading && !percent {
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
        // voicing).
        let known = self.root_items(stem).iter().any(|i| {
            matches!(
                i.secondary_pos,
                SecondaryPos::ProperNoun | SecondaryPos::Abbreviation
            )
        });
        if known {
            return Vec::new();
        }
        // An upper-case stem is an abbreviation, read letter by letter unless it reads as a
        // word (TCDD'nin "tecedede", NATO'nun).
        let original_stem = clean.rfind(APOSTROPHES).map_or(clean, |i| &clean[..i]);
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
        let stems: Vec<StemTransition> = stems::generate(&guessed, 0, &ENGINE.roots)
            .into_iter()
            .filter(|s| joined.starts_with(s.surface.as_ref()))
            .collect();
        let candidates: Vec<(&StemTransition, &DictionaryItem)> =
            stems.iter().map(|s| (s, &guessed)).collect();
        let mut results = Vec::new();
        self.search(&candidates, &joined, clean, true, &mut results);
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
    MorphParse {
        surface: original.to_string(),
        lemma: item.lemma.clone(),
        root: item.root.clone(),
        primary_pos: item.primary_pos,
        pos,
        secondary_pos: item.secondary_pos,
        morpheme_tags: tags.into_iter().map(str::to_string).collect(),
        formatted,
        analysis: zemberek_analysis(graph, item, stem, steps, input),
        score,
        yields: false,
        rank,
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

/// A path in Zemberek's default format (`AnalysisFormatters.DEFAULT`): the dictionary item,
/// then the stem with its POS morpheme and each suffix morpheme, `surface:Id` or `Id`, joined
/// by `+`, with `|` before a derivational morpheme and `→` after it. Pnon and Nom are left out,
/// as Zemberek's `SingleAnalysis.fromSearchPath` leaves them out.
fn zemberek_analysis(
    graph: &Graph,
    item: &DictionaryItem,
    stem: &StemTransition,
    steps: &[Step],
    input: &str,
) -> String {
    let mut out = String::with_capacity(item.lemma.len() + 2 * input.len() + 6 * steps.len() + 16);
    push_zemberek_item(&mut out, &item.lemma, item.primary_pos, item.secondary_pos);
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

/// Sorts by score, then fewer tags, then away from the 2nd-person possessive (much rarer than
/// the 3rd in running text) and from a nominal predicate (kitab-ım "my book" before "I am a
/// book"), then by `formatted`; and drops repeats of the same reading (`formatted` plus its
/// secondary POS), so two Zemberek readings that are one reading in akana's tags (an empty
/// P3sg or Pnon after a CompoundP3sg noun) keep the first one's `analysis`.
fn finish(mut results: Vec<MorphParse>) -> Vec<MorphParse> {
    order(&mut results);
    let mut seen = HashSet::with_capacity(results.len());
    let first: Vec<bool> = results
        .iter()
        .map(|p| seen.insert((p.formatted.as_str(), p.secondary_pos)))
        .collect();
    let mut first = first.into_iter();
    results.retain(|_| first.next().unwrap_or(true));
    results
}

/// Most likely first; ties go to fewer tags, then away from the 2nd-person possessive and the
/// copula (much rarer than the 3rd person and the bare noun), then to the string.
fn order(results: &mut [MorphParse]) {
    let p2sg = |p: &MorphParse| p.morpheme_tags.iter().any(|t| t == "P2sg");
    let predicate = |p: &MorphParse| {
        p.primary_pos != PrimaryPos::Verb && p.morpheme_tags.iter().any(|t| t.starts_with("Cop"))
    };
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.morpheme_tags.len().cmp(&b.morpheme_tags.len()))
            .then_with(|| p2sg(a).cmp(&p2sg(b)))
            .then_with(|| predicate(a).cmp(&predicate(b)))
            .then_with(|| a.formatted.cmp(&b.formatted))
    });
}
