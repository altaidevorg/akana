//! Turkish morphological analyzer: a port of Zemberek's `RuleBasedAnalyzer` over the
//! morphotactic graph of [`super::turkish_morphotactics`] and the stems of
//! [`super::stems`].

use super::dictionary::{DictionaryItem, RootLexicon};
use super::morphotactics::{Context, Graph, Morpheme as M, PathView};
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
    /// How likely the reading is: a spelling factor (1 for a bare root, 0.95 for a suffixed
    /// word, 0.6 for a name's suffixes written without an apostrophe) times the reading's corpus
    /// prior relative to the word's most likely reading, so the best reading keeps its factor.
    /// Analyses come sorted by it.
    pub score: f32,
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

/// The second-person tags, which the prior scales by its register factor (see
/// `examples/morph_prior.rs`).
pub const SECOND_PERSON: [&str; 4] = ["P2sg", "P2pl", "A2sg", "A2pl"];

/// The ratio between adjacent levels of a reading's prior relative to its word's best reading.
const PRIOR_STEP: f32 = 1.02;

/// Log probabilities of items ([`prior_item_key`]) and of shapes ([`prior_shape_key`]) given
/// their POS (the shape's first tag): P(item) * P(shape | POS) is how likely a reading is.
/// The keys are hashed ([`KeyHash`]), so looking a reading up builds no string.
pub(super) struct Prior {
    items: KeyMap,
    shapes: KeyMap,
    unseen_item: f32,
    /// A shape not in the file, per POS; [`Prior::unseen_shape`] for a POS not in it either.
    unseen_shapes: HashMap<String, f32>,
    unseen_shape: f32,
    /// Per-morpheme backoff for a shape not in the file: log P(tag | POS) over the overt
    /// morphemes of the POS's shapes, `POS+tag` -> log probability. An unseen shape scores the
    /// smoothing count times the probabilities of its morphemes, so one of fewer and more
    /// common morphemes outranks one of more or rarer morphemes.
    tag_penalties: HashMap<String, f32>,
    /// The penalty of a tag a POS's shapes never show.
    unseen_tag_penalty: f32,
    /// Log of the factor for second-person readings.
    second_person: f32,
    /// P(capitalised | name) and P(capitalised | other word).
    capitalised_name: f32,
    capitalised_other: f32,
}

impl Prior {
    /// Reads `I<TAB>key<TAB>count` and `S<TAB>key<TAB>count` lines, smoothing every count by
    /// one half as the estimator does. A damaged file only loses the ranking.
    fn parse(gz: &[u8]) -> Self {
        use std::io::Read;
        let mut text = String::new();
        if flate2::read::GzDecoder::new(gz)
            .read_to_string(&mut text)
            .is_err()
        {
            text.clear();
        }
        // Borrowed from the decompressed text: no key is copied before it is hashed.
        let mut items: Vec<(&str, f64)> = Vec::new();
        let mut shapes: Vec<(&str, f64)> = Vec::new();
        let mut second_person = 0.0;
        // Without the file's estimates the word's case says nothing.
        let mut capitalised_name = 0.5f64;
        let mut capitalised_other = 0.5f64;
        for line in text.lines() {
            let mut fields = line.splitn(3, '\t');
            let (Some(kind), Some(key), Some(count)) =
                (fields.next(), fields.next(), fields.next())
            else {
                continue;
            };
            let Ok(count) = count.parse::<f64>() else {
                continue;
            };
            match kind {
                "I" => items.push((key, count)),
                "S" => shapes.push((key, count)),
                "R" if key == "second-person" && count > 0.0 => second_person = count.ln() as f32,
                "R" if key == "capitalised-name" && count > 0.0 && count < 1.0 => {
                    capitalised_name = count
                }
                "R" if key == "capitalised-other" && count > 0.0 && count < 1.0 => {
                    capitalised_other = count
                }
                _ => {}
            }
        }
        let total = items.iter().map(|(_, c)| c + 0.5).sum::<f64>() + 0.5;
        let unseen_item = (0.5 / total).ln() as f32;
        let items = items
            .iter()
            .map(|&(k, c)| (KeyHash::of(k), ((c + 0.5) / total).ln() as f32))
            .collect();
        fn pos_of(shape: &str) -> &str {
            shape.split('+').next().unwrap_or_default()
        }
        let mut totals: HashMap<&str, f64> = HashMap::new();
        // P(tag | POS) over the overt morphemes of the POS's shapes.
        let mut tag_counts: HashMap<(&str, &str), f64> = HashMap::new();
        let mut tag_totals: HashMap<&str, f64> = HashMap::new();
        for &(k, c) in &shapes {
            *totals.entry(pos_of(k)).or_insert(0.5) += c + 0.5;
            let mut tags = k.split('+');
            let pos = tags.next().unwrap_or_default();
            for tag in tags.filter(|t| !matches!(*t, "A3sg" | "Pnon" | "Nom")) {
                *tag_counts.entry((pos, tag)).or_insert(0.0) += c;
                *tag_totals.entry(pos).or_insert(0.5) += c;
            }
        }
        let shapes = shapes
            .iter()
            .map(|&(k, c)| {
                let total = totals[pos_of(k)];
                (KeyHash::of(k), ((c + 0.5) / total).ln() as f32)
            })
            .collect();
        let unseen_shapes = totals
            .iter()
            .map(|(pos, total)| (pos.to_string(), (0.5 / total).ln() as f32))
            .collect();
        let tag_penalties = tag_counts
            .iter()
            .map(|(&(pos, tag), c)| {
                let total = tag_totals[pos];
                (format!("{pos}+{tag}"), ((c + 0.5) / total).ln() as f32)
            })
            .collect();
        let unseen_tag_penalty =
            (0.5 / tag_totals.values().copied().fold(0.5, f64::max)).ln() as f32;
        Prior {
            items,
            shapes,
            unseen_item,
            unseen_shapes,
            unseen_shape: 0.5f32.ln(),
            tag_penalties,
            unseen_tag_penalty,
            second_person,
            capitalised_name: capitalised_name as f32,
            capitalised_other: capitalised_other as f32,
        }
    }

    /// Log P(item) of a dictionary item, for choosing among homographs (hak the noun).
    pub(super) fn item_log_p(&self, item: &DictionaryItem) -> f32 {
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

    /// `capitalised`: the word as written starts with a capital.
    fn log_p(&self, p: &MorphParse, capitalised: bool) -> f32 {
        let item = self
            .items
            .get(&KeyHash::of_parts(item_key_parts(p)))
            .copied();
        let shape_key = shape_key_parts(p).fold(KeyHash::new(), KeyHash::feed).0;
        let shape = self.shapes.get(&shape_key).copied().unwrap_or_else(|| {
            let pos = p.morpheme_tags.first().map_or("", String::as_str);
            let morphemes: f32 = p.morpheme_tags[1..]
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
                .get(pos)
                .copied()
                .unwrap_or(self.unseen_shape)
                + morphemes
        });
        let register = if p
            .morpheme_tags
            .iter()
            .any(|t| SECOND_PERSON.contains(&t.as_str()))
        {
            self.second_person
        } else {
            0.0
        };
        let p_capitalised = if prior_is_name(p) {
            self.capitalised_name
        } else {
            self.capitalised_other
        };
        let case = if capitalised {
            p_capitalised
        } else {
            1.0 - p_capitalised
        };
        item.unwrap_or(self.unseen_item) + shape + register + case.ln()
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

/// The prior's key for a reading's shape: its tag sequence (Noun+P3sg+Loc).
pub fn prior_shape_key(p: &MorphParse) -> String {
    shape_key_parts(p).collect()
}

fn item_key_parts(p: &MorphParse) -> [&str; 7] {
    let secondary = match p.secondary_pos.as_str() {
        "" => "-",
        s => s,
    };
    let pos = p.primary_pos.as_str();
    [&p.lemma, " ", &p.root, " ", pos, " ", secondary]
}

fn shape_key_parts(p: &MorphParse) -> impl Iterator<Item = &str> {
    p.morpheme_tags
        .iter()
        .enumerate()
        .flat_map(|(i, t)| [if i == 0 { "" } else { "+" }, t.as_str()])
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
type KeyMap = HashMap<u64, f32, std::hash::BuildHasherDefault<IdentityHasher>>;

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
            let logs: Vec<f32> = parses.iter().map(|p| prior.log_p(p, capitalised)).collect();
            let best = logs.iter().copied().fold(f32::MIN, f32::max);
            for (p, log) in parses.iter_mut().zip(logs) {
                // In steps of 2%: readings the corpus cannot tell apart (A3pl+Gen and
                // A3pl+P2sg always compete in the same words) come out of the estimator equal
                // up to rounding, and tie, so the analyzer's order decides between them.
                p.score *= PRIOR_STEP.powf(((log - best) / PRIOR_STEP.ln()).round());
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
        for (i, steps) in found {
            let (stem, item) = &candidates[i];
            if !apostrophe && !suffixes_without_apostrophe(&engine.graph, item, &steps) {
                continue;
            }
            results.push(render(engine, item, stem, &steps, input, original));
        }
    }
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
            if tail.is_empty() && t.has_surface() {
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

    let formatted = format!(
        "[{}:{}] {}:{}",
        item.lemma,
        item.primary_pos.as_str(),
        item.root,
        tags.join("+")
    );
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
    let score = if bare {
        if named && !is_upper {
            0.65
        } else if item.primary_pos == PrimaryPos::Verb {
            0.95
        } else {
            1.0
        }
    } else if named {
        if has_apostrophe {
            0.98
        } else {
            0.6
        }
    } else {
        0.95
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
        score,
    }
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
/// book"), then by the analysis string; and drops repeats of the same reading (the analysis
/// string plus its secondary POS).
fn finish(mut results: Vec<MorphParse>) -> Vec<MorphParse> {
    order(&mut results);
    let mut seen = HashSet::new();
    results.retain(|p| seen.insert((p.formatted.clone(), p.secondary_pos)));
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
