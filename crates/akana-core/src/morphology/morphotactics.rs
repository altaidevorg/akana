//! The morphotactic graph: morphemes, the states a word passes through, and the suffix
//! transitions between them, after Zemberek's `TurkishMorphotactics`.
//!
//! A transition carries a morphophonemic template (`+yA`, `>dA`, `lI~k`), so its surface is
//! decided by the letters before it rather than listed allomorph by allomorph, and a condition
//! on what the word has passed through so far. Conditions read only a [`Context`] that the
//! search keeps up to date step by step, so a sub-search that failed from one context fails
//! from every equal one and can be remembered.

use super::dictionary::DictionaryItem;
use super::phonetics::{attributes_of, devoice, PhoneticAttrs};
use super::pos::{PrimaryPos, RootAttr, SecondaryPos};

macro_rules! morphemes {
    ($($name:ident = $id:literal, $deriv:literal, $pos:expr;)*) => {
        /// A morpheme of Zemberek's inventory; `id()` is its Zemberek name.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        #[repr(u8)]
        pub enum Morpheme { $($name,)* }

        impl Morpheme {
            pub const ALL: &'static [Morpheme] = &[$(Morpheme::$name,)*];

            /// Zemberek's id (`A3pl`, `Inf1`, `Narr`).
            pub fn id(self) -> &'static str {
                match self { $(Morpheme::$name => $id,)* }
            }

            /// Whether the morpheme opens a new inflectional group (a derivation).
            pub fn is_derivational(self) -> bool {
                match self { $(Morpheme::$name => $deriv,)* }
            }

            /// The POS a POS morpheme stands for.
            pub fn pos(self) -> Option<PrimaryPos> {
                match self { $(Morpheme::$name => $pos,)* }
            }
        }
    };
}

morphemes! {
    Root = "Root", false, None;
    Noun = "Noun", false, Some(PrimaryPos::Noun);
    Adj = "Adj", false, Some(PrimaryPos::Adj);
    Verb = "Verb", false, Some(PrimaryPos::Verb);
    Pron = "Pron", false, Some(PrimaryPos::Pron);
    Adv = "Adv", false, Some(PrimaryPos::Adv);
    Conj = "Conj", false, Some(PrimaryPos::Conj);
    Punc = "Punc", false, Some(PrimaryPos::Punc);
    Ques = "Ques", false, Some(PrimaryPos::Q);
    Postp = "Postp", false, Some(PrimaryPos::Postp);
    Det = "Det", false, Some(PrimaryPos::Det);
    Num = "Num", false, Some(PrimaryPos::Num);
    Dup = "Dup", false, Some(PrimaryPos::Dup);
    Interj = "Interj", false, Some(PrimaryPos::Interj);
    A1sg = "A1sg", false, None;
    A2sg = "A2sg", false, None;
    A3sg = "A3sg", false, None;
    A1pl = "A1pl", false, None;
    A2pl = "A2pl", false, None;
    A3pl = "A3pl", false, None;
    Pnon = "Pnon", false, None;
    P1sg = "P1sg", false, None;
    P2sg = "P2sg", false, None;
    P3sg = "P3sg", false, None;
    P1pl = "P1pl", false, None;
    P2pl = "P2pl", false, None;
    P3pl = "P3pl", false, None;
    Nom = "Nom", false, None;
    Dat = "Dat", false, None;
    Acc = "Acc", false, None;
    Abl = "Abl", false, None;
    Loc = "Loc", false, None;
    Ins = "Ins", false, None;
    Gen = "Gen", false, None;
    Equ = "Equ", false, None;
    Dim = "Dim", true, None;
    Ness = "Ness", true, None;
    With = "With", true, None;
    Without = "Without", true, None;
    Related = "Related", true, None;
    JustLike = "JustLike", true, None;
    Rel = "Rel", true, None;
    Agt = "Agt", true, None;
    Become = "Become", true, None;
    Acquire = "Acquire", true, None;
    Ly = "Ly", true, None;
    Caus = "Caus", true, None;
    Recip = "Recip", true, None;
    Reflex = "Reflex", true, None;
    Able = "Able", true, None;
    Pass = "Pass", true, None;
    Inf1 = "Inf1", true, None;
    Inf2 = "Inf2", true, None;
    Inf3 = "Inf3", true, None;
    ActOf = "ActOf", true, None;
    PastPart = "PastPart", true, None;
    NarrPart = "NarrPart", true, None;
    FutPart = "FutPart", true, None;
    PresPart = "PresPart", true, None;
    AorPart = "AorPart", true, None;
    NotState = "NotState", true, None;
    FeelLike = "FeelLike", true, None;
    EverSince = "EverSince", true, None;
    Repeat = "Repeat", true, None;
    Almost = "Almost", true, None;
    Hastily = "Hastily", true, None;
    Stay = "Stay", true, None;
    Start = "Start", true, None;
    AsIf = "AsIf", true, None;
    While = "While", true, None;
    When = "When", true, None;
    SinceDoingSo = "SinceDoingSo", true, None;
    AsLongAs = "AsLongAs", true, None;
    ByDoingSo = "ByDoingSo", true, None;
    Adamantly = "Adamantly", true, None;
    AfterDoingSo = "AfterDoingSo", true, None;
    WithoutHavingDoneSo = "WithoutHavingDoneSo", true, None;
    WithoutBeingAbleToHaveDoneSo = "WithoutBeingAbleToHaveDoneSo", true, None;
    Zero = "Zero", true, None;
    Cop = "Cop", false, None;
    Neg = "Neg", false, None;
    Unable = "Unable", false, None;
    Pres = "Pres", false, None;
    Past = "Past", false, None;
    Narr = "Narr", false, None;
    Cond = "Cond", false, None;
    Prog1 = "Prog1", false, None;
    Prog2 = "Prog2", false, None;
    Aor = "Aor", false, None;
    Fut = "Fut", false, None;
    Imp = "Imp", false, None;
    Opt = "Opt", false, None;
    Desr = "Desr", false, None;
    Neces = "Neces", false, None;
}

/// A set of morphemes (there are fewer than 128).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct MorphemeSet(u128);

impl MorphemeSet {
    pub fn of(morphemes: &[Morpheme]) -> Self {
        Self(morphemes.iter().fold(0, |s, &m| s | 1u128 << m as u8))
    }
    #[inline]
    pub fn contains(self, m: Morpheme) -> bool {
        self.0 & 1u128 << m as u8 != 0
    }
    #[inline]
    pub fn insert(&mut self, m: Morpheme) {
        self.0 |= 1u128 << m as u8;
    }
    #[inline]
    pub fn intersects(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StateId(pub u16);

/// A set of states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct StateSet([u64; 6]);

impl StateSet {
    pub fn of(states: &[StateId]) -> Self {
        let mut s = Self::default();
        for &st in states {
            s.insert(st);
        }
        s
    }
    #[inline]
    pub fn insert(&mut self, s: StateId) {
        self.0[(s.0 / 64) as usize] |= 1 << (s.0 % 64);
    }
    #[inline]
    pub fn contains(&self, s: StateId) -> bool {
        self.0[(s.0 / 64) as usize] & 1 << (s.0 % 64) != 0
    }
    #[inline]
    pub fn intersects(&self, other: &Self) -> bool {
        self.0.iter().zip(other.0.iter()).any(|(a, b)| a & b != 0)
    }
}

/// A state of the graph. A terminal state may end a word; a derivative state opens a new
/// inflectional group; a POS-root state starts one (noun_S, verbRoot_S).
#[derive(Debug, Clone)]
pub struct State {
    pub name: &'static str,
    pub morpheme: Morpheme,
    pub terminal: bool,
    pub derivative: bool,
    pub pos_root: bool,
}

/// One token of a suffix template.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Token {
    Letter(char),
    /// A: a or e by the last vowel.
    AVowel,
    /// I: ı, i, u or ü by the last vowel.
    IVowel,
    /// +x: x only after a vowel (the buffer consonants y, s, n).
    Append(char),
    /// >x: x devoiced after a voiceless consonant (>dA: da/ta).
    DevoiceFirst(char),
    /// ~k: a final k; the next suffix must start with a consonant.
    LastVoiced(char),
    /// !ğ: a final ğ; the next suffix must start with a vowel.
    LastNotVoiced(char),
}

fn tokenize(template: &str) -> Vec<Token> {
    let mut out = Vec::new();
    let mut it = template.chars().peekable();
    while let Some(c) = it.next() {
        out.push(match c {
            '+' => match it.next() {
                Some('I') => Token::IVowel,
                Some('A') => Token::AVowel,
                Some(n) => Token::Append(n),
                None => break,
            },
            '>' => Token::DevoiceFirst(it.next().unwrap_or('d')),
            '~' => Token::LastVoiced(it.next().unwrap_or('k')),
            '!' => Token::LastNotVoiced(it.next().unwrap_or('ğ')),
            'I' => Token::IVowel,
            'A' => Token::AVowel,
            c => Token::Letter(c),
        });
    }
    out
}

/// The phonetic attributes a template's surface depends on, packed into a cache index.
const SURFACE_KEYS: usize = 64;

fn surface_key(a: PhoneticAttrs) -> usize {
    let bits = [
        PhoneticAttrs::LAST_LETTER_VOWEL,
        PhoneticAttrs::LAST_VOWEL_FRONTAL,
        PhoneticAttrs::LAST_VOWEL_BACK,
        PhoneticAttrs::LAST_VOWEL_ROUNDED,
        PhoneticAttrs::LAST_VOWEL_UNROUNDED,
        PhoneticAttrs::LAST_LETTER_VOICELESS,
    ];
    bits.iter()
        .enumerate()
        .fold(0, |k, (i, &b)| if a.contains(b) { k | 1 << i } else { k })
}

fn attrs_of_key(key: usize) -> PhoneticAttrs {
    let bits = [
        PhoneticAttrs::LAST_LETTER_VOWEL,
        PhoneticAttrs::LAST_VOWEL_FRONTAL,
        PhoneticAttrs::LAST_VOWEL_BACK,
        PhoneticAttrs::LAST_VOWEL_ROUNDED,
        PhoneticAttrs::LAST_VOWEL_UNROUNDED,
        PhoneticAttrs::LAST_LETTER_VOICELESS,
    ];
    bits.iter()
        .enumerate()
        .fold(PhoneticAttrs::empty(), |a, (i, &b)| {
            if key & 1 << i != 0 {
                a | b
            } else {
                a
            }
        })
}

/// Zemberek's `SurfaceTransition.generateSurface`: the template's surface after letters with
/// the attributes `path`, or `None` when a harmonising vowel has nothing to agree with.
fn generate_surface(tokens: &[Token], path: PhoneticAttrs) -> Option<String> {
    let mut sb = String::new();
    for (index, &token) in tokens.iter().enumerate() {
        let attrs = attributes_of(&sb, path);
        match token {
            Token::Letter(c) | Token::LastVoiced(c) | Token::LastNotVoiced(c) => sb.push(c),
            Token::AVowel => {
                if index == 0 && path.contains(PhoneticAttrs::LAST_LETTER_VOWEL) {
                    continue;
                }
                if attrs.contains(PhoneticAttrs::LAST_VOWEL_BACK) {
                    sb.push('a');
                } else if attrs.contains(PhoneticAttrs::LAST_VOWEL_FRONTAL) {
                    sb.push('e');
                } else {
                    return None;
                }
            }
            Token::IVowel => {
                if index == 0 && path.contains(PhoneticAttrs::LAST_LETTER_VOWEL) {
                    continue;
                }
                let front = attrs.contains(PhoneticAttrs::LAST_VOWEL_FRONTAL);
                let back = attrs.contains(PhoneticAttrs::LAST_VOWEL_BACK);
                let round = attrs.contains(PhoneticAttrs::LAST_VOWEL_ROUNDED);
                let unround = attrs.contains(PhoneticAttrs::LAST_VOWEL_UNROUNDED);
                sb.push(match (front, back, round, unround) {
                    (true, _, _, true) => 'i',
                    (_, true, _, true) => 'ı',
                    (_, true, true, _) => 'u',
                    (true, _, true, _) => 'ü',
                    _ => return None,
                });
            }
            Token::Append(c) => {
                if attrs.contains(PhoneticAttrs::LAST_LETTER_VOWEL) {
                    sb.push(c);
                }
            }
            Token::DevoiceFirst(c) => {
                sb.push(if attrs.contains(PhoneticAttrs::LAST_LETTER_VOICELESS) {
                    devoice(c)
                } else {
                    c
                })
            }
        }
    }
    Some(sb)
}

/// A dictionary item a condition names: lemma and POS, and the secondary POS if given.
#[derive(Debug, Clone)]
pub struct ItemRef {
    pub lemma: &'static str,
    pub pos: PrimaryPos,
    pub secondary: Option<SecondaryPos>,
}

impl ItemRef {
    pub fn matches(&self, item: &DictionaryItem) -> bool {
        item.lemma == self.lemma
            && item.primary_pos == self.pos
            && self.secondary.is_none_or(|s| s == item.secondary_pos)
    }
}

/// A condition on taking a transition (Zemberek's `Conditions`).
#[derive(Debug, Clone)]
pub enum Cond {
    True,
    Not(Box<Cond>),
    And(Vec<Cond>),
    Or(Vec<Cond>),
    /// The root carries a root attribute.
    HasRootAttr(RootAttr),
    /// The letters so far have a phonetic attribute.
    HasPhonetic(PhoneticAttrs),
    RootIsAny(Vec<ItemRef>),
    SecondaryPosIs(SecondaryPos),
    /// The stem's surface is one of these (de/ye vs di/yi).
    RootSurfaceIsAny(Vec<&'static str>),
    PrevMorphemeIsAny(MorphemeSet),
    PrevStateIsAny(StateSet),
    /// True when there is no previous state.
    PrevStateIsNot(StateId),
    LastDerivationIsAny(StateSet),
    CurrentGroupContainsAny(StateSet),
    PreviousGroupContainsAny(StateSet),
    PreviousGroupContainsMorpheme(MorphemeSet),
    NoSurfaceAfterDerivation,
    ContainsMorpheme(MorphemeSet),
    /// A registered morpheme sequence occurs anywhere in the path (index into
    /// [`Graph::sequences`]).
    ContainsSequence(usize),
    /// The path ends with a registered sequence.
    TailSequence(usize),
    HasTail,
    HasAnySuffixSurface,
}

impl Cond {
    pub fn and(self, other: Cond) -> Cond {
        match self {
            Cond::True => other,
            Cond::And(mut v) => {
                v.push(other);
                Cond::And(v)
            }
            c => Cond::And(vec![c, other]),
        }
    }
    pub fn or(self, other: Cond) -> Cond {
        match self {
            Cond::Or(mut v) => {
                v.push(other);
                Cond::Or(v)
            }
            c => Cond::Or(vec![c, other]),
        }
    }
    pub fn and_not(self, other: Cond) -> Cond {
        self.and(other.not())
    }
    #[allow(clippy::should_implement_trait)]
    pub fn not(self) -> Cond {
        Cond::Not(Box::new(self))
    }
}

/// Number of morphemes of the path kept for sequence conditions (the longest is 7).
pub const RECENT: usize = 7;

/// What a condition may know about the path so far. It is updated incrementally as the search
/// takes each transition; two paths with equal contexts at the same input position and state
/// have the same continuations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Context {
    pub current: StateId,
    pub previous: Option<StateId>,
    pub last_derivation: Option<StateId>,
    /// States since (and including) the last derivative state, the root excluded.
    pub current_group: StateSet,
    /// The group before that.
    pub previous_group: StateSet,
    /// The morphemes of the states in `previous_group`.
    pub previous_group_morphemes: MorphemeSet,
    /// The morphemes of the states in `current_group`.
    pub current_group_morphemes: MorphemeSet,
    /// Every morpheme so far, the root's included.
    pub morphemes: MorphemeSet,
    /// The last morphemes, most recent last (`Root` pads the start).
    pub recent: [Morpheme; RECENT],
    /// Registered sequences seen so far (bit per [`Graph::sequences`] entry).
    pub sequences_seen: u32,
    pub surface_since_derivation: bool,
    pub any_surface: bool,
}

impl Context {
    pub fn at_root(root: StateId, morpheme: Morpheme) -> Self {
        let mut recent = [Morpheme::Root; RECENT];
        recent[RECENT - 1] = morpheme;
        Context {
            current: root,
            previous: None,
            last_derivation: None,
            current_group: StateSet::default(),
            previous_group: StateSet::default(),
            previous_group_morphemes: MorphemeSet::default(),
            current_group_morphemes: MorphemeSet::default(),
            morphemes: MorphemeSet::of(&[morpheme]),
            recent,
            sequences_seen: 0,
            surface_since_derivation: false,
            any_surface: false,
        }
    }

    /// The context after taking a transition to `to` with a (possibly empty) surface.
    pub fn step(&self, graph: &Graph, to: StateId, has_surface: bool) -> Self {
        let state = &graph.states[to.0 as usize];
        let mut c = *self;
        c.previous = Some(self.current);
        c.current = to;
        c.morphemes.insert(state.morpheme);
        c.recent.copy_within(1.., 0);
        c.recent[RECENT - 1] = state.morpheme;
        if state.derivative {
            c.previous_group = self.current_group;
            c.previous_group_morphemes = self.current_group_morphemes;
            c.current_group = StateSet::of(&[to]);
            c.current_group_morphemes = MorphemeSet::of(&[state.morpheme]);
            c.last_derivation = Some(to);
            c.surface_since_derivation = false;
        } else {
            c.current_group.insert(to);
            c.current_group_morphemes.insert(state.morpheme);
            c.surface_since_derivation |= has_surface;
        }
        c.any_surface |= has_surface;
        for (i, seq) in graph.sequences.iter().enumerate() {
            if c.recent.ends_with(seq) {
                c.sequences_seen |= 1 << i;
            }
        }
        c
    }
}

/// What a condition is evaluated against: the context plus the root and the letters left.
pub struct PathView<'a> {
    pub ctx: &'a Context,
    pub item: &'a DictionaryItem,
    pub stem_surface: &'a str,
    pub attrs: PhoneticAttrs,
    pub tail_empty: bool,
}

impl Cond {
    pub fn eval(&self, graph: &Graph, p: &PathView) -> bool {
        match self {
            Cond::True => true,
            Cond::Not(c) => !c.eval(graph, p),
            Cond::And(v) => v.iter().all(|c| c.eval(graph, p)),
            Cond::Or(v) => v.iter().any(|c| c.eval(graph, p)),
            Cond::HasRootAttr(a) => p.item.attributes.contains(*a),
            Cond::HasPhonetic(a) => p.attrs.contains(*a),
            Cond::RootIsAny(items) => items.iter().any(|r| r.matches(p.item)),
            Cond::SecondaryPosIs(s) => p.item.secondary_pos == *s,
            Cond::RootSurfaceIsAny(s) => s.contains(&p.stem_surface),
            Cond::PrevMorphemeIsAny(set) => p
                .ctx
                .previous
                .is_some_and(|s| set.contains(graph.states[s.0 as usize].morpheme)),
            Cond::PrevStateIsAny(set) => p.ctx.previous.is_some_and(|s| set.contains(s)),
            Cond::PrevStateIsNot(s) => p.ctx.previous != Some(*s),
            Cond::LastDerivationIsAny(set) => {
                p.ctx.last_derivation.is_some_and(|s| set.contains(s))
            }
            Cond::CurrentGroupContainsAny(set) => p.ctx.current_group.intersects(set),
            Cond::PreviousGroupContainsAny(set) => p.ctx.previous_group.intersects(set),
            Cond::PreviousGroupContainsMorpheme(set) => {
                p.ctx.previous_group_morphemes.intersects(*set)
            }
            Cond::NoSurfaceAfterDerivation => !p.ctx.surface_since_derivation,
            Cond::ContainsMorpheme(set) => p.ctx.morphemes.intersects(*set),
            Cond::ContainsSequence(i) => p.ctx.sequences_seen & 1 << i != 0,
            Cond::TailSequence(i) => p.ctx.recent.ends_with(&graph.sequences[*i]),
            Cond::HasTail => !p.tail_empty,
            Cond::HasAnySuffixSurface => p.ctx.any_surface,
        }
    }
}

/// A suffix transition with its surfaces precomputed for every phonetic context.
#[derive(Debug, Clone)]
pub struct Transition {
    pub from: StateId,
    pub to: StateId,
    pub template: &'static str,
    pub condition: Cond,
    tokens: Vec<Token>,
    surfaces: Vec<Option<Box<str>>>,
}

impl Transition {
    /// Whether the template is non-empty. A template like `I` still counts when it produces
    /// nothing after a vowel (oku-r is Aor `Ir`).
    pub fn has_surface(&self) -> bool {
        !self.tokens.is_empty()
    }

    /// The surface after letters with these attributes.
    #[inline]
    pub fn surface(&self, attrs: PhoneticAttrs) -> Option<&str> {
        self.surfaces[surface_key(attrs)].as_deref()
    }

    /// Whether the template ends in `~k` (the next suffix starts with a consonant) or `!ğ`
    /// (with a vowel, and the word cannot end here).
    pub fn ends_voiced(&self) -> Option<bool> {
        match self.tokens.last() {
            Some(Token::LastVoiced(_)) => Some(false),
            Some(Token::LastNotVoiced(_)) => Some(true),
            _ => None,
        }
    }
}

/// The automatic condition of a template on the stem's expectations (Zemberek's
/// `SuffixTransition.conditionsFromTemplate`): a vowel-initial suffix may not follow a stem
/// that expects a consonant (kitap-ı), and a consonant-initial one not a stem that expects a
/// vowel (kitab-da).
fn template_condition(template: &str) -> Option<Cond> {
    let lower: Vec<char> = template
        .chars()
        .map(|c| match c {
            'I' => 'ı',
            'A' => 'a',
            c => c,
        })
        .collect();
    let first = *lower.first()?;
    let vowel = |c: char| super::phonetics::is_vowel(c);
    let mut c = None;
    if first == '>' || !vowel(first) {
        c = Some(Cond::HasPhonetic(PhoneticAttrs::EXPECTS_VOWEL).not());
    }
    if (first == '+' && lower.get(2).copied().is_some_and(vowel)) || vowel(first) {
        c = Some(Cond::HasPhonetic(PhoneticAttrs::EXPECTS_CONSONANT).not());
    }
    c
}

/// The graph: states, transitions indexed by their source state, and the morpheme sequences
/// conditions look for.
pub struct Graph {
    pub states: Vec<State>,
    pub transitions: Vec<Transition>,
    outgoing: Vec<Vec<u32>>,
    pub sequences: Vec<Vec<Morpheme>>,
}

impl Graph {
    pub fn new() -> Self {
        Graph {
            states: Vec::new(),
            transitions: Vec::new(),
            outgoing: Vec::new(),
            sequences: Vec::new(),
        }
    }

    pub fn state(
        &mut self,
        name: &'static str,
        morpheme: Morpheme,
        terminal: bool,
        derivative: bool,
        pos_root: bool,
    ) -> StateId {
        let id = StateId(self.states.len() as u16);
        assert!((id.0 as usize) < 64 * 6, "more states than StateSet holds");
        self.states.push(State {
            name,
            morpheme,
            terminal,
            derivative,
            pos_root,
        });
        self.outgoing.push(Vec::new());
        id
    }

    /// Adds a transition; the template's own vowel/consonant condition is added to `condition`.
    pub fn add(&mut self, from: StateId, to: StateId, template: &'static str, condition: Cond) {
        let condition = match template_condition(template) {
            Some(auto) => auto.and(condition),
            None => condition,
        };
        let tokens = tokenize(template);
        let surfaces = (0..SURFACE_KEYS)
            .map(|k| generate_surface(&tokens, attrs_of_key(k)).map(String::into_boxed_str))
            .collect();
        self.outgoing[from.0 as usize].push(self.transitions.len() as u32);
        self.transitions.push(Transition {
            from,
            to,
            template,
            condition,
            tokens,
            surfaces,
        });
    }

    /// Copies the transitions leaving `source` so that they also leave `target`.
    pub fn copy_outgoing(&mut self, target: StateId, source: StateId) {
        let ids = self.outgoing[source.0 as usize].clone();
        for id in ids {
            let t = self.transitions[id as usize].clone();
            self.outgoing[target.0 as usize].push(self.transitions.len() as u32);
            self.transitions.push(Transition { from: target, ..t });
        }
    }

    /// Removes the transitions from `state` into states of morpheme `m`.
    pub fn remove_outgoing_to(&mut self, state: StateId, m: Morpheme) {
        let transitions = &self.transitions;
        let states = &self.states;
        self.outgoing[state.0 as usize]
            .retain(|&id| states[transitions[id as usize].to.0 as usize].morpheme != m);
    }

    /// Registers a morpheme sequence for `ContainsSequence` / `TailSequence` conditions.
    pub fn sequence(&mut self, seq: &[Morpheme]) -> usize {
        assert!(seq.len() <= RECENT);
        if let Some(i) = self.sequences.iter().position(|s| s == seq) {
            return i;
        }
        self.sequences.push(seq.to_vec());
        assert!(self.sequences.len() <= 32);
        self.sequences.len() - 1
    }

    /// The transitions leaving a state, with their indices into [`Graph::transitions`].
    #[inline]
    pub fn outgoing(&self, state: StateId) -> impl Iterator<Item = (u32, &Transition)> {
        self.outgoing[state.0 as usize]
            .iter()
            .map(move |&id| (id, &self.transitions[id as usize]))
    }

    pub fn state_by_name(&self, name: &str) -> Option<StateId> {
        self.states
            .iter()
            .position(|s| s.name == name)
            .map(|i| StateId(i as u16))
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn surface(template: &str, before: &str) -> Option<String> {
        generate_surface(
            &tokenize(template),
            attributes_of(before, PhoneticAttrs::empty()),
        )
    }

    #[test]
    fn templates_follow_the_stem() {
        assert_eq!(surface(">dA", "kitap").as_deref(), Some("ta"));
        assert_eq!(surface(">dA", "ev").as_deref(), Some("de"));
        assert_eq!(surface("+yA", "araba").as_deref(), Some("ya"));
        assert_eq!(surface("+yA", "ev").as_deref(), Some("e"));
        assert_eq!(surface("+sI", "araba").as_deref(), Some("sı"));
        assert_eq!(surface("+sI", "göz").as_deref(), Some("ü"));
        assert_eq!(surface("Ir", "oku").as_deref(), Some("r"));
        assert_eq!(surface("Ir", "gel").as_deref(), Some("ir"));
        assert_eq!(surface("+nIn", "araba").as_deref(), Some("nın"));
        assert_eq!(surface("lI~k", "göz").as_deref(), Some("lük"));
        assert_eq!(surface("+yAcA!ğ", "gel").as_deref(), Some("eceğ"));
        // A vowel-less sequence keeps the vowel attributes before it.
        let after_m = attributes_of("m", attributes_of("gel", PhoneticAttrs::empty()));
        assert_eq!(
            generate_surface(&tokenize("Iyor"), after_m).as_deref(),
            Some("iyor")
        );
    }
}
