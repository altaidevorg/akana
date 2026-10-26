//! Syntactic analysis: the morphological analyzer's readings split into inflectional groups
//! (IGs) with Universal Dependencies features.
//!
//! The groups come from the morphotactic path of each reading, as in Zemberek: a derivative
//! state (PastPart, Caus, Ness, ...) opens a group, and the POS-root state that follows it gives
//! the group's POS. Zero derivations are eliminated as in Öztürel et al. (FSMNLP 2019): a zero
//! derivation to a noun (güzel-ler, ev-de-ki-ler) re-categorises the group it follows, and only
//! the copula (zero derivation to a nominal predicate: evde-yim, güzel-di) opens a group, which
//! UD treebanks split off as the auxiliary.

use super::types::{InflectionalGroup, SyntacticParse};
use crate::morphology::analyzer::ENGINE;
use crate::morphology::morphotactics::Morpheme as M;
use crate::morphology::{MorphParse, PrimaryPos, SecondaryPos, TurkishMorphology};
use std::collections::BTreeMap;

/// The syntactic analyzer. It shares the morphological analyzer's lexicon and ranking, so its
/// parses come most likely first.
#[derive(Clone, Default)]
pub struct TurkishSyntacticMorphology {
    morphology: TurkishMorphology,
}

impl TurkishSyntacticMorphology {
    pub fn new() -> Self {
        Self {
            morphology: TurkishMorphology::new(),
        }
    }

    /// An analyzer over a morphological analyzer of one's own (a custom lexicon).
    pub fn with_morphology(morphology: TurkishMorphology) -> Self {
        Self { morphology }
    }

    /// The parses of a word, most likely first, each split into inflectional groups.
    pub fn analyze(&self, word: &str) -> Vec<SyntacticParse> {
        let mut parses: Vec<SyntacticParse> = Vec::new();
        for reading in self.morphology.analyze(word) {
            // Readings that differ only in what zero derivations hide come out equal (güzel-ler
            // from the adjective and from the noun).
            let parse = Self::from_parse(word, &reading);
            if !parses.iter().any(|p| p.formatted == parse.formatted) {
                parses.push(parse);
            }
        }
        parses
    }

    /// The inflectional groups of one morphological reading (a disambiguated one, say). A
    /// reading not made by the analyzer (deserialized, or the disambiguator's unknown word) has no
    /// path and gives one group with its POS.
    pub fn from_parse(surface: &str, parse: &MorphParse) -> SyntacticParse {
        let groups = groups_of(parse);
        let root_pos = groups.first().map_or("UNK", |g| g.pos.as_str()).to_string();
        SyntacticParse::new(
            surface,
            &parse.root,
            root_pos,
            groups,
            parse.secondary_pos == SecondaryPos::ProperNoun,
        )
    }
}

/// One group while the path is read.
struct Group {
    /// The POS of the group's last POS-root state.
    pos: PrimaryPos,
    /// The morpheme that opened the group (`Zero` for the copula); `None` for the root's group.
    derivation: Option<M>,
    /// The POS of the group it was derived from.
    source: Option<PrimaryPos>,
    /// A nominal predicate (the copula, değil, the question particle, imek): no default polarity.
    predicate: bool,
    features: BTreeMap<String, String>,
    /// Mood and Voice values, which UD Turkish concatenates when a word has several (GenPot,
    /// CauPass).
    moods: Vec<&'static str>,
    voices: Vec<&'static str>,
    /// The last tense or mood suffix of the group: a past or evidential after it is the copula.
    tense: Option<M>,
}

impl Group {
    fn new(pos: PrimaryPos, derivation: Option<M>, source: Option<PrimaryPos>) -> Self {
        Group {
            pos,
            derivation,
            source,
            predicate: false,
            features: BTreeMap::new(),
            moods: Vec::new(),
            voices: Vec::new(),
            tense: None,
        }
    }

    fn set(&mut self, key: &str, value: &str) {
        self.features.insert(key.to_string(), value.to_string());
    }

    /// The UD features of an inflectional morpheme.
    fn inflect(&mut self, m: M) {
        let person = |m: M| match m {
            M::A1sg | M::A1pl | M::P1sg | M::P1pl => "1",
            M::A2sg | M::A2pl | M::P2sg | M::P2pl => "2",
            _ => "3",
        };
        let number = |m: M| match m {
            M::A1sg | M::A2sg | M::A3sg | M::P1sg | M::P2sg | M::P3sg => "Sing",
            _ => "Plur",
        };
        match m {
            M::A1sg | M::A2sg | M::A3sg | M::A1pl | M::A2pl | M::A3pl => {
                self.set("Number", number(m));
                self.set("Person", person(m));
            }
            M::P1sg | M::P2sg | M::P3sg | M::P1pl | M::P2pl | M::P3pl => {
                self.set("Number[psor]", number(m));
                self.set("Person[psor]", person(m));
            }
            M::Nom | M::Dat | M::Acc | M::Abl | M::Loc | M::Ins | M::Gen | M::Equ => {
                self.set("Case", m.id());
            }
            M::Neg => self.set("Polarity", "Neg"),
            M::Unable => {
                self.set("Polarity", "Neg");
                self.moods.push("Pot");
            }
            M::Pres => self.set("Tense", "Pres"),
            M::Past | M::Narr => {
                match self.tense {
                    // A second past is the pluperfect (gelmişti, gittiydi).
                    Some(M::Past | M::Narr) => self.set("Tense", "Pqp"),
                    // The future in the past is prospective (gelecekti).
                    Some(M::Fut) => {
                        self.set("Tense", "Past");
                        self.set("Aspect", "Prosp");
                    }
                    // After another tense or mood the aspect stays (geliyordu, gelirdi).
                    Some(_) => self.set("Tense", "Past"),
                    None => {
                        self.set("Tense", "Past");
                        self.set("Aspect", "Perf");
                    }
                }
                self.set("Evident", if m == M::Past { "Fh" } else { "Nfh" });
            }
            M::Prog1 | M::Prog2 => {
                self.set("Tense", "Pres");
                self.set("Aspect", "Prog");
            }
            M::Aor => {
                self.set("Tense", "Pres");
                self.set("Aspect", "Hab");
                self.moods.push("Gen");
            }
            M::Fut => {
                self.set("Tense", "Fut");
                self.set("Aspect", "Perf");
            }
            M::Neces => {
                self.set("Tense", "Pres");
                self.moods.push("Nec");
            }
            M::Imp => self.moods.push("Imp"),
            M::Opt => self.moods.push("Opt"),
            M::Desr => self.moods.push("Des"),
            M::Cond => self.moods.push("Cnd"),
            M::Cop => self.moods.push("Gen"),
            _ => {}
        }
        if matches!(
            m,
            M::Prog1 | M::Prog2 | M::Aor | M::Fut | M::Past | M::Narr | M::Opt | M::Desr | M::Neces
        ) {
            self.tense = Some(m);
        }
    }

    fn finish(mut self) -> InflectionalGroup {
        let derived = self.derivation.is_some();
        let from_verb = self.source == Some(PrimaryPos::Verb);
        let pos = match self.pos {
            PrimaryPos::Noun if derived && from_verb => "NOMP",
            PrimaryPos::Noun => "NN",
            PrimaryPos::Adj if derived => "ADJP",
            PrimaryPos::Adj => "ADJ",
            PrimaryPos::Verb => "VB",
            PrimaryPos::Adv => "ADV",
            PrimaryPos::Pron => "PRON",
            PrimaryPos::Num => "NUM",
            PrimaryPos::Det => "DET",
            PrimaryPos::Conj => "CONJ",
            PrimaryPos::Postp => "POSTP",
            PrimaryPos::Interj => "INTERJ",
            PrimaryPos::Q => "QUES",
            PrimaryPos::Dup => "DUP",
            PrimaryPos::Punc => "PUNC",
            PrimaryPos::Unknown => "UNK",
        };
        if !self.voices.is_empty() {
            self.voices.sort_unstable();
            self.set("Voice", &self.voices.concat());
        }
        let verbal = self.pos == PrimaryPos::Verb || self.pos == PrimaryPos::Q;
        if verbal {
            // Every finite verb ends in a person (Zemberek's paths carry the silent 3rd person);
            // a tense or mood before a converb or participle is not finite (gel-ir-ken,
            // gel-ebil-ir "able to come").
            let finite = self.features.contains_key("Person");
            if finite {
                if self.moods.is_empty() {
                    self.moods.push("Ind");
                }
                self.set("VerbForm", "Fin");
            }
            if self.pos == PrimaryPos::Verb && !self.predicate {
                self.features
                    .entry("Polarity".to_string())
                    .or_insert_with(|| "Pos".to_string());
            }
        }
        if !self.moods.is_empty() {
            self.moods.sort_unstable();
            self.set("Mood", &self.moods.concat());
        }
        if let Some(form) = self.derivation.filter(|_| from_verb).and_then(|m| match m {
            M::Agt => None,
            M::PresPart | M::NarrPart | M::AorPart => Some("Part"),
            _ => match self.pos {
                PrimaryPos::Noun => Some("Vnoun"),
                PrimaryPos::Adj => Some("Part"),
                PrimaryPos::Adv => Some("Conv"),
                _ => None,
            },
        }) {
            self.set("VerbForm", form);
        }
        let derivation = self.derivation.map(|m| match m {
            M::Zero => "Cop",
            // The -DIk and -(y)AcAk nominalisations (gel-diğ-imiz-de), as Öztürel et al. name
            // them; as adjectives they are participles (gel-diğ-imiz gün).
            M::PastPart if self.pos == PrimaryPos::Noun => "PastNom",
            M::FutPart if self.pos == PrimaryPos::Noun => "FutNom",
            m => m.id(),
        });
        InflectionalGroup {
            pos: pos.to_string(),
            derivation: derivation.map(str::to_string),
            features: self.features,
        }
    }
}

/// The inflectional groups of a reading, read off its morphotactic path.
fn groups_of(parse: &MorphParse) -> Vec<InflectionalGroup> {
    let engine = &*ENGINE;
    let graph = &engine.graph;
    let path = &parse.path;
    let Some(root) = path.root else {
        return vec![Group::new(parse.primary_pos, None, None).finish()];
    };
    let root_state = &graph.states[root.0 as usize];
    let mut current = Group::new(
        root_state.morpheme.pos().unwrap_or(parse.primary_pos),
        None,
        None,
    );
    current.predicate = engine.roots.nominal_verb.contains(root);
    let mut groups = Vec::new();
    // A zero derivation waiting for the POS root it leads to.
    let mut zero = false;
    for step in &path.steps {
        let state = &graph.states[step.state.0 as usize];
        let m = state.morpheme;
        // A postposition read as a noun (sonra-sı) reaches its noun state without a POS root.
        if state.pos_root || (zero && m.pos().is_some()) {
            let Some(pos) = m.pos() else { continue };
            if zero && engine.roots.nominal_verb.contains(step.state) {
                // The copula: a group of its own.
                let source = current.pos;
                groups.push(std::mem::replace(
                    &mut current,
                    Group::new(pos, Some(M::Zero), Some(source)),
                ));
                current.predicate = true;
            } else if !(zero && current.pos == PrimaryPos::Num) {
                // Zero derivation eliminated: the group takes the POS it inflects as (güzel-ler
                // is a noun); a numeral stays one (1923'te, iki-si).
                current.pos = pos;
            }
            zero = false;
            continue;
        }
        if state.derivative {
            if m == M::Zero {
                zero = true;
            } else if !step.overt && matches!(m, M::Recip | M::Reflex) {
                // A lexically reciprocal or reflexive verb (dövüş, giyin): a voice, no group.
                current
                    .voices
                    .push(if m == M::Recip { "Rcp" } else { "Rfl" });
            } else {
                let source = current.pos;
                groups.push(std::mem::replace(
                    &mut current,
                    Group::new(source, Some(m), Some(source)),
                ));
                match m {
                    M::Caus => current.voices.push("Cau"),
                    M::Pass => current.voices.push("Pass"),
                    M::Recip => current.voices.push("Rcp"),
                    M::Reflex => current.voices.push("Rfl"),
                    M::Able => current.moods.push("Pot"),
                    _ => {}
                }
            }
            continue;
        }
        current.inflect(m);
    }
    groups.push(current);
    groups.into_iter().map(Group::finish).collect()
}
