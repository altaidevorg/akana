//! Rule identity and selection.
//!
//! Every check the grammar checker runs is a [`GrammarRule`] with a stable string id
//! (`de-da-after-verb`, `mi-merged`, ...). A [`RuleSet`] says which rules a check runs.
//!
//! The default set holds the rules that leave correct, edited Turkish alone: each raises a
//! false alarm on at most 1 in 1,000 clean sentences (7,500 court sentences, 4,452 Wikipedia
//! sentences), and where GECTurk has its error type at least 80 % of its GECTurk findings are
//! on a labelled error. Where GECTurk has no such error, or does not label the errors a rule
//! finds, every finding of the rule in all the measured text was read and judged instead. The
//! other rules are experimental: off unless asked for, by id or all together with
//! [`RuleSet::with_experimental`]. The README's grammar section has the measurements, and
//! `tests/grammar_precision_gate.rs` keeps the default set quiet on the committed clean corpus.

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;
use std::str::FromStr;

use super::ErrorCategory;

macro_rules! grammar_rules {
    ($(
        $variant:ident = $id:literal, $category:ident, $default:literal, $confidence:literal,
            $summary:literal;
    )*) => {
        /// One check of the grammar checker. The id ([`GrammarRule::id`]) is stable and is what
        /// the Python binding, the CLI and serialized findings use.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub enum GrammarRule {
            $( #[doc = $summary] $variant, )*
        }

        impl GrammarRule {
            /// Every rule, in a fixed order.
            pub const ALL: &'static [GrammarRule] = &[ $( GrammarRule::$variant, )* ];

            /// The rule's stable id.
            pub fn id(self) -> &'static str {
                match self { $( GrammarRule::$variant => $id, )* }
            }

            /// The error category its findings report.
            pub fn category(self) -> ErrorCategory {
                match self { $( GrammarRule::$variant => ErrorCategory::$category, )* }
            }

            /// Whether the rule is in the default set; the others are experimental.
            pub fn is_default(self) -> bool {
                match self { $( GrammarRule::$variant => $default, )* }
            }

            /// The rule's measured precision, (right + 1) / (findings + 2) over its findings
            /// on GECTurk (test and movie reviews; by GECTurk's labels where it has the error
            /// type, judged by hand otherwise, with the clean-text findings). 0.5 for a rule
            /// that never fired in the measured text. Every finding of the rule carries it.
            pub fn confidence(self) -> f32 {
                match self { $( GrammarRule::$variant => $confidence, )* }
            }

            /// What the rule looks for, in one line.
            pub fn summary(self) -> &'static str {
                match self { $( GrammarRule::$variant => $summary, )* }
            }
        }
    };
}

grammar_rules! {
    Capitalization = "capitalization", Capitalization, true, 0.90,
        "The text starts with a lower-case letter.";
    MiMerged = "mi-merged", ParticleMi, true, 0.99,
        "The question particle written attached: geldinmi.";
    MiMergedPerson = "mi-merged-person", ParticleMi, true, 0.95,
        "The question particle with a person ending written attached: biliyormusun.";
    MiHarmony = "mi-harmony", ParticleMi, true, 0.67,
        "A separate question particle out of vowel harmony: gitti mı.";
    DeDaHardened = "de-da-hardened", CliticDeDa, true, 0.88,
        "The conjunction de/da written te/ta: Veli te geldi.";
    DeDaMergedFixed = "de-da-merged-fixed", CliticDeDa, true, 0.99,
        "A fixed list of words with the conjunction de/da attached or after an apostrophe: yada, hemde, ya'da.";
    DeDaAfterVerb = "de-da-after-verb", CliticDeDa, true, 0.99,
        "The conjunction de/da attached to a finite verb or a word in a case: gittide, onuda.";
    LocativeSplit = "locative-split", CliticDeDa, true, 0.99,
        "The locative or ablative suffix written apart from its noun: hakkın da.";
    KiLexicalizedSplit = "ki-lexicalized-split", CliticKi, true, 0.98,
        "A lexicalised ki written apart: san ki, oysa ki.";
    KiMergedFixed = "ki-merged-fixed", CliticKi, true, 0.99,
        "A fixed list of words with the conjunction ki attached: demekki, iyiki.";
    KiAfterVerb = "ki-after-verb", CliticKi, true, 0.99,
        "The conjunction ki attached to a verb: duydumki.";
    OrdinalDotApostrophe = "ordinal-dot-apostrophe", ApostropheNumberDate, true, 0.5,
        "An ordinal written with both a period and a suffix: 2.'nci.";
    NumberSuffix = "number-suffix", ApostropheNumberDate, true, 0.90,
        "A locative or ablative suffix after a number that does not match how the number is read: 1923'de, 1940'da.";
    ApostropheDerivational = "apostrophe-derivational", ApostropheProperNoun, true, 0.98,
        "An apostrophe before a plural or derivational suffix of a proper noun: Türk'ler, Ankara'lı.";
    ApostrophePluralName = "apostrophe-plural-name", ApostropheProperNoun, false, 0.67,
        "An apostrophe after the plural of a proper noun: Ahmetler'in, Fransızlar'ın.";
    Reduplication = "reduplication", ReduplicationOrthography, true, 0.90,
        "A fixed list of reduplications written as one word: elele.";
    EbilSplit = "ebil-split", CompoundWordOrthography, true, 0.99,
        "The ability verb -ebil written apart: ola bilir.";
    IndefiniteCompound = "indefinite-compound", CompoundWordOrthography, true, 0.96,
        "An indefinite determiner written apart: bir çok, hiç bir.";
    CompoundVerbMerged = "compound-verb-merged", CompoundWordOrthography, true, 0.83,
        "A fixed list of compound verbs written as one word: terketmek.";
    CompoundVerbSplit = "compound-verb-split", CompoundWordOrthography, true, 0.86,
        "A fixed list of compound verbs written apart: hiss etmek.";
    ConsonantAssimilation = "consonant-assimilation", ConsonantAssimilation, true, 0.94,
        "A suffix not hardened after a voiceless consonant: kitapda.";
    ConsonantSoftening = "consonant-softening", ConsonantSoftening, true, 0.94,
        "A fixed list of stems not softened before a vowel: kitapı.";
    VowelDrop = "vowel-drop", VowelDropping, true, 0.96,
        "A fixed list of stems keeping the vowel that drops before a vowel: şehire, omuzundan.";
    YorNarrowing = "yor-narrowing", VowelHarmony, true, 0.5,
        "a/e not narrowed before -yor: başlayor.";
    OverNarrowing = "over-narrowing", VowelHarmony, true, 0.99,
        "a/e narrowed outside -yor: başlıyan, yapmıya.";
    ColloquialForm = "colloquial-form", VowelDropping, true, 0.91,
        "A fixed list of colloquial forms: burda, yalnış.";
    PluralHarmony = "plural-harmony", VowelHarmony, false, 0.5,
        "The plural suffix out of vowel harmony on a word the analyser does not know: evlar.";
    QuantityPlural = "quantity-plural", QuantityPluralClash, false, 0.18,
        "A plural noun after a number or quantifier: üç elmalar.";
    GenitivePossessive = "genitive-possessive", GenitivePossessiveClash, false, 0.25,
        "A noun without the possessive after benim: benim ev.";
    NeNePolarity = "ne-ne-polarity", CorrelativeConjunctionPolarity, false, 0.2,
        "A negative verb with ne ... ne: ne gelmedi ne gitmedi.";
    NegationConcord = "negation-concord", CorrelativeConjunctionPolarity, false, 0.05,
        "A positive verb after kimse or asla in its sentence: kimse geldi (detection only).";
    InanimatePluralVerb = "inanimate-plural-verb", SubjectVerbAgreement, false, 0.04,
        "A plural verb after a plural subject that opens the sentence: Ağaçlar döküldüler.";
    Tautology = "tautology", TautologyRedundancy, true, 0.5,
        "A fixed list of redundant pairs: geri iade, birlikte beraber.";
}

impl GrammarRule {
    /// The rule with this id.
    pub fn from_id(id: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|r| r.id() == id)
    }

    /// Whether the rule is off unless asked for.
    pub fn is_experimental(self) -> bool {
        !self.is_default()
    }

    fn bit(self) -> u64 {
        1 << (self as u32)
    }
}

impl fmt::Display for GrammarRule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id())
    }
}

/// An id that names no rule.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("unknown grammar rule '{0}'")]
pub struct UnknownRule(pub String);

impl FromStr for GrammarRule {
    type Err = UnknownRule;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_id(s.trim()).ok_or_else(|| UnknownRule(s.to_string()))
    }
}

impl Serialize for GrammarRule {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.id())
    }
}

impl<'de> Deserialize<'de> for GrammarRule {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let id = String::deserialize(deserializer)?;
        id.parse().map_err(serde::de::Error::custom)
    }
}

/// The rules a check runs. [`RuleSet::default`] is the default set.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct RuleSet(u64);

impl RuleSet {
    /// No rule.
    pub const fn empty() -> Self {
        Self(0)
    }

    /// Every rule, experimental ones included.
    pub fn all() -> Self {
        GrammarRule::ALL.iter().copied().collect()
    }

    /// The rules that meet the precision bar.
    pub fn default_rules() -> Self {
        GrammarRule::ALL
            .iter()
            .copied()
            .filter(|r| r.is_default())
            .collect()
    }

    /// The rules that do not meet it.
    pub fn experimental() -> Self {
        GrammarRule::ALL
            .iter()
            .copied()
            .filter(|r| r.is_experimental())
            .collect()
    }

    /// The rules with these ids.
    pub fn from_ids<I, S>(ids: I) -> Result<Self, UnknownRule>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        ids.into_iter().map(|id| id.as_ref().parse()).collect()
    }

    /// These rules and every experimental one.
    pub fn with_experimental(self) -> Self {
        Self(self.0 | Self::experimental().0)
    }

    /// These rules and `rule`.
    pub fn with(mut self, rule: GrammarRule) -> Self {
        self.enable(rule);
        self
    }

    /// These rules without `rule`.
    pub fn without(mut self, rule: GrammarRule) -> Self {
        self.disable(rule);
        self
    }

    pub fn enable(&mut self, rule: GrammarRule) {
        self.0 |= rule.bit();
    }

    pub fn disable(&mut self, rule: GrammarRule) {
        self.0 &= !rule.bit();
    }

    pub fn contains(&self, rule: GrammarRule) -> bool {
        self.0 & rule.bit() != 0
    }

    pub fn is_empty(&self) -> bool {
        self.0 == 0
    }

    /// The rules in the set, in [`GrammarRule::ALL`] order.
    pub fn iter(&self) -> impl Iterator<Item = GrammarRule> + '_ {
        GrammarRule::ALL
            .iter()
            .copied()
            .filter(move |&r| self.contains(r))
    }
}

impl Default for RuleSet {
    fn default() -> Self {
        Self::default_rules()
    }
}

impl FromIterator<GrammarRule> for RuleSet {
    fn from_iter<I: IntoIterator<Item = GrammarRule>>(iter: I) -> Self {
        let mut set = Self::empty();
        for rule in iter {
            set.enable(rule);
        }
        set
    }
}

impl fmt::Debug for RuleSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set()
            .entries(self.iter().map(GrammarRule::id))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_unique_and_round_trip() {
        assert!(GrammarRule::ALL.len() <= 64, "RuleSet is a u64 bit set");
        for (i, &rule) in GrammarRule::ALL.iter().enumerate() {
            assert_eq!(rule as usize, i);
            assert_eq!(GrammarRule::from_id(rule.id()), Some(rule));
            let json = serde_json::to_string(&rule).unwrap();
            assert_eq!(json, format!("\"{}\"", rule.id()));
            assert_eq!(serde_json::from_str::<GrammarRule>(&json).unwrap(), rule);
        }
    }

    #[test]
    fn sets_compose() {
        let default = RuleSet::default();
        assert_eq!(default.with_experimental(), RuleSet::all());
        for rule in GrammarRule::ALL.iter().copied() {
            assert_eq!(default.contains(rule), rule.is_default());
            assert!(!default.without(rule).contains(rule));
            assert!(RuleSet::empty().with(rule).contains(rule));
        }
        assert_eq!(
            RuleSet::from_ids(["mi-merged", "tautology"]).unwrap(),
            RuleSet::empty()
                .with(GrammarRule::MiMerged)
                .with(GrammarRule::Tautology)
        );
        assert_eq!(
            RuleSet::from_ids(["nope"]),
            Err(UnknownRule("nope".to_string()))
        );
    }
}
