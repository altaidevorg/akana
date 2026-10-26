//! Tests for the syntactic analyzer: inflectional groups (IGs) with UD features, built from the
//! morphological analyzer's morphotactic paths.

use akana_core::morphology::{is_derivational_tag, MorphologicalDisambiguator, TurkishMorphology};
use akana_core::syntactic_morphology::{SyntacticParse, TurkishSyntacticMorphology};

const PARADIGMS: &str = include_str!("data/morphology/paradigms.tsv");
const FREQUENT_TYPES: &str = include_str!("data/morphology/frequent_types.tsv");
const ROUNDTRIP_WORDS: &str = include_str!("data/morphology/roundtrip_words.txt");

fn analyze(word: &str) -> Vec<SyntacticParse> {
    TurkishSyntacticMorphology::new().analyze(word)
}

/// Fails unless one parse of `word` is exactly `formatted`, and lists them all when none is.
fn assert_parse(word: &str, formatted: &str) -> SyntacticParse {
    let parses = analyze(word);
    parses
        .iter()
        .find(|p| p.formatted == formatted)
        .cloned()
        .unwrap_or_else(|| {
            let all: Vec<&str> = parses.iter().map(|p| p.formatted.as_str()).collect();
            panic!("{word}: no parse {formatted}\n  got {all:#?}")
        })
}

fn feature<'a>(parse: &'a SyntacticParse, group: usize, key: &str) -> Option<&'a str> {
    parse.inflectional_groups[group]
        .features
        .get(key)
        .map(String::as_str)
}

#[test]
fn test_syntactic_morphology_root_cross_categorization() {
    // The shared lexicon lists güzel as an adjective, an adverb and a noun: three readings, no
    // zero derivation.
    let parses = analyze("güzel");
    let pos_list: Vec<&str> = parses.iter().map(|p| p.root_pos.as_str()).collect();
    assert!(pos_list.contains(&"ADJ"));
    assert!(pos_list.contains(&"ADV"));
    assert!(pos_list.contains(&"NN"));
    assert!(parses.iter().all(|p| p.inflectional_groups.len() == 1));
    assert_parse("güzel", "(güzel[ADJ])+[Proper=False]");
}

#[test]
fn test_syntactic_morphology_inflectional_groups_derivation() {
    // "geldiğimizde" -> the verb's group and the -DIk nominalisation's.
    let parse = assert_parse(
        "geldiğimizde",
        "(gel[VB]+[Polarity=Pos])([NOMP]-PastNom+[Case=Loc]+[Number=Sing]+[Number[psor]=Plur]\
         +[Person=3]+[Person[psor]=1]+[VerbForm=Vnoun])+[Proper=False]",
    );
    assert_eq!(parse.root, "gel");
    assert_eq!(parse.root_pos, "VB");
    let ig1 = &parse.inflectional_groups[1];
    assert_eq!(ig1.pos, "NOMP");
    assert_eq!(ig1.derivation.as_deref(), Some("PastNom"));
    assert_eq!(feature(&parse, 1, "Case"), Some("Loc"));
}

#[test]
fn test_syntactic_morphology_nominal_inflections() {
    let parse = assert_parse(
        "evlerimizde",
        "(ev[NN]+[Case=Loc]+[Number=Plur]+[Number[psor]=Plur]+[Person=3]+[Person[psor]=1])\
         +[Proper=False]",
    );
    assert_eq!(parse.root_pos, "NN");
    assert_eq!(
        analyze("evlerimizde")[0],
        parse,
        "the best reading comes first"
    );
    assert_parse(
        "Ankara'da",
        "(ankara[NN]+[Case=Loc]+[Number=Sing]+[Person=3])+[Proper=True]",
    );
}

#[test]
fn zero_derivations_are_eliminated() {
    // An adjective inflected as a noun is a noun, so güzel the adjective and güzel the noun give
    // one parse; only the copula opens a group.
    let plural: Vec<SyntacticParse> = analyze("güzeller")
        .into_iter()
        .filter(|p| p.inflectional_groups.len() == 1)
        .collect();
    assert_eq!(plural.len(), 1, "{plural:#?}");
    assert_eq!(
        plural[0].formatted,
        "(güzel[NN]+[Case=Nom]+[Number=Plur]+[Person=3])+[Proper=False]"
    );
    // A postposition read as a noun.
    assert_parse(
        "sonrası",
        "(sonra[NN]+[Case=Nom]+[Number=Sing]+[Number[psor]=Sing]+[Person=3]+[Person[psor]=3])\
         +[Proper=False]",
    );
    // A numeral stays a numeral.
    assert_parse(
        "1923'te",
        "(1923[NUM]+[Case=Loc]+[Number=Sing]+[Person=3])+[Proper=False]",
    );
    // -ki then a zero derivation to a noun: the -ki group becomes the noun.
    assert_parse(
        "evdekiler",
        "(ev[NN]+[Case=Loc]+[Number=Sing]+[Person=3])([NN]-Rel+[Case=Nom]+[Number=Plur]\
         +[Person=3])+[Proper=False]",
    );
}

#[test]
fn participles() {
    assert_parse(
        "gelen",
        "(gel[VB]+[Polarity=Pos])([ADJP]-PresPart+[VerbForm=Part])+[Proper=False]",
    );
    assert_parse(
        "gelecek",
        "(gel[VB]+[Polarity=Pos])([ADJP]-FutPart+[VerbForm=Part])+[Proper=False]",
    );
    assert_parse(
        "gelmiş",
        "(gel[VB]+[Polarity=Pos])([ADJP]-NarrPart+[VerbForm=Part])+[Proper=False]",
    );
    // -(y)AcAk as a noun, then the past copula.
    assert_parse(
        "gelecekti",
        "(gel[VB]+[Polarity=Pos])([NOMP]-FutNom+[Case=Nom]+[Number=Sing]+[Person=3]\
         +[VerbForm=Vnoun])([VB]-Cop+[Aspect=Perf]+[Evident=Fh]+[Mood=Ind]+[Number=Sing]\
         +[Person=3]+[Tense=Past]+[VerbForm=Fin])+[Proper=False]",
    );
    // The aorist participle after -Abil: the ability group is not finite.
    assert_parse(
        "gelebilir",
        "(gel[VB]+[Polarity=Pos])([VB]-Able+[Mood=Pot]+[Polarity=Pos])([ADJP]-AorPart\
         +[VerbForm=Part])+[Proper=False]",
    );
}

#[test]
fn verbal_nouns() {
    for (word, derivation) in [("gelmek", "Inf1"), ("gelme", "Inf2"), ("geliş", "Inf3")] {
        assert_parse(
            word,
            &format!(
                "(gel[VB]+[Polarity=Pos])([NOMP]-{derivation}+[Case=Nom]+[Number=Sing]\
                 +[Person=3]+[VerbForm=Vnoun])+[Proper=False]"
            ),
        );
    }
    // An agent noun is no verbal noun.
    assert_parse(
        "okuyucu",
        "(oku[VB]+[Polarity=Pos])([NOMP]-Agt+[Case=Nom]+[Number=Sing]+[Person=3])\
         +[Proper=False]",
    );
}

#[test]
fn converbs() {
    for (word, derivation) in [
        ("gelince", "When"),
        ("gelerek", "ByDoingSo"),
        ("gelip", "AfterDoingSo"),
        ("gelmeden", "WithoutHavingDoneSo"),
    ] {
        assert_parse(
            word,
            &format!("(gel[VB]+[Polarity=Pos])([ADV]-{derivation}+[VerbForm=Conv])+[Proper=False]"),
        );
    }
    // The aorist before -ken has no person: not finite.
    assert_parse(
        "gelirken",
        "(gel[VB]+[Aspect=Hab]+[Mood=Gen]+[Polarity=Pos]+[Tense=Pres])([ADV]-While\
         +[VerbForm=Conv])+[Proper=False]",
    );
}

#[test]
fn voice() {
    // Each voice suffix opens a group, as in Zemberek.
    assert_parse(
        "yaptırıldı",
        "(yap[VB]+[Polarity=Pos])([VB]-Caus+[Polarity=Pos]+[Voice=Cau])([VB]-Pass\
         +[Aspect=Perf]+[Evident=Fh]+[Mood=Ind]+[Number=Sing]+[Person=3]+[Polarity=Pos]\
         +[Tense=Past]+[VerbForm=Fin]+[Voice=Pass])+[Proper=False]",
    );
    assert_parse(
        "yıkandı",
        "(yıka[VB]+[Polarity=Pos])([VB]-Pass+[Aspect=Perf]+[Evident=Fh]+[Mood=Ind]\
         +[Number=Sing]+[Person=3]+[Polarity=Pos]+[Tense=Past]+[VerbForm=Fin]+[Voice=Pass])\
         +[Proper=False]",
    );
}

#[test]
fn ability_and_inability() {
    // UD Turkish concatenates moods: the aorist's Gen and the ability's Pot.
    assert_parse(
        "gelebilir",
        "(gel[VB]+[Polarity=Pos])([VB]-Able+[Aspect=Hab]+[Mood=GenPot]+[Number=Sing]\
         +[Person=3]+[Polarity=Pos]+[Tense=Pres]+[VerbForm=Fin])+[Proper=False]",
    );
    // Negation stays in the group it marks.
    assert_parse(
        "gelmeyebilir",
        "(gel[VB]+[Polarity=Neg])([VB]-Able+[Aspect=Hab]+[Mood=GenPot]+[Number=Sing]\
         +[Person=3]+[Polarity=Pos]+[Tense=Pres]+[VerbForm=Fin])+[Proper=False]",
    );
    assert_parse(
        "gelemedi",
        "(gel[VB]+[Aspect=Perf]+[Evident=Fh]+[Mood=Pot]+[Number=Sing]+[Person=3]\
         +[Polarity=Neg]+[Tense=Past]+[VerbForm=Fin])+[Proper=False]",
    );
}

#[test]
fn nominal_derivations() {
    assert_parse(
        "güzellik",
        "(güzel[ADJ])([NN]-Ness+[Case=Nom]+[Number=Sing]+[Person=3])+[Proper=False]",
    );
    assert_parse(
        "evli",
        "(ev[NN]+[Case=Nom]+[Number=Sing]+[Person=3])([ADJP]-With)+[Proper=False]",
    );
    assert_parse(
        "evsiz",
        "(ev[NN]+[Case=Nom]+[Number=Sing]+[Person=3])([ADJP]-Without)+[Proper=False]",
    );
    assert_parse(
        "kitapçık",
        "(kitap[NN]+[Case=Nom]+[Number=Sing]+[Person=3])([NN]-Dim+[Case=Nom]+[Number=Sing]\
         +[Person=3])+[Proper=False]",
    );
    assert_parse(
        "kitapçı",
        "(kitap[NN]+[Case=Nom]+[Number=Sing]+[Person=3])([NN]-Agt+[Case=Nom]+[Number=Sing]\
         +[Person=3])+[Proper=False]",
    );
    assert_parse(
        "güzelleşti",
        "(güzel[ADJ])([VB]-Become+[Aspect=Perf]+[Evident=Fh]+[Mood=Ind]+[Number=Sing]\
         +[Person=3]+[Polarity=Pos]+[Tense=Past]+[VerbForm=Fin])+[Proper=False]",
    );
}

#[test]
fn copula() {
    // The copula is a group of its own (UD splits it off as the auxiliary).
    assert_parse(
        "evdeyim",
        "(ev[NN]+[Case=Loc]+[Number=Sing]+[Person=3])([VB]-Cop+[Mood=Ind]+[Number=Sing]\
         +[Person=1]+[Tense=Pres]+[VerbForm=Fin])+[Proper=False]",
    );
    assert_parse(
        "güzeldi",
        "(güzel[ADJ])([VB]-Cop+[Aspect=Perf]+[Evident=Fh]+[Mood=Ind]+[Number=Sing]\
         +[Person=3]+[Tense=Past]+[VerbForm=Fin])+[Proper=False]",
    );
    assert_parse(
        "hastaymış",
        "(hasta[ADJ])([VB]-Cop+[Aspect=Perf]+[Evident=Nfh]+[Mood=Ind]+[Number=Sing]\
         +[Person=3]+[Tense=Past]+[VerbForm=Fin])+[Proper=False]",
    );
    // -DIr is the generalizing mood.
    assert_parse(
        "öğretmendir",
        "(öğretmen[NN]+[Case=Nom]+[Number=Sing]+[Person=3])([VB]-Cop+[Mood=Gen]+[Number=Sing]\
         +[Person=3]+[Tense=Pres]+[VerbForm=Fin])+[Proper=False]",
    );
    // değil and the question particle are predicates themselves: no copula group.
    assert_parse(
        "değilim",
        "(değil[VB]+[Mood=Ind]+[Number=Sing]+[Person=1]+[Polarity=Neg]+[Tense=Pres]\
         +[VerbForm=Fin])+[Proper=False]",
    );
    assert_parse(
        "misin",
        "(mi[QUES]+[Mood=Ind]+[Number=Sing]+[Person=2]+[Tense=Pres]+[VerbForm=Fin])\
         +[Proper=False]",
    );
}

#[test]
fn tense_aspect_mood() {
    let verb = |features: &str| format!("(gel[VB]{features})+[Proper=False]");
    for (word, features) in [
        (
            "gelmişti",
            "+[Aspect=Perf]+[Evident=Fh]+[Mood=Ind]+[Number=Sing]+[Person=3]+[Polarity=Pos]\
             +[Tense=Pqp]+[VerbForm=Fin]",
        ),
        (
            "geliyordu",
            "+[Aspect=Prog]+[Evident=Fh]+[Mood=Ind]+[Number=Sing]+[Person=3]+[Polarity=Pos]\
             +[Tense=Past]+[VerbForm=Fin]",
        ),
        (
            "gelecekti",
            "+[Aspect=Prosp]+[Evident=Fh]+[Mood=Ind]+[Number=Sing]+[Person=3]+[Polarity=Pos]\
             +[Tense=Past]+[VerbForm=Fin]",
        ),
        (
            "gelirse",
            "+[Aspect=Hab]+[Mood=CndGen]+[Number=Sing]+[Person=3]+[Polarity=Pos]+[Tense=Pres]\
             +[VerbForm=Fin]",
        ),
        (
            "gelse",
            "+[Mood=Des]+[Number=Sing]+[Person=3]+[Polarity=Pos]+[VerbForm=Fin]",
        ),
        (
            "gelmeli",
            "+[Mood=Nec]+[Number=Sing]+[Person=3]+[Polarity=Pos]+[Tense=Pres]+[VerbForm=Fin]",
        ),
        (
            "gelelim",
            "+[Mood=Opt]+[Number=Plur]+[Person=1]+[Polarity=Pos]+[VerbForm=Fin]",
        ),
        (
            "gelme",
            "+[Mood=Imp]+[Number=Sing]+[Person=2]+[Polarity=Neg]+[VerbForm=Fin]",
        ),
    ] {
        assert_parse(word, &verb(features));
    }
}

#[test]
fn parses_are_unique_and_ranked() {
    let morphology = TurkishMorphology::new();
    for word in ["evde", "kitapları", "güzeller", "yüzü", "gelirken"] {
        let parses = analyze(word);
        let mut formatted: Vec<&str> = parses.iter().map(|p| p.formatted.as_str()).collect();
        let n = formatted.len();
        formatted.sort_unstable();
        formatted.dedup();
        assert_eq!(formatted.len(), n, "{word}: repeated parses");
        // The best morphological reading's groups come first.
        let best = morphology.analyze(word);
        assert_eq!(
            parses[0],
            TurkishSyntacticMorphology::from_parse(word, &best[0]),
            "{word}"
        );
    }
    assert!(analyze("xqzzt").is_empty());
    assert!(analyze("").is_empty());
}

#[test]
fn readings_without_a_path_give_one_group() {
    // The disambiguator's reading of an unknown word was not made by the analyzer.
    let unknown = MorphologicalDisambiguator::new().disambiguate(&["xqzzt"]);
    let parse = TurkishSyntacticMorphology::from_parse("xqzzt", &unknown[0]);
    assert_eq!(parse.formatted, "(xqzzt[UNK])+[Proper=False]");
    // Nor was a reading read back from JSON.
    let ev = &TurkishMorphology::new().analyze("evde")[0];
    let json = serde_json::to_string(ev).unwrap();
    let back: akana_core::morphology::MorphParse = serde_json::from_str(&json).unwrap();
    assert_eq!(
        TurkishSyntacticMorphology::from_parse("evde", &back).formatted,
        "(ev[NN])+[Proper=False]"
    );
}

/// What a legacy tag of [`akana_core::morphology::MorphParse::morpheme_tags`] must show in the
/// groups: a feature value (a Mood or Voice value may be one of several concatenated), or the
/// derivation that opens the next group.
enum Shows {
    Feature(&'static str, &'static [&'static str]),
    Features(&'static [(&'static str, &'static str)]),
    Derivation(&'static [&'static str]),
    Nothing,
}

fn shows(tag: &str) -> Shows {
    use Shows::*;
    match tag {
        "Noun" | "Verb" | "Adj" | "Adv" | "Pron" | "Num" | "Conj" | "Postp" | "Interj" | "Q"
        | "Punc" | "Dup" | "Det" => Nothing,
        "Nom" | "Dat" | "Acc" | "Abl" | "Loc" | "Ins" | "Gen" | "Equ" => Feature("Case", &[]),
        "A1sg" | "CopPres1sg" => Features(&[("Person", "1"), ("Number", "Sing")]),
        "A2sg" | "CopPres2sg" => Features(&[("Person", "2"), ("Number", "Sing")]),
        "A3sg" => Features(&[("Person", "3"), ("Number", "Sing")]),
        "A1pl" | "CopPres1pl" => Features(&[("Person", "1"), ("Number", "Plur")]),
        "A2pl" | "CopPres2pl" => Features(&[("Person", "2"), ("Number", "Plur")]),
        "A3pl" | "CopPres3pl" => Features(&[("Person", "3"), ("Number", "Plur")]),
        "P1sg" => Features(&[("Person[psor]", "1"), ("Number[psor]", "Sing")]),
        "P2sg" => Features(&[("Person[psor]", "2"), ("Number[psor]", "Sing")]),
        "P3sg" => Features(&[("Person[psor]", "3"), ("Number[psor]", "Sing")]),
        "P1pl" => Features(&[("Person[psor]", "1"), ("Number[psor]", "Plur")]),
        "P2pl" => Features(&[("Person[psor]", "2"), ("Number[psor]", "Plur")]),
        "P3pl" => Features(&[("Person[psor]", "3"), ("Number[psor]", "Plur")]),
        "Neg" => Feature("Polarity", &["Neg"]),
        "NegAbil" => Features(&[("Polarity", "Neg"), ("Mood", "Pot")]),
        // The last past of a group decides its evidentiality (gel-miş-ti is Fh).
        "Past" | "CopPast" | "Evid" | "CopEvid" => Feature("Evident", &[]),
        "Prog" => Feature("Aspect", &["Prog"]),
        "Aor" | "Cop" | "CopPres3sg" => Feature("Mood", &["Gen"]),
        "Fut" => Feature("Tense", &["Fut", "Past"]),
        "Imp" => Feature("Mood", &["Imp"]),
        "Opt" => Feature("Mood", &["Opt"]),
        "Cond" => Feature("Mood", &["Des"]),
        "CopCond" => Feature("Mood", &["Cnd"]),
        "Nec" => Feature("Mood", &["Nec"]),
        "Abil" => Derivation(&["Able"]),
        "RelClitic" => Derivation(&["Rel"]),
        "BecomeVerb" => Derivation(&["Become"]),
        "Inf" => Derivation(&["Inf1"]),
        "ActN" => Derivation(&["Inf2"]),
        "Manner" => Derivation(&["Inf3"]),
        "PastPart" => Derivation(&["PastPart", "PastNom"]),
        "FutPart" => Derivation(&["FutPart", "FutNom"]),
        "AdvInce" => Derivation(&["When"]),
        "AdvErek" => Derivation(&["ByDoingSo"]),
        "AdvIp" => Derivation(&["AfterDoingSo"]),
        t if is_derivational_tag(t) => Derivation(&[]),
        t => panic!("legacy tag {t} has no expected IG feature"),
    }
}

/// Mood and Voice values are concatenations of single values (GenPot, CauPass).
fn has_value(actual: &str, key: &str, value: &str) -> bool {
    if key == "Mood" || key == "Voice" {
        actual.contains(value)
    } else {
        actual == value
    }
}

#[test]
fn every_morpheme_of_the_gold_words_maps_to_its_group() {
    let morphology = TurkishMorphology::new();
    let mut words: Vec<&str> = Vec::new();
    for line in PARADIGMS.lines().chain(FREQUENT_TYPES.lines()) {
        if let Some(word) = line.split('\t').next().filter(|_| !line.starts_with('#')) {
            words.push(word);
        }
    }
    words.extend(ROUNDTRIP_WORDS.lines().filter(|l| !l.starts_with('#')));
    words.retain(|w| !w.is_empty());
    let mut readings = 0;
    for word in words {
        for reading in morphology.analyze(word) {
            readings += 1;
            let parse = TurkishSyntacticMorphology::from_parse(word, &reading);
            let groups = &parse.inflectional_groups;
            let context = || format!("{word}: {} -> {}", reading.formatted, parse.formatted);
            assert!(groups[0].derivation.is_none(), "{}", context());
            assert!(
                groups[1..].iter().all(|g| g.derivation.is_some()),
                "{}",
                context()
            );
            // The groups' derivations, but the copula's (which the legacy tags show as an
            // inflection), in the order of the legacy derivational tags.
            let mut derivations = groups[1..]
                .iter()
                .filter_map(|g| g.derivation.as_deref())
                .filter(|d| *d != "Cop");
            for tag in &reading.morpheme_tags[1..] {
                match shows(tag) {
                    Shows::Derivation(names) => {
                        let d = derivations.next();
                        let ok = d.is_some_and(|d| {
                            if names.is_empty() {
                                d == tag.as_str()
                            } else {
                                names.contains(&d)
                            }
                        });
                        assert!(ok, "{tag} vs derivation {d:?}: {}", context());
                    }
                    Shows::Feature(key, values) => {
                        let ok = groups.iter().any(|g| {
                            g.features.get(key).is_some_and(|v| {
                                values.is_empty() || values.iter().any(|x| has_value(v, key, x))
                            })
                        });
                        assert!(ok, "{tag}: no {key}={values:?}: {}", context());
                    }
                    Shows::Features(pairs) => {
                        let ok = groups.iter().any(|g| {
                            pairs.iter().all(|(key, value)| {
                                g.features
                                    .get(*key)
                                    .is_some_and(|v| has_value(v, key, value))
                            })
                        });
                        assert!(ok, "{tag}: no {pairs:?} in one group: {}", context());
                    }
                    Shows::Nothing => {}
                }
            }
            assert!(
                derivations.next().is_none(),
                "a group no legacy tag shows: {}",
                context()
            );
        }
    }
    assert!(readings > 2000, "{readings} readings");
}
