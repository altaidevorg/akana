//! `MorphParse::pos` is the POS after the last derivation (Zemberek's SingleAnalysis.getPos),
//! and the disambiguator and parser use it instead of the root's POS.
use akana_core::morphology::*;
use akana_core::parser::*;

fn pos_of(word: &str, root: &str, tag: &str) -> PrimaryPos {
    TurkishMorphology::new()
        .analyze(word)
        .into_iter()
        .find(|p| p.root == root && p.morpheme_tags.iter().any(|t| t == tag))
        .unwrap_or_else(|| panic!("{word}: no {root} parse with {tag}"))
        .pos
}

#[test]
fn derivations_set_the_pos() {
    // Noun root, verb derivation, converb: the word is an adverbial, not a noun.
    assert_eq!(pos_of("başlayınca", "baş", "NounToVerb"), PrimaryPos::Adv);
    assert_eq!(pos_of("başlayınca", "başla", "AdvInce"), PrimaryPos::Adv);
    // Noun root + -lAş: a finite verb.
    assert_eq!(pos_of("anlaşıyorum", "an", "BecomeVerb"), PrimaryPos::Verb);
    // Participle + possessive + case: a nominalised clause (Zemberek's Zero -> Noun).
    assert_eq!(pos_of("olduğunu", "ol", "PastPart"), PrimaryPos::Noun);
    assert_eq!(pos_of("giden", "git", "PresPart"), PrimaryPos::Adj);
    assert_eq!(pos_of("gelmek", "gel", "Inf"), PrimaryPos::Noun);
    // Inflected adjective: zero-derived noun.
    assert_eq!(pos_of("güzeli", "güzel", "Acc"), PrimaryPos::Noun);
    // The copula keeps the nominal POS (unlike Zemberek, as UD expects).
    assert_eq!(pos_of("mutluyduk", "mutlu", "CopPast"), PrimaryPos::Adj);
}

#[test]
fn root_pos_is_kept() {
    let parses = TurkishMorphology::new().analyze("başlayınca");
    let noun_derived = parses.iter().find(|p| p.root == "baş").expect("baş");
    assert_eq!(noun_derived.primary_pos, PrimaryPos::Noun);
}

fn parse(sentence: &str) -> Vec<DependencyNode> {
    let tokens: Vec<&str> = sentence.split_whitespace().collect();
    TurkishDependencyParser::new().parse(&tokens).nodes
}

fn node<'a>(nodes: &'a [DependencyNode], form: &str) -> &'a DependencyNode {
    nodes.iter().find(|n| n.form == form).expect(form)
}

#[test]
fn converbs_are_verbs_heading_adverbial_clauses() {
    let nodes = parse("Yağmur yağmaya başlayınca herkes en yakın kafeye koştu");
    let conv = node(&nodes, "başlayınca");
    assert_eq!(conv.upos, "VERB");
    assert_eq!(conv.lemma, "başla");
    assert_eq!(conv.deprel, "advcl");
    assert_eq!(node(&nodes, "koştu").deprel, "root");
}

#[test]
fn derived_verbs_are_verbs() {
    let nodes = parse("Şimdi iş arkadaşlarımla çok iyi anlaşıyorum ve işimi seviyorum");
    let verb = node(&nodes, "anlaşıyorum");
    assert_eq!(verb.upos, "VERB");
    assert!(verb.feats.starts_with("Verb"), "{}", verb.feats);
}

#[test]
fn participles_modify_the_following_noun() {
    let nodes = parse("Okula giden çocuk geldi");
    let part = node(&nodes, "giden");
    assert_eq!(part.upos, "VERB");
    assert_eq!(part.deprel, "acl");
    assert_eq!(part.head, 3);
    assert_eq!(node(&nodes, "geldi").deprel, "root");
}

#[test]
fn nominalised_clauses_are_not_the_root() {
    let nodes = parse("Eve döndüğümde kapının açık olduğunu görünce çok korktum");
    assert_eq!(node(&nodes, "korktum").deprel, "root");
    assert_eq!(node(&nodes, "olduğunu").deprel, "obj");
    assert_eq!(node(&nodes, "görünce").deprel, "advcl");
}
