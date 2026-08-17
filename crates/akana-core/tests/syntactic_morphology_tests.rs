//! Tests for Google-style Syntactic Expressive Morphology (FSMNLP 2019).

use akana_core::syntactic_morphology::TurkishSyntacticMorphology;

#[test]
fn test_syntactic_morphology_root_cross_categorization() {
    let morph = TurkishSyntacticMorphology::new();

    // "güzel" should produce ADJ, ADV, NN parses without zero-derivation
    let parses = morph.analyze("güzel");
    assert!(!parses.is_empty());

    let pos_list: Vec<&str> = parses.iter().map(|p| p.root_pos.as_str()).collect();
    assert!(pos_list.contains(&"ADJ"));
    assert!(pos_list.contains(&"ADV"));
}

#[test]
fn test_syntactic_morphology_inflectional_groups_derivation() {
    let morph = TurkishSyntacticMorphology::new();

    // "geldiğimizde" -> 2 Inflectional Groups (Verb root tier + PastNom derivation tier)
    let parses = morph.analyze("geldiğimizde");
    assert!(!parses.is_empty());

    let parse = parses.iter().find(|p| p.root == "gel").expect("Must parse 'gel'");
    assert_eq!(parse.root_pos, "VB");
    assert!(parse.inflectional_groups.len() >= 2, "Must contain at least 2 Inflectional Groups");

    let ig0 = &parse.inflectional_groups[0];
    assert_eq!(ig0.pos, "VB");
    assert_eq!(ig0.features.get("Polarity").unwrap(), "Pos");

    let ig1 = &parse.inflectional_groups[1];
    assert_eq!(ig1.pos, "NOMP");
    assert_eq!(ig1.derivation.as_deref(), Some("PastNom"));
    assert_eq!(ig1.features.get("Case").unwrap(), "Loc");
}

#[test]
fn test_syntactic_morphology_nominal_inflections() {
    let morph = TurkishSyntacticMorphology::new();

    let parses = morph.analyze("evlerimizde");
    assert!(!parses.is_empty());

    let parse = parses.iter().find(|p| p.root == "ev").expect("Must parse 'ev'");
    assert_eq!(parse.root_pos, "NN");

    let ig = &parse.inflectional_groups[0];
    assert_eq!(ig.features.get("PersonNumber").unwrap(), "A3pl");
    assert_eq!(ig.features.get("Possessive").unwrap(), "P1pl");
    assert_eq!(ig.features.get("Case").unwrap(), "Loc");
}
