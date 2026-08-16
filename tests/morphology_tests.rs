use akana_core::morphology::*;

#[test]
fn test_nominal_morphology_analysis() {
    let morph = TurkishMorphology::new();

    // 1. Root with Voicing (kitap -> kitaba, kitabım)
    let parses_kitabim = morph.analyze("kitabım");
    assert!(!parses_kitabim.is_empty());
    assert_eq!(parses_kitabim[0].root, "kitap");
    assert!(parses_kitabim.iter().any(|p| p.morpheme_tags.contains(&"P1sg".to_string())));

    // 2. Root with Vowel Drop (burun -> burnum)
    let parses_burnum = morph.analyze("burnum");
    assert!(!parses_burnum.is_empty());
    assert_eq!(parses_burnum[0].root, "burun");

    // 3. Root with Consonant Doubling (hak -> hakkım)
    let parses_hakkim = morph.analyze("hakkım");
    assert!(!parses_hakkim.is_empty());
    assert_eq!(parses_hakkim[0].root, "hak");

    // 4. Plural + Case (evlerimizde)
    let parses_evler = morph.analyze("evlerimizde");
    assert!(!parses_evler.is_empty());
    assert_eq!(parses_evler[0].root, "ev");
}

#[test]
fn test_verbal_morphology_analysis() {
    let morph = TurkishMorphology::new();

    // Past tense (geldi)
    let parses_geldi = morph.analyze("geldi");
    assert!(!parses_geldi.is_empty());
    assert_eq!(parses_geldi[0].root, "gel");
    assert!(parses_geldi.iter().any(|p| p.morpheme_tags.contains(&"Past".to_string())));

    // Progressive tense (geliyorum)
    let parses_geliyorum = morph.analyze("geliyorum");
    assert!(!parses_geliyorum.is_empty());
    assert_eq!(parses_geliyorum[0].root, "gel");
    assert!(parses_geliyorum.iter().any(|p| p.morpheme_tags.contains(&"Prog".to_string())));
}

#[test]
fn test_morphology_generation() {
    let generator = TurkishGenerator::new();

    // Noun generation with voicing and possessive
    let surface_kitap = generator.generate("kitap", &["Noun", "A3sg", "P1sg", "Dat"]);
    assert_eq!(surface_kitap, Some("kitabıma".to_string()));

    // Noun generation with vowel drop
    let surface_burun = generator.generate("burun", &["Noun", "A3sg", "P1sg"]);
    assert_eq!(surface_burun, Some("burnum".to_string()));

    // Noun generation with consonant doubling
    let surface_hak = generator.generate("hak", &["Noun", "A3sg", "P1sg"]);
    assert_eq!(surface_hak, Some("hakkım".to_string()));

    // Verb generation with progressive tense
    let surface_gel = generator.generate("gel", &["Verb", "Prog", "A1sg"]);
    assert_eq!(surface_gel, Some("geliyorum".to_string()));
}

#[test]
fn test_morphology_disambiguation() {
    let disambiguator = MorphologicalDisambiguator::new();
    let tokens = vec!["Ali", "güzel", "kitap", "okudu"];
    let disambiguated = disambiguator.disambiguate(&tokens);

    assert_eq!(disambiguated.len(), 4);
    assert_eq!(disambiguated[0].lemma, "ali");
    assert_eq!(disambiguated[1].primary_pos, pos::PrimaryPos::Adj);
    assert_eq!(disambiguated[2].primary_pos, pos::PrimaryPos::Noun);
    assert_eq!(disambiguated[3].primary_pos, pos::PrimaryPos::Verb);
}
