use akana_core::morphology::*;

#[test]
fn test_nominal_morphology_analysis() {
    let morph = TurkishMorphology::new();

    // 1. Root with Voicing (kitap -> kitaba, kitabım, kitabına, kitabımız)
    let parses_kitabim = morph.analyze("kitabım");
    assert!(!parses_kitabim.is_empty());
    assert_eq!(parses_kitabim[0].root, "kitap");
    assert!(parses_kitabim.iter().any(|p| p.morpheme_tags.contains(&"P1sg".to_string())));

    let parses_agaca = morph.analyze("ağaca");
    assert!(!parses_agaca.is_empty());
    assert_eq!(parses_agaca[0].root, "ağaç");
    assert!(parses_agaca.iter().any(|p| p.morpheme_tags.contains(&"Dat".to_string())));

    // 2. Root with Vowel Drop (burun -> burnum, akıl -> aklı, şehir -> şehre)
    let parses_burnum = morph.analyze("burnum");
    assert!(!parses_burnum.is_empty());
    assert_eq!(parses_burnum[0].root, "burun");

    let parses_sehre = morph.analyze("şehre");
    assert!(!parses_sehre.is_empty());
    assert_eq!(parses_sehre[0].root, "şehir");

    // 3. Root with Consonant Doubling (hak -> hakkım, his -> hissi, af -> affı)
    let parses_hakkim = morph.analyze("hakkım");
    assert!(!parses_hakkim.is_empty());
    assert_eq!(parses_hakkim[0].root, "hak");

    let parses_hissi = morph.analyze("hissi");
    assert!(!parses_hissi.is_empty());
    assert_eq!(parses_hissi[0].root, "his");

    // 4. Plural + Case + Possessive (evlerimizde, okullarımızdan)
    let parses_evler = morph.analyze("evlerimizde");
    assert!(!parses_evler.is_empty());
    assert_eq!(parses_evler[0].root, "ev");
    assert!(parses_evler.iter().any(|p| p.morpheme_tags.contains(&"A3pl".to_string())));
    assert!(parses_evler.iter().any(|p| p.morpheme_tags.contains(&"P1pl".to_string())));
    assert!(parses_evler.iter().any(|p| p.morpheme_tags.contains(&"Loc".to_string())));
}

#[test]
fn test_relative_clitic_ki() {
    let morph = TurkishMorphology::new();

    // Relative clitic -ki: evdeki, masadaki
    let parses_evdeki = morph.analyze("evdeki");
    assert!(!parses_evdeki.is_empty());
    assert_eq!(parses_evdeki[0].root, "ev");
    assert!(parses_evdeki.iter().any(|p| p.morpheme_tags.contains(&"RelClitic".to_string())));
}

#[test]
fn test_diminutives() {
    let morph = TurkishMorphology::new();

    // Diminutive: evcik, kedicik, küçücük
    let parses_evcik = morph.analyze("evcik");
    assert!(!parses_evcik.is_empty());
    assert_eq!(parses_evcik[0].root, "ev");
    assert!(parses_evcik.iter().any(|p| p.morpheme_tags.contains(&"Dim".to_string())));

    let parses_kucucuk = morph.analyze("küçücük");
    assert!(!parses_kucucuk.is_empty());
    assert_eq!(parses_kucucuk[0].root, "küçük");
}

#[test]
fn test_compound_decomposition() {
    let decomposer = CompoundDecomposer::new();

    let decomp_denizalti = decomposer.decompose("denizaltı");
    assert!(!decomp_denizalti.is_empty());
    assert_eq!(decomp_denizalti[0].part1, "deniz");
    assert_eq!(decomp_denizalti[0].part2, "altı");

    let decomp_akbaba = decomposer.decompose("akbaba");
    assert!(!decomp_akbaba.is_empty());
    assert_eq!(decomp_akbaba[0].part1, "ak");
    assert_eq!(decomp_akbaba[0].part2, "baba");
}

#[test]
fn test_verbal_morphology_analysis() {
    let morph = TurkishMorphology::new();

    // Past tense (geldi, gitti, baktı)
    let parses_geldi = morph.analyze("geldi");
    assert!(!parses_geldi.is_empty());
    assert_eq!(parses_geldi[0].root, "gel");
    assert!(parses_geldi.iter().any(|p| p.morpheme_tags.contains(&"Past".to_string())));

    // Progressive tense (geliyorum, gidiyor, bakıyor)
    let parses_geliyorum = morph.analyze("geliyorum");
    assert!(!parses_geliyorum.is_empty());
    assert_eq!(parses_geliyorum[0].root, "gel");
    assert!(parses_geliyorum.iter().any(|p| p.morpheme_tags.contains(&"Prog".to_string())));

    // Future tense (gelecek, yapacak)
    let parses_gelecek = morph.analyze("gelecek");
    assert!(!parses_gelecek.is_empty());
    assert_eq!(parses_gelecek[0].root, "gel");
    assert!(parses_gelecek.iter().any(|p| p.morpheme_tags.contains(&"Fut".to_string())));

    // Negation (gelmedi)
    let parses_gelmedi = morph.analyze("gelmedi");
    assert!(!parses_gelmedi.is_empty());
    assert_eq!(parses_gelmedi[0].root, "gel");
    assert!(parses_gelmedi.iter().any(|p| p.morpheme_tags.contains(&"Neg".to_string())));
}

#[test]
fn test_derivational_morphology() {
    let morph = TurkishMorphology::new();

    // Derivations: kitaplık, gözlük
    let parses_kitaplik = morph.analyze("kitaplık");
    assert!(!parses_kitaplik.is_empty());
    assert_eq!(parses_kitaplik[0].root, "kitap");
    assert!(parses_kitaplik.iter().any(|p| p.morpheme_tags.contains(&"Ness".to_string())));

    // Derivation with Agent (-ci): evci, yolcu
    let parses_yolcu = morph.analyze("yolcu");
    assert!(!parses_yolcu.is_empty());
    assert_eq!(parses_yolcu[0].root, "yol");

    // Verb -> Noun (-gi): sevgi, bilgi
    let parses_sevgi = morph.analyze("sevgi");
    assert!(!parses_sevgi.is_empty());
    assert_eq!(parses_sevgi[0].root, "sev");

    // Verb -> Adj (-gen): çalışkan
    let parses_caliskan = morph.analyze("çalışkan");
    assert!(!parses_caliskan.is_empty());
    assert_eq!(parses_caliskan[0].root, "çalış");
}

#[test]
fn test_dynamic_dictionary_loading() {
    let mut morph = TurkishMorphology::new();
    morph.load_dictionary_str("kuvars kuvars Noun\nblokzincir blokzincir Noun");

    let parses_kuvars = morph.analyze("kuvars");
    assert!(!parses_kuvars.is_empty());
    assert_eq!(parses_kuvars[0].root, "kuvars");

    let parses_blok = morph.analyze("blokzincir");
    assert!(!parses_blok.is_empty());
    assert_eq!(parses_blok[0].root, "blokzincir");
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

    let surface_sehir = generator.generate("şehir", &["Noun", "A3sg", "Dat"]);
    assert_eq!(surface_sehir, Some("şehre".to_string()));

    // Noun generation with consonant doubling
    let surface_hak = generator.generate("hak", &["Noun", "A3sg", "P1sg"]);
    assert_eq!(surface_hak, Some("hakkım".to_string()));

    // Verb generation with progressive tense
    let surface_gel = generator.generate("gel", &["Verb", "Prog", "A1sg"]);
    assert_eq!(surface_gel, Some("geliyorum".to_string()));

    // Verb generation with future tense and softening
    let surface_git = generator.generate("git", &["Verb", "Fut", "A1sg"]);
    assert_eq!(surface_git, Some("gideceğim".to_string()));
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
