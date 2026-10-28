use akana_core::morphology::*;

#[test]
fn test_nominal_morphology_analysis() {
    let morph = TurkishMorphology::new();

    // 1. Root with Voicing (kitap -> kitaba, kitabım, kitabına, kitabımız)
    let parses_kitabim = morph.analyze("kitabım");
    assert!(!parses_kitabim.is_empty());
    assert!(parses_kitabim
        .iter()
        .any(|p| p.root == "kitap" && p.morpheme_tags.contains(&"P1sg".to_string())));

    let parses_agaca = morph.analyze("ağaca");
    assert!(!parses_agaca.is_empty());
    assert!(parses_agaca
        .iter()
        .any(|p| p.root == "ağaç" && p.morpheme_tags.contains(&"Dat".to_string())));

    // 2. Root with Vowel Drop (burun -> burnum, akıl -> aklı, şehir -> şehre)
    let parses_burnum = morph.analyze("burnum");
    assert!(!parses_burnum.is_empty());
    assert!(parses_burnum.iter().any(|p| p.root == "burun"));

    let parses_sehre = morph.analyze("şehre");
    assert!(!parses_sehre.is_empty());
    assert!(parses_sehre.iter().any(|p| p.root == "şehir"));

    // 3. Root with Consonant Doubling (hak -> hakkım, his -> hissi, af -> affı)
    let parses_hakkim = morph.analyze("hakkım");
    assert!(!parses_hakkim.is_empty());
    assert!(parses_hakkim.iter().any(|p| p.root == "hak"));

    let parses_hissi = morph.analyze("hissi");
    assert!(!parses_hissi.is_empty());
    assert!(parses_hissi.iter().any(|p| p.root == "his"));

    // 4. Plural + Case + Possessive (evlerimizde, okullarımızdan)
    let parses_evler = morph.analyze("evlerimizde");
    assert!(!parses_evler.is_empty());
    assert!(parses_evler.iter().any(|p| p.root == "ev"
        && p.morpheme_tags.contains(&"A3pl".to_string())
        && p.morpheme_tags.contains(&"Loc".to_string())));
}

#[test]
fn test_relative_clitic_ki() {
    let morph = TurkishMorphology::new();

    // Relative clitic -ki: evdeki, masadaki
    let parses_evdeki = morph.analyze("evdeki");
    assert!(!parses_evdeki.is_empty());
    assert!(parses_evdeki
        .iter()
        .any(|p| p.root == "ev" && p.morpheme_tags.contains(&"RelClitic".to_string())));
}

#[test]
fn test_diminutives() {
    let morph = TurkishMorphology::new();

    // Diminutive: evcik, kedicik, küçücük
    let parses_evcik = morph.analyze("evcik");
    assert!(!parses_evcik.is_empty());
    assert!(parses_evcik
        .iter()
        .any(|p| p.root == "ev" && p.morpheme_tags.contains(&"Dim".to_string())));

    let parses_kucucuk = morph.analyze("küçücük");
    assert!(!parses_kucucuk.is_empty());
    assert!(parses_kucucuk
        .iter()
        .any(|p| p.root == "küçük" || p.root == "küçücük"));
}

#[test]
fn test_compound_decomposition() {
    let decomposer = CompoundDecomposer::new();

    let decomp_denizalti = decomposer.decompose("denizaltı");
    assert!(!decomp_denizalti.is_empty());
    assert_eq!(decomp_denizalti[0].part1, "deniz");
    assert_eq!(decomp_denizalti[0].part2, "altı");

    // A productive pattern on lexicon roots, through the word's inflection.
    let decomp_basbakan = decomposer.decompose("başbakanlar");
    assert!(!decomp_basbakan.is_empty());
    assert_eq!(decomp_basbakan[0].part1, "baş");
    assert_eq!(decomp_basbakan[0].part2, "bakanlar");

    // Words whose halves merely happen to be words are not compounds (mah + keme, hare + ket).
    let morphology = TurkishMorphology::new();
    for word in [
        "mahkeme",
        "yüksek",
        "sistem",
        "kültür",
        "matematik",
        "kapsam",
        "toplam",
        "hareket",
        "dikkat",
        "temmuz",
        "kaybet",
        "başkan",
        // Compounds that are words of their own, whose parts mislead (review 2 MSTAB-07).
        "anayasa",
        "anavatan",
        "cephane",
        "gözaltı",
        "olağanüstü",
        "hanımeli",
    ] {
        assert!(decomposer.decompose(word).is_empty(), "{word} decomposed");
        assert_eq!(
            CompoundDecomposer::split_lexical_compound(&morphology, word),
            None,
            "{word} split"
        );
    }
    for (word, first, head) in [
        ("cumhurbaşkanı", "cumhur", "başkan"),
        ("zeytinyağı", "zeytin", "yağ"),
        ("denizaltı", "deniz", "alt"),
        ("başsavcı", "baş", "savcı"),
        ("anabilim", "ana", "bilim"),
        // Compounds the lexicon does not mark (review 2 RANK2-12).
        ("bilgisayar", "bilgi", "sayar"),
        ("kahverengi", "kahve", "renk"),
        ("ayakkabı", "ayak", "kap"),
        ("bilirkişi", "bilir", "kişi"),
        ("ateşkes", "ateş", "kes"),
        ("tebliğname", "tebliğ", "name"),
        ("huzurevi", "huzur", "ev"),
    ] {
        assert_eq!(
            CompoundDecomposer::split_lexical_compound(&morphology, word),
            Some((first.to_string(), head.to_string())),
            "{word}"
        );
    }
}

#[test]
fn test_verbal_morphology_analysis() {
    let morph = TurkishMorphology::new();

    // Past tense (geldi, gitti, baktı)
    let parses_geldi = morph.analyze("geldi");
    assert!(!parses_geldi.is_empty());
    assert_eq!(parses_geldi[0].root, "gel");
    assert!(parses_geldi
        .iter()
        .any(|p| p.morpheme_tags.contains(&"Past".to_string())));

    // Progressive tense (geliyorum, gidiyor, bakıyor)
    let parses_geliyorum = morph.analyze("geliyorum");
    assert!(!parses_geliyorum.is_empty());
    assert_eq!(parses_geliyorum[0].root, "gel");
    assert!(parses_geliyorum
        .iter()
        .any(|p| p.morpheme_tags.contains(&"Prog".to_string())));

    // Future tense (gelecek, yapacak)
    let parses_gelecek = morph.analyze("gelecek");
    assert!(!parses_gelecek.is_empty());
    assert!(parses_gelecek
        .iter()
        .any(|p| p.root == "gel" && p.morpheme_tags.contains(&"Fut".to_string())));

    // Negation (gelmedi)
    let parses_gelmedi = morph.analyze("gelmedi");
    assert!(!parses_gelmedi.is_empty());
    assert!(parses_gelmedi
        .iter()
        .any(|p| p.root == "gel" && p.morpheme_tags.contains(&"Neg".to_string())));
}

#[test]
fn test_derivational_morphology() {
    let morph = TurkishMorphology::new();

    // Derivations: kitaplık, gözlük
    let parses_kitaplik = morph.analyze("kitaplık");
    assert!(!parses_kitaplik.is_empty());
    assert!(parses_kitaplik
        .iter()
        .any(|p| p.root == "kitap" && p.morpheme_tags.contains(&"Ness".to_string())));

    // Derivation with Agent (-ci): evci, yolcu
    let parses_yolcu = morph.analyze("yolcu");
    assert!(!parses_yolcu.is_empty());
    assert!(parses_yolcu.iter().any(|p| p.root == "yol"));

    // -gI and -GAn are not productive (as in Zemberek): sevgi and çalışkan are lexicon words.
    let parses_sevgi = morph.analyze("sevgi");
    assert!(parses_sevgi.iter().any(|p| p.root == "sevgi"));
    assert!(!parses_sevgi.iter().any(|p| p.root == "sev"));
    let parses_caliskan = morph.analyze("çalışkan");
    assert!(parses_caliskan.iter().any(|p| p.root == "çalışkan"));
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

/// Review 2 RANK2-06 and MORPH2-10: names and abbreviations, the two progressives, the stems
/// that change, compounds and the spelling of tags.
#[test]
fn generation_writes_names_progressives_and_compounds() {
    let morph = TurkishMorphology::new();
    let generate = |word: &str, tags: &str| {
        let tags: Vec<&str> = tags.split('+').collect();
        morph.generate(word, &tags)
    };
    for (word, tags, expected) in [
        // A name's case, possessive and copula after an apostrophe, its plural and
        // derivations without one; an abbreviation's suffixes follow its pronunciation.
        ("Türkiye", "Noun+Loc", Some("Türkiye'de")),
        ("türkiye", "Noun+Loc", Some("Türkiye'de")),
        ("Ankara", "Noun+Dat", Some("Ankara'ya")),
        ("Ali", "Noun+Gen", Some("Ali'nin")),
        ("Ali", "Noun+P3sg", Some("Ali'si")),
        ("Ali", "Noun+CopPres3sg", Some("Ali'dir")),
        ("Ahmet", "Noun+A3pl+Dat", Some("Ahmetlere")),
        ("Ankara", "Noun+With", Some("Ankaralı")),
        ("Türkçe", "Noun+Gen", Some("Türkçenin")),
        ("Türkiye", "Noun", Some("Türkiye")),
        ("TBMM", "Noun+Loc", Some("TBMM'de")),
        ("TL", "Noun+P3sg", Some("TL'si")),
        ("HMK", "Noun+Gen", Some("HMK'nın")),
        // The abbreviation when it is asked for in capitals, the common noun otherwise.
        ("ABD", "Noun+Gen", Some("ABD'nin")),
        ("abd", "Noun+Gen", Some("abdın")),
        // Prog is -Iyor, also on a stem that changes and on de-/ye- (review 3 MORPH3-07),
        // but -mAktA before the copula -DIr, as written Turkish has it; Prog1 and Prog2 ask for
        // one of the two.
        ("git", "Verb+Prog+A1sg", Some("gidiyorum")),
        ("et", "Verb+Prog", Some("ediyor")),
        ("tat", "Verb+Prog", Some("tadıyor")),
        ("demek", "Verb+Prog", Some("diyor")),
        ("yemek", "Verb+Prog+A3pl", Some("yiyorlar")),
        ("git", "Verb+Prog2+A1sg", Some("gitmekteyim")),
        ("bulunmak", "Verb+Prog+CopPres3sg", Some("bulunmaktadır")),
        ("bulunmak", "Verb+Prog1+CopPres3sg", Some("bulunuyordur")),
        ("gelmek", "Verb+Prog+CopPast", Some("geliyordu")),
        ("bulunmak", "Verb+Prog2+CopPres3sg", Some("bulunmaktadır")),
        ("gerekmek", "Verb+Prog2+CopPres3sg", Some("gerekmektedir")),
        ("gel", "Verb+Prog1+A1sg", Some("geliyorum")),
        // The special stems: de/ye, and the root as spelt before içer-/bur-.
        ("de", "Verb+AdvIp", Some("deyip")),
        ("de", "Verb+AdvInce", Some("deyince")),
        ("ye", "Verb+AdvInce", Some("yiyince")),
        ("yukarı", "Noun+Loc", Some("yukarıda")),
        ("ben", "Pron+A1sg+Dat", Some("bana")),
        // A CompoundP3sg noun derives and pluralises on its stem without the possessive.
        ("cumhurbaşkanı", "Noun+Ness", Some("cumhurbaşkanlık")),
        ("cumhurbaşkanı", "Noun+A3pl+Dat", Some("cumhurbaşkanlarına")),
        ("cumhurbaşkanı", "Noun+Dat", Some("cumhurbaşkanına")),
        ("zeytinyağı", "Noun+With", Some("zeytinyağlı")),
        // Tags in any case; an unknown tag is no word.
        ("kitap", "noun+a3sg+p1sg+dat", Some("kitabıma")),
        ("kitap", "NOUN+P1SG+DAT", Some("kitabıma")),
        ("kitap", "Noun+Foo", None),
        // The spellings analyze reports (display_lemma) are folded as analyze folds words
        // (review 3 MORPH3-07, BIND3-06).
        ("kâğıt", "Noun+Loc", Some("kağıtta")),
        ("hâkim", "Noun+Acc", Some("hakimi")),
        ("e-posta", "Noun+Dat", Some("epostaya")),
        ("Hakkâri", "Noun+Loc", Some("Hakkâri'de")),
        // An acronym as display_lemma spells it, with a Latin I (review 4 BIND4-05).
        ("FIFA", "Noun+Dat", Some("FIFA'ya")),
        ("IBAN", "Noun+Gen", Some("IBAN'ın")),
        ("ışık", "Noun+Loc", Some("ışıkta")),
        // Review 4 MORPH4-08: an overt dative, a pronoun's full instrumental, the item spelt
        // as asked, the request's capitals.
        ("karşı", "Noun+Dat", Some("karşıya")),
        ("bu", "Pron+Ins", Some("bununla")),
        ("o", "Pron+Ins", Some("onunla")),
        ("ben", "Pron+A1sg+Ins", Some("benimle")),
        ("dâhil", "Noun+P3sg+Loc", Some("dahilinde")),
        ("NATO", "Noun+Gen", Some("NATO'nun")),
        ("Yönetmelik", "Noun+Gen", Some("Yönetmeliğin")),
        ("KİTAP", "Noun+Dat", Some("KİTABA")),
    ] {
        assert_eq!(generate(word, tags).as_deref(), expected, "{word} {tags}");
    }
    // The full stem of a compound takes no derivation.
    for word in ["cumhurbaşkanılık", "cumhurbaşkanılığı", "zeytinyağılı"] {
        assert!(morph.analyze(word).is_empty(), "{word}");
    }
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

#[test]
fn test_frequency_prior_ranking() {
    // The embedded corpus prior ranks each word's readings; without it these come out in the
    // analyzer's tie order or worse.
    let morphology = TurkishMorphology::new();
    let best = |word: &str| morphology.analyze(word)[0].formatted.clone();
    // Third person over the second, which court text almost never uses.
    assert_eq!(best("kararının"), "[karar:Noun] karar:Noun+P3sg+Gen");
    assert_eq!(best("kişilerin"), "[kişi:Noun] kişi:Noun+A3pl+Gen");
    // The common item over its rare homograph (hak the noun, not the adjective).
    assert_eq!(best("hakkı"), "[hak:Noun] hak:Noun+P3sg");
    // A passive participle over a derived noun (düzenle-n-en, not düzen-len-en).
    assert_eq!(
        best("düzenlenen"),
        "[düzenlemek:Verb] düzenle:Verb+Pass+PresPart"
    );
    // Case is evidence, weighed against how often each reading is meant: capitalised Ali is
    // the name, but a capitalised Yani (the start of a sentence) is still the conjunction.
    assert_eq!(
        morphology.analyze("Ali")[0].secondary_pos,
        SecondaryPos::ProperNoun
    );
    assert_eq!(best("Yani"), "[yani:Conj] yani:Conj");
    // Without the prior, readings keep the spelling factor alone.
    let flat = TurkishMorphology::new().with_prior(false);
    assert!(flat.analyze("kişilerin").iter().all(|p| p.score >= 0.9));
}

#[test]
fn prior_was_estimated_for_this_lexicon_and_graph() {
    // prior.tsv.gz records the lexicon and morphotactic graph it was estimated for. After a
    // change to either, re-run examples/morph_prior.rs (see its docs), or the prior's keys
    // stop describing the analyzer's readings.
    assert_eq!(
        akana_core::morphology::embedded_prior_inventory(),
        akana_core::morphology::prior_inventory()
    );
}

#[test]
fn homographs_have_their_own_prior_keys() {
    let morphology = TurkishMorphology::new();
    // ay "moon" and ay "month" (the lexicon's second ay, idx=1) are separate items.
    let keys: std::collections::HashSet<String> = morphology
        .analyze("aya")
        .iter()
        .filter(|p| p.lemma == "ay" && p.primary_pos == PrimaryPos::Noun)
        .map(akana_core::morphology::prior_item_key)
        .collect();
    assert!(keys.contains("ay ay Noun -"), "{keys:?}");
    assert!(keys.contains("ay ay Noun Time #1"), "{keys:?}");
}

/// Review 2 MORPH2-12: user dictionaries also take Zemberek's syntax, as its dictionaries and
/// akana's supplements write it, and the loader returns the lines it cannot read.
#[test]
fn dictionary_lines_in_zemberek_syntax_and_rejected_lines() {
    let mut morph = TurkishMorphology::new();
    let rejected = morph.load_dictionary_str(
        "kriptopara [P:Noun; A:CompoundP3sg; Roots:kripto-para]\n\
         foobarlamak\n\
         fıstıkçap [A:Voicing]\n\
         XYZK [P:Abbrv; Pr:iksiyezetka]\n\
         bozuk [P:Foo]\n\
         eksik satır\n\
         acik [P:Noun\n",
    );
    assert_eq!(rejected, ["bozuk [P:Foo]", "eksik satır", "acik [P:Noun"]);
    assert!(morph
        .analyze("foobarladı")
        .iter()
        .any(|p| p.lemma == "foobarlamak" && p.primary_pos == PrimaryPos::Verb));
    assert!(morph
        .analyze("fıstıkçabı")
        .iter()
        .any(|p| p.root == "fıstıkçap" && p.morpheme_tags.contains(&"Acc".to_string())));
    assert!(morph
        .analyze("kriptoparası")
        .iter()
        .any(|p| p.root == "kriptopara"));
    assert!(morph
        .analyze("XYZK'ya")
        .iter()
        .any(|p| p.secondary_pos == SecondaryPos::Abbreviation));
    // No line of the built-in lexicon is rejected.
    let mut lexicon = RootLexicon::empty();
    let builtin = include_str!("../src/morphology/zemberek_lexicon.txt");
    assert!(lexicon.load_from_str(builtin).is_empty());
}

/// Review 2 MORPH-09: the fullwidth apostrophe of East Asian keyboards is an apostrophe.
#[test]
fn fullwidth_apostrophe() {
    let morph = TurkishMorphology::new();
    assert!(morph
        .analyze("Ankara＇da")
        .iter()
        .any(|p| p.lemma == "ankara" && p.morpheme_tags.contains(&"Loc".to_string())));
}

/// Review 2 PARSER2-02: one inventory of non-finite verb forms (verbal nouns, participles,
/// converbs), the full set the analyzer emits, which the parser, readability and the syntactic
/// analyzer share.
#[test]
fn non_finite_verb_forms_come_from_one_inventory() {
    let morph = TurkishMorphology::new();
    let forms = |word: &str| -> Vec<(&'static str, VerbForm)> {
        morph
            .analyze(word)
            .iter()
            .filter_map(verb_form)
            .map(|f| (f.tag, f.form))
            .collect()
    };
    for (word, tag, form) in [
        ("gelmek", "Inf", VerbForm::Vnoun),
        ("gelince", "AdvInce", VerbForm::Conv),
        ("gelip", "AdvIp", VerbForm::Conv),
        ("gelirken", "While", VerbForm::Conv),
        ("güldükçe", "AsLongAs", VerbForm::Conv),
        ("gelircesine", "AsIf", VerbForm::Conv),
        ("gelemeden", "WithoutBeingAbleToHaveDoneSo", VerbForm::Conv),
        ("gelen", "PresPart", VerbForm::Part),
        ("gelmiş", "NarrPart", VerbForm::Part),
        ("okunmaz", "AorPart", VerbForm::Part),
        ("geldiği", "PastPart", VerbForm::Vnoun),
        ("geldiği", "PastPart", VerbForm::Part),
    ] {
        assert!(
            forms(word).contains(&(tag, form)),
            "{word}: {:?}",
            forms(word)
        );
    }
    // -(y)ken after a noun or adjective, -cA on an adjective, and a verb form made an ordinary
    // word again are no verb forms.
    for word in ["çocukken", "hastayken", "büyükçe", "okuyucu", "bilinmezlik"] {
        assert!(forms(word).is_empty(), "{word}: {:?}", forms(word));
    }
}
