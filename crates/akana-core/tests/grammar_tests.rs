use akana_core::grammar::{ErrorCategory, GrammarRule, RuleSet, TurkishGrammarChecker};

#[test]
fn test_grammar_checker_initialization() {
    let checker = TurkishGrammarChecker::new();
    let res = checker.check("Bugün hava çok güzel.");
    assert!(res.findings.is_empty());
    assert_eq!(res.corrected, "Bugün hava çok güzel.");
}

#[test]
fn test_clitic_de_da() {
    let checker = TurkishGrammarChecker::new();

    // 1. Separate "te" / "ta" error
    let res = checker.check("Ali de geldi, Veli te geldi.");
    assert!(res
        .findings
        .iter()
        .any(|f| f.category == ErrorCategory::CliticDeDa));
    assert_eq!(res.corrected, "Ali de geldi, Veli de geldi.");

    // 2. Attached de/da on finite verb
    let res2 = checker.check("Gittide geri dönmedi.");
    assert!(res2
        .findings
        .iter()
        .any(|f| f.category == ErrorCategory::CliticDeDa));
    assert_eq!(res2.corrected, "Gitti de geri dönmedi.");
}

#[test]
fn test_clitic_ki() {
    let checker = TurkishGrammarChecker::new();

    // 1. Attached ki on verb
    let res = checker.check("Duydumki unutmuşsun gözlerimin rengini.");
    assert!(res
        .findings
        .iter()
        .any(|f| f.category == ErrorCategory::CliticKi));
    assert_eq!(res.corrected, "Duydum ki unutmuşsun gözlerimin rengini.");

    // 2. SOMBAHÇEMİ exception separated
    let res2 = checker.check("Madem ki bilmiyorsun, neden konuşuyorsun?");
    assert!(res2
        .findings
        .iter()
        .any(|f| f.category == ErrorCategory::CliticKi));
    assert_eq!(res2.corrected, "Mademki bilmiyorsun, neden konuşuyorsun?");
}

#[test]
fn test_particle_mi() {
    let checker = TurkishGrammarChecker::new();

    // 1. Merged question particle
    let res = checker.check("Geldinmi dün akşam?");
    assert!(res
        .findings
        .iter()
        .any(|f| f.category == ErrorCategory::ParticleMi));
    assert_eq!(res.corrected, "Geldin mi dün akşam?");

    // 2. Merged with person suffix
    let res2 = checker.check("Biliyormusun bu şarkıyı?");
    assert!(res2
        .findings
        .iter()
        .any(|f| f.category == ErrorCategory::ParticleMi));
    assert_eq!(res2.corrected, "Biliyor musun bu şarkıyı?");

    // 3. Standalone vowel harmony violation
    let res3 = checker.check("Sen de gördün mı onu?");
    assert!(res3
        .findings
        .iter()
        .any(|f| f.category == ErrorCategory::ParticleMi));
    assert_eq!(res3.corrected, "Sen de gördün mü onu?");
}

#[test]
fn test_consonant_assimilation() {
    let checker = TurkishGrammarChecker::new();

    // kitapda -> kitapta
    let res = checker.check("Kitapda ilginç bilgiler var.");
    assert!(res
        .findings
        .iter()
        .any(|f| f.category == ErrorCategory::ConsonantAssimilation));
    assert_eq!(res.corrected, "Kitapta ilginç bilgiler var.");

    // sokakdan -> sokaktan
    let res2 = checker.check("Sokakdan sesler geliyor.");
    assert!(res2
        .findings
        .iter()
        .any(|f| f.category == ErrorCategory::ConsonantAssimilation));
    assert_eq!(res2.corrected, "Sokaktan sesler geliyor.");
}

#[test]
fn test_vowel_drop() {
    let checker = TurkishGrammarChecker::new();

    // oğulu -> oğlu
    let res = checker.check("Onun oğulu okula gitti.");
    assert!(res
        .findings
        .iter()
        .any(|f| f.category == ErrorCategory::VowelDropping));
    assert_eq!(res.corrected, "Onun oğlu okula gitti.");

    // akılı also reads as akı-lı, burunu as bur-un-u, alınan as a verb, hükümet has its own
    // root: left alone
    for text in [
        "Onun akılı çok karışık.",
        "Alınan karar.",
        "Hükümet toplandı.",
    ] {
        assert_eq!(checker.correct(text), text);
    }

    // şehire -> şehre
    let res2 = checker.check("Yarın şehire gideceğiz.");
    assert!(res2
        .findings
        .iter()
        .any(|f| f.category == ErrorCategory::VowelDropping));
    assert_eq!(res2.corrected, "Yarın şehre gideceğiz.");
}

#[test]
fn test_consonant_softening() {
    let checker = TurkishGrammarChecker::new();

    // kitapı -> kitabı
    let res = checker.check("Bu kitapı mutlaka okumalısın.");
    assert!(res
        .findings
        .iter()
        .any(|f| f.category == ErrorCategory::ConsonantSoftening));
    assert_eq!(res.corrected, "Bu kitabı mutlaka okumalısın.");
}

#[test]
fn test_apostrophe_proper_nouns_and_numbers() {
    let checker = TurkishGrammarChecker::new();

    // Ahmetler'in -> Ahmetlerin (plural suffix on proper noun must not have apostrophe); the
    // rule is experimental, as names ending in -ler keep it (Silahlı Kuvvetler'in)
    let experimental = TurkishGrammarChecker::with_rules(RuleSet::default().with_experimental());
    let res = experimental.check("Ahmetler'in evi buraya çok yakın.");
    assert!(res
        .findings
        .iter()
        .any(|f| f.category == ErrorCategory::ApostropheProperNoun));
    assert_eq!(res.corrected, "Ahmetlerin evi buraya çok yakın.");

    // 2.'nci -> 2'nci (double ordinal)
    let res2 = checker.check("Yarışmada 2.'nci oldu.");
    assert!(res2
        .findings
        .iter()
        .any(|f| f.category == ErrorCategory::ApostropheNumberDate));
    assert_eq!(res2.corrected, "Yarışmada 2'nci oldu.");

    // 1923'de -> 1923'te (consonant assimilation on number)
    let res3 = checker.check("Cumhuriyet 1923'de kuruldu.");
    assert!(res3
        .findings
        .iter()
        .any(|f| f.category == ErrorCategory::ApostropheNumberDate));
    assert_eq!(res3.corrected, "Cumhuriyet 1923'te kuruldu.");
}

#[test]
fn test_reduplications() {
    let checker = TurkishGrammarChecker::new();

    // elele -> el ele
    let res = checker.check("Çocuklar elele yürüyorlardı.");
    assert!(res
        .findings
        .iter()
        .any(|f| f.category == ErrorCategory::ReduplicationOrthography));
    assert_eq!(res.corrected, "Çocuklar el ele yürüyorlardı.");

    // yanyana -> yan yana
    let res2 = checker.check("İki bina yanyana inşa edilmiş.");
    assert!(res2
        .findings
        .iter()
        .any(|f| f.category == ErrorCategory::ReduplicationOrthography));
    assert_eq!(res2.corrected, "İki bina yan yana inşa edilmiş.");
}

#[test]
fn test_quantity_plural_agreement() {
    // experimental: it has no animacy or idiom knowledge, so it is off by default
    let checker = TurkishGrammarChecker::with_rules(RuleSet::default().with_experimental());
    assert_eq!(
        TurkishGrammarChecker::new().correct("Pazardan üç elmalar aldım."),
        "Pazardan üç elmalar aldım."
    );

    // üç elmalar -> üç elma
    let res = checker.check("Pazardan üç elmalar aldım.");
    assert!(res
        .findings
        .iter()
        .any(|f| f.category == ErrorCategory::QuantityPluralClash));
    assert_eq!(res.corrected, "Pazardan üç elma aldım.");

    // birçok insanlar -> birçok insan
    let res2 = checker.check("Mitinge birçok insanlar katıldı.");
    assert!(res2
        .findings
        .iter()
        .any(|f| f.category == ErrorCategory::QuantityPluralClash));
    assert_eq!(res2.corrected, "Mitinge birçok insan katıldı.");
}

#[test]
fn test_compound_verbs() {
    let checker = TurkishGrammarChecker::new();

    // terketmek -> terk etmek
    let res = checker.check("Evi terketmek zorunda kaldı.");
    assert!(res
        .findings
        .iter()
        .any(|f| f.category == ErrorCategory::CompoundWordOrthography));
    assert_eq!(res.corrected, "Evi terk etmek zorunda kaldı.");

    // hiss etmek -> hissetmek
    let res2 = checker.check("Kendini çok iyi hiss etti.");
    assert!(res2
        .findings
        .iter()
        .any(|f| f.category == ErrorCategory::CompoundWordOrthography));
    assert_eq!(res2.corrected, "Kendini çok iyi hissetti.");
}

#[test]
fn test_tautology_redundancy() {
    let checker = TurkishGrammarChecker::new();

    // henüz hala -> henüz
    let res = checker.check("O henüz hala buraya gelmedi.");
    assert!(res
        .findings
        .iter()
        .any(|f| f.category == ErrorCategory::TautologyRedundancy));
    assert_eq!(res.corrected, "O henüz buraya gelmedi.");

    // birlikte beraber -> birlikte
    let res2 = checker.check("Onlar birlikte beraber çalışıyorlar.");
    assert!(res2
        .findings
        .iter()
        .any(|f| f.category == ErrorCategory::TautologyRedundancy));
    assert_eq!(res2.corrected, "Onlar birlikte çalışıyorlar.");
}

/// Correct sentences the rules used to rewrite (the 2026-10-20 review, GN-01 to GN-19): the
/// default checker leaves them alone.
#[test]
fn default_rules_leave_correct_text_alone() {
    let checker = TurkishGrammarChecker::new();
    for text in [
        "Mahkeme önceki kararda direnmiştir.",
        "Davacı tahliye tehdidi altında bulunmaktadır.",
        "Toplantıda karar alındığında tutanak düzenlenir.",
        "Mahkemenin hükmü kesinleşti.",
        "Filmi yirmi kişi izledi.",
        "Cumhuriyet 1923'te kuruldu.",
        "Saat 5'te buluşalım.",
        "Hükümet yeni bir karar aldı.",
        "Bu karar alınan önlemlere dayanır.",
        "Yetki ve tepki arasında bir ilişki var.",
        "Sözleşmedeki hükümler geçerlidir.",
        "Ona bile söylemedim.",
        "Durum mahkemeye bildirdi.",
        "Dosyada bilirkişi raporu var.",
        "Türkçe bilen kişiler başvurabilir.",
        "Ali'nin evi Bolu'da.",
        "Hitler'in ve Kepler'in adı geçti.",
        "Kılıçdaroğlu'nun konuşması uzundu.",
        "STK'lar ve CHP'li vekiller toplandı.",
        "Maç 2-1'lik skorla bitti.",
        "Takdir hakkı mahkemeye aittir.",
        "Bunun için de çalıştık.",
        "Kitap bende kaldı.",
        "Saat birde buluşalım.",
        "Bir şeyler oldu.",
        "Beş dolar verdi.",
        "Benim için önemli.",
        "Kendi özel hayatına karışma.",
        "Kalabalık bir takım halinde yürüdü.",
        "Davacı tarafın da kabulünde olduğu üzere süre geçmiştir.",
        "1.000'de bir görülen bir durum.",
    ] {
        let res = checker.check(text);
        assert!(
            res.findings.is_empty(),
            "{text:?}: {:?}",
            res.findings
                .iter()
                .map(|f| (f.rule.id(), &f.original_text, &f.replacement))
                .collect::<Vec<_>>()
        );
        assert_eq!(res.corrected, text);
    }
}

#[test]
fn number_suffix_follows_how_the_number_is_read() {
    let checker = TurkishGrammarChecker::new();
    for (text, fixed) in [
        ("Cumhuriyet 1923'de kuruldu.", "Cumhuriyet 1923'te kuruldu."),
        ("Cumhuriyet 1923'da kuruldu.", "Cumhuriyet 1923'te kuruldu."),
        ("Savaş 1940'da başladı.", "Savaş 1940'ta başladı."),
        ("Darbe 1960'da oldu.", "Darbe 1960'ta oldu."),
        ("Bu 1990'ta oldu.", "Bu 1990'da oldu."),
        ("Saat 5'den sonra gel.", "Saat 5'ten sonra gel."),
        (
            "Karar 15.03.1993'de verildi.",
            "Karar 15.03.1993'te verildi.",
        ),
        ("Sayfa 100'te bitiyor.", "Sayfa 100'de bitiyor."),
    ] {
        assert_eq!(checker.correct(text), fixed, "{text}");
    }
}

#[test]
fn corrections_keep_the_case() {
    let checker = TurkishGrammarChecker::new();
    assert_eq!(checker.correct("Yada gelme."), "Ya da gelme.");
    assert_eq!(checker.correct("KİTAPDA YAZIYOR."), "KİTAPTA YAZIYOR.");
    assert_eq!(checker.correct("Sağol kardeşim."), "Sağ ol kardeşim.");
    assert_eq!(
        checker.correct("iPhone satışları arttı."),
        "iPhone satışları arttı."
    );
    assert_eq!(checker.correct("Elele verdiler."), "El ele verdiler.");
}

#[test]
fn question_particle_harmony_reads_the_lexicon() {
    // saat, gol and kalp take front-vowel suffixes (InverseHarmony): "Saat mi", not "Saat mı"
    let checker = TurkishGrammarChecker::new();
    for text in ["Saat mi geldi?", "Gol mü attı?", "Kalp mi ağrıyor?"] {
        assert_eq!(checker.correct(text), text);
    }
    assert_eq!(checker.correct("Gitti mı?"), "Gitti mi?");
}

#[test]
fn proper_noun_suffix_apostrophe() {
    let checker = TurkishGrammarChecker::new();
    assert_eq!(
        checker.correct("Ankara'lı bir öğrenci."),
        "Ankaralı bir öğrenci."
    );
    assert_eq!(checker.correct("Türk'ler geldi."), "Türkler geldi.");
    // the plural of a name before the apostrophe is experimental (Silahlı Kuvvetler'in, a
    // name ending in -ler, keeps it)
    assert_eq!(checker.correct("Ahmetler'in evi."), "Ahmetler'in evi.");
    let all = RuleSet::empty().with(GrammarRule::ApostrophePluralName);
    assert_eq!(
        checker.correct_with_rules("Ahmetler'in evi.", all),
        "Ahmetlerin evi."
    );
}

#[test]
fn rules_can_be_chosen() {
    let mut checker = TurkishGrammarChecker::new();
    let text = "Ali de geldi, Veli te geldi.";
    checker.disable_rule(GrammarRule::DeDaHardened);
    assert_eq!(checker.correct(text), text);
    checker.enable_rule(GrammarRule::DeDaHardened);
    assert_eq!(checker.correct(text), "Ali de geldi, Veli de geldi.");

    let only = RuleSet::from_ids(["de-da-hardened"]).unwrap();
    let res = checker.check_with_rules("veli te geldi.", only);
    assert_eq!(
        res.findings.iter().map(|f| f.rule).collect::<Vec<_>>(),
        [GrammarRule::DeDaHardened]
    );
    assert_eq!(
        res.findings[0].confidence,
        GrammarRule::DeDaHardened.confidence()
    );
    assert_eq!(
        serde_json::to_value(&res.findings[0]).unwrap()["rule"],
        "de-da-hardened"
    );
    assert!(checker
        .check_with_rules(text, RuleSet::empty())
        .findings
        .is_empty());
}

#[test]
fn findings_of_0_8_0_still_deserialize() {
    use akana_core::grammar::GrammarFinding;
    let checker = TurkishGrammarChecker::new();
    let result = checker.check("Veli te geldi.");
    let found = result
        .findings
        .iter()
        .find(|f| f.category == ErrorCategory::CliticDeDa)
        .expect("a de/da finding");
    let json = serde_json::to_value(found).unwrap();
    let back: GrammarFinding = serde_json::from_value(json.clone()).unwrap();
    assert_eq!(back.rule, found.rule);
    // 0.8.0 wrote no rule: the finding takes the first rule of its category.
    let mut old = json;
    old.as_object_mut().unwrap().remove("rule");
    let back: GrammarFinding = serde_json::from_value(old.clone()).unwrap();
    assert_eq!(back.category, ErrorCategory::CliticDeDa);
    assert_eq!(back.rule.category(), ErrorCategory::CliticDeDa);
    // No rule reports SpellingTypo, so such a finding cannot have come from 0.8.0.
    old["category"] = "SpellingTypo".into();
    assert!(serde_json::from_value::<GrammarFinding>(old).is_err());
}

#[test]
fn sentence_level_rules_stay_in_their_sentence() {
    let checker = TurkishGrammarChecker::with_rules(RuleSet::all());
    // the two "ne" are in different sentences, and the first is a question word
    let text = "Ne yapacağımı bilmiyorum. Ne düşündüğünü sor.";
    assert!(
        !checker
            .check(text)
            .findings
            .iter()
            .any(|f| f.rule == GrammarRule::NeNePolarity),
        "{:?}",
        checker.check(text).findings
    );
    // the verb after kimse is negative; a later clause may be positive
    let text = "Kimse gelmedi ama sonra herkes geldi.";
    assert!(!checker
        .check(text)
        .findings
        .iter()
        .any(|f| f.rule == GrammarRule::NegationConcord));
}

/// Review 2 GN2-03: a genitive with -da written on it is split ("Hakkının da"), but not a
/// word one letter away from a possessed locative, which is that word misspelt (aranında for
/// oranında, bölünmeninde for bölünmesinde).
#[test]
fn misspelt_locatives_are_not_split_as_genitives() {
    let checker = TurkishGrammarChecker::new();
    let split = |text: &str| {
        checker
            .check(text)
            .findings
            .iter()
            .any(|f| f.rule == GrammarRule::DeDaAfterVerb)
    };
    assert!(split("Hakkınında karar verildi."));
    assert!(split("Faturasınında ödenmesi gerekir."));
    assert!(split("Evede gitti."));
    assert!(!split("Faiz aranında artış oldu."));
    assert!(!split("Davanın alınında görüldüğü gibi."));
    // Review 3 GRAM3-02: only a frequent word of another root is a likelier reading; the
    // token's own root's locative (bölünmesinde) or a rare word (eninde, izinde) is not.
    assert!(split("Bölünmeninde etkisi var."));
    assert!(split("Benim seninde gelmeni isterim."));
    assert!(split("Bu ülkeninde vatandaşıydı."));
}

#[test]
fn corrections_stay_as_recorded() {
    // Review 3 GRAM3-04: what the checker writes for sentences with an error and for their
    // look-alikes, exactly; the precision gate guards only clean text.
    let checker = TurkishGrammarChecker::new();
    let mut wrong = Vec::new();
    for line in include_str!("data/grammar_recall.tsv")
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
    {
        let (input, expected) = line.split_once('\t').expect("input<TAB>expected");
        let got = checker.correct(input);
        if got != expected {
            wrong.push(format!("{input:?}: {got:?}, recorded {expected:?}"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
