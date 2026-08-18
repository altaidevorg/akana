use akana_core::grammar::{ErrorCategory, TurkishGrammarChecker};

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
    assert!(res.findings.iter().any(|f| f.category == ErrorCategory::CliticDeDa));
    assert_eq!(res.corrected, "Ali de geldi, Veli de geldi.");

    // 2. Attached de/da on finite verb
    let res2 = checker.check("Gittide geri dönmedi.");
    assert!(res2.findings.iter().any(|f| f.category == ErrorCategory::CliticDeDa));
    assert_eq!(res2.corrected, "Gitti de geri dönmedi.");
}

#[test]
fn test_clitic_ki() {
    let checker = TurkishGrammarChecker::new();

    // 1. Attached ki on verb
    let res = checker.check("Duydumki unutmuşsun gözlerimin rengini.");
    assert!(res.findings.iter().any(|f| f.category == ErrorCategory::CliticKi));
    assert_eq!(res.corrected, "Duydum ki unutmuşsun gözlerimin rengini.");

    // 2. SOMBAHÇEMİ exception separated
    let res2 = checker.check("Madem ki bilmiyorsun, neden konuşuyorsun?");
    assert!(res2.findings.iter().any(|f| f.category == ErrorCategory::CliticKi));
    assert_eq!(res2.corrected, "Mademki bilmiyorsun, neden konuşuyorsun?");
}

#[test]
fn test_particle_mi() {
    let checker = TurkishGrammarChecker::new();

    // 1. Merged question particle
    let res = checker.check("Geldinmi dün akşam?");
    assert!(res.findings.iter().any(|f| f.category == ErrorCategory::ParticleMi));
    assert_eq!(res.corrected, "Geldin mi dün akşam?");

    // 2. Merged with person suffix
    let res2 = checker.check("Biliyormusun bu şarkıyı?");
    assert!(res2.findings.iter().any(|f| f.category == ErrorCategory::ParticleMi));
    assert_eq!(res2.corrected, "Biliyor musun bu şarkıyı?");

    // 3. Standalone vowel harmony violation
    let res3 = checker.check("Sen de gördün mı onu?");
    assert!(res3.findings.iter().any(|f| f.category == ErrorCategory::ParticleMi));
    assert_eq!(res3.corrected, "Sen de gördün mü onu?");
}

#[test]
fn test_consonant_assimilation() {
    let checker = TurkishGrammarChecker::new();

    // kitapda -> kitapta
    let res = checker.check("Kitapda ilginç bilgiler var.");
    assert!(res.findings.iter().any(|f| f.category == ErrorCategory::ConsonantAssimilation));
    assert_eq!(res.corrected, "Kitapta ilginç bilgiler var.");

    // sokakdan -> sokaktan
    let res2 = checker.check("Sokakdan sesler geliyor.");
    assert!(res2.findings.iter().any(|f| f.category == ErrorCategory::ConsonantAssimilation));
    assert_eq!(res2.corrected, "Sokaktan sesler geliyor.");
}

#[test]
fn test_vowel_drop() {
    let checker = TurkishGrammarChecker::new();

    // akılı -> aklı
    let res = checker.check("Onun akılı çok karışık.");
    assert!(res.findings.iter().any(|f| f.category == ErrorCategory::VowelDropping));
    assert_eq!(res.corrected, "Onun aklı çok karışık.");

    // şehire -> şehre
    let res2 = checker.check("Yarın şehire gideceğiz.");
    assert!(res2.findings.iter().any(|f| f.category == ErrorCategory::VowelDropping));
    assert_eq!(res2.corrected, "Yarın şehre gideceğiz.");
}

#[test]
fn test_consonant_softening() {
    let checker = TurkishGrammarChecker::new();

    // kitapı -> kitabı
    let res = checker.check("Bu kitapı mutlaka okumalısın.");
    assert!(res.findings.iter().any(|f| f.category == ErrorCategory::ConsonantSoftening));
    assert_eq!(res.corrected, "Bu kitabı mutlaka okumalısın.");
}

#[test]
fn test_apostrophe_proper_nouns_and_numbers() {
    let checker = TurkishGrammarChecker::new();

    // Ahmetler'in -> Ahmetlerin (plural suffix on proper noun must not have apostrophe)
    let res = checker.check("Ahmetler'in evi buraya çok yakın.");
    assert!(res.findings.iter().any(|f| f.category == ErrorCategory::ApostropheProperNoun));
    assert_eq!(res.corrected, "Ahmetlerin evi buraya çok yakın.");

    // 2.'nci -> 2'nci (double ordinal)
    let res2 = checker.check("Yarışmada 2.'nci oldu.");
    assert!(res2.findings.iter().any(|f| f.category == ErrorCategory::ApostropheNumberDate));
    assert_eq!(res2.corrected, "Yarışmada 2'nci oldu.");

    // 1923'de -> 1923'te (consonant assimilation on number)
    let res3 = checker.check("Cumhuriyet 1923'de kuruldu.");
    assert!(res3.findings.iter().any(|f| f.category == ErrorCategory::ApostropheNumberDate));
    assert_eq!(res3.corrected, "Cumhuriyet 1923'te kuruldu.");
}

#[test]
fn test_reduplications() {
    let checker = TurkishGrammarChecker::new();

    // elele -> el ele
    let res = checker.check("Çocuklar elele yürüyorlardı.");
    assert!(res.findings.iter().any(|f| f.category == ErrorCategory::ReduplicationOrthography));
    assert_eq!(res.corrected, "Çocuklar el ele yürüyorlardı.");

    // yanyana -> yan yana
    let res2 = checker.check("İki bina yanyana inşa edilmiş.");
    assert!(res2.findings.iter().any(|f| f.category == ErrorCategory::ReduplicationOrthography));
    assert_eq!(res2.corrected, "İki bina yan yana inşa edilmiş.");
}

#[test]
fn test_quantity_plural_agreement() {
    let checker = TurkishGrammarChecker::new();

    // üç elmalar -> üç elma
    let res = checker.check("Pazardan üç elmalar aldım.");
    assert!(res.findings.iter().any(|f| f.category == ErrorCategory::QuantityPluralClash));
    assert_eq!(res.corrected, "Pazardan üç elma aldım.");

    // birçok insanlar -> birçok insan
    let res2 = checker.check("Mitinge birçok insanlar katıldı.");
    assert!(res2.findings.iter().any(|f| f.category == ErrorCategory::QuantityPluralClash));
    assert_eq!(res2.corrected, "Mitinge birçok insan katıldı.");
}

#[test]
fn test_compound_verbs() {
    let checker = TurkishGrammarChecker::new();

    // terketmek -> terk etmek
    let res = checker.check("Evi terketmek zorunda kaldı.");
    assert!(res.findings.iter().any(|f| f.category == ErrorCategory::CompoundWordOrthography));
    assert_eq!(res.corrected, "Evi terk etmek zorunda kaldı.");

    // hiss etmek -> hissetmek
    let res2 = checker.check("Kendini çok iyi hiss etti.");
    assert!(res2.findings.iter().any(|f| f.category == ErrorCategory::CompoundWordOrthography));
    assert_eq!(res2.corrected, "Kendini çok iyi hissetti.");
}

#[test]
fn test_tautology_redundancy() {
    let checker = TurkishGrammarChecker::new();

    // henüz hala -> henüz
    let res = checker.check("O henüz hala buraya gelmedi.");
    assert!(res.findings.iter().any(|f| f.category == ErrorCategory::TautologyRedundancy));
    assert_eq!(res.corrected, "O henüz buraya gelmedi.");

    // birlikte beraber -> birlikte
    let res2 = checker.check("Onlar birlikte beraber çalışıyorlar.");
    assert!(res2.findings.iter().any(|f| f.category == ErrorCategory::TautologyRedundancy));
    assert_eq!(res2.corrected, "Onlar birlikte çalışıyorlar.");
}
