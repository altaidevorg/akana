use akana_core::phonology::*;

#[test]
fn test_turkish_casing() {
    assert_eq!(to_turkish_lower("İSTANBUL"), "istanbul");
    assert_eq!(to_turkish_lower("ILIK"), "ılık");
    assert_eq!(to_turkish_lower("IŞIK"), "ışık");
    assert_eq!(to_turkish_lower("İĞNE"), "iğne");
    assert_eq!(to_turkish_lower("ÇOCUK"), "çocuk");
    assert_eq!(to_turkish_lower("ÖĞRETMEN"), "öğretmen");
    assert_eq!(to_turkish_lower("ŞEMSİYE"), "şemsiye");

    assert_eq!(to_turkish_upper("istanbul"), "İSTANBUL");
    assert_eq!(to_turkish_upper("ılık"), "ILIK");
    assert_eq!(to_turkish_upper("ışık"), "IŞIK");
    assert_eq!(to_turkish_upper("türkçe"), "TÜRKÇE");
    assert_eq!(to_turkish_upper("ağaç"), "AĞAÇ");

    assert_eq!(to_turkish_title("istanbul ve ankara"), "İstanbul Ve Ankara");
    assert_eq!(to_turkish_title("ılık süt ve taze ekmek"), "Ilık Süt Ve Taze Ekmek");
    assert_eq!(to_turkish_title("türk dil kurumu"), "Türk Dil Kurumu");
}

#[test]
fn test_vowel_harmony() {
    // Major vowel harmony (Büyük Ünlü Uyumu: all back or all front)
    assert!(check_major_vowel_harmony("okul"));
    assert!(check_major_vowel_harmony("çiçek"));
    assert!(check_major_vowel_harmony("ağaç"));
    assert!(check_major_vowel_harmony("türkiye"));
    assert!(check_major_vowel_harmony("gözlük"));
    assert!(check_major_vowel_harmony("ayakkabı"));
    assert!(check_major_vowel_harmony("kelebek"));

    // Non-harmonic words
    assert!(!check_major_vowel_harmony("kitap"));
    assert!(!check_major_vowel_harmony("kalem"));
    assert!(!check_major_vowel_harmony("tiyatro"));
    assert!(!check_major_vowel_harmony("otobüs"));
    assert!(!check_major_vowel_harmony("televizyon"));
    assert!(!check_major_vowel_harmony("dünya"));

    // Minor vowel harmony (Küçük Ünlü Uyumu)
    assert!(check_minor_vowel_harmony("çocuk"));
    assert!(check_minor_vowel_harmony("gözlük"));
    assert!(check_minor_vowel_harmony("kapı"));
    assert!(check_minor_vowel_harmony("odun"));
    assert!(check_minor_vowel_harmony("bilezik"));
}

#[test]
fn test_consonant_softening() {
    assert_eq!(soften_consonant('p', false), 'b');
    assert_eq!(soften_consonant('ç', false), 'c');
    assert_eq!(soften_consonant('t', false), 'd');
    assert_eq!(soften_consonant('k', false), 'ğ');
    assert_eq!(soften_consonant('k', true), 'g');

    assert_eq!(apply_stem_softening("kitap"), "kitab");
    assert_eq!(apply_stem_softening("ağaç"), "ağac");
    assert_eq!(apply_stem_softening("kanat"), "kanad");
    assert_eq!(apply_stem_softening("ayak"), "ayağ");
    assert_eq!(apply_stem_softening("renk"), "reng");
    assert_eq!(apply_stem_softening("bebek"), "bebeğ");
}

#[test]
fn test_vowel_drop() {
    assert_eq!(apply_vowel_drop("burun"), Some("burn".to_string()));
    assert_eq!(apply_vowel_drop("akıl"), Some("akl".to_string()));
    assert_eq!(apply_vowel_drop("şehir"), Some("şehr".to_string()));
    assert_eq!(apply_vowel_drop("karın"), Some("karn".to_string()));
    assert_eq!(apply_vowel_drop("ağız"), Some("ağz".to_string()));
    assert_eq!(apply_vowel_drop("alın"), Some("aln".to_string()));
    assert_eq!(apply_vowel_drop("oğul"), Some("oğl".to_string()));
    assert_eq!(apply_vowel_drop("fikir"), Some("fikr".to_string()));
}

#[test]
fn test_consonant_doubling() {
    assert_eq!(apply_consonant_doubling("hak"), "hakk");
    assert_eq!(apply_consonant_doubling("his"), "hiss");
    assert_eq!(apply_consonant_doubling("af"), "aff");
    assert_eq!(apply_consonant_doubling("sır"), "sırr");
    assert_eq!(apply_consonant_doubling("hat"), "hatt");
    assert_eq!(apply_consonant_doubling("zam"), "zamm");
}

#[test]
fn test_harmonic_vowel_selectors() {
    assert_eq!(get_a_type_harmonic_vowel("kitap"), 'a');
    assert_eq!(get_a_type_harmonic_vowel("ev"), 'e');
    assert_eq!(get_a_type_harmonic_vowel("çiçek"), 'e');
    assert_eq!(get_a_type_harmonic_vowel("okul"), 'a');

    assert_eq!(get_i_type_harmonic_vowel("kitap"), 'ı');
    assert_eq!(get_i_type_harmonic_vowel("ev"), 'i');
    assert_eq!(get_i_type_harmonic_vowel("okul"), 'u');
    assert_eq!(get_i_type_harmonic_vowel("göz"), 'ü');
}
