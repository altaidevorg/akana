use akana_core::phonology::*;

#[test]
fn test_turkish_casing() {
    assert_eq!(to_turkish_lower("İSTANBUL"), "istanbul");
    assert_eq!(to_turkish_lower("ILIK"), "ılık");
    assert_eq!(to_turkish_lower("IŞIK"), "ışık");
    assert_eq!(to_turkish_lower("İĞNE"), "iğne");

    assert_eq!(to_turkish_upper("istanbul"), "İSTANBUL");
    assert_eq!(to_turkish_upper("ılık"), "ILIK");
    assert_eq!(to_turkish_upper("ışık"), "IŞIK");
    assert_eq!(to_turkish_upper("türkçe"), "TÜRKÇE");

    assert_eq!(to_turkish_title("istanbul ve ankara"), "İstanbul Ve Ankara");
    assert_eq!(to_turkish_title("ılık süt"), "Ilık Süt");
}

#[test]
fn test_vowel_harmony() {
    // Major vowel harmony (Büyük Ünlü Uyumu)
    assert!(check_major_vowel_harmony("kitap"));
    assert!(check_major_vowel_harmony("okul"));
    assert!(check_major_vowel_harmony("çiçek"));
    assert!(check_major_vowel_harmony("ağaç"));
    assert!(check_major_vowel_harmony("türkiye"));
    assert!(!check_major_vowel_harmony("kalem"));
    assert!(!check_major_vowel_harmony("tiyatro"));
    assert!(!check_major_vowel_harmony("otobüs"));

    // Minor vowel harmony (Küçük Ünlü Uyumu)
    assert!(check_minor_vowel_harmony("çocuk"));
    assert!(check_minor_vowel_harmony("gözlük"));
    assert!(check_minor_vowel_harmony("kapı"));
    assert!(check_minor_vowel_harmony("odun"));
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
}

#[test]
fn test_vowel_drop() {
    assert_eq!(apply_vowel_drop("burun"), Some("burn".to_string()));
    assert_eq!(apply_vowel_drop("akıl"), Some("akl".to_string()));
    assert_eq!(apply_vowel_drop("şehir"), Some("şehr".to_string()));
    assert_eq!(apply_vowel_drop("karın"), Some("karn".to_string()));
    assert_eq!(apply_vowel_drop("ağız"), Some("ağz".to_string()));
}

#[test]
fn test_consonant_doubling() {
    assert_eq!(apply_consonant_doubling("hak"), "hakk");
    assert_eq!(apply_consonant_doubling("his"), "hiss");
    assert_eq!(apply_consonant_doubling("af"), "aff");
    assert_eq!(apply_consonant_doubling("sır"), "sırr");
}
