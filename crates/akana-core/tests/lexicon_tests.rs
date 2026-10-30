//! Root lexicon and loader regressions: the generated lexicon must keep Zemberek's POS and
//! secondary POS (scripts/build_full_lexicon.py), and the loader must read them.
use akana_core::morphology::*;

fn has(parses: &[MorphParse], root: &str, pos: PrimaryPos, tags: &[&str]) -> bool {
    parses.iter().any(|p| {
        p.root == root
            && p.primary_pos == pos
            && tags.iter().all(|t| p.morpheme_tags.iter().any(|m| m == t))
    })
}

#[test]
fn nouns_ending_in_mak_mek_stay_nouns() {
    // Zemberek lists `yemek [P:Noun]`, `parmak [P:Noun]`, `ırmak [P:Noun]`; the old build
    // forced every -mak/-mek word to a verb with a fabricated root (par-, ır-).
    let morph = TurkishMorphology::new();
    assert!(has(
        &morph.analyze("yemeğini"),
        "yemek",
        PrimaryPos::Noun,
        &["P3sg", "Acc"]
    ));
    assert!(has(
        &morph.analyze("parmağı"),
        "parmak",
        PrimaryPos::Noun,
        &[]
    ));
    let irmakta = morph.analyze("ırmakta");
    assert!(has(&irmakta, "ırmak", PrimaryPos::Noun, &["Loc"]));
    assert!(!irmakta.iter().any(|p| p.root == "ır"));
    // The verb is still there.
    assert!(has(
        &morph.analyze("yemek"),
        "ye",
        PrimaryPos::Verb,
        &["Inf"]
    ));
}

#[test]
fn verb_stems_have_no_noun_homographs() {
    // A readability word list used to add bare verb stems (gel, çık, söyle) as nouns, and the
    // location gazetteer added villages such as Kork; both made past-tense verbs read as
    // Noun+CopPast.
    let morph = TurkishMorphology::new();
    for word in ["geldi", "çıktı", "söyledi", "korktum"] {
        let parses = morph.analyze(word);
        assert!(
            parses.iter().any(|p| p.primary_pos == PrimaryPos::Verb
                && p.morpheme_tags.iter().any(|t| t == "Past")),
            "{word}: no Verb+Past parse"
        );
        // A copular reading of a common noun, or of a proper noun scored like one, is the
        // defect. (Abbreviation readings such as sö-yle-di stay, scored as unlikely.)
        assert!(
            !parses
                .iter()
                .any(|p| p.morpheme_tags.iter().any(|t| t == "CopPast")
                    && (p.secondary_pos == SecondaryPos::None || p.score > 0.9)),
            "{word}: spurious Noun+CopPast parse"
        );
    }
}

#[test]
fn proper_nouns_from_the_lexicon_file_are_proper() {
    // `aachen` exists only in the generated lexicon, whose secondary POS column says
    // `ProperNoun`; the loader used to read that as an attribute only.
    let morph = TurkishMorphology::new();
    let parses = morph.analyze("Aachen");
    assert!(!parses.is_empty());
    assert!(parses
        .iter()
        .all(|p| p.secondary_pos == SecondaryPos::ProperNoun));
    // Written lower-case it is an unlikely reading; its case needs an apostrophe (aachen'de).
    let lower = morph.analyze("aachen");
    assert!(lower
        .iter()
        .any(|p| p.secondary_pos == SecondaryPos::ProperNoun && p.score < 0.9));
    assert!(morph.analyze("aachende").is_empty());
    assert!(!morph.analyze("aachen'de").is_empty());
}

#[test]
fn a_name_without_its_apostrophe_reads_only_through_a_consonant_ending() {
    // A name written with its capital and a case ending but no apostrophe reads as the name,
    // at the factor of a name's suffixes without one (plan step D1).
    let morph = TurkishMorphology::new();
    for (word, root, case) in [
        ("Türkiyede", "türkiye", "Loc"),
        ("Aachende", "aachen", "Loc"),
    ] {
        let parses = morph.analyze(word);
        assert_eq!(parses[0].root, root, "{word}");
        assert_eq!(parses[0].morpheme_tags, ["Noun", case], "{word}");
        assert!(parses.iter().all(|p| p.score <= 0.6), "{word}");
    }
    // A vowel after a name is as often the end of a name the lexicon lacks, and -ye after a
    // name in -i ends many (Nuriye).
    for (word, root) in [
        ("Saide", "said"),
        ("Türkeri", "türker"),
        ("Özbakım", "özbak"),
        ("Nuriye", "nuri"),
    ] {
        assert!(
            morph.analyze(word).iter().all(|p| p.root != root),
            "{word} read through {root}"
        );
    }
}

#[test]
fn a_latin_i_reads_as_i_where_the_turkish_reading_finds_no_word() {
    // Plan step D3. Text typed without the Turkish keyboard writes İ as I, and so do foreign
    // acronyms, whose lemmas have i.
    let morph = TurkishMorphology::new();
    let lemma = |word: &str| morph.analyze(word).first().map(|p| p.lemma.clone());
    assert_eq!(lemma("KITAPLAR").as_deref(), Some("kitap"));
    assert_eq!(lemma("INSAN").as_deref(), Some("insan"));
    assert_eq!(lemma("FBI").as_deref(), Some("fbi"));
    assert_eq!(lemma("fbi").as_deref(), Some("fbi"));
    assert_eq!(lemma("COVID'e").as_deref(), Some("kovid"));
    // A known acronym takes its suffixes by the lexicon's pronunciation (CIA "siyayey", FBI
    // "febeı"); an unknown one by either reading of its I, with the lemma spelled accordingly.
    assert_eq!(lemma("CIA'in").as_deref(), Some("cia"));
    assert_eq!(lemma("FBI'nın").as_deref(), Some("fbi"));
    assert_eq!(lemma("XYZI'nin").as_deref(), Some("xyzi"));
    assert_eq!(lemma("XYZI'nın").as_deref(), Some("xyzı"));
    // A Turkish word in capitals keeps its ı (KIRATLI'nın).
    assert_eq!(lemma("KIRATLI'nın").as_deref(), Some("kıratlı"));
    // The Turkish reading stays where it makes a word, and Roman numerals stay numbers.
    assert_eq!(lemma("ISI").as_deref(), Some("ısı"));
    assert_eq!(morph.analyze("IV")[0].primary_pos, PrimaryPos::Num);
    // Turkish words in capitals that Zemberek's list took for acronyms are gone.
    assert!(morph
        .analyze("iki")
        .iter()
        .all(|p| p.secondary_pos != SecondaryPos::Abbreviation));
}

#[test]
fn old_orthography_reads_as_nonstandard_with_the_modern_lemma() {
    // Plan step D5: -iyle, a or e narrowed before y, -mAğA. Search finds the lemma; spelling
    // checks see no correct word.
    let morph = TurkishMorphology::new();
    for (word, lemma) in [
        ("karariyle", "karar"),
        ("maksadiyle", "maksat"),
        ("olmıyan", "olmak"),
        ("sağlıyacak", "sağlamak"),
        ("vermeğe", "vermek"),
    ] {
        let parses = morph.analyze(word);
        assert_eq!(
            parses.first().map(|p| p.lemma.as_str()),
            Some(lemma),
            "{word}"
        );
        assert!(
            parses.iter().all(|p| p.nonstandard && p.score <= 0.5),
            "{word}"
        );
        assert!(morph.analyze_standard(word).is_empty(), "{word}");
    }
    // An acute accent folds like a circumflex, and the modern forms stay standard.
    for word in ["hálá", "káğıt", "eviyle", "damağa"] {
        assert!(!morph.analyze_standard(word).is_empty(), "{word}");
    }
}

#[test]
fn usage_variants_of_alternations_read_as_nonstandard() {
    // Plan step D5: second stems with one alternation flipped (akana-variants.dict).
    let morph = TurkishMorphology::new();
    for (variant, standard, lemma) in [
        ("temerrüte", "temerrüde", "temerrüt"),
        ("gaspı", "gasbı", "gasp"),
        ("fuhuşa", "fuhşa", "fuhuş"),
        ("hakında", "hakkında", "hak"),
    ] {
        let parses = morph.analyze(variant);
        assert_eq!(
            parses.first().map(|p| p.lemma.as_str()),
            Some(lemma),
            "{variant}"
        );
        assert!(parses.iter().all(|p| p.nonstandard), "{variant}");
        assert!(morph.analyze_standard(variant).is_empty(), "{variant}");
        // The standard spelling reads as before, and first.
        let standard = morph.analyze(standard);
        assert!(
            !standard.is_empty() && !standard[0].nonstandard,
            "{standard:?}"
        );
    }
}

#[test]
fn an_unknown_word_backs_off_to_a_guessed_noun_stem() {
    // Plan step D6: the stem search indexes a word the lexicon cannot read by.
    let morph = TurkishMorphology::new();
    for (word, stem) in [
        ("hiperbariğe", Some("hiperbarik")),
        ("istanbulda", Some("istanbul")),
        ("kampüslerin", Some("kampüs")),
        ("dosyada", Some("dosya")),
        ("hiperbariklerimizden", Some("hiperbarik")),
        // Review 3 MORPH3-02: a vowel the word may end in is no inflection to take off, so a
        // vowel-final word and its inflected forms are one stem (zorbanta, not zorbant).
        ("zorbanta", None),
        ("zorbantalar", Some("zorbanta")),
        ("zorbantanın", Some("zorbanta")),
        // Derivations, -m and -n stay on.
        ("kitapçı", None),
        ("bitcoin", None),
        ("instagram", None),
    ] {
        assert_eq!(morph.backoff_stem(word).as_deref(), stem, "{word}");
    }
}

#[test]
fn spellings_are_judged_in_the_case_they_are_written_in() {
    // Plan step D7: a name's reading does not make a word written in lower case a spelling,
    // nor an abbreviation's the dictionary spells in capitals; analyze() still reads them.
    let morph = TurkishMorphology::new();
    let standard = |w: &str| !morph.analyze_standard(w).is_empty();
    assert!(standard("Ankara'da") && !standard("ankara'da"));
    // A name without its apostrophe reads, but is no spelling (review 3 NORM3-04).
    assert!(!standard("Türkiyede") && !morph.analyze("Türkiyede").is_empty());
    assert!(!morph.analyze("varmı").is_empty() && !standard("varmı"));
    // Abbreviations the dictionary spells in lower case, and dotted ones, are spellings.
    for word in ["km", "vb", "bkz", "vb.", "TBMM'de"] {
        assert!(standard(word), "{word}");
    }
}

#[test]
fn abbreviations_are_not_common_nouns() {
    // Zemberek's `K [P:Abbrv]` became the common noun `k`, so `ki` read as k-ı (Acc).
    let morph = TurkishMorphology::new();
    let parses = morph.analyze("ki");
    assert_eq!(parses[0].primary_pos, PrimaryPos::Conj);
    for p in parses.iter().filter(|p| p.root == "k") {
        assert_eq!(p.secondary_pos, SecondaryPos::Abbreviation);
        assert!(p.score < 0.9);
    }
}

#[test]
fn loader_reads_secondary_pos_and_determiners() {
    let mut lexicon = RootLexicon::empty();
    lexicon.load_from_str(
        "zorlu zorlu Noun ProperNoun ProperNoun\nşey şey Det None\nabc abc Noun Abbrv\n",
    );
    let zorlu = &lexicon.get_items("zorlu").expect("zorlu")[0];
    assert_eq!(zorlu.secondary_pos, SecondaryPos::ProperNoun);
    assert!(zorlu.attributes.contains(RootAttr::PROPER_NOUN));
    assert_eq!(
        lexicon.get_items("şey").expect("şey")[0].primary_pos,
        PrimaryPos::Det
    );
    assert_eq!(
        lexicon.get_items("abc").expect("abc")[0].secondary_pos,
        SecondaryPos::Abbreviation
    );
}

fn chosen(sentence: &str) -> Vec<MorphParse> {
    let tokens: Vec<&str> = sentence.split_whitespace().collect();
    MorphologicalDisambiguator::new().disambiguate(&tokens)
}

#[test]
fn question_particle_contrast_keeps_the_verb() {
    // The EchoCraft contrast: mI after the subject, the object or the verb.
    for sentence in [
        "Ayşe mi okula geldi",
        "Ayşe okula mı geldi",
        "Ayşe okula geldi mi",
    ] {
        let parses = chosen(sentence);
        let verb = parses.iter().find(|p| p.surface == "geldi").expect("geldi");
        assert_eq!(verb.primary_pos, PrimaryPos::Verb, "{sentence}");
        assert!(verb.morpheme_tags.iter().any(|t| t == "Past"), "{sentence}");
        let q = parses
            .iter()
            .find(|p| matches!(p.surface.as_str(), "mi" | "mı"))
            .expect("mI");
        assert_eq!(q.primary_pos, PrimaryPos::Q, "{sentence}");
    }
}

#[test]
fn final_past_tense_verbs_are_verbs() {
    for (sentence, word) in [
        (
            "Eve döndüğümde kapının açık olduğunu görünce çok korktum",
            "korktum",
        ),
        ("İlk başta bunu yapmak bana çok zor geldi", "geldi"),
    ] {
        let parses = chosen(sentence);
        let p = parses.iter().find(|p| p.surface == word).expect(word);
        assert_eq!(p.primary_pos, PrimaryPos::Verb, "{sentence}");
    }
}

#[test]
fn determiners_before_nouns_and_pronouns_alone() {
    assert_eq!(chosen("Bu kitabı okudun")[0].primary_pos, PrimaryPos::Det);
    assert_eq!(chosen("bu güzel kitap")[0].primary_pos, PrimaryPos::Det);
    assert_eq!(chosen("Bu doğru")[0].primary_pos, PrimaryPos::Pron);
}

#[test]
fn misspellings_and_references_read_with_the_tdk_lemma() {
    // Review 3 LEX3-01/02: Zemberek's non-TDK spellings and its Ref: pointers were lemmas of
    // their own, and the spell checker took them for TDK spellings. A misspelling now reads
    // as nonstandard with the TDK lemma; a reference or a spelling twin TDK lists takes its
    // target's lemma; an inflected form listed as a word (mevkii) reads as the word's.
    let morph = TurkishMorphology::new();
    for (word, lemma, nonstandard) in [
        ("şöförün", "şoför", true),
        ("dökümanı", "doküman", true),
        ("orjinalinde", "orijinal", true),
        ("motorsiklet", "motosiklet", true),
        ("herşey", "herşey", true),
        ("hükümetin", "hükumet", false),
        ("kastı", "kasıt", false),
        ("rızkı", "rızık", false),
        ("mevkii", "mevki", false),
        ("bayiine", "bayi", false),
    ] {
        let p = &morph.analyze(word)[0];
        assert_eq!(
            (p.lemma.as_str(), p.nonstandard),
            (lemma, nonstandard),
            "{word}"
        );
    }
    assert!(morph.analyze_standard("şöför").is_empty());
    assert!(!morph.analyze_standard("hükümet").is_empty());
}

#[test]
fn names_that_are_misspelt_words_are_gone() {
    // Review 3 NORM3-01, MORPH3-06: corpus "names" that were common words without their
    // Turkish letters made egitim a word, and the ASCII-I twin ıstanbul kept Istanbul from
    // its İ reading.
    let morph = TurkishMorphology::new();
    assert!(morph.analyze("egitim").is_empty());
    assert!(morph.analyze("Ayrica").is_empty());
    assert_eq!(morph.analyze("Istanbul")[0].lemma, "istanbul");
    // A real name spelled so stays (ASCII_NAMES in the build).
    assert!(morph
        .analyze("Akçil")
        .iter()
        .any(|p| p.secondary_pos == SecondaryPos::ProperNoun));
}

#[test]
fn homographs_have_their_own_prior_keys() {
    // Review 3 LEX3-02: dahil "interference" and dâhil "included" shared a prior key (lemma,
    // root, POS, secondary POS and homograph number), so the prior could not tell them apart.
    let morph = TurkishMorphology::new();
    let numbers: std::collections::HashSet<u8> = morph
        .root_items("dahil")
        .iter()
        .filter(|item| item.primary_pos == PrimaryPos::Noun)
        .map(|item| item.homograph)
        .collect();
    assert_eq!(numbers.len(), 2, "{numbers:?}");
}

#[test]
fn context_picks_lemma_and_case_as_review_3_found() {
    // Review 3 RANK3-01..05: (sentence, word, lemma, primary POS, last tag).
    for (sentence, word, lemma, pos, last) in [
        (
            "İtirazı varsa , bildirsin .",
            "varsa",
            "var",
            PrimaryPos::Adj,
            "CopCond",
        ),
        (
            "Bir sorun vardı .",
            "vardı",
            "var",
            PrimaryPos::Adj,
            "CopPast",
        ),
        (
            "Saat beşte eve vardı .",
            "vardı",
            "varmak",
            PrimaryPos::Verb,
            "Past",
        ),
        (
            "Sözleşmeden doğan alacak ödenmedi .",
            "doğan",
            "doğmak",
            PrimaryPos::Verb,
            "PresPart",
        ),
        (
            "Örneğin , bu karar yanlıştır .",
            "Örneğin",
            "örneğin",
            PrimaryPos::Conj,
            "Conj",
        ),
        (
            "Bu konu ile ilgili karar verildi .",
            "ile",
            "ile",
            PrimaryPos::Postp,
            "Postp",
        ),
        (
            "Ali ile Veli geldi .",
            "ile",
            "ile",
            PrimaryPos::Conj,
            "Conj",
        ),
        (
            "Bu filmi beğendim .",
            "filmi",
            "film",
            PrimaryPos::Noun,
            "Acc",
        ),
        (
            "Mahkemenin kararı kesinleşti .",
            "kararı",
            "karar",
            PrimaryPos::Noun,
            "P3sg",
        ),
        (
            "Diğerleri de geldi .",
            "Diğerleri",
            "diğer",
            PrimaryPos::Adj,
            "P3sg",
        ),
        (
            "Daha önce film çevirdi .",
            "önce",
            "önce",
            PrimaryPos::Adv,
            "Adv",
        ),
        (
            "Dilekçe ve ekleri incelendi .",
            "ekleri",
            "ek",
            PrimaryPos::Noun,
            "P3sg",
        ),
        (
            "Kesinleşmiş karar uygulanır .",
            "Kesinleşmiş",
            "kesinleşmek",
            PrimaryPos::Verb,
            "NarrPart",
        ),
    ] {
        let parses = chosen(sentence);
        let p = parses.iter().find(|p| p.surface == word).expect(word);
        assert_eq!(
            (
                p.lemma.as_str(),
                p.primary_pos,
                p.morpheme_tags.last().map(String::as_str)
            ),
            (lemma, pos, Some(last)),
            "{sentence}: {}",
            p.formatted
        );
    }
}
