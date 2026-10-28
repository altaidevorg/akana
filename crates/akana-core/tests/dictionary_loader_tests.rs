//! User dictionaries are read as the lexicon build reads the same lines, and a line the loader
//! cannot read in full is returned, never half-read (review 3 LEX3-05, BIND3-05).

use akana_core::morphology::*;
use std::path::PathBuf;

/// The one item a single line loads, or the rejected line.
fn load(line: &str) -> Result<DictionaryItem, Vec<String>> {
    let mut lexicon = RootLexicon::empty();
    let rejected = lexicon.load_from_str(line);
    if !rejected.is_empty() {
        return Err(rejected);
    }
    let items: Vec<&DictionaryItem> = lexicon.items().collect();
    assert_eq!(items.len(), 1, "{line}");
    Ok(items[0].clone())
}

#[test]
fn zemberek_lines_read_as_the_build_reads_them() {
    // A capitalised word is a proper noun and never a verb (Kızılırmak is no verb kızılır-),
    // and a name gets no voicing (Selçuk: Selçuk'u).
    let river = load("Kızılırmak").unwrap();
    assert_eq!(
        (river.root.as_str(), river.primary_pos, river.secondary_pos),
        ("kızılırmak", PrimaryPos::Noun, SecondaryPos::ProperNoun)
    );
    let name = load("Selçuk").unwrap();
    assert_eq!(name.secondary_pos, SecondaryPos::ProperNoun);
    assert!(!name.attributes.contains(RootAttr::VOICING));
    // A verb is written in lower case, has more than three letters and is no CompoundP3sg.
    assert_eq!(load("foobarlamak").unwrap().primary_pos, PrimaryPos::Verb);
    assert_eq!(load("mak").unwrap().primary_pos, PrimaryPos::Noun);
    let finger = load("ortaparmak [A:CompoundP3sg; Roots:orta-parmak]").unwrap();
    assert_eq!(
        (finger.root.as_str(), finger.primary_pos),
        ("ortaparmak", PrimaryPos::Noun)
    );
    // A postposition's case, and the first secondary POS.
    let ait = load("ait [P:Postp,PCDat]").unwrap();
    assert_eq!(
        (ait.primary_pos, ait.governed_case.as_deref()),
        (PrimaryPos::Postp, Some("Dat"))
    );
    assert_eq!(
        load("xyzq [P:Noun,Abbrv,Prop]").unwrap().secondary_pos,
        SecondaryPos::Abbreviation
    );
    // An acronym's Latin I is i in its lemma; read letter by letter it is ı.
    let iban = load("IBAN [P:Noun,Abbrv]").unwrap();
    assert_eq!(
        (iban.lemma.as_str(), iban.pronunciation()),
        ("iban", "iban")
    );
    let fbi = load("FBI [P:Noun,Abbrv]").unwrap();
    assert_eq!((fbi.lemma.as_str(), fbi.pronunciation()), ("fbi", "febeı"));
    assert_eq!(fbi.display_lemma(), "FBI");
    // A comment may follow the bracket (akana-variants.dict writes one on every line).
    assert!(load("kahvaltı [P:Noun; A:NoVoicing] # a comment").is_ok());
    // Zemberek's attributes that akana has no use for are allowed.
    assert!(load("abartı [P:Noun; A:Ext, NoQuote]").is_ok());
}

#[test]
fn lines_that_cannot_be_read_in_full_are_returned() {
    for line in [
        // a misspelt attribute, in both formats
        "kitapçığımsı [P:Noun; A:Voicng]",
        "kitapçığımsı kitapçığımsı Noun Voicng",
        // an unknown secondary POS or field
        "foo [P:Noun,Bogus]",
        "foo [P:Noun; Prr:fu]",
        "foo foo Noun pron=fu",
        // a voice name on a noun
        "kalemlik [P:Noun; A:Reflexive]",
        // text after the bracket that is no comment
        "foo [P:Noun] bar",
        // roots no word reaches: empty, a ligature, a control character
        "- [P:Noun]",
        "'' '' Noun",
        "\u{fb01}lm [P:Noun]",
        "fo\u{0007}o [P:Noun]",
    ] {
        assert_eq!(load(line), Err(vec![line.to_string()]), "{line:?}");
    }
}

#[test]
fn a_byte_order_mark_and_decomposed_letters_are_read() {
    let mut morphology = TurkishMorphology::new();
    let rejected = morphology.load_dictionary_str(
        "\u{feff}bomkelime [P:Noun]\ns\u{0327}orbanc\u{0327}\u{0131}k [P:Noun]",
    );
    assert!(rejected.is_empty(), "{rejected:?}");
    assert!(morphology
        .analyze("bomkelimeler")
        .iter()
        .any(|p| p.root == "bomkelime"));
    assert!(morphology
        .analyze("şorbançıkta")
        .iter()
        .any(|p| p.root == "şorbançık"));
}

/// A line of Zemberek's dictionaries loads through the user-dictionary path as the build's
/// `make_entry` reads it: compared with the built-in item of the same lemma, root, POS,
/// secondary POS and homograph number, it has the same attributes, pronunciation and
/// postposition case. Only words with one line in all six dictionaries are compared: the build
/// also merges the lines of one word (a sense takes another's phonology, circumflex twins are
/// one item, an earlier file wins), which a user dictionary's single line has no part in. Words
/// akana's own dictionaries change and `Ref:` lines are left out too.
#[test]
fn zemberek_dictionaries_load_as_the_build_reads_their_lines() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../scripts/lexicon");
    let builtin = RootLexicon::new();
    // Words akana's own dictionaries name (supplements, respellings, removals).
    let mut touched = std::collections::HashSet::new();
    for file in [
        "akana-legal.dict",
        "akana-extra.dict",
        "akana-variants.dict",
        "akana-spelling.dict",
        "akana-remove.dict",
    ] {
        let text = std::fs::read_to_string(dir.join(file)).unwrap();
        for line in text.lines().filter(|l| !l.trim_start().starts_with('#')) {
            let word = line.split(['#', '[']).next().unwrap_or("").trim();
            if !word.is_empty() {
                touched.extend(load(word).map(|i| i.lemma));
            }
        }
    }
    let files = [
        "master-dictionary.dict",
        "non-tdk.dict",
        "proper.dict",
        "abbreviations.dict",
        "person-names.dict",
        "proper-from-corpus.dict",
    ];
    let mut lines = Vec::new();
    let mut rejected = Vec::new();
    let mut per_lemma: std::collections::HashMap<String, usize> = Default::default();
    for file in files {
        let text = std::fs::read_to_string(dir.join("zemberek").join(file)).unwrap();
        for line in text.lines().map(str::trim) {
            if line.is_empty() || line.starts_with('#') || line.contains("P:Punc") {
                continue;
            }
            match load(line) {
                Ok(item) => {
                    *per_lemma.entry(item.lemma.clone()).or_default() += 1;
                    lines.push((file, line.to_string(), item));
                }
                Err(r) => rejected.extend(r),
            }
        }
    }
    // Attributes the build adds from its own lists, not from the line.
    let build_only =
        RootAttr::RARE | RootAttr::NONSTANDARD | RootAttr::LEXICALIZED | RootAttr::NOT_LEXICALIZED;
    let (mut compared, mut corpus_dropped) = (0, 0);
    let (mut dropped, mut differ) = (Vec::new(), Vec::new());
    for (file, line, item) in &lines {
        if per_lemma[&item.lemma] > 1 || touched.contains(&item.lemma) || line.contains("Ref:") {
            continue;
        }
        let same = builtin.get_items(&item.root).and_then(|items| {
            items.iter().find(|b| {
                b.lemma == item.lemma
                    && b.primary_pos == item.primary_pos
                    && b.secondary_pos == item.secondary_pos
                    && b.homograph == item.homograph
            })
        });
        let Some(same) = same else {
            if *file == "proper-from-corpus.dict" {
                corpus_dropped += 1;
            } else {
                dropped.push(line.as_str());
            }
            continue;
        };
        compared += 1;
        let found = (
            same.attributes - build_only,
            same.pronunciation(),
            same.governed_case.as_deref(),
        );
        let loaded = (
            item.attributes - build_only,
            item.pronunciation(),
            item.governed_case.as_deref(),
        );
        if found != loaded {
            differ.push(format!(
                "{file}: {line}: built in {found:?}, loaded {loaded:?}"
            ));
        }
    }
    // The build drops these too: roots with characters that are no letters or digits.
    assert_eq!(
        rejected,
        [
            "Alm. [Pr:almanca; P:Abbrv]",
            "İng. [Pr:ingilizce; P:Abbrv]",
            "am. [Pr:eyem; P:Abbrv]",
        ]
    );
    // The build's removals (akana-remove.dict) that the words above miss, written with a
    // Latin I; and the corpus names its junk filter drops (a loader that lost Prop would miss
    // thousands).
    assert_eq!(
        dropped,
        [
            "Iv [P:Abbrv]",
            "Şıh [P:Abbrv]",
            "Vı [P:Abbrv]",
            "Vıı [P:Abbrv]",
            "FIGC [Pr:figo; P:Abbrv]",
        ]
    );
    assert!(
        corpus_dropped < 400,
        "{corpus_dropped} corpus names not in the lexicon"
    );
    assert!(compared > 50_000, "only {compared} lines compared");
    assert!(
        differ.is_empty(),
        "{} of {compared} lines differ:
{}",
        differ.len(),
        differ[..differ.len().min(40)].join(
            "
"
        )
    );
}
