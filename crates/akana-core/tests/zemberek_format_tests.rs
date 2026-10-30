//! `MorphParse::analysis` is the analysis in Zemberek's own format (`SingleAnalysis.formatLong`,
//! `AnalysisFormatters.DEFAULT`). The expected strings are Zemberek's: the examples of its
//! morphology README verbatim, and for the other words the path its `TurkishMorphotactics`
//! takes (zeytinyağı-na, numZeroDeriv_S, vUnable_S copying vNeg_S, nounZeroDeriv_S -> nVerb_S).
use akana_core::morphology::*;

fn analyses(word: &str) -> Vec<String> {
    TurkishMorphology::new()
        .analyze(word)
        .into_iter()
        .map(|p| p.analysis)
        .collect()
}

/// The best analysis is Zemberek's `expected[0]`, and Zemberek's other readings are there too.
fn assert_zemberek(word: &str, expected: &[&str]) {
    let got = analyses(word);
    assert_eq!(
        got.first().map(String::as_str),
        Some(expected[0]),
        "{word}: {got:#?}"
    );
    for e in &expected[1..] {
        assert!(got.iter().any(|g| g == e), "{word}: no {e} in {got:#?}");
    }
}

#[test]
fn inflection_and_derivation_as_zemberek_writes_them() {
    assert_zemberek("kitabıma", &["[kitap:Noun] kitab:Noun+A3sg+ım:P1sg+a:Dat"]);
    assert_zemberek(
        "geldiğimizde",
        &["[gelmek:Verb] gel:Verb|diğ:PastPart→Noun+A3sg+imiz:P1pl+de:Loc"],
    );
    // CompoundP3sg: the dative is -nA on the whole word (zeytinyağı-na); the 2nd-person
    // possessive goes on it too, as Zemberek's a3sg_S allows.
    assert_zemberek(
        "cumhurbaşkanına",
        &[
            "[cumhurbaşkanı:Noun] cumhurbaşkanı:Noun+A3sg+na:Dat",
            "[cumhurbaşkanı:Noun] cumhurbaşkanı:Noun+A3sg+n:P2sg+a:Dat",
        ],
    );
    assert_zemberek(
        "evdeyim",
        &["[ev:Noun] ev:Noun+A3sg+de:Loc|Zero→Verb+Pres+yim:A1sg"],
    );
    // -AmA is Unable, inflectional like Neg (Zemberek's vUnable_S).
    assert_zemberek(
        "gelemeyecekmişsiniz",
        &["[gelmek:Verb] gel:Verb+eme:Unable+yecek:Fut+miş:Narr+siniz:A2pl"],
    );
}

#[test]
fn zemberek_readme_examples() {
    assert_zemberek(
        "kitabımızsa",
        &["[kitap:Noun] kitab:Noun+A3sg+ımız:P1pl|Zero→Verb+sa:Cond+A3sg"],
    );
    assert_zemberek(
        "okuyacağım",
        &[
            "[okumak:Verb] oku:Verb+yacağ:Fut+ım:A1sg",
            "[okumak:Verb] oku:Verb|yacağ:FutPart→Noun+A3sg+ım:P1sg",
        ],
    );
    assert_zemberek("diyor", &["[demek:Verb] di:Verb+yor:Prog1+A3sg"]);
    assert_zemberek(
        "kalemin",
        &[
            "[kalem:Noun] kalem:Noun+A3sg+in:Gen",
            "[kalem:Noun] kalem:Noun+A3sg+in:P2sg",
            "[kale:Noun] kale:Noun+A3sg+m:P1sg+in:Gen",
        ],
    );
    assert_zemberek(
        "yağacak",
        &[
            "[yağmak:Verb] yağ:Verb+acak:Fut+A3sg",
            "[yağmak:Verb] yağ:Verb|acak:FutPart→Adj",
        ],
    );
    assert_zemberek(
        "yarın",
        &[
            "[yarın:Adv] yarın:Adv",
            "[yarmak:Verb] yar:Verb+Imp+ın:A2pl",
            "[yar:Noun] yar:Noun+A3sg+ın:Gen",
            "[yar:Noun] yar:Noun+A3sg+ın:P2sg",
            "[yarı:Noun] yarı:Noun+A3sg+n:P2sg",
            "[yarın:Noun,Time] yarın:Noun+A3sg",
            "[yarı:Adj] yarı:Adj|Zero→Noun+A3sg+n:P2sg",
        ],
    );
    assert_zemberek(
        "kışı",
        &[
            "[kış:Noun,Time] kış:Noun+A3sg+ı:P3sg",
            "[kış:Noun,Time] kış:Noun+A3sg+ı:Acc",
        ],
    );
    assert_zemberek("okut", &["[okumak:Verb] oku:Verb|t:Caus→Verb+Imp+A2sg"]);
    // The lemma as the dictionary spells it (`display_lemma`), as Zemberek prints it: a name
    // with its capital, an abbreviation in capitals, a loanword with its circumflex.
    assert_zemberek(
        "Ankara'dan",
        &["[Ankara:Noun,Prop] ankara:Noun+A3sg+dan:Abl"],
    );
    assert_zemberek("TBMM'de", &["[TBMM:Noun,Abbrv] tbmm:Noun+A3sg+de:Loc"]);
    assert_zemberek(
        "kâğıdı",
        &[
            "[kâğıt:Noun] kağıd:Noun+A3sg+ı:P3sg",
            "[kâğıt:Noun] kağıd:Noun+A3sg+ı:Acc",
        ],
    );
}

#[test]
fn display_lemma_keeps_the_dictionary_spelling() {
    let morphology = TurkishMorphology::new();
    let first = |word: &str| {
        let p = morphology
            .analyze(word)
            .into_iter()
            .next()
            .expect("analysed");
        (p.lemma.clone(), p.display_lemma().to_string())
    };
    assert_eq!(first("Ankara'ya"), ("ankara".into(), "Ankara".into()));
    assert_eq!(first("İstanbul'da"), ("istanbul".into(), "İstanbul".into()));
    assert_eq!(first("kâğıt"), ("kağıt".into(), "kâğıt".into()));
    assert_eq!(first("kağıt"), ("kağıt".into(), "kâğıt".into()));
    assert_eq!(first("e-postayı"), ("eposta".into(), "e-posta".into()));
    assert_eq!(first("geldi"), ("gelmek".into(), "gelmek".into()));
    // kar "snow" and kâr "profit" are one item, which keeps the plain spelling.
    assert_eq!(first("kar"), ("kar".into(), "kar".into()));
    // A name the lexicon does not know is shown as the default spelling of its lemma.
    assert_eq!(first("Mamyak'ın"), ("mamyak".into(), "Mamyak".into()));
    // A user dictionary may write the lemma as it is spelled.
    let mut lexicon = RootLexicon::empty();
    lexicon.load_from_str("hâkim hakim Adj\nNATO nato Noun Abbrv\nzeytin zeytin Noun");
    let items: Vec<String> = lexicon
        .items()
        .map(|i| format!("{}/{}", i.lemma, i.display_lemma()))
        .collect();
    assert_eq!(items, ["hakim/hâkim", "nato/NATO", "zeytin/zeytin"]);
}

#[test]
fn numbers_and_particles() {
    // Zemberek's numeral analysis: the number is lemma and stem (copyFor), through its last
    // word (üç-te) and numZeroDeriv_S.
    assert_zemberek(
        "1923'te",
        &["[1923:Num,Card] 1923:Num|Zero→Noun+A3sg+te:Loc"],
    );
    assert_zemberek("15.", &["[15.:Num,Ord] 15.:Num"]);
    // PrimaryPos.Question is `Ques` in Zemberek, `Q` in akana's tags.
    assert_zemberek("mı", &["[mı:Ques] mı:Ques+Pres+A3sg"]);
}

#[test]
fn legacy_tags_are_unchanged() {
    let best = TurkishMorphology::new().analyze("geldiğimizde").remove(0);
    assert_eq!(best.morpheme_tags, ["Verb", "PastPart", "P1pl", "Loc"]);
    assert_eq!(best.formatted, "[gelmek:Verb] gel:Verb+PastPart+P1pl+Loc");
}

#[test]
fn derivation_boundaries_are_the_derivative_states() {
    // The format puts `|` before a derivational morpheme, as Zemberek's formatter does; in
    // the graph those are exactly the derivative states.
    for state in &morphotactics_graph().states {
        assert_eq!(
            state.derivative,
            state.morpheme.is_derivational(),
            "{}",
            state.name
        );
    }
}

#[test]
fn every_analysis_of_running_text_is_well_formed() {
    let morphology = TurkishMorphology::new();
    let text = "Mahkemece verilen kararın bozulmasına, dosyanın yerel mahkemesine gönderilmesine \
                oybirliğiyle karar verildi. Davacının 15.03.2021 tarihli dilekçesinde %20'sini \
                istediği anlaşılmaktadır; TBMM'de görüşülemeyecekmiş.";
    for word in text.split(|c: char| c.is_whitespace() || matches!(c, ',' | ';')) {
        for p in morphology.analyze(word.trim_end_matches('.')) {
            let a = &p.analysis;
            let (item, body) = a.split_once("] ").unwrap_or_else(|| panic!("{a}"));
            assert!(item.starts_with('[') && item.contains(':'), "{a}");
            assert!(!body.contains("Pnon") && !body.contains("Nom"), "{a}");
            assert!(!body.ends_with('→') && !body.contains("+|"), "{a}");
            // The suffix surfaces spell the word after the stem.
            let stem = body.split(':').next().unwrap_or_default();
            let suffixes: String = body
                .split(['+', '|', '→'])
                .skip(1)
                .filter_map(|m| m.split_once(':').map(|(s, _)| s))
                .collect();
            if p.primary_pos != PrimaryPos::Num && !word.contains(['\'', '-']) {
                let lower = akana_core::phonology::to_turkish_lower(word.trim_end_matches('.'));
                assert_eq!(format!("{stem}{suffixes}"), lower, "{a}");
            }
        }
    }
}

#[test]
fn disambiguated_and_serialized_parses_carry_it() {
    let chosen = MorphologicalDisambiguator::new().disambiguate(&["Evdeyim", "zxqyw", "."]);
    assert_eq!(
        chosen[0].analysis,
        "[ev:Noun] ev:Noun+A3sg+de:Loc|Zero→Verb+Pres+yim:A1sg"
    );
    assert_eq!(chosen[1].analysis, "[zxqyw:Unk] zxqyw:Unk");

    // Serialized parses written before the field existed still load.
    let mut json = serde_json::to_value(&chosen[0]).unwrap();
    assert_eq!(json["analysis"], chosen[0].analysis.as_str());
    json.as_object_mut().unwrap().remove("analysis");
    let old: MorphParse = serde_json::from_value(json).unwrap();
    assert_eq!(old.analysis, "");
    assert_eq!(old.formatted, chosen[0].formatted);
}
