//! Engines share the built-in lexicon and the embedding weights, and batches run on several
//! threads; none of that may change a result.

use akana_core::morphology::*;
use akana_core::parser::TurkishDependencyParser;
use akana_core::tokenization::{SentenceSegmenter, TurkishTokenizer};
use akana_core::{AkanaEngine, TurkishEmbeddings, TurkishHybridEmbeddings};

fn has_root(m: &TurkishMorphology, word: &str, root: &str) -> bool {
    m.analyze(word).iter().any(|p| p.root == root)
}

#[test]
fn added_items_stay_with_their_analyzer() {
    let shared = TurkishMorphology::new();
    let mut custom = TurkishMorphology::new();
    custom.load_dictionary_str("zxqy zxqy Noun");
    let before_add = custom.clone();
    custom.add_item(DictionaryItem::new(
        "wqer",
        "wqer",
        PrimaryPos::Noun,
        SecondaryPos::None,
        RootAttr::NONE,
    ));

    assert!(has_root(&custom, "zxqyler", "zxqy"));
    assert!(has_root(&custom, "wqerler", "wqer"));
    assert!(has_root(&before_add, "zxqyler", "zxqy"));
    assert!(!has_root(&before_add, "wqerler", "wqer"));
    for m in [&shared, &TurkishMorphology::new()] {
        assert!(!has_root(m, "zxqyler", "zxqy"));
        assert!(!has_root(m, "wqerler", "wqer"));
        assert!(has_root(m, "kitaplar", "kitap"));
    }
    assert!(!TurkishMorphology::builtin_lexicon().contains_root("zxqy"));
}

#[test]
fn dictionary_files_still_load() {
    let path = std::env::temp_dir().join("akana_shared_engines_dictionary.txt");
    std::fs::write(&path, "# custom\nzxqy zxqy Noun\r\nfırçala fırçala Verb\n").unwrap();
    let mut m = TurkishMorphology::new();
    m.load_dictionary_file(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    assert!(has_root(&m, "zxqyler", "zxqy"));
    assert!(!has_root(&TurkishMorphology::new(), "zxqyler", "zxqy"));
    assert!(m.load_dictionary_file("no/such/dictionary.txt").is_err());
}

#[test]
fn parse_disambiguated_matches_parse() {
    let parser = TurkishDependencyParser::new();
    let disambiguator = MorphologicalDisambiguator::new();
    for sentence in [
        "Ali eve gitti .",
        "Bu da mı gitti ?",
        "Öğretmen öğrencilere yeni aldığı kitapları dağıttı .",
        "",
    ] {
        let tokens: Vec<&str> = sentence.split_whitespace().collect();
        let parses = disambiguator.disambiguate(&tokens);
        assert_eq!(
            parser.parse_disambiguated(&tokens, &parses).to_conllu(),
            parser.parse(&tokens).to_conllu()
        );
    }
}

#[test]
fn analyze_document_matches_the_separate_steps() {
    let text = "Ali eve gitti. Öğretmen öğrencilere yeni aldığı kitapları dağıttı! \
                Bu da mı gitti? Şükrü İğdır'da çalışıyor.";
    let disambiguator = MorphologicalDisambiguator::new();
    let parser = TurkishDependencyParser::new();
    let doc = AkanaEngine::new().analyze_document(text);
    let sentences = SentenceSegmenter::segment(text);
    assert_eq!(doc.sentences.len(), sentences.len());
    for (got, sent) in doc.sentences.iter().zip(sentences) {
        let tokens = TurkishTokenizer::tokenize_words(sent.text);
        assert_eq!(got.parses, disambiguator.disambiguate(&tokens));
        assert_eq!(
            got.dependency_tree.to_conllu(),
            parser.parse(&tokens).to_conllu()
        );
    }
}

#[test]
fn batches_match_single_embeddings() {
    // Large enough to be split across threads.
    let texts: Vec<String> = (0..1500)
        .map(|i| {
            format!(
                "Belge {i}: Ankara'daki toplantıda {} konu konuşuldu.",
                i % 7
            )
        })
        .collect();
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    let emb = TurkishEmbeddings::new();
    let single: Vec<Vec<f32>> = refs.iter().map(|t| emb.embed(t)).collect();
    assert_eq!(emb.embed_batch(&refs), single);
    assert_eq!(
        emb.embed_batch_packed_2bit(&refs),
        refs.iter()
            .map(|t| emb.embed_packed_2bit(t))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        emb.embed_batch_packed_1bit(&refs),
        refs.iter()
            .map(|t| emb.embed_packed_1bit(t))
            .collect::<Vec<_>>()
    );
    let hybrid = TurkishHybridEmbeddings::new();
    assert_eq!(
        hybrid.embed_hybrid_batch(&refs),
        refs.iter()
            .map(|t| hybrid.embed_hybrid(t))
            .collect::<Vec<_>>()
    );
    // Engines share the weights, so a second engine gives the same vectors.
    assert_eq!(
        TurkishEmbeddings::new().embed_batch(&refs[..10]),
        single[..10]
    );
}
