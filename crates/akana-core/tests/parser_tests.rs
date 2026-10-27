use akana_core::parser::*;

#[test]
fn test_dependency_parser() {
    let parser = TurkishDependencyParser::new();
    let tokens = vec!["Ali", "güzel", "kitabı", "okudu"];
    let tree = parser.parse(&tokens);

    assert_eq!(tree.nodes.len(), 4);
    // Main verb is the root
    let root_node = tree.nodes.iter().find(|n| n.deprel == "root").unwrap();
    assert_eq!(root_node.form, "okudu");
    assert_eq!(root_node.head, 0);

    // Subject
    let subj_node = tree.nodes.iter().find(|n| n.deprel == "nsubj").unwrap();
    assert_eq!(subj_node.form, "Ali");

    // Adjective modifier
    let amod_node = tree.nodes.iter().find(|n| n.deprel == "amod").unwrap();
    assert_eq!(amod_node.form, "güzel");
    assert_eq!(amod_node.head, 3); // modifies "kitabı"
}

/// Review 2 PARSER2-01: a sentence without a verb has its nominal predicate as the root, never
/// the punctuation; var, yok and a word before a question particle are predicates; punctuation
/// is PUNCT; a postposition marks the word before it.
#[test]
fn nominal_predicates_are_roots_and_punctuation_never_is() {
    let parser = TurkishDependencyParser::new();
    let root = |sentence: &str| {
        let tokens: Vec<&str> = sentence.split(' ').collect();
        let tree = parser.parse(&tokens);
        let root = tree.nodes.iter().find(|n| n.deprel == "root").unwrap();
        for node in tree
            .nodes
            .iter()
            .filter(|n| matches!(n.form.as_str(), "." | "?"))
        {
            assert_eq!(
                (node.upos.as_str(), node.deprel.as_str()),
                ("PUNCT", "punct")
            );
        }
        root.form.clone()
    };
    assert_eq!(root("Hava çok soğuk ."), "soğuk");
    assert_eq!(root("Bugün okulda sınav var ."), "var");
    assert_eq!(root("Evde kimse yok ."), "yok");
    assert_eq!(root("Bu karar doğru mu ?"), "doğru");
    assert_eq!(root("Ali öğretmen değil ."), "değil");
    assert_eq!(root("Sen bu akşam bize gelecek misin ?"), "gelecek");

    let tree = parser.parse(&["Dilekçe", "ile", "başvurdu", "."]);
    let ile = &tree.nodes[1];
    assert_eq!(
        (ile.upos.as_str(), ile.deprel.as_str(), ile.head),
        ("ADP", "case", 1)
    );
    assert_eq!(tree.nodes[0].deprel, "obl");
}
