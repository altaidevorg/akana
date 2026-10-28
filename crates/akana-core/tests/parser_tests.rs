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
    // değil negates the nominal before it, which is the predicate, as in UD Turkish (review 3
    // PARSER3-03); UD labels it aux (review 4 DOWN4-06).
    assert_eq!(root("Ali öğretmen değil ."), "öğretmen");
    let tree = parser.parse(&["Ali", "öğretmen", "değil", "."]);
    assert_eq!(
        (
            tree.nodes[2].upos.as_str(),
            tree.nodes[2].deprel.as_str(),
            tree.nodes[2].head
        ),
        ("AUX", "aux", 2)
    );
    // A postposition is no root, nor punctuation (review 3 PARSER3-03).
    assert_eq!(root("Bu senin için ."), "senin");
    assert_eq!(root("mi ?"), "mi");
    assert_eq!(root("Sen bu akşam bize gelecek misin ?"), "gelecek");

    let tree = parser.parse(&["Dilekçe", "ile", "başvurdu", "."]);
    let ile = &tree.nodes[1];
    assert_eq!(
        (ile.upos.as_str(), ile.deprel.as_str(), ile.head),
        ("ADP", "case", 1)
    );
    assert_eq!(tree.nodes[0].deprel, "obl");
}

/// Review 2 PARSER2-02: every converb the analyzer emits heads an adverbial clause, every
/// participle before a noun a relative clause.
#[test]
fn every_non_finite_form_is_a_verb() {
    let parser = TurkishDependencyParser::new();
    let node = |sentence: &str, form: &str| {
        let tokens: Vec<&str> = sentence.split(' ').collect();
        let tree = parser.parse(&tokens);
        let n = tree.nodes.iter().find(|n| n.form == form).unwrap().clone();
        (n.upos, n.deprel)
    };
    assert_eq!(
        node("Güldükçe güzelleşiyor .", "Güldükçe"),
        ("VERB".to_string(), "advcl".to_string())
    );
    assert_eq!(
        node("Kaçarcasına gitti .", "Kaçarcasına"),
        ("VERB".to_string(), "advcl".to_string())
    );
    assert_eq!(
        node("Kırılmış camları topladı .", "Kırılmış"),
        ("VERB".to_string(), "acl".to_string())
    );
}

/// Review 3 PARSER3-02/04/07: relations measured on UD Turkish (scripts/parser_ud_eval.py).
#[test]
fn compounds_conjuncts_names_and_relative_clauses() {
    let parser = TurkishDependencyParser::new();
    // Each word's (form, head, relation).
    let arcs = |sentence: &str| -> Vec<(String, usize, String)> {
        let tokens: Vec<&str> = sentence.split(' ').collect();
        let tree = parser.parse(&tokens);
        tree.nodes
            .iter()
            .map(|n| (n.form.clone(), n.head, n.deprel.clone()))
            .collect()
    };
    let arc = |sentence: &str, form: &str| {
        let found = arcs(sentence).into_iter().find(|a| a.0 == form).unwrap();
        (found.1, found.2)
    };
    // A bare noun before a light verb is a compound with it; before another passive verb its
    // subject, before another active one its object.
    assert_eq!(
        arc("Mahkeme davanın reddine karar verdi .", "karar"),
        (5, "compound".to_string())
    );
    assert_eq!(
        arc("Sonra karar açıklandı .", "karar"),
        (3, "nsubj".to_string())
    );
    assert_eq!(arc("Ali kitap okudu .", "kitap"), (3, "obj".to_string()));
    // Coordinated clauses: the first conjunct heads the second, and a name opening the second
    // clause is its subject.
    for sentence in ["Ali geldi , Veli gitti .", "Ali geldi ve Veli gitti ."] {
        assert_eq!(arc(sentence, "geldi"), (0, "root".to_string()));
        assert_eq!(arc(sentence, "gitti"), (2, "conj".to_string()));
        assert_eq!(arc(sentence, "Veli"), (5, "nsubj".to_string()));
    }
    // A name of several words is one, headed by its first word.
    assert_eq!(
        arc("Ahmet Yılmaz'ın kararı onandı .", "Yılmaz'ın"),
        (1, "flat".to_string())
    );
    // A relative clause reaches its noun past a determiner; a number counts its noun.
    assert_eq!(
        arc("Bilinmeyen bir sebeple gitti .", "Bilinmeyen"),
        (3, "acl".to_string())
    );
    assert_eq!(
        arc("İki yıl önce taşındı .", "İki"),
        (2, "nummod".to_string())
    );
}

/// Review 4 DOWN4-06/07/08: değil in each conjunct; names side by side join up to a case,
/// which joins only a person's name; a bare noun before a passive verb is its subject, but a
/// light verb's compound, as the UD treebanks mostly have it.
#[test]
fn conjunct_negation_person_names_and_passive_subjects() {
    let parser = TurkishDependencyParser::new();
    let parse = |sentence: &str| {
        let tokens: Vec<&str> = sentence.split(' ').collect();
        parser.parse(&tokens)
    };
    let rel = |tree: &DependencyTree, form: &str| {
        let node = tree.nodes.iter().find(|n| n.form == form).unwrap();
        let head = match node.head {
            0 => "ROOT".to_string(),
            h => tree.nodes[h - 1].form.clone(),
        };
        (node.deprel.clone(), head)
    };
    let tree = parse("Sanık suçlu değil , masumdur .");
    assert_eq!(rel(&tree, "suçlu"), ("root".into(), "ROOT".into()));
    assert_eq!(rel(&tree, "değil"), ("aux".into(), "suçlu".into()));
    assert_eq!(rel(&tree, "masumdur"), ("conj".into(), "suçlu".into()));
    let tree = parse("Mustafa Kemal Atatürk geldi .");
    assert_eq!(rel(&tree, "Kemal"), ("flat".into(), "Mustafa".into()));
    assert_eq!(rel(&tree, "Atatürk"), ("flat".into(), "Mustafa".into()));
    let tree = parse("Ankara Türkiye'nin başkentidir .");
    assert_ne!(rel(&tree, "Türkiye'nin").0, "flat");
    assert_ne!(rel(&tree, "Ankara").0, "nmod:poss");
    let tree = parse("Ahmet Yılmaz'ın kitabı geldi .");
    assert_eq!(rel(&tree, "Yılmaz'ın"), ("flat".into(), "Ahmet".into()));
    let tree = parse("Karar açıklandı .");
    assert_eq!(rel(&tree, "Karar").0, "nsubj");
    let tree = parse("Karar verildi .");
    assert_eq!(rel(&tree, "Karar").0, "compound");
}
