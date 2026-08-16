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
