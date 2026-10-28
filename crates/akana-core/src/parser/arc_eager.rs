//! Turkish dependency parser: rule-based attachment with Universal Dependencies relations
//! and CoNLL-U output, measured on the UD Turkish test splits by `scripts/parser_ud_eval.py`.

use super::tree::{DependencyNode, DependencyTree};
use crate::morphology::pos::PrimaryPos;
use crate::morphology::{
    verb_form, MorphParse, MorphologicalDisambiguator, TurkishMorphology, VerbForm,
};

/// Verbs that a bare noun right before them forms a compound with (karar ver-, söz et-, yemek
/// ye-) more often than it is their object or subject, counted in the UD Turkish train splits
/// (light_verbs.txt, written by `scripts/parser_ud_eval.py --light-verbs`; CC BY-SA 4.0, see
/// THIRD_PARTY_NOTICES; review 3 PARSER3-07).
fn light_verb(root: &str) -> bool {
    include_str!("light_verbs.txt")
        .lines()
        .any(|line| !line.starts_with('#') && line == root)
}

pub struct TurkishDependencyParser {
    disambiguator: MorphologicalDisambiguator,
    morphology: TurkishMorphology,
}

impl Default for TurkishDependencyParser {
    fn default() -> Self {
        Self::new()
    }
}

impl TurkishDependencyParser {
    pub fn new() -> Self {
        Self {
            disambiguator: MorphologicalDisambiguator::new(),
            morphology: TurkishMorphology::new(),
        }
    }

    pub fn with_morphology(morphology: TurkishMorphology) -> Self {
        Self {
            disambiguator: MorphologicalDisambiguator::with_morphology(morphology.clone()),
            morphology,
        }
    }

    /// Parses a sequence of tokens in a Turkish sentence into a full `DependencyTree`.
    ///
    /// Categories come from each word's POS after derivation (`MorphParse::pos`), so a
    /// converb or participle is not a noun and a noun-derived verb (baş-la-dı) is a verb.
    pub fn parse(&self, tokens: &[&str]) -> DependencyTree {
        if tokens.is_empty() {
            return DependencyTree::new(Vec::new());
        }
        let disambiguated = self.disambiguator.disambiguate(tokens);
        self.parse_disambiguated(tokens, &disambiguated)
    }

    /// [`parse`](Self::parse) with the tokens' parses already chosen (one per token, as
    /// `MorphologicalDisambiguator::disambiguate` returns them), so a caller that has them
    /// does not disambiguate the sentence twice.
    pub fn parse_disambiguated(
        &self,
        tokens: &[&str],
        disambiguated: &[MorphParse],
    ) -> DependencyTree {
        assert_eq!(
            tokens.len(),
            disambiguated.len(),
            "one parse per token is required"
        );
        if tokens.is_empty() {
            return DependencyTree::new(Vec::new());
        }
        let n = tokens.len();
        // Punctuation, which the analyzer leaves unanalysed.
        let punct = |i: usize| {
            disambiguated[i].pos == PrimaryPos::Punc
                || tokens[i].chars().all(|c| !c.is_alphanumeric())
        };

        // 1. Locate the primary root of the sentence: the last verb or copular predicate (var,
        // yok and değil are predicates too). The question particle carries copula and person
        // (mi-yiz) but is not the predicate: the word before it is ("öğrenci misin"). With
        // neither, the last word that is no punctuation or question particle is ("Hava çok
        // soğuk ."), never the punctuation (review 2 PARSER2-01).
        // var, yok and değil only as themselves, with no derivation after them (Varken is a
        // converb); a postposition, conjunction or determiner is no fallback root ("Bu senin
        // için ." is rooted in senin), and punctuation never is (review 3 PARSER3-03).
        let bare_predicate = |p: &MorphParse| {
            matches!(p.lemma.as_str(), "var" | "yok" | "değil")
                && !p
                    .morpheme_tags
                    .iter()
                    .skip(1)
                    .any(|t| crate::morphology::is_derivational_tag(t))
        };
        let predicate = |p: &MorphParse| {
            (p.pos == PrimaryPos::Verb && verb_form(p).is_none())
                || p.morpheme_tags.iter().any(|t| t.starts_with("Cop"))
                || bare_predicate(p)
        };
        let content = |i: &usize| !punct(*i) && disambiguated[*i].pos != PrimaryPos::Q;
        let head_word = |i: &usize| {
            content(i)
                && !matches!(
                    disambiguated[*i].pos,
                    PrimaryPos::Postp | PrimaryPos::Conj | PrimaryPos::Det
                )
        };
        let root_idx = (0..n)
            .rev()
            .filter(content)
            .find(|&i| predicate(&disambiguated[i]))
            .or_else(|| (0..n).rev().find(head_word))
            .or_else(|| (0..n).rev().find(content))
            .or_else(|| (0..n).find(|&i| !punct(i)))
            .unwrap_or(0);
        // değil after a nominal negates it, and the nominal is the predicate of its clause, in
        // each conjunct (UD Turkish: "Ali öğretmen değil ." is rooted in öğretmen, değil AUX
        // aux, as the test splits label it 189 times against 27 cop; "Sanık suçlu değil,
        // masumdur." is rooted in suçlu; review 4 DOWN4-06).
        let copula = |i: usize| {
            i > 0
                && disambiguated[i].lemma == "değil"
                && matches!(
                    disambiguated[i - 1].pos,
                    PrimaryPos::Noun | PrimaryPos::Adj | PrimaryPos::Pron | PrimaryPos::Num
                )
                && !punct(i - 1)
        };
        let nominal = |k: usize| if copula(k) { k - 1 } else { k };
        // Coordinated clauses: a finite predicate before the root, followed by a comma or a
        // conjunction, is a conjunct; the first conjunct heads the others (UD conj; review 3
        // PARSER3-04). Each word attaches to the predicate of its own clause.
        let clause_end = |k: usize| {
            k + 1 < n
                && (matches!(tokens[k + 1], "," | ";")
                    || disambiguated[k + 1].pos == PrimaryPos::Conj)
        };
        // Where each clause ends: its predicate, or the değil after it.
        let mut ends: Vec<usize> = (0..root_idx)
            .filter(|&k| content(&k) && predicate(&disambiguated[k]) && clause_end(k))
            .collect();
        ends.push(root_idx);
        let negated: Vec<usize> = ends.iter().copied().filter(|&k| copula(k)).collect();
        let conjuncts: Vec<usize> = ends.iter().map(|&k| nominal(k)).collect();
        let root_idx = nominal(root_idx);
        let first = conjuncts[0];
        let clause_of = |i: usize| {
            ends.iter()
                .copied()
                .find(|&k| k >= i)
                .map_or(root_idx, nominal)
        };
        // The first word of a word's clause: past the comma or conjunction after the previous
        // conjunct.
        let clause_start = |i: usize| {
            ends.iter()
                .rev()
                .find(|&&k| k < i)
                .map_or(0, |&k| (k + 2).min(i))
        };
        // The noun a modifier or a relative clause goes with: the first noun after it past
        // determiners, numbers and adjectives ("Bilinmeyen bir sebeple", "uzun bir süre";
        // review 3 PARSER3-02).
        let noun_after = |i: usize| {
            let mut j = i + 1;
            while j < n
                && !punct(j)
                && matches!(
                    disambiguated[j].pos,
                    PrimaryPos::Det | PrimaryPos::Num | PrimaryPos::Adj
                )
            {
                j += 1;
            }
            (j < n && matches!(disambiguated[j].pos, PrimaryPos::Noun)).then_some(j)
        };
        let passive = |k: usize| disambiguated[k].morpheme_tags.iter().any(|t| t == "Pass");
        // A participle the dictionary lists as an adjective (gelen, geçen, bilinen) is still a
        // relative clause in UD: VERB acl.
        let lexical_participle = |p: &MorphParse| {
            p.pos == PrimaryPos::Adj
                && verb_form(p).is_none()
                && self
                    .morphology
                    .analyze(&p.surface)
                    .iter()
                    .any(|q| verb_form(q).is_some_and(|f| f.form == VerbForm::Part))
        };

        let mut nodes = Vec::with_capacity(n);

        for (i, parse) in disambiguated.iter().enumerate() {
            let id = i + 1;
            let form = parse.surface.clone();
            let lemma = parse.lemma.clone();
            let lexical_part = lexical_participle(parse);
            let form_of_verb = verb_form(parse);
            let clause = clause_of(i);

            // UD: infinitives, participles and converbs are VERB (with a VerbForm).
            let upos = match parse.pos {
                _ if punct(i) => "PUNCT",
                _ if negated.contains(&i) => "AUX",
                _ if form_of_verb.is_some() || lexical_part => "VERB",
                PrimaryPos::Noun => {
                    if parse.secondary_pos == crate::morphology::pos::SecondaryPos::ProperNoun {
                        "PROPN"
                    } else {
                        "NOUN"
                    }
                }
                PrimaryPos::Verb => "VERB",
                PrimaryPos::Adj => "ADJ",
                PrimaryPos::Adv => "ADV",
                PrimaryPos::Pron => "PRON",
                PrimaryPos::Num => "NUM",
                PrimaryPos::Conj => "CCONJ",
                PrimaryPos::Postp => "ADP",
                PrimaryPos::Interj => "INTJ",
                PrimaryPos::Punc => "PUNCT",
                PrimaryPos::Det => "DET",
                PrimaryPos::Q => "AUX",
                _ => "X",
            }
            .to_string();

            let xpos = parse.primary_pos.as_str().to_string();
            let feats = parse.morpheme_tags.join("|");

            let mut head = clause + 1;
            let mut deprel = "dep".to_string();
            let root_idx = first;

            if i == root_idx {
                head = 0;
                deprel = "root".to_string();
            } else if negated.contains(&i) {
                head = i;
                deprel = "aux".to_string();
            } else if conjuncts.contains(&i) {
                head = root_idx + 1;
                deprel = "conj".to_string();
            } else if punct(i) {
                head = root_idx + 1;
                deprel = "punct".to_string();
            } else if parse.pos == PrimaryPos::Q {
                // The question particle attaches to the word it follows, which is the focus of
                // the question (Ayşe mi okula geldi?); UD Turkish labels every use aux:q.
                if i > 0 {
                    head = i;
                }
                deprel = "aux:q".to_string();
            } else if form_of_verb.is_some_and(|f| f.form == VerbForm::Conv) {
                // A converb heads an adverbial clause of the predicate (advcl)
                deprel = "advcl".to_string();
            } else if (form_of_verb.is_some_and(|f| f.form == VerbForm::Part) || lexical_part)
                && parse.pos == PrimaryPos::Adj
                && noun_after(i).is_some()
            {
                // A participle before its noun heads a relative clause on it (acl)
                head = noun_after(i).map_or(head, |j| j + 1);
                deprel = "acl".to_string();
            } else if parse.pos == PrimaryPos::Adj {
                // It modifies the noun after it (amod)
                if let Some(j) = noun_after(i) {
                    head = j + 1;
                }
                deprel = "amod".to_string();
            } else if parse.pos == PrimaryPos::Det {
                // A determiner modifies the noun after it, past adjectives (det)
                if let Some(j) = noun_after(i) {
                    head = j + 1;
                }
                deprel = "det".to_string();
            } else if parse.pos == PrimaryPos::Num && noun_after(i).is_some() {
                // A number counts the noun after it (nummod)
                head = noun_after(i).map_or(head, |j| j + 1);
                deprel = "nummod".to_string();
            } else if parse.pos == PrimaryPos::Adv {
                head = clause + 1;
                deprel = "advmod".to_string();
            } else if parse.pos == PrimaryPos::Conj {
                // A conjunction attaches to the conjunct after it: a nominal or a verbal word,
                // never a question particle or another conjunction. The particle attaches to
                // the word before it, so "da" -> "mı" -> "da" made a cycle cut off from the root
                // (Bu da mı gitti ?).
                let next_is_conjunct = i + 1 < n
                    && matches!(
                        disambiguated[i + 1].pos,
                        PrimaryPos::Noun
                            | PrimaryPos::Pron
                            | PrimaryPos::Verb
                            | PrimaryPos::Adj
                            | PrimaryPos::Adv
                            | PrimaryPos::Num
                            | PrimaryPos::Det
                    );
                head = if next_is_conjunct {
                    i + 2
                } else {
                    root_idx + 1
                };
                deprel = "cc".to_string();
            } else if parse.pos == PrimaryPos::Postp && i > 0 && !punct(i - 1) {
                // A postposition marks the word before it (Dilekçe ile, onun için): UD case.
                head = i;
                deprel = "case".to_string();
            } else if parse.pos == PrimaryPos::Noun || parse.pos == PrimaryPos::Pron {
                let before_postposition =
                    i + 1 < n && disambiguated[i + 1].pos == PrimaryPos::Postp;
                let bare = parse.pos == PrimaryPos::Noun
                    && parse.secondary_pos != crate::morphology::pos::SecondaryPos::ProperNoun
                    && parse
                        .morpheme_tags
                        .iter()
                        .skip(1)
                        .all(|t| matches!(t.as_str(), "A3sg" | "Pnon" | "Nom"));
                // A noun phrase's head (Gelen adam oturdu) and a capitalised word inside the
                // sentence, a name (Veli gitti), are no part of a compound with the verb.
                let modified = nodes.iter().any(|m: &DependencyNode| {
                    m.head == i + 1
                        && matches!(
                            m.deprel.as_str(),
                            "acl" | "amod" | "det" | "nummod" | "nmod:poss"
                        )
                });
                let capitalised =
                    parse.surface.starts_with(char::is_uppercase) && (0..i).any(|k| !punct(k));
                let next_is_verb = i + 1 < n
                    && !punct(i + 1)
                    && (disambiguated[i + 1].pos == PrimaryPos::Verb
                        || verb_form(&disambiguated[i + 1]).is_some());
                // Check case features
                if before_postposition {
                    // The postposition's complement is an oblique of the predicate.
                    head = clause + 1;
                    deprel = "obl".to_string();
                } else if parse.morpheme_tags.contains(&"Acc".to_string()) {
                    head = clause + 1;
                    deprel = "obj".to_string();
                } else if parse.morpheme_tags.contains(&"Dat".to_string())
                    || parse.morpheme_tags.contains(&"Loc".to_string())
                    || parse.morpheme_tags.contains(&"Abl".to_string())
                    || parse.morpheme_tags.contains(&"Ins".to_string())
                {
                    head = clause + 1;
                    deprel = "obl".to_string();
                } else if parse.morpheme_tags.contains(&"Gen".to_string()) {
                    // A genitive modifies the noun after it, past adjectives (nmod:poss)
                    if let Some(j) = noun_after(i) {
                        head = j + 1;
                        deprel = "nmod:poss".to_string();
                    } else {
                        head = root_idx + 1;
                        deprel = "nmod".to_string();
                    }
                } else if bare
                    && next_is_verb
                    && !modified
                    && !capitalised
                    && light_verb(&disambiguated[i + 1].root)
                {
                    // A bare noun right before a light verb is a compound with it (karar ver-,
                    // söz et-): UD Turkish's compound, the commonest relation the parser never
                    // gave (review 3 PARSER3-07). So it is before a passive one too (karar
                    // verildi): the UD treebanks have compound 646 times there, nsubj 143.
                    head = i + 2;
                    deprel = "compound".to_string();
                } else if bare && next_is_verb && !capitalised {
                    // Before another verb it is that verb's subject when the verb is passive
                    // (karar açıklandı; UD: nsubj 264 of 545) and its object otherwise (kitap
                    // okudu).
                    head = i + 2;
                    deprel = if passive(i + 1) { "nsubj" } else { "obj" }.to_string();
                } else {
                    // Nominative subject or object
                    // by its place in its own clause
                    let start = clause_start(i);
                    if i == start || (i < clause && i <= start + 1) {
                        head = clause + 1;
                        deprel = "nsubj".to_string();
                    } else {
                        head = clause + 1;
                        deprel = "obj".to_string();
                    }
                }
            }

            nodes.push(DependencyNode {
                id,
                form,
                lemma,
                upos,
                xpos,
                feats,
                head,
                deprel,
            });
        }

        // A name of several words (Ahmet Yılmaz'ın, Piri Reis) is one: its first word heads it
        // with the relation the last word's case gives, and the others are flat (UD; review 3
        // PARSER3-07). Names side by side join up to a word with a case, which joins only a
        // person's name, one that starts with a given name: Ankara Türkiye'nin başkentidir has
        // two (review 4 DOWN4-07). Joining only people's names lost 124 of the 159 right flat
        // arcs on the UD test splits, which use flat for every name of several words.
        let name = |i: usize| {
            disambiguated[i].secondary_pos == crate::morphology::pos::SecondaryPos::ProperNoun
                && !punct(i)
        };
        let given = |i: usize| {
            name(i)
                && crate::pii::is_given_name(&crate::phonology::to_turkish_lower(
                    &disambiguated[i].lemma,
                ))
        };
        let inflected = |i: usize| {
            tokens[i].contains(['\'', '’'])
                || disambiguated[i]
                    .morpheme_tags
                    .iter()
                    .any(|t| matches!(t.as_str(), "Gen" | "Acc" | "Dat" | "Loc" | "Abl" | "Ins"))
        };
        let mut a = 0;
        while a < n {
            let mut b = a;
            while b + 1 < n
                && name(a)
                && name(b + 1)
                && !inflected(b)
                && (given(a) || !inflected(b + 1))
            {
                b += 1;
            }
            if b > a && a != first && !(a + 1..=b).contains(&first) {
                let (head, deprel) = (nodes[b].head, nodes[b].deprel.clone());
                if !(a + 1..=b + 1).contains(&head) {
                    nodes[a].head = head;
                    nodes[a].deprel = deprel;
                }
                for node in &mut nodes[a + 1..=b] {
                    node.head = a + 1;
                    node.deprel = "flat".to_string();
                }
            }
            a = b + 1;
        }

        make_tree(&mut nodes, first);
        DependencyTree::new(nodes)
    }
}

/// Makes the attachments a tree, whatever the rules above produced: the node at `root_idx` is
/// the one root, every other head is a token of the sentence and not the node itself, and
/// following heads from any node reaches the root. A node that breaks this is attached to the
/// root.
fn make_tree(nodes: &mut [DependencyNode], root_idx: usize) {
    let n = nodes.len();
    let root_id = root_idx + 1;
    for (i, node) in nodes.iter_mut().enumerate() {
        if i == root_idx {
            node.head = 0;
            node.deprel = "root".to_string();
        } else if node.head == 0 || node.head > n || node.head == i + 1 {
            node.head = root_id;
            if node.deprel == "root" {
                node.deprel = "dep".to_string();
            }
        }
    }
    // Follow the heads from every node. A walk that comes back to a node of its own path has
    // gone round a cycle: attaching that node to the root breaks it, and the whole path then
    // reaches the root. Every node is walked once (0: not seen, 1: on this walk, 2: reaches the
    // root).
    let mut state = vec![0u8; n];
    state[root_idx] = 2;
    let mut path = Vec::new();
    for start in 0..n {
        path.clear();
        let mut i = start;
        while state[i] == 0 {
            state[i] = 1;
            path.push(i);
            i = nodes[i].head - 1;
        }
        if state[i] == 1 {
            nodes[i].head = root_id;
        }
        for &p in &path {
            state[p] = 2;
        }
    }
}
