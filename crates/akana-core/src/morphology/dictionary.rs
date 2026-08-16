//! Turkish Lexicon, Dictionary items, and built-in root entries.

use super::pos::{PrimaryPos, RootAttr, SecondaryPos};
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DictionaryItem {
    pub lemma: String,
    pub root: String,
    pub primary_pos: PrimaryPos,
    pub secondary_pos: SecondaryPos,
    pub attributes: RootAttr,
}

impl DictionaryItem {
    pub fn new(
        lemma: impl Into<String>,
        root: impl Into<String>,
        primary_pos: PrimaryPos,
        secondary_pos: SecondaryPos,
        attributes: RootAttr,
    ) -> Self {
        Self {
            lemma: lemma.into(),
            root: root.into(),
            primary_pos,
            secondary_pos,
            attributes,
        }
    }
}

pub struct RootLexicon {
    items_by_root: HashMap<String, Vec<DictionaryItem>>,
}

impl Default for RootLexicon {
    fn default() -> Self {
        Self::new()
    }
}

impl RootLexicon {
    pub fn new() -> Self {
        let mut lexicon = Self {
            items_by_root: HashMap::new(),
        };
        lexicon.load_builtin_lexicon();
        lexicon
    }

    pub fn add_item(&mut self, item: DictionaryItem) {
        let key = item.root.clone();
        self.items_by_root.entry(key).or_default().push(item);
    }

    pub fn get_items(&self, root: &str) -> Option<&Vec<DictionaryItem>> {
        self.items_by_root.get(root)
    }

    pub fn contains_root(&self, root: &str) -> bool {
        self.items_by_root.contains_key(root)
    }

    fn load_builtin_lexicon(&mut self) {
        // Essential core vocabulary with accurate POS and phonological flags
        let words = [
            // Nouns with Voicing (kitap -> kitaba, ağaç -> ağaca, kanat -> kanada, ayak -> ayağı)
            ("kitap", "kitap", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOICING),
            ("ağaç", "ağaç", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOICING),
            ("kanat", "kanat", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOICING),
            ("ayak", "ayak", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOICING),
            ("bebek", "bebek", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOICING),
            ("çiçek", "çiçek", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOICING),
            ("köpek", "köpek", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOICING),
            ("renk", "renk", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOICING),
            ("yurt", "yurt", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOICING),
            ("dert", "dert", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOICING),
            ("kurt", "kurt", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOICING),
            ("kalp", "kalp", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOICING | RootAttr::INVERSE_HARMONY),

            // Nouns with Vowel Drop (burun -> burnu, akıl -> aklı, şehir -> şehri)
            ("burun", "burun", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOWEL_DROP),
            ("akıl", "akıl", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOWEL_DROP),
            ("karın", "karın", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOWEL_DROP),
            ("şehir", "şehir", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOWEL_DROP),
            ("ağız", "ağız", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOWEL_DROP),
            ("alın", "alın", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOWEL_DROP),
            ("oğul", "oğul", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOWEL_DROP),
            ("fikir", "fikir", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOWEL_DROP),
            ("resim", "resim", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOWEL_DROP),
            ("vakit", "vakit", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOWEL_DROP | RootAttr::VOICING),

            // Nouns with Consonant Doubling (hak -> hakkı, his -> hissi, af -> affı)
            ("hak", "hak", PrimaryPos::Noun, SecondaryPos::None, RootAttr::CONSONANT_DOUBLING),
            ("his", "his", PrimaryPos::Noun, SecondaryPos::None, RootAttr::CONSONANT_DOUBLING),
            ("af", "af", PrimaryPos::Noun, SecondaryPos::None, RootAttr::CONSONANT_DOUBLING),
            ("hat", "hat", PrimaryPos::Noun, SecondaryPos::None, RootAttr::CONSONANT_DOUBLING),
            ("sır", "sır", PrimaryPos::Noun, SecondaryPos::None, RootAttr::CONSONANT_DOUBLING),
            ("zam", "zam", PrimaryPos::Noun, SecondaryPos::None, RootAttr::CONSONANT_DOUBLING),

            // Loanwords with Inverse Harmony (saat -> saatler, alkol -> alkolü)
            ("saat", "saat", PrimaryPos::Noun, SecondaryPos::Time, RootAttr::INVERSE_HARMONY),
            ("alkol", "alkol", PrimaryPos::Noun, SecondaryPos::None, RootAttr::INVERSE_HARMONY),
            ("rol", "rol", PrimaryPos::Noun, SecondaryPos::None, RootAttr::INVERSE_HARMONY),
            ("gol", "gol", PrimaryPos::Noun, SecondaryPos::None, RootAttr::INVERSE_HARMONY),
            ("petrol", "petrol", PrimaryPos::Noun, SecondaryPos::None, RootAttr::INVERSE_HARMONY),
            ("sembol", "sembol", PrimaryPos::Noun, SecondaryPos::None, RootAttr::INVERSE_HARMONY),
            ("harf", "harf", PrimaryPos::Noun, SecondaryPos::None, RootAttr::INVERSE_HARMONY),

            // Regular Nouns
            ("ev", "ev", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("araba", "araba", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("kapı", "kapı", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("kedi", "kedi", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("masa", "masa", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("oda", "oda", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("okul", "okul", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("sınıf", "sınıf", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("yol", "yol", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("gün", "gün", PrimaryPos::Noun, SecondaryPos::Time, RootAttr::NONE),
            ("yıl", "yıl", PrimaryPos::Noun, SecondaryPos::Time, RootAttr::NONE),
            ("hafta", "hafta", PrimaryPos::Noun, SecondaryPos::Time, RootAttr::NONE),
            ("ay", "ay", PrimaryPos::Noun, SecondaryPos::Time, RootAttr::NONE),
            ("gece", "gece", PrimaryPos::Noun, SecondaryPos::Time, RootAttr::NONE),
            ("sabah", "sabah", PrimaryPos::Noun, SecondaryPos::Time, RootAttr::NONE),
            ("akşam", "akşam", PrimaryPos::Noun, SecondaryPos::Time, RootAttr::NONE),
            ("su", "su", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("adam", "adam", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("kadın", "kadın", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("çocuk", "çocuk", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOICING),
            ("arkadaş", "arkadaş", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("kardeş", "kardeş", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("insan", "insan", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("deniz", "deniz", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("göz", "göz", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("el", "el", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("kol", "kol", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("baş", "baş", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("dil", "dil", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("yazılım", "yazılım", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),

            // Proper Nouns
            ("türkiye", "türkiye", PrimaryPos::Noun, SecondaryPos::ProperNoun, RootAttr::PROPER_NOUN),
            ("türk", "türk", PrimaryPos::Noun, SecondaryPos::ProperNoun, RootAttr::PROPER_NOUN),
            ("türkçe", "türkçe", PrimaryPos::Noun, SecondaryPos::ProperNoun, RootAttr::PROPER_NOUN),
            ("ankara", "ankara", PrimaryPos::Noun, SecondaryPos::ProperNoun, RootAttr::PROPER_NOUN),
            ("istanbul", "istanbul", PrimaryPos::Noun, SecondaryPos::ProperNoun, RootAttr::PROPER_NOUN),
            ("izmir", "izmir", PrimaryPos::Noun, SecondaryPos::ProperNoun, RootAttr::PROPER_NOUN),
            ("ahmet", "ahmet", PrimaryPos::Noun, SecondaryPos::ProperNoun, RootAttr::PROPER_NOUN | RootAttr::VOICING),
            ("mehmet", "mehmet", PrimaryPos::Noun, SecondaryPos::ProperNoun, RootAttr::PROPER_NOUN | RootAttr::VOICING),
            ("ali", "ali", PrimaryPos::Noun, SecondaryPos::ProperNoun, RootAttr::PROPER_NOUN),
            ("ayşe", "ayşe", PrimaryPos::Noun, SecondaryPos::ProperNoun, RootAttr::PROPER_NOUN),
            ("fatma", "fatma", PrimaryPos::Noun, SecondaryPos::ProperNoun, RootAttr::PROPER_NOUN),
            ("akana", "akana", PrimaryPos::Noun, SecondaryPos::ProperNoun, RootAttr::PROPER_NOUN),

            // Verbs
            ("gel", "gel", PrimaryPos::Verb, SecondaryPos::None, RootAttr::NONE),
            ("git", "git", PrimaryPos::Verb, SecondaryPos::None, RootAttr::VOICING), // git -> gidiyor
            ("et", "et", PrimaryPos::Verb, SecondaryPos::None, RootAttr::VOICING),   // et -> ediyor
            ("tat", "tat", PrimaryPos::Verb, SecondaryPos::None, RootAttr::VOICING), // tat -> tadıyor
            ("güt", "güt", PrimaryPos::Verb, SecondaryPos::None, RootAttr::VOICING),
            ("yap", "yap", PrimaryPos::Verb, SecondaryPos::None, RootAttr::NONE),
            ("al", "al", PrimaryPos::Verb, SecondaryPos::None, RootAttr::NONE),
            ("ver", "ver", PrimaryPos::Verb, SecondaryPos::None, RootAttr::NONE),
            ("ol", "ol", PrimaryPos::Verb, SecondaryPos::None, RootAttr::NONE),
            ("kal", "kal", PrimaryPos::Verb, SecondaryPos::None, RootAttr::NONE),
            ("bak", "bak", PrimaryPos::Verb, SecondaryPos::None, RootAttr::NONE),
            ("gör", "gör", PrimaryPos::Verb, SecondaryPos::None, RootAttr::NONE),
            ("sev", "sev", PrimaryPos::Verb, SecondaryPos::None, RootAttr::NONE),
            ("yaz", "yaz", PrimaryPos::Verb, SecondaryPos::None, RootAttr::NONE),
            ("oku", "oku", PrimaryPos::Verb, SecondaryPos::None, RootAttr::NONE),
            ("bil", "bil", PrimaryPos::Verb, SecondaryPos::None, RootAttr::NONE),
            ("bul", "bul", PrimaryPos::Verb, SecondaryPos::None, RootAttr::NONE),
            ("koş", "koş", PrimaryPos::Verb, SecondaryPos::None, RootAttr::NONE),
            ("dur", "dur", PrimaryPos::Verb, SecondaryPos::None, RootAttr::NONE),
            ("ye", "ye", PrimaryPos::Verb, SecondaryPos::None, RootAttr::VOWEL_MUTATION), // ye -> yiyor
            ("de", "de", PrimaryPos::Verb, SecondaryPos::None, RootAttr::VOWEL_MUTATION), // de -> diyor

            // Adjectives
            ("güzel", "güzel", PrimaryPos::Adj, SecondaryPos::None, RootAttr::NONE),
            ("iyi", "iyi", PrimaryPos::Adj, SecondaryPos::None, RootAttr::NONE),
            ("kötü", "kötü", PrimaryPos::Adj, SecondaryPos::None, RootAttr::NONE),
            ("büyük", "büyük", PrimaryPos::Adj, SecondaryPos::None, RootAttr::VOICING),
            ("küçük", "küçük", PrimaryPos::Adj, SecondaryPos::None, RootAttr::VOICING),
            ("yeni", "yeni", PrimaryPos::Adj, SecondaryPos::None, RootAttr::NONE),
            ("eski", "eski", PrimaryPos::Adj, SecondaryPos::None, RootAttr::NONE),
            ("hızlı", "hızlı", PrimaryPos::Adj, SecondaryPos::None, RootAttr::NONE),
            ("yavaş", "yavaş", PrimaryPos::Adj, SecondaryPos::None, RootAttr::NONE),
            ("zor", "zor", PrimaryPos::Adj, SecondaryPos::None, RootAttr::NONE),
            ("kolay", "kolay", PrimaryPos::Adj, SecondaryPos::None, RootAttr::NONE),
            ("ak", "ak", PrimaryPos::Adj, SecondaryPos::None, RootAttr::NONE),
            ("kara", "kara", PrimaryPos::Adj, SecondaryPos::None, RootAttr::NONE),

            // Pronouns
            ("ben", "ben", PrimaryPos::Pron, SecondaryPos::Personal, RootAttr::NONE),
            ("sen", "sen", PrimaryPos::Pron, SecondaryPos::Personal, RootAttr::NONE),
            ("o", "o", PrimaryPos::Pron, SecondaryPos::Personal, RootAttr::NONE),
            ("biz", "biz", PrimaryPos::Pron, SecondaryPos::Personal, RootAttr::NONE),
            ("siz", "siz", PrimaryPos::Pron, SecondaryPos::Personal, RootAttr::NONE),
            ("onlar", "onlar", PrimaryPos::Pron, SecondaryPos::Personal, RootAttr::NONE),
            ("bu", "bu", PrimaryPos::Pron, SecondaryPos::Demonstrative, RootAttr::NONE),
            ("şu", "şu", PrimaryPos::Pron, SecondaryPos::Demonstrative, RootAttr::NONE),
            ("kendi", "kendi", PrimaryPos::Pron, SecondaryPos::Reflexive, RootAttr::NONE),
            ("kim", "kim", PrimaryPos::Pron, SecondaryPos::Question, RootAttr::NONE),
            ("ne", "ne", PrimaryPos::Pron, SecondaryPos::Question, RootAttr::NONE),

            // Adverbs
            ("çok", "çok", PrimaryPos::Adv, SecondaryPos::Quantitive, RootAttr::NONE),
            ("az", "az", PrimaryPos::Adv, SecondaryPos::Quantitive, RootAttr::NONE),
            ("daha", "daha", PrimaryPos::Adv, SecondaryPos::None, RootAttr::NONE),
            ("en", "en", PrimaryPos::Adv, SecondaryPos::None, RootAttr::NONE),
            ("hiç", "hiç", PrimaryPos::Adv, SecondaryPos::None, RootAttr::NONE),
            ("hep", "hep", PrimaryPos::Adv, SecondaryPos::None, RootAttr::NONE),
            ("şimdi", "şimdi", PrimaryPos::Adv, SecondaryPos::Time, RootAttr::NONE),
            ("sonra", "sonra", PrimaryPos::Adv, SecondaryPos::Time, RootAttr::NONE),
            ("önce", "önce", PrimaryPos::Adv, SecondaryPos::Time, RootAttr::NONE),

            // Conjunctions & Postpositions
            ("ve", "ve", PrimaryPos::Conj, SecondaryPos::None, RootAttr::NONE),
            ("ile", "ile", PrimaryPos::Conj, SecondaryPos::None, RootAttr::NONE),
            ("veya", "veya", PrimaryPos::Conj, SecondaryPos::None, RootAttr::NONE),
            ("ama", "ama", PrimaryPos::Conj, SecondaryPos::None, RootAttr::NONE),
            ("fakat", "fakat", PrimaryPos::Conj, SecondaryPos::None, RootAttr::NONE),
            ("ancak", "ancak", PrimaryPos::Conj, SecondaryPos::None, RootAttr::NONE),
            ("çünkü", "çünkü", PrimaryPos::Conj, SecondaryPos::None, RootAttr::NONE),
            ("için", "için", PrimaryPos::Postp, SecondaryPos::None, RootAttr::NONE),
            ("gibi", "gibi", PrimaryPos::Postp, SecondaryPos::None, RootAttr::NONE),
            ("kadar", "kadar", PrimaryPos::Postp, SecondaryPos::None, RootAttr::NONE),
            ("göre", "göre", PrimaryPos::Postp, SecondaryPos::None, RootAttr::NONE),

            // Numerals
            ("bir", "bir", PrimaryPos::Num, SecondaryPos::Cardinal, RootAttr::NONE),
            ("iki", "iki", PrimaryPos::Num, SecondaryPos::Cardinal, RootAttr::NONE),
            ("üç", "üç", PrimaryPos::Num, SecondaryPos::Cardinal, RootAttr::NONE),
            ("dört", "dört", PrimaryPos::Num, SecondaryPos::Cardinal, RootAttr::VOICING),
            ("beş", "beş", PrimaryPos::Num, SecondaryPos::Cardinal, RootAttr::NONE),
            ("altı", "altı", PrimaryPos::Num, SecondaryPos::Cardinal, RootAttr::NONE),
            ("yedi", "yedi", PrimaryPos::Num, SecondaryPos::Cardinal, RootAttr::NONE),
            ("sekiz", "sekiz", PrimaryPos::Num, SecondaryPos::Cardinal, RootAttr::NONE),
            ("dokuz", "dokuz", PrimaryPos::Num, SecondaryPos::Cardinal, RootAttr::NONE),
            ("on", "on", PrimaryPos::Num, SecondaryPos::Cardinal, RootAttr::NONE),
            ("yüz", "yüz", PrimaryPos::Num, SecondaryPos::Cardinal, RootAttr::NONE),
            ("bin", "bin", PrimaryPos::Num, SecondaryPos::Cardinal, RootAttr::NONE),
            ("milyon", "milyon", PrimaryPos::Num, SecondaryPos::Cardinal, RootAttr::NONE),
        ];

        for (lemma, root, primary, sec, attr) in words {
            self.add_item(DictionaryItem::new(lemma, root, primary, sec, attr));
        }
    }
}
