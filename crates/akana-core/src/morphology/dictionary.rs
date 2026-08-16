//! Turkish Lexicon, Dictionary items, prefix indexing, and built-in modern vocabulary.

use super::pos::{PrimaryPos, RootAttr, SecondaryPos};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
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

    pub fn empty() -> Self {
        Self {
            items_by_root: HashMap::new(),
        }
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

    pub fn count(&self) -> usize {
        self.items_by_root.values().map(|v| v.len()).sum()
    }

    /// Loads custom dictionary lines in format:
    /// `lemma root POS [SecondaryPOS] [Attributes]`
    /// e.g. `bilgisayar bilgisayar Noun` or `kitap kitap Noun Voicing`
    pub fn load_from_str(&mut self, text: &str) {
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 3 {
                continue;
            }

            let lemma = parts[0];
            let root = parts[1];
            let primary_pos = match parts[2].to_lowercase().as_str() {
                "noun" => PrimaryPos::Noun,
                "verb" => PrimaryPos::Verb,
                "adj" => PrimaryPos::Adj,
                "adv" => PrimaryPos::Adv,
                "pron" => PrimaryPos::Pron,
                "num" => PrimaryPos::Num,
                "conj" => PrimaryPos::Conj,
                "postp" => PrimaryPos::Postp,
                "interj" => PrimaryPos::Interj,
                "q" => PrimaryPos::Q,
                _ => PrimaryPos::Noun,
            };

            let mut sec_pos = SecondaryPos::None;
            let mut attrs = RootAttr::NONE;

            for &p in &parts[3..] {
                match p.to_lowercase().as_str() {
                    "prop" | "proper" => sec_pos = SecondaryPos::ProperNoun,
                    "time" => sec_pos = SecondaryPos::Time,
                    "voicing" => attrs |= RootAttr::VOICING,
                    "voweldrop" => attrs |= RootAttr::VOWEL_DROP,
                    "doubling" => attrs |= RootAttr::CONSONANT_DOUBLING,
                    "inverseharmony" => attrs |= RootAttr::INVERSE_HARMONY,
                    "propernoun" => attrs |= RootAttr::PROPER_NOUN,
                    "vowelmutation" => attrs |= RootAttr::VOWEL_MUTATION,
                    _ => {}
                }
            }

            self.add_item(DictionaryItem::new(lemma, root, primary_pos, sec_pos, attrs));
        }
    }

    /// Loads dictionary items from a text file.
    pub fn load_from_file<P: AsRef<Path>>(&mut self, path: P) -> std::io::Result<()> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        for line in reader.lines() {
            let l = line?;
            self.load_from_str(&l);
        }
        Ok(())
    }

    fn load_builtin_lexicon(&mut self) {
        // High-coverage modern Turkish vocabulary across multiple domains
        let words = [
            // --- Nouns with Voicing (p->b, ç->c, t->d, k->ğ/g) ---
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
            ("ekmek", "ekmek", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOICING),
            ("kapak", "kapak", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOICING),
            ("tabak", "tabak", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOICING),
            ("bardak", "bardak", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOICING),
            ("çorap", "çorap", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOICING),
            ("kulak", "kulak", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOICING),
            ("dolap", "dolap", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOICING),
            ("hesap", "hesap", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOICING),
            ("sebep", "sebep", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOICING),
            ("cevap", "cevap", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOICING),
            ("kasap", "kasap", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOICING),
            ("kırbaç", "kırbaç", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOICING),
            ("araç", "araç", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOICING),
            ("amaç", "amaç", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOICING),
            ("sonuç", "sonuç", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOICING),
            ("inanç", "inanç", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOICING),
            ("kazanç", "kazanç", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOICING),
            ("sevinç", "sevinç", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOICING),
            ("korkunç", "korkunç", PrimaryPos::Adj, SecondaryPos::None, RootAttr::VOICING),
            ("bayrak", "bayrak", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOICING),
            ("toprak", "toprak", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOICING),
            ("yaprak", "yaprak", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOICING),
            ("sokak", "sokak", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOICING),
            ("kaynak", "kaynak", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOICING),
            ("örnek", "örnek", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOICING),
            ("ördek", "ördek", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOICING),
            ("böcek", "böcek", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOICING),
            ("etek", "etek", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOICING),
            ("gömlek", "gömlek", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOICING),
            ("bilek", "bilek", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOICING),
            ("kemik", "kemik", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOICING),
            ("gözlük", "gözlük", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOICING),
            ("sözlük", "sözlük", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOICING),
            ("yüzük", "yüzük", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOICING),
            ("kütük", "kütük", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOICING),
            ("türkü", "türkü", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("şarkı", "şarkı", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),

            // --- Nouns with Vowel Drop (burun -> burnu, akıl -> aklı, şehir -> şehri) ---
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
            ("isim", "isim", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOWEL_DROP),
            ("cisim", "cisim", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOWEL_DROP),
            ("nesil", "nesil", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOWEL_DROP),
            ("asıl", "asıl", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOWEL_DROP),
            ("sabır", "sabır", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOWEL_DROP),
            ("omuz", "omuz", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOWEL_DROP),
            ("göğüs", "göğüs", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOWEL_DROP),
            ("boyun", "boyun", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOWEL_DROP),
            ("bağır", "bağır", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOWEL_DROP),
            ("zehir", "zehir", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOWEL_DROP),
            ("nehir", "nehir", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOWEL_DROP),
            ("emir", "emir", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOWEL_DROP),
            ("devir", "devir", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOWEL_DROP),
            ("tavır", "tavır", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOWEL_DROP),
            ("kavim", "kavim", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOWEL_DROP),
            ("metin", "metin", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOWEL_DROP),

            // --- Nouns with Consonant Doubling (hak -> hakkı, his -> hissi, af -> affı) ---
            ("hak", "hak", PrimaryPos::Noun, SecondaryPos::None, RootAttr::CONSONANT_DOUBLING),
            ("his", "his", PrimaryPos::Noun, SecondaryPos::None, RootAttr::CONSONANT_DOUBLING),
            ("af", "af", PrimaryPos::Noun, SecondaryPos::None, RootAttr::CONSONANT_DOUBLING),
            ("hat", "hat", PrimaryPos::Noun, SecondaryPos::None, RootAttr::CONSONANT_DOUBLING),
            ("sır", "sır", PrimaryPos::Noun, SecondaryPos::None, RootAttr::CONSONANT_DOUBLING),
            ("zam", "zam", PrimaryPos::Noun, SecondaryPos::None, RootAttr::CONSONANT_DOUBLING),
            ("had", "had", PrimaryPos::Noun, SecondaryPos::None, RootAttr::CONSONANT_DOUBLING),
            ("tıp", "tıp", PrimaryPos::Noun, SecondaryPos::None, RootAttr::CONSONANT_DOUBLING | RootAttr::VOICING),
            ("ret", "ret", PrimaryPos::Noun, SecondaryPos::None, RootAttr::CONSONANT_DOUBLING | RootAttr::VOICING),
            ("zam", "zam", PrimaryPos::Noun, SecondaryPos::None, RootAttr::CONSONANT_DOUBLING),
            ("haz", "haz", PrimaryPos::Noun, SecondaryPos::None, RootAttr::CONSONANT_DOUBLING),
            ("şan", "şan", PrimaryPos::Noun, SecondaryPos::None, RootAttr::CONSONANT_DOUBLING),
            ("şer", "şer", PrimaryPos::Noun, SecondaryPos::None, RootAttr::CONSONANT_DOUBLING),
            ("fen", "fen", PrimaryPos::Noun, SecondaryPos::None, RootAttr::CONSONANT_DOUBLING),

            // --- Loanwords with Inverse Harmony (saat -> saatler, alkol -> alkolü) ---
            ("saat", "saat", PrimaryPos::Noun, SecondaryPos::Time, RootAttr::INVERSE_HARMONY),
            ("alkol", "alkol", PrimaryPos::Noun, SecondaryPos::None, RootAttr::INVERSE_HARMONY),
            ("rol", "rol", PrimaryPos::Noun, SecondaryPos::None, RootAttr::INVERSE_HARMONY),
            ("gol", "gol", PrimaryPos::Noun, SecondaryPos::None, RootAttr::INVERSE_HARMONY),
            ("petrol", "petrol", PrimaryPos::Noun, SecondaryPos::None, RootAttr::INVERSE_HARMONY),
            ("sembol", "sembol", PrimaryPos::Noun, SecondaryPos::None, RootAttr::INVERSE_HARMONY),
            ("harf", "harf", PrimaryPos::Noun, SecondaryPos::None, RootAttr::INVERSE_HARMONY),
            ("kontrol", "kontrol", PrimaryPos::Noun, SecondaryPos::None, RootAttr::INVERSE_HARMONY),
            ("santral", "santral", PrimaryPos::Noun, SecondaryPos::None, RootAttr::INVERSE_HARMONY),
            ("general", "general", PrimaryPos::Noun, SecondaryPos::None, RootAttr::INVERSE_HARMONY),
            ("moral", "moral", PrimaryPos::Noun, SecondaryPos::None, RootAttr::INVERSE_HARMONY),
            ("helal", "helal", PrimaryPos::Noun, SecondaryPos::None, RootAttr::INVERSE_HARMONY),
            ("hilal", "hilal", PrimaryPos::Noun, SecondaryPos::None, RootAttr::INVERSE_HARMONY),
            ("kemal", "kemal", PrimaryPos::Noun, SecondaryPos::None, RootAttr::INVERSE_HARMONY),
            ("cemal", "cemal", PrimaryPos::Noun, SecondaryPos::None, RootAttr::INVERSE_HARMONY),
            ("hayal", "hayal", PrimaryPos::Noun, SecondaryPos::None, RootAttr::INVERSE_HARMONY),
            ("meal", "meal", PrimaryPos::Noun, SecondaryPos::None, RootAttr::INVERSE_HARMONY),
            ("sual", "sual", PrimaryPos::Noun, SecondaryPos::None, RootAttr::INVERSE_HARMONY),
            ("misal", "misal", PrimaryPos::Noun, SecondaryPos::None, RootAttr::INVERSE_HARMONY),
            ("itidal", "itidal", PrimaryPos::Noun, SecondaryPos::None, RootAttr::INVERSE_HARMONY),
            ("usul", "usul", PrimaryPos::Noun, SecondaryPos::None, RootAttr::INVERSE_HARMONY),
            ("kabul", "kabul", PrimaryPos::Noun, SecondaryPos::None, RootAttr::INVERSE_HARMONY),
            ("mahsul", "mahsul", PrimaryPos::Noun, SecondaryPos::None, RootAttr::INVERSE_HARMONY),
            ("meçhul", "meçhul", PrimaryPos::Noun, SecondaryPos::None, RootAttr::INVERSE_HARMONY),
            ("meşgul", "meşgul", PrimaryPos::Noun, SecondaryPos::None, RootAttr::INVERSE_HARMONY),
            ("makul", "makul", PrimaryPos::Noun, SecondaryPos::None, RootAttr::INVERSE_HARMONY),

            // --- Tech, Science, AI & Digital Modern Terms ---
            ("yazılım", "yazılım", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("donanım", "donanım", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("bilişim", "bilişim", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("bilgisayar", "bilgisayar", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("internet", "internet", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("ağ", "ağ", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("veri", "veri", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("veritabanı", "veritabanı", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("sunucu", "sunucu", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("istemci", "istemci", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("yapay", "yapay", PrimaryPos::Adj, SecondaryPos::None, RootAttr::NONE),
            ("zeka", "zeka", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("algoritma", "algoritma", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("model", "model", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("kod", "kod", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("program", "program", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("uygulama", "uygulama", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("geliştirici", "geliştirici", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("arayüz", "arayüz", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("sistem", "sistem", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("işlemci", "işlemci", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("bellek", "bellek", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOICING),
            ("ekran", "ekran", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("kamera", "kamera", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("sensör", "sensör", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("robot", "robot", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("otomasyon", "otomasyon", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("bulut", "bulut", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),

            // --- Regular Nouns ---
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
            ("anne", "anne", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("baba", "baba", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("öğrenci", "öğrenci", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("öğretmen", "öğretmen", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("doktor", "doktor", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("hastane", "hastane", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("eczane", "eczane", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("ilaç", "ilaç", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOICING),
            ("hasta", "hasta", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("sağlık", "sağlık", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOICING),
            ("devlet", "devlet", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("hükümet", "hükümet", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("kanun", "kanun", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("yasa", "yasa", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("mahkeme", "mahkeme", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("avukat", "avukat", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("hâkim", "hâkim", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("savcı", "savcı", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("sanat", "sanat", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("edebiyat", "edebiyat", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("müzik", "müzik", PrimaryPos::Noun, SecondaryPos::None, RootAttr::VOICING),
            ("sinema", "sinema", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("tiyatro", "tiyatro", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("tarih", "tarih", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("coğrafya", "coğrafya", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("matematik", "matematik", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("fizik", "fizik", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("kimya", "kimya", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("biyoloji", "biyoloji", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("felsefe", "felsefe", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("sosyoloji", "sosyoloji", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("psikoloji", "psikoloji", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("ekonomi", "ekonomi", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("para", "para", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("banka", "banka", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("şirket", "şirket", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("fabrika", "fabrika", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("iş", "iş", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("işçi", "işçi", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("patron", "patron", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("müdür", "müdür", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("memur", "memur", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),

            // --- Proper Nouns ---
            ("türkiye", "türkiye", PrimaryPos::Noun, SecondaryPos::ProperNoun, RootAttr::PROPER_NOUN),
            ("türk", "türk", PrimaryPos::Noun, SecondaryPos::ProperNoun, RootAttr::PROPER_NOUN),
            ("türkçe", "türkçe", PrimaryPos::Noun, SecondaryPos::ProperNoun, RootAttr::PROPER_NOUN),
            ("ankara", "ankara", PrimaryPos::Noun, SecondaryPos::ProperNoun, RootAttr::PROPER_NOUN),
            ("istanbul", "istanbul", PrimaryPos::Noun, SecondaryPos::ProperNoun, RootAttr::PROPER_NOUN),
            ("izmir", "izmir", PrimaryPos::Noun, SecondaryPos::ProperNoun, RootAttr::PROPER_NOUN),
            ("bursa", "bursa", PrimaryPos::Noun, SecondaryPos::ProperNoun, RootAttr::PROPER_NOUN),
            ("antalya", "antalya", PrimaryPos::Noun, SecondaryPos::ProperNoun, RootAttr::PROPER_NOUN),
            ("adana", "adana", PrimaryPos::Noun, SecondaryPos::ProperNoun, RootAttr::PROPER_NOUN),
            ("trabzon", "trabzon", PrimaryPos::Noun, SecondaryPos::ProperNoun, RootAttr::PROPER_NOUN),
            ("konya", "konya", PrimaryPos::Noun, SecondaryPos::ProperNoun, RootAttr::PROPER_NOUN),
            ("ahmet", "ahmet", PrimaryPos::Noun, SecondaryPos::ProperNoun, RootAttr::PROPER_NOUN | RootAttr::VOICING),
            ("mehmet", "mehmet", PrimaryPos::Noun, SecondaryPos::ProperNoun, RootAttr::PROPER_NOUN | RootAttr::VOICING),
            ("ali", "ali", PrimaryPos::Noun, SecondaryPos::ProperNoun, RootAttr::PROPER_NOUN),
            ("ayşe", "ayşe", PrimaryPos::Noun, SecondaryPos::ProperNoun, RootAttr::PROPER_NOUN),
            ("fatma", "fatma", PrimaryPos::Noun, SecondaryPos::ProperNoun, RootAttr::PROPER_NOUN),
            ("mustafa", "mustafa", PrimaryPos::Noun, SecondaryPos::ProperNoun, RootAttr::PROPER_NOUN),
            ("kemal", "kemal", PrimaryPos::Noun, SecondaryPos::ProperNoun, RootAttr::PROPER_NOUN),
            ("atatürk", "atatürk", PrimaryPos::Noun, SecondaryPos::ProperNoun, RootAttr::PROPER_NOUN | RootAttr::VOICING),
            ("akana", "akana", PrimaryPos::Noun, SecondaryPos::ProperNoun, RootAttr::PROPER_NOUN),

            // --- Verbs ---
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
            ("anla", "anla", PrimaryPos::Verb, SecondaryPos::None, RootAttr::NONE),
            ("anlat", "anlat", PrimaryPos::Verb, SecondaryPos::None, RootAttr::NONE),
            ("dinle", "dinle", PrimaryPos::Verb, SecondaryPos::None, RootAttr::NONE),
            ("öğren", "öğren", PrimaryPos::Verb, SecondaryPos::None, RootAttr::NONE),
            ("öğret", "öğret", PrimaryPos::Verb, SecondaryPos::None, RootAttr::NONE),
            ("çalış", "çalış", PrimaryPos::Verb, SecondaryPos::None, RootAttr::NONE),
            ("konuş", "konuş", PrimaryPos::Verb, SecondaryPos::None, RootAttr::NONE),
            ("otur", "otur", PrimaryPos::Verb, SecondaryPos::None, RootAttr::NONE),
            ("kalk", "kalk", PrimaryPos::Verb, SecondaryPos::None, RootAttr::NONE),
            ("gül", "gül", PrimaryPos::Verb, SecondaryPos::None, RootAttr::NONE),
            ("ağla", "ağla", PrimaryPos::Verb, SecondaryPos::None, RootAttr::NONE),
            ("uyu", "uyu", PrimaryPos::Verb, SecondaryPos::None, RootAttr::NONE),
            ("uyan", "uyan", PrimaryPos::Verb, SecondaryPos::None, RootAttr::NONE),
            ("düşün", "düşün", PrimaryPos::Verb, SecondaryPos::None, RootAttr::NONE),
            ("hisset", "hisset", PrimaryPos::Verb, SecondaryPos::None, RootAttr::VOICING),
            ("başla", "başla", PrimaryPos::Verb, SecondaryPos::None, RootAttr::NONE),
            ("bitir", "bitir", PrimaryPos::Verb, SecondaryPos::None, RootAttr::NONE),
            ("kazan", "kazan", PrimaryPos::Verb, SecondaryPos::None, RootAttr::NONE),
            ("kaybet", "kaybet", PrimaryPos::Verb, SecondaryPos::None, RootAttr::VOICING),
            ("yardım", "yardım", PrimaryPos::Noun, SecondaryPos::None, RootAttr::NONE),
            ("tanı", "tanı", PrimaryPos::Verb, SecondaryPos::None, RootAttr::NONE),
            ("unut", "unut", PrimaryPos::Verb, SecondaryPos::None, RootAttr::NONE),
            ("hatırla", "hatırla", PrimaryPos::Verb, SecondaryPos::None, RootAttr::NONE),

            // --- Adjectives ---
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
            ("uzun", "uzun", PrimaryPos::Adj, SecondaryPos::None, RootAttr::NONE),
            ("kısa", "kısa", PrimaryPos::Adj, SecondaryPos::None, RootAttr::NONE),
            ("geniş", "geniş", PrimaryPos::Adj, SecondaryPos::None, RootAttr::NONE),
            ("dar", "dar", PrimaryPos::Adj, SecondaryPos::None, RootAttr::NONE),
            ("yüksek", "yüksek", PrimaryPos::Adj, SecondaryPos::None, RootAttr::VOICING),
            ("alçak", "alçak", PrimaryPos::Adj, SecondaryPos::None, RootAttr::VOICING),
            ("ağır", "ağır", PrimaryPos::Adj, SecondaryPos::None, RootAttr::NONE),
            ("hafif", "hafif", PrimaryPos::Adj, SecondaryPos::None, RootAttr::NONE),
            ("sıcak", "sıcak", PrimaryPos::Adj, SecondaryPos::None, RootAttr::VOICING),
            ("soğuk", "soğuk", PrimaryPos::Adj, SecondaryPos::None, RootAttr::VOICING),
            ("zengin", "zengin", PrimaryPos::Adj, SecondaryPos::None, RootAttr::NONE),
            ("fakir", "fakir", PrimaryPos::Adj, SecondaryPos::None, RootAttr::NONE),
            ("mutlu", "mutlu", PrimaryPos::Adj, SecondaryPos::None, RootAttr::NONE),
            ("üzgün", "üzgün", PrimaryPos::Adj, SecondaryPos::None, RootAttr::NONE),
            ("genç", "genç", PrimaryPos::Adj, SecondaryPos::None, RootAttr::VOICING),
            ("yaşlı", "yaşlı", PrimaryPos::Adj, SecondaryPos::None, RootAttr::NONE),
            ("temiz", "temiz", PrimaryPos::Adj, SecondaryPos::None, RootAttr::NONE),
            ("kirli", "kirli", PrimaryPos::Adj, SecondaryPos::None, RootAttr::NONE),
            ("doğru", "doğru", PrimaryPos::Adj, SecondaryPos::None, RootAttr::NONE),
            ("yanlış", "yanlış", PrimaryPos::Adj, SecondaryPos::None, RootAttr::NONE),

            // --- Pronouns ---
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
            ("nere", "nere", PrimaryPos::Pron, SecondaryPos::Question, RootAttr::NONE),
            ("hangi", "hangi", PrimaryPos::Pron, SecondaryPos::Question, RootAttr::NONE),
            ("kaç", "kaç", PrimaryPos::Pron, SecondaryPos::Question, RootAttr::NONE),

            // --- Adverbs ---
            ("çok", "çok", PrimaryPos::Adv, SecondaryPos::Quantitive, RootAttr::NONE),
            ("az", "az", PrimaryPos::Adv, SecondaryPos::Quantitive, RootAttr::NONE),
            ("daha", "daha", PrimaryPos::Adv, SecondaryPos::None, RootAttr::NONE),
            ("en", "en", PrimaryPos::Adv, SecondaryPos::None, RootAttr::NONE),
            ("hiç", "hiç", PrimaryPos::Adv, SecondaryPos::None, RootAttr::NONE),
            ("hep", "hep", PrimaryPos::Adv, SecondaryPos::None, RootAttr::NONE),
            ("şimdi", "şimdi", PrimaryPos::Adv, SecondaryPos::Time, RootAttr::NONE),
            ("sonra", "sonra", PrimaryPos::Adv, SecondaryPos::Time, RootAttr::NONE),
            ("önce", "önce", PrimaryPos::Adv, SecondaryPos::Time, RootAttr::NONE),
            ("erken", "erken", PrimaryPos::Adv, SecondaryPos::Time, RootAttr::NONE),
            ("geç", "geç", PrimaryPos::Adv, SecondaryPos::Time, RootAttr::NONE),
            ("bazen", "bazen", PrimaryPos::Adv, SecondaryPos::Time, RootAttr::NONE),
            ("genellikle", "genellikle", PrimaryPos::Adv, SecondaryPos::None, RootAttr::NONE),
            ("mutlaka", "mutlaka", PrimaryPos::Adv, SecondaryPos::None, RootAttr::NONE),
            ("kesinlikle", "kesinlikle", PrimaryPos::Adv, SecondaryPos::None, RootAttr::NONE),
            ("belki", "belki", PrimaryPos::Adv, SecondaryPos::None, RootAttr::NONE),
            ("birlikte", "birlikte", PrimaryPos::Adv, SecondaryPos::None, RootAttr::NONE),

            // --- Conjunctions & Postpositions ---
            ("ve", "ve", PrimaryPos::Conj, SecondaryPos::None, RootAttr::NONE),
            ("ile", "ile", PrimaryPos::Conj, SecondaryPos::None, RootAttr::NONE),
            ("veya", "veya", PrimaryPos::Conj, SecondaryPos::None, RootAttr::NONE),
            ("ama", "ama", PrimaryPos::Conj, SecondaryPos::None, RootAttr::NONE),
            ("fakat", "fakat", PrimaryPos::Conj, SecondaryPos::None, RootAttr::NONE),
            ("ancak", "ancak", PrimaryPos::Conj, SecondaryPos::None, RootAttr::NONE),
            ("çünkü", "çünkü", PrimaryPos::Conj, SecondaryPos::None, RootAttr::NONE),
            ("oysa", "oysa", PrimaryPos::Conj, SecondaryPos::None, RootAttr::NONE),
            ("madem", "madem", PrimaryPos::Conj, SecondaryPos::None, RootAttr::NONE),
            ("halbuki", "halbuki", PrimaryPos::Conj, SecondaryPos::None, RootAttr::NONE),
            ("için", "için", PrimaryPos::Postp, SecondaryPos::None, RootAttr::NONE),
            ("gibi", "gibi", PrimaryPos::Postp, SecondaryPos::None, RootAttr::NONE),
            ("kadar", "kadar", PrimaryPos::Postp, SecondaryPos::None, RootAttr::NONE),
            ("göre", "göre", PrimaryPos::Postp, SecondaryPos::None, RootAttr::NONE),
            ("karşı", "karşı", PrimaryPos::Postp, SecondaryPos::None, RootAttr::NONE),
            ("rağmen", "rağmen", PrimaryPos::Postp, SecondaryPos::None, RootAttr::NONE),
            ("doğru", "doğru", PrimaryPos::Postp, SecondaryPos::None, RootAttr::NONE),

            // --- Numerals ---
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
            ("yirmi", "yirmi", PrimaryPos::Num, SecondaryPos::Cardinal, RootAttr::NONE),
            ("otuz", "otuz", PrimaryPos::Num, SecondaryPos::Cardinal, RootAttr::NONE),
            ("kırk", "kırk", PrimaryPos::Num, SecondaryPos::Cardinal, RootAttr::NONE),
            ("elli", "elli", PrimaryPos::Num, SecondaryPos::Cardinal, RootAttr::NONE),
            ("altmış", "altmış", PrimaryPos::Num, SecondaryPos::Cardinal, RootAttr::NONE),
            ("yetmiş", "yetmiş", PrimaryPos::Num, SecondaryPos::Cardinal, RootAttr::NONE),
            ("seksen", "seksen", PrimaryPos::Num, SecondaryPos::Cardinal, RootAttr::NONE),
            ("doksan", "doksan", PrimaryPos::Num, SecondaryPos::Cardinal, RootAttr::NONE),
            ("yüz", "yüz", PrimaryPos::Num, SecondaryPos::Cardinal, RootAttr::NONE),
            ("bin", "bin", PrimaryPos::Num, SecondaryPos::Cardinal, RootAttr::NONE),
            ("milyon", "milyon", PrimaryPos::Num, SecondaryPos::Cardinal, RootAttr::NONE),
            ("milyar", "milyar", PrimaryPos::Num, SecondaryPos::Cardinal, RootAttr::NONE),
        ];

        for (lemma, root, primary, sec, attr) in words {
            self.add_item(DictionaryItem::new(lemma, root, primary, sec, attr));
        }

        // Ingest the complete ~93,000 Turkish dictionary entries and roots
        let full_lex = include_str!("zemberek_lexicon.txt");
        self.load_from_str(full_lex);
    }
}
