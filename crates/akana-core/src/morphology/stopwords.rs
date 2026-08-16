//! Curated Turkish Stopword filter and cleaner.

use std::collections::HashSet;
use lazy_static::lazy_static;
use crate::phonology::to_turkish_lower;

lazy_static! {
    static ref DEFAULT_STOPWORDS: HashSet<&'static str> = {
        let mut set = HashSet::new();
        let words = [
            "ve", "veya", "ile", "ama", "ancak", "fakat", "lakin", "çünkü", "oysa", "oysaki",
            "halbuki", "madem", "mademki", "meğer", "nitekim", "zira",
            "bu", "şu", "o", "bunlar", "şunlar", "onlar", "buna", "şuna", "ona",
            "bunu", "şunu", "onu", "bunda", "şunda", "onda", "bundan", "şundan", "ondan",
            "ben", "sen", "biz", "siz", "bana", "sana", "bize", "size", "beni", "seni", "bizi", "sizi",
            "bende", "sende", "bizde", "sizde", "benden", "senden", "bizden", "sizden",
            "benim", "senin", "onun", "bizim", "sizin", "onların",
            "gibi", "kadar", "için", "göre", "doğru", "karşı", "rağmen", "karşın", "dolayı", "ötürü",
            "de", "da", "ki", "mi", "mı", "mu", "mü", "misin", "mısın", "musun", "müsün",
            "ya", "hem", "ne", "gerek", "ister", "ha",
            "daha", "en", "pek", "çok", "az", "fazla", "gayet", "oldukça", "biraz",
            "her", "bazı", "bütün", "tüm", "kimi", "birçok", "çoğu", "hiçbiri", "herkes", "kimse",
            "şey", "şeyler", "biri", "birisi", "başkası", "diğeri", "öteki", "hiç",
            "nasıl", "neden", "niçin", "niye", "kim", "kime", "kimi", "kimde", "kimden",
            "nerede", "nereye", "nereden", "neresi", "neresini", "hangisi", "kaç", "kaçı",
            "var", "yok", "olan", "olarak", "eden", "ederek", "yapan", "yapılan",
            "edilmiş", "edilen", "olduğu", "olduğunu", "oldukları", "olmak", "etmek", "yapmak",
            "ise", "iken", "idi", "imiş", "yine", "tekrar", "artık", "zaten", "henüz", "şimdi"
        ];
        for w in words {
            set.insert(w);
        }
        set
    };
}

pub struct TurkishStopwords {
    stopwords: HashSet<String>,
}

impl Default for TurkishStopwords {
    fn default() -> Self {
        Self::new()
    }
}

impl TurkishStopwords {
    pub fn new() -> Self {
        let mut set = HashSet::new();
        for &w in DEFAULT_STOPWORDS.iter() {
            set.insert(w.to_string());
        }
        Self { stopwords: set }
    }

    pub fn with_custom(words: impl IntoIterator<Item = String>) -> Self {
        let mut set = HashSet::new();
        for w in words {
            set.insert(to_turkish_lower(&w));
        }
        Self { stopwords: set }
    }

    /// Checks if a word is a stopword.
    #[inline]
    pub fn is_stopword(&self, word: &str) -> bool {
        let lower = to_turkish_lower(word);
        self.stopwords.contains(&lower)
    }

    /// Filters out stopwords from a slice of tokens.
    pub fn filter_tokens<'a>(&self, tokens: &[&'a str]) -> Vec<&'a str> {
        tokens.iter()
            .copied()
            .filter(|t| !self.is_stopword(t))
            .collect()
    }
}
