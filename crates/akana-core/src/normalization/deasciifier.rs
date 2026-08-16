//! Restores Turkish diacritics to ASCII Turkish text (De-asciifier).

use std::collections::HashMap;
use lazy_static::lazy_static;

lazy_static! {
    /// Common Turkish word mappings for accurate de-asciification
    static ref DEASCII_DICT: HashMap<&'static str, &'static str> = {
        let mut m = HashMap::new();
        let entries = [
            ("turkce", "türkçe"), ("turk", "türk"), ("turkiye", "türkiye"),
            ("cok", "çok"), ("az", "az"), ("daha", "daha"),
            ("agac", "ağaç"), ("agaci", "ağacı"), ("agaclar", "ağaçlar"),
            ("cocuk", "çocuk"), ("cocuklar", "çocuklar"),
            ("guzel", "güzel"), ("guzellik", "güzellik"),
            ("ogrenci", "öğrenci"), ("ogretmen", "öğretmen"), ("ogretmenler", "öğretmenler"),
            ("kitap", "kitap"), ("kitaplar", "kitaplar"), ("kitabi", "kitabı"),
            ("okul", "okul"), ("okullar", "okullar"),
            ("hizli", "hızlı"), ("hizlica", "hızlıca"),
            ("caliskan", "çalışkan"), ("calismak", "çalışmak"), ("calisiyor", "çalışıyor"),
            ("yazi", "yazı"), ("yazilim", "yazılım"),
            ("degil", "değil"), ("dunya", "dünya"), ("gun", "gün"), ("gunler", "günler"),
            ("gunes", "güneş"), ("ay", "ay"), ("yildiz", "yıldız"),
            ("goz", "göz"), ("gozluk", "gözlük"), ("kulak", "kulak"),
            ("yurek", "yürek"), ("gonul", "gönül"), ("sevgi", "sevgi"),
            ("arkadas", "arkadaş"), ("kardes", "kardeş"),
            ("birlik", "birlik"), ("beraberlik", "beraberlik"),
            ("buyuk", "büyük"), ("kucuk", "küçük"),
            ("saglik", "sağlık"), ("yasam", "yaşam"), ("hayat", "hayat"),
            ("cagdas", "çağdaş"), ("bilim", "bilim"), ("sanat", "sanat"),
            ("edebiyat", "edebiyat"), ("dil", "dil"), ("tarih", "tarih"),
            ("cogur", "çoğur"), ("cogul", "çoğul"), ("tekil", "tekil"),
            ("kosul", "koşul"), ("durum", "durum"), ("olay", "olay"),
            ("yol", "yol"), ("yolcu", "yolcu"), ("yolculuk", "yolculuk"),
            ("deniz", "deniz"), ("gok", "gök"), ("yer", "yer"),
            ("sehir", "şehir"), ("ulke", "ülke"), ("devlet", "devlet"),
            ("insan", "insan"), ("insanlar", "insanlar"),
            ("yapilmis", "yapılmış"), ("olmus", "olmuş"), ("gelmis", "gelmiş"),
            ("gitti", "gitti"), ("geldi", "geldi"), ("gordu", "gördü"),
            ("bakar", "bakar"), ("yapar", "yapar"), ("eder", "eder"),
            ("icin", "için"), ("cunku", "çünkü"), ("eger", "eğer"),
            ("belki", "belki"), ("simdi", "şimdi"), ("sonra", "sonra"), ("once", "önce")
        ];
        for (k, v) in entries {
            m.insert(k, v);
        }
        m
    };
}

pub struct TurkishDeasciifier;

impl TurkishDeasciifier {
    /// Deasciifies a word or text, converting ASCII representations back to Turkish letters.
    pub fn deasciify(text: &str) -> String {
        let mut result = String::with_capacity(text.len());
        let tokens: Vec<&str> = text.split_inclusive(|c: char| !c.is_alphabetic()).collect();

        for token in tokens {
            let alphabetic_part: String = token.chars().filter(|c| c.is_alphabetic()).collect();
            let non_alphabetic_part: String = token.chars().filter(|c| !c.is_alphabetic()).collect();

            if alphabetic_part.is_empty() {
                result.push_str(token);
                continue;
            }

            let lower = alphabetic_part.to_lowercase();

            if let Some(&correct) = DEASCII_DICT.get(lower.as_str()) {
                let is_upper = alphabetic_part.chars().next().map_or(false, |c| c.is_uppercase());
                let is_all_upper = alphabetic_part.len() > 1 && alphabetic_part.chars().all(|c| c.is_uppercase());

                let restored = if is_all_upper {
                    super::super::phonology::to_turkish_upper(correct)
                } else if is_upper {
                    super::super::phonology::to_turkish_title(correct)
                } else {
                    correct.to_string()
                };

                result.push_str(&restored);
                result.push_str(&non_alphabetic_part);
            } else {
                result.push_str(token);
            }
        }

        result
    }
}
