//! Turkish Number-to-Words, Words-to-Number, Ordinals, and Currency Converter.

use lazy_static::lazy_static;
use std::collections::HashMap;

lazy_static! {
    static ref DIGITS: [&'static str; 10] =
        ["sıfır", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"];
    static ref TENS: [&'static str; 10] =
        ["", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan"];
    static ref SCALES: [&'static str; 7] = [
        "",
        "bin",
        "milyon",
        "milyar",
        "trilyon",
        "katrilyon",
        "kentilyon"
    ];
    static ref WORD_TO_VAL: HashMap<&'static str, i64> = {
        let mut m = HashMap::new();
        m.insert("sıfır", 0);
        m.insert("bir", 1);
        m.insert("iki", 2);
        m.insert("üç", 3);
        m.insert("dört", 4);
        m.insert("beş", 5);
        m.insert("altı", 6);
        m.insert("yedi", 7);
        m.insert("sekiz", 8);
        m.insert("dokuz", 9);

        m.insert("on", 10);
        m.insert("yirmi", 20);
        m.insert("otuz", 30);
        m.insert("kırk", 40);
        m.insert("elli", 50);
        m.insert("altmış", 60);
        m.insert("yetmiş", 70);
        m.insert("seksen", 80);
        m.insert("doksan", 90);

        m.insert("yüz", 100);
        m.insert("bin", 1000);
        m.insert("milyon", 1_000_000);
        m.insert("milyar", 1_000_000_000);
        m.insert("trilyon", 1_000_000_000_000);
        m
    };
}

pub struct TurkishNumberConverter;

impl TurkishNumberConverter {
    /// Converts an integer into Turkish written words.
    /// Example: 1923 -> "bin dokuz yüz yirmi üç", 105 -> "yüz beş", -42 -> "eksi kırk iki".
    pub fn number_to_words(n: i64) -> String {
        if n == 0 {
            return "sıfır".to_string();
        }

        let is_negative = n < 0;
        // As u64: the magnitude of i64::MIN is no i64 (review 3 BIND3-08).
        let mut groups: Vec<u64> = Vec::new();
        let mut temp = n.unsigned_abs();
        while temp > 0 {
            groups.push(temp % 1000);
            temp /= 1000;
        }

        let mut result_parts: Vec<String> = Vec::new();

        for (i, &group) in groups.iter().enumerate().rev() {
            if group == 0 {
                continue;
            }

            let group_words = Self::convert_triplet(group);

            // Special Turkish rule: for 1000-1999, we say "bin", NOT "bir bin"
            if i == 1 && group == 1 {
                result_parts.push("bin".to_string());
            } else {
                result_parts.push(group_words);
                if i > 0 && i < SCALES.len() {
                    result_parts.push(SCALES[i].to_string());
                }
            }
        }

        let text = result_parts.join(" ");
        if is_negative {
            format!("eksi {text}")
        } else {
            text
        }
    }

    fn convert_triplet(n: u64) -> String {
        let mut parts = Vec::new();
        let hundreds = n / 100;
        let remainder = n % 100;
        let tens = remainder / 10;
        let units = remainder % 10;

        if hundreds > 0 {
            if hundreds > 1 {
                parts.push(DIGITS[hundreds as usize].to_string());
            }
            parts.push("yüz".to_string());
        }

        if tens > 0 {
            parts.push(TENS[tens as usize].to_string());
        }

        if units > 0 {
            parts.push(DIGITS[units as usize].to_string());
        }

        parts.join(" ")
    }

    /// Converts a number to its Turkish ordinal form (e.g. 1 -> "birinci", 2 -> "ikinci", 3 -> "üçüncü", 4 -> "dördüncü").
    pub fn ordinal_to_words(n: i64) -> String {
        let words = Self::number_to_words(n);
        let last_word = words.split_whitespace().last().unwrap_or("sıfır");

        let ordinal_suffix = match last_word {
            "bir" => "inci",
            "iki" => "nci",
            "üç" => "üncü",
            "dört" => "üncü", // dördüncü
            "beş" => "inci",
            "altı" => "ncı",
            "yedi" => "nci",
            "sekiz" => "inci",
            "dokuz" => "uncu",
            "on" => "uncu",
            "yirmi" => "nci",
            "otuz" => "uncu",
            "kırk" => "ıncı",
            "elli" => "nci",
            "altmış" => "ıncı",
            "yetmiş" => "inci",
            "seksen" => "inci",
            "doksan" => "ıncı",
            "yüz" => "üncü",
            "bin" => "inci",
            "milyon" => "uncu",
            "milyar" => "ıncı",
            "trilyon" => "uncu",
            "sıfır" => "ıncı",
            _ => "inci",
        };

        // Handle consonant mutation for "dört" -> "dörd"
        let base = if last_word == "dört" {
            let prefix = if words.contains(' ') {
                let idx = words.rfind(' ').unwrap();
                format!("{} dörd", &words[..idx])
            } else {
                "dörd".to_string()
            };
            prefix
        } else {
            words
        };

        format!("{base}{ordinal_suffix}")
    }

    /// Converts currency amounts into Turkish words.
    /// Example: (1250.50, "TL") -> "bin iki yüz elli lira elli kuruş". An amount
    /// [`try_currency_to_words`](Self::try_currency_to_words) takes no words of is "".
    pub fn currency_to_words(amount: f64, currency: &str) -> String {
        Self::try_currency_to_words(amount, currency).unwrap_or_default()
    }

    /// [`currency_to_words`](Self::currency_to_words), or `None` for an amount that is not
    /// finite or has more kuruş than an `i64` holds. The amount is rounded to whole kuruş
    /// once, so 2.999 is "üç lira" and -1.5 "eksi bir lira elli kuruş" (review 3 BIND3-08).
    pub fn try_currency_to_words(amount: f64, currency: &str) -> Option<String> {
        let kurus = (amount.abs() * 100.0).round();
        if !kurus.is_finite() || kurus >= i64::MAX as f64 {
            return None;
        }
        // Below 2^63, so the cast is exact up to f64's own precision.
        let kurus = kurus as u64;
        let (main_unit, cents) = (kurus / 100, kurus % 100);

        let (curr_name, sub_name) = match currency.to_uppercase().as_str() {
            "TL" | "TRY" => ("lira", "kuruş"),
            "USD" | "$" => ("dolar", "sent"),
            "EUR" | "€" => ("avro", "sent"),
            "GBP" | "£" => ("sterlin", "peni"),
            _ => ("lira", "kuruş"),
        };

        let mut parts = Vec::new();
        if amount < 0.0 && kurus > 0 {
            parts.push("eksi".to_string());
        }
        // Both below 2^63 / 100.
        parts.push(Self::number_to_words(main_unit as i64));
        parts.push(curr_name.to_string());

        if cents > 0 {
            parts.push(Self::number_to_words(cents as i64));
            parts.push(sub_name.to_string());
        }

        Some(parts.join(" "))
    }

    /// Parses written Turkish number words back into an integer.
    /// Example: "bin dokuz yüz yirmi üç" -> Ok(1923), "beş yüz" -> Ok(500).
    pub fn words_to_number(text: &str) -> Result<i64, String> {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return Err("Empty input".to_string());
        }

        let mut total: i64 = 0;
        let mut current_triplet: i64 = 0;

        for &w in &words {
            let lower = super::super::phonology::to_turkish_lower(w);
            if let Some(&val) = WORD_TO_VAL.get(lower.as_str()) {
                if val == 1000 {
                    if current_triplet == 0 {
                        current_triplet = 1;
                    }
                    total += current_triplet * 1000;
                    current_triplet = 0;
                } else if val >= 1_000_000 {
                    if current_triplet == 0 {
                        current_triplet = 1;
                    }
                    total += current_triplet * val;
                    current_triplet = 0;
                } else if val == 100 {
                    if current_triplet == 0 {
                        current_triplet = 1;
                    }
                    current_triplet *= 100;
                } else {
                    current_triplet += val;
                }
            } else {
                return Err(format!("Unrecognized number word: {w}"));
            }
        }

        total += current_triplet;
        Ok(total)
    }
}
