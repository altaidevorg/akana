//! Fast conversion of Turkish characters to ASCII equivalents.

pub struct TurkishAsciifier;

impl TurkishAsciifier {
    /// Converts Turkish characters with diacritics to their ASCII counterparts.
    /// e.g. "Türkçe Sözlük" -> "Turkce Sozluk"
    pub fn asciify(s: &str) -> String {
        let mut result = String::with_capacity(s.len());
        for c in s.chars() {
            match c {
                'ç' => result.push('c'),
                'Ç' => result.push('C'),
                'ğ' => result.push('g'),
                'Ğ' => result.push('G'),
                'ı' => result.push('i'),
                'I' => result.push('I'),
                'i' => result.push('i'),
                'İ' => result.push('I'),
                'ö' => result.push('o'),
                'Ö' => result.push('O'),
                'ş' => result.push('s'),
                'Ş' => result.push('S'),
                'ü' => result.push('u'),
                'Ü' => result.push('U'),
                'â' => result.push('a'),
                'Â' => result.push('A'),
                'î' => result.push('i'),
                'Î' => result.push('I'),
                'û' => result.push('u'),
                'Û' => result.push('U'),
                _ => result.push(c),
            }
        }
        result
    }
}
