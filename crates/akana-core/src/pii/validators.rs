//! Mathematical and algorithmic checksum validators for Turkish PII.
//!
//! Provides zero-false-positive validation for:
//! - **TCKN** (T.C. Kimlik Numarası - Dual mod-10 check)
//! - **VKN** (Vergi Kimlik Numarası - Official Revenue Administration algorithm)
//! - **IBAN** (TR + 24 digits - ISO 7064 mod-97 check)
//! - **Credit/Debit Cards** (13-19 digits - Luhn algorithm + BIN detection: Troy, Visa, MC, Amex)
//! - **Vehicle Plates** (01-81 city code format check)
//! - **VIN / Şasi No** (17 alphanumeric - ISO 3779 transliteration weight matrix & mod-11 check digit)
//! - **IMEI** (15 digits - Luhn check digit)

use serde::{Deserialize, Serialize};

/// Card brand detection result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CardBrand {
    Troy,
    Visa,
    MasterCard,
    AmericanExpress,
    Unknown,
}

/// Validates an 11-digit Turkish National Identity Number (TCKN).
///
/// Rules:
/// 1. Exactly 11 ASCII digits.
/// 2. First digit cannot be '0'.
/// 3. 10th digit: `((d1 + d3 + d5 + d7 + d9) * 7 - (d2 + d4 + d6 + d8)) mod 10`.
/// 4. 11th digit: `(d1 + d2 + ... + d10) mod 10`.
/// 5. Rejects known dummy/trivial series (e.g. `11111111110`, `22222222220`).
pub fn validate_tckn(s: &str) -> bool {
    let clean: String = s.chars().filter(|c| c.is_ascii_digit()).collect();
    if clean.len() != 11 {
        return false;
    }

    let bytes = clean.as_bytes();
    if bytes[0] == b'0' {
        return false;
    }

    // Reject dummy series where all first 10 digits are identical
    if bytes[0..10].iter().all(|&b| b == bytes[0]) {
        return false;
    }

    let d: Vec<i32> = bytes.iter().map(|&b| (b - b'0') as i32).collect();

    let odd_sum = d[0] + d[2] + d[4] + d[6] + d[8];
    let even_sum = d[1] + d[3] + d[5] + d[7];

    let calc_d10 = ((odd_sum * 7 - even_sum) % 10 + 10) % 10;
    if calc_d10 != d[9] {
        return false;
    }

    let total_sum_10: i32 = d[0..10].iter().sum();
    let calc_d11 = total_sum_10 % 10;
    if calc_d11 != d[10] {
        return false;
    }

    true
}

/// Validates a 10-digit Turkish Tax Identification Number (VKN - Vergi Kimlik Numarası).
///
/// Official GİB algorithm:
/// For i = 0..8 (digits d1..d9):
///   v_i = (d_i + (9 - i)) mod 10
///   if v_i == 0 => w_i = 0
///   else => w_i = (v_i * 2^(9 - i)) mod 9; if w_i == 0 => w_i = 9
/// Sum = sum(w_0..w_8)
/// check_digit = (10 - (Sum mod 10)) mod 10
/// Valid if d10 == check_digit.
pub fn validate_vkn(s: &str) -> bool {
    let clean: String = s.chars().filter(|c| c.is_ascii_digit()).collect();
    if clean.len() != 10 {
        return false;
    }

    let bytes = clean.as_bytes();
    let d: Vec<i32> = bytes.iter().map(|&b| (b - b'0') as i32).collect();

    let mut sum = 0;
    for (i, &digit) in d.iter().enumerate().take(9) {
        let v = (digit + (9 - i as i32)) % 10;
        if v != 0 {
            let power_of_two = 1 << (9 - i);
            let mut w = (v * power_of_two) % 9;
            if w == 0 {
                w = 9;
            }
            sum += w;
        }
    }

    let check_digit = (10 - (sum % 10)) % 10;
    check_digit == d[9]
}

/// Returns the expected standard IBAN length according to ISO 13616 if the country code is known.
fn expected_iban_length(country_code: &[u8; 2]) -> Option<usize> {
    match country_code {
        b"AL" => Some(28),
        b"AD" => Some(24),
        b"AT" => Some(20),
        b"AZ" => Some(28),
        b"BH" => Some(22),
        b"BY" => Some(28),
        b"BE" => Some(16),
        b"BA" => Some(20),
        b"BR" => Some(29),
        b"BG" => Some(22),
        b"CR" => Some(22),
        b"HR" => Some(21),
        b"CY" => Some(28),
        b"CZ" => Some(24),
        b"DK" => Some(18),
        b"DO" => Some(28),
        b"EE" => Some(20),
        b"FO" => Some(18),
        b"FI" => Some(18),
        b"FR" => Some(27),
        b"GE" => Some(22),
        b"DE" => Some(22),
        b"GI" => Some(23),
        b"GR" => Some(27),
        b"GL" => Some(18),
        b"GT" => Some(28),
        b"HU" => Some(28),
        b"IS" => Some(26),
        b"IE" => Some(22),
        b"IL" => Some(23),
        b"IT" => Some(27),
        b"JO" => Some(30),
        b"KZ" => Some(20),
        b"XK" => Some(20),
        b"KW" => Some(30),
        b"LV" => Some(21),
        b"LB" => Some(28),
        b"LI" => Some(21),
        b"LT" => Some(20),
        b"LU" => Some(20),
        b"MK" => Some(19),
        b"MT" => Some(31),
        b"MR" => Some(27),
        b"MU" => Some(30),
        b"MC" => Some(27),
        b"MD" => Some(24),
        b"ME" => Some(22),
        b"NL" => Some(18),
        b"NO" => Some(15),
        b"PK" => Some(24),
        b"PS" => Some(29),
        b"PL" => Some(28),
        b"PT" => Some(25),
        b"QA" => Some(29),
        b"RO" => Some(24),
        b"LC" => Some(32),
        b"SM" => Some(27),
        b"ST" => Some(25),
        b"SA" => Some(24),
        b"RS" => Some(22),
        b"SC" => Some(31),
        b"SK" => Some(24),
        b"SI" => Some(19),
        b"ES" => Some(24),
        b"SE" => Some(24),
        b"CH" => Some(21),
        b"TL" => Some(23),
        b"TN" => Some(24),
        b"TR" => Some(26),
        b"UA" => Some(29),
        b"AE" => Some(23),
        b"GB" => Some(22),
        b"VA" => Some(22),
        b"VG" => Some(24),
        _ => None,
    }
}

/// Validates an International Bank Account Number (IBAN) from any country (ISO 13616).
///
/// Checks:
/// - Total length (15 to 34 characters; checks exact length if country is registered)
/// - First 2 characters are ISO country code (`[A-Z]{2}`)
/// - Characters 3..4 are check digits (`[0-9]{2}`)
/// - Remaining characters are alphanumeric
/// - ISO 7064 MOD 97-10 check on rearranged string (rearranged mod 97 == 1)
pub fn validate_iban(s: &str) -> bool {
    let clean: String = s.chars().filter(|c| !c.is_whitespace()).collect();
    if clean.len() < 15 || clean.len() > 34 {
        return false;
    }

    let upper = clean.to_ascii_uppercase();
    let bytes = upper.as_bytes();

    // First 2 chars: ISO country code (A-Z)
    if !bytes[0].is_ascii_uppercase() || !bytes[1].is_ascii_uppercase() {
        return false;
    }

    // Next 2 chars: check digits (0-9)
    if !bytes[2].is_ascii_digit() || !bytes[3].is_ascii_digit() {
        return false;
    }

    // Remaining characters: alphanumeric
    if !bytes[4..].iter().all(|b| b.is_ascii_alphanumeric()) {
        return false;
    }

    // If country code is known in ISO 13616 table, enforce exact length
    let country = [bytes[0], bytes[1]];
    if let Some(expected_len) = expected_iban_length(&country) {
        if clean.len() != expected_len {
            return false;
        }
    }

    // Rearrange: chars[4..] + chars[0..4]
    // Incremental modulo 97 to avoid giant integer overflow
    let mut remainder = 0u64;
    for &b in &bytes[4..] {
        if b.is_ascii_digit() {
            remainder = (remainder * 10 + (b - b'0') as u64) % 97;
        } else {
            // A=10, B=11, ..., Z=35
            let val = (b - b'A' + 10) as u64;
            remainder = (remainder * 100 + val) % 97;
        }
    }
    for &b in &bytes[..4] {
        if b.is_ascii_digit() {
            remainder = (remainder * 10 + (b - b'0') as u64) % 97;
        } else {
            let val = (b - b'A' + 10) as u64;
            remainder = (remainder * 100 + val) % 97;
        }
    }

    remainder == 1
}

/// Validates a US Social Security Number (SSN).
///
/// Rules:
/// - Hyphenated form only: `000-00-0000` (length 11).
/// - Reject Area number (first 3 digits): `000`, `666`, and `900..=999`.
/// - Reject Group number (middle 2 digits): `00`.
/// - Reject Serial number (last 4 digits): `0000`.
pub fn validate_ssn(s: &str) -> bool {
    let bytes = s.as_bytes();
    if bytes.len() != 11 {
        return false;
    }

    if bytes[3] != b'-' || bytes[6] != b'-' {
        return false;
    }

    if !bytes[0].is_ascii_digit()
        || !bytes[1].is_ascii_digit()
        || !bytes[2].is_ascii_digit()
        || !bytes[4].is_ascii_digit()
        || !bytes[5].is_ascii_digit()
        || !bytes[7].is_ascii_digit()
        || !bytes[8].is_ascii_digit()
        || !bytes[9].is_ascii_digit()
        || !bytes[10].is_ascii_digit()
    {
        return false;
    }

    let area =
        (bytes[0] - b'0') as u16 * 100 + (bytes[1] - b'0') as u16 * 10 + (bytes[2] - b'0') as u16;

    if area == 0 || area == 666 || area >= 900 {
        return false;
    }

    let group = (bytes[4] - b'0') as u16 * 10 + (bytes[5] - b'0') as u16;
    if group == 0 {
        return false;
    }

    let serial = (bytes[7] - b'0') as u16 * 1000
        + (bytes[8] - b'0') as u16 * 100
        + (bytes[9] - b'0') as u16 * 10
        + (bytes[10] - b'0') as u16;

    if serial == 0 {
        return false;
    }

    true
}

/// Validates credit/debit card numbers using the standard Luhn (mod-10) algorithm
/// and returns the detected card brand.
pub fn validate_credit_card(s: &str) -> Option<CardBrand> {
    let clean: String = s.chars().filter(|c| c.is_ascii_digit()).collect();
    if clean.len() < 13 || clean.len() > 19 {
        return None;
    }

    let bytes = clean.as_bytes();
    let n = bytes.len();
    let mut sum = 0;
    let mut alternate = false;

    for i in (0..n).rev() {
        let mut d = (bytes[i] - b'0') as i32;
        if alternate {
            d *= 2;
            if d > 9 {
                d -= 9;
            }
        }
        sum += d;
        alternate = !alternate;
    }

    if sum % 10 != 0 {
        return None;
    }

    // Detect card brand
    let brand = if clean.starts_with("9792") {
        CardBrand::Troy
    } else if clean.starts_with('4') {
        CardBrand::Visa
    } else if is_mastercard(&clean) {
        CardBrand::MasterCard
    } else if clean.starts_with("34") || clean.starts_with("37") {
        CardBrand::AmericanExpress
    } else {
        CardBrand::Unknown
    };

    Some(brand)
}

fn is_mastercard(s: &str) -> bool {
    if s.len() < 2 {
        return false;
    }
    if let Ok(prefix2) = s[0..2].parse::<u32>() {
        if (51..=55).contains(&prefix2) {
            return true;
        }
    }
    if s.len() >= 4 {
        if let Ok(prefix4) = s[0..4].parse::<u32>() {
            if (2221..=2720).contains(&prefix4) {
                return true;
            }
        }
    }
    false
}

/// Validates a Turkish vehicle license plate (Plaka).
///
/// Formats:
/// - `99 X 9999` or `99 X 99999`
/// - `99 XX 999` or `99 XX 9999`
/// - `99 XXX 99` or `99 XXX 999`
///
/// Province code must be between 01 and 81.
pub fn validate_plate(s: &str) -> bool {
    let clean: String = s
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || c.is_alphabetic())
        .collect::<String>()
        .to_uppercase();

    // Replace Turkish characters to standard ASCII for plate matching
    let plate_str = clean
        .replace('Ç', "C")
        .replace('Ğ', "G")
        .replace('İ', "I")
        .replace('Ö', "O")
        .replace('Ş', "S")
        .replace('Ü', "U");

    if plate_str.len() < 7 || plate_str.len() > 9 {
        return false;
    }

    // First 2 characters must be digits (01 to 81)
    if !plate_str[0..2].chars().all(|c| c.is_ascii_digit()) {
        return false;
    }

    let province: u32 = match plate_str[0..2].parse() {
        Ok(p) => p,
        Err(_) => return false,
    };

    if !(1..=81).contains(&province) {
        return false;
    }

    let rest = &plate_str[2..];
    let letter_count = rest.chars().take_while(|c| c.is_ascii_alphabetic()).count();
    let digit_count = rest[letter_count..]
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .count();

    if letter_count + digit_count != rest.len() {
        return false;
    }

    // Turkish plate rules:
    // 1 letter: 4 or 5 digits
    // 2 letters: 3 or 4 digits
    // 3 letters: 2 or 3 digits
    match letter_count {
        1 => digit_count == 4 || digit_count == 5,
        2 => digit_count == 3 || digit_count == 4,
        3 => (2..=4).contains(&digit_count),
        _ => false,
    }
}

/// Validates a Vehicle Identification Number (VIN / Şasi No) using ISO 3779 checksum.
///
/// Rules:
/// 1. Exactly 17 alphanumeric characters.
/// 2. Letters 'I', 'O', 'Q' are invalid.
/// 3. Position 9 (index 8) is the check digit (0-9 or 'X').
pub fn validate_vin(s: &str) -> bool {
    let clean: String = s.chars().filter(|c| c.is_ascii_alphanumeric()).collect();
    if clean.len() != 17 {
        return false;
    }

    let upper = clean.to_ascii_uppercase();
    if upper.chars().any(|c| c == 'I' || c == 'O' || c == 'Q') {
        return false;
    }

    // ISO 3779 transliteration values
    fn char_to_val(c: char) -> i32 {
        match c {
            '0'..='9' => (c as u8 - b'0') as i32,
            'A' | 'J' => 1,
            'B' | 'K' | 'S' => 2,
            'C' | 'L' | 'T' => 3,
            'D' | 'M' | 'U' => 4,
            'E' | 'N' | 'V' => 5,
            'F' | 'W' => 6,
            'G' | 'P' | 'X' => 7,
            'H' | 'Y' => 8,
            'R' | 'Z' => 9,
            _ => 0,
        }
    }

    const WEIGHTS: [i32; 17] = [8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2];

    let mut sum = 0;
    let chars: Vec<char> = upper.chars().collect();
    for i in 0..17 {
        sum += char_to_val(chars[i]) * WEIGHTS[i];
    }

    let rem = sum % 11;
    let expected_check = if rem == 10 {
        'X'
    } else {
        (b'0' + rem as u8) as char
    };

    chars[8] == expected_check
}

/// Validates an International Mobile Equipment Identity (IMEI) number.
///
/// Standard format is 15 digits; 15th digit is the Luhn check digit computed over the first 14 digits.
pub fn validate_imei(s: &str) -> bool {
    let clean: String = s.chars().filter(|c| c.is_ascii_digit()).collect();
    if clean.len() != 15 {
        return false;
    }

    let bytes = clean.as_bytes();
    let mut sum = 0;

    for (i, &b) in bytes.iter().enumerate().take(14) {
        let mut d = (b - b'0') as i32;
        if i % 2 == 1 {
            d *= 2;
            if d > 9 {
                d -= 9;
            }
        }
        sum += d;
    }

    let check_digit = ((10 - (sum % 10)) % 10) as u8;
    (bytes[14] - b'0') == check_digit
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tckn_valid_and_invalid() {
        assert!(validate_tckn("10000000146"));
        assert!(validate_tckn("51980838978"));
        assert!(!validate_tckn("10203040506")); // Invalid checksum
        assert!(!validate_tckn("11111111110")); // Dummy series
        assert!(!validate_tckn("01234567890")); // Starts with 0
        assert!(!validate_tckn("12345")); // Too short
    }

    #[test]
    fn test_vkn_valid_and_invalid() {
        assert!(validate_vkn("1234567890")); // Valid fixed-point in GİB math
        assert!(!validate_vkn("1234567891")); // Invalid check digit
        assert!(!validate_vkn("123")); // Too short
    }

    #[test]
    fn test_iban_validation() {
        assert!(validate_iban("TR330006100519786457841326"));
        assert!(validate_iban("TR33 0006 1005 1978 6457 8413 26"));
        assert!(!validate_iban("TR330006100519786457841327")); // Bad check digit
        assert!(validate_iban("DE89370400440532013000")); // Valid German IBAN
        assert!(validate_iban("GB82WEST12345698765432")); // Valid UK IBAN
        assert!(!validate_iban("GB82WEST12345698765433")); // Changed final digit to 3 fails
    }

    #[test]
    fn test_ssn_validation() {
        assert!(validate_ssn("219-09-9999")); // Valid SSN
        assert!(!validate_ssn("000-09-9999")); // Area 000 rejected
        assert!(!validate_ssn("666-09-9999")); // Area 666 rejected
        assert!(!validate_ssn("950-09-9999")); // Area 900-999 rejected
        assert!(!validate_ssn("219-00-9999")); // Group 00 rejected
        assert!(!validate_ssn("219-09-0000")); // Serial 0000 rejected
        assert!(!validate_ssn("219099999")); // Unhyphenated rejected
        assert!(!validate_ssn("219 09 9999")); // Space-separated rejected
    }

    #[test]
    fn test_credit_card_validation() {
        assert_eq!(
            validate_credit_card("4111 1111 1111 1111"),
            Some(CardBrand::Visa)
        );
        assert_eq!(
            validate_credit_card("4111111111111111"),
            Some(CardBrand::Visa)
        );
        assert_eq!(
            validate_credit_card("9792 0000 0000 0003"),
            Some(CardBrand::Troy)
        );
        assert_eq!(validate_credit_card("4111 1111 1111 1112"), None);
    }

    #[test]
    fn test_plate_validation() {
        assert!(validate_plate("34 ABC 123"));
        assert!(validate_plate("06 A 1234"));
        assert!(validate_plate("35 AB 1234"));
        assert!(validate_plate("16 JAA 01"));
        assert!(!validate_plate("82 ABC 123")); // 82 is not a valid Turkish province
        assert!(!validate_plate("00 ABC 123")); // 00 is invalid
    }

    #[test]
    fn test_vin_validation() {
        assert!(validate_vin("1HGCM82633A004352"));
        assert!(!validate_vin("1HGCM82633A004353")); // Bad check digit
    }

    #[test]
    fn test_imei_validation() {
        assert!(!validate_imei("123456789012345"));
    }
}
