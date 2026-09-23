//! Turkish spelled-out numbers PII detector.
//!
//! Handles spoken or textual representation of numbers in Turkish prompts:
//! - "tc dokuz dokuz yedi yedi bir yedi dokuz altı sekiz dokuz iki"
//! - "kart no dört sekiz dört yedi dokuz bir bir altı dört sıfır iki iki altı dört sıfır dokuz"
//! - "IBAN TR sekiz bir bir sıfır bir dört dört dokuz sekiz altı sıfır bir"
//! - "müşteri no sıfır yedi doksan sekiz yirmi dört altmış sekiz"
//! - "kart güvenlik kodum yedi sıfır bir"

use super::{PiiEntity, PiiType};
use lazy_static::lazy_static;
use regex::Regex;
use stringzilla::StringZilla;

lazy_static! {
    /// Matches a consecutive sequence of Turkish number words (3 or more tokens).
    static ref SPELLED_NUMBER_SEQUENCE_REGEX: Regex = Regex::new(
        r"(?ix)
        \b
        (?:s[ıiIİ]f[ıiIİ]r|b[iİıI]r|[iİıI]k[iİıI]|[üuÜU][çcÇC]|d[öoÖO]rt|be[şsŞS]|alt[ıiIİ]|yed[iİıI]|sek[iİıI]z|dokuz|on|y[iİıI]rm[iİıI]|otuz|k[ıiIİ]rk|ell[iİıI]|altm[ıiIİ][şsŞS]|yetm[iİıI][şsŞS]|seksen|doksan|y[üuÜU]z|b[iİıI]n)
        (?:\s+(?:s[ıiIİ]f[ıiIİ]r|b[iİıI]r|[iİıI]k[iİıI]|[üuÜU][çcÇC]|d[öoÖO]rt|be[şsŞS]|alt[ıiIİ]|yed[iİıI]|sek[iİıI]z|dokuz|on|y[iİıI]rm[iİıI]|otuz|k[ıiIİ]rk|ell[iİıI]|altm[ıiIİ][şsŞS]|yetm[iİıI][şsŞS]|seksen|doksan|y[üuÜU]z|b[iİıI]n)){2,25}
        \b"
    ).unwrap();

    /// Context trigger preceding the spelled number sequence.
    static ref SPELLED_TRIGGER_REGEX: Regex = Regex::new(
        r"(?ix)
        \b
        (?:
            (?P<tckn>t\.?c\.?(?:\s*k[iİıI]ml[iİıI]k(?:\s*no(?:su)?)?|\s*no)?|k[iİıI]ml[iİıI]k\s*no)
            |
            (?P<iban>[iİıI]ban\s*no|[iİıI]ban(?:[ıiIİ]m)?|[iİıI]ban)
            |
            (?P<card>kred[iİıI]\s*kart[ıiIİ]|kart\s*no(?:su)?|kart)
            |
            (?P<customer>m[üuÜU][şsŞS]ter[iİıI]\s*no(?:su)?|muster[iİıI]\s*no)
            |
            (?P<account>hesap(?:\s*no(?:su)?)?|hesap|[öoÖO]deme\s*tal[iİıI]mat[ıiIİ](?:n[ıiIİ])?)
            |
            (?P<cvv>kart\s*g[üuÜU]venl[iİıI]k\s*kod(?:um|u)?|g[üuÜU]venl[iİıI]k\s*kodu|cvv2?|cvc2?)
            |
            (?P<pin>p[iİıI]n\s*kodu|p[iİıI]n)
        )
        \b
        [:\s]*
        (?P<tr>tr)?
        \s*
        $"
    ).unwrap();
}

/// Detects spelled-out Turkish numbers preceded by financial/identity context triggers.
pub fn detect_spelled_numbers(text: &str, out: &mut Vec<PiiEntity>) {
    // Fast SIMD pre-filter: A spelled number sequence is ONLY emitted if preceded by a context trigger
    let lower = text.to_lowercase();
    const SPELLED_TRIGGERS: &[&str] = &[
        "tc",
        "kimlik",
        "iban",
        "kart",
        "müşteri",
        "musteri",
        "hesap",
        "ödeme",
        "odeme",
        "cvv",
        "cvc",
        "pin",
        "güvenlik",
        "guvenlik",
    ];
    let has_trigger = SPELLED_TRIGGERS
        .iter()
        .any(|&trig| lower.sz_find(trig).is_some());
    if !has_trigger {
        return;
    }

    for mat in SPELLED_NUMBER_SEQUENCE_REGEX.find_iter(text) {
        let seq_start = mat.start();
        let seq_end = mat.end();
        let matched_text = mat.as_str();

        // Safely check context up to 55 chars before the number sequence
        let lookback_start = match text[..seq_start].char_indices().rev().nth(55) {
            Some((idx, _)) => idx,
            None => 0,
        };
        let prefix = &text[lookback_start..seq_start];

        if let Some(cap) = SPELLED_TRIGGER_REGEX.captures(prefix.trim_end()) {
            let (pii_type, label) = if cap.name("tckn").is_some() {
                (PiiType::Tckn, "TCKN")
            } else if cap.name("iban").is_some() {
                (PiiType::Iban, "IBAN")
            } else if cap.name("card").is_some() {
                (PiiType::CreditCard, "KART")
            } else if cap.name("cvv").is_some() {
                (PiiType::CreditCard, "CVV")
            } else if cap.name("pin").is_some() {
                (PiiType::Credentials, "PIN")
            } else if cap.name("customer").is_some() {
                (PiiType::AccountNo, "MUSTERI_NO")
            } else if cap.name("account").is_some() {
                (PiiType::AccountNo, "HESAP_NO")
            } else {
                continue;
            };

            // If IBAN and preceded by "TR ", include TR in the entity span
            let (final_start, final_text) =
                if cap.name("tr").is_some() || prefix.to_uppercase().ends_with("TR ") {
                    let tr_idx = prefix
                        .sz_rfind("TR ")
                        .or_else(|| prefix.sz_rfind("tr "))
                        .unwrap_or(seq_start - lookback_start);
                    let actual_start = lookback_start + tr_idx;
                    (actual_start, text[actual_start..seq_end].to_string())
                } else {
                    (seq_start, matched_text.to_string())
                };

            out.push(PiiEntity {
                text: final_text.clone(),
                label: label.to_string(),
                pii_type,
                start: final_start,
                end: seq_end,
                confidence: 0.98,
                stem: final_text,
                suffix: None,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spelled_tckn_and_iban() {
        let text = "adım Burak Akgündüz tc dokuz dokuz yedi yedi bir yedi dokuz altı sekiz dokuz iki. onay bekliyor.";
        let mut out = Vec::new();
        detect_spelled_numbers(text, &mut out);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].label, "TCKN");
        assert_eq!(
            out[0].text,
            "dokuz dokuz yedi yedi bir yedi dokuz altı sekiz dokuz iki"
        );

        let text2 = "ödeme için IBAN TR sekiz bir bir sıfır bir dört dört dokuz sekiz altı sıfır bir aktarılsın.";
        let mut out2 = Vec::new();
        detect_spelled_numbers(text2, &mut out2);
        assert_eq!(out2.len(), 1);
        assert_eq!(out2[0].label, "IBAN");
        assert!(out2[0].text.contains("TR"));
    }
}
