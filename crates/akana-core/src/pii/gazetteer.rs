//! Lexical gazetteer and dictionaries for Turkish PII detection.
//!
//! Contains:
//! - Curated Turkish First Names (NVİ census & historical registry)
//! - Common Turkish Surnames
//! - Ambiguous / Polysemous Names (requiring contextual proof)
//! - Honorific Titles and Official Roles
//! - Kinship & Persona Markers
//! - Turkish Provinces (81 iller) and major districts
//! - KVKK Article 6 Special Category Dictionaries (Blood types, Health, Religion, etc.)
//!
//! The name and place lists live in [`crate::lexicon`], which is built without the `pii`
//! feature (speech and the parser read them); they are re-exported here unchanged.

use crate::lexicon::insert_with_ascii;
use lazy_static::lazy_static;
use std::collections::HashSet;

pub use crate::lexicon::{DISTRICTS, GIVEN_NAMES, POLYSEMOUS_NAMES, PROVINCES, SURNAMES};

lazy_static! {
    /// Comprehensive list of Turkish honorific titles, professional designations, and institutional role triggers.
    pub static ref TITLE_TRIGGERS: HashSet<String> = {
        let mut set = HashSet::new();
        insert_with_ascii(&mut set, &[
            "sayın", "prof.", "prof", "dr.", "dr", "doç.", "doç", "yrd.", "yrd",
            "av.", "av", "müh.", "müh", "bakan", "bakanı", "başkan", "başkanı",
            "vali", "valisi", "kaymakam", "kaymakamı", "rektör", "rektörü",
            "dekan", "dekanı", "müdür", "müdürü", "öğretmen", "öğretmeni",
            "cumhurbaşkanı", "başbakan", "general", "albay", "kaptan", "binbaşı",
            "teğmen", "astsubay", "çavuş", "bey", "hanım", "efendi", "paşa",
            "hoca", "hocam", "uzm.", "uzm", "uzman", "uzmanı", "milletvekili",
            "hakim", "savcı", "komiser", "başkomiser", "mimar", "eczacı",
            "doktor", "tabip", "hemşire", "ebe", "psikolog", "sosyolog",
            "prof. dr.", "prof dr", "doç. dr.", "doç dr", "uzm. dr.", "uzm dr",
            "avukat", "stajyer", "müfettiş", "uzman yardımcısı", "başkan yardımcısı",
            // Enterprise, banking, and legal role indicators
            "müşteri", "müşterisi", "müşterimiz", "kullanıcı", "kullanıcısı", "kullanıcımız",
            "hasta", "hastası", "hastamız", "çalışan", "çalışanı", "çalışanımız",
            "personel", "personeli", "personelimiz", "yetkili", "yetkilisi", "yetkilimiz",
            "müvekkil", "müvekkili", "müvekkilimiz", "davacı", "davalı", "sigortalı",
            "abone", "abonesi", "abonemiz", "üye", "üyesi", "üyemiz", "yolcu", "yolcusu",
            "öğrenci", "öğrencisi", "veli", "velisi", "borçlu", "alacaklı"
        ]);
        set
    };

    /// Kinship terms often preceding or following family personal data.
    pub static ref KINSHIP_TERMS: HashSet<String> = {
        let mut set = HashSet::new();
        insert_with_ascii(&mut set, &[
            "anne", "annesi", "baba", "babası", "eş", "eşi", "kız", "kızı",
            "oğul", "oğlu", "kardeş", "kardeşi", "abi", "abisi", "abla", "ablası",
            "teyze", "teyzesi", "amca", "amcası", "dayı", "dayısı", "hala", "halası",
            "yeğen", "yeğeni", "kuzen", "kuzeni", "dede", "dedesi", "nine", "ninesi"
        ]);
        set
    };

    /// Street and address level designator triggers.
    pub static ref ADDRESS_TRIGGERS: HashSet<String> = {
        let mut set = HashSet::new();
        insert_with_ascii(&mut set, &[
            "mahallesi", "mah.", "mah", "caddesi", "cad.", "cad",
            "sokağı", "sok.", "sok", "bulvarı", "bulv.", "bulv",
            "meydanı", "sitesi", "apartmanı", "apt.", "apt",
            "bloğu", "blok", "kat:", "kat", "daire:", "daire", "d:", "no:", "no"
        ]);
        set
    };

    /// Blood type patterns (KVKK Md. 6).
    pub static ref BLOOD_TYPES: HashSet<String> = {
        let mut set = HashSet::new();
        insert_with_ascii(&mut set, &[
            "a rh+", "a rh-", "a rh (+)", "a rh (-)", "a rh pozitif", "a rh negatif",
            "b rh+", "b rh-", "b rh (+)", "b rh (-)", "b rh pozitif", "b rh negatif",
            "ab rh+", "ab rh-", "ab rh (+)", "ab rh (-)", "ab rh pozitif", "ab rh negatif",
            "0 rh+", "0 rh-", "0 rh (+)", "0 rh (-)", "0 rh pozitif", "0 rh negatif",
            "sıfır rh+", "sıfır rh-", "sıfır rh pozitif", "sıfır rh negatif"
        ]);
        set
    };

    /// Health, diagnosis, symptom, and treatment terms (KVKK Md. 6).
    pub static ref HEALTH_TERMS: HashSet<String> = {
        let mut set = HashSet::new();
        insert_with_ascii(&mut set, &[
            "diyabet", "şeker hastalığı", "hipertansiyon", "tansiyon", "kanser",
            "tümör", "lösemi", "kemoterapi", "radyoterapi", "depresyon", "bipolar",
            "şizofreni", "anksiyete", "panik atak", "astım", "bronşit", "koah",
            "epilepsi", "sara", "alzheimer", "demans", "parkinson", "hepatit",
            "hepatit b", "hepatit c", "aids", "hiv", "koroner", "enfarktüs", "kalp krizi",
            "anjiyo", "bypass", "diyaliz", "böbrek yetmezliği", "otizm", "down sendromu",
            "engelli", "engellilik", "özürlü", "işitme engelli", "görme engelli",
            "ortopedik engelli", "zihinsel engelli", "sağlık kurulu raporu", "ilaç raporu",
            "psikiyatri", "onkoloji", "kardiyoloji", "nöroloji", "ortopedi"
        ]);
        set
    };

    /// Religious and philosophical belief terms (KVKK Md. 6).
    pub static ref RELIGION_TERMS: HashSet<String> = {
        let mut set = HashSet::new();
        insert_with_ascii(&mut set, &[
            "müslüman", "islam", "hristiyan", "hıristiyan", "yahudi", "musevi",
            "alevi", "sünni", "caferi", "ortodoks", "katolik", "protestan",
            "budist", "hindu", "ateist", "deist", "agnostik", "şinto", "zerdüşt"
        ]);
        set
    };

    /// Standard institutional and functional email local-parts that represent public organizational
    /// channels rather than individual personal data (KVKK non-PII / institutional role addresses).
    pub static ref CORPORATE_EMAIL_PREFIXES: HashSet<String> = {
        let mut set = HashSet::new();
        insert_with_ascii(&mut set, &[
            "info", "destek", "iletisim", "yardim", "help", "support",
            "musteri.hizmetleri", "musterihizmetleri", "satis", "sales",
            "contact", "admin", "administrator", "webmaster", "postmaster",
            "ik", "hr", "kariyer", "career", "muhasebe", "finans",
            "hukuk", "legal", "kvkk", "privacy", "guvenlik", "security",
            "basin", "press", "media", "pazarlama", "marketing",
            "bilgi", "operasyon", "lojistik", "siparis", "reklam"
        ]);
        set
    };
}
