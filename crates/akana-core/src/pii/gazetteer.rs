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

use crate::normalization::TurkishAsciifier;
use lazy_static::lazy_static;
use std::collections::HashSet;

fn insert_with_ascii(set: &mut HashSet<String>, items: &[&str]) {
    for &item in items {
        set.insert(item.to_string());
        let asc = TurkishAsciifier::asciify(item);
        if asc != item {
            set.insert(asc);
        }
    }
}

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

    /// Names that are homonyms / polysemous with common nouns, adjectives, or verbs.
    pub static ref POLYSEMOUS_NAMES: HashSet<String> = {
        let mut set = HashSet::new();
        insert_with_ascii(&mut set, &[
            "deniz", "barış", "gül", "toprak", "kaya", "demir", "bahar", "güneş",
            "can", "çiçek", "umut", "zafer", "dilek", "sevgi", "melek", "kartal",
            "aslan", "bulut", "yağmur", "rüzgar", "derya", "yiğit", "aydın",
            "mutlu", "oğuz", "ilker", "pınar", "başak", "sarp", "savaş", "yavuz",
            "özgür", "mert", "bora", "koray", "onur", "taner", "alper", "bilge",
            "damla", "defne", "engin", "erdem", "fırat", "meriç", "dicle", "tuna",
            "inanç", "özlem", "duygu", "ezgi", "çağrı", "şafak", "ufuk", "sevinç",
            "huzur", "ışık", "doğa", "evren", "volkan", "görkem", "sefa", "cihan",
            "arda", "efe", "eren", "ozan", "doruk", "yaman", "çetin", "coşkun"
        ]);
        set
    };

    /// Curated list of popular and traditional Turkish Given Names.
    pub static ref GIVEN_NAMES: HashSet<String> = {
        let mut set = HashSet::new();
        insert_with_ascii(&mut set, &[
            // Male names
            "mehmet", "mustafa", "ahmet", "ali", "hüseyin", "hasan", "ibrahim", "ismail",
            "osman", "halil", "süleyman", "yusuf", "ömer", "ramazan", "murat", "mahmut",
            "salih", "kemal", "recep", "fatih", "şaban", "abdullah", "emre", "adem",
            "hakan", "adem", "bekir", "cemal", "burak", "serkan", "selim", "cihan",
            "orhan", "sinan", "bülent", "tarık", "sedat", "erhan", "metin", "vedat",
            "levent", "kaan", "kerem", "batuhan", "furkan", "tolga", "berk", "berkay",
            "alperen", "oğuzhan", "gökhan", "ferhat", "serdar", "engin", "kenan", "semih",
            "tayfun", "volkan", "koray", "tuncay", "taner", "ercan", "şerif", "muzaffer",
            "necati", "sabri", "sadık", "rasim", "nihat", "zeki", "hamza", "yakup",
            "ilyas", "yunus", "bilal", "harun", "mikail", "lokman", "davut", "eyüp",
            "enes", "muhammed", "muhammet", "mirac", "berat", "ayaz", "eymen", "alparslan",
            "göktuğ", "metehan", "emirhan", "bedirhan", "doruk", "poyraz", "kuzey", "rüzgar",
            "çınar", "atlas", "toprak", "baran", "mert", "arda", "kerim", "selçuk",
            "caner", "alp", "alper", "cenk", "cem", "çağlar", "çağrı", "serhat",
            "kadir", "turgut", "tevfik", "cüneyt", "zafer", "tarık", "uğur", "veysel",
            "veli", "şinasi", "sinan", "rıza", "suat", "ferdi", "sadettin", "samet",

            // Female names
            "fatma", "ayşe", "emine", "hatice", "zeynep", "elif", "meryem", "özlem",
            "sevim", "filiz", "hülya", "zehra", "hanife", "dilek", "yasemin", "songül",
            "sultan", "rabia", "esra", "ebru", "büşra", "merve", "kübra", "tuğba",
            "gülşen", "ayten", "nurten", "nuray", "gülay", "nuran", "neriman", "sevil",
            "güler", "leyla", "nermin", "nesrin", "figen", "aslı", "şule", "banu",
            "belgin", "berna", "arzu", "didem", "derya", "damla", "defne", "ece",
            "ecem", "gamze", "hande", "ilknur", "ipek", "irem", "jale", "melike",
            "melis", "melisa", "nihal", "nilgün", "nurgül", "pelin", "rüya", "selin",
            "serap", "seval", "sezen", "simge", "sinem", "sude", "şeyma", "tuba",
            "tülay", "vildan", "yağmur", "yıldız", "yonca", "zerrin", "zuhal", "zümrüt",
            "asude", "azra", "bade", "begüm", "beril", "beren", "ceyda", "dilara",
            "eda", "ela", "elçin", "esma", "eylül", "gaye", "gökçe", "gözde",
            "gülce", "hale", "hazan", "hilal", "ılgın", "ırmak", "idil", "ilayda",
            "mina", "miray", "neva", "nisa", "öykü", "sare", "su", "şevval",
            "alara", "derin", "duru", "lina", "masal", "mira", "parla", "asya",
            "aylin", "burcu", "ceren", "gizem", "neslihan", "sedef",
            "nur", "cansu", "reyhan", "sema", "demet", "sevda", "sibel", "meltem",
            "buket", "inci", "betül", "tülin", "buse", "funda", "gülfem", "serra", "seçil"
        ]);
        set
    };

    /// Most frequent Turkish surnames.
    pub static ref SURNAMES: HashSet<String> = {
        let mut set = HashSet::new();
        insert_with_ascii(&mut set, &[
            "yılmaz", "kaya", "demir", "çelik", "şahin", "yıldız", "yıldırım", "öztürk",
            "aydın", "özdemir", "arslan", "doğan", "kılıç", "aslan", "çetin", "kara",
            "koç", "kurt", "özkan", "şimşek", "polat", "özcan", "korkmaz", "çakır",
            "erdoğan", "yavuz", "avcı", "şen", "acar", "keskin", "yüksel", "güler",
            "aksoy", "güneş", "bozkurt", "aktaş", "bulut", "ünal", "özmen", "özbal",
            "turan", "can", "gül", "özer", "sarıkaya", "tekeli", "akın", "aydem",
            "bayram", "coşkun", "duran", "ergin", "göksu", "gündoğdu", "inan", "ışık",
            "kahraman", "kaplan", "karaca", "karagöz", "karahan", "karataş", "kocaman",
            "mutlu", "ocak", "orhan", "sarı", "savaş", "soylu", "taş", "tekin",
            "topal", "toprak", "tunç", "türker", "türkmen", "uğur", "uzun", "varol",
            "yalçın", "yanık", "yücel", "zorlu", "özkaya", "albayrak", "barış", "bayrak",
            "bilgin", "bostancı", "çavuş", "çelebi", "çiftçi", "dalgıç", "duman", "durgun",
            "ekici", "elmas", "engin", "ergün", "genç", "göçer", "gök", "güzel",
            "hasanoğlu", "ilhan", "ipek", "karabulut", "kartal", "kavuk", "koca", "mert",
            "mor", "nalbant", "narin", "ocakoğlu", "oflu", "okur", "pamuk", "pehlivan",
            "sağlam", "sezer", "sürek", "tamgör", "tanrıverdi", "tatar", "torun", "tüfekçi",
            "akar", "gündüz", "aydoğan", "kök", "ekinci", "bakır", "yalçınkaya", "kandemir",
            "dağ", "karakuş", "şentürk", "çakmak", "özbek", "güngör", "yaman", "kocabaş", "başaran"
        ]);
        set
    };

    /// 81 Provinces of Turkey.
    pub static ref PROVINCES: HashSet<String> = {
        let mut set = HashSet::new();
        insert_with_ascii(&mut set, &[
            "adana", "adıyaman", "afyonkarahisar", "ağrı", "amasya", "ankara", "antalya",
            "artvin", "aydın", "balıkesir", "bilecik", "bingöl", "bitlis", "bolu", "burdur",
            "bursa", "çanakkale", "çankırı", "çorum", "denizli", "diyarbakır", "edirne",
            "elazığ", "erzincan", "erzurum", "eskişehir", "gaziantep", "giresun", "gümüşhane",
            "hakkari", "hatay", "ısparta", "mersin", "istanbul", "izmir", "kars", "kastamonu",
            "kayseri", "kırklareli", "kırşehir", "kocaeli", "konya", "kütahya", "malatya",
            "manisa", "kahramanmaraş", "mardin", "muğla", "muş", "nevşehir", "niğde", "ordu",
            "rize", "sakarya", "samsun", "siirt", "sinop", "sivas", "tekirdağ", "tokat",
            "trabzon", "tunceli", "şanlıurfa", "uşak", "van", "yozgat", "zonguldak",
            "aksaray", "bayburt", "karaman", "kırıkkale", "batman", "şırnak", "bartın",
            "ardahan", "ığdır", "yalova", "karabük", "kilis", "osmaniye", "düzce"
        ]);
        set
    };

    /// Major Turkish districts (ilçeler) commonly found in addresses.
    pub static ref DISTRICTS: HashSet<String> = {
        let mut set = HashSet::new();
        insert_with_ascii(&mut set, &[
            "kadıköy", "beşiktaş", "üsküdar", "şişli", "bakırköy", "maltepe", "kartal",
            "pendik", "ümraniye", "fatih", "beyoğlu", "sarıyer", "beylikdüzü", "ataşehir",
            "çankaya", "keçiören", "yenimahalle", "mamak", "etimesgut", "sincan", "altındağ",
            "gölbaşı", "konak", "karşıyaka", "bornova", "buca", "çiğli", "gaziemir", "balçova",
            "muratpaşa", "kepez", "konyaaltı", "alanya", "manavgat", "nilüfer", "osmangazi",
            "yıldırım", "seyhan", "çukurova", "yüreğir", "şahinbey", "şehitkamil", "melikgazi",
            "kocasinan", "odunpazarı", "tepebaşı", "izmit", "gebze", "darıca", "körfez"
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
