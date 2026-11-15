//! The plain word lists: Turkish given names, surnames and names that are also everyday words,
//! the 81 provinces and major districts. Each set holds the lower-case (Turkish rules) entries and
//! their ASCII-folded variants.
//!
//! Moved from `pii::gazetteer`, which re-exports them.

use super::insert_with_ascii;
use lazy_static::lazy_static;
use std::collections::HashSet;

lazy_static! {
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
}
