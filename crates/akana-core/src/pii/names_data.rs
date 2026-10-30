//! Word lists for person-name detection beyond gazetteer.rs: more common Turkish given names
//! and surnames, common international first names, role/kinship/self-introduction triggers,
//! words that end a name, and well-known public figures (for the optional filter).
//!
//! All entries are lower case (Turkish rules); gazetteer.rs adds ASCII-folded variants.

/// Common Turkish given names, in addition to gazetteer.rs's GIVEN_NAMES.
pub(crate) const COMMON_GIVEN_NAMES: &[&str] = &[
    "abdullah", "abdurrahman", "adem", "adnan", "ahmet", "akın", "alaattin", "alp", "alparslan",
    "alperen", "arif", "asım", "atakan", "ataberk", "atilla", "aybüke", "aydan", "aygül", "aykut",
    "aylin", "aynur", "ayten", "aysel", "aysun", "azra", "bahattin", "banu", "batuhan", "bayram",
    "bedirhan", "berat", "beren", "berk", "berkay", "berke", "betül", "beyza", "bilal", "birsen",
    "büşra", "burcu", "burhan", "cafer", "canan", "cansu", "celal", "cem", "cemal", "cemile",
    "cengiz", "ceren", "ceyda", "cihangir", "cüneyt", "çağatay", "çağla", "dilara", "dilek",
    "dursun", "ebru", "ecrin", "eda", "egemen", "ekin", "ekrem", "elçin", "elanur", "elvan", "emel",
    "emin", "emine", "emircan", "emirhan", "enes", "ercan", "erdal", "erdinç", "erhan", "erkan",
    "ersin", "ertan", "ertuğrul", "esma", "esra", "eylül", "ezel", "fadime", "fahri", "faruk",
    "fatih", "fatma", "feride", "feyza", "figen", "fikret", "filiz", "furkan", "gamze", "gizem",
    "gökay", "gökçe", "gökçen", "gökhan", "gönül", "gülay", "gülsüm", "gülşen", "gürkan", "habibe", "hacer", "halide",
    "hakan", "halil", "halime", "hamza", "handan", "hande", "harun", "hasan", "hatice", "havva",
    "hayrettin", "hilal", "hira", "hülya", "hüseyin", "ibrahim", "ilayda", "ilhan", "ilkay", "irem",
    "ismail", "işıl", "kadir", "kâmil", "kamil", "kemal", "kenan", "kerem", "kevser", "kübra",
    "levent", "mahmut", "mehmet", "melih", "melike", "melis", "meltem", "merve", "metin", "mevlüt",
    "mine", "muhammed", "muharrem", "murat", "musa", "mustafa", "naci", "nazlı", "necati", "necla",
    "nermin", "neslihan", "nesrin", "nevin", "nihat", "nilay", "nilüfer", "nisa", "nur", "nuran",
    "nurettin", "nurhan", "nuri", "nurten", "oğuzhan", "okan", "orhan", "osman", "ömer", "özcan",
    "özge", "özgen", "özkan", "pelin", "perihan", "rabia", "ramazan", "recep", "remzi", "reşat", "rukiye",
    "rümeysa", "sabri", "sadık", "safiye", "salih", "sami", "sedat", "sedef", "selahattin",
    "selçuk", "selim", "selin", "sema", "semih", "semra", "serap", "serdar", "serhat", "serkan",
    "sevda", "sevgül", "sevim", "seyfi", "sibel", "sinan", "songül", "suat", "sultan", "süleyman",
    "şahin", "şenol", "şerife", "şeyma", "şükrü", "tahsin", "talha", "tamer", "tarık", "tayfun",
    "tolga", "tuba", "tuğba", "tuğçe", "tuncay", "tuncer", "turan", "turgay", "turgut", "türkan",
    "uğur", "ümit", "ümmühan", "vedat", "veli", "yakup", "yaprak", "yasemin", "yaşar", "yeliz",
    "yeşim", "yılmaz", "yıldız", "yunus", "yusuf", "zafer", "zehra", "zekeriya", "zeki", "zeliha",
    "zerrin", "zeynep", "zübeyde", "zülfikar", "abdülkadir", "abdülhamit", "ahsen", "almila",
    "asel", "asya", "ayaz", "aybars", "bera", "berra", "buse", "çınar", "defne", "demet", "derin",
    "duru", "ecem", "ela", "elif", "eflin", "erva", "eymen", "göktuğ", "hiranur", "ilkin", "kaan",
    "kuzey", "lina", "meryem", "mira", "miraç", "miray", "nehir", "nil", "poyraz", "sare", "selen",
    "su", "tuana", "umay", "yağız", "zümra", "taha", "yiğit", "ömer faruk", "elif nur", "zeynep su",
    "nazım", "hikmet", "nedim", "baki", "tarkan", "muammer", "zühtü", "basri", "recai", "rıdvan",
    "muhterem", "kadri", "hicabi", "irfan",
    // more common names, many of them also words (Hakkı, Sedat, İsmet, Evrim)
    "abdülkerim", "akif", "alper", "aslı", "ayhan", "bekir", "bülent", "cahit", "caner", "cavit",
    "celil", "cevdet", "cevat", "ceyhun", "çiğdem", "davut", "doğan", "durmuş", "ebubekir", "ece",
    "emrah", "ender", "enver", "ergün", "erol", "esat", "eşref", "ethem", "evrim", "eyüp", "fadıl",
    "fahrettin", "faik", "ferhat", "ferit", "feyzi", "fikri", "fuat", "gani", "gökmen", "gülizar",
    "gülten", "güner", "hacı", "hafize", "haluk", "hamdi", "hamit", "hanife", "haydar", "hayri",
    "hayriye", "hıdır", "hilmi", "hulusi", "hüsnü", "ilyas", "ismet", "kâzım", "kazım",
    "kemalettin", "kubilay", "lütfi", "lütfü", "mahir", "mehdi", "melahat", "memduh", "mesut",
    "mithat", "muhittin", "muhsin", "mukadder", "mümtaz", "münevver", "müzeyyen", "nail", "naime",
    "nazmi", "necdet", "necip", "nedret", "nejat", "nevzat", "nihal", "nimet", "nusret", "nüket",
    "oktay", "pakize", "rahmi", "raif", "rasim", "rauf", "refik", "reha", "reşit", "rıfat",
    "sabahattin", "sabit", "sacit", "sadettin", "saffet", "sait", "saliha", "sebahattin", "seçkin",
    "selma", "semiha", "sena", "sencer", "serpil", "sertaç", "sevil", "seyit", "sezai", "sezgin",
    "sırrı", "suna", "süheyla", "şadiye", "şakir", "şemsettin", "şenay", "şevket", "şevki",
    "şükriye", "tacettin", "tahir", "talat", "tanju", "tevfik", "tufan", "tülay", "türker", "ülkü",
    "ümran", "vahit", "vasfi", "vecdi", "vehbi", "yalçın", "yasin", "yeter", "yıldıray", "yücel",
    "zekai", "zekiye", "ziya", "zuhal", "nurullah", "seray", "cemil", "şennur", "mesude",
    "selahaddin", "onurhan", "sabahat", "azize", "vecihe", "renas", "dıjvar", "serruh", "tugay",
    "gülşah", "yavuzhan", "erkam", "cebrail", "namık", "süreyya", "füsun", "abdulhakim", "mohammed",
    "rüştü", "fatin", "vural", "beyti", "muzaffer",
    // Given names of official-form names in Constitutional Court headers that no list had
    // (review 3 PII3-04).
    "nejdet", "hüsamettin", "heysem", "nahit", "hamdullah", "gülbin", "abdulhalim", "feyzullah",
    "fehmi", "ziver", "izzettin", "saadettin", "mecit", "veysi", "selami", "utku", "abdulkerim",
    "cumali", "isa",
    // Given names of court decisions read as other words (Kâni Vrana: the adjective kani, Hasan
    // Gürsel: gür, Onurhan Solmaz: solmak; pre-release audit).
    "kâni", "gürsel", "solmaz",
];

/// Common Turkish surnames, in addition to gazetteer.rs's SURNAMES.
pub(crate) const COMMON_SURNAMES: &[&str] = &[
    "acar", "akbulut", "akgül", "akın", "akkaya", "akman", "akpınar", "aksoy", "aktaş", "aktürk",
    "akyol", "akyüz", "alkan", "altıntaş", "altun", "arıkan", "arslan", "aslan", "ateş", "avcı",
    "aydemir", "aydın", "aydoğan", "aygün", "aykaç", "baysal", "bayram", "bektaş", "bilgin",
    "bilgiç", "bozkurt", "budak", "bulut", "cengiz", "coşkun", "çakır", "çakmak", "çalışkan",
    "çelik", "çetin", "çiftçi", "çınar", "çolak", "demir", "demirci", "demirel", "dinç", "doğan",
    "doğru", "duman", "durmaz", "duran", "ekinci", "elmas", "erbil", "erdem", "erdoğan", "eren",
    "ergin", "ergün", "erol", "ersoy", "ertekin", "ertuğrul", "eryılmaz", "esen", "gök", "gökçe",
    "güler", "gültekin", "gümüş", "gündoğdu", "güneş", "güngör", "günay", "güven", "güzel",
    "ışık", "kahraman", "kalkan", "kaplan", "kara", "karaca", "karadağ", "karakaya", "karakuş",
    "karaman", "karataş", "kaya", "kaymaz", "keskin", "kılıç", "kılınç", "kırkpınar", "koç",
    "kocabaş", "kocaman", "korkmaz", "kurt", "kuru", "küçük", "metin", "oral", "ozan", "öz",
    "özbek", "özcan", "özdemir", "özer", "özgür", "özkan", "öztürk", "parlak", "polat", "sarı",
    "savaş", "sevim", "sezer", "soylu", "sönmez", "şahin", "şen", "şimşek", "taş", "taşdemir",
    "tekin", "temel", "tok", "toprak", "tuncer", "turan", "tunç", "türk", "uçar", "uysal", "uzun",
    "ünal", "üstün", "yalçın", "yaman", "yavuz", "yazıcı", "yıldırım", "yıldız", "yılmaz", "yiğit",
    "yücel", "yüksel", "yurt", "zengin", "çağlayan", "kandemir", "tanrıverdi", "yıldızhan",
    "kızılkaya", "akdoğan", "sağlam", "sezgin", "tatlı", "ulusoy", "koçak", "gökcan", "bağcı",
    "akıncı", "özgüldür", "hakyemez", "akyel", "güleç", "kuz", "özkaya", "fidan", "akçil",
    "kıratlı", "özeren",
    "zorlu", "evren",
    // Surnames of court decisions read as other words (Serruh KALELİ: kale, Mehmet ERTEN: erte,
    // Sacit ADALI: ada; pre-release audit).
    "kaleli", "kömürcü", "erten", "adalı", "özler", "tuğcu", "başarır", "baltalı",
    // Round 2: applicants and members cited by name (Hikmet Kopar: kopmak, Emine Dilek Onaran:
    // onarmak, Abdullah Topçu: top, Sünuhi ARSAN: arsa) and Sabancı (saban). Not Bilin and Edin,
    // which at a sentence's start are mostly bilmek's and etmek's imperatives (Bilin ki ...).
    "arsan", "mencik", "onaran", "kalyoncu", "kopar", "topçu", "balcı", "gayretli", "balıkçı",
    "sabancı", "sökmen", "sürmeli", "akün",
];

/// Common international first names, leaving out ones that are everyday English words.
pub(crate) const INTERNATIONAL_GIVEN_NAMES: &[&str] = &[
    "james", "john", "robert", "michael", "david", "william", "richard", "joseph", "thomas",
    "charles", "christopher", "daniel", "matthew", "anthony", "donald", "steven", "paul", "andrew",
    "joshua", "kenneth", "kevin", "brian", "george", "timothy", "ronald", "edward", "jason",
    "jeffrey", "ryan", "jacob", "gary", "nicholas", "eric", "jonathan", "stephen", "larry",
    "justin", "scott", "brandon", "benjamin", "samuel", "gregory", "alexander", "patrick", "jack",
    "dennis", "jerry", "tyler", "aaron", "henry", "peter", "nathan", "zachary", "kyle", "noah",
    "ethan", "jeremy", "christian", "keith", "roger", "terry", "sean", "austin", "mary", "patricia",
    "jennifer", "linda", "elizabeth", "barbara", "susan", "jessica", "sarah", "karen", "nancy",
    "lisa", "margaret", "betty", "sandra", "ashley", "dorothy", "kimberly", "emily", "donna",
    "michelle", "carol", "amanda", "melissa", "deborah", "stephanie", "rebecca", "laura", "sharon",
    "cynthia", "kathleen", "amy", "angela", "shirley", "anna", "brenda", "pamela", "emma", "nicole",
    "helen", "samantha", "katherine", "christine", "debra", "rachel", "catherine", "carolyn",
    "janet", "maria", "heather", "diane", "julie", "olivia", "sophia", "isabella", "mia",
    "charlotte", "amelia", "harper", "evelyn", "abigail", "ella", "chloe", "hans", "jürgen",
    "klaus", "wolfgang", "stefan", "andreas", "markus", "pierre", "jean", "françois", "marie",
    "giovanni", "giuseppe", "marco", "luca", "francesca", "carlos", "juan", "josé", "miguel",
    "javier", "sofia", "ivan", "dmitri", "sergei", "olga", "natasha", "mohammed", "ahmed", "ali",
    "fatima", "omar", "hassan", "yusuf", "aisha",
    // more European, Russian, Spanish, Italian and Arabic names
    "martin", "carl", "august", "anton", "friedrich", "albert", "pyotr", "alexandra", "walter",
    "bernhard", "otto", "karl", "wilhelm", "heinrich", "ludwig", "franz", "fritz", "gustav",
    "hermann", "ernst", "rudolf", "dieter", "helmut", "gerhard", "günter", "horst", "werner",
    "ingrid", "ursula", "greta", "hannah", "louis", "henri", "jacques", "michel", "philippe",
    "alain", "claude", "nicolas", "antoine", "sophie", "camille", "isabelle", "nathalie", "giorgio",
    "alessandro", "antonio", "francesco", "matteo", "lorenzo", "paolo", "roberto", "sergio",
    "valentina", "chiara", "giulia", "pedro", "pablo", "diego", "luis", "manuel", "fernando",
    "jorge", "alejandro", "ricardo", "rafael", "lucía", "carmen", "isabel", "elena", "vladimir",
    "nikolai", "mikhail", "alexei", "andrei", "boris", "yuri", "igor", "oleg", "tatiana",
    "svetlana", "irina", "natalia", "ekaterina", "anastasia", "hiroshi", "takeshi", "kenji",
    "ahmad", "mahmoud", "khalid", "nelson", "oliver", "harry", "arthur", "alfred", "ernest",
    "frank", "fred", "howard", "leonard", "lewis", "ralph", "raymond", "wayne", "eugene", "harold",
    "jesse", "bruce", "gerald", "lawrence", "philip", "russell", "victor", "vincent", "wesley",
    "julia", "victoria", "alice", "lily", "lucy", "megan", "nellie", "eliseo", "serhiy", "sergo",
    "alex", "sam", "theodore", "franklin", "dwight",
];

/// Given names whose everyday sense is frequent in modern text (metin "text", emin "sure",
/// kuzey "north", rüya "dream", eylül "September"). Like gazetteer.rs's POLYSEMOUS_NAMES they
/// are names only with evidence: a title, a role, a surname, or a capital mid-sentence.
pub(crate) const AMBIGUOUS_NAMES: &[&str] = &[
    "akın", "arzu", "ayaz", "bayram", "çağlar", "demet", "derin", "duru", "egemen", "ekin", "emin",
    "eylül", "fahri", "filiz", "gaye", "gizem", "gönül", "gözde", "haziran", "hilal", "ilkin",
    "inci", "ipek", "ırmak", "kuzey", "levent", "masal", "metin", "nehir", "nisan", "nur", "öykü",
    "ramazan", "rıza", "rüya", "sadık", "sevda", "simge", "su", "sultan", "ümit", "veli", "yaprak",
    "yıldız", "zeki", "ela", "mine", "tuna", "hale", "nil", "bahar", "ada", "sevgi", "yaren",
    "aras", "christian", "austin", "charlotte", "carol", "heather", "harper", "jean",
    "hakkı",
    // the same court headers' given names that are also everyday words (review 3 PII3-04)
    "bahadır", "önder", "adil", "aziz", "ulaş", "servet", "halis", "şeref", "salim", "kudret",
];

/// Given names that are far more often something else: a celestial body, a direction, a region
/// or an abstract noun (Güneş'in, Ay'a, Kuzey'de, Ege'de, Kader). Like the provinces and
/// districts that are names (Aydın, Fatih) and the names in person_names.txt that are common
/// words (Kanun, Beyan, Umman), they need a title, a role or a known name next to them; an
/// apostrophe or a capital mid-sentence is not enough.
pub(crate) const RARE_NAMES: &[&str] = &[
    "ay", "gün", "güneş", "dünya", "kuzey", "güney", "kader", "hayat", "ilke", "asya", "ege",
    "toros", "evren", "doğa", "ışık", "bulut", "yağmur", "rüzgar", "rüzgâr", "toprak", "kaya",
    "demir", "çiçek", "melek", "kartal", "aslan",
];

/// Lower-case words that are given names but also everyday English or code words: a lower-case
/// "mine" or "nil" alone is not taken as a name.
pub(crate) const LOWERCASE_COLLISIONS: &[&str] = &[
    "mine", "tuna", "nil", "hale", "mira", "lina", "aria", "maya", "dana", "sara", "nora", "elle",
    "lara", "ella", "jack", "will", "mark", "bill", "rose", "grace", "hope", "joy", "may", "june",
    "april", "summer", "paris", "chase", "hunter", "max", "sky", "star",
];

/// Honorifics before a name: any capitalised word after them is taken as one (Sayın Kırkpınar,
/// Dr. Yücel, Prof. Dr. Ahmet).
pub(crate) const HONORIFICS: &[&str] = &[
    "sayın", "sn", "prof", "dr", "doç", "yrd", "av", "müh", "uzm", "öğr", "gör", "op", "ord", "mr",
    "mrs", "ms", "miss", "sir", "dear", "madam", "mx",
];

/// Titles after a name, which end it: Mehmet Bey, Fatma Hanım'a, Ahmet Hoca.
pub(crate) const POST_TITLES: &[&str] = &[
    "bey", "beyin", "beye", "beyi", "beyden", "beyle", "beyefendi", "hanım", "hanımın", "hanıma",
    "hanımı", "hanımdan", "hanımla", "hanımefendi", "efendi", "paşa", "ağa", "hoca", "hocam",
    "usta", "abi", "abla", "amca", "teyze", "dayı", "hala", "enişte", "yenge", "baba", "ana",
    "sultan", "çavuş", "reis",
];

/// Words that introduce a name after a colon: "Ad Soyad: ...", "İmza: ...", "Name: ...".
pub(crate) const COLON_TRIGGERS: &[&str] = &[
    "ad", "adı", "adım", "isim", "ismi", "name", "full name", "first name", "last name", "surname",
    "soyad", "soyadı", "ad soyad", "ad soyadı", "adı soyadı", "adı ve soyadı", "imza", "yazar",
    "author", "hazırlayan", "gönderen", "kimden", "sorumlu", "yetkili", "contact", "iletişim",
    "ilgili kişi", "müşteri", "hasta", "öğrenci", "veli",
];

/// Words right before a name that mark it as one: roles in legal, medical, business and school
/// texts, possessive kinship terms, and self-introductions.
pub(crate) const NAME_TRIGGERS: &[&str] = &[
    // legal and official roles
    "başvurucu", "başvuran", "sanık", "sanıklar", "müşteki", "mağdur", "mağdure", "tanık",
    "şüpheli", "vekil", "vekili", "vekilim", "katılan", "müdahil", "şikayetçi", "mirasçı", "murisi",
    "kiracı", "kiracısı", "kiraya", "işçi", "işveren", "sürücü", "sürücüsü", "alıcı", "satıcı",
    "kefil", "temsilci", "temsilcisi", "üye", "hâkim", "hakim", "yargıç", "savcı", "noter",
    "katip", "kâtip",
    "bilirkişi", "arabulucu", "imza", "hazırlayan", "gönderen", "onaylayan",
    // possessive kinship and relations
    "annem", "babam", "eşim", "kızım", "oğlum", "kardeşim", "abim", "ablam", "ağabeyim", "teyzem",
    "amcam", "dayım", "halam", "yeğenim", "kuzenim", "dedem", "ninem", "anneannem", "babaannem",
    "torunum", "gelinim", "damadım", "kayınvalidem", "kayınpederim", "sevgilim", "nişanlım",
    "arkadaşım", "komşum", "patronum", "müdürüm", "öğretmenim", "doktorum", "avukatım",
    "müvekkilim", "danışmanım", "asistanım",
    // English roles
    "mr", "mr.", "mrs", "mrs.", "ms", "ms.", "miss", "sir", "dear", "dr", "prof", "colleague",
    "client", "patient", "applicant", "defendant", "plaintiff", "witness", "tenant",
    // military ranks, offices and professions
    "tümgeneral", "korgeneral", "orgeneral", "tuğgeneral", "mareşal", "amiral", "tümamiral",
    "koramiral", "oramiral", "tuğamiral", "yüzbaşı", "üsteğmen", "kurmay", "onbaşı", "başçavuş",
    "astsubay", "polis memuru", "komutan", "komutanı", "yazar", "şair", "ressam", "besteci",
    "yönetmen", "oyuncu", "sanatçı", "şarkıcı", "futbolcu", "antrenör", "gazeteci", "muhabir",
    "başkanvekili",
    "başsavcı", "başsavcısı", "başsavcıvekili", "savcısı", "hakimi", "yargıcı", "müsteşar",
    "müsteşarı", "büyükelçi", "büyükelçisi", "konsolos", "başkonsolos", "sekreteri", "senatör",
    "senatörü", "profesör", "doçent", "eşi", "kızı", "oğlu", "annesi", "babası", "kardeşi",
];

/// Roles that may stand before an organisation as well as a person (Davacı Şirket, Müşteri
/// Hizmetleri, Üye Devletler): after them a common word is not taken as a name, as it is
/// after a title or a kinship word (Cumhurbaşkanı İsmet İnönü, kızım Derin).
pub(crate) const ORG_ROLES: &[&str] = &[
    "müşteri", "müşterisi", "müşterimiz", "kullanıcı", "kullanıcısı", "kullanıcımız", "müvekkil",
    "müvekkili", "müvekkilimiz", "müvekkilim", "davacı", "davalı", "abone", "abonesi", "abonemiz",
    "üye", "üyesi", "üyemiz", "borçlu", "alacaklı", "kiracı", "kiracısı", "kiraya", "işveren",
    "alıcı", "satıcı", "kefil", "temsilci", "temsilcisi", "gönderen", "hazırlayan", "onaylayan",
    "başvurucu", "başvuran", "şikayetçi", "katılan", "müdahil", "mirasçı", "imza", "yetkili",
    "yetkilisi", "yetkilimiz", "client", "applicant", "defendant", "plaintiff", "tenant",
];

/// Phrases right before a name in a self-introduction or a signature. "ben" alone is a common
/// word, so only a known given name after it is taken.
pub(crate) const SELF_INTRO: &[&str] = &[
    "adım", "ismim", "my name is",
    "saygılarımla", "sevgilerimle", "regards", "best regards", "kind regards", "sincerely",
    "thanks", "teşekkürler", "cheers",
];

/// Words before a name that also stand before the names of things (Darbenin Adı Balyoz,
/// this is Paris): only a known given name after them is taken, as after "ben".
pub(crate) const KNOWN_NAME_INTRO: &[&str] = &[
    "ben", "adı", "ismi", "adını", "ismini", "i am", "i'm", "this is", "it's", "im",
];

/// Lower-case particles inside a name (Karl von Frisch, Ludwig van Beethoven, Leonardo da
/// Vinci, Osama bin Laden); "da" and "de" only after a name that is not Turkish, where they
/// are the clitic (Ahmet de geldi).
pub(crate) const NAME_PARTICLES: &[&str] = &[
    "von", "van", "der", "den", "del", "della", "di", "du", "le", "la", "bin", "ibn", "al", "el",
    "dos", "das", "ter", "ten", "zu", "da", "de",
];

/// Capitalised words that end a name instead of continuing it (Ahmet Bey, Kemal Paşa Lisesi,
/// Ahmet Pazartesi geldi).
pub(crate) const NAME_STOPPERS: &[&str] = &[
    "ocak", "şubat", "mart", "nisan", "mayıs", "haziran", "temmuz", "ağustos", "eylül", "ekim",
    "kasım", "aralık", "pazartesi", "salı", "çarşamba", "perşembe", "cuma", "cumartesi", "pazar",
    "lisesi", "okulu", "üniversitesi", "fakültesi", "hastanesi", "caddesi", "sokak", "sokağı",
    "mahallesi", "bulvarı", "meydanı", "parkı", "camii", "vakfı", "derneği", "holding", "a.ş.",
    "ltd.", "şti.", "anonim", "limited", "şirketi", "bakanlığı", "müdürlüğü", "başkanlığı",
    "belediyesi", "valiliği", "mahkemesi", "dairesi", "kurulu", "partisi", "köprüsü", "havalimanı",
    "stadı", "salonu", "merkezi", "enstitüsü", "kütüphanesi", "müzesi", "denizi", "gölü", "nehri",
    "dağı", "dağları", "ovası", "körfezi", "boğazı", "adası", "kuvvetleri", "komutanlığı",
    "bayramı", "günü", "haftası", "ödülü", "kupası", "ligi", "and", "ve", "ile", "the", "of", "inc",
    "llc", "gmbh", "corp", "university", "street",
    // words of a decision's heading and signature block, which end the official form's name
    // (Mehmet KAYA KARAR; review 4 DOWN4-09)
    "karar", "kararı", "tarih", "mühür",
    // companies; the countries, regions and cities abroad (FOREIGN_PLACES) and the
    // nationalities (DEMONYMS) end a name too
    "google", "microsoft", "apple", "amazon", "facebook", "twitter",
    // legal and institution words capitalised as terms
    "grand", "prix", "bank", "kanun", "kanunu", "yönetmelik", "tüzük", "genelge", "kararname",
    "anayasa", "mahkeme", "daire", "kurul", "bakanlık", "kurum", "kurumu", "ticaret", "sanayi",
    "san.", "tic.", "idaresi", "idaresinde", "müdürlüğünde",
    // A word's period is trimmed before the list is looked up (LTD. is ltd), and company words
    // end a name (Hayat Sigorta A.Ş.; review 3 PII3-05, PII3-06).
    "ltd", "şti", "tic", "sigorta", "sigortası", "sigortaları", "bankası", "inşaat", "holdingi",
    "diğerleri", "digerleri", "başvurusu", "basvurusu",
];

/// Countries, regions and large cities outside the province list: places (NER's LOC), and words
/// that end a name.
pub(crate) const FOREIGN_PLACES: &[&str] = &[
    "türkiye", "almanya", "fransa", "ingiltere", "amerika", "abd", "rusya", "italya", "ispanya",
    "yunanistan", "irak", "iran", "suriye", "azerbaycan", "japonya", "çin", "hollanda", "belçika",
    "avusturya", "isviçre", "kıbrıs", "bulgaristan", "gürcistan", "ukrayna", "mısır", "israil",
    "filistin", "katar", "arabistan", "kanada", "brezilya", "hindistan", "pakistan", "afganistan",
    "avrupa", "asya", "afrika", "anadolu", "trakya", "londra", "berlin", "moskova", "atina", "roma",
    "viyana", "brüksel", "bakü", "tahran", "bağdat", "şam", "kahire", "tokyo", "pekin", "germany",
    "france", "england", "america", "russia", "turkey", "europe", "london", "york", "kore",
    "vietnam", "tayland", "endonezya", "malezya", "singapur", "avustralya", "meksika", "arjantin",
    "şili", "kolombiya", "venezuela", "küba", "portekiz", "polonya", "çekya", "macaristan",
    "romanya", "sırbistan", "hırvatistan", "bosna", "makedonya", "karadağ", "arnavutluk",
    "slovenya", "slovakya", "danimarka", "norveç", "isveç", "finlandiya", "estonya", "letonya",
    "litvanya", "belarus", "moldova", "kazakistan", "özbekistan", "türkmenistan", "kırgızistan",
    "tacikistan", "moğolistan", "ürdün", "lübnan", "kuveyt", "bahreyn", "umman", "yemen", "libya",
    "tunus", "cezayir", "fas", "sudan", "etiyopya", "kenya", "nijerya", "somali", "irlanda",
    "iskoçya", "galler", "izlanda", "lüksemburg", "monako", "vatikan", "malta", "tayvan",
    "filipinler", "bangladeş", "nepal", "kosova", "kafkasya", "balkanlar", "ortadoğu", "delhi",
];

/// Nationalities and demonyms: no person's name (NER), and words that end a name.
pub(crate) const DEMONYMS: &[&str] = &[
    "türk", "kürt", "alman", "ingiliz", "fransız", "amerikalı", "rus", "arap", "yunan", "ermeni",
    "laz", "çerkez", "gürcü", "boşnak", "arnavut",
];

/// Words a company's name ends with (Hayat Sigorta, Axa Sigorta A.Ş., Koç Holding): the words
/// before them name the company (Doğan Holding), unless one is a given name that is only a name
/// (Ahmet Yılmaz İnşaat is a person's business; review 3 PII3-05). As `Word` reads them: lower case,
/// without the period (A.Ş. is the initials a, ş).
pub(crate) const COMPANY_WORDS: &[&str] = &[
    "sigorta", "sigortası", "sigortaları", "bankası", "bank", "holding", "holdingi", "inşaat",
    "ltd", "şti", "anonim", "limited", "şirketi", "ticaret", "tic", "sanayi", "san", "kooperatifi",
];

/// Well-known public figures (historical, political, literary, artistic, scientific) whose
/// names are not private data, for `TurkishPiiEngine::set_preserve_public_figures`. Matched as
/// whole phrases, and a detected name is kept only when it lies inside one: Mustafa Kemal
/// Yılmaz or a private Nedim is still masked. Single words are left out when they are also
/// common given names (Nedim, Baki, Alparslan, Tarkan).
pub(crate) const PUBLIC_FIGURES: &[&str] = &[
    // founders, statesmen
    "atatürk", "mustafa kemal", "mustafa kemal atatürk", "gazi mustafa kemal", "ismet inönü",
    "inönü", "celal bayar", "adnan menderes", "cemal gürsel", "cevdet sunay", "fahri korutürk",
    "kenan evren", "turgut özal", "süleyman demirel", "ahmet necdet sezer", "abdullah gül",
    "recep tayyip erdoğan", "bülent ecevit", "necmettin erbakan", "tansu çiller", "mesut yılmaz",
    "alparslan türkeş", "deniz baykal", "kemal kılıçdaroğlu", "devlet bahçeli", "fevzi çakmak",
    "kazım karabekir", "kâzım karabekir", "ali fuat cebesoy", "rauf orbay", "latife hanım",
    "halide edib", "halide edib adıvar", "nene hatun", "kara fatma",
    // Ottoman and earlier
    "osman gazi", "orhan gazi", "fatih sultan mehmet", "fatih sultan mehmed", "yavuz sultan selim",
    "kanuni sultan süleyman", "kanuni", "ii. abdülhamit", "sultan abdülhamit",
    "barbaros hayrettin paşa", "barbaros hayreddin paşa", "piri reis", "mimar sinan",
    "evliya çelebi", "katip çelebi", "hezarfen ahmet çelebi", "sultan alparslan", "hürrem sultan",
    "kösem sultan", "mehmet ali paşa", "enver paşa", "talat paşa", "cemal paşa",
    // thought, literature, poetry
    "mevlana", "mevlâna", "mevlana celaleddin rumi", "rumi", "yunus emre", "hacı bektaş veli",
    "pir sultan abdal", "karacaoğlan", "aşık veysel", "âşık veysel", "nasreddin hoca",
    "nasrettin hoca", "mehmet akif ersoy", "mehmet akif", "namık kemal", "ziya paşa",
    "tevfik fikret", "yahya kemal", "yahya kemal beyatlı", "nazım hikmet", "nazım hikmet ran",
    "orhan veli", "orhan veli kanık", "cahit sıtkı tarancı", "necip fazıl kısakürek",
    "ahmet hamdi tanpınar", "sabahattin ali", "sait faik", "sait faik abasıyanık", "yaşar kemal",
    "aziz nesin", "orhan pamuk", "elif şafak", "oğuz atay", "attila ilhan", "cemal süreya",
    "can yücel", "ahmet arif", "ömer seyfettin", "reşat nuri güntekin", "halit ziya uşaklıgil",
    "peyami safa", "ziya gökalp", "fuzuli", "nef'i",
    // science
    "aziz sancar", "cahit arf", "gazi yaşargil", "feza gürsey", "ali kuşçu", "uluğ bey", "farabi",
    "ibn sina", "biruni", "harezmi", "cezeri", "el cezeri",
    // music, film, sports
    "barış manço", "cem karaca", "zeki müren", "sezen aksu", "müslüm gürses", "neşet ertaş",
    "ajda pekkan", "ibrahim tatlıses", "kemal sunal", "türkan şoray", "yılmaz güney", "şener şen",
    "cüneyt arkın", "fatma girik", "hülya koçyiğit", "adile naşit", "münir özkul",
    "naim süleymanoğlu", "hakan şükür", "arda turan", "nuri şahin",
    // international
    "albert einstein", "einstein", "isaac newton", "newton", "charles darwin", "darwin",
    "william shakespeare", "shakespeare", "leonardo da vinci", "da vinci", "napoleon",
    "napoleon bonaparte", "karl marx", "sigmund freud", "mozart", "beethoven", "bach",
    "aristoteles", "aristotle", "platon", "plato", "sokrates", "socrates", "galileo", "marie curie",
    "nikola tesla", "tesla", "abraham lincoln", "winston churchill", "gandhi", "mahatma gandhi",
    "nelson mandela", "martin luther king", "barack obama", "donald trump", "joe biden",
    "vladimir putin", "putin", "elon musk", "bill gates", "steve jobs",
];
