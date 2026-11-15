//! Word lists of the person-name lexicon and the place gazetteer beyond gazetteer.rs: more
//! common Turkish given names and surnames, common international first names, given names that
//! need evidence, places abroad, and demonyms.
//!
//! All entries are lower case (Turkish rules); the lexicon adds ASCII-folded variants.

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
