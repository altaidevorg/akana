//! Word lists for person-name detection beyond gazetteer.rs: more common Turkish given names
//! and surnames, common international first names, role/kinship/self-introduction triggers,
//! words that end a name, and well-known public figures (for the optional filter).
//!
//! All entries are lower case (Turkish rules); gazetteer.rs adds ASCII-folded variants.

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
