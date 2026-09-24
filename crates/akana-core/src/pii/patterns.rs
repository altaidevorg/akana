//! Regex and syntactic pattern extractors for Turkish PII entities.
//!
//! Recognizes:
//! - Turkish Mobile and Fixed-Line Phone Numbers
//! - Email Addresses (Standard and Obfuscated)
//! - IP Addresses and Port Numbers
//! - Passport & Driver License Numbers
//! - Birth Dates, Age & Age Ranges
//! - Financial Triggers (CVV, SKT/Expiry, Account No, Customer No, Salary)
//! - Credentials & Secrets (Passwords, PINs)
//! - Multi-token Turkish Postal Addresses
//! - KVKK Article 6 Sensitive Data (Blood type, Health conditions, Religion)

use lazy_static::lazy_static;
use regex::Regex;

lazy_static! {
    /// Phone regex matching Turkish mobile, landline, 0850, and 444 numbers.
    ///
    /// Matches:
    /// - `+90 532 123 45 67`, `+905321234567`
    /// - `0532 123 45 67`, `0 532 123 45 67`
    /// - `(0212) 123 45 67`, `0212 123 45 67`
    /// - `532 123 45 67`, `5321234567`
    /// - `0850 123 45 67`, `444 0 123`, `444 1234`
    pub static ref PHONE_REGEX: Regex = Regex::new(
        r"(?x)
        (?:(?:\+90|0090)\s*|\b)
        (?:
            (?:\(?0?[2-5][0-9]{2}\)?[\s.-]?[0-9]{3}[\s.-]?[0-9]{2}[\s.-]?[0-9]{2})
            |
            (?:\(?0?850\)?[\s.-]?[0-9]{3}[\s.-]?[0-9]{2}[\s.-]?[0-9]{2})
            |
            (?:444[\s.-]?[0-9][\s.-]?[0-9]{3})
            |
            (?:444[\s.-]?[0-9]{4})
        )
        \b"
    ).unwrap();

    /// Standard RFC 5322 email regex.
    pub static ref EMAIL_REGEX: Regex = Regex::new(
        r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"
    ).unwrap();

    /// Obfuscated email regex (e.g. `ahmet [at] gmail [dot] com`, `ali (at) sirket . com`).
    pub static ref EMAIL_OBFUSCATED_REGEX: Regex = Regex::new(
        r"(?i)\b(?:[A-Z0-9._%+-]+\s*(?:\[at\]|\(at\))\s*[A-Z0-9.-]+\s*(?:\[dot\]|\(dot\)|\.)\s*[A-Z]{2,}|[A-Z0-9._%+-]+@[A-Z0-9.-]+\s*(?:\[dot\]|\(dot\)|\.)\s*[A-Z]{2,})\b"
    ).unwrap();

    /// IPv4 address regex.
    pub static ref IPV4_REGEX: Regex = Regex::new(
        r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
    ).unwrap();

    /// IPv6 address regex.
    pub static ref IPV6_REGEX: Regex = Regex::new(
        r"(?i)\b(?:[0-9a-f]{1,4}:){7}[0-9a-f]{1,4}\b|\b(?:[0-9a-f]{1,4}:){1,7}:|:(?::[0-9a-f]{1,4}){1,7}\b"
    ).unwrap();

    /// Port number following 'portu', 'portunda', etc., or ':PORT'
    pub static ref PORT_REGEX: Regex = Regex::new(
        r"(?i)\b(\d{1,5})\s*(?:nolu\s+|numaralı\s+)?port(?:u|unda|undan|una)?\b|:([0-9]{2,5})\b"
    ).unwrap();

    /// Turkish Passport number pattern (e.g. `U12345678`, `A12345678`, `EP1234567`).
    pub static ref PASSPORT_REGEX: Regex = Regex::new(
        r"(?i)\b(?:pasaport(?:\s*no(?:su)?)?[:\s]*)?([UuAa][0-9]{7,8}|[Ee][Pp][0-9]{7,8})\b"
    ).unwrap();

    /// Age expression: `42 yaşında`, `35 yaşındayım`.
    pub static ref AGE_REGEX: Regex = Regex::new(
        r"(?i)\b([1-9][0-9]?)\s*(?:yaşında|yaşındayım|yaşındaki)\b"
    ).unwrap();

    /// Age range: `20-30 yaş arası`, `40 - 50 yaşlarında`.
    pub static ref AGE_RANGE_REGEX: Regex = Regex::new(
        r"(?i)\b([1-9][0-9]?)\s*[-–]\s*([1-9][0-9]?)\s*(?:yaş|yaşlarında|yaş\s*arası|yaş\s*aralığı(?:nda)?)\b"
    ).unwrap();

    /// Card CVV / CVC code in context.
    pub static ref CVV_REGEX: Regex = Regex::new(
        r"(?i)\b(?:cvv(?:(?:\s*['’]?\s*s?[iı])|\s+kodu)?|cvc2?|güvenlik\s+kodu(?:m|nu)?|güvenlik\s+no(?:su)?)[:\s]*([0-9]{3,4})\b"
    ).unwrap();

    /// Card expiration date (SKT): `12/26`, `08/2028`.
    pub static ref CARD_EXPIRY_REGEX: Regex = Regex::new(
        r"(?i)\b(?:skt|son\s+kullanma\s+tarihi|exp(?:\.|\s+date)?)[:\s]*(0[1-9]|1[0-2])\s*[/.-]\s*([0-9]{2}|20[2-3][0-9])\b"
    ).unwrap();

    /// Bank Account No / Customer No in context.
    pub static ref ACCOUNT_NO_REGEX: Regex = Regex::new(
        r"(?i)\b(?:m[üuÜU][şsŞS]ter[iİıI]\s*(?:no(?:su)?|numaras[ıiIİ]|id(?:s[iİıI])?|kodu)|muster[iİıI]\s*no)[:\s]*([A-Za-z0-9_-]{4,16})\b|\b(?:hesap(?:\s*no(?:su)?)?)[:\s]*([0-9]{3,5}[- ][0-9]{5,8}(?:[- ][0-9]{2,4})?|[0-9]{6,16})\b|\b(?:[öoÖO]deme\s+tal[iİıI]mat[ıiIİ](?:n[ıiIİ])?)[:\s]*([0-9]{3,5}[- ][0-9]{5,8}(?:[- ][0-9]{2,4})?)\b"
    ).unwrap();

    /// Turkish bank account standard format: 4-7-3 digits (e.g. 5427-5551073-146).
    pub static ref BANK_ACCOUNT_FORMAT_REGEX: Regex = Regex::new(
        r"\b\d{3,5}-\d{5,8}-\d{2,4}\b"
    ).unwrap();

    /// Tax Identification Number (VKN / Vergi Kimlik No) in context.
    pub static ref VERGI_NO_REGEX: Regex = Regex::new(
        r"(?i)\b(?:verg[iİıI]\s*(?:k[iİıI]ml[iİıI]k)?\s*no(?:su)?|vkn)[\s:]*([0-9]{10})\b"
    ).unwrap();

    /// Password / Secret credentials in context.
    pub static ref CREDENTIALS_REGEX: Regex = Regex::new(
        r"(?i)\b(?:şifre(?:m|si)?|parola(?:m|sı)?|pin(?:\s*kodu)?|kullanıcı\s+şifresi)[:\s]*([A-Za-z0-9!@#$%^&*()_+=\-]{4,32})\b"
    ).unwrap();

    /// Salary / Income amounts in context: `maaşım 45.000 TL`, `aylık gelir: 60000 TRY`.
    pub static ref SALARY_REGEX: Regex = Regex::new(
        r"(?i)\b(?:maaş(?:ım|ı)?|aylık\s+gelir(?:im|i)?|ücret(?:im|i)?)[:\s]*((?:aylık\s+)?[0-9]{1,3}(?:[.,][0-9]{3})*(?:[.,][0-9]{2})?|[0-9]+)\s*(?:tl|try|lira|₺)\b"
    ).unwrap();

    /// Driver's License number in context (supports new EU/TR format e.g. F1839528 or D6923973).
    pub static ref DRIVER_LICENSE_REGEX: Regex = Regex::new(
        r"(?i)\b(?:ehl[iİıI]yet(?:\s*no(?:su)?)?|s[üuÜU]r[üuÜU]c[üuÜU]\s+belges[iİıI](?:\s*no(?:su)?)?|[şsŞS]of[öoÖO]r\s+belge(?:s[iİıI]|m)?(?:\s*no(?:su)?)?)(?:[\s:]+(?:yen[iİıI]lend[iİıI]|ver[iİıI]ld[iİıI]))?[:\s]*([A-Z]?[0-9]{6,8})\b"
    ).unwrap();

    /// SGK / SSK / Bağkur registration number in context.
    pub static ref SGK_NO_REGEX: Regex = Regex::new(
        r"(?i)\b(?:sgk|ssk|bağkur)(?:\s*(?:sicil\s+no|sicil\s+numarası|no))?[:\s]*([0-9- ]{9,16})\b"
    ).unwrap();

    /// Positive birth date regex: matches when explicit birth triggers are present.
    /// E.g. `doğum tarihi: 12.04.1985`, `14 Nisan 1993 doğumlu`, `09.09.1992 doğanlara`, `26.05.1953'te doğan`
    pub static ref BIRTH_DATE_POSITIVE_REGEX: Regex = Regex::new(
        r"(?ix)
        (?:
            (?:doğum\s+tarihi(?:m|miz)?[:\s]*)
            (?:
                ([0-3]?[0-9])\s+(ocak|şubat|mart|nisan|mayıs|haziran|temmuz|ağustos|eylül|ekim|kasım|aralık)\s+(19[2-9][0-9]|20[0-2][0-9])
                |
                (0[1-9]|[12][0-9]|3[01])[./-](0[1-9]|1[0-2])[./-](19[2-9][0-9]|20[0-2][0-9])
                |
                (19[2-9][0-9]|20[0-2][0-9])-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])
            )
        )
        |
        (?:
            (?:
                ([0-3]?[0-9])\s+(ocak|şubat|mart|nisan|mayıs|haziran|temmuz|ağustos|eylül|ekim|kasım|aralık)\s+(19[2-9][0-9]|20[0-2][0-9])
                |
                (0[1-9]|[12][0-9]|3[01])[./-](0[1-9]|1[0-2])[./-](19[2-9][0-9]|20[0-2][0-9])
                |
                (19[2-9][0-9]|20[0-2][0-9])-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])
            )
            (?:\s*(?:'de|'da|'te|'ta|'e|'a)?\s*(?:doğumlu(?:yum)?|doğan(?:lar)?(?:a|a\s+özel)?))
        )"
    ).unwrap();

    /// Positive private date trigger regex: matches personal lifecycle/account dates.
    /// E.g. `fatura kesim tarihim: 01.10.2025`, `randevu günüm 2 mayıs 2011`, `teslimat tarihi 17 ağustos 2001`,
    /// `mezuniyet tarihi 2 Aralık 1957`, `kayıt tarihim ayın 16'sı`
    pub static ref PRIVATE_DATE_TRIGGER_REGEX: Regex = Regex::new(
        r"(?ix)
        \b(?:
            fatura\s+kesim\s+tarihi(?:m)?
            |son\s+ödeme\s+tarihi(?:m)?
            |ödeme\s+itiraz\s+tarihi(?:m)?
            |iptal\s+talebi\s+tarihi(?:m)?
            |servis\s+(?:ziyaret\s+)?tarihi(?:m)?
            |abonelik\s+(?:başlangıç|başlangıcı|tarihi)(?:m)?
            |üyelik\s+tarihi(?:m)?
            |randevu\s+(?:günü|tarihi)(?:m)?
            |kurulum\s+(?:randevum?|günü|tarihi)(?:m)?
            |muayene\s+randevum?
            |seçtiğim\s+teslim(?:at)?\s+(?:günü|tarihi)(?:m)?
            |teslim(?:at)?\s+(?:günü|tarihi)(?:m)?
            |adresime\s+teslim\s+tarihi
            |rezervasyon\s+tarihi(?:m)?
            |işlem\s+tarihi(?:m)?
            |mezuniyet\s+tarihi(?:m)?
            |sınav\s+kayıt\s+tarihi(?:m)?
            |kayıt\s+tarihi(?:m)?
            |üniversite\s+kayıt\s+tarihi(?:m)?
            |işe\s+giriş\s+tarihi(?:m)?
            |işten\s+ayrılış\s+tarihi(?:m)?
            |sözleşme\s+(?:başlangıç|bitiş|imza)?\s*tarihi(?:m)?
            |başvuru\s+tarihi(?:m)?
            |talep\s+tarihi(?:m)?
            |doğum\s+tarihi(?:m)?
        )\s*(?:için)?[:\s]+
        (
            (?:(?:pazartesi|salı|sali|çarşamba|carsamba|perşembe|persembe|cuma|cumartesi|pazar)\s+)?[0-3]?[0-9]\s+(?:ocak|şubat|mart|nisan|mayıs|haziran|temmuz|ağustos|eylül|ekim|kasım|aralık)(?:\s+(?:19[2-9][0-9]|20[0-3][0-9]))?
            |
            (?:0[1-9]|[12][0-9]|3[01])[./-](?:0[1-9]|1[0-2])[./-](?:19[2-9][0-9]|20[0-3][0-9])
            |
            (?:19[2-9][0-9]|20[0-3][0-9])-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])
            |
            ayın\s+[0-3]?[0-9](?:\'?(?:i|si|sı|ü|u))?
        )\b"
    ).unwrap();

    /// Date preceding a lifecycle role trigger with `olan`:
    /// E.g. `4 mayıs 1995 olan ödeme itiraz tarihim`, `13.12.2026 olan kayıt tarihim`
    pub static ref PRIVATE_DATE_POST_TRIGGER_REGEX: Regex = Regex::new(
        r"(?ix)
        \b(
            (?:(?:pazartesi|salı|sali|çarşamba|carsamba|perşembe|persembe|cuma|cumartesi|pazar)\s+)?[0-3]?[0-9]\s+(?:ocak|şubat|mart|nisan|mayıs|haziran|temmuz|ağustos|eylül|ekim|kasım|aralık)(?:\s+(?:19[2-9][0-9]|20[0-3][0-9]))?
            |
            (?:0[1-9]|[12][0-9]|3[01])[./-](?:0[1-9]|1[0-2])[./-](?:19[2-9][0-9]|20[0-3][0-9])
            |
            (?:19[2-9][0-9]|20[0-3][0-9])-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])
            |
            ayın\s+[0-3]?[0-9](?:\'?(?:i|si|sı|ü|u))?
        )\s+olan\s+(?:[^\s,;.]+\s+){1,3}(?:tarihi(?:m)?|günü(?:m)?|randevum?)\b"
    ).unwrap();

    /// Turkish calendar date candidate regex (for embedding context disambiguation).
    pub static ref TURKISH_DATE_CANDIDATE_REGEX: Regex = Regex::new(
        r"(?i)\b([0-3]?[0-9])\s+(ocak|şubat|mart|nisan|mayıs|haziran|temmuz|ağustos|eylül|ekim|kasım|aralık)(?:\s+(19[2-9][0-9]|20[0-3][0-9]))?\b"
    ).unwrap();

    /// Numeric date candidate regex: `12.04.1985`, `12/04/1985`, `1985-04-12`.
    pub static ref NUMERIC_DATE_CANDIDATE_REGEX: Regex = Regex::new(
        r"\b(?:(0[1-9]|[12][0-9]|3[01])[./-](0[1-9]|1[0-2])[./-](19[2-9][0-9]|20[0-3][0-9])|(19[2-9][0-9]|20[0-3][0-9])-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01]))\b"
    ).unwrap();

    /// Private URLs containing sensitive query tokens, auth sessions, or sensitive routes.
    pub static ref PRIVATE_URL_REGEX: Regex = Regex::new(
        r#"(?i)\bhttps?://[^\s<>"'{}|\\^`]+(?:[?&](?:token|k|key|oturum|session|dosya|auth|kod|secret)=[^\s<>"'{}|\\^`]+|/(?:dosya|paylas|u|ticket|kisisel|private|secure)/[^\s<>"'{}|\\^`]+)"#
    ).unwrap();

    /// Contextual private link trigger: `kişisel teslim linki https://...`, `özel link: https://...`
    pub static ref PRIVATE_URL_TRIGGER_REGEX: Regex = Regex::new(
        r#"(?i)\b(?:özel\s+(?:dosya\s+)?linki?|kişisel\s+(?:dosya\s+)?linki?|teslim\s+linki?|takip\s+linki?|belge\s+linki?|hesap\s+linki?|profil\s+linki?|özel\s+url)[:\s]+(https?://[^\s<>"'{}|\\^`]+)"#
    ).unwrap();

    /// MAC Hardware Address regex (e.g. `1F:FA:B2:B3:CF:1D` or `00-14-22-01-23-45`).
    pub static ref MAC_ADDRESS_REGEX: Regex = Regex::new(
        r"(?i)\b(?:mac(?:\s*adresi?)?[:\s]*)?((?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2})\b"
    ).unwrap();

    /// Cryptocurrency Wallet Address (BTC, ETH, TRON, etc.).
    pub static ref CRYPTO_WALLET_REGEX: Regex = Regex::new(
        r"\b(bc1[a-zA-HJ-NP-Z0-9]{25,59}|0x[a-fA-F0-9]{40}|T[A-Za-z1-9]{33})\b"
    ).unwrap();

    /// IMEI device number (15 digits).
    pub static ref IMEI_REGEX: Regex = Regex::new(
        r"(?i)\b(?:imei(?:\s*no(?:su)?)?[:\s]*)?([0-9]{15})\b"
    ).unwrap();

    /// Vehicle Registration / Ruhsat No (e.g. `ruhsat no FY-568391` or `EM-123456`).
    pub static ref RUHSAT_NO_REGEX: Regex = Regex::new(
        r"(?i)\b(?:ruhsat(?:\s*no(?:su)?)?)[:\s]*([A-Z]{2}[- ][0-9]{6})\b"
    ).unwrap();

    /// Contract / Policy / Reference numbers.
    pub static ref CONTRACT_NO_REGEX: Regex = Regex::new(
        r"(?i)\b(?:sözleşme(?:\s*no(?:su)?)?|sozlesme_no)[:\s]*([A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)+|[0-9]{5,10})\b"
    ).unwrap();

    pub static ref POLICY_NO_REGEX: Regex = Regex::new(
        r"(?i)\b(?:poliçe(?:\s*no(?:su)?)?|police_no)[:\s]*([A-Za-z0-9]{2,6}[-/][A-Za-z0-9]{4,12}|[A-Za-z0-9]{8,14})\b"
    ).unwrap();

    pub static ref REFERENCE_NO_REGEX: Regex = Regex::new(
        r"(?i)\b(?:referans(?:[ıiIİ])?(?:\s*olarak)?(?:\s*no(?:su)?)?)[:\s]*(REF-[0-9]{4}-[0-9]{3,5}|RF[0-9]{6,10}|[A-ZÇĞİÖŞÜ][a-zçğıöşü]+(?:\s+[A-ZÇĞİÖŞÜ][a-zçğıöşü]+)+)\b"
    ).unwrap();

    /// Vehicle Chassis / VIN number in context (e.g. `şasi no 6Z9741GSRLHSCNA73` or `şasi F7B6HT2ETFG9716N2`).
    pub static ref CHASSIS_NO_REGEX: Regex = Regex::new(
        r"(?i)\b(?:şasi(?:\s*no(?:su)?)?|sasi(?:\s*no(?:su)?)?|vin)[:\s]*([A-HJ-NPR-Z0-9]{17})\b"
    ).unwrap();

    /// Engine / Motor number in context (e.g. `motor N66D336569`).
    pub static ref MOTOR_NO_REGEX: Regex = Regex::new(
        r"(?i)\b(?:motor(?:\s*no(?:su)?)?)[:\s]*([A-Z0-9]{8,14})\b"
    ).unwrap();

    /// PIN code in context (e.g. `pin 690676`).
    pub static ref PIN_REGEX: Regex = Regex::new(
        r"(?i)\b(?:pin(?:(?:\s*['’]?\s*i)|\s+kodu)?|kullanıcı\s+pini?)[:\s]*([0-9]{4,8})\b"
    ).unwrap();

    /// Credit rating / Findeks score in context (e.g. `kredi notu 570`).
    pub static ref CREDIT_SCORE_REGEX: Regex = Regex::new(
        r"(?i)\b(?:kredi\s+notu(?:m|nu)?|findeks(?:\s*notu)?)[:\s]*([0-9]{3,4})\b"
    ).unwrap();

    /// Mother's maiden name in context (e.g. `annemin kızlık soyadı Karadeniz`).
    pub static ref MOTHER_MAIDEN_REGEX: Regex = Regex::new(
        r"(?i)\b(?:anne(?:min|m)?\s+kızlık\s+soyadı(?:m|nız|nın)?|anne\s+kizlik\s+soyadi)[:\s]*([A-ZÇĞİÖŞÜa-zçğıöşü]+)\b"
    ).unwrap();

    /// Mother / Father given names in context (e.g. `anne adı Sevsevil`, `baba adı Bedri`).
    pub static ref PARENT_NAMES_REGEX: Regex = Regex::new(
        r"(?i)\b(?:anne|annem(?:in)?|baba|babam(?:ın)?)\s+adı(?:m|nın|sı)?[:\s]*([A-ZÇĞİÖŞÜ][a-zçğıöşü]+)\b"
    ).unwrap();

    /// Birthplace in context (e.g. `doğum yerim Denizli`).
    pub static ref BIRTHPLACE_REGEX: Regex = Regex::new(
        r"(?i)\b(?:doğum\s+yer(?:i|im|iniz)?|dogum\s+yeri)[:\s]*([A-ZÇĞİÖŞÜ][a-zçğıöşü]+)\b"
    ).unwrap();

    /// Gender identity in context (e.g. `cinsiyeti kadın`, `cinsiyet: erkek`, `24 yaşında kadın`).
    pub static ref GENDER_REGEX: Regex = Regex::new(
        r"(?i)\b(?:c[iİıI]ns[iİıI]yet(?:[iİıI]|m)?|c[iİıI]ns[iİıI]yet[iİıI])[\s:]*(kad[ıiIİ]n|k[ıiIİ]z|erkek|bayan|d[iİıI][şsŞS][iİıI]|bay|er)\b|\b(?:ya[şsŞS][ıiIİ]nda|ya[şsŞS][ıiIİ]ndak[iİıI])\s+(kad[ıiIİ]n|erkek|k[ıiIİ]z|bayan)\b"
    ).unwrap();

    /// Nationality / Demonym in context (e.g. `uyruğu Türk`, `uyruğu Alman`).
    pub static ref NATIONALITY_REGEX: Regex = Regex::new(
        r"(?i)\b(?:uyruğu|uyruk|vatandaşlığı|vatandaşlık)[:\s]*([A-ZÇĞİÖŞÜa-zçğıöşü]+)\b"
    ).unwrap();

    /// Geolocation Coordinates (e.g. `41.6849, 31.2242`).
    pub static ref COORDINATES_REGEX: Regex = Regex::new(
        r"\b([1-4][0-9]\.[0-9]{4,7})[,\s]+([2-4][0-9]\.[0-9]{4,7})\b"
    ).unwrap();

    /// Digital Signature / E-Signature digest/hash/cert serial.
    pub static ref SIGNATURE_REGEX: Regex = Regex::new(
        r"(?i)\b(?:dijital\s+imza(?:\s*özeti)?\s+[A-Za-z0-9]+|imza\s*hash\s+[a-f0-9]+|e-imza(?:\s*sertifika(?:\s*seri\s*no)?)?[:\s]*[0-9]+)\b"
    ).unwrap();

    /// Criminal / Judicial record (e.g. `adli para cezası kaydı`, `KABAHAT KAYDI MEVCUT`).
    pub static ref CRIMINAL_RECORD_REGEX: Regex = Regex::new(
        r"(?i)\b(?:adli\s+para\s+cezası\s+kaydı|kabahat\s+kaydı(?:\s*mevcut)?|denetimli\s+serbestlik\s+kaydı|icra\s+takibi\s+kaydı|adli\s+sicil\s+kaydı|sabıka\s+kaydı)\b"
    ).unwrap();

    /// Health Data under KVKK Article 6.
    pub static ref HEALTH_RECORD_REGEX: Regex = Regex::new(
        r"(?i)\b(?:sağlık\s+raporu(?:m)?(?:nda)?|rahatsızlığı|kronik\s+rahatsızlık|engel\s+durumu)[:\s]*([A-ZÇĞİÖŞÜa-zçğıöşü\s]{4,35})\b"
    ).unwrap();

    /// Disability status under KVKK Article 6.
    pub static ref DISABILITY_REGEX: Regex = Regex::new(
        r"(?i)\b(?:engel\s*durumu|engell[iİıI]l[iİıI]k(?:\s*oran[ıiIİ])?|engell[iİıI](?:\s*raporu)?|[öoÖO]z[üuÜU]r\s*durumu)[\s:]*([a-zA-ZçğıöşüÇĞİÖŞÜ0-9%\s]+?)(?:\s+(?:olarak|sebeb[iİıI]yle|neden[iİıI]yle|var)|[,;.\n]|$)"
    ).unwrap();

    /// Trade Union Membership under KVKK Article 6.
    pub static ref TRADE_UNION_REGEX: Regex = Regex::new(
        r"(?i)\b(?:sendika(?:\s*bilgisi)?|sendikası)[:\s]*([A-ZÇĞİÖŞÜa-zçğıöşü\s-]+?(?:Sendikası|Sen|-İş)(?:\s+üyesi)?)\b|\b([A-ZÇĞİÖŞÜa-zçğıöşü\s-]+?(?:Sendikası|Sen|-İş)\s+üyesi)\b"
    ).unwrap();

    /// Username in context.
    pub static ref USERNAME_REGEX: Regex = Regex::new(
        r"(?i)\b(?:kullan[ıiIİ]c[ıiIİ]\s+ad[ıiIİ](?:m|n[ıiIİ]z|s[ıiIİ])?|kullan[iİıI]c[iİıI]\s*ad[iİıI]|username)[:\s]*([A-Za-z0-9_.-]{4,24})\b|\b([A-Za-z0-9_.-]{4,24})'(?:[iİıI]n|[ıiIİ]n|[üuÜU]n|[uU]n)\s+oturumlar[ıiIİ]\b"
    ).unwrap();

    /// Biometric Data under KVKK Article 6.
    pub static ref BIOMETRIC_REGEX: Regex = Regex::new(
        r"(?i)\b(?:avuç\s+içi\s+damar\s+izi|avuc\s+ici\s+damar\s+izi|parmak\s*izi|yüz\s*taraması|yuz\s*taramasi|iris\s*taraması|iris\s*taramasi|retina\s*taraması|retina\s*taramasi|biyometrik\s*(?:veri|imza|kayıt|giriş))\b"
    ).unwrap();

    /// Family / Marital status.
    pub static ref FAMILY_STATUS_REGEX: Regex = Regex::new(
        r"(?i)\b(?:a[iİıI]le\s+durumu|meden[iİıI]\s+hal[iİıI])[:\s]*([a-zçğıöşüA-ZÇĞİÖŞÜ0-9\s]+?)(?:[,;.\n]|$)"
    ).unwrap();

    /// Ethnic Origin under KVKK Article 6.
    pub static ref ETHNIC_ORIGIN_REGEX: Regex = Regex::new(
        r"(?i)\b(?:etn[iİıI]k\s+k[öoÖO]ken(?:[iİıI])?|etn[iİıI]k\s+k[öoÖO]ken)[\s:]*(?:alan[iİıI]\s*)?([A-ZÇĞİÖŞÜa-zçğıöşü]+)\b"
    ).unwrap();

    /// Workplace / Institution.
    pub static ref WORKPLACE_REGEX: Regex = Regex::new(
        r"(?i)\b(?:[iİıI][şsŞS]yer[iİıI]|[iİıI]syer[iİıI]|kurum|[şsŞS][iİıI]rket)[:\s]*([A-ZÇĞİÖŞÜ][A-Za-zÇĞİÖŞÜçğıöşü0-9\s.&-]+?(?:Hastan[iİıI]s[iİıI]|Beled[iİıI]yes[iİıI]|M[üuÜU]d[üuÜU]rl[üuÜU][ğgĞG][üuÜU]|Bakanl[ıiIİ][ğgĞG][ıiIİ]|A\.Ş\.|AŞ|Ltd\.\s*[ŞS]t[iİıI]\.|[ÜU]n[iİıI]vers[iİıI]tes[iİıI]|Hold[iİıI]ng|Anon[iİıI]m\s*[ŞS][iİıI]rket[iİıI]))\b|\b([A-ZÇĞİÖŞÜ][A-Za-zÇĞİÖŞÜçğıöşü0-9\s&-]{2,50}?(?:A\.Ş\.|AŞ|Ltd\.\s*[ŞS]t[iİıI]\.|[ŞS][iİıI]rket[iİıI]|[ÜU]n[iİıI]vers[iİıI]tes[iİıI]|Beled[iİıI]yes[iİıI]))\s+b[üuÜU]nyes[iİıI]nde\b"
    ).unwrap();

    /// Hardware / Device ID in context (e.g. `cihaz id DEV-99381290`).
    pub static ref DEVICE_ID_REGEX: Regex = Regex::new(
        r"(?i)\b(?:c[iİıI]haz\s+id(?:s[iİıI])?|dev[iİıI]ce\s+id|c[iİıI]haz\s+(?:e[şsŞS]le[şsŞS]t[iİıI]rme\s+)?kodu)[:\s]*([A-Za-z0-9_-]{8,36})\b"
    ).unwrap();

    /// Employee Registry / Sicil No in context (e.g. `sicil SC3907`).
    pub static ref SICIL_NO_REGEX: Regex = Regex::new(
        r"(?i)\b(?:sicil(?:\s*no(?:su)?)?)[:\s]*([A-Za-z0-9]{4,10})\b"
    ).unwrap();

    /// Vehicle License Plate with explicit trigger keyword.
    pub static ref PLATE_TRIGGER_REGEX: Regex = Regex::new(
        r"(?i)\b(?:plaka(?:\s*no(?:su)?)?)[:\s]*([0-8][0-9]\s*[A-ZÇĞİÖŞÜ]{1,3}\s*[0-9]{2,5})\b"
    ).unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phone_regex() {
        let text = "Bana 0532 123 45 67 veya +90 542 321 45 67 veya 0212 456 78 90 numarasından ulaşabilirsiniz.";
        let matches: Vec<_> = PHONE_REGEX.find_iter(text).map(|m| m.as_str()).collect();
        assert_eq!(matches.len(), 3);
    }

    #[test]
    fn test_email_regex() {
        let text = "İletişim: ahmet.yilmaz@example.com veya ali [at] domain [dot] com";
        assert!(EMAIL_REGEX.is_match(text));
        assert!(EMAIL_OBFUSCATED_REGEX.is_match(text));

        // Obfuscated email variations
        assert!(EMAIL_OBFUSCATED_REGEX.is_match("ali (at) sirket . com"));
        assert!(EMAIL_OBFUSCATED_REGEX.is_match("user@gmail [dot] com"));
        assert!(EMAIL_OBFUSCATED_REGEX.is_match("user@gmail (dot) com"));
        assert!(EMAIL_OBFUSCATED_REGEX.is_match("mehmet [at] sirket.com"));
        assert!(EMAIL_OBFUSCATED_REGEX.is_match("mehmet (at) sirket.com"));

        // File mentions must NOT match EMAIL_OBFUSCATED_REGEX or EMAIL_REGEX
        assert!(!EMAIL_OBFUSCATED_REGEX.is_match("update @README.md"));
        assert!(!EMAIL_OBFUSCATED_REGEX.is_match("and @handoff.md"));
        assert!(!EMAIL_OBFUSCATED_REGEX.is_match("see @file.ext"));
        assert!(!EMAIL_REGEX.is_match("update @README.md"));
        assert!(!EMAIL_REGEX.is_match("and @handoff.md"));
        assert!(!EMAIL_REGEX.is_match("see @file.ext"));
    }

    #[test]
    fn test_ip_and_port() {
        let text = "Sunucu 192.168.1.100 üzerinde 8080 portu ile çalışıyor.";
        assert!(IPV4_REGEX.is_match(text));
        assert!(PORT_REGEX.is_match(text));
    }

    #[test]
    fn test_cvv_and_skt() {
        let text = "Kart CVV: 789 ve SKT: 12/28";
        assert!(CVV_REGEX.is_match(text));
        assert!(CARD_EXPIRY_REGEX.is_match(text));
    }

    #[test]
    fn test_age_and_range() {
        let text1 = "Müşteri 42 yaşında olduğunu belirtti.";
        let text2 = "25 - 35 yaş aralığında adaylar aranıyor.";
        assert!(AGE_REGEX.is_match(text1));
        assert!(AGE_RANGE_REGEX.is_match(text2));
    }

    #[test]
    fn test_ethnic_regex() {
        let text = "ETNİK KÖKEN ALANI RUM YAZILMIŞ BU ALAN ZORUNLU MU.";
        let cap = ETHNIC_ORIGIN_REGEX.captures(text);
        assert!(
            cap.is_some(),
            "ETHNIC_ORIGIN_REGEX failed to match {:?}",
            text
        );
        assert_eq!(cap.unwrap().get(1).unwrap().as_str(), "RUM");
    }
}
