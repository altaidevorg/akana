//! The contract between the sparse and hybrid encoders and the indexes built with them.
//!
//! An index stores [`encoder_info`] when it is built and compares it with the running library's
//! when it is opened ([`compatibility`]):
//! - the same fingerprint: the library produces exactly the vectors the index holds;
//! - the same [`SPARSE_SPEC`] major and [`HYBRID_LAYOUT`]: vectors may differ slightly. Each
//!   release that changes them reports how many, and keeps a stale index's retrieval within
//!   0.005 MRR@10 of a rebuilt one (the bump policy below), so the index needs no rebuild;
//! - a different major or layout: term ids, term definitions or the vector layout changed, and
//!   old and new vectors do not match. Rebuild.
//!
//! Bump policy for a release:
//! - patch: the golden fixtures (`tests/data/embeddings/golden.jsonl`) and a 5,000-chunk court
//!   sweep are bit-identical; the fingerprint may still change (a resource's bytes changed with
//!   no effect on vectors);
//! - minor: at most 5 % of court hybrid vectors change, and a stale index's MRR@10 is within
//!   0.005 of a rebuilt one on RagTurk and Turkuaz;
//! - major: anything else, and any change to term ids, the hybrid layout, the tokenizer, the
//!   dense model, the lemma conventions, the stopwords or the compound rules.
//!
//! Spec 1 of the sparse encoder: text is split into sentences and words (the akana tokenizer);
//! each word is analysed and disambiguated in its sentence; its term is the reading's lemma
//! (lower-case, no circumflex), plus the parts of a compound (the lexicon's, and those listed
//! in `morphology::compound`, but not a compound that is a word of its own: anayasa), but for
//! a word the dictionary lists as a postposition or an adverb, whose term is fixed (that word,
//! or the verb an adverb is a form of: `sparse::frozen_form`); stopwords and function lemmas
//! (`sparse::STOP_LEMMAS`: a lemma whose items are all pronouns, conjunctions, postpositions,
//! determiners or question particles, and a listed few) are dropped; a term's id is the 32-bit
//! FNV-1a of its string ([`super::hash_term_32bit`]); its value is BM25-saturated term
//! frequency (k1 = 1.2) times its POS weight times its function-lemma weight. The hybrid vector packs the 2-bit dense embedding (64 bytes) with the
//! 16 heaviest sparse terms (24-bit id and an 8-bit weight each; ties go to the lower id).

use crate::fingerprint::Fingerprint;
use serde::Serialize;
use std::sync::LazyLock;

/// The sparse encoder's spec version (major, minor); see the module docs.
pub const SPARSE_SPEC: (u16, u16) = (1, 0);

/// The hybrid vector's byte layout (64-byte 2-bit dense half, 16 slots of 24-bit id + 8-bit
/// weight).
pub const HYBRID_LAYOUT: u16 = 1;

/// The bundled dense model: Model2Vec static embeddings, 256 dimensions, TurboQuant 2-bit.
pub const DENSE_MODEL: &str = "model2vec-tr-256-turboquant-2bit";

/// What an index records about the library that built it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct EncoderInfo {
    /// [`SPARSE_SPEC`] as `major.minor`.
    pub sparse_spec: String,
    pub hybrid_layout: u16,
    pub dense_model: &'static str,
    /// [`crate::morphology::ANALYSIS_VERSION`].
    pub analysis_version: &'static str,
    /// [`crate::morphology::analysis_fingerprint`].
    pub analysis_fingerprint: String,
    /// The encoders' fingerprint: every resource below, the constants that define terms and
    /// weights, and the vectors of a built-in probe set.
    pub fingerprint: String,
    /// Each embedded resource's fingerprint, by name.
    pub resources: Vec<(&'static str, String)>,
}

/// How an index built by another library relates to this one ([`compatibility`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compatibility {
    /// The same fingerprint: identical vectors.
    Identical,
    /// The same spec major and layout: usable without a rebuild.
    Compatible,
    /// Rebuild the index.
    Rebuild,
}

impl Compatibility {
    pub fn as_str(self) -> &'static str {
        match self {
            Compatibility::Identical => "identical",
            Compatibility::Compatible => "compatible",
            Compatibility::Rebuild => "rebuild",
        }
    }
}

/// Texts whose vectors the fingerprint covers, so that a change in code that the resources do
/// not show (a rule, a weight, a tie-break) still changes it. They span court and encyclopedic
/// text, apostrophes, numerals, abbreviations, capitals, compounds and a few unusual inputs.
const PROBE_TEXTS: &[&str] = &[
    "Davacı vekili, müvekkilinin işe iadesine karar verilmesini talep etmiştir.",
    "Mahkemece davanın kabulüne karar verilmiş, hüküm davalı vekili tarafından temyiz edilmiştir.",
    "Dosyadaki bilgi ve belgelere göre, başvurucunun adil yargılanma hakkı ihlal edilmiştir.",
    "Anayasa Mahkemesi, 6216 sayılı Kanun'un 45. maddesi uyarınca başvuruyu inceledi.",
    "Hukuk Muhakemeleri Kanunu'nun 353. maddesi gereğince istinaf başvurusunun esastan reddine.",
    "İİK'nın 68. maddesinde öngörülen belgelerden hiçbiri sunulmamıştır.",
    "Sanığın uyuşturucu madde bulundurmaktan cezalandırılmasına oybirliğiyle karar verildi.",
    "Çoğunluk görüşüne katılmıyoruz.",
    "Bilirkişi raporunda kusur oranları %75 ve %25 olarak belirlenmiştir.",
    "Dava 12.03.2019 tarihinde açılmış, 1.250.000,00 TL tazminat istenmiştir.",
    "TBMM'de kabul edilen kanun 1924'te Resmî Gazete'de yayımlandı.",
    "Av. Ahmet Yılmaz ile Tic. ve San. Ltd. Şti. arasındaki sözleşme feshedildi.",
    "Sanık hakkında TCK.nun 53. maddesi ve H.U.M.K.nun 427. maddesi uygulandı.",
    "Ankara Büyükşehir Belediyesi, toplu taşıma ücretlerini yeniden düzenledi.",
    "İstanbul, Türkiye'nin en kalabalık şehridir ve Boğaziçi'nin iki yakasında kuruludur.",
    "Osmanlı İmparatorluğu 1299 yılında kurulmuş ve altı yüzyıl boyunca hüküm sürmüştür.",
    "Fotosentez, bitkilerin güneş ışığını kimyasal enerjiye dönüştürdüğü süreçtir.",
    "Mustafa Kemal Atatürk, Cumhuriyet'in ilk cumhurbaşkanı olarak seçildi.",
    "Kahverengi ayakkabılarını giyip bilgisayarın başına oturdu.",
    "Cumhurbaşkanlığı Kararnamesi ile zeytinyağı ihracatına yeni düzenleme getirildi.",
    "Denizaltı, Karadeniz'in derin sularında üç gün boyunca seyretti.",
    "Bana bir kitap ver.",
    "Sözümü anla!",
    "Siyasi partiler ister iktidarda ister muhalefette olsunlar, kanuna uymak zorundadır.",
    "Yarın sabah erkenden yola çıkacağız, hava çok güzel olacakmış.",
    "Napıyorsun ya, bi dakka bekle!",
    "Kitapları masanın üstüne bıraktım ama kimse okumadı.",
    "Öğrencilerin büyük çoğunluğu sınavı başarıyla geçti.",
    "Çocuklar bahçede top oynarken yağmur başladı.",
    "Bu kararın, 2709 sayılı Türkiye Cumhuriyeti Anayasası'nın 13. maddesine aykırı olduğu açıktır.",
    "Hükmün açıklanmasının geri bırakılmasına ilişkin karar kesinleşmiştir.",
    "Davalı idarenin işleminin iptaline, yargılama giderlerinin davalı üzerinde bırakılmasına.",
    "Kaynaklanan zararın tazmini için ihtarname gönderilmiştir.",
    "ABD'nin ve AB'nin yaptırım kararları uluslararası ticareti etkiledi.",
    "COVID-19 salgını sırasında uzaktan eğitime geçildi.",
    "Prof. Dr. Ayşe Kaya, XIX. yüzyıl edebiyatı üzerine bir konferans verdi.",
    "Saat 17.00'de toplanan kurul 3'er kişilik komisyonlar oluşturdu.",
    "Türkiyede ve istanbulda yaşayanların sayısı arttı.",
    "KANUN HÜKMÜNDE KARARNAME İLE DÜZENLENEN HUSUSLAR",
    "ı i I İ ç Ç ğ Ğ ö Ö ş Ş ü Ü â î û",
    "Kur'an-ı Kerim'in meali yayımlandı.",
    "Ve, veya, ile, ama, fakat, ancak, çünkü, eğer, ise.",
    "1, 2, 3, 4, 5, 10, 100, 1000, 1.000.000, 3,14",
    "e-posta adresine gönderilen belge @akana ve #nlp etiketleriyle paylaşıldı.",
    "\u{feff}Metnin başında bayt sırası işareti var.",
    "Sıfır\u{200b}genişlikli boşluk ve birleştirici\u{200d}karakterler.",
    "",
    "   ",
    ".",
    "a",
    "Ahmet'in, Ayşe'nin ve Mehmet'in kitapları.",
    "Yargıtay 9. Hukuk Dairesinin bozma ilamına uyularak yeniden hüküm kuruldu.",
    "Danıştay İdari Dava Daireleri Kurulu, ısrar kararını onadı.",
    "Kişisel Verilerin Korunması Kanunu kapsamında açık rıza alınmamıştır.",
    "Tapu kaydında taşınmazın malikinin davalı olduğu görülmektedir.",
    "Kira bedelinin tespiti davasında hak ve nesafet ilkesi uygulanır.",
    "Başvuru, açıkça dayanaktan yoksun olduğundan kabul edilemez bulunmuştur.",
    "Kediler ve köpekler en yaygın evcil hayvanlardır.",
    "Dünyanın en yüksek dağı Everest'tir.",
    "Ekonomik büyüme oranı geçen yılın aynı dönemine göre yüzde 4,5 arttı.",
    "Futbol takımı şampiyonluk maçını 2-1 kazandı.",
    "Müzisyen yeni albümünü dijital platformlarda yayımladı.",
    "Tarım ve Orman Bakanlığı kuraklıkla mücadele planını açıkladı.",
    "Gelmeyecekmişsiniz gibi davranmayın, hepinizi bekliyoruz.",
    "Avrupalılaştıramadıklarımızdanmışsınızcasına konuşuyorsunuz.",
];

/// The probe set's vectors, sparse and hybrid, as bytes.
fn probe_bytes() -> Vec<u8> {
    let sparse = super::MorphologicalSparseEncoder::new();
    let hybrid = super::TurkishHybridEmbeddings::new();
    let mut out = Vec::new();
    for text in PROBE_TEXTS {
        let v = sparse.encode_sparse(text);
        for ((id, value), term) in v.indices.iter().zip(&v.values).zip(&v.terms) {
            out.extend_from_slice(&id.to_le_bytes());
            out.extend_from_slice(&value.to_bits().to_le_bytes());
            out.extend_from_slice(term.as_bytes());
            out.push(0);
        }
        out.push(0xff);
        out.extend_from_slice(&hybrid.embed_hybrid(text));
    }
    out
}

static ENCODER_INFO: LazyLock<EncoderInfo> = LazyLock::new(|| {
    let resources: Vec<(&'static str, String)> = vec![
        (
            "tokenizer.json",
            crate::fingerprint::of(super::TOKENIZER_JSON),
        ),
        (
            "turboquant_weights.npz",
            crate::fingerprint::of(super::WEIGHTS_NPZ),
        ),
        (
            "function_lemmas.tsv",
            crate::fingerprint::of(super::sparse::FUNCTION_LEMMAS_TSV.as_bytes()),
        ),
        (
            "analysis",
            crate::morphology::analysis_fingerprint().to_string(),
        ),
    ];
    let mut fp = Fingerprint::new().section(
        "spec",
        format!(
            "sparse {}.{} hybrid {HYBRID_LAYOUT} dense {DENSE_MODEL}",
            SPARSE_SPEC.0, SPARSE_SPEC.1
        )
        .as_bytes(),
    );
    for (name, hash) in &resources {
        fp = fp.section(name, hash.as_bytes());
    }
    fp = fp
        .section("sparse", super::sparse::fingerprint_constants().as_bytes())
        .section(
            "stopwords",
            crate::morphology::stopwords_fingerprint_constants().as_bytes(),
        )
        .section(
            "compounds",
            crate::morphology::compound_fingerprint_constants().as_bytes(),
        )
        .section("probe", &probe_bytes());
    EncoderInfo {
        sparse_spec: format!("{}.{}", SPARSE_SPEC.0, SPARSE_SPEC.1),
        hybrid_layout: HYBRID_LAYOUT,
        dense_model: DENSE_MODEL,
        analysis_version: crate::morphology::ANALYSIS_VERSION,
        analysis_fingerprint: crate::morphology::analysis_fingerprint().to_string(),
        fingerprint: fp.hex(),
        resources,
    }
});

/// What this library's encoders are: spec, layout, dense model, analysis version and the
/// fingerprints. Computed once per process (it hashes the embedded resources and encodes the
/// probe set: a fraction of a second).
pub fn encoder_info() -> &'static EncoderInfo {
    &ENCODER_INFO
}

/// How an index built by a library with `stored` info relates to this one.
pub fn compatibility(
    stored_sparse_spec: &str,
    stored_layout: u16,
    stored_fingerprint: &str,
) -> Compatibility {
    let info = encoder_info();
    // The spec and layout first: a record whose fingerprint matches but whose spec does not is
    // not this library's (a hand-edited or corrupt record), and must not pass as identical.
    let major = |spec: &str| spec.split('.').next().and_then(|m| m.parse::<u16>().ok());
    if stored_layout != info.hybrid_layout || major(stored_sparse_spec) != Some(SPARSE_SPEC.0) {
        Compatibility::Rebuild
    } else if stored_fingerprint == info.fingerprint {
        Compatibility::Identical
    } else {
        Compatibility::Compatible
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoder_info_is_complete_and_stable() {
        let info = encoder_info();
        assert_eq!(
            info.sparse_spec,
            format!("{}.{}", SPARSE_SPEC.0, SPARSE_SPEC.1)
        );
        assert_eq!(info.fingerprint.len(), 32);
        assert_eq!(info.resources.len(), 4);
        assert_eq!(probe_bytes(), probe_bytes());
        assert_eq!(
            compatibility(&info.sparse_spec, info.hybrid_layout, &info.fingerprint),
            Compatibility::Identical
        );
        assert_eq!(
            compatibility(&info.sparse_spec, info.hybrid_layout, "0"),
            Compatibility::Compatible
        );
        assert_eq!(
            compatibility("0.9", info.hybrid_layout, "0"),
            Compatibility::Rebuild
        );
        assert_eq!(
            compatibility(&info.sparse_spec, info.hybrid_layout + 1, "0"),
            Compatibility::Rebuild
        );
    }
}
