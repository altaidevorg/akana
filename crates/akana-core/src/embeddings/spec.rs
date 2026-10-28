//! The contract between the encoders and the indexes built with them.
//!
//! An index stores [`encoder_info`] when it is built and compares it with the running library's
//! when it is opened ([`compatibility_of`]), for what it holds ([`IndexKind`]):
//! - the same fingerprint of that kind: the library was built from the same inputs (resources,
//!   the source of the analysis and encoder path, `Cargo.lock`, the Unicode version) and
//!   produces exactly the vectors the index holds;
//! - sparse or hybrid vectors with the same [`SPARSE_SPEC`] major (and [`HYBRID_LAYOUT`], and
//!   dense model and weights, for hybrids): vectors may differ slightly. Each release that
//!   changes them reports how many, and keeps a stale index's retrieval within 0.005 MRR@10 of
//!   a rebuilt one (the bump policy below), so the index needs no rebuild;
//! - anything else, a record that is not one [`encoder_info`] wrote included: term ids, term
//!   definitions, the dense model or the vector layout changed, and old and new vectors do not
//!   match. Rebuild. A dense index is identical or rebuilt: its vectors are the model's.
//!
//! Bump policy for a release, measured with `scripts/encoder_churn.py` (`--declared` exits
//! non-zero above the declared class):
//! - patch: the golden fixtures (`tests/data/embeddings/golden.jsonl`) and the court, RagTurk
//!   and Turkuaz documents' vectors are bit-identical; the fingerprint may still change (code
//!   or a resource changed with no effect on vectors);
//! - minor: at most 5 % of the documents' sparse vectors and of their hybrid vectors change in
//!   each of court, RagTurk and Turkuaz, and a stale index's MRR@10 (hybrid, and sparse by dot
//!   product) is within 0.005 of a rebuilt one on RagTurk and Turkuaz;
//! - major: anything else, and any change to term ids, the hybrid layout, the tokenizer, the
//!   dense model, the lemma conventions, the stopwords or the compound rules.
//!
//! CI holds a release to its spec: more than 5 % of the golden texts with other vectors than
//! the release that froze the spec (`tests/data/embeddings/golden-1.0.jsonl`, never rewritten)
//! fails unless [`SPARSE_SPEC`]'s major changed.
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
//! frequency (k1 = 1.2) times its POS weight times its function-lemma weight.
//!
//! The hybrid vector packs the 2-bit dense embedding (64 bytes) with the 16 heaviest sparse
//! terms, ties going to the lower 32-bit id. Each is a 24-bit id ([`super::term_id_24bit`]: the
//! 32-bit id xor its top byte, masked to 24 bits, 0 read as 1) and an 8-bit weight (the value
//! / 4 · 255 in f32, truncated and clamped to [1, 255]), in ascending 24-bit id; of two of the
//! 16 whose 24-bit ids collide the lighter is dropped, and its slot stays empty (zero bytes).

use crate::fingerprint::Fingerprint;
use serde::Serialize;
use std::sync::LazyLock;

/// The sparse encoder's spec version (major, minor); see the module docs. 1.0 is frozen as
/// released with akana 0.9.0: a later change that moves vectors follows the bump policy
/// above, and the golden fixtures record each one.
pub const SPARSE_SPEC: (u16, u16) = (1, 0);

/// The hybrid vector's byte layout (64-byte 2-bit dense half, 16 slots of 24-bit id + 8-bit
/// weight).
pub const HYBRID_LAYOUT: u16 = 1;

/// The bundled dense model: Model2Vec static embeddings, 256 dimensions, TurboQuant 2-bit.
pub const DENSE_MODEL: &str = "model2vec-tr-256-turboquant-2bit";

/// What an index records about the library that built it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct EncoderInfo {
    /// The akana release, for reports (compatibility reads the fingerprints).
    pub akana_version: &'static str,
    /// The Unicode version of the Rust standard library the library was built with, whose
    /// tables lower-case and classify characters.
    pub unicode_version: String,
    /// [`SPARSE_SPEC`] as `major.minor`.
    pub sparse_spec: String,
    pub hybrid_layout: u16,
    pub dense_model: &'static str,
    /// [`crate::morphology::ANALYSIS_VERSION`].
    pub analysis_version: &'static str,
    /// [`crate::morphology::analysis_fingerprint`].
    pub analysis_fingerprint: String,
    /// What sparse vectors depend on: the sparse spec, the analyses, the function-lemma table,
    /// the constants that define terms and weights, the source of the encoder path,
    /// `Cargo.lock`, the Unicode version, and the sparse vectors of a built-in probe set.
    pub sparse_fingerprint: String,
    /// What dense vectors depend on: the dense model, its tokenizer and weights, the Unicode
    /// version, and the dense vectors (float and packed) of the probe set.
    pub dense_fingerprint: String,
    /// Both, the hybrid layout and the probe set's hybrid vectors.
    pub fingerprint: String,
    /// Each embedded resource's fingerprint, by name.
    pub resources: Vec<(&'static str, String)>,
}

/// How an index built by another library relates to this one ([`compatibility_of`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compatibility {
    /// The same fingerprint: identical vectors.
    Identical,
    /// The same spec major (and layout and dense model): usable without a rebuild.
    Compatible,
    /// Rebuild the index.
    Rebuild,
}

/// What an index holds, which decides the parts of [`EncoderInfo`] it depends on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexKind {
    /// Sparse vectors or their terms (`sparse_embed`, a BM25 index of the terms).
    Sparse,
    /// Dense vectors (`embed`, the packed 2-bit and 1-bit vectors).
    Dense,
    /// Hybrid vectors (`embed_hybrid`).
    Hybrid,
}

impl IndexKind {
    /// `sparse`, `dense` or `hybrid`.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "sparse" => Some(IndexKind::Sparse),
            "dense" => Some(IndexKind::Dense),
            "hybrid" => Some(IndexKind::Hybrid),
            _ => None,
        }
    }
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

/// The probe set's sparse vectors (ids, value bits, terms) as bytes.
fn sparse_probe_bytes() -> Vec<u8> {
    let sparse = super::MorphologicalSparseEncoder::new();
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
    }
    out
}

/// The probe set's dense vectors, float and packed, as bytes.
fn dense_probe_bytes() -> Vec<u8> {
    let dense = super::TurkishEmbeddings::new();
    let mut out = Vec::new();
    for text in PROBE_TEXTS {
        for x in dense.embed(text) {
            out.extend_from_slice(&x.to_bits().to_le_bytes());
        }
        out.extend_from_slice(&dense.embed_packed_2bit(text));
        out.extend_from_slice(&dense.embed_packed_1bit(text));
    }
    out
}

/// The probe set's hybrid vectors as bytes.
fn hybrid_probe_bytes() -> Vec<u8> {
    let hybrid = super::TurkishHybridEmbeddings::new();
    PROBE_TEXTS
        .iter()
        .flat_map(|text| hybrid.embed_hybrid(text))
        .collect()
}

/// The Unicode version of the standard library this was built with (`16.0.0`).
fn unicode_version() -> String {
    let (major, minor, update) = char::UNICODE_VERSION;
    format!("{major}.{minor}.{update}")
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
            "stop_lemmas.txt",
            crate::fingerprint::of(super::sparse::STOP_LEMMAS_TXT.as_bytes()),
        ),
        (
            "lemma_weights.tsv",
            crate::fingerprint::of(super::sparse::LEMMA_WEIGHTS_TSV.as_bytes()),
        ),
        (
            "analysis",
            crate::morphology::analysis_fingerprint().to_string(),
        ),
    ];
    let resource = |name: &str| {
        resources
            .iter()
            .find(|(n, _)| *n == name)
            .map_or(&[][..], |(_, hash)| hash.as_bytes())
    };
    let unicode = unicode_version();
    let sparse = Fingerprint::new()
        .section(
            "spec",
            format!("sparse {}.{}", SPARSE_SPEC.0, SPARSE_SPEC.1).as_bytes(),
        )
        .section("function_lemmas.tsv", resource("function_lemmas.tsv"))
        .section("stop_lemmas.txt", resource("stop_lemmas.txt"))
        .section("lemma_weights.tsv", resource("lemma_weights.tsv"))
        .section("analysis", resource("analysis"))
        .section("sparse", super::sparse::fingerprint_constants().as_bytes())
        .section(
            "stopwords",
            crate::morphology::stopwords_fingerprint_constants().as_bytes(),
        )
        .section(
            "compounds",
            crate::morphology::compound_fingerprint_constants().as_bytes(),
        )
        .section("sources", env!("AKANA_ENCODER_SOURCES").as_bytes())
        .section("Cargo.lock", env!("AKANA_CARGO_LOCK").as_bytes())
        .section("unicode", unicode.as_bytes())
        .section("probe", &sparse_probe_bytes())
        .hex();
    let dense = Fingerprint::new()
        .section("model", DENSE_MODEL.as_bytes())
        .section("tokenizer.json", resource("tokenizer.json"))
        .section("turboquant_weights.npz", resource("turboquant_weights.npz"))
        .section("unicode", unicode.as_bytes())
        .section("probe", &dense_probe_bytes())
        .hex();
    let fingerprint = Fingerprint::new()
        .section("layout", &HYBRID_LAYOUT.to_le_bytes())
        .section("sparse", sparse.as_bytes())
        .section("dense", dense.as_bytes())
        .section("probe", &hybrid_probe_bytes())
        .hex();
    EncoderInfo {
        akana_version: env!("CARGO_PKG_VERSION"),
        unicode_version: unicode,
        sparse_spec: format!("{}.{}", SPARSE_SPEC.0, SPARSE_SPEC.1),
        hybrid_layout: HYBRID_LAYOUT,
        dense_model: DENSE_MODEL,
        analysis_version: crate::morphology::ANALYSIS_VERSION,
        analysis_fingerprint: crate::morphology::analysis_fingerprint().to_string(),
        sparse_fingerprint: sparse,
        dense_fingerprint: dense,
        fingerprint,
        resources,
    }
});

/// What this library's encoders are: release, Unicode version, spec, layout, dense model,
/// analysis version and the fingerprints. Computed once per process (it hashes the embedded
/// resources and encodes the probe set: a fraction of a second).
pub fn encoder_info() -> &'static EncoderInfo {
    &ENCODER_INFO
}

/// How an index of `kind` built by a library whose [`encoder_info`] was `stored` (as JSON: the
/// object `encoder_info` serialises to, as an index stores it) relates to this one. A record
/// that is not such an object, or lacks or mistypes a field the kind depends on, needs a
/// rebuild: nothing says what built the index.
///
/// - Sparse: rebuild unless the sparse spec major is this one's; identical with the same
///   `sparse_fingerprint`; else compatible.
/// - Dense: identical with the same `dense_model` and `dense_fingerprint`; else rebuild.
/// - Hybrid: rebuild unless the layout, the sparse spec major and the dense part are this
///   one's; identical with the same `fingerprint`; else compatible.
pub fn compatibility_of(kind: IndexKind, stored: &serde_json::Value) -> Compatibility {
    let info = encoder_info();
    let text = |key: &str| stored.get(key).and_then(serde_json::Value::as_str);
    let fingerprint =
        |key: &str| text(key).filter(|f| f.len() == 32 && f.bytes().all(|b| b.is_ascii_hexdigit()));
    let spec_major = text("sparse_spec").and_then(|spec| {
        let (major, minor) = spec.split_once('.')?;
        minor.parse::<u16>().ok()?;
        major.parse::<u16>().ok()
    });
    let layout = stored
        .get("hybrid_layout")
        .and_then(serde_json::Value::as_u64);
    let sparse_same_major = spec_major == Some(SPARSE_SPEC.0);
    let dense_same = text("dense_model") == Some(info.dense_model)
        && fingerprint("dense_fingerprint") == Some(info.dense_fingerprint.as_str());
    match kind {
        IndexKind::Sparse => match fingerprint("sparse_fingerprint") {
            Some(f) if sparse_same_major && f == info.sparse_fingerprint => {
                Compatibility::Identical
            }
            Some(_) if sparse_same_major => Compatibility::Compatible,
            _ => Compatibility::Rebuild,
        },
        IndexKind::Dense if dense_same => Compatibility::Identical,
        IndexKind::Dense => Compatibility::Rebuild,
        IndexKind::Hybrid => {
            let parts_same = sparse_same_major
                && dense_same
                && layout == Some(u64::from(info.hybrid_layout))
                && fingerprint("sparse_fingerprint").is_some();
            match fingerprint("fingerprint") {
                Some(f) if parts_same && f == info.fingerprint => Compatibility::Identical,
                Some(_) if parts_same => Compatibility::Compatible,
                _ => Compatibility::Rebuild,
            }
        }
    }
}

/// [`encoder_info`] as the JSON object an index stores.
pub fn encoder_info_json() -> serde_json::Value {
    let info = encoder_info();
    let resources: serde_json::Map<String, serde_json::Value> = info
        .resources
        .iter()
        .map(|(name, hash)| (name.to_string(), serde_json::Value::from(hash.as_str())))
        .collect();
    serde_json::json!({
        "akana_version": info.akana_version,
        "unicode_version": info.unicode_version,
        "sparse_spec": info.sparse_spec,
        "hybrid_layout": info.hybrid_layout,
        "dense_model": info.dense_model,
        "analysis_version": info.analysis_version,
        "analysis_fingerprint": info.analysis_fingerprint,
        "sparse_fingerprint": info.sparse_fingerprint,
        "dense_fingerprint": info.dense_fingerprint,
        "fingerprint": info.fingerprint,
        "resources": resources,
    })
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
        for f in [
            &info.fingerprint,
            &info.sparse_fingerprint,
            &info.dense_fingerprint,
        ] {
            assert_eq!(f.len(), 32);
        }
        assert_eq!(info.resources.len(), 6);
        assert_eq!(sparse_probe_bytes(), sparse_probe_bytes());
        assert_eq!(dense_probe_bytes(), dense_probe_bytes());
    }

    /// Review 3 MSTAB3-01, MSTAB3-11, BIND3-01: each kind of index is judged by what its
    /// vectors depend on, and a record that is not one `encoder_info` wrote needs a rebuild.
    #[test]
    fn compatibility_is_judged_per_kind_of_index() {
        use Compatibility::{Compatible, Identical, Rebuild};
        use IndexKind::{Dense, Hybrid, Sparse};
        let stored = encoder_info_json();
        let with = |key: &str, value: serde_json::Value| {
            let mut record = stored.clone();
            record[key] = value;
            record
        };
        let check = |record: &serde_json::Value| {
            [Sparse, Dense, Hybrid].map(|kind| compatibility_of(kind, record))
        };
        let other = serde_json::json!("0123456789abcdef0123456789abcdef");
        assert_eq!(check(&stored), [Identical, Identical, Identical]);
        // A sparse change within the spec: sparse and hybrid indexes stay, dense ones are
        // untouched.
        let sparse_moved = with("sparse_fingerprint", other.clone());
        let sparse_moved = {
            let mut r = sparse_moved;
            r["fingerprint"] = other.clone();
            r
        };
        assert_eq!(check(&sparse_moved), [Compatible, Identical, Compatible]);
        // Another dense model or other weights: dense and hybrid indexes are rebuilt, sparse
        // ones are untouched (a release that swaps the dense model keeps the sparse spec).
        for record in [
            with("dense_model", serde_json::json!("other-model-512")),
            with("dense_fingerprint", other.clone()),
        ] {
            assert_eq!(check(&record), [Identical, Rebuild, Rebuild]);
        }
        // Another sparse spec major: sparse and hybrid indexes are rebuilt.
        assert_eq!(
            check(&with("sparse_spec", serde_json::json!("2.0"))),
            [Rebuild, Identical, Rebuild]
        );
        assert_eq!(
            check(&with("hybrid_layout", serde_json::json!(2))),
            [Identical, Identical, Rebuild]
        );
        // Malformed records: nothing says what built the index.
        for record in [
            serde_json::json!({}),
            serde_json::json!("1.0"),
            serde_json::Value::Null,
            with("sparse_spec", serde_json::json!(1.0)),
            with("sparse_spec", serde_json::json!("1")),
            with("hybrid_layout", serde_json::Value::Null),
            with("hybrid_layout", serde_json::json!(-1)),
            with("hybrid_layout", serde_json::json!(1.0)),
            with("fingerprint", serde_json::json!("")),
        ] {
            assert_eq!(compatibility_of(Hybrid, &record), Rebuild, "{record}");
        }
        let mut no_sparse = stored.clone();
        no_sparse
            .as_object_mut()
            .unwrap()
            .remove("sparse_fingerprint");
        assert_eq!(check(&no_sparse), [Rebuild, Identical, Rebuild]);
    }
}
