//! Cliché, bureaucratic connector, inflated hedges, and rhetorical translationese pattern matcher for Turkish text.
//! Accelerated with StringZilla SIMD substring searching.

use regex::Regex;
use lazy_static::lazy_static;
use stringzilla::StringZilla;
use crate::phonology::to_turkish_lower;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClicheCategory {
    BureaucraticConnector,
    ConclusionHedge,
    RhetoricalCalque,
    InflatedHedging,
    PassiveInflation,
    CliticAnomaly,
    SemicolonConjunction,
    IndefiniteArticleInflation,
    TricolonParallelList,
    HypophoraQuestion,
    HumanAuthenticityMarker,
    PunctuationAnomaly,
    PredicateRepetition,
    RhythmMonotony,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatchedPattern {
    pub category: ClicheCategory,
    pub pattern_name: &'static str,
    pub text: String,
    pub start: usize,
    pub end: usize,
    pub explanation: &'static str,
    pub suggestion: &'static str,
}

struct LiteralCliche {
    pattern_name: &'static str,
    phrase: &'static str,
    category: ClicheCategory,
    explanation: &'static str,
    suggestion: &'static str,
}

lazy_static! {
    // 1. Literal / Exact Clichés (Accelerated via StringZilla SIMD)
    static ref LITERAL_CLICHES: Vec<LiteralCliche> = vec![
        // Bureaucratic Connectors
        LiteralCliche {
            pattern_name: "bu_baglamda",
            phrase: "bu bağlamda",
            category: ClicheCategory::BureaucraticConnector,
            explanation: "Bürokratik ve yapay geçiş kalıbı; AI metinlerinde aşırı tekrarlanır.",
            suggestion: "Cümleyi doğrudan başlatın veya 'Bu yüzden', 'Dolayısıyla' gibi daha doğal bağlaçlar kullanın.",
        },
        LiteralCliche {
            pattern_name: "bu_dogrultuda",
            phrase: "bu doğrultuda",
            category: ClicheCategory::BureaucraticConnector,
            explanation: "Bürokratik geçiş kalıbı; metni mekanikleştiren yapay bağlaç.",
            suggestion: "Doğrudan amaca yönelik bağlaç veya fiil kullanın.",
        },
        LiteralCliche {
            pattern_name: "bu_cercevede",
            phrase: "bu çerçevede",
            category: ClicheCategory::BureaucraticConnector,
            explanation: "Bürokratik geçiş kalıbı.",
            suggestion: "Kaldırın veya 'Böylece' ile sadeleştirin.",
        },
        LiteralCliche {
            pattern_name: "bu_kapsamda",
            phrase: "bu kapsamda",
            category: ClicheCategory::BureaucraticConnector,
            explanation: "Bürokratik geçiş kalıbı.",
            suggestion: "Kaldırın ve eylemi doğrudan söyleyin.",
        },
        LiteralCliche {
            pattern_name: "bunun_yani_sira",
            phrase: "bunun yanı sıra",
            category: ClicheCategory::BureaucraticConnector,
            explanation: "Aşırı tekrarlanan listeleyici geçiş ifadesi.",
            suggestion: "'Ayrıca', 'Üstelik' veya doğrudan 've' kullanın.",
        },
        LiteralCliche {
            pattern_name: "ote_yandan",
            phrase: "öte yandan",
            category: ClicheCategory::BureaucraticConnector,
            explanation: "İngilizce 'on the other hand' çevirisi olarak sıkça suistimal edilen kalıp.",
            suggestion: "'Oysa', 'Fakat' veya 'Buna karşın' tercih edin.",
        },
        LiteralCliche {
            pattern_name: "soz_konusu",
            phrase: "söz konusu",
            category: ClicheCategory::BureaucraticConnector,
            explanation: "Metni gereksiz yere resmileştiren ve dolaylı kılan bürokratik tamlama.",
            suggestion: "İlgili ismi doğrudan adlandırın veya 'bu' zamirini kullanın.",
        },
        LiteralCliche {
            pattern_name: "bununla_birlikte",
            phrase: "bununla birlikte",
            category: ClicheCategory::BureaucraticConnector,
            explanation: "Aşırı kullanılan karşıtlık bağlacı.",
            suggestion: "'Ama', 'Fakat' veya 'Ancak' ile sadeleştirin.",
        },

        // Conclusion & Hedge Clichés
        LiteralCliche {
            pattern_name: "goz_ardi_edilmemelidir",
            phrase: "göz ardı edilmemelidir",
            category: ClicheCategory::ConclusionHedge,
            explanation: "AI metinlerinde sıkça rastlanan yapay uyarı/tembih kalıbı.",
            suggestion: "Doğrudan gerçeği ifade edin.",
        },
        LiteralCliche {
            pattern_name: "unutulmamalidir_ki",
            phrase: "unutulmamalıdır ki",
            category: ClicheCategory::ConclusionHedge,
            explanation: "Didaktik ve yapay anlatım başlatıcısı.",
            suggestion: "Kaldırın ve yargıyı doğrudan söyleyin.",
        },
        LiteralCliche {
            pattern_name: "ozetlemek_gerekirse",
            phrase: "özetlemek gerekirse",
            category: ClicheCategory::ConclusionHedge,
            explanation: "Yapay AI sonuç bölümü başlatıcısı.",
            suggestion: "Doğrudan sonuca girin.",
        },

        // Inflated Hedging & Adverbs
        LiteralCliche {
            pattern_name: "buyuk_olcude",
            phrase: "büyük ölçüde",
            category: ClicheCategory::InflatedHedging,
            explanation: "Yapay genelleme / riskten kaçınma dolgusu.",
            suggestion: "Gereksizse kaldırın.",
        },
        LiteralCliche {
            pattern_name: "suphesiz_ki",
            phrase: "şüphesiz ki",
            category: ClicheCategory::InflatedHedging,
            explanation: "Aşırı didaktik yapay kesinlik vurgusu.",
            suggestion: "Cümleye doğrudan başlayın.",
        },
        LiteralCliche {
            pattern_name: "son_derece",
            phrase: "son derece",
            category: ClicheCategory::InflatedHedging,
            explanation: "Abartılı sıfat niteleyicisi.",
            suggestion: "'Çok' veya doğrudan güçlü bir sıfat kullanın.",
        },

        // Clitic Anomalies
        LiteralCliche {
            pattern_name: "baglac_ki_bitisik",
            phrase: "şuki",
            category: ClicheCategory::CliticAnomaly,
            explanation: "Bağlaç 'ki' ayrı yazılmalıdır (TDK kuralı).",
            suggestion: "'şu ki' şeklinde ayrı yazın.",
        },
        LiteralCliche {
            pattern_name: "baglac_ki_bitisik",
            phrase: "buki",
            category: ClicheCategory::CliticAnomaly,
            explanation: "Bağlaç 'ki' ayrı yazılmalıdır (TDK kuralı).",
            suggestion: "'bu ki' şeklinde ayrı yazın.",
        },
        LiteralCliche {
            pattern_name: "baglac_ki_bitisik",
            phrase: "öyleki",
            category: ClicheCategory::CliticAnomaly,
            explanation: "Bağlaç 'ki' ayrı yazılmalıdır (TDK kuralı).",
            suggestion: "'öyle ki' şeklinde ayrı yazın.",
        },
        LiteralCliche {
            pattern_name: "baglac_ki_bitisik",
            phrase: "tabiki",
            category: ClicheCategory::CliticAnomaly,
            explanation: "Bağlaç 'ki' ayrı yazılmalıdır (TDK kuralı).",
            suggestion: "'tabii ki' şeklinde ayrı yazın.",
        },
        LiteralCliche {
            pattern_name: "hersey_bitisik",
            phrase: "herşey",
            category: ClicheCategory::CliticAnomaly,
            explanation: "'şey' sözcüğü her zaman ayrı yazılır.",
            suggestion: "'her şey' olarak düzeltin.",
        },
        LiteralCliche {
            pattern_name: "hersey_bitisik",
            phrase: "birşey",
            category: ClicheCategory::CliticAnomaly,
            explanation: "'şey' sözcüğü her zaman ayrı yazılır.",
            suggestion: "'bir şey' olarak düzeltin.",
        },

        // Human Authenticity Markers
        LiteralCliche {
            pattern_name: "insani_ritim_belirteci",
            phrase: "daha doğrusu",
            category: ClicheCategory::HumanAuthenticityMarker,
            explanation: "Doğal insan Türkçesine ait samimi bakış açısı / öz-düzeltme belirteci.",
            suggestion: "Doğal üslup korunmalıdır.",
        },
        LiteralCliche {
            pattern_name: "insani_ritim_belirteci",
            phrase: "bana kalırsa",
            category: ClicheCategory::HumanAuthenticityMarker,
            explanation: "Doğal insan Türkçesine ait samimi bakış açısı belirteci.",
            suggestion: "Doğal üslup korunmalıdır.",
        },
        LiteralCliche {
            pattern_name: "insani_ritim_belirteci",
            phrase: "en azından",
            category: ClicheCategory::HumanAuthenticityMarker,
            explanation: "Doğal insan Türkçesine ait bakış açısı belirteci.",
            suggestion: "Doğal üslup korunmalıdır.",
        },
        LiteralCliche {
            pattern_name: "insani_ritim_belirteci",
            phrase: "açıkçası",
            category: ClicheCategory::HumanAuthenticityMarker,
            explanation: "Doğal insan Türkçesine ait samimi itiraf / bakış açısı belirteci.",
            suggestion: "Doğal üslup korunmalıdır.",
        },
        LiteralCliche {
            pattern_name: "insani_ritim_belirteci",
            phrase: "nitekim",
            category: ClicheCategory::HumanAuthenticityMarker,
            explanation: "Doğal Türkçe düşünsel bağlantı bağlacı.",
            suggestion: "Doğal üslup korunmalıdır.",
        },
    ];

    // 2. Dynamic / Regex Patterns (Where wildcards or intervening words are necessary)
    static ref DYNAMIC_PATTERNS: Vec<(&'static str, Regex, ClicheCategory, &'static str, &'static str)> = vec![
        (
            "noktali_virgul_baglac",
            Regex::new(r";\s*(?:aynı zamanda|dolayısıyla|bununla birlikte|öte yandan|ancak|fakat|bu nedenle|bu yüzden|bu doğrultuda|bu bağlamda|ayrıca|üstelik|zira|nitekim)\b").unwrap(),
            ClicheCategory::SemicolonConjunction,
            "TDK noktalama kurallarına göre noktalı virgülden (;) sonra bağlaç gelmez. AI, iki bağımsız cümleyi noktayla bölmek yerine bu yapıyı kurar.",
            "Noktalı virgülü kaldırıp cümleyi nokta ile bitirin veya bağlacı çıkarın."
        ),
        (
            "rol_oynamaktadir",
            Regex::new(r"(?i)\b(?:kritik|önemli|büyük|belirleyici|temel)\s+bir\s+rol\s+(?:oynamaktadır|oynar|üstlenmektedir|oynamaktadırlar)\b").unwrap(),
            ClicheCategory::ConclusionHedge,
            "Klasik AI kapanış ve şişirme kalıbı (İngilizce 'plays a key role' çevirisi).",
            "Öznenin neyi nasıl değiştirdiğini somut eylem fiiliyle ifade edin."
        ),
        (
            "onem_tasimaktadir",
            Regex::new(r"(?i)\b(?:hayati|büyük|kritik|özel)\s+bir\s+önem\s+(?:taşımaktadır|arz\s+etmektedir|taşır)\b").unwrap(),
            ClicheCategory::ConclusionHedge,
            "AI metinlerinin en belirgin yapay önem atfetme kalıbı.",
            "Kalıbı kaldırın; ifadenin neden önemli olduğunu doğrudan aktarın."
        ),
        (
            "oneme_sahiptir",
            Regex::new(r"(?i)\b(?:büyük|önemli|kritik)\s+bir\s+öneme\s+sahiptir\b").unwrap(),
            ClicheCategory::ConclusionHedge,
            "Yapay önem vurgusu kalıbı.",
            "Somut eylemle bitirin."
        ),
        (
            "vazgecilmez",
            Regex::new(r"(?i)\bvazgeçilmez\s+bir\s+(?:unsur|parça|hale\s+gelmiştir|parçasıdır|öğe)\b").unwrap(),
            ClicheCategory::ConclusionHedge,
            "Yapay genelleme ve abartı klişesi.",
            "Metnin ana fikrini somutlaştırın veya cümleyi önceki yargıyla bitirin."
        ),
        (
            "sadece_degil_ayni_zamanda",
            Regex::new(r"(?i)\b(?:sadece|yalnızca|salt)\s+[^.;!?\n]+?\s+değil,\s*(?:aynı\s+zamanda|bununla\s+birlikte)\b").unwrap(),
            ClicheCategory::RhetoricalCalque,
            "İngilizce 'Not only X, but also Y' retorik yapısının Türkçe'ye zorlama çevirisi.",
            "İki yargıyı iki ayrı doğal cümleye bölün veya 'hem X hem Y' yapısını kullanın."
        ),
        (
            "sinirli_kalmayip",
            Regex::new(r"(?i)\b(?:sadece|yalnızca)?\s*[^.;!?\n]+?\s+ile\s+sınırlı\s+kalmayıp\b").unwrap(),
            ClicheCategory::RhetoricalCalque,
            "Zorlama retorik ikileme çevirisi.",
            "Cümleleri sadeleştirin."
        ),
        (
            "olmakla_birlikte_tasimaktadir",
            Regex::new(r"(?i)\b[^.;!?\n]+?\s+olmakla\s+birlikte,\s*[^.;!?\n]+?\s+niteliğini\s+(?:de\s+)?taşımaktadır\b").unwrap(),
            ClicheCategory::RhetoricalCalque,
            "Ağır yapay akademik ikileme.",
            "Aktif fiillerle iki bağımsız cümle kurun."
        ),
        (
            "otesinde_boyutu",
            Regex::new(r"(?i)\b[^.;!?\n]+?'?[ıinunün]?\s+ötesinde,\s*[^.;!?\n]+?\s+boyutu\b").unwrap(),
            ClicheCategory::RhetoricalCalque,
            "İngilizce 'Beyond X, the dimension of Y' retorik çevirisi.",
            "Sade Türkçe cümle yapısına dönüştürün."
        ),
        (
            "etkin_bir_sekilde",
            Regex::new(r"(?i)\b(?:etkin|başarılı|kapsamlı|verimli|titiz)\s+bir\s+şekilde\b").unwrap(),
            ClicheCategory::InflatedHedging,
            "İngilizce '-ly' zarflarının yapay 'bir şekilde' dolgusuyla çevrilmesi.",
            "Doğrudan zarfı kullanın (Örn: 'etkince', 'başarıyla', 'titizlikle') veya eylemi netleştirin."
        ),
        (
            "ortaya_konulmaktadir",
            Regex::new(r"(?i)\b(?:ortaya\s+konulmaktadır|görülmektedir|belirtilmektedir|ele\s+alınmaktadır|vurgulanmaktadır)\b").unwrap(),
            ClicheCategory::PassiveInflation,
            "Yapay edilgen çatı kalıbı (İngilizce 'it is shown that...' çevirisi).",
            "Cümleyi etken çatıyla ve gerçek öznesiyle ifade edin."
        ),
        (
            "sifat_bir_isim_enflasyonu",
            Regex::new(r"(?i)\b(?:önemli|kritik|büyük|belirleyici|temel|merkezi|stratejik|kapsamlı|başarılı|etkili|güçlü)\s+bir\s+(?:adım|rol|aşama|süreç|unsur|faktör|örnek|çalışma|katkı|dönüm\s+noktası)\b").unwrap(),
            ClicheCategory::IndefiniteArticleInflation,
            "İngilizce 'a/an' belgisiz tanımlığının Türkçeye yapay olarak birebir çevrilmesi.",
            "'bir' sözcüğünü kaldırarak doğal Türkçe tamlaması kurun (Örn: 'önemli adım', 'kritik süreç')."
        ),
        (
            "uclu_simetrik_liste",
            Regex::new(r"(?i)\b[abcçdefgğhıijklmnoöprsştuüvyz]+,\s+[abcçdefgğhıijklmnoöprsştuüvyz]+\s+(?:ve|veya)\s+[abcçdefgğhıijklmnoöprsştuüvyz]+\b").unwrap(),
            ClicheCategory::TricolonParallelList,
            "AI'nin simetrik üçlü liste kalıbı (Tricolon). Metne yapay ve mekanik bir ritim verir.",
            "Listeyi sadeleştirin veya cümle içine doğal eylemlerle dağıtın."
        ),
        (
            "sozde_soru_retorigi",
            Regex::new(r"(?i)\b(?:peki\s+(?:bu\s+)?(?:neden|nasıl|ne\s+anlama|ne\s+demek)|nasıl\s+mümkün\s+olabilir)\s*[^.?!]*?\?").unwrap(),
            ClicheCategory::HypophoraQuestion,
            "AI'nin paragraflar arası yapay merak/bağlantı üretmek için kullandığı sözde soru kalıbı.",
            "Soruyu kaldırıp doğrudan açıklama veya argümana geçin."
        ),
    ];
}

pub struct ClicheMatcher;

impl ClicheMatcher {
    /// Finds all bureaucratic connectors, conclusion hedges, rhetorical calques, and hedging patterns in text.
    /// Utilizes StringZilla SIMD substring searches for fast phrase matching.
    pub fn find_all(text: &str) -> Vec<MatchedPattern> {
        let mut matches = Vec::new();
        let lower = to_turkish_lower(text);

        // 1. Fast StringZilla SIMD search for literal clichés and phrases
        for item in LITERAL_CLICHES.iter() {
            let needle = item.phrase;
            let needle_len = needle.len();
            let mut offset = 0;

            while offset < lower.len() {
                if let Some(pos) = lower[offset..].sz_find(needle) {
                    let start = offset + pos;
                    let end = start + needle_len;

                    // Word boundary check (cannot be part of a larger word)
                    let prev_ok = if start == 0 {
                        true
                    } else {
                        let prev_char = lower[..start].chars().next_back().unwrap();
                        !prev_char.is_alphabetic()
                    };

                    let next_ok = if end >= lower.len() {
                        true
                    } else {
                        let next_char = lower[end..].chars().next().unwrap();
                        !next_char.is_alphabetic()
                    };

                    if prev_ok && next_ok {
                        let text_slice = if end <= text.len() && text.is_char_boundary(start) && text.is_char_boundary(end) {
                            text[start..end].to_string()
                        } else {
                            needle.to_string()
                        };

                        matches.push(MatchedPattern {
                            category: item.category.clone(),
                            pattern_name: item.pattern_name,
                            text: text_slice,
                            start,
                            end,
                            explanation: item.explanation,
                            suggestion: item.suggestion,
                        });
                    }

                    offset = end;
                } else {
                    break;
                }
            }
        }

        // 2. Dynamic regex search for variable-length patterns (e.g. "sadece ... değil, aynı zamanda")
        for (name, re, cat, exp, sug) in DYNAMIC_PATTERNS.iter() {
            for m in re.find_iter(text) {
                matches.push(MatchedPattern {
                    category: cat.clone(),
                    pattern_name: name,
                    text: m.as_str().to_string(),
                    start: m.start(),
                    end: m.end(),
                    explanation: exp,
                    suggestion: sug,
                });
            }
        }

        matches.sort_by_key(|m| m.start);
        matches
    }
}
