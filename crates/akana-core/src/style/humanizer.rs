//! Actionable LLM Humanizer Prompt and Feedback Generator.

use super::detector::{TurkishStyleAuditor, StyleAuditReport};

pub struct TurkishHumanizer;

impl TurkishHumanizer {
    /// Generates a structured, actionable prompt instructing an LLM (or human editor)
    /// to humanize the text according to its detected AI signatures and register.
    pub fn generate_prompt(text: &str, register: &str) -> String {
        let auditor = TurkishStyleAuditor::new();
        let report = auditor.audit(text);

        Self::format_prompt(text, register, &report)
    }

    /// Formats the audit report into a ready-to-paste prompt.
    pub fn format_prompt(text: &str, register: &str, report: &StyleAuditReport) -> String {
        let mut instructions: Vec<String> = Vec::new();
        let mut step = 1;

        // 1. Predicate Tense Variety Rule
        if report.metrics.predicates.mektedir_count > 0 {
            instructions.push(format!(
                "{}. **Yüklem Çeşitliliği:** Metindeki {} adet '-mektedir/-maktadır' yüklemini geçmiş zaman (-di), geniş zaman (-r), şimdiki zaman (-yor) ve aktif eylemlerle değiştir.",
                step, report.metrics.predicates.mektedir_count
            ));
            step += 1;
        }

        // 2. Clichés and Bureaucratic Connectors Rule
        let total_cliches = report.metrics.bureaucratic_connectors_count
            + report.metrics.conclusion_cliches_count
            + report.metrics.rhetorical_calques_count;

        if total_cliches > 0 {
            let mut list = Vec::new();
            if report.metrics.bureaucratic_connectors_count > 0 {
                list.push("bürokratik geçiş bağlaçlarını ('Bu bağlamda', 'Bu doğrultuda', 'Bu çerçevede')");
            }
            if report.metrics.conclusion_cliches_count > 0 {
                list.push("kapanış klişelerini ('kritik bir rol oynamaktadır', 'hayati önem taşımaktadır', 'vazgeçilmez bir hale gelmiştir')");
            }
            if report.metrics.rhetorical_calques_count > 0 {
                list.push("İngilizce retorik çevirilerini ('Sadece ... değil, aynı zamanda ...')");
            }

            instructions.push(format!(
                "{}. **Kalıp ve Klişe Temizliği:** Metinde bulunan {} temizle. Düşünceleri dolaylı kalıplara sokmadan doğrudan ve somut aktar.",
                step, list.join(", ")
            ));
            step += 1;
        }

        // 3. Punctuation Cleanliness Rule
        let mut punct_items = Vec::new();
        let has_semicolon_conj = report.findings.iter().any(|f| f.category == "SemicolonConjunction");
        if has_semicolon_conj {
            punct_items.push("noktalı virgülden (;) sonra gelen bağlaçları kaldırıp cümleyi nokta ile böl");
        }
        if report.metrics.punctuation.em_dash_count > 0 {
            punct_items.push("cümle içi uzun tireleri (—) virgüle çevir veya cümleyi böl");
        }
        if report.metrics.punctuation.semicolon_count >= 2 && !has_semicolon_conj {
            punct_items.push("noktalı virgülleri (;) kaldırarak iki bağımsız cümle oluştur");
        }
        if report.metrics.punctuation.inline_colon_count >= 2 {
            punct_items.push("cümle içi açıklama iki noktalarını (:) doğal bağlaçlarla değiştir");
        }

        if !punct_items.is_empty() {
            instructions.push(format!(
                "{}. **Noktalama Düzenlemesi:** TDK kurallarına ve doğal Türkçeye uygun olarak {}.",
                step, punct_items.join(", ")
            ));
            step += 1;
        }

        // 4. Structure & Rhetoric (Tricolon & Hypophora)
        let has_tricolon = report.findings.iter().any(|f| f.category == "TricolonParallelList");
        let has_hypophora = report.findings.iter().any(|f| f.category == "HypophoraQuestion");
        if has_tricolon || has_hypophora {
            let mut struct_items = Vec::new();
            if has_tricolon {
                struct_items.push("mekanik 3'lü simetrik listeleri ('hızlı, güvenilir ve etkili') kırıp doğal anlatıma dönüştür");
            }
            if has_hypophora {
                struct_items.push("paragraf başı yapay sözde soruları ('Peki bu neden önemli?') kaldırıp doğrudan konuya gir");
            }
            instructions.push(format!(
                "{}. **Yapay Retorik ve Liste Temizliği:** {}.",
                step, struct_items.join(" ve ")
            ));
            step += 1;
        }

        // 5. Rhythm and Burstiness Rule
        if report.metrics.rhythm.is_low_burstiness || report.metrics.rhythm.short_sentences_count == 0 {
            instructions.push(format!(
                "{}. **Cümle Ritim ve Nefesini Canlandır (Burstiness):** Cümleler ortalama {:.1} kelime ile tekdüze uzunluktadır. En az bir uzun cümleyi bölerek araya 1-4 kelimelik kısa, net nefes cümleleri ekle.",
                step, report.metrics.rhythm.mean_sentence_length
            ));
        }

        // 5. Register Specific Guidance
        let reg_lower = register.to_lowercase();
        let register_guidance = if reg_lower.contains("hukuk") || reg_lower.contains("idari") {
            "Metin türü: **Hukuki-İdari**. Terimsel kesinliği koru, sadece yapay çeviri kalıplarını ve noktalama bozukluklarını temizle. Konuşma dili unsurları ekleme."
        } else if reg_lower.contains("akademi") || reg_lower.contains("kurumsal") {
            "Metin türü: **Akademik-Kurumsal**. Ciddiyeti ve nesnelliği koru, ancak '-mektedir' monotonluğunu ve 'bu bağlamda' gibi dolgu kalıplarını kaldır."
        } else if reg_lower.contains("analitik") || reg_lower.contains("haber") || reg_lower.contains("gazete") {
            "Metin türü: **Analitik-Gazetecilik**. Akıcı, dinamik ve net bir üslup kullan. Gereksiz dolguları at, somut verilere ve eylemlere odaklan."
        } else if reg_lower.contains("edebi") || reg_lower.contains("yaratıcı") {
            "Metin türü: **Edebi-Yaratıcı**. Duyusal detayları, ritim çeşitliliğini ve canlı Türkçe metaforlarını zenginleştir."
        } else {
            // Default: Deneme / Blog
            "Metin türü: **Deneme / Blog / Serbest Anlatı**. Samimi, nefes alan, canlı ve insani bir Türkçe ritmi yakala. Kuru ve mekanik AI tonunu tamamen kır."
        };

        let mut prompt = String::new();
        prompt.push_str("# Türkçe Metin Doğallaştırma (Humanizer) Yönergesi\n\n");
        prompt.push_str(&format!("**Hedef Register:** {}\n", register_guidance));
        prompt.push_str(&format!("**Tespit Edilen AI Skoru:** {:.1} / 100 ({})\n\n", report.ai_score, report.verdict));
        prompt.push_str("Lütfen aşağıdaki metni şu somut kurallara uyarak insan Türkçesine dönüştür:\n\n");

        for inst in &instructions {
            prompt.push_str(inst);
            prompt.push('\n');
        }

        prompt.push_str("\n### İşlenecek Metin:\n");
        prompt.push_str("```text\n");
        prompt.push_str(text.trim());
        prompt.push_str("\n```\n");

        prompt
    }
}
