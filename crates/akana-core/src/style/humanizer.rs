//! Actionable LLM Humanizer Prompt and Feedback Generator.
//!
//! Two entry points:
//!
//! - [`TurkishHumanizer::humanize`]: the detector-driven pipeline. The AI-text detector
//!   ([`crate::aidetect::ai_text_detect`]) decides first; only a text judged AI gets a
//!   "why AI" report ([`crate::aidetect::why_ai_report`]) and a rewrite prompt aimed at
//!   exactly the signals that made it look AI (strongest features, found patterns, most
//!   AI-like sections), followed by the style auditor's rules.
//! - [`TurkishHumanizer::generate_prompt`]: the style-auditor-only prompt, for any text.

use serde::Serialize;

use super::detector::{StyleAuditReport, TurkishStyleAuditor};
use crate::aidetect::{ai_reasons, ai_text_detect, feature_sentence, AiAnalysis};

pub struct TurkishHumanizer;

/// Result of [`TurkishHumanizer::humanize`].
#[derive(Debug, Clone, Serialize)]
pub struct HumanizeResult {
    /// Detector output with its explanation and "why AI" report.
    pub analysis: AiAnalysis,
    /// Rewrite prompt; `None` unless the detector judged the text AI.
    pub prompt: Option<String>,
}

/// How to move a feature towards human text, by feature key; `None` falls back to a
/// generic sentence built from the AI and human averages.
fn feature_advice(key: &str) -> Option<&'static str> {
    Some(match key {
        "wordlm_llr_mean" | "wordlm_ai_share" => {
            "Kelime seçimi ve ek dizilimi kalıplaşmış AI Türkçesine benziyor: hazır, \
             cilalı ifadeler yerine gündelik, somut ve kişisel kelimeler kullan; aynı \
             fikri kendi cümlenle söyle."
        }
        "pat_evidence" | "pat_phrase_n" | "pat_sent_start_n" | "pat_opener" | "pat_closer" => {
            "Aşağıda listelenen AI kalıplarını (giriş, kapanış, cümle başı ve hazır \
             ifadeler) metinden çıkar ya da tamamen farklı sözcüklerle yeniden yaz."
        }
        _ => return None,
    })
}

/// Rewrite advice for a feature: move this text's value towards the human average. The
/// direction comes from where the value is, not from the averages alone (a value can be
/// AI-like while already past the human average).
fn generic_advice(f: &crate::aidetect::FeatureContribution) -> String {
    let dir = match (f.ai_mean, f.human_mean) {
        (_, Some(h)) if f.value > h => "azalt (insan metinlerindeki düzeye yaklaştır)",
        (_, Some(h)) if f.value < h => "artır (insan metinlerindeki düzeye yaklaştır)",
        (Some(a), Some(h)) if h < a => "azalt (insan metinlerindeki düzeye yaklaştır)",
        (Some(_), Some(_)) => "artır (insan metinlerindeki düzeye yaklaştır)",
        _ => "insan metinlerindeki düzeye yaklaştır",
    };
    format!("{} Bunu {dir}.", feature_sentence(f))
}

impl TurkishHumanizer {
    /// Detector-driven pipeline: detect → "why AI" report → targeted rewrite prompt (only
    /// when the text is judged AI).
    pub fn humanize(text: &str, register: &str) -> HumanizeResult {
        let analysis = ai_text_detect(text);
        let prompt = analysis.is_ai().then(|| {
            let audit = TurkishStyleAuditor::new().audit(text);
            Self::format_targeted_prompt(text, register, &analysis, &audit)
        });
        HumanizeResult { analysis, prompt }
    }

    /// Rewrite prompt built from the detector's explanation, then the style audit rules.
    pub fn format_targeted_prompt(
        text: &str,
        register: &str,
        analysis: &AiAnalysis,
        audit: &StyleAuditReport,
    ) -> String {
        let mut instructions: Vec<String> = Vec::new();
        let mut pattern_rule = false;
        let mut token_rule = false;
        for f in ai_reasons(analysis, 6) {
            match feature_advice(&f.key) {
                Some(advice) if f.key.starts_with("pat_") => {
                    if !pattern_rule && !analysis.patterns.is_empty() {
                        pattern_rule = true;
                        instructions.push(format!("**AI kalıpları:** {advice}"));
                    }
                }
                Some(advice) => {
                    if !token_rule {
                        token_rule = true;
                        instructions.push(format!("**Kelime ve ek dizilimi:** {advice}"));
                    }
                }
                None => instructions.push(generic_advice(f)),
            }
        }
        if !pattern_rule && !analysis.patterns.is_empty() {
            instructions.push(format!(
                "**AI kalıpları:** {}",
                feature_advice("pat_evidence").unwrap_or_default()
            ));
        }
        instructions.extend(Self::audit_instructions(audit));

        let mut prompt = String::new();
        prompt.push_str("# Türkçe Metin Doğallaştırma (Humanizer) Yönergesi\n\n");
        prompt.push_str(&format!(
            "**Hedef Register:** {}\n",
            Self::register_guidance(register)
        ));
        prompt.push_str(&format!(
            "**AI tespiti:** AI olasılığı %{:.0} (eşik %{:.0}). Aşağıdaki kurallar, metni AI \
             gibi gösteren ölçümlerden çıkarıldı; en etkiliden başlayarak uygula. Anlamı, \
             bilgileri ve dili (Türkçe) koru.\n\n",
            analysis.ai_probability * 100.0,
            analysis.threshold * 100.0
        ));
        prompt.push_str(
            "Lütfen aşağıdaki metni şu somut kurallara uyarak insan Türkçesine dönüştür:\n\n",
        );
        for (i, inst) in instructions.iter().enumerate() {
            prompt.push_str(&format!("{}. {inst}\n", i + 1));
        }
        if !analysis.patterns.is_empty() {
            prompt.push_str("\n### Çıkarılacak / yeniden yazılacak kalıplar:\n");
            for p in analysis.patterns.iter().take(12) {
                prompt.push_str(&format!("- «{}» ({})\n", p.pattern, p.kind));
            }
        }
        let hot: Vec<_> = analysis
            .sections
            .iter()
            .filter(|s| s.ai_probability >= analysis.threshold)
            .take(3)
            .collect();
        if !hot.is_empty() {
            prompt.push_str("\n### Önce bu bölümlere odaklan (en çok AI'a benzeyenler):\n");
            for s in hot {
                prompt.push_str(&format!(
                    "- {}–{}. kelimeler: «{}…»\n",
                    s.start_word,
                    s.end_word,
                    s.text.chars().take(80).collect::<String>()
                ));
            }
        }
        prompt.push_str("\n### İşlenecek Metin:\n");
        prompt.push_str("```text\n");
        prompt.push_str(text.trim());
        prompt.push_str("\n```\n");
        prompt
    }

    /// Generates a structured, actionable prompt instructing an LLM (or human editor)
    /// to humanize the text according to its detected AI signatures and register.
    pub fn generate_prompt(text: &str, register: &str) -> String {
        let auditor = TurkishStyleAuditor::new();
        let report = auditor.audit(text);

        Self::format_prompt(text, register, &report)
    }

    /// Formats the audit report into a ready-to-paste prompt.
    pub fn format_prompt(text: &str, register: &str, report: &StyleAuditReport) -> String {
        let instructions = Self::audit_instructions(report);
        let register_guidance = Self::register_guidance(register);

        let mut prompt = String::new();
        prompt.push_str("# Türkçe Metin Doğallaştırma (Humanizer) Yönergesi\n\n");
        prompt.push_str(&format!("**Hedef Register:** {register_guidance}\n"));
        prompt.push_str(&format!(
            "**Tespit Edilen AI Skoru:** {:.1} / 100 ({})\n\n",
            report.ai_score, report.verdict
        ));
        prompt.push_str(
            "Lütfen aşağıdaki metni şu somut kurallara uyarak insan Türkçesine dönüştür:\n\n",
        );

        for (i, inst) in instructions.iter().enumerate() {
            prompt.push_str(&format!("{}. {inst}\n", i + 1));
        }

        prompt.push_str("\n### İşlenecek Metin:\n");
        prompt.push_str("```text\n");
        prompt.push_str(text.trim());
        prompt.push_str("\n```\n");

        prompt
    }

    /// Rule-based rewrite instructions from a style audit (unnumbered).
    pub fn audit_instructions(report: &StyleAuditReport) -> Vec<String> {
        let mut instructions: Vec<String> = Vec::new();

        // 1. Predicate Tense Variety Rule
        if report.metrics.predicates.mektedir_count > 0 {
            instructions.push(format!(
                "**Yüklem Çeşitliliği:** Metindeki {} adet '-mektedir/-maktadır' yüklemini geçmiş zaman (-di), geniş zaman (-r), şimdiki zaman (-yor) ve aktif eylemlerle değiştir.",
                report.metrics.predicates.mektedir_count
            ));
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
                "**Kalıp ve Klişe Temizliği:** Metinde bulunan {} temizle. Düşünceleri dolaylı kalıplara sokmadan doğrudan ve somut aktar.",
                list.join(", ")
            ));
        }

        // 3. Punctuation Cleanliness Rule
        let mut punct_items = Vec::new();
        let has_semicolon_conj = report
            .findings
            .iter()
            .any(|f| f.category == "SemicolonConjunction");
        if has_semicolon_conj {
            punct_items.push(
                "noktalı virgülden (;) sonra gelen bağlaçları kaldırıp cümleyi nokta ile böl",
            );
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
                "**Noktalama Düzenlemesi:** TDK kurallarına ve doğal Türkçeye uygun olarak {}.",
                punct_items.join(", ")
            ));
        }

        // 4. Structure & Rhetoric (Tricolon & Hypophora)
        let has_tricolon = report
            .findings
            .iter()
            .any(|f| f.category == "TricolonParallelList");
        let has_hypophora = report
            .findings
            .iter()
            .any(|f| f.category == "HypophoraQuestion");
        if has_tricolon || has_hypophora {
            let mut struct_items = Vec::new();
            if has_tricolon {
                struct_items.push("mekanik 3'lü simetrik listeleri ('hızlı, güvenilir ve etkili') kırıp doğal anlatıma dönüştür");
            }
            if has_hypophora {
                struct_items.push("paragraf başı yapay sözde soruları ('Peki bu neden önemli?') kaldırıp doğrudan konuya gir");
            }
            instructions.push(format!(
                "**Yapay Retorik ve Liste Temizliği:** {}.",
                struct_items.join(" ve ")
            ));
        }

        // 5. Rhythm and Burstiness Rule
        if report.metrics.rhythm.is_low_burstiness
            || report.metrics.rhythm.short_sentences_count == 0
        {
            instructions.push(format!(
                "**Cümle Ritim ve Nefesini Canlandır (Burstiness):** Cümleler ortalama {:.1} kelime ile tekdüze uzunluktadır. En az bir uzun cümleyi bölerek araya 1-4 kelimelik kısa, net nefes cümleleri ekle.",
                report.metrics.rhythm.mean_sentence_length
            ));
        }

        instructions
    }

    /// Register-specific tone guidance line.
    pub fn register_guidance(register: &str) -> &'static str {
        let reg_lower = register.to_lowercase();
        if reg_lower.contains("hukuk") || reg_lower.contains("idari") {
            "Metin türü: **Hukuki-İdari**. Terimsel kesinliği koru, sadece yapay çeviri kalıplarını ve noktalama bozukluklarını temizle. Konuşma dili unsurları ekleme."
        } else if reg_lower.contains("akademi") || reg_lower.contains("kurumsal") {
            "Metin türü: **Akademik-Kurumsal**. Ciddiyeti ve nesnelliği koru, ancak '-mektedir' monotonluğunu ve 'bu bağlamda' gibi dolgu kalıplarını kaldır."
        } else if reg_lower.contains("analitik")
            || reg_lower.contains("haber")
            || reg_lower.contains("gazete")
        {
            "Metin türü: **Analitik-Gazetecilik**. Akıcı, dinamik ve net bir üslup kullan. Gereksiz dolguları at, somut verilere ve eylemlere odaklan."
        } else if reg_lower.contains("edebi") || reg_lower.contains("yaratıcı") {
            "Metin türü: **Edebi-Yaratıcı**. Duyusal detayları, ritim çeşitliliğini ve canlı Türkçe metaforlarını zenginleştir."
        } else {
            // Default: Deneme / Blog
            "Metin türü: **Deneme / Blog / Serbest Anlatı**. Samimi, nefes alan, canlı ve insani bir Türkçe ritmi yakala. Kuru ve mekanik AI tonunu tamamen kır."
        }
    }
}

#[cfg(test)]
mod tests {
    use super::generic_advice;
    use crate::aidetect::FeatureContribution;

    fn feature(value: f64, ai_mean: f64, human_mean: f64) -> FeatureContribution {
        FeatureContribution {
            key: "k".into(),
            name: "n".into(),
            group: "g".into(),
            desc: "d".into(),
            value,
            ai_mean: Some(ai_mean),
            human_mean: Some(human_mean),
            contribution: 0.1,
        }
    }

    #[test]
    fn advice_moves_the_value_towards_the_human_average() {
        // AI average below the human one, but this text is already above it: reduce.
        assert!(generic_advice(&feature(4.5, 1.6, 2.3)).contains("Bunu azalt"));
        assert!(generic_advice(&feature(1.0, 1.6, 2.3)).contains("Bunu artır"));
        assert!(generic_advice(&feature(1.0, 0.95, 0.52)).contains("Bunu azalt"));
        // At the human average: fall back to the direction of the averages.
        assert!(generic_advice(&feature(0.52, 0.95, 0.52)).contains("Bunu azalt"));
    }
}
