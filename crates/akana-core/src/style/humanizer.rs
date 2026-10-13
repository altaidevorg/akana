//! Actionable LLM Humanizer Prompt and Feedback Generator.
//!
//! [`TurkishHumanizer::generate_prompt`] audits the text ([`TurkishStyleAuditor::audit`]).
//! When the audit's AI-text detector ([`StyleAuditReport::ai_detection`]) judges the text AI,
//! the prompt starts with the signals that made it look AI (strongest model inputs, found AI
//! phrases, most AI-like sections) and continues with the rule-based instructions; otherwise
//! it is the rule-based prompt alone.

use std::fmt::Write as _;

use super::ai_detector::{
    ai_signals, pct, sections, signal_sentence, AiDetection, AiPattern, AiSignal, PatternKind,
};
use super::detector::{StyleAuditReport, TurkishStyleAuditor};

pub struct TurkishHumanizer;

/// Detector instructions listed at most, AI phrases listed at most, and AI-like sections
/// pointed to at most.
const MAX_DETECTOR_RULES: usize = 6;
const MAX_PROMPT_PATTERNS: usize = 12;
const MAX_HOT_SECTIONS: usize = 3;
/// Characters of a section quoted in the prompt.
const SECTION_QUOTE_CHARS: usize = 80;

/// Model inputs that are no writing advice: moving them towards human text would mean adding
/// typos or slang (`oov_rate`), lower-case sentence starts, `!!` / `?!` / `...`, changing the
/// length, or chasing abstract statistics. They stay in the "why AI" report.
const NO_ADVICE: &[&str] = &[
    "oov_rate",
    "log_words",
    "lower_start_share",
    "multi_excl",
    "interrobang",
    "multi_q",
    "ellipsis",
    "word_entropy",
    "heaps_exponent",
    "compression_ratio",
];

/// Inputs that share one instruction.
#[derive(Clone, Copy, PartialEq, Eq)]
enum AdviceGroup {
    /// Token-transition scores.
    WordChoice,
    /// Mined AI pattern counts.
    Patterns,
}

const WORD_CHOICE_INPUTS: &[&str] = &["wordlm_llr_mean", "wordlm_ai_share"];
const PATTERN_INPUTS: &[&str] = &[
    "pat_evidence",
    "pat_phrase_n",
    "pat_sent_start_n",
    "pat_opener",
    "pat_closer",
];

impl AdviceGroup {
    fn of(key: &str) -> Option<Self> {
        if WORD_CHOICE_INPUTS.contains(&key) {
            Some(Self::WordChoice)
        } else if PATTERN_INPUTS.contains(&key) {
            Some(Self::Patterns)
        } else {
            None
        }
    }

    fn instruction(self) -> &'static str {
        match self {
            Self::WordChoice => {
                "**Kelime ve ek dizilimi:** Kelime seçimi ve ek dizilimi kalıplaşmış AI \
                 Türkçesine benziyor: hazır, cilalı ifadeler yerine gündelik, somut ve kişisel \
                 kelimeler kullan; aynı fikri kendi cümlenle söyle."
            }
            Self::Patterns => {
                "**AI kalıpları:** Aşağıda listelenen AI kalıplarını (giriş, kapanış, cümle başı \
                 ve hazır ifadeler) metinden çıkar ya da tamamen farklı sözcüklerle yeniden yaz."
            }
        }
    }
}

/// Conjunctions that make a two-word phrase a topic pair ("ve enerji", "sanat ve") rather
/// than a style habit ("öneme sahiptir", "katkıda bulunur").
const TOPIC_PAIR_WORDS: &[&str] = &["ve", "ile", "veya"];

/// Found patterns worth a rewrite instruction: all but two-word topic pairs, which stay in
/// the "why AI" report only.
fn rewrite_patterns(d: &AiDetection) -> impl Iterator<Item = &AiPattern> {
    d.patterns.iter().filter(|p| !is_topic_pair(p))
}

fn is_topic_pair(p: &AiPattern) -> bool {
    let words: Vec<&str> = p.pattern.split(' ').collect();
    p.kind == PatternKind::Phrase
        && words.len() == 2
        && words.iter().any(|w| TOPIC_PAIR_WORDS.contains(w))
}

/// Rewrite advice for one input: move this text's value towards the human average. The
/// direction comes from where the value is, not from the averages alone (a value can be
/// AI-like while already past the human average).
fn signal_advice(s: &AiSignal) -> String {
    let towards = "(insan metinlerindeki düzeye yaklaştır)";
    let reduce = if s.value == s.human_mean {
        s.human_mean < s.ai_mean
    } else {
        s.value > s.human_mean
    };
    let verb = if reduce { "azalt" } else { "artır" };
    format!("{} Bunu {verb} {towards}.", signal_sentence(s))
}

/// Rewrite instructions from the detector's strongest AI-ward signals (unnumbered).
fn detector_instructions(d: &AiDetection) -> Vec<String> {
    let has_patterns = rewrite_patterns(d).next().is_some();
    let mut instructions = Vec::new();
    let mut groups_done: Vec<AdviceGroup> = Vec::new();
    let signals = ai_signals(d)
        .filter(|s| !NO_ADVICE.contains(&s.key.as_str()))
        .take(MAX_DETECTOR_RULES);
    for s in signals {
        match AdviceGroup::of(&s.key) {
            Some(g) if groups_done.contains(&g) => {}
            Some(AdviceGroup::Patterns) if !has_patterns => {}
            Some(g) => {
                groups_done.push(g);
                instructions.push(g.instruction().to_string());
            }
            None => instructions.push(signal_advice(s)),
        }
    }
    if has_patterns && !groups_done.contains(&AdviceGroup::Patterns) {
        instructions.push(AdviceGroup::Patterns.instruction().to_string());
    }
    instructions
}

/// The found AI phrases and the most AI-like sections of the text, as prompt sections.
fn push_detector_sections(prompt: &mut String, d: &AiDetection, text: &str) {
    let mut patterns = rewrite_patterns(d).take(MAX_PROMPT_PATTERNS).peekable();
    if patterns.peek().is_some() {
        prompt.push_str("\n### Çıkarılacak / yeniden yazılacak kalıplar:\n");
        for p in patterns {
            let _ = writeln!(prompt, "- «{}» ({})", p.pattern, p.kind.label_tr());
        }
    }
    // The most AI-like sections above the threshold, shown in text order.
    let mut hot: Vec<_> = sections(text)
        .into_iter()
        .filter(|s| s.ai_probability >= d.threshold)
        .collect();
    hot.sort_by(|a, b| b.ai_probability.total_cmp(&a.ai_probability));
    hot.truncate(MAX_HOT_SECTIONS);
    hot.sort_by_key(|s| s.start_word);
    if !hot.is_empty() {
        prompt.push_str("\n### Önce bu bölümlere odaklan (en çok AI'a benzeyenler):\n");
        for s in hot {
            let quote: String = s.text.chars().take(SECTION_QUOTE_CHARS).collect();
            let _ = writeln!(
                prompt,
                "- {}–{}. kelimeler: «{quote}…»",
                s.start_word, s.end_word
            );
        }
    }
}

impl TurkishHumanizer {
    /// Generates a structured, actionable prompt instructing an LLM (or human editor)
    /// to humanize the text according to its detected AI signatures and register.
    pub fn generate_prompt(text: &str, register: &str) -> String {
        let auditor = TurkishStyleAuditor::new();
        let report = auditor.audit(text);

        Self::format_prompt(text, register, &report)
    }

    /// Formats the audit report into a ready-to-paste prompt.
    ///
    /// When the report's AI-text detector (`ai_detection`) judges the text AI, the prompt
    /// starts with the signals that made it look AI (strongest model features, found AI
    /// phrases, most AI-like sections) before the rule-based instructions; otherwise it is
    /// the rule-based prompt alone.
    pub fn format_prompt(text: &str, register: &str, report: &StyleAuditReport) -> String {
        let detected = report
            .ai_detection
            .as_ref()
            .filter(|a| a.is_ai() && a.ai_probability.is_some());
        let mut instructions = detected.map(detector_instructions).unwrap_or_default();
        instructions.extend(Self::audit_instructions(report));
        let register_guidance = Self::register_guidance(register);

        let mut prompt = String::new();
        prompt.push_str("# Türkçe Metin Doğallaştırma (Humanizer) Yönergesi\n\n");
        prompt.push_str(&format!("**Hedef Register:** {register_guidance}\n"));
        match detected {
            // One headline verdict: the trained detector's. The rule-based score is shown
            // as what it is, a count of style findings, without its own verdict.
            Some(a) => {
                let _ = writeln!(
                    prompt,
                    "**AI tespiti:** AI olasılığı {} (eşik {}). İlk kurallar, metni AI gibi \
                     gösteren ölçümlerden çıkarıldı; en etkiliden başlayarak uygula. Anlamı, \
                     bilgileri ve dili (Türkçe) koru.",
                    pct(a.ai_probability.unwrap_or_default()),
                    pct(a.threshold)
                );
                let _ = writeln!(
                    prompt,
                    "**Stil bulguları skoru:** {:.1} / 100",
                    report.ai_score
                );
            }
            None => {
                let _ = writeln!(
                    prompt,
                    "**Tespit Edilen AI Skoru:** {:.1} / 100 ({})",
                    report.ai_score, report.verdict
                );
            }
        }
        prompt.push('\n');
        prompt.push_str(
            "Lütfen aşağıdaki metni şu somut kurallara uyarak insan Türkçesine dönüştür:\n\n",
        );

        for (i, inst) in instructions.iter().enumerate() {
            let _ = writeln!(prompt, "{}. {inst}", i + 1);
        }
        if let Some(a) = detected {
            push_detector_sections(&mut prompt, a, text);
        }

        prompt.push_str("\n### İşlenecek Metin:\n");
        prompt.push_str("```text\n");
        prompt.push_str(text.trim());
        prompt.push_str("\n```\n");

        prompt
    }

    /// Rule-based rewrite instructions from a style audit (unnumbered).
    fn audit_instructions(report: &StyleAuditReport) -> Vec<String> {
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
    fn register_guidance(register: &str) -> &'static str {
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
    use super::*;
    use crate::style::ai_detector::detect;
    use crate::style::ai_detector::model_input_keys;
    use crate::style::ai_detector::samples::AI_TEXT;

    fn signal(value: f64, ai_mean: f64, human_mean: f64) -> AiSignal {
        AiSignal {
            key: "k".into(),
            name: "n".into(),
            value,
            ai_mean,
            human_mean,
            contribution: 0.1,
        }
    }

    #[test]
    fn advice_moves_the_value_towards_the_human_average() {
        // AI average below the human one, but this text is already above it: reduce.
        assert!(signal_advice(&signal(4.5, 1.6, 2.3)).contains("Bunu azalt"));
        assert!(signal_advice(&signal(1.0, 1.6, 2.3)).contains("Bunu artır"));
        assert!(signal_advice(&signal(1.0, 0.95, 0.52)).contains("Bunu azalt"));
        // At the human average: fall back to the direction of the averages.
        assert!(signal_advice(&signal(0.52, 0.95, 0.52)).contains("Bunu azalt"));
    }

    #[test]
    fn no_advice_inputs_never_become_instructions() {
        let d = detect(AI_TEXT);
        assert!(d.is_ai());
        for line in detector_instructions(&d) {
            for s in d
                .signals
                .iter()
                .filter(|s| NO_ADVICE.contains(&s.key.as_str()))
            {
                assert!(!line.contains(&format!("**{}**", s.name)), "{line}");
            }
        }
    }

    #[test]
    fn advice_keys_are_model_inputs() {
        let keys = model_input_keys();
        for key in NO_ADVICE
            .iter()
            .chain(WORD_CHOICE_INPUTS)
            .chain(PATTERN_INPUTS)
        {
            assert!(keys.contains(key), "{key} is not a model input");
        }
    }

    #[test]
    fn topic_pairs_are_not_rewrite_instructions() {
        let d = detect(AI_TEXT);
        let (pairs, kept): (Vec<_>, Vec<_>) = d.patterns.iter().partition(|p| is_topic_pair(p));
        assert!(!pairs.is_empty() && !kept.is_empty());
        let prompt = TurkishHumanizer::generate_prompt(AI_TEXT, "blog");
        for p in pairs {
            assert!(
                !prompt.contains(&format!("«{}»", p.pattern)),
                "{}",
                p.pattern
            );
        }
        for p in kept.iter().take(MAX_PROMPT_PATTERNS) {
            assert!(
                prompt.contains(&format!("«{}»", p.pattern)),
                "{}",
                p.pattern
            );
        }
    }

    #[test]
    fn ai_prompt_has_one_headline_verdict() {
        let prompt = TurkishHumanizer::generate_prompt(AI_TEXT, "blog");
        assert!(prompt.contains("**AI tespiti:**"));
        assert!(prompt.contains("**Stil bulguları skoru:**"));
        assert!(!prompt.contains("Tespit Edilen AI Skoru"));
        assert!(prompt.contains("### Çıkarılacak / yeniden yazılacak kalıplar:"));
    }
}
