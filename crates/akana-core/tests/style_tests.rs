use std::collections::BTreeMap;

use akana_core::style::{AiVerdict, StyleAuditReport, TurkishHumanizer, TurkishStyleAuditor};
use serde::Deserialize;

#[test]
fn test_ai_style_auditor_on_synthetic_text() {
    let synthetic_ai_text = "Yapay zeka teknolojileri, modern dünyada kritik bir rol oynamaktadır. \
        Bu bağlamda —özellikle veri analitiği alanında— hayati bir önem taşımaktadır; \
        aynı zamanda sadece işletmeler için değil, aynı zamanda bireyler için de vazgeçilmez bir hale gelmiştir.";

    let auditor = TurkishStyleAuditor::new();
    let report = auditor.audit(synthetic_ai_text);

    assert!(report.ai_score >= 60.0);
    assert!(!report.findings.is_empty());
    assert!(report.metrics.punctuation.em_dash_count >= 1);
    assert!(report.metrics.conclusion_cliches_count >= 2);
    assert!(report.metrics.rhetorical_calques_count >= 1);
    assert!(report.metrics.bureaucratic_connectors_count >= 1);
}

#[test]
fn test_ai_style_auditor_on_human_text() {
    let natural_human_text = "Dün sabah erkenden uyandım. Hava buz gibiydi. \
        Çantamı alıp hızlıca dışarı çıktım çünkü vapuru kaçırmak istemiyordum. \
        İskelede sıcak bir çay içtim.";

    let auditor = TurkishStyleAuditor::new();
    let report = auditor.audit(natural_human_text);

    assert!(report.ai_score < 25.0);
    assert_eq!(report.verdict, "Doğal İnsan Metni");
    assert_eq!(report.metrics.punctuation.em_dash_count, 0);
    assert_eq!(report.metrics.conclusion_cliches_count, 0);
}

#[test]
fn test_humanizer_prompt_generation() {
    let text = "Bu doğrultuda eğitim sistemleri hayati bir önem taşımaktadır.";
    let prompt = TurkishHumanizer::generate_prompt(text, "blog");

    assert!(prompt.contains("Humanizer"));
    assert!(prompt.contains("Kalıp ve Klişe Temizliği"));
}

/// `humanize_prompt` output of the rule-based humanizer before the AI-text detector was
/// added, for texts the detector does not judge AI (`tests/data/humanizer_rule_prompts.json`).
#[derive(Deserialize)]
struct RulePromptCase {
    text: String,
    prompts: BTreeMap<String, String>,
}

#[derive(Deserialize)]
struct RulePrompts {
    cases: Vec<RulePromptCase>,
}

fn rule_prompts() -> RulePrompts {
    serde_json::from_str(include_str!("data/humanizer_rule_prompts.json")).expect("valid JSON")
}

#[test]
fn test_humanizer_keeps_the_rule_prompt_for_non_ai_text() {
    let data = rule_prompts();
    assert_eq!(data.cases.len(), 2);
    for case in &data.cases {
        assert_eq!(case.prompts.len(), 5);
        for (register, want) in &case.prompts {
            let got = TurkishHumanizer::generate_prompt(&case.text, register);
            assert_eq!(&got, want, "register {register}");
        }
    }
}

#[test]
fn test_audit_report_carries_the_ai_detector() {
    let data = rule_prompts();
    let report = TurkishStyleAuditor::new().audit(&data.cases[0].text);
    let detection = report
        .ai_detection
        .as_ref()
        .expect("audit runs the detector");
    assert_eq!(detection.verdict, AiVerdict::Human);
    assert!(detection.report.starts_with("# AI tespit raporu"));

    // Round trip through JSON (the Python binding's path).
    let json = serde_json::to_string(&report).expect("serializable");
    let back: StyleAuditReport = serde_json::from_str(&json).expect("deserializable");
    let back = back.ai_detection.expect("kept");
    assert_eq!(back.verdict, detection.verdict);
    assert_eq!(back.report, detection.report);
    assert_eq!(back.signals.len(), detection.signals.len());
}

#[test]
fn test_audit_report_json_without_ai_detection_still_parses() {
    let report = TurkishStyleAuditor::new().audit("Kısa bir metin.");
    let mut json = serde_json::to_value(&report).expect("serializable");
    json.as_object_mut().expect("object").remove("ai_detection");
    let back: StyleAuditReport = serde_json::from_value(json).expect("deserializable");
    assert!(back.ai_detection.is_none());
}
