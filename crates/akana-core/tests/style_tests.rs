use akana_core::style::{TurkishStyleAuditor, TurkishHumanizer};

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
