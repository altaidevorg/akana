"""
Akana Turkish AI Writing Signature Auditor & Humanizer Engine
Demonstrates:
1. Auditing synthetic AI text for punctuation anomalies, rhythm monotony, bureaucratic clichés, and rhetorical calques.
2. Auditing natural human-written Turkish text to verify clean baseline scores.
3. Generating an actionable, register-aware LLM rewrite prompt to humanize the text.
"""

import akana

def main():
    print("=" * 80)
    print(" AKANA TURKISH AI WRITING STYLE AUDITOR & HUMANIZER")
    print("=" * 80)

    # 1. Sample AI-Generated Text
    ai_sample = (
        "Yapay zeka teknolojileri, modern dünyada bireylerin ve kurumların hayatında kritik bir rol oynamaktadır. "
        "Bu bağlamda —özellikle doğal dil işleme ve makine öğrenimi alanında— hayati bir önem taşımaktadır; "
        "aynı zamanda sadece büyük işletmeler için değil, aynı zamanda günlük kullanıcılar için de vazgeçilmez bir hale gelmiştir. "
        "Bu doğrultuda geliştirilen algoritmalar, karar alma süreçlerinde etkin bir şekilde kullanılmaktadır."
    )

    print("\n" + "#" * 80)
    print(" SAMPLE 1: TYPICAL AI-GENERATED TURKISH TEXT")
    print("#" * 80)
    print(f"\nOriginal Text:\n\"{ai_sample}\"\n")

    report1 = akana.audit_ai_style(ai_sample)
    print(f"[*] AI Score: {report1.ai_score:.1f} / 100.0")
    print(f"[*] Verdict:  {report1.verdict}")
    print(f"[*] Metrics Summary:")
    print(f"    • Total sentences: {report1.metrics['rhythm']['total_sentences']}, Total words: {report1.metrics['rhythm']['total_words']}")
    print(f"    • Mean sentence length: {report1.metrics['rhythm']['mean_sentence_length']:.1f} words (Std Dev: {report1.metrics['rhythm']['std_dev_sentence_length']:.1f})")
    print(f"    • '-mektedir/-maktadır' predicates: {report1.metrics['predicates']['mektedir_count']} (%{report1.metrics['predicates']['mektedir_ratio']*100:.0f})")
    print(f"    • Punctuation anomalies: {report1.metrics['punctuation']['em_dash_count']} em-dashes (—), {report1.metrics['punctuation']['semicolon_count']} semicolons (;)")
    print(f"    • Clichés & Calques: {report1.metrics['bureaucratic_connectors_count']} bureaucratic, {report1.metrics['conclusion_cliches_count']} conclusion hedges, {report1.metrics['rhetorical_calques_count']} rhetorical calques")

    print(f"\n[*] Diagnostic Findings ({len(report1.findings)}):")
    for idx, f in enumerate(report1.findings, 1):
        print(f"    {idx}. [{f.severity:<8}] ({f.category}) {f.message}")
        if f.suggestion:
            print(f"       -> Öneri: {f.suggestion}")

    # 2. Generate Actionable Humanizer Prompt
    print("\n" + "-" * 80)
    print(" GENERATED ACTIONABLE HUMANIZER PROMPT (for Claude / ChatGPT / Human Editor):")
    print("-" * 80)
    prompt = akana.humanize_prompt(ai_sample, register="blog")
    print(prompt)

    # 3. Sample Natural Human-Written Text
    human_sample = (
        "Dün sabah erken saatlerde uyandım. Dışarıda dondurucu bir soğuk vardı. "
        "Aceleyle çantamı hazırlayıp iskeleye doğru yürüdüm çünkü ilk vapuru kaçırmak istemiyordum. "
        "Martılar çığlık çığlığa uçuşuyordu. Vapurun güvertesinde sıcak bir çay içtim; içim ısındı."
    )

    print("\n" + "#" * 80)
    print(" SAMPLE 2: NATURAL HUMAN-WRITTEN TURKISH TEXT")
    print("#" * 80)
    print(f"\nOriginal Text:\n\"{human_sample}\"\n")

    report2 = akana.audit_ai_style(human_sample)
    print(f"[*] AI Score: {report2.ai_score:.1f} / 100.0")
    print(f"[*] Verdict:  {report2.verdict}")
    print(f"[*] Findings Count: {len(report2.findings)}")
    print(f"[*] Predicates: {report2.metrics['predicates']['mektedir_count']} '-mektedir' endings")
    print(f"[*] Rhythm: Short sentences count = {report2.metrics['rhythm']['short_sentences_count']}, Mean = {report2.metrics['rhythm']['mean_sentence_length']:.1f} words")

    print("\n" + "=" * 80)
    print(" AUDIT DEMONSTRATION COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    main()
