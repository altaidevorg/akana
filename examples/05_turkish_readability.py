"""
Example 05: Modern Turkish Readability Analysis (Kalyoncu 2025 vs Classic Metrics)
"""

import akana

def display_report(title: str, text: str):
    print("=" * 70)
    print(f"  {title.upper()}")
    print("=" * 70)
    print(f"Sample Text:\n\"{text.strip()}\"\n")

    report = akana.analyze_readability(text)
    s = report.statistics

    print("--- 1. LINGUISTIC & STRUCTURAL VARIABLES ---")
    print(f"  • Total Sentences:              {s.total_sentences}")
    print(f"  • Total Words:                  {s.total_words}")
    print(f"  • Total Syllables:              {s.total_syllables}")
    print(f"  • OCU (Avg Words/Sentence):     {s.words_per_sentence:.2f}")
    print(f"  • OSU (Avg Syllables/Word):     {s.syllables_per_word:.2f}")
    print(f"  • KL  (Unfamiliar Word %):      {s.unfamiliar_word_ratio:.2f}% ({s.unfamiliar_words} / {s.total_words})")
    print(f"  • TTR (Type-Token Ratio %):     {s.type_token_ratio:.2f}% ({s.single_occurrence_words} unique stems)")
    print(f"  • KCO (Complex Sentence %):     {s.complex_sentence_ratio:.2f}% ({s.complex_sentences} sentences)")
    print(f"  • YCO (Fiilimsi / Clause Rate): {s.clause_ratio:.2f} ({s.total_fiilimsiler} fiilimsiler)")

    print("\n--- 2. KALYONCU (2025) READABILITY SUITE ---")
    print(f"  [1] Formula 1 (Comprehensive, R²=0.99) : Score = {report.kalyoncu_formula1.score:>6.2f} -> {report.kalyoncu_formula1.grade_level}")
    print(f"  [2] Formula 2 (Standard, R²=0.82)      : Score = {report.kalyoncu_formula2.score:>6.2f} -> {report.kalyoncu_formula2.grade_level}")
    print(f"  [3] Formula 3 (Clause-based, R²=0.83)  : Score = {report.kalyoncu_formula3.score:>6.2f} -> {report.kalyoncu_formula3.grade_level}")
    print(f"  [4] Formula 4 (Practical, R²=0.78)     : Score = {report.kalyoncu_formula4.score:>6.2f} -> {report.kalyoncu_formula4.grade_level}")

    print("\n--- 3. COMPARATIVE CLASSIC FORMULAS ---")
    print(f"  • Ateşman (1997)                       : Score = {report.atesman.score:>6.2f} -> {report.atesman.grade_level}")
    print(f"  • Çetinkaya-Uzun (2010)                : Score = {report.cetinkaya_uzun.score:>6.2f} -> {report.cetinkaya_uzun.grade_level}")
    print(f"  • Bezirci-Yılmaz (2010)                : Score = {report.bezirci_yilmaz.score:>6.2f} -> {report.bezirci_yilmaz.grade_level}")
    print("\n")

def main():
    # 1. Primary School Level Children's Story
    children_story = """
    Küçük çocuk bahçede neşeyle koşuyordu. Güneş pırıl pırıl parlıyor, 
    kuşlar ağaçların dallarında cıvıldıyordu. Annesi onu eve çağırdı ancak 
    çocuk oyuna devam etmek istedi. Bütün gün arkadaşlarıyla oynadıktan 
    sonra yorgun bir şekilde eve döndü ve mutlu bir uykuya daldı.
    """

    # 2. Academic / Scientific Thesis Abstract
    academic_abstract = """
    Türkçenin morfolojik karmaşıklığı ve sondan eklemeli yapısı, doğal dil işleme 
    modellerinde sözcük kökü ile biçimbirim dizilimlerinin derinlemesine ayrıştırılmasını 
    zorunlu kılmaktadır. Bu bağlamda geliştirilen algoritmalar, bağlamsal çokanlamlılığı 
    giderme süreçlerinde olasılıksal geçiş matrisleri ve sentaktik bağımlılık ağaçlarından 
    istifade etmektedir. İstatiksel modelleme neticesinde elde edilen parametreler, metinlerin 
    okunabilirlik düzeylerinin tayininde yüksek belirleyicilik sergilemektedir.
    """

    # 3. News / Journalistic Text
    news_text = """
    Ulaştırma Bakanlığı, yüksek hızlı tren hatlarının genişletilmesi amacıyla yeni bir 
    altyapı projesini duyurdu. Yapılan açıklamaya göre Ankara ile İzmir arasındaki seyahat süresi 
    üç buçuk saate inecek. Projenin önümüzdeki yıl tamamlanarak halkın hizmetine sunulması planlanıyor.
    """

    display_report("Sample 1: Children's Story (İlkokul / Ortaokul Düzeyi)", children_story)
    display_report("Sample 2: Academic Paper (Lisans / Lisansüstü Düzeyi)", academic_abstract)
    display_report("Sample 3: Daily News Article (Genel Okur Düzeyi)", news_text)

if __name__ == "__main__":
    main()
