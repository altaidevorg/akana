"""
Example 05: Modern Turkish Readability Analysis (Kalyoncu 2025 vs Classic Metrics)
"""

import akana

def display_report(title: str, text: str):
    print("=" * 75)
    print(f"  {title.upper()}")
    print("=" * 75)
    preview = text.strip()[:220] + "..." if len(text.strip()) > 220 else text.strip()
    print(f"Sample Text Preview:\n\"{preview}\"\n")

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
    # 1. Primary School Fable (Thesis Benchmark Text: İki Horoz - 3. & 4. Sınıf Düzeyi)
    iki_horoz_story = """
    Bir ormanın kıyısında, büyük bir çiftlik varmış. Bu çiftlikte pek çok hayvan yetiştirilirmiş. 
    Atlar, eşekler, inekler ve kümesteki tüm hayvanlar dostça geçinirlermiş. Birbirleriyle hiç kavga etmezlermiş. 
    Kümesteki hayvanların başkanı yaşlı bir horozmuş. Bu horoz çok adaletliymiş. Tüm hayvanlara dostça davranırmış. 
    Herkesin hakkına saygı gösterir, kimseyi incitmezmiş. Çiftliğin sahibi bir sabah kümesin kapısını açmış. 
    Yanında yeni bir horoz varmış. Diğerlerinin meraklı bakışları arasında onu kümese bırakmış. 
    Herkes yeni gelenin çevresinde toplanmış. Ona “Aramıza hoş geldin.” demişler. Ancak yeni horoz çok kibirliymiş. 
    Kendini diğer hayvanlardan üstün görürmüş. Kimseyle konuşmaz, kimseye selam vermezmiş. 
    Eski horoz onun bu davranışlarına çok üzülmüş. Yanına gidip tatlı bir dille konuşmak istemiş. 
    Ona dostluğun ve paylaşmanın önemini anlatmış. Kibirli horoz bu sözlere hiç aldırmamış. 
    Zaman geçtikçe çiftlikteki hayvanlar kibirli horozdan uzaklaşmışlar. Sonunda yalnız kalan horoz hatasını anlamış.
    """

    # 2. Academic / Scientific Article (Lisans & Lisansüstü Düzeyi)
    academic_paper = """
    Türkçenin morfolojik karmaşıklığı ve sondan eklemeli yapısı, doğal dil işleme modellerinde 
    sözcük kökü ile biçimbirim dizilimlerinin derinlemesine ayrıştırılmasını zorunlu kılmaktadır. 
    Bu bağlamda geliştirilen algoritmalar, bağlamsal çokanlamlılığı giderme süreçlerinde 
    olasılıksal geçiş matrisleri ve sentaktik bağımlılık ağaçlarından istifade etmektedir. 
    Geleneksel eğitim anlayışında eğiticinin anlattıklarını ya da gösterdiklerini öğrenmekle 
    yükümlü olan öğrenci, günümüzde bilgiye kendisi erişen ve bilgiyi yapılandıran bir profile dönüşmüştür. 
    Bu epistemolojik dönüşüm, metinlerin okunabilirlik ve anlaşılabilirlik düzeylerinin belirlenmesinde 
    çok değişkenli istatistiksel parametrelerin kullanılmasını zorunlu hale getirmiştir.
    """

    # 3. Daily News & Journalistic Report (Orta Düzey / 7-8. Sınıf)
    news_report = """
    Ulaştırma ve Altyapı Bakanlığı, şehirler arası demiryolu ağını modernize etmek amacıyla 
    hazırlanan yeni yatırım programını kamuoyuna açıkladı. Proje kapsamında mevcut demiryolu hatları 
    yenilenecek ve yüksek hızlı tren seferleri artırılacak. Yapılan açıklamada Ankara ve İzmir 
    arasındaki seyahat süresinin önemli ölçüde kısalacağı vurgulandı. Çalışmaların belirlenen takvime 
    uygun olarak devam ettiği ve projenin önümüzdeki yıl tamamlanacağı bildirildi.
    """

    display_report("Sample 1: Primary School Story (İki Horoz - 3 ve 4. Sınıf)", iki_horoz_story)
    display_report("Sample 2: Academic Paper (Lisans & Lisansüstü Düzeyi)", academic_paper)
    display_report("Sample 3: News Report (Genel Okur & Ortaokul Düzeyi)", news_report)

if __name__ == "__main__":
    main()
