"""
Akana Turkish NLP Suite Example
Demonstrates:
1. TDK-compliant Syllabification & Hyphenation
2. Turkish Number Converter (Cardinals, Ordinals, Currencies, Words-to-Number)
3. Morphological Stemmer & Stopword Filtering
4. Named Entity Recognition (PER, LOC, ORG, DATE, MONEY, PERCENT)
5. Keyword & Keyphrase Extraction (Turkish RAKE)
6. Morphological Extractive Text Summarization (TextRank)
7. Morphology-Guided Spell Checking & De-asciification
"""

import akana

def main():
    print("=" * 70)
    print(" AKANA TURKISH NLP SUITE DEMO")
    print("=" * 70)

    # 1. Syllabification & Hyphenation
    print("\n--- 1. SYLLABIFICATION & HYPHENATION (Heceleme) ---")
    words = ["Türkçe", "bilgisayar", "öğretmenlerimiz", "ilkokul", "cumhuriyet"]
    for w in words:
        sylls = akana.syllabify(w)
        hyphenated = akana.hyphenate(w, "-")
        count = akana.count_syllables(w)
        print(f"Word: {w:<16} -> Syllables: {sylls} (Count: {count}, Hyphenated: '{hyphenated}')")

    # 2. Number Conversion
    print("\n--- 2. TURKISH NUMBER CONVERSION ---")
    nums = [1923, 105, 1000, 2026, 1250500]
    for n in nums:
        cardinal = akana.number_to_words(n)
        ordinal = akana.ordinal_to_words(n)
        print(f"Number {n:<10} -> Cardinal: '{cardinal}' | Ordinal: '{ordinal}'")

    currency_val = 1450.75
    print(f"Currency {currency_val} TL -> '{akana.currency_to_words(currency_val, 'TL')}'")
    print(f"Currency 250.50 USD -> '{akana.currency_to_words(250.50, 'USD')}'")

    parsed_num = akana.words_to_number("bin dokuz yüz yirmi üç")
    print(f"Words 'bin dokuz yüz yirmi üç' -> Integer: {parsed_num}")

    # 3. Stemmer & Stopwords
    print("\n--- 3. STEMMER & STOPWORD FILTERING ---")
    sample_tokens = ["bu", "güzel", "kitaplarımızda", "ve", "defterlerimizde", "yazılan", "bilgiler"]
    cleaned_tokens = akana.remove_stopwords(sample_tokens)
    stems = [akana.stem(t) for t in cleaned_tokens]
    print(f"Original tokens: {sample_tokens}")
    print(f"After stopword filtering: {cleaned_tokens}")
    print(f"Morphological stems: {stems}")

    # 4. Named Entity Recognition (NER)
    print("\n--- 4. NAMED ENTITY RECOGNITION (NER) ---")
    ner_text = (
        "Sayın Prof. Dr. Ahmet Yılmaz, 16 Ağustos 2026 tarihinde İstanbul Üniversitesi "
        "bünyesindeki konferansta konuştu. Etkinlik için 500 TL katılım ücreti belirlendi "
        "ve katılımcıların yüzde 80'i onayladı."
    )
    entities = akana.extract_entities(ner_text)
    print(f"Text:\n\"{ner_text}\"\n")
    print(f"Detected Entities ({len(entities)}):")
    for e in entities:
        print(f"  • [{e.label:<7}] '{e.text}' (Offset: {e.start}-{e.end})")

    # 5. Keyword Extraction (RAKE + Morphology)
    print("\n--- 5. TURKISH KEYWORD EXTRACTION ---")
    doc_text = (
        "Doğal dil işleme ve morfolojik analiz algoritmaları, Türkçe metinlerin "
        "doğru çözümlenmesinde ve yapay zeka sistemlerinin geliştirilmesinde kritik rol oynar. "
        "Sondan eklemeli dil yapısı nedeniyle Türkçe morfolojik çözümleme yüksek başarım gerektirir."
    )
    keywords = akana.extract_keywords(doc_text, top_k=5)
    print(f"Text:\n\"{doc_text}\"\n")
    print("Top Keywords & Keyphrases:")
    for kw in keywords:
        print(f"  • {kw['keyword']:<35} (Score: {kw['score']:.2f})")

    # 6. Extractive Summarization (TextRank)
    print("\n--- 6. EXTRACTIVE TEXT SUMMARIZATION (TextRank) ---")
    long_doc = (
        "Türkiye Cumhuriyeti 1923 yılında Gazi Mustafa Kemal Atatürk önderliğinde kuruldu. "
        "Ankara, Türkiye'nin başkenti ve idari merkezidir. "
        "Türkçe, zengin morfolojik yapısıyla dünya dilleri arasında özel bir yere sahiptir. "
        "Akana, modern ve yüksek hızlı bir Türkçe doğal dil işleme kütüphanesidir. "
        "Gelişmiş algoritmaları sayesinde morfolojik analiz, sözdizimsel ayrıştırma ve okunabilirlik "
        "ölçümünü milisaniyeler içinde gerçekleştirir."
    )
    summary = akana.summarize(long_doc, max_sentences=2)
    print("Original Text (5 sentences):")
    print(long_doc)
    print("\nExtracted Summary (Top 2 sentences):")
    for idx, s in enumerate(summary, 1):
        print(f"  {idx}. {s}")

    # 7. Morphology-Guided Spellcheck & De-asciification
    print("\n--- 7. MORPHOLOGY-GUIDED SPELLCHECK & DE-ASCIIFICATION ---")
    dirty_ascii = "turkce nlp kutuphanede cok hizli calisiyor ve ogrenci kitap okuyor"
    clean_turkish = akana.deasciify(dirty_ascii)
    print(f"ASCII input:    '{dirty_ascii}'")
    print(f"Deasciified:    '{clean_turkish}'")

    spell = akana.SpellChecker()
    typo_word = "ktap"
    is_valid = spell.is_correct("kitaplarımız")
    print(f"Is 'kitaplarımız' valid word? {is_valid}")
    suggestions = spell.suggest(typo_word, max_distance=2, max_suggestions=3)
    print(f"Suggestions for '{typo_word}': {[s['word'] for s in suggestions]}")

    print("\n" + "=" * 70)
    print(" ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()
