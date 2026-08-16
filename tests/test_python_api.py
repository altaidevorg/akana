"""
Comprehensive Python unit test suite for Akana.
"""

import pytest
import akana

def test_casing_and_harmony():
    # Turkish specific casing
    assert akana.to_turkish_lower("İSTANBUL") == "istanbul"
    assert akana.to_turkish_lower("ILIK") == "ılık"
    assert akana.to_turkish_upper("istanbul") == "İSTANBUL"
    assert akana.to_turkish_upper("ılık") == "ILIK"
    assert akana.to_turkish_title("istanbul ve ankara") == "İstanbul Ve Ankara"

    # Vowel harmony
    assert akana.check_major_vowel_harmony("okul") is True
    assert akana.check_major_vowel_harmony("çiçek") is True
    assert akana.check_major_vowel_harmony("kitap") is False

    assert akana.check_minor_vowel_harmony("çocuk") is True
    assert akana.check_minor_vowel_harmony("odun") is True

def test_normalization():
    # Asciification
    assert akana.asciify("Türkçe, Çağdaş, Şiir, Ağaç") == "Turkce, Cagdas, Siir, Agac"

    # Deasciification
    assert akana.deasciify("turkce nlp cok hizli calisiyor") == "türkçe nlp çok hızlı çalışıyor"
    assert akana.deasciify("ogrenci kutuphanede kitap okuyor") == "öğrenci kütüphanede kitap okuyor"

    # Informal text normalization
    assert akana.normalize_informal("yapcam dedim ve geliyom") == "yapacağım dedim ve geliyorum"
    assert akana.normalize_informal("nooldu ya") == "ne oldu ya"
    assert akana.normalize_informal("çooookk güzel") == "çok güzel"

def test_morphology_analysis():
    morph = akana.Morphology()

    # Noun with voicing and possessive
    parses = morph.analyze("kitabım")
    assert len(parses) > 0
    top = parses[0]
    assert top["root"] == "kitap"
    assert "P1sg" in top["morphemes"]

    # Noun with vowel drop
    parses_burnum = morph.analyze("burnum")
    assert len(parses_burnum) > 0
    assert any(p["root"] == "burun" for p in parses_burnum)

    # Noun with consonant doubling
    parses_hak = morph.analyze("hakkım")
    assert len(parses_hak) > 0
    assert any(p["root"] == "hak" for p in parses_hak)

    # Verb with progressive tense
    parses_gel = morph.analyze("geliyorum")
    assert len(parses_gel) > 0
    assert parses_gel[0]["root"] == "gel"
    assert "Prog" in parses_gel[0]["morphemes"]

    # Relative clitic -ki
    parses_evdeki = morph.analyze("evdeki")
    assert len(parses_evdeki) > 0
    assert any("RelClitic" in p["morphemes"] for p in parses_evdeki)

    # Diminutive -cik
    parses_evcik = morph.analyze("evcik")
    assert len(parses_evcik) > 0
    assert any("Dim" in p["morphemes"] for p in parses_evcik)

def test_compound_decomposer():
    analyses = akana.decompose_compound("denizaltı")
    assert len(analyses) > 0
    assert analyses[0]["part1"] == "deniz"
    assert analyses[0]["part2"] == "altı"

    analyses2 = akana.decompose_compound("akbaba")
    assert len(analyses2) > 0
    assert analyses2[0]["part1"] == "ak"
    assert analyses2[0]["part2"] == "baba"

def test_dynamic_dictionary_loading():
    morph = akana.Morphology()
    morph.load_dictionary_str("blokzincir blokzincir Noun\nkuvars kuvars Noun")

    parses = morph.analyze("blokzincir")
    assert len(parses) > 0
    assert parses[0]["root"] == "blokzincir"

def test_morphology_generation():
    morph = akana.Morphology()

    # Kitabıma
    surface1 = morph.generate("kitap", ["Noun", "A3sg", "P1sg", "Dat"])
    assert surface1 == "kitabıma"

    # Burnum
    surface2 = morph.generate("burun", ["Noun", "A3sg", "P1sg"])
    assert surface2 == "burnum"

    # Hakkım
    surface3 = morph.generate("hak", ["Noun", "A3sg", "P1sg"])
    assert surface3 == "hakkım"

    # Geliyorum
    surface4 = morph.generate("gel", ["Verb", "Prog", "A1sg"])
    assert surface4 == "geliyorum"

def test_spellchecker():
    spell = akana.SpellChecker()
    assert spell.is_correct("kitap") is True
    assert spell.is_correct("ktap") is False

    suggestions = spell.suggest("ktap", max_distance=2, max_suggestions=5)
    assert len(suggestions) > 0
    assert any(s["word"] == "kitap" for s in suggestions)

def test_disambiguation():
    disambiguator = akana.Disambiguator()
    tokens = ["Ali", "güzel", "kitap", "okudu"]
    parses = disambiguator.disambiguate(tokens)

    assert len(parses) == 4
    assert parses[0]["lemma"] == "ali"
    assert parses[1]["primary_pos"] == "Adj"
    assert parses[2]["primary_pos"] == "Noun"
    assert parses[3]["primary_pos"] == "Verb"

def test_dependency_parser():
    parser = akana.DependencyParser()
    tokens = ["Ali", "güzel", "kitabı", "okudu"]
    conllu = parser.parse_conllu(tokens)

    assert "okudu" in conllu
    assert "root" in conllu
    assert "nsubj" in conllu
    assert "amod" in conllu

def test_document_pipeline():
    doc = akana.analyze("Ak Ana, Türk mitolojisinde deniz tanrıçasıdır. Ali güzel bir kitap okudu.")
    assert len(doc.sentences) == 2
    assert "Ak" in doc.sentences[0].tokens
    assert "kitap" in doc.sentences[1].tokens
    assert len(doc.sentences[0].parses) > 0

def test_readability_analysis():
    sample_text = (
        "Küçük çocuk bahçede neşeyle koşuyordu. Güneş pırıl pırıl parlıyor, "
        "kuşlar ağaçların dallarında cıvıldıyordu. Annesi onu eve çağırdı ancak "
        "çocuk oyuna devam etmek istedi."
    )
    report = akana.analyze_readability(sample_text)
    assert report.statistics.total_sentences >= 2
    assert report.statistics.total_words >= 15
    assert report.kalyoncu_formula1.score > 0
    assert len(report.kalyoncu_formula1.grade_level) > 0
    assert report.atesman.score > 0
    assert len(report.atesman.grade_level) > 0

def test_syllabification():
    assert akana.syllabify("Türkçe") == ["Türk", "çe"]
    assert akana.syllabify("araba") == ["a", "ra", "ba"]
    assert akana.hyphenate("bilgisayar") == "bil-gi-sa-yar"
    assert akana.count_syllables("öğretmenlerimiz") == 6

def test_number_conversion():
    assert akana.number_to_words(1923) == "bin dokuz yüz yirmi üç"
    assert akana.number_to_words(105) == "yüz beş"
    assert akana.number_to_words(0) == "sıfır"
    assert akana.ordinal_to_words(1) == "birinci"
    assert akana.ordinal_to_words(4) == "dördüncü"
    assert akana.currency_to_words(1250.50, "TL") == "bin iki yüz elli lira elli kuruş"
    assert akana.words_to_number("bin dokuz yüz yirmi üç") == 1923

def test_stemmer_and_stopwords():
    assert akana.stem("kitaplarımızda") in ["kitap", "kitab"]
    assert akana.is_stopword("ve") is True
    assert akana.is_stopword("bilgisayar") is False
    filtered = akana.remove_stopwords(["bu", "güzel", "kitap", "ve", "defter"])
    assert "ve" not in filtered
    assert "bu" not in filtered
    assert "kitap" in filtered

def test_named_entity_recognition():
    text = "Prof. Dr. Ahmet Yılmaz 16 Ağustos 2026 tarihinde İstanbul Üniversitesi bünyesinde 500 TL ödeme yaptı."
    entities = akana.extract_entities(text)
    assert len(entities) >= 3
    labels = [e.label for e in entities]
    assert "PER" in labels or "DATE" in labels
    assert "MONEY" in labels

def test_keyword_extraction():
    text = "Doğal dil işleme ve morfolojik analiz algoritmaları Türkçe metinlerin çözümlenmesinde büyük rol oynar."
    keywords = akana.extract_keywords(text, top_k=5)
    assert len(keywords) > 0
    assert "keyword" in keywords[0]
    assert "score" in keywords[0]

def test_summarization():
    text = (
        "Türkiye Cumhuriyeti 1923 yılında kuruldu. Başkenti Ankara'dır. "
        "Türkçe, Türk dilleri ailesine ait sondan eklemeli zengin bir dildir. "
        "Günümüzde milyonlarca insan tarafından konuşulmaktadır."
    )
    summary = akana.summarize(text, max_sentences=2)
    assert len(summary) == 2


