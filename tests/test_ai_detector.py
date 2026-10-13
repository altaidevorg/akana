"""The AI-text detector in the style auditor (`audit_ai_style(text).ai_detection`) and the
detector-driven `humanize_prompt`."""

import akana

# Sample texts from the detector's evaluation data, not written by hand: WildChat answers
# (ODC-BY) and Turkish Wikisource texts (public domain).
# WildChat (ODC-BY), o1-preview-2024-09-12 answer, id 151b38df05e571c8
AI_TEXT = 'En katma değerli ve ölçeklenebilir kompozit malzemeler arasında **karbon fiber takviyeli polimerler (CFRP)** öne çıkmaktadır. CFRP, yüksek mukavemet/ağırlık oranı, mükemmel yorulma direnci ve korozyon direnci gibi üstün mekanik özellikleri sayesinde havacılık ve uzay sanayii, otomotiv, rüzgar enerjisi ve spor ekipmanları gibi çeşitli endüstrilerde yaygın olarak kullanılmaktadır.\n\nBu malzemelerin üretimi, reçine aktarım kalıplama (RTM) ve otoklav laminasyonu gibi ölçeklenebilir üretim teknikleriyle gerçekleştirilebilir. Ayrıca, otomasyon ve robotik teknolojilerinin entegrasyonu ile üretim süreçleri daha verimli hale getirilebilir, bu da büyük ölçekli üretime olanak tanır.\n\nKarbon fiber kompozitlerin yüksek katma değeri, ürünlerin performansını artırırken ağırlığı azaltması ve enerji verimliliğine katkıda bulunmasından kaynaklanmaktadır. Bu özellikler, özellikle yakıt tasarrufu ve azaltılmış emisyonların önemli olduğu sektörlerde büyük ekonomik ve çevresel faydalar sağlamaktadır.'

# Turkish Wikisource (public domain), a folk poem, id 984244ffc4199c8c
POEM = 'İki keklik bir kayada ötüyor\nÖtme de keklik derdim bana yetiyor,\nAman güzelim yetiyor.\nBu hasretlik bu ayrılık yetiyor\nYazma da müstantik uğraşırım kız benim\nAman güzelim kız benim\nMantosu oymalı iskarpini boyalı kız benim\nAman güzelim kız benim\nİki de keklik bir kayada su içer\nDertli de kerem dertsizlere dert açar\nAman güzelim dert açar\nBuna da asrı sevda derler tez geçer\nYazma da müstantik uğraşırım kız benim\nAman güzelim kız benim\nMantosu oymalı iskarpini boyalı kız benim\nAman güzelim kız benim'

# WildChat (ODC-BY), gpt-4o-2024-08-06 answer the academic rule flags, id d3d3a89f0be18699
ACADEMIC_CUES_TEXT = 'Elbette, Chain-of-Thought Prompting yöntemiyle adım adım inceleyelim:\n\n1. **"Narrator" kelimesi**: \n   - "Narrator" kelimesindeki harflere bakalım: N, a, r, r, a, t, o, r.\n   - Burada üç tane \'r\' harfi var.\n\n2. **"Strawberry" kelimesi**:\n   - "Strawberry" kelimesindeki harflere bakalım: S, t, r, a, w, b, e, r, r, y.\n   - Burada da üç tane \'r\' harfi var.\n\n3. **"Kelimeler arasındaki toplam \'r\' sayısı"**: \n   - "Narrator" kelimesinde 3 \'r\' + "Strawberry" kelimesinde 3 \'r\' = Toplam 6 \'r\' harfi bulunuyor.\n\nSonuç olarak, "Narrator" ve "Strawberry" kelimelerindeki toplam \'r\' harfi sayısı 6.'

# Turkish Wikisource (public domain), id 176742819e0e7a3f: a Nasreddin Hoca story.
HUMAN_TEXT = "Nasreddin Hoca'nın bir oğlu varmış. Ne denirse tam tersini yaparmış. Bir gün Nasreddin Hoca ile oğlu Akşehir'e gidip oradan iki çuval tuz almışlar. Eşeğe yükleyip köylerine dönmek için yola koyulmuşlar. Köylerinin yakınındaki dereye geldiklerinde, Nasreddin Hoca, derenin en sığ yerinden karşı tarafa geçmiş. Oğlu da eşeği sudan geçirmeye çalışıyormuş. Ama eşeğin sırtındaki çuvallardan biri, neredeyse suya değmek üzereymiş. Hoca, oğlunun ters huyunu bildiği için seslenmiş:\n– Sevgili oğlum. Çuvallardan biri suya değecek. Biraz daha asıl da, iyice suya gömülsün.\nNe var ki çocuğun o gün uysallığı üzerindeymiş. Tutup, babasının dediğini yapmış. Neticede iki çuval birden suyun içine düşmüş. Nasreddin Hoca:\n– Ne yapıyorsun şaşkın oğlum! diye bağırmış, ama bu arada tuzlar da suda eriyip gitmiş.\nNasreddin Hoca'nın oğlu, babasının kızgınlığı karşısında şöyle söyleniyormuş:\n– Kırk yılda bir baba sözü dinleyelim dedik, gene de yaranamadık."


def detect(text):
    return akana.audit_ai_style(text).ai_detection


def test_ai_answer_is_ai():
    d = detect(AI_TEXT)
    assert d["verdict"] == "ai"
    assert d["threshold"] <= d["ai_probability"] <= 1.0
    assert len(d["patterns"]) >= 3
    assert d["signals"] and {"key", "name", "value", "contribution"} <= set(d["signals"][0])
    assert d["report"].startswith("# AI tespit raporu")
    assert "kanıt değildir" in d["report"]


def test_human_story_is_human():
    d = detect(HUMAN_TEXT)
    assert d["verdict"] == "human"
    assert d["ai_probability"] < d["uncertain_low"]


def test_rule_based_score_is_kept_next_to_the_model():
    report = akana.audit_ai_style(AI_TEXT)
    assert 0.0 <= report.ai_score <= 100.0
    assert report.verdict
    assert report.ai_detection["verdict"] == "ai"


def test_no_decision_without_enough_words_or_for_poems():
    for text in ["", "Elbette, yardımcı olabilirim.", POEM]:
        d = detect(text)
        assert d["verdict"] in {"too_short", "out_of_scope"}
        assert d["ai_probability"] is None
        assert d["signals"] == [] and d["patterns"] == []
    assert detect(POEM)["scope"]["kind"] == "poem"


def test_academic_cues_only_warn():
    d = detect(ACADEMIC_CUES_TEXT)
    assert d["scope"]["kind"] == "academic"
    assert d["verdict"] != "out_of_scope"


def test_humanize_prompt_uses_the_detector_for_ai_text():
    d = detect(AI_TEXT)
    prompt = akana.humanize_prompt(AI_TEXT)
    assert "**AI tespiti:**" in prompt and "Hedef Register" in prompt
    assert "Tespit Edilen AI Skoru" not in prompt
    assert d["patterns"][0]["pattern"] in prompt
    # Texts the detector does not judge AI keep the rule-based prompt.
    human = akana.humanize_prompt(HUMAN_TEXT)
    assert "Tespit Edilen AI Skoru" in human and "AI tespiti" not in human
