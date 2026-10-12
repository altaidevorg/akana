"""Tests of the AI-text detector: akana.ai_text_detect and akana.humanize."""

import math

import akana

# Sample texts taken from the frozen golden sets (akana-data/golden), not written by hand.
# OpenAI golden v1, wildchat_tr (o1-preview-2024-09-12), id 151b38df05e571c8
AI_TEXT = 'En katma değerli ve ölçeklenebilir kompozit malzemeler arasında **karbon fiber takviyeli polimerler (CFRP)** öne çıkmaktadır. CFRP, yüksek mukavemet/ağırlık oranı, mükemmel yorulma direnci ve korozyon direnci gibi üstün mekanik özellikleri sayesinde havacılık ve uzay sanayii, otomotiv, rüzgar enerjisi ve spor ekipmanları gibi çeşitli endüstrilerde yaygın olarak kullanılmaktadır.\n\nBu malzemelerin üretimi, reçine aktarım kalıplama (RTM) ve otoklav laminasyonu gibi ölçeklenebilir üretim teknikleriyle gerçekleştirilebilir. Ayrıca, otomasyon ve robotik teknolojilerinin entegrasyonu ile üretim süreçleri daha verimli hale getirilebilir, bu da büyük ölçekli üretime olanak tanır.\n\nKarbon fiber kompozitlerin yüksek katma değeri, ürünlerin performansını artırırken ağırlığı azaltması ve enerji verimliliğine katkıda bulunmasından kaynaklanmaktadır. Bu özellikler, özellikle yakıt tasarrufu ve azaltılmış emisyonların önemli olduğu sektörlerde büyük ekonomik ve çevresel faydalar sağlamaktadır.'

# Claude golden v2, Turkish Wikisource poem (public domain), id 984244ffc4199c8c
POEM = 'İki keklik bir kayada ötüyor\nÖtme de keklik derdim bana yetiyor,\nAman güzelim yetiyor.\nBu hasretlik bu ayrılık yetiyor\nYazma da müstantik uğraşırım kız benim\nAman güzelim kız benim\nMantosu oymalı iskarpini boyalı kız benim\nAman güzelim kız benim\nİki de keklik bir kayada su içer\nDertli de kerem dertsizlere dert açar\nAman güzelim dert açar\nBuna da asrı sevda derler tez geçer\nYazma da müstantik uğraşırım kız benim\nAman güzelim kız benim\nMantosu oymalı iskarpini boyalı kız benim\nAman güzelim kız benim'

# OpenAI golden v1, wildchat_tr (gpt-4o-2024-08-06), id d3d3a89f0be18699: a chat answer the academic rule flags
ACADEMIC_CUES_TEXT = 'Elbette, Chain-of-Thought Prompting yöntemiyle adım adım inceleyelim:\n\n1. **"Narrator" kelimesi**: \n   - "Narrator" kelimesindeki harflere bakalım: N, a, r, r, a, t, o, r.\n   - Burada üç tane \'r\' harfi var.\n\n2. **"Strawberry" kelimesi**:\n   - "Strawberry" kelimesindeki harflere bakalım: S, t, r, a, w, b, e, r, r, y.\n   - Burada da üç tane \'r\' harfi var.\n\n3. **"Kelimeler arasındaki toplam \'r\' sayısı"**: \n   - "Narrator" kelimesinde 3 \'r\' + "Strawberry" kelimesinde 3 \'r\' = Toplam 6 \'r\' harfi bulunuyor.\n\nSonuç olarak, "Narrator" ve "Strawberry" kelimelerindeki toplam \'r\' harfi sayısı 6.'


def test_analysis_fields():
    a = akana.ai_text_detect(AI_TEXT)
    assert 0.0 <= a["ai_probability"] <= 1.0
    assert a["verdict"] in {"ai", "uncertain", "human", "too_short", "out_of_scope"}
    assert len(a["features"]) == 75
    total = a["bias"] + sum(f["contribution"] for f in a["features"])
    assert math.isclose(total, a["raw_score"], abs_tol=1e-6)
    assert a["verdict"] == "ai"
    assert len(a["patterns"]) >= 3
    assert a["tokens"] and "score" in a["tokens"][0]


def test_short_and_out_of_scope():
    assert akana.ai_text_detect("Elbette, yardımcı olabilirim.")["verdict"] == "too_short"
    a = akana.ai_text_detect(POEM)
    assert a["verdict"] == "out_of_scope"
    assert a["scope"]["kind"] == "şiir"
    assert akana.ai_text_detect(AI_TEXT)["scope"] is None
    # academic text is only flagged, the decision is still made
    a = akana.ai_text_detect(ACADEMIC_CUES_TEXT)
    assert a["scope"]["kind"] == "akademik"
    assert a["verdict"] in {"ai", "uncertain", "human"}


def test_report_is_turkish_markdown():
    r = akana.ai_text_detect(AI_TEXT)["report"]
    assert r.startswith("# AI tespit raporu")
    assert "kanıt değildir" in r


def test_humanize_pipeline():
    r = akana.humanize(AI_TEXT)
    assert set(r) == {"analysis", "prompt"}
    if r["analysis"]["verdict"] == "ai":
        assert "İşlenecek Metin" in r["prompt"]
        assert r["analysis"]["patterns"][0]["pattern"] in r["prompt"]
    else:
        assert r["prompt"] is None
    short = akana.humanize("Kısa bir metin.")
    assert short["prompt"] is None
    # the style-auditor-only prompt is unchanged
    assert "İşlenecek Metin" in akana.humanize_prompt(AI_TEXT)


def test_public_api():
    assert {"ai_text_detect", "humanize"} <= set(akana.__all__)
    assert not any(n.startswith("_ai") for n in akana.__all__)
