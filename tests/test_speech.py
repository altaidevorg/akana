"""The phonemiser's Python API: `phonemize`, `phonemize_detailed` and `verbalize`.

`tts-v1` is checked against its frozen golden file (EchoCraft's shipped voice was trained on it),
`tts-v2` against its own.
"""

import json
from pathlib import Path

import pytest

import akana

DATA = Path(__file__).resolve().parents[1] / "crates/akana-core/tests/data/speech"


def test_default_profile_is_tts_v1():
    assert akana.phonemize("Dağ.") == akana.phonemize("Dağ.", profile="tts-v1")
    assert akana.phonemize_detailed("Dağ.")["profile"] == "tts-v1"


def test_unknown_profile_raises():
    for name in ["literature-v9", "espeak-compat"]:
        with pytest.raises(ValueError):
            akana.phonemize("merhaba", profile=name)


# --- profile tts-v1 ---------------------------------------------------------------------------

TTS_TYPES = {"vowel", "stop", "vstop", "fricative", "vfricative", "nasal", "liquid", "pause"}


def _vowel_levels(tokens):
    """Per word, the stress levels of its vowels."""
    words, cur = [], None
    for t in tokens:
        if t["type"] == "pause":
            continue
        if t["word"]:
            cur = []
            words.append(cur)
        if t["type"] == "vowel" and cur is not None:
            cur.append(t["stress"])
    return words


def test_tts_v1_schema_and_edges():
    tokens = akana.phonemize("Kitabı okudum ama beğenmedim.", profile="tts-v1")
    assert tokens[0]["ph"] == "_" and tokens[-1]["ph"] == "_"
    for t in tokens:
        assert set(t) == {"ph", "stress", "word", "lengthen", "mark", "type", "ipa"}
        assert t["type"] in TTS_TYPES
        assert 0 <= t["stress"] <= 7
        assert (t["ipa"] is None) == (t["type"] == "pause")
        if t["ipa"] is not None:
            # the aligner writes a long ı as ɯɯ (it has no ɯː)
            assert t["lengthen"] == (t["ipa"].endswith("ː") or t["ipa"] == "ɯɯ")
    # a short pause before the conjunction, and the clause's punctuation on every token
    assert [t["ph"] for t in tokens if t["type"] == "pause"] == ["_", "_!", "_"]
    assert {t["mark"] for t in tokens} == {"."}


def test_tts_v1_stress_from_the_literature():
    def levels(text):
        return _vowel_levels(akana.phonemize(text, profile="tts-v1"))

    # pre-accenting negation: GELmedi; the adverb before it carries the nucleus
    assert levels("Hâlâ gelmedi.") == [[7, 1], [2, 1, 1]]
    # the question clitic is unstressed and its host carries the nucleus
    assert levels("Serkan mı bavulu hazırladı?")[:2] == [[1, 7], [0]]
    # Sezer's rule for a place name, initial stress of an adverb
    lv = levels("Ankara'ya gittik, sonra döndük.")
    assert lv[0][0] == 7 and lv[2] == [7, 1]


def test_tts_v1_one_nucleus_per_clause():
    for text in ["Bugün hava çok güzel.", "Eve geldim, yemek yedim ve uyudum.", "Ne zaman gelecek?"]:
        d = akana.phonemize_detailed(text, profile="tts-v1")
        clauses = {}
        for w in d["words"]:
            clauses.setdefault(w["clause"], []).append(w["prominence"])
        for levels in clauses.values():
            assert levels.count(7) == 1, (text, levels)


def test_tts_v1_long_vowels_and_geminates():
    ipa = [t["ipa"] for t in akana.phonemize("Hâlâ hissetmek, kalıbı.", profile="tts-v1") if t["ipa"]]
    assert ipa[:4] == ["h", "aː", "l", "aː"]
    # the aligner has tː but no sː: a doubled s stays two tokens
    assert ipa[4:11] == ["h", "ɪ", "s", "s", "ɛ", "t", "m"]
    toks = akana.phonemize("Sıhhat, dikkat, ağaç.", profile="tts-v1")
    # a geminate closes the syllable before it: dɪk.kat, lax ɪ
    assert [t["ipa"] for t in toks if t["ipa"]][6:11] == ["d", "ɪ", "kː", "a", "t"]


def test_tts_v1_detailed_provenance():
    text = "Ankara'ya gittik."
    d = akana.phonemize_detailed(text, profile="tts-v1")
    assert d["profile"] == "tts-v1" and d["version"].startswith("tts-v1.")
    assert d["words"][0]["stress_source"] == "sezer"
    for w in d["words"]:
        assert w["stress_source"] and w["segment_source"]
        assert 0 <= w["prominence"] <= 7
        assert text[w["start"]:w["end"]] == w["text"]
    assert d["tokens"] == akana.phonemize(text, profile="tts-v1")


def test_tts_v1_odd_input_never_raises():
    for text in ["", "...", "̂", "â", "-i", "’", "😀 merhaba", "x" * 1000, "(ve)", "\"mı\""]:
        tokens = akana.phonemize(text, profile="tts-v1")
        assert tokens[0]["ph"] == "_" and tokens[-1]["ph"] == "_"
        assert akana.phonemize(text, profile="tts-v1") == tokens


TTS_GOLDEN = DATA / "tts_v1_golden.jsonl"
TTS_FIELDS = ("ph", "stress", "word", "lengthen", "mark", "type", "ipa")


def test_tts_v1_golden_is_frozen():
    rows = [json.loads(line) for line in TTS_GOLDEN.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 60
    version = akana.phonemize_detailed("a", profile="tts-v1")["version"]
    for row in rows:
        assert row["version"] == version
        got = [[t[f] for f in TTS_FIELDS] for t in akana.phonemize(row["text"], profile="tts-v1")]
        assert got == row["tokens"], row["text"]


# --- verbaliser -------------------------------------------------------------------------------


def test_verbalize_reads_what_a_screen_reader_meets():
    cases = {
        "Toplantı 3'te başlıyor.": "Toplantı üçte başlıyor.",
        "Fiyat ₺850'ye düştü.": "Fiyat sekiz yüz elli liraya düştü.",
        "Saat 14:30'da": "Saat on dört otuzda",
        "12.03.2024 tarihinde": "on iki mart iki bin yirmi dört tarihinde",
        "%25'lik artış": "yüzde yirmi beşlik artış",
        "hız 90 km/sa": "hız saatte doksan kilometre",
        "II. Dünya Savaşı": "ikinci Dünya Savaşı",
        "TBMM'de": "tebememede",
        "2/3'ü": "üçte ikisi",
        "Dr. Ahmet Bey, vb. şeyler": "doktor Ahmet Bey, ve benzeri şeyler",
    }
    for text, spoken in cases.items():
        assert akana.verbalize(text)["text"] == spoken, text


def test_verbalize_keeps_spans():
    text = "Fiyat ₺850'ye düştü."
    v = akana.verbalize(text)
    (seg,) = v["segments"]
    assert text[seg["start"]:seg["end"]] == "₺850'ye" == seg["source"]
    assert text.encode()[seg["start_byte"]:seg["end_byte"]].decode() == "₺850'ye"
    assert seg["spoken"] == "sekiz yüz elli liraya" and seg["kind"] == "money"
    # every word the phonemiser reads from the amount points back at it
    d = akana.phonemize_detailed(text, profile="tts-v1")
    amount = [w for w in d["words"] if w["normalized"] == "money"]
    assert [w["reading"] for w in amount] == ["sekiz", "yüz", "elli", "liraya"]
    assert all(text[w["start"]:w["end"]] == "₺850'ye" for w in amount)


def test_tts_v1_keeps_words_written_in_letters():
    # tts-v1's output on plain text is frozen: abbreviations and foreign words stay as written
    assert akana.verbalize("vb. online", words=False)["segments"] == []
    assert akana.verbalize("vb. online")["text"] == "ve benzeri onlayn"
    plain = "Sınıfsal, etnik, dinsel, politik, vb. farklılıklar bir kenara itilir."
    assert akana.phonemize_detailed(plain, profile="tts-v1")["words"][4]["normalized"] is None


# --- profile tts-v2 ---------------------------------------------------------------------------


def test_tts_v2_golden_is_frozen():
    rows = [json.loads(line) for line in (DATA / "tts_v2_golden.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 100
    version = akana.phonemize_detailed("a", profile="tts-v2")["version"]
    for row in rows:
        assert row["version"] == version
        got = [[t[f] for f in TTS_FIELDS] for t in akana.phonemize(row["text"], profile="tts-v2")]
        assert got == row["tokens"], row["text"]


def test_tts_v2_writes_ipa_and_the_literature_segments():
    tokens = akana.phonemize("Kardeşim erken bir ekmek aldı.", profile="tts-v2")
    for t in tokens:
        assert t["ph"].startswith("_") == (t["type"] == "pause")
        if t["ipa"] is not None:
            assert t["ipa"].startswith(t["ph"]) or t["ipa"] == "ɯɯ"
    ph = " ".join(t["ph"] for t in tokens)
    # /ɾ/ a tap except before a pause; palatal k next to a front vowel; e as æ before a coda r;
    # a final ı is not lowered
    assert ph.startswith("_ k a ɾ d e ʃ i m")
    assert "æ ɾ c æ n" in ph and "b i ɾ" in ph and "e c m e c" in ph
    assert ph.endswith("a ɫ d ɯ _")


def test_tts_v2_stress_rules_tts_v1_lacked():
    def stressed(text, profile):
        w = akana.phonemize_detailed(text, profile=profile)["words"][0]
        vowels = [p for p in w["phones"] if p["ph"] in "aeiouyɛæɪʊɯœøɔEIOUYW@&"]
        return max(range(len(vowels)), key=lambda k: (vowels[k]["stress"], -k))

    # nerede (Dursunoğlu 2006), seninki (Kabak & Vogel 2001), yapıver (-Iver), Hindistan (-istan)
    assert stressed("Nerede?", "tts-v1") == 2 and stressed("Nerede?", "tts-v2") == 0
    assert stressed("Seninki geldi.", "tts-v2") == 1
    assert stressed("Yapıver.", "tts-v2") == 1
    assert stressed("Hindistan'a gitti.", "tts-v2") == 2


# --- review of 2026-11-08 ---------------------------------------------------------------------


def test_review_regressions():
    import unicodedata

    # non-ASCII digits: no panic, edge pauses kept
    for text in ["1١11", "1２３４", "12.03.٢٠٢٤", "Tarih: 1.1.٢٠"]:
        for profile in ["tts-v1", "tts-v2"]:
            tokens = akana.phonemize(text, profile=profile)
            assert tokens[0]["ph"] == "_" and tokens[-1]["ph"] == "_"
    # tts-v1 reads text of letters and punctuation as written
    for text in ["TBMM bugün", "II. Dünya Savaşı", "M. Kemal", "www.ornek.com", "vb. online"]:
        assert akana.verbalize(text, words=False)["segments"] == [], text
    # tts-v2 reads decomposed text as composed text
    nfc = "Çiçek güzel, İstanbul'a gitti."
    nfd = unicodedata.normalize("NFD", nfc)
    words = akana.phonemize_detailed(nfd, profile="tts-v2")["words"]
    assert [w["reading"] for w in words] == ["çiçek", "güzel", "istanbul'a", "gitti"]
    assert akana.phonemize(nfd, profile="tts-v2") == akana.phonemize(nfc, profile="tts-v2")
    for w in words:
        assert unicodedata.normalize("NFC", nfd[w["start"]:w["end"]]) == unicodedata.normalize("NFC", w["text"])


# --- tts-v2: the literature consensus of 2026-11-08 ---------------------------------------------


def _v2(text):
    """tts-v2 as EchoCraft's probe prints it: IPA, `ˈ` before the most prominent vowel."""
    toks = [t for t in akana.phonemize(text, profile="tts-v2") if t["type"] != "pause"]
    top = max((t["stress"] for t in toks if t["type"] == "vowel"), default=0)
    out, marked = [], False
    for t in toks:
        if t["word"] and out:
            out.append(" ")
        if t["type"] == "vowel" and t["stress"] == top and not marked:
            out.append("ˈ")
            marked = True
        out.append(t["ipa"])
    return "".join(out)


def test_tts_v2_literature_consensus():
    cases = {
        # S1 /ɾ/: a tap, devoiced only before a pause
        "gelir.": "ɟelˈir̝̊",
        "bir de": "bˈiɾ dɛ",
        "var mı?": "vˈaɾ mɯ",
        "her şey": "hæɾ ʃˈej",
        "sonra": "sˈonɾa",
        # S2 -mAz, pekmez; S3 hem, elbise, Belçika; S4 word-final /e/
        "gelmez": "ɟælmˈæz",
        "pekmez": "pecmˈæz",
        "hem": "hˈem",
        "elbise": "elbisˈɛ",
        "Belçika": "bˈeltʃika",
        "dede geldi": "dedˈɛ ɟældɪ",
        # T1 -en adverbs, T2 -CA, T3 place names, T4 compounds, T5 intensives
        "kısmen": "kˈɯsmæn",
        "nispeten": "nispˈetæn",
        "çocukça": "tʃodʒˈuktʃa",
        "ayrıca": "ajɾˈɯdʒa",
        "Denizli": "denˈizlɪ",
        "Harbiye": "hˈaɾbijɛ",
        "Süleymaniye": "sylejmˈanijɛ",
        "bugün": "bˈuɟyn",
        "başbakan": "bˈaʃbakan",
        "kapkara": "kˈapkaɾa",
        "bembeyaz": "bˈembejaz",
        "yapayalnız": "jˈapajaɫnɯz",
    }
    for text, want in cases.items():
        assert _v2(text) == want, text
    assert "doˈandʒɯɫaɾa" in _v2("Biz Doğancılar'a gittik.")
    # no fricated r anywhere: the aligner symbol r̝ is gone
    assert all(t["ipa"] != "r̝" for t in akana.phonemize("Ruh, sonra kardeş ekran.", profile="tts-v2"))
