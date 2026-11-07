"""The phonemiser's Python API: `phonemize` and `phonemize_detailed` (profile `espeak-compat`).

The fixture test reads EchoCraft's eSpeak reference fixtures from the directory in
``AKANA_ESPEAK_FIXTURES`` and skips without it.
"""

import json
import os
from pathlib import Path

import pytest

import akana

GOLDEN = Path(__file__).resolve().parents[1] / "crates/akana-core/tests/data/speech/espeak_compat_golden.jsonl"
FIELDS = ("ph", "stress", "word", "lengthen", "mark")


def test_token_schema():
    tokens = akana.phonemize("Dağ.")
    assert tokens[0] == {"ph": "_", "stress": 0, "word": 0, "lengthen": 0, "mark": ".", "type": "pause", "ipa": None}
    assert tokens[-1]["ph"] == "_"
    vowel = [t for t in tokens if t["type"] == "vowel"][0]
    assert vowel == {"ph": "a", "stress": 7, "word": 0, "lengthen": 1, "mark": ".", "type": "vowel", "ipa": "aː"}


def test_golden_sentences():
    rows = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 60
    for row in rows:
        got = [[t[f] for f in FIELDS] for t in akana.phonemize(row["text"])]
        assert got == row["tokens"], row["text"]


def test_detailed_has_provenance_and_character_spans():
    text = "Çay ister misiniz? # 3 bardak"
    d = akana.phonemize_detailed(text)
    assert d["profile"] == "espeak-compat"
    assert d["version"].startswith("espeak-compat-")
    words = d["words"]
    # the joined pair is one word, spanning both
    assert words[1]["text"] == "ister misiniz"
    assert text[words[1]["start"]:words[1]["end"]] == "ister misiniz"
    for w in words:
        assert w["stress_source"] and w["segment_source"]
        assert all(p["ph"] for p in w["phones"])
    # the symbol is not read in this profile, and says so; the digit is read as a number
    assert [text[w["start"]:w["end"]] for w in d["warnings"]] == ["#"]
    assert words[2]["reading"] == "üç" and words[2]["normalized"] == "number"
    assert words[0]["normalized"] is None
    assert d["tokens"] == akana.phonemize(text)


def test_unknown_profile_raises():
    with pytest.raises(ValueError):
        akana.phonemize("merhaba", profile="literature-v9")


def test_odd_input_never_raises():
    for text in ["", "...", "̂", "â", "-i", "’", "😀 merhaba", "x" * 1000]:
        tokens = akana.phonemize(text)
        assert tokens[0]["ph"] == "_" and tokens[-1]["ph"] == "_"


@pytest.mark.skipif("AKANA_ESPEAK_FIXTURES" not in os.environ, reason="AKANA_ESPEAK_FIXTURES is not set")
def test_fixture_validation_parity():
    plan = Path(os.environ["AKANA_ESPEAK_FIXTURES"]) / "espeak-plan.jsonl"
    rows = [json.loads(line) for line in plan.read_text(encoding="utf-8").splitlines()]
    val = [r for r in rows if r["split"] == "validation"]
    exact = 0
    for r in val:
        got = [[t[f] for f in FIELDS] for t in akana.phonemize(r["text"])]
        want = [[t[f] for f in FIELDS] for t in r["tokens"]]
        exact += got == want
    assert exact / len(val) >= 0.97


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


TTS_GOLDEN = GOLDEN.with_name("tts_v1_golden.jsonl")
TTS_FIELDS = ("ph", "stress", "word", "lengthen", "mark", "type", "ipa")


def test_tts_v1_golden_is_frozen():
    rows = [json.loads(line) for line in TTS_GOLDEN.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 60
    version = akana.phonemize_detailed("a", profile="tts-v1")["version"]
    for row in rows:
        assert row["version"] == version
        got = [[t[f] for f in TTS_FIELDS] for t in akana.phonemize(row["text"], profile="tts-v1")]
        assert got == row["tokens"], row["text"]
