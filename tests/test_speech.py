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
