"""Tests of the PII evaluation and training helpers (no model, no akana build needed).

    .venv/Scripts/python.exe -m pytest scripts/pii_eval/test_pii_tools.py -q
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
for sub in ("pii_eval", "pii_train", "pii_data"):
    sys.path.insert(0, str(SCRIPTS / sub))

import datasets  # noqa: E402
import hardcase_collect  # noqa: E402
import score  # noqa: E402
import stress  # noqa: E402
import systems  # noqa: E402
import tagger  # noqa: E402
import values  # noqa: E402


# ------------------------------------------------------------------ scorer


@pytest.mark.parametrize("text,span,core", [
    ("Ahmet Kaya'nın", (0, 14), "Ahmet Kaya"),       # a suffix after an apostrophe is not part of the value
    ("Ahmet Kaya\x92nın", (0, 14), "Ahmet Kaya"),    # cp1252 right quote read as Latin-1
    ("Ahmet Kayanın", (0, 13), "Ahmet Kayanın"),     # a glued suffix stays
    ("(Ahmet Kaya),", (0, 13), "Ahmet Kaya"),        # surrounding punctuation
    (" 0532 111 22 33.", (0, 16), "0532 111 22 33"),
    ("Dyt. Gizem Demir", (0, 16), "Gizem Demir"),     # titles and honorifics are not personal data
    ("Prof. Dr. Ali Kaya'nın", (0, 22), "Ali Kaya"),
    ("Psikolog Gonca Taş Maden", (0, 24), "Gonca Taş Maden"),
    ("Ayşe Hanım", (0, 10), "Ayşe"),
    ("Kaya Bey'e", (0, 10), "Kaya"),
    ("Bayram Yılmaz", (0, 13), "Bayram Yılmaz"),     # a name that starts like a title stays whole
    ("Hakim Yılmaz", (0, 12), "Hakim Yılmaz"),       # Hakim is also a given name: never trimmed
    ("DR. AHMET KAYA BEY", (0, 18), "AHMET KAYA"),
    ("Dr.", (0, 3), "Dr"),                            # nothing but a title: left as it is
])
def test_core_trims_edges_and_apostrophe_suffixes(text, span, core):
    s, e = score.core(text, *span)
    assert text[s:e] == core


def _score(text, gold, preds):
    row = datasets.Row(text, [(s, e, c, c) for s, e, c in gold])
    per_row, _, _, _ = score.evaluate([row], lambda t: [(s, e, c, c) for s, e, c in preds])
    return score.summary(score.total_of(per_row))


def test_exact_match_scores_one_and_leaks_nothing():
    text = "Ahmet Kaya 0532 111 22 33 numarasından aradı."
    gold = [(0, 10, "person"), (11, 25, "phone")]
    s = _score(text, gold, gold)
    assert s["exact_f1"] == 1.0 and s["full_cover_r"] == 1.0 and s["char_leak"] == 0.0


def test_partial_mask_is_an_overlap_but_not_full_cover():
    text = "Ahmet Kaya 0532 111 22 33 numarasından aradı."
    s = _score(text, [(0, 10, "person")], [(0, 5, "person")])
    assert s["exact_f1"] == 0.0 and s["overlap_f1"] == 1.0 and s["full_cover_r"] == 0.0 and s["char_leak"] > 0.4


def test_flag_on_a_pii_free_row_counts():
    s = _score("Kampanya 31.12.2026 tarihine kadar geçerlidir.", [], [(9, 19, "date")])
    assert s["neg_flagged"] == 1.0


def test_ignored_values_count_neither_way():
    text = "Ahmet Kaya 62 yaşında, kan grubu 0 Rh negatif."
    row = datasets.Row(text, [(0, 10, "person", "AD")], ignore=[(11, 13), (33, 45)])
    preds = [(0, 10, "person", "AD"), (11, 13, "account", "TCKN"), (33, 34, "account", "TCKN")]
    per_row, _, _, _ = score.evaluate([row], lambda t: preds)
    s = score.summary(score.total_of(per_row))
    assert s["exact_p"] == 1.0 and s["exact_r"] == 1.0 and s["overmask"] == 0.0
    only_ignored = datasets.Row("62 yaşında", [], ignore=[(0, 2)])
    per_row, _, _, _ = score.evaluate([only_ignored], lambda t: [])
    assert score.total_of(per_row)["neg_rows"] == 0


def _cached(repo: str) -> bool:
    """Whether a pinned evaluation dataset is in the local Hugging Face cache (CI has none)."""
    try:
        datasets.snapshot(repo)
    except SystemExit:
        return False
    return True


@pytest.mark.skipif(not _cached("cagrigungor/turkish-pii-masking-benchmark"),
                    reason="Saturday Labs is not in the local Hugging Face cache")
def test_saturday_in_scheme_keeps_every_value_once():
    for full, scoped in zip(datasets.saturday(), datasets.saturday_in_scheme()):
        assert len(scoped.gold) + len(scoped.ignore) == len(full.gold)
        assert all(g[3] not in datasets.SATURDAY_OUT_OF_SCHEME for g in scoped.gold)


# ------------------------------------------------------------------ tagger helpers


def test_normalize_keeps_offsets_and_turkish_letters():
    for text in ["Bugün Ahmet Yılmaz\U0001F389 ile kutladık!", "Ahmet´in notu ❤️ ok", "Çağla Öztürk İğdır'da", "ascii only"]:
        out = tagger.normalize(text)
        assert len(out) == len(text)
        assert all(a == b or b == " " for a, b in zip(text, out))
        assert all(a == b for a, b in zip(text, out) if a.isalnum() or a.isascii())
    assert tagger.normalize("Yılmaz\U0001F389") == "Yılmaz "
    assert tagger.normalize("Çağla Öztürk") == "Çağla Öztürk"


def test_token_tags_and_spans_round_trip():
    offsets = [(0, 0), (0, 5), (6, 10), (10, 13), (14, 19), (0, 0)]  # [CLS] Ahmet Kaya 'nın aradı [SEP]
    tags = tagger.token_tags(offsets, [(0, 10, "PERSON")])
    assert [tagger.TAGS[t] if t >= 0 else None for t in tags] == [None, "B-PERSON", "I-PERSON", "O", "O", None]
    real = [k for k, t in enumerate(tags) if t >= 0]
    assert tagger.spans_from_tags([offsets[k] for k in real], [tags[k] for k in real]) == [(0, 10, "PERSON")]


@pytest.mark.parametrize("text,span,label,snapped", [
    ("Kristof Kolomb geldi", (0, 2), "PERSON", "Kristof"),
    ("dün pınarla konuştum", (4, 9), "PERSON", "pınarla"),
    ("Ahmet'in kitabı", (0, 5), "PERSON", "Ahmet"),            # the suffix after an apostrophe stays out
    ("Ahmet-Can geldi", (0, 5), "PERSON", "Ahmet"),            # a name does not widen over a hyphen
    ("karar 29/9/2016, § 42", (6, 8), "DATE", "29/9/2016"),
    ("kod SRM-2021-K2VM6 okundu", (4, 5), "ID", "SRM-2021-K2VM6"),
    ("e-posta:ali.veli@x.com", (12, 22), "EMAIL", "ali.veli@x.com"),  # not over the colon
    ("Tel:05321234567", (4, 15), "PHONE", "05321234567"),
])
def test_snap_widens_spans_to_whole_words(text, span, label, snapped):
    (s, e, lab), = tagger.snap(text, [(*span, label)])
    assert (text[s:e], lab) == (snapped, label)


def test_decode_takes_each_token_from_its_most_central_window():
    np = pytest.importorskip("numpy")

    def lp(tags):
        out = np.full((len(tags), len(tagger.TAGS)), np.log(0.01))
        for k, t in enumerate(tags):
            out[k, tagger.TAG_ID[t]] = np.log(0.9)
        return out

    # "Bugün Kristof Kolomb geldi": tokens Bugün | Kr | ##istof | Kolomb | geldi
    offs = [(0, 5), (6, 8), (8, 13), (14, 20), (21, 26)]
    first = (0, offs[:2], lp(["O", "B-PERSON"]))  # ends after "Kr": no right context
    second = (0, offs, lp(["O", "B-PERSON", "I-PERSON", "I-PERSON", "O"]))
    assert tagger.decode([first, second], 1) == [[(6, 20, "PERSON")]]
    assert tagger.decode([second, first], 1) == [[(6, 20, "PERSON")]]


def test_an_operating_point_can_raise_some_labels_only():
    np = pytest.importorskip("numpy")
    # "Ali 12.04.2026": both values just below the O tag, Ali as a person, the date as a date
    offs = [(0, 3), (4, 14)]
    lp = np.full((2, len(tagger.TAGS)), np.log(0.01))
    lp[:, 0] = np.log(0.5)
    lp[0, tagger.TAG_ID["B-PERSON"]] = lp[1, tagger.TAG_ID["B-DATE"]] = np.log(0.4)
    scored = [(0, offs, lp)]
    assert tagger.decode(scored, 1) == [[]]
    assert tagger.decode(scored, 1, 0.5) == [[(0, 3, "PERSON"), (4, 14, "DATE")]]  # lowering O raises every label
    assert tagger.decode(scored, 1, 0.5, labels=("PERSON",)) == [[(0, 3, "PERSON")]]
    assert tagger.decode(scored, 1, 0.5, labels=("PERSON", "DATE")) == tagger.decode(scored, 1, 0.5)


def test_an_ensemble_teacher_is_the_mean_of_tempered_distributions():
    torch = pytest.importorskip("torch")
    import train_tagger
    a, b = torch.randn(2, 5, len(tagger.TAGS)), torch.randn(2, 5, len(tagger.TAGS))
    assert torch.allclose(train_tagger.teacher_logp([a], 2.0), torch.log_softmax(a / 2, -1), atol=1e-6)
    mean = (torch.softmax(a / 2, -1) + torch.softmax(b / 2, -1)) / 2
    assert torch.allclose(train_tagger.teacher_logp([a, b], 2.0).exp(), mean, atol=1e-6)


def test_snap_merges_spans_that_meet():
    assert tagger.snap("Mehmetcan", [(0, 3, "PERSON"), (6, 9, "PERSON")]) == [(0, 9, "PERSON")]


def test_viterbi_obeys_bio():
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(0)
    for _ in range(200):
        lp = np.log(rng.dirichlet(np.ones(len(tagger.TAGS)), size=12))
        path = [tagger.TAGS[t] for t in tagger.viterbi(lp)]
        for prev, cur in zip(["O"] + path, path):
            if cur.startswith("I-"):
                assert prev != "O" and prev[2:] == cur[2:]


# ------------------------------------------------------------------ policy


@pytest.mark.parametrize("value,public", [
    ("444 0 123", True), ("Tel: 444 0 123", True), ("0850 222 33 44", True), ("112", True),
    ("dört dört dört bir iki üç dört", True),
    ("0532 123 45 67", False), ("Tel: 0212 555 12 34", False),
    ("sıfır beş üç iki bir iki üç dört beş altı yedi", False), ("beş yüz otuz iki", False),
])
def test_public_phone_policy(value, public):
    assert systems.public_value(value, "PHONE") is public


def test_merge_keeps_tagger_boundaries_and_adds_what_it_missed():
    # "ayse.kaya@x.com. Klinik ..." : akana's e-mail pattern overruns into the next word
    model = [(0, 15, "EMAIL")]
    rules = [(0, 23, "EMAIL", "EMAIL"), (40, 51, "ID", "TCKN")]
    assert systems.merge_spans(model, rules) == [(0, 15, "EMAIL"), (40, 51, "ID")]


def test_merge_lets_a_checksummed_identifier_repair_a_split_span():
    model = [(10, 14, "ID"), (15, 34, "ID"), (50, 55, "PERSON")]
    rules = [(10, 42, "ID", "IBAN")]
    assert systems.merge_spans(model, rules) == [(10, 42, "ID"), (50, 55, "PERSON")]


def test_role_mailboxes_are_public_and_personal_ones_are_not():
    assert systems.public_value("destek@firma.com.tr", "EMAIL")
    assert systems.public_value("kvkk [at] firma [dot] com", "EMAIL")
    assert not systems.public_value("ayse.kaya@gmail.com", "EMAIL")


# ------------------------------------------------------------------ data generation and collection


def test_number_words():
    assert values.number_words("532") == "beş yüz otuz iki"
    assert values.number_words("100") == "yüz"
    assert values.number_words("07") == "sıfır yedi"
    assert values.number_words("45") == "kırk beş"


def test_spoken_phone_reads_out_a_mobile_number():
    words = set(values.ONES + values.TENS + ["sıfır", "yüz"]) - {""}
    rng = random.Random(5)
    for _ in range(100):
        p = values.spoken_phone(rng)
        tokens = re.findall(r"\w+", p)
        assert all(t in words or t.isdigit() for t in tokens), p
        assert not systems.public_value(p, "PHONE")


def test_agreement_key_is_what_training_sees():
    personal = hardcase_collect.key([3, 9, "DATE", {"personal": True}])
    assert personal == (3, 9, "DATE")
    assert hardcase_collect.key([3, 9, "DATE", {}]) == personal  # unmarked counts as personal, as in training
    assert hardcase_collect.key([3, 9, "DATE", {"personal": False}]) is None  # public dates are not learnt
    assert hardcase_collect.key([3, 9, "URL", {"personal": False}]) is None
    assert hardcase_collect.key([3, 9, "SENSITIVE", {"subtype": "health"}]) is None
    assert hardcase_collect.key([0, 4, "PERSON", {"public": True}]) == hardcase_collect.key([0, 4, "PERSON", {}])
    # the tagger learns every phone and mailbox; the policy decides which stay unmasked
    assert hardcase_collect.key([0, 3, "PHONE", {"public": True}]) == (0, 3, "PHONE")
    assert hardcase_collect.key([0, 9, "EMAIL", {"public": True}]) == (0, 9, "EMAIL")


GOLD_UNIT = {"id": "court:x:1", "domain": "court", "text": "112 Acil Servis, 0532 111 22 33, 1/3/1970",
             "spans": [[0, 3, "PHONE", {"public": True}], [16, 30, "PHONE", {}], [32, 40, "DATE", {"personal": False}]]}


def test_real_gold_leaves_public_numbers_and_dates_to_the_policy(tmp_path, monkeypatch):
    (tmp_path / ".cache" / "pii_data" / "real").mkdir(parents=True)
    (tmp_path / ".cache" / "pii_data" / "real" / "gold_dev.jsonl").write_text(json.dumps(GOLD_UNIT) + "\n", encoding="utf-8")
    monkeypatch.setattr(datasets, "REPO_ROOT", tmp_path)
    [row] = datasets.real_gold("dev")
    assert [(s, e) for s, e, *_ in row.gold] == [(16, 30)]


def test_court_citation_dates_become_personal():
    import relabel_dates
    text = "(Ahmet Sağlam [2. B.], B. No: 2013/3351, 18/9/2013, § 42) ve (Onur Doğanay, B. No: 2013/1977,9/1/2014)"
    s1, s2 = text.index("18/9/2013"), text.index("9/1/2014")
    row = {"text": text, "spans": [[1, 13, "PERSON", {}], [s1, s1 + 9, "DATE", {"personal": False}]]}
    out, flipped, added = relabel_dates.relabel(row)
    assert (flipped, added) == (1, 1)
    assert out["spans"][1] == [s1, s1 + 9, "DATE", {"personal": True}]
    assert out["spans"][-1] == [s2, s2 + 8, "DATE", {"personal": True}]
    assert relabel_dates.relabel({"text": "Kanun 1/3/1970 tarihinde yürürlüğe girdi.", "spans": []})[1:] == (0, 0)


def test_dev_gold_is_scored_like_the_test_gold(tmp_path):
    import data
    path = tmp_path / "gold_dev.jsonl"
    path.write_text(json.dumps(GOLD_UNIT) + "\n", encoding="utf-8")
    [row] = data.gold(path)
    assert row["spans"] == [(16, 30, "PHONE")]


def test_stress_generators_make_valid_cards_and_ibans():
    rng = random.Random(1)
    for _ in range(50):
        c = stress.luhn(rng)
        assert sum(sum(divmod(2 * int(x), 10)) if i % 2 else int(x) for i, x in enumerate(reversed(c))) % 10 == 0
        iban = stress.iban(rng)
        assert int("".join(str(int(ch, 36)) for ch in iban[4:] + iban[:4])) % 97 == 1
