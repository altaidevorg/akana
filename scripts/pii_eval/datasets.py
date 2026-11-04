"""Datasets of the PII evaluation, read from the local Hugging Face cache at pinned revisions.

Every loader returns `Row`s whose gold spans carry a coarse class (the label set of
openai/privacy-filter, which BTX24, Belgin and yusuf-said share) and the dataset's own label.

Roles: `TEST` sets report results and are never trained or tuned on; nothing in the code or the
tests may be copied from them (contamination.py checks). BTX24 train and Belgin train may train
models; the old rules were tuned on BTX24 test, Belgin train and Saturday Labs, so their figures
there are not held-out figures.

Fetch a set with `hf download --repo-type dataset <repo> --revision <rev>`.
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HUB = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub")

REVISIONS = {
    "yusuf-said/turkish-privacy-filter-dataset": "fc2b9a3e197484d22a8201317cc722baacf053b7",
    "negentropi/belgin-pii-dataset": "72e010b507097c2ef29798cf3814a52a69a04a1b",
    "BTX24/turkish-privacy-pii-ner": "45a8e18edb57224b14368c910f701e064dbb1326",
    "cagrigungor/turkish-pii-masking-benchmark": "b2e6e68b260873cda53e90e623c1fa35d74fc7eb",
}

COARSE = ("person", "address", "email", "phone", "url", "date", "account", "secret")

# the datasets' labels -> coarse class
DATASET_COARSE = {
    "private_person": "person", "private_address": "address", "private_email": "email",
    "private_phone": "phone", "private_url": "url", "private_date": "date",
    "account_number": "account", "secret": "secret", "tckn": "account", "vkn": "account",
    "iban": "account",
}


@dataclass
class Row:
    text: str
    gold: list = field(default_factory=list)  # (start, end, coarse, label)
    slice: str = ""
    rid: str = ""
    ignore: list = field(default_factory=list)  # (start, end): values outside the scored scheme


def snapshot(repo: str) -> Path:
    path = HUB / ("datasets--" + repo.replace("/", "--")) / "snapshots" / REVISIONS[repo]
    if not path.is_dir():
        sys.exit(f"{repo}@{REVISIONS[repo][:8]} is not in {HUB}: "
                 f"hf download --repo-type dataset {repo} --revision {REVISIONS[repo]}")
    return path


def _privacy_filter_rows(path: Path, slice_key: str) -> list[Row]:
    rows = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        gold = []
        for key, spans in r["spans"].items():
            label = key.split(":")[0]
            gold += [(s, e, DATASET_COARSE[label], label) for s, e in spans]
        info = r.get("info") or {}
        rows.append(Row(r["text"], gold, info.get(slice_key, ""), r.get("id") or info.get("id", "")))
    return rows


def yusuf_said() -> list[Row]:
    """5,110 synthetic support, log, form and OCR texts, 1,273 of them without PII. Its generator
    writes almost every phone number with 12 national digits, so its phone figure measures
    tolerance for malformed numbers."""
    return _privacy_filter_rows(snapshot("yusuf-said/turkish-privacy-filter-dataset") / "tr_privacy_tr_curated.jsonl", "scenario")


def yusuf_said_split(part: str) -> list[Row]:
    """yusuf-said has one split. A fixed 30 % of its rows (by a hash of the row id) is a dev part
    that may be inspected to diagnose errors; the other 70 % is the test part, only reported. Split
    on 2026-11-01, after the full set had been scored but before any of its errors were read."""
    import hashlib
    rows = yusuf_said()
    dev = {r.rid for r in rows if int(hashlib.sha1(r.rid.encode()).hexdigest(), 16) % 10 < 3}
    return [r for r in rows if (r.rid in dev) == (part == "dev")]


def belgin(split: str) -> list[Row]:
    """Synthetic hard cases in complaint style; `split` is train, validation or test."""
    return _privacy_filter_rows(snapshot("negentropi/belgin-pii-dataset") / "data" / f"{split}.jsonl", "case")


def btx24(split: str) -> list[Row]:
    """Machine-generated templates, ten labels, no PII-free rows."""
    rows = []
    for line in open(snapshot("BTX24/turkish-privacy-pii-ner") / f"{split}.jsonl", encoding="utf-8"):
        r = json.loads(line)
        rows.append(Row(r["text"], [(x["start"], x["end"], DATASET_COARSE[x["category"]], x["category"]) for x in r["label"]]))
    return rows


TAG = re.compile(r"\[[A-ZÇĞİÖŞÜ_0-9]+\]")


def saturday(categories=("tam", "negatif")) -> list[Row]:
    """Saturday Labs' rows that mask every value ("tam") or none ("negatif"). The gold spans come
    from aligning the input with the expected output; their label is the benchmark's tag (AD,
    TCKN, ...), also used as the coarse class, since its 53 tags do not map onto eight."""
    rows = []
    path = snapshot("cagrigungor/turkish-pii-masking-benchmark") / "benchmark_1000.csv"
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        if r["kategori"] not in categories:
            continue
        text, expected = r["input"], r["expected_output"]
        parts, tags = TAG.split(expected), TAG.findall(expected)
        pos, values = 0, []
        for i, part in enumerate(parts):
            j = text.find(part, pos) if part else pos
            if j < 0:
                break
            if i > 0:
                values.append((tags[i - 1][1:-1], pos, j))
            pos = j + len(part)
        else:
            if len(parts) > 1 and not parts[-1]:
                values[-1] = (values[-1][0], values[-1][1], len(text))
            gold = [(s, e, tag, tag) for tag, s, e in values if e > s]
            rows.append(Row(text, gold, r["kategori"] + ":" + r["ozellikler"], r["id"]))
    return rows


# Saturday Labs tags outside the tagger's label scheme: demographic attributes, KVKK special
# categories (a policy layer's job), workplaces, birthplaces, coordinates and signature records
SATURDAY_OUT_OF_SCHEME = {"UYRUK", "YAS", "CINSIYET", "MAAS", "KREDI_NOTU", "KAN_GRUBU", "CEZA_KAYDI", "SAGLIK",
                          "BIYOMETRIK", "ETNIK_KOKEN", "ENGEL_DURUMU", "DIN", "SENDIKA", "AILE", "DOGUM_YERI",
                          "ISYERI", "KONUM", "IMZA"}


def saturday_in_scheme() -> list[Row]:
    """Saturday Labs scored on the values the tagger's scheme covers (names, contact details,
    identifiers, secrets, birth dates); a detection on any other tagged value counts neither for
    nor against a system."""
    rows = []
    for r in saturday():
        keep = [g for g in r.gold if g[3] not in SATURDAY_OUT_OF_SCHEME]
        rows.append(Row(r.text, keep, r.slice, r.rid, [(g[0], g[1]) for g in r.gold if g[3] in SATURDAY_OUT_OF_SCHEME]))
    return rows


def wikiner_persons(split: str = "test", root: Path | None = None) -> list[Row]:
    """Real Wikipedia sentences of Turkish WikiNER with their person spans (B-PER/I-PER), the
    tokens joined with spaces except before closing punctuation. Prepared by
    scripts/prepare_wikiner.py into .cache/ner/wikiner."""
    path = (root or REPO_ROOT / ".cache" / "ner" / "wikiner") / f"{split}.tsv"
    if not path.is_file():
        sys.exit(f"{path} is missing: run scripts/prepare_wikiner.py")
    sentences, current = [], []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            if current:
                sentences.append(current)
            current = []
            continue
        current.append(tuple(line.split("\t")))
    if current:
        sentences.append(current)
    rows = []
    for k, sentence in enumerate(sentences):
        text, pos = "", []
        for token, tag in sentence:
            if text and not re.match(r"^[.,;:!?)\]’'”]", token) and not text.endswith(("(", "[", "“")):
                text += " "
            pos.append((len(text), len(text) + len(token), tag))
            text += token
        gold, i = [], 0
        while i < len(pos):
            if pos[i][2] == "B-PER":
                s, e, j = pos[i][0], pos[i][1], i + 1
                while j < len(pos) and pos[j][2] == "I-PER":
                    e, j = pos[j][1], j + 1
                gold.append((s, e, "person", "PER"))
                i = j
            else:
                i += 1
        rows.append(Row(text, gold, "", f"wikiner-{split}-{k}"))
    return rows


GOLD_COARSE = {"PERSON": "person", "USERNAME": "person", "ADDRESS": "address", "PHONE": "phone",
               "EMAIL": "email", "URL": "url", "ID": "account", "SECRET": "secret", "DATE": "date"}


def real_gold(pool: str) -> list[Row]:
    """Real Turkish text (court, Wikipedia, news, tweets, reviews, voice-assistant requests, LLM
    prompts, comments, word problems) with gold labels: two teacher passes adjudicated and reviewed
    (scripts/pii_data). Every person counts, public figures included; dates and URLs only when
    personal; phones and e-mails unless public (call-centre, emergency and role numbers and
    mailboxes, which the policy leaves unmasked); sensitive-category spans are not scored here. The
    slice is the domain."""
    path = REPO_ROOT / ".cache" / "pii_data" / "real" / f"gold_{pool}.jsonl"
    if not path.is_file():
        sys.exit(f"{path} is missing: see scripts/pii_data")
    rows = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        gold = []
        for s, e, label, attrs in r["spans"]:
            if (label not in GOLD_COARSE or (label in ("DATE", "URL") and attrs.get("personal") is False)
                    or (label in ("PHONE", "EMAIL") and attrs.get("public") is True)):
                continue
            gold.append((s, e, GOLD_COARSE[label], label))
        rows.append(Row(r["text"], gold, r["domain"], r["id"]))
    return rows


# name -> (loader, role, whether the gold labels every coarse class or only persons)
SETS = {
    "real-gold": (lambda: real_gold("test"), "test", "all"),
    "real-gold-dev": (lambda: real_gold("dev"), "dev", "all"),
    "yusuf-said": (yusuf_said, "test", "all"),
    "yusuf-test": (lambda: yusuf_said_split("test"), "test", "all"),
    "yusuf-dev": (lambda: yusuf_said_split("dev"), "dev", "all"),
    "belgin-test": (lambda: belgin("test"), "test", "all"),
    "belgin-validation": (lambda: belgin("validation"), "dev", "all"),
    "btx24-test": (lambda: btx24("test"), "test", "all"),
    "saturday": (saturday, "test", "saturday"),
    "saturday-in-scheme": (saturday_in_scheme, "test", "saturday"),
    "wikiner-persons": (wikiner_persons, "test", "person"),
    "belgin-train": (lambda: belgin("train"), "train", "all"),
    "btx24-validation": (lambda: btx24("validation"), "dev", "all"),
}

DEFAULT_SETS = ("real-gold", "yusuf-test", "belgin-test", "belgin-validation", "btx24-test", "saturday", "wikiner-persons")
