"""Training data for the PII tagger: one label scheme over every source, and case and spelling
augmentation for informal text.

Labels: PERSON (public figures included), USERNAME, ADDRESS, PHONE and EMAIL (public numbers and
role mailboxes included: a policy step decides about those), URL (personal ones only), ID, SECRET,
DATE (personal ones only). Sources: the synthetic files of make_synthetic.py, WikiNER's training
persons, and teacher-labelled real text (labels/<run>_<pass>.jsonl or gold files).
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "pii_eval"))
import datasets  # noqa: E402

LABELS = ("PERSON", "USERNAME", "ADDRESS", "PHONE", "EMAIL", "URL", "ID", "SECRET", "DATE")
SYN = ROOT / ".cache" / "pii_data" / "synthetic"
REAL = ROOT / ".cache" / "pii_data" / "real"


def keep(label: str, attrs: dict) -> str | None:
    """The training label of a span, or None for one the tagger should leave alone."""
    if label not in LABELS:
        return None
    if label in ("URL", "DATE") and attrs.get("personal") is False:
        return None
    return label  # public phones and role mailboxes are learnt too: the policy decides what stays


def policy_public(label: str, attrs: dict | None) -> bool:
    """A gold value the deployed policy leaves unmasked by design (call-centre, emergency and role
    numbers and mailboxes): not counted when scoring."""
    return label in ("PHONE", "EMAIL") and (attrs or {}).get("public") is True


# values the teacher labels wrongly in a known way: MASSIVE's voice assistant is called "olly"
NOT_PERSONS = {"olly"}


def _clean(text: str, spans) -> dict:
    out = []
    for s, e, label, *rest in spans:
        attrs = rest[0] if rest else {}
        if label == "PERSON" and text[s:e].lower() in NOT_PERSONS:
            continue
        lab = keep(label, attrs or {})
        if lab and 0 <= s < e <= len(text):
            out.append((s, e, lab))
    out.sort()
    kept, end = [], -1
    for s, e, lab in out:  # no overlaps: the first span wins
        if s >= end:
            kept.append((s, e, lab))
            end = e
    return {"text": text, "spans": kept}


# real-text domains whose source declares no licence or a non-commercial one (tweets-turkish,
# turkish_product_reviews, gsm8k_tr, RagTurk's Wikipedia packaging): --permissive leaves them out
UNCLEAR_LICENCE = ("tweets", "reviews", "word_problems", "wikipedia")


def jsonl(path: Path, limit: int | None = None, rng: random.Random | None = None, skip_domains: tuple = ()) -> list[dict]:
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    if skip_domains:
        rows = [r for r in rows if r.get("id", "").split(":", 1)[0] not in skip_domains]
    if limit is not None and limit < len(rows):
        rows = (rng or random.Random(0)).sample(rows, limit)
    return [_clean(r["text"], r["spans"]) for r in rows]


def wikiner_train() -> list[dict]:
    return [_clean(r.text, [(s, e, "PERSON", {}) for s, e, *_ in r.gold]) for r in datasets.wikiner_persons("train")]


def tr_lower(s: str) -> str:
    return s.replace("I", "ı").replace("İ", "i").lower()


def tr_upper(s: str) -> str:
    return s.replace("i", "İ").replace("ı", "I").upper()


ASCII = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")


def augment(rows: list[dict], rng: random.Random, lower=0.2, upper=0.04, ascii_=0.1) -> list[dict]:
    """Copies in lower case, capitals or typed without Turkish letters (each keeps every offset)."""
    out = []
    for r in rows:
        x = rng.random()
        if x < lower:
            out.append({"text": tr_lower(r["text"]), "spans": r["spans"]})
        elif x < lower + upper:
            out.append({"text": tr_upper(r["text"]), "spans": r["spans"]})
        if rng.random() < ascii_:
            out.append({"text": r["text"].translate(ASCII), "spans": r["spans"]})
    return out


HARD = ROOT / ".cache" / "pii_data" / "hardcases"


def training_mix(seed: int = 0, btx24: int = 40000, scenario: int = 20000, real_runs: tuple = (), wikiner: bool = True,
                 hard_runs: tuple = (), hard_repeat: int = 1, skip_domains: tuple = (), real_repeat: int = 1) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    rows += jsonl(SYN / "scenario.jsonl", scenario, rng)
    rows += jsonl(SYN / "btx24.jsonl", btx24, rng)
    rows += jsonl(SYN / "belgin.jsonl")
    if wikiner:
        rows += wikiner_train()
    for run in real_runs:  # teacher-labelled real text, a tenth of the mix unless repeated
        rows += jsonl(REAL / "labels" / f"{run}.jsonl", skip_domains=skip_domains) * real_repeat
    for run in hard_runs:  # generated technical and form texts both labellers agree on
        rows += jsonl(HARD / f"{run}.jsonl") * hard_repeat
    rows += augment(rows, rng)
    rng.shuffle(rows)
    return rows


def unlabelled(pools: tuple, limit: int | None = None, seed: int = 0, skip_domains: tuple = ()) -> list[dict]:
    """Units of sample pools in .cache/pii_data/real (training side only) without labels, for a
    teacher to label on the fly: {"text", "spans": None}."""
    rng = random.Random(seed)
    rows = []
    for pool in pools:
        units = [u["text"] for u in map(json.loads, open(REAL / f"{pool}.jsonl", encoding="utf-8"))
                 if u["domain"] not in skip_domains]
        if limit is not None and limit < len(units):
            units = rng.sample(units, limit)
        rows += [{"text": u, "spans": None} for u in units]
    return rows


COARSE_TO_LABEL = {"person": "PERSON", "address": "ADDRESS", "email": "EMAIL", "phone": "PHONE", "url": "URL",
                   "date": "DATE", "account": "ID", "secret": "SECRET"}


def dev_sets() -> dict:
    sets = {name: [_clean(r.text, [(s, e, COARSE_TO_LABEL[c], {}) for s, e, c, _ in r.gold]) for r in rows]
            for name, rows in (("belgin-validation", datasets.belgin("validation")),
                               ("yusuf-dev", datasets.yusuf_said_split("dev")))}
    gold_dev = REAL / "gold_dev.jsonl"
    if gold_dev.is_file():
        sets["gold-dev"] = gold(gold_dev)
    return sets


def gold(path: Path) -> list[dict]:
    """A real-text gold split as scored (datasets.real_gold scores the test split the same way):
    public numbers and mailboxes are the policy's, not values to find."""
    return [_clean(r["text"], [sp for sp in r["spans"] if not policy_public(sp[2], sp[3] if len(sp) > 3 else None)])
            for r in map(json.loads, open(path, encoding="utf-8"))]
