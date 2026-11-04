"""Synthetic Turkish PII examples: the scenario templates of templates.py, and the BTX24 and Belgin
training rows turned into templates (their values replaced by slots, a case suffix after an
apostrophe re-inflected for the new value), filled with values.py's generators.

    python scripts/pii_data/make_synthetic.py --out .cache/pii_data/synthetic --scenario 20000 --btx24 1 --belgin 4
Writes scenario.jsonl, btx24.jsonl, belgin.jsonl: {"text", "spans": [[start, end, type, attrs]], "source"}.
Only training splits are read; held-out rows never become templates.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "pii_eval"))
import datasets  # noqa: E402
import templates as T  # noqa: E402
import values as V  # noqa: E402

SLOT = re.compile(r"\{([A-Z_]+)(?::([a-z]+))?(?:\|([a-z]+)(~)?)?\}")
APOSTROPHES = "'’"
CASES = {
    "gen": ("in", "ın", "un", "ün", "nin", "nın", "nun", "nün"), "dat": ("e", "a", "ye", "ya"),
    "acc": ("i", "ı", "u", "ü", "yi", "yı", "yu", "yü"), "loc": ("de", "da", "te", "ta"),
    "abl": ("den", "dan", "ten", "tan"), "ins": ("le", "la", "yle", "yla"), "pl": ("ler", "lar"),
}
CASE_OF = {s: c for c, forms in CASES.items() for s in forms}
DATASET_SLOT = {"private_person": "PERSON", "private_address": "ADDRESS", "private_email": "EMAIL",
                "private_phone": "PHONE", "private_url": "URL", "private_date": "DATE", "account_number": "ID",
                "secret": "SECRET", "tckn": "TCKN", "vkn": "VKN", "iban": "IBAN"}
PUBLIC_FIGURES = [p for p in V._rust_list(V.PII_SRC / "names_data.rs", "PUBLIC_FIGURES") if " " in p]


def first_name(rng):
    g = V.given_name(rng)
    r = rng.random()
    return V.tr_lower(g) if r < 0.25 else V.tr_upper(g) if r < 0.32 else V.tr_title(g)


def value_of(slot: str, variant: str | None, rng: random.Random):
    """(text, label type or None, attributes)."""
    if slot == "PERSON":
        if variant == "public":
            return V.tr_title(rng.choice(PUBLIC_FIGURES)), "PERSON", {"public": True}
        return (first_name(rng) if variant == "first" else V.person(rng)), "PERSON", {"public": False}
    if slot == "HANDLE":
        return V.handle(rng), "USERNAME", {}
    if slot == "PHONE":
        kind = variant or rng.choices(["mobile", "landline", "malformed", "foreign", "spoken"], [0.7, 0.18, 0.07, 0.05, 0.04])[0]
        return V.phone(rng, kind), "PHONE", {"public": kind == "callcentre"}
    if slot == "EMAIL":
        kind = variant or rng.choices(["personal", "work", "obfuscated"], [0.75, 0.2, 0.05])[0]
        return V.email(rng, kind), "EMAIL", {"public": kind == "role"}
    if slot == "URL":
        if variant == "public":
            return V.url(rng, personal=False), None, {}
        return V.url(rng), "URL", {"personal": True}
    if slot == "ID":
        text, attrs = V.identifier(rng)
        return text, "ID", attrs
    if slot in ("TCKN", "VKN"):
        return (V.tckn(rng, valid=rng.random() < 0.9) if slot == "TCKN" else V.vkn(rng)), "ID", {"subtype": "national"}
    if slot == "IBAN":
        return V.iban_tr(rng, valid=rng.random() < 0.9), "ID", {"subtype": "financial"}
    if slot == "CARD":
        return V.card(rng), "ID", {"subtype": "financial"}
    if slot == "PLATE":
        return V.plate(rng), "ID", {"subtype": "vehicle"}
    if slot == "CUSTOMER":
        return V.record_id(rng, "customer"), "ID", {"subtype": "customer"}
    if slot == "ORDER":
        return V.record_id(rng, "transaction"), "ID", {"subtype": "transaction"}
    if slot == "SECRET":
        return V.secret(rng), "SECRET", {}
    if slot == "DOC":
        return V.document_code(rng), None, {}
    if slot == "HASH":  # a checksum or commit of a public file
        return "".join(rng.choice("0123456789abcdef") for _ in range(rng.choice([7, 40, 64]))), None, {}
    if slot == "DATE":
        if variant == "public":
            return V.date(rng, years=(2000, 2029), with_time=False), None, {}
        return V.date(rng, with_time=True if variant == "time" else None), "DATE", {"personal": True}
    if slot == "ADDRESS":
        return V.address(rng), "ADDRESS", {}
    if slot == "ORG_NAMED":
        return rng.choice(T.ORG_NAMED), None, {}
    if slot == "STREET_NAMED":
        return rng.choice(T.STREET_NAMED), None, {}
    if slot == "AMOUNT":
        return rng.choice([f"{rng.randint(10, 99999):,}".replace(",", "."), f"{rng.randint(1, 9999)},{rng.randint(0, 99):02d}"]), None, {}
    if slot == "VERSION":
        return rng.choice([f"v{rng.randint(1, 9)}.{rng.randint(0, 20)}.{rng.randint(0, 30)}", f"{rng.randint(1, 5)}.{rng.randint(0, 9)}.{rng.randint(0, 9)}-rc{rng.randint(1, 5)}"]), None, {}
    if slot == "LAW":
        return rng.choice(T.LAWS), None, {}
    if slot == "STOCK":
        return rng.choice([f"STK-{rng.randint(10000, 99999)}", f"SKU{rng.randint(100000, 999999)}", f"{rng.choice(['A', 'B', 'X'])}{rng.randint(100, 999)}-{rng.randint(10, 99)}"]), None, {}
    if slot == "CITY":
        return rng.choice(T.CITIES), None, {}
    raise ValueError(f"unknown slot {slot}")


def fill(template: str, rng: random.Random, source: str) -> dict:
    template = template.replace("{{", "\x00").replace("}}", "\x01")
    text, spans, at = "", [], 0
    for m in SLOT.finditer(template):
        text += template[at:m.start()]
        slot, variant, case, glued = m.groups()
        value, label, attrs = value_of(slot, variant, rng)
        if case:
            written, masked = V.inflect(value, case, None if glued else rng.choice(APOSTROPHES))
        else:
            written, masked = value, value
        start = len(text)
        text += written
        if label:
            spans.append([start, start + len(masked), label, attrs])
        at = m.end()
    text += template[at:]
    return {"text": text.replace("\x00", "{").replace("\x01", "}"), "spans": spans, "source": source}


def templatise(row) -> str | None:
    """A dataset row as a template: each value a slot, a case suffix after it a slot case."""
    text, out, at = row.text, "", 0
    for s, e, _, label in sorted(row.gold, key=lambda g: g[0]):
        if s < at:
            return None  # overlapping gold
        slot = DATASET_SLOT.get(label)
        if not slot:
            return None
        out += text[at:s].replace("{", "{{").replace("}", "}}")
        tail = re.match(rf"[{APOSTROPHES}]([a-zçğıöşü]+)", text[e:])
        if slot == "PERSON" and tail and tail.group(1) in CASE_OF:
            out += "{PERSON|" + CASE_OF[tail.group(1)] + "}"
            e += tail.end()
        else:
            out += "{" + slot + "}"
        at = e
    return out + text[at:].replace("{", "{{").replace("}", "}}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(datasets.REPO_ROOT / ".cache" / "pii_data" / "synthetic"))
    ap.add_argument("--scenario", type=int, default=20000, help="examples from the scenario templates")
    ap.add_argument("--btx24", type=int, default=1, help="fills per BTX24 training row")
    ap.add_argument("--belgin", type=int, default=4, help="fills per Belgin training row")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "scenario.jsonl", "w", encoding="utf-8") as f:
        for k in range(args.scenario):
            f.write(json.dumps(fill(T.TEMPLATES[k % len(T.TEMPLATES)], rng, "scenario"), ensure_ascii=False) + "\n")
    for name, rows, fills in (("btx24", datasets.btx24("train"), args.btx24), ("belgin", datasets.belgin("train"), args.belgin)):
        n = skipped = 0
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                tpl = templatise(row)
                if tpl is None:
                    skipped += 1
                    continue
                # the row itself (its public dates, call-centre numbers, ... stay as negatives)
                f.write(json.dumps({"text": row.text, "spans": [[s, e, DATASET_SLOT[l] if DATASET_SLOT[l] not in ("TCKN", "VKN", "IBAN") else "ID", {}] for s, e, _, l in row.gold],
                                    "source": name + "-original"}, ensure_ascii=False) + "\n")
                n += 1
                for _ in range(fills if row.gold else 0):
                    f.write(json.dumps(fill(tpl, rng, name), ensure_ascii=False) + "\n")
                    n += 1
        print(f"{name}: {n} examples ({skipped} rows skipped)")
    print(f"scenario: {args.scenario} examples from {len(T.TEMPLATES)} templates")


if __name__ == "__main__":
    main()
