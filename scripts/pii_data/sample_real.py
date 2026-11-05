"""Samples real Turkish text into labelling units for the PII teacher.

Ten domains: court decisions (Yargıtay, Danıştay, emsal, Anayasa Mahkemesi), Wikipedia
(RagTurk), news (Turkuaz), tweets, product reviews, voice-assistant requests (MASSIVE tr),
prompts sent to LLMs, social-media comments and translated maths word problems. A unit is a
paragraph or a run of sentences of at most ~400 characters.

Documents are split by a hash of their id before sampling: 15 % may give test units, 10 % dev
units, the rest training units, so a name in a test unit does not come back in training text from
the same decision or article. Within a pool, 70 % of the units are "rich" (a digit, an @, a
capital inside a sentence or a listed given name: where PII is likely) and 30 % are plain, so the
test pool also measures false alarms on ordinary text.

    python scripts/pii_data/sample_real.py --out .cache/pii_data/real
Writes test.jsonl, dev.jsonl, train.jsonl: {"id", "domain", "rich", "text"}, and sources.json.
"""
from __future__ import annotations

import argparse
import csv
import glob
import gzip
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HUB = Path(os.environ.get("HF_HUB_CACHE") or Path.home() / ".cache" / "huggingface" / "hub")
MAX_UNIT = 400

# domain -> (units in test, dev, train)
TARGETS = {
    "court": (360, 180, 3000),
    "wikipedia": (240, 120, 1500),
    "news": (240, 120, 1500),
    "tweets": (240, 120, 1500),
    "reviews": (120, 60, 600),
    "assistant": (240, 120, 1200),
    "prompts": (120, 60, 400),
    "social": (120, 60, 600),
    "word_problems": (120, 60, 600),
}

SOURCES = {
    "court": "Yargıtay, Danıştay, emsal and Anayasa Mahkemesi decisions (akana's morph_sample/eval)",
    "wikipedia": "RagTurk Wikipedia articles (CC BY-SA)",
    "news": "Turkuaz news (eneSadi/turkuaz-rag)",
    "tweets": "yankihue/tweets-turkish",
    "reviews": "fthbrmnby/turkish_product_reviews",
    "assistant": "MASSIVE tr (mteb/amazon_massive_intent, CC BY 4.0)",
    "prompts": "Turkish prompt-injection and over-refusal sets (beratcmn, fevziegeyurtsevenler, AltaySec)",
    "social": "Overfit-GM/turkish-toxic-language",
    "word_problems": "malhajar/gsm8k_tr-v0.2 (GSM8K, MIT)",
}


def hub(pattern: str) -> list[str]:
    return sorted(glob.glob(str(HUB / pattern)))


def court_docs():
    """Decisions of each court file, cut at the header lines that start a decision."""
    starts = {
        "yargitay": re.compile(r"^(\d+\. (Hukuk|Ceza) Dairesi|Hukuk Genel Kurulu|Ceza Genel Kurulu) .*\d{4}/\d+ E\."),
        "danistay": re.compile(r"^Danıştay .*\d{4}/\d+ E\."),
        "emsal": re.compile(r"^T\.C\.$"),
        "aym_norm": re.compile(r"^ANAYASA MAHKEMESİ KARARI$"),
        "aym_bb": re.compile(r"^TÜRKİYE CUMHURİYETİ$"),
    }
    for name, start in starts.items():
        path = ROOT / ".cache" / "morph_sample" / "eval" / f"{name}.txt"
        doc, k = [], 0
        for line in open(path, encoding="utf-8"):
            line = line.rstrip("\n")
            if start.match(line) and doc:
                yield f"{name}-{k}", "\n".join(doc)
                doc, k = [], k + 1
            doc.append(line)
        if doc:
            yield f"{name}-{k}", "\n".join(doc)


def wikipedia_docs():
    for path in sorted(glob.glob(str(ROOT / ".cache" / "rag_benchmarks" / "ragturk" / "*.json"))):
        data = json.load(open(path, encoding="utf-8"))
        if not isinstance(data, dict) or "article" not in data:
            continue  # index files of the benchmark, not articles
        article = data["article"]
        text = re.sub(r"\s*==+[^=]+==+\s*", "\n\n", article["content"])
        yield f"wiki-{article['id']}", text


def news_docs():
    for line in open(ROOT / ".cache" / "rag_benchmarks" / "turkuaz.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r.get("text"):
            yield f"news-{r['doc']}", r["text"]


def tweets_docs():
    rows = list(csv.reader(open(hub("datasets--yankihue--tweets-turkish/snapshots/*/data.csv")[0], encoding="utf-8")))[1:]
    for k, r in enumerate(rows):
        yield f"tweet-{k}", r[0]


def parquet_docs(pattern, column, prefix):
    import pyarrow.parquet as pq
    for path in hub(pattern):
        for k, text in enumerate(pq.read_table(path).column(column).to_pylist()):
            yield f"{prefix}-{Path(path).stem}-{k}", text or ""


def assistant_docs():
    for path in hub("datasets--mteb--amazon_massive_intent/snapshots/*/*/tr.json.gz"):
        split = Path(path).parent.name
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                yield f"massive-{split}-{r['id']}", r["text"]


def prompt_docs():
    files = [
        ("datasets--beratcmn--turkish-prompt-injections/snapshots/*/data/*.jsonl", "text"),
        ("datasets--fevziegeyurtsevenler--turkish-prompt-injection/snapshots/*/*.jsonl", "payload"),
        ("datasets--AltaySec--turkish-llm-injection/snapshots/*/data/*.jsonl", "prompt"),
        ("datasets--fevziegeyurtsevenler--turkish-over-refusal-set/snapshots/*/*.jsonl", "prompt"),
        ("datasets--fevziegeyurtsevenler--guardrail-arena/snapshots/*/*.jsonl", "text"),
    ]
    for pattern, key in files:
        dataset = pattern.split("/")[0].removeprefix("datasets--")
        for path in hub(pattern):
            for k, line in enumerate(open(path, encoding="utf-8")):
                r = json.loads(line)
                if r.get("language", r.get("lang", "tr")) not in ("tr", None):
                    continue
                if r.get(key):
                    yield f"prompt-{dataset}-{Path(path).stem}-{k}", r[key]


def social_docs():
    rows = csv.DictReader(open(hub("datasets--Overfit-GM--turkish-toxic-language/snapshots/*/*.csv")[0], encoding="utf-8"))
    for k, r in enumerate(rows):
        yield f"social-{k}", r["text"]


def word_problem_docs():
    yield from parquet_docs("datasets--malhajar--gsm8k_tr-v0.2/snapshots/*/data/*.parquet", "question", "gsm8k")


LOADERS = {
    "court": court_docs,
    "wikipedia": wikipedia_docs,
    "news": news_docs,
    "tweets": tweets_docs,
    "reviews": lambda: parquet_docs("datasets--fthbrmnby--turkish_product_reviews/snapshots/*/data/*.parquet", "sentence", "review"),
    "assistant": assistant_docs,
    "prompts": prompt_docs,
    "social": social_docs,
    "word_problems": word_problem_docs,
}

SENTENCE_END = re.compile(r"(?<=[.!?…])\s+(?=[\"“(]?[A-ZÇĞİÖŞÜ0-9])")


def units_of(text: str):
    """Paragraphs, or runs of their sentences, of at most MAX_UNIT characters."""
    for para in re.split(r"\n\s*\n|\n(?=[A-ZÇĞİÖŞÜ0-9(\"“-])", text):
        para = re.sub(r"[ \t]+", " ", para).strip()
        if len(para) < 20:
            continue
        if len(para) <= MAX_UNIT:
            yield para
            continue
        run = ""
        for sentence in SENTENCE_END.split(para):
            if run and len(run) + 1 + len(sentence) > MAX_UNIT:
                yield run
                run = ""
            run = f"{run} {sentence}".strip()
            if len(run) > MAX_UNIT:
                yield run[:MAX_UNIT].rsplit(" ", 1)[0]
                run = ""
        if len(run) >= 20:
            yield run


def given_names() -> set[str]:
    path = ROOT / "crates" / "akana-core" / "src" / "pii" / "person_names.txt"
    return {w.strip().lower() for w in open(path, encoding="utf-8") if w.strip() and not w.startswith("#")}


def is_rich(text: str, names: set[str]) -> bool:
    if re.search(r"\d|@|https?://|www\.", text):
        return True
    if re.search(r"[a-zçğıöşü,;:]\s+[A-ZÇĞİÖŞÜ][a-zçğıöşü]", text):
        return True
    words = re.findall(r"[a-zçğıöşü]+", text.replace("I", "ı").replace("İ", "i").lower())
    return any(w in names for w in words if len(w) > 2)


def pool_of(doc_id: str) -> str:
    h = int(hashlib.sha1(doc_id.encode("utf-8")).hexdigest(), 16) % 100
    return "test" if h < 15 else "dev" if h < 25 else "train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / ".cache" / "pii_data" / "real"))
    ap.add_argument("--seed", type=int, default=20261101)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = given_names()
    rng = random.Random(args.seed)
    pools = {"test": [], "dev": [], "train": []}
    for domain, loader in LOADERS.items():
        candidates = {"test": ([], []), "dev": ([], []), "train": ([], [])}
        seen = set()
        for doc_id, text in loader():
            pool = pool_of(f"{domain}:{doc_id}")
            for k, unit in enumerate(units_of(text)):
                key = unit.lower()
                if key in seen:
                    continue
                seen.add(key)
                rich = is_rich(unit, names)
                candidates[pool][0 if rich else 1].append({"id": f"{domain}:{doc_id}:{k}", "domain": domain, "rich": rich, "text": unit})
        for i, pool in enumerate(("test", "dev", "train")):
            want = TARGETS[domain][i]
            rich, plain = candidates[pool]
            rng.shuffle(rich)
            rng.shuffle(plain)
            n_rich = min(len(rich), round(want * 0.7))
            chosen = rich[:n_rich] + plain[:want - n_rich]
            if len(chosen) < want:
                chosen += rich[n_rich:n_rich + want - len(chosen)]
            pools[pool] += chosen
            print(f"{domain:13s} {pool:5s} {len(chosen):5d} units (rich {sum(u['rich'] for u in chosen)}; "
                  f"available {len(rich)} rich, {len(plain)} plain)")
    for pool, units in pools.items():
        rng.shuffle(units)
        with open(out / f"{pool}.jsonl", "w", encoding="utf-8") as f:
            for u in units:
                f.write(json.dumps(u, ensure_ascii=False) + "\n")
        print(f"{pool}: {len(units)} units")
    json.dump(SOURCES, open(out / "sources.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
