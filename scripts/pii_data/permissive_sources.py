"""Rebuilds the three real-text training domains whose sources do not allow publishing the trained
weights: Wikipedia units cut from tr.wikipedia.org's own text (CC BY-SA 4.0) instead of RagTurk's
packaging (CC BY-NC-SA), maths word problems translated by us from GSM8K (MIT) instead of gsm8k_tr
(no licence), and product reviews from Turkish NLP Suite's MusteriYorumlari (CC BY-SA 4.0 as its
curators declare) instead of turkish_product_reviews (no licence).

The Wikipedia units keep v7's labels (their text is unchanged); the word problems and reviews go to
the teacher workflow through the pool p2_pool.jsonl (teacher_batches.py, teacher_collect.py).

    python scripts/pii_data/permissive_sources.py fetch-wikipedia       # article texts + revision ids
    python scripts/pii_data/permissive_sources.py wikipedia             # labels/p2_wikipedia.jsonl
    python scripts/pii_data/permissive_sources.py translate-batches     # batches/p2_translate/T_*.jsonl
    python scripts/pii_data/permissive_sources.py collect-translations --journal <journal.jsonl>
    python scripts/pii_data/permissive_sources.py pool                  # p2_pool.jsonl: word problems + reviews
    python scripts/pii_data/permissive_sources.py pool-dates            # p3_pool.jsonl: reviews that mention a date
    python scripts/pii_data/permissive_sources.py keep                  # labels/silver{1,2}_keep.jsonl

Run with `python -I` (it reads downloaded data); it finds its sibling sample_real.py itself.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sample_real  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
REAL = ROOT / ".cache" / "pii_data" / "real"
LABELS = REAL / "labels"
P2 = REAL / "p2"
SEED = 20261105
OLD = ("silver1_A_cit", "silver2_A_cit")  # v7's teacher-labelled real text
REBUILT = ("wikipedia", "reviews", "word_problems")
GSM8K = ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866", "main/train-00000-of-00001.parquet")
REVIEWS = ("turkish-nlp-suite/MusteriYorumlari", "7579c67929d30832253649fb21bab99185c8b9cd", "data/train.jsonl")
WIKI_API = "https://tr.wikipedia.org/w/api.php"
USER_AGENT = "akana-pii-data/0.1 (+https://github.com/altaidevorg/akana)"
# the evaluation side: no rebuilt unit may share text with these
EVAL_FILES = ("test.jsonl", "dev.jsonl", "dev2.jsonl", "gold_dev.jsonl", "gold_dev2.jsonl", "gold_test.jsonl")


def old_rows(domain: str | None = None) -> list[tuple[str, dict]]:
    out = []
    for name in OLD:
        for line in open(LABELS / f"{name}.jsonl", encoding="utf-8"):
            r = json.loads(line)
            if domain is None or r["id"].split(":", 1)[0] == domain:
                out.append((name, r))
    return out


def write_jsonl(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def collapse(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


# --- Wikipedia --------------------------------------------------------------------------------

def fetch_article(title: str) -> dict:
    q = {"action": "query", "format": "json", "formatversion": "2", "redirects": "1", "prop": "extracts|revisions",
         "explaintext": "1", "rvprop": "ids|timestamp", "titles": title, "maxlag": "5"}
    req = urllib.request.Request(WIKI_API + "?" + urllib.parse.urlencode(q), headers={"User-Agent": USER_AGENT})
    err: Exception | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.load(r)
            if data.get("error", {}).get("code") == "maxlag":
                time.sleep(5 * (attempt + 1))
                continue
            page = data["query"]["pages"][0]
            rev = (page.get("revisions") or [{}])[0]
            return {"title": page.get("title"), "pageid": page.get("pageid"), "revid": rev.get("revid"),
                    "timestamp": rev.get("timestamp"), "text": page.get("extract") or "", "missing": page.get("missing", False)}
        except Exception as e:  # noqa: BLE001 - retried, then raised
            err = e
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"{title}: {err}")


def fetch_wikipedia(args):
    """The current plain text and revision id of every article behind v7's Wikipedia units, from
    tr.wikipedia.org's API (articles already fetched are kept)."""
    meta = {}
    for p in glob.glob(str(ROOT / ".cache" / "rag_benchmarks" / "ragturk" / "*.json")):
        d = json.load(open(p, encoding="utf-8"))
        if isinstance(d, dict) and "article" in d:
            meta["wiki-" + d["article"]["id"]] = (d["article"]["title"], d["article"]["url"])
    keys = sorted({r["id"].split(":")[1] for _, r in old_rows("wikipedia")})
    path = P2 / "wikipedia_articles.jsonl"
    done = {json.loads(l)["key"] for l in open(path, encoding="utf-8")} if path.exists() else set()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        for k, key in enumerate(keys):
            if key in done:
                continue
            title, url = meta[key]
            if "/wiki/" in url:
                title = urllib.parse.unquote(url.rsplit("/wiki/", 1)[1]).replace("_", " ")
            r = fetch_article(title)
            r.update({"key": key, "url": url})
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            if k % 20 == 0:
                print(f"{k}/{len(keys)} {title}: revision {r['revid']}")
            time.sleep(0.5)
    print(f"{len(keys)} articles in {path}")


def wikipedia(args):
    """v7's Wikipedia units, each cut from its article's text on tr.wikipedia.org (whitespace
    collapsed as v7's units were); a unit not found word for word is dropped. The labels are v7's:
    the text is the same."""
    arts = {r["key"]: r for r in map(json.loads, open(P2 / "wikipedia_articles.jsonl", encoding="utf-8"))}
    rows, used, dropped = [], Counter(), 0
    for _, r in old_rows("wikipedia"):
        key = r["id"].split(":")[1]
        art = arts.get(key)
        m = None
        if art and not art.get("missing"):
            pattern = r"\s+".join(map(re.escape, r["text"].split()))
            m = re.search(pattern, art["text"])
        if not m or collapse(m.group(0)) != r["text"]:
            dropped += 1
            continue
        text = collapse(m.group(0))
        k = r["id"].rsplit(":", 1)[1]
        source = {"title": art["title"], "pageid": art["pageid"], "revid": art["revid"],
                  "url": f"https://tr.wikipedia.org/w/index.php?oldid={art['revid']}", "offset": m.start()}
        rows.append({"id": f"wikipedia:trwiki-{art['pageid']}-{art['revid']}:{k}", "domain": "wikipedia", "text": text,
                     "spans": r["spans"], "problems": [], "source": source})
        used[(art["title"], art["pageid"], art["revid"])] += 1
    n = write_jsonl(LABELS / "p2_wikipedia.jsonl", rows)
    with open(P2 / "wikipedia_attribution.tsv", "w", encoding="utf-8", newline="\n") as f:
        f.write("title\tpageid\trevision\tpermalink\tunits\n")
        for (title, pageid, revid), units in sorted(used.items()):
            f.write(f"{title}\t{pageid}\t{revid}\thttps://tr.wikipedia.org/w/index.php?oldid={revid}\t{units}\n")
    spans = sum(len(r["spans"]) for r in rows)
    print(f"{n} Wikipedia units from {len(used)} articles ({spans} labelled values); {dropped} dropped (not in the article)")


# --- word problems ----------------------------------------------------------------------------

def hub_file(spec: tuple[str, str, str]) -> Path:
    from huggingface_hub import hf_hub_download
    repo, revision, name = spec
    return Path(hf_hub_download(repo, name, repo_type="dataset", revision=revision))


def translate_batches(args):
    """GSM8K train questions sampled for translation, as batch files for the workflow."""
    import pyarrow.parquet as pq
    questions = pq.read_table(hub_file(GSM8K)).column("question").to_pylist()
    rows = sorted(random.Random(SEED).sample(range(len(questions)), args.questions))
    out = REAL / "batches" / "p2_translate"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("T_*.jsonl"):
        old.unlink()
    paths = []
    for b, k in enumerate(range(0, len(rows), args.size)):
        path = out / f"T_{b:03d}.jsonl"
        write_jsonl(path, ({"id": f"gsm8k-train-{i}", "text": questions[i]} for i in rows[k:k + args.size]))
        paths.append(str(path))
    print(json.dumps(paths))


def collect_translations(args):
    """The workflow's translations ({"items": [{"id", "tr"}]}) from its journal, with the English."""
    import pyarrow.parquet as pq
    questions = pq.read_table(hub_file(GSM8K)).column("question").to_pylist()
    wanted = [json.loads(l)["id"] for p in sorted((REAL / "batches" / "p2_translate").glob("T_*.jsonl"))
              for l in open(p, encoding="utf-8")]
    found = {}

    def walk(obj):
        if isinstance(obj, dict):
            items = obj.get("items")
            if isinstance(items, list) and items and all(isinstance(i, dict) and "id" in i and "tr" in i for i in items):
                for i in items:
                    found[i["id"]] = i["tr"]
                return
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)
        elif isinstance(obj, str) and '"items"' in obj:
            try:
                walk(json.loads(obj))
            except ValueError:
                pass

    for line in open(args.journal, encoding="utf-8"):
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if rec.get("type") == "result":
            walk(rec.get("result"))
    rows, missing, flagged = [], [], []
    for uid in wanted:
        tr = (found.get(uid) or "").strip()
        if not tr:
            missing.append(uid)
            continue
        en = questions[int(uid.rsplit("-", 1)[1])]
        rows.append({"id": uid, "en": en, "tr": tr})
        # checks for a reader, not filters: the digits should survive translation (separators
        # aside), and so should the names, which are kept as written
        if Counter(re.findall(r"\d+", en)) != Counter(re.findall(r"\d+", tr)):
            flagged.append({"id": uid, "check": "digits", "en": en, "tr": tr})
        names = {w for w in re.findall(r"(?<=[a-z,;] )[A-Z][a-z]{2,}", en)}
        lost = sorted(w for w in names if w not in tr)
        if lost:
            flagged.append({"id": uid, "check": f"names {lost}", "en": en, "tr": tr})
    n = write_jsonl(P2 / "gsm8k_tr.jsonl", rows)
    write_jsonl(P2 / "gsm8k_tr_flagged.jsonl", flagged)
    kinds = Counter(f["check"].split()[0] for f in flagged)
    print(f"{n} translations of {len(wanted)}; missing {missing[:10]}{'...' if len(missing) > 10 else ''}; "
          f"flagged for reading: {dict(kinds)} -> {P2 / 'gsm8k_tr_flagged.jsonl'}")


# --- the pool for the teacher -----------------------------------------------------------------

def sample_real_lower(s: str) -> str:
    return s.replace("I", "ı").replace("İ", "i").lower()


def eval_shingles() -> tuple[set, set]:
    """8-word shingles and whole short texts of every dev and test unit and gold row."""
    long_, short = set(), set()
    for name in EVAL_FILES:
        path = REAL / name
        if not path.exists():
            continue
        for line in open(path, encoding="utf-8"):
            text = json.loads(line)["text"]
            words = re.findall(r"\w+", sample_real_lower(text))
            short.add(" ".join(words))
            long_ |= {" ".join(words[i:i + 8]) for i in range(len(words) - 7)}
    return long_, short


def leaks(text: str, long_: set, short: set) -> bool:
    words = re.findall(r"\w+", sample_real_lower(text))
    if " ".join(words) in short:
        return True
    return any(" ".join(words[i:i + 8]) in long_ for i in range(len(words) - 7))


def pool(args):
    """Units for the teacher: every unit of the translated word problems, and review units of
    MusteriYorumlari sampled as sample_real.py samples a domain (70 % rich)."""
    names = sample_real.given_names()
    long_, short = eval_shingles()
    units, report = [], Counter()
    for r in map(json.loads, open(P2 / "gsm8k_tr.jsonl", encoding="utf-8")):
        for k, unit in enumerate(sample_real.units_of(r["tr"])):
            if leaks(unit, long_, short):
                report["word problems: leak"] += 1
                continue
            units.append({"id": f"word_problems:{r['id']}-tr:{k}", "domain": "word_problems",
                          "rich": sample_real.is_rich(unit, names), "text": unit})
    report["word problems"] = len(units)
    reviews = [json.loads(l)["text"] for l in open(hub_file(REVIEWS), encoding="utf-8")]
    rich, plain, seen = [], [], set()
    for row, text in enumerate(reviews):
        for k, unit in enumerate(sample_real.units_of(text or "")):
            key = unit.lower()
            if key in seen:
                continue
            seen.add(key)
            if leaks(unit, long_, short):
                report["reviews: leak"] += 1
                continue
            u = {"id": f"reviews:musteri-train-{row}:{k}", "domain": "reviews", "rich": False, "text": unit}
            if sample_real.is_rich(unit, names):
                u["rich"] = True
                rich.append(u)
            else:
                plain.append(u)
    rng = random.Random(SEED)
    rng.shuffle(rich)
    rng.shuffle(plain)
    n_rich = min(len(rich), round(args.reviews * 0.7))
    chosen = rich[:n_rich] + plain[:args.reviews - n_rich]
    report["reviews"] = len(chosen)
    report["reviews rich"] = sum(u["rich"] for u in chosen)
    units += chosen
    rng.shuffle(units)
    n = write_jsonl(REAL / "p2_pool.jsonl", units)
    print(f"{n} units -> {REAL / 'p2_pool.jsonl'}: {dict(report)}")


MONTHS = r"(ocak|şubat|subat|mart|nisan|mayıs|mayis|haziran|temmuz|ağustos|agustos|eylül|eylul|ekim|kasım|kasim|aralık|aralik)"
# a day and month ("15 kasımda", "6 haziran") or a numeric date ("09.04.2016", "02.01.17")
DATE_MENTION = re.compile(r"\b\d{1,2}[./]\d{1,2}[./](?:\d{2}|\d{4})\b|\b\d{1,2}\s*" + MONTHS + r"\w*", re.I)


def pool_dates(args):
    """Round P3's addition: every MusteriYorumlari unit that mentions a date and is not in
    p2_pool.jsonl. A random sample of reviews draws almost none (129 of 69,244 units), and P2's
    review units, nearly all without a date, taught the models to leave order and delivery dates in
    reviews untagged."""
    names = sample_real.given_names()
    long_, short = eval_shingles()
    have = {json.loads(l)["text"].lower() for l in open(REAL / "p2_pool.jsonl", encoding="utf-8")}
    units, seen, report = [], set(), Counter()
    for row, text in enumerate(json.loads(l)["text"] for l in open(hub_file(REVIEWS), encoding="utf-8")):
        for k, unit in enumerate(sample_real.units_of(text or "")):
            key = unit.lower()
            if key in seen:
                continue
            seen.add(key)
            if not DATE_MENTION.search(unit):
                continue
            if key in have:
                report["already in p2_pool"] += 1
                continue
            if leaks(unit, long_, short):
                report["leak"] += 1
                continue
            units.append({"id": f"reviews:musteri-train-{row}:{k}", "domain": "reviews",
                          "rich": sample_real.is_rich(unit, names), "text": unit})
    random.Random(SEED).shuffle(units)
    n = write_jsonl(REAL / "p3_pool.jsonl", units)
    print(f"{n} units -> {REAL / 'p3_pool.jsonl'}: {dict(report)}")


def keep(args):
    """v7's teacher label files without the three rebuilt domains."""
    for name in OLD:
        out = name.replace("_A_cit", "_keep")
        rows = [r for r in map(json.loads, open(LABELS / f"{name}.jsonl", encoding="utf-8"))
                if r["id"].split(":", 1)[0] not in REBUILT]
        n = write_jsonl(LABELS / f"{out}.jsonl", rows)
        print(f"{out}: {n} units ({dict(Counter(r['id'].split(':', 1)[0] for r in rows))})")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch-wikipedia").set_defaults(fn=fetch_wikipedia)
    sub.add_parser("wikipedia").set_defaults(fn=wikipedia)
    t = sub.add_parser("translate-batches")
    t.add_argument("--questions", type=int, default=808, help="v7's word-problem units came from 808 questions")
    t.add_argument("--size", type=int, default=34)
    t.set_defaults(fn=translate_batches)
    c = sub.add_parser("collect-translations")
    c.add_argument("--journal", required=True)
    c.set_defaults(fn=collect_translations)
    p = sub.add_parser("pool")
    p.add_argument("--reviews", type=int, default=920, help="v7 had 920 review units")
    p.set_defaults(fn=pool)
    sub.add_parser("pool-dates").set_defaults(fn=pool_dates)
    sub.add_parser("keep").set_defaults(fn=keep)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
