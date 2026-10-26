"""How much one build's sparse and hybrid vectors differ from another's, and what that costs an
index that is not rebuilt: the bump class of `crates/akana-core/src/embeddings/spec.rs`.

    # with build A importable as `akana` (installed, or on PYTHONPATH):
    python scripts/encoder_churn.py dump --out a.dump.gz
    # with build B:
    python scripts/encoder_churn.py dump --out b.dump.gz
    python scripts/encoder_churn.py compare a.dump.gz b.dump.gz

`dump` encodes:
- court chunks: the corpus sample's eval text in sentences packed up to 256 words, the first
  1,000 chunks of each subset. They are cut without akana, so that two builds encode the same
  texts even when their tokenizers differ;
- RagTurk: chunks and questions, as `benchmark_rag_real_data.py` builds them;
- Turkuaz: texts and questions, cached in `.cache/rag_benchmarks/turkuaz.jsonl` after the
  first streaming download.

It records each text's sparse vector (ids, f32 bits, terms) and hybrid bytes, and the build's
`akana.encoder_info()`, so a dump says which build made it.

`compare` reports:
- the share of changed vectors;
- exact-search hybrid MRR@10 of B's questions against B's documents (rebuilt) and against A's
  documents (stale);
- the class:
  - patch: no vector changed;
  - minor: at most 5 % of court hybrid vectors changed, and stale MRR@10 within 0.005 of
    rebuilt on every set;
  - major: anything else, or a different spec major or hybrid layout.

Run the commands from the repository root (the court sample is `.cache/morph_sample/eval`).
"""

import argparse
import gzip
import hashlib
import json
import re
import struct
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

COURT_SUBSETS = ["aym_bb", "aym_norm", "danistay", "emsal", "yargitay"]
CHUNKS_PER_SUBSET = 1000
CHUNK_WORDS = 256
SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")
TURKUAZ_CACHE = Path(".cache/rag_benchmarks/turkuaz.jsonl")


def chunk(decision: str) -> list[str]:
    """A decision's sentences packed into chunks of up to CHUNK_WORDS whitespace-separated
    words; a longer sentence is a chunk of its own."""
    chunks, current, words = [], [], 0
    for sentence in SENTENCE_END.split(decision.strip()):
        n = len(sentence.split())
        if current and words + n > CHUNK_WORDS:
            chunks.append(" ".join(current))
            current, words = [], 0
        if n:
            current.append(sentence)
            words += n
    if current:
        chunks.append(" ".join(current))
    return chunks


def court_chunks(sample_dir: Path) -> list[str]:
    texts = []
    for subset in COURT_SUBSETS:
        path = sample_dir / f"{subset}.txt"
        if not path.exists():
            continue
        chunks = []
        # Documents are separated by a line holding only \x1e (scripts/morph_corpus_sample.py).
        for decision in path.read_text(encoding="utf-8").split("\n\x1e\n"):
            chunks.extend(chunk(decision))
            if len(chunks) >= CHUNKS_PER_SUBSET:
                break
        texts.extend(chunks[:CHUNKS_PER_SUBSET])
    return texts


def turkuaz() -> tuple[list[dict], list[dict]]:
    """Turkuaz texts and questions, from the on-disk cache (written on the first run)."""
    if not TURKUAZ_CACHE.exists():
        from benchmark_rag_real_data import fetch_turkuaz_dataset

        docs, queries = fetch_turkuaz_dataset(max_samples=500)
        if not queries:
            return [], []
        TURKUAZ_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with TURKUAZ_CACHE.open("w", encoding="utf-8") as f:
            for d in docs:
                f.write(json.dumps({"doc": d["key"], "text": d["text"]}, ensure_ascii=False) + "\n")
            for q in queries:
                f.write(json.dumps({"question": q["question"], "gold": sorted(q["gold_keys"])}, ensure_ascii=False) + "\n")
    docs, queries = [], []
    for line in TURKUAZ_CACHE.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if "doc" in row:
            docs.append({"key": row["doc"], "text": row["text"]})
        else:
            queries.append({"question": row["question"], "gold_keys": set(row["gold"])})
    return docs, queries


def encode(texts: list[str]) -> list[dict]:
    import akana

    hybrids = akana.embed_batch_hybrid(texts) if texts else []
    out = []
    for text, hybrid in zip(texts, hybrids):
        s = akana.sparse_embed(text)
        sparse = [
            [i, struct.unpack("<I", struct.pack("<f", v))[0], t]
            for i, v, t in zip(s["indices"], s["values"], s["terms"])
        ]
        out.append({"sparse": sparse, "hybrid": bytes(hybrid).hex()})
    return out


def build_info() -> dict:
    """The build's encoder_info(); a build from before the spec (no encoder_info) had spec 1.0's
    term ids and layout."""
    import akana

    if hasattr(akana, "encoder_info"):
        return akana.encoder_info()
    return {"sparse_spec": "1.0", "hybrid_layout": 1, "fingerprint": "pre-spec build"}


def dump(args) -> None:
    from benchmark_rag_real_data import fetch_ragturk_dataset

    sets = {"court": {"docs": court_chunks(Path(args.sample)), "queries": [], "gold": []}}
    r_docs, r_queries = fetch_ragturk_dataset(max_articles=300)
    t_docs, t_queries = turkuaz()
    for name, docs, queries in (("ragturk", r_docs, r_queries), ("turkuaz", t_docs, t_queries)):
        position = {d["key"]: i for i, d in enumerate(docs)}
        sets[name] = {
            "docs": [d["text"] for d in docs],
            "queries": [q["question"] for q in queries],
            "gold": [sorted(position[k] for k in q["gold_keys"]) for q in queries],
        }
    with gzip.open(args.out, "wt", encoding="utf-8") as f:
        f.write(json.dumps({"encoder_info": build_info()}) + "\n")
        for name, s in sets.items():
            digests = {k: hashlib.sha256("\x00".join(s[k]).encode("utf-8")).hexdigest() for k in ("docs", "queries")}
            f.write(json.dumps({"set": name, "gold": s["gold"], "texts": digests}) + "\n")
            for kind in ("docs", "queries"):
                for i, v in enumerate(encode(s[kind])):
                    f.write(json.dumps({"set": name, "kind": kind, "i": i, **v}) + "\n")
    sizes = ", ".join(f"{n} {len(s['docs'])} docs / {len(s['queries'])} questions" for n, s in sets.items())
    print(f"wrote {args.out}: {sizes}; fingerprint {build_info()['fingerprint']}")


def load(path: str) -> tuple[dict, dict]:
    sets: dict = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        info = json.loads(f.readline())["encoder_info"]
        for line in f:
            row = json.loads(line)
            s = sets.setdefault(row["set"], {"docs": [], "queries": [], "gold": [], "texts": {}})
            if "kind" in row:
                s[row["kind"]].append(row)
            else:
                s["gold"] = row["gold"]
                s["texts"] = row.get("texts", {})
    return info, sets


def mrr_at_10(queries: list[bytes], docs: list[bytes], gold: list[list[int]]) -> float:
    import akana

    emb = akana.Embeddings()
    total = 0.0
    for q, g in zip(queries, gold):
        scored = sorted(range(len(docs)), key=lambda i: (emb.distance_hybrid(q, docs[i]), i))[:10]
        rank = next((r for r, i in enumerate(scored, 1) if i in g), None)
        total += 1.0 / rank if rank else 0.0
    return total / max(1, len(queries))


def compare(args) -> int:
    info_a, a = load(args.old)
    info_b, b = load(args.new)
    print(f"A {args.old}: spec {info_a['sparse_spec']}, fingerprint {info_a['fingerprint']}")
    print(f"B {args.new}: spec {info_b['sparse_spec']}, fingerprint {info_b['fingerprint']}")
    # Both dumps must encode the same texts, or the comparison measures the data, not the build.
    differ = [
        f"{name} {kind}"
        for name in sorted(set(a) | set(b))
        for kind in ("docs", "queries")
        if a.get(name, {}).get("texts", {}).get(kind) != b.get(name, {}).get("texts", {}).get(kind)
    ]
    if differ:
        print(f"the dumps encode different texts ({', '.join(differ)}): dump both builds from the same data")
        return 2
    major = (
        info_a["sparse_spec"].split(".")[0] != info_b["sparse_spec"].split(".")[0]
        or info_a["hybrid_layout"] != info_b["hybrid_layout"]
    )
    changed_any = False
    court_share = 0.0
    worst_gap = 0.0
    for name in sorted(set(a) & set(b)):
        sa, sb = a[name], b[name]
        for kind in ("docs", "queries"):
            va, vb = sa[kind], sb[kind]
            if not va:
                continue
            hybrid = sum(x["hybrid"] != y["hybrid"] for x, y in zip(va, vb))
            sparse = sum(x["sparse"] != y["sparse"] for x, y in zip(va, vb))
            terms = sum({t for *_, t in x["sparse"]} != {t for *_, t in y["sparse"]} for x, y in zip(va, vb))
            changed_any |= hybrid > 0 or sparse > 0
            print(f"{name} {kind}: hybrid changed {hybrid}/{len(va)} ({100 * hybrid / len(va):.2f} %), "
                  f"sparse values {sparse} ({100 * sparse / len(va):.2f} %), term sets {terms} ({100 * terms / len(va):.2f} %)")
            if name == "court" and kind == "docs":
                court_share = hybrid / len(va)
        if sb["queries"] and not major:
            q = [bytes.fromhex(x["hybrid"]) for x in sb["queries"]]
            rebuilt = mrr_at_10(q, [bytes.fromhex(x["hybrid"]) for x in sb["docs"]], sb["gold"])
            stale = mrr_at_10(q, [bytes.fromhex(x["hybrid"]) for x in sa["docs"]], sb["gold"])
            worst_gap = max(worst_gap, abs(rebuilt - stale))
            print(f"{name}: hybrid MRR@10 rebuilt {rebuilt:.4f}, stale {stale:.4f} (gap {rebuilt - stale:+.4f})")
    if major:
        cls = "major"
    elif not changed_any:
        cls = "patch"
    elif court_share <= 0.05 and worst_gap <= 0.005:
        cls = "minor"
    else:
        cls = "major"
    print(f"class: {cls} (court hybrid vectors changed {100 * court_share:.2f} %, largest stale gap {worst_gap:.4f})")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--out", required=True)
    d.add_argument("--sample", default=".cache/morph_sample/eval")
    c = sub.add_parser("compare")
    c.add_argument("old")
    c.add_argument("new")
    args = parser.parse_args()
    sys.exit({"dump": dump, "compare": compare}[args.command](args) or 0)


if __name__ == "__main__":
    main()
