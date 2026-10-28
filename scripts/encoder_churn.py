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
- the share of changed vectors, sparse and hybrid, per set;
- exact-search MRR@10 of B's questions against B's documents (rebuilt) and against A's
  documents (stale): hybrid vectors, and sparse vectors scored by their dot product (what a
  sparse or BM25-style index of the terms sees);
- the class:
  - patch: no vector changed;
  - minor: at most 5 % of the documents' sparse vectors and of their hybrid vectors changed in
    every set (court, RagTurk, Turkuaz), and every stale MRR@10 is within 0.005 of rebuilt;
  - major: anything else, or a different spec major, hybrid layout or dense part.

With `--declared CLASS` it exits 3 when the measured class is above the declared one, so a
release script can stop on it. `dump` refuses (exit 1) when a set is missing or incomplete:
the court sample's five subsets, the 300 pinned RagTurk articles and the 500 Turkuaz samples;
`compare` refuses (exit 2) dumps of different or missing texts (review 3 MSTAB3-03, BIND3-03).

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
            sys.exit(f"missing {path}: the court sample has five subsets (scripts/morph_corpus_sample.py)")
        chunks = []
        # Documents are separated by a line holding only \x1e (scripts/morph_corpus_sample.py).
        for decision in path.read_text(encoding="utf-8").split("\n\x1e\n"):
            chunks.extend(chunk(decision))
            if len(chunks) >= CHUNKS_PER_SUBSET:
                break
        texts.extend(chunks[:CHUNKS_PER_SUBSET])
    return texts


def turkuaz() -> tuple[list[dict], list[dict]]:
    """Turkuaz texts and questions, from the benchmark's on-disk cache (written on the first
    run)."""
    from benchmark_rag_real_data import fetch_turkuaz_dataset

    return fetch_turkuaz_dataset(max_samples=500)


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
    r_docs, r_queries = fetch_ragturk_dataset(max_articles=300, strict=True)
    t_docs, t_queries = turkuaz()
    if len(t_queries) == 0 or not r_queries:
        sys.exit("RagTurk or Turkuaz is unavailable: run benchmark_rag_real_data.py online once to cache them")
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


def sparse_values(row: dict) -> dict[int, float]:
    return {i: struct.unpack("<f", struct.pack("<I", bits))[0] for i, bits, _ in row["sparse"]}


def sparse_mrr_at_10(queries: list[dict], docs: list[dict], gold: list[list[int]]) -> float:
    """MRR@10 of the sparse vectors scored by their dot product, as a sparse index ranks them
    (documents that share no term with the question are not ranked)."""
    postings: dict[int, list[tuple[int, float]]] = {}
    for n, d in enumerate(docs):
        for i, w in sparse_values(d).items():
            postings.setdefault(i, []).append((n, w))
    total = 0.0
    for q, g in zip(queries, gold):
        scores: dict[int, float] = {}
        for i, w in sparse_values(q).items():
            for n, dw in postings.get(i, ()):
                scores[n] = scores.get(n, 0.0) + w * dw
        scored = sorted(scores, key=lambda n: (-scores[n], n))[:10]
        rank = next((r for r, n in enumerate(scored, 1) if n in g), None)
        total += 1.0 / rank if rank else 0.0
    return total / max(1, len(queries))


CLASSES = ["patch", "minor", "major"]


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
    empty = [name for name in ("court", "ragturk", "turkuaz") if not a.get(name, {}).get("docs")]
    if empty:
        print(f"the dumps hold no {', '.join(empty)} documents: dump both builds with the full data")
        return 2
    major = (
        info_a["sparse_spec"].split(".")[0] != info_b["sparse_spec"].split(".")[0]
        or info_a["hybrid_layout"] != info_b["hybrid_layout"]
        # A build from before the spec (0.8.0) records no dense model, and one from 0.9.0's
        # development no dense fingerprint: what is not recorded is not compared.
        or any(
            None not in (info_a.get(key), info_b.get(key)) and info_a[key] != info_b[key]
            for key in ("dense_model", "dense_fingerprint")
        )
    )
    changed_any = False
    worst_share = 0.0
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
            if kind == "docs":
                worst_share = max(worst_share, hybrid / len(va), sparse / len(va))
        if sb["queries"] and not major:
            q = [bytes.fromhex(x["hybrid"]) for x in sb["queries"]]
            rebuilt = mrr_at_10(q, [bytes.fromhex(x["hybrid"]) for x in sb["docs"]], sb["gold"])
            stale = mrr_at_10(q, [bytes.fromhex(x["hybrid"]) for x in sa["docs"]], sb["gold"])
            worst_gap = max(worst_gap, abs(rebuilt - stale))
            print(f"{name}: hybrid MRR@10 rebuilt {rebuilt:.4f}, stale {stale:.4f} (gap {rebuilt - stale:+.4f})")
            s_rebuilt = sparse_mrr_at_10(sb["queries"], sb["docs"], sb["gold"])
            s_stale = sparse_mrr_at_10(sb["queries"], sa["docs"], sb["gold"])
            worst_gap = max(worst_gap, abs(s_rebuilt - s_stale))
            print(f"{name}: sparse MRR@10 rebuilt {s_rebuilt:.4f}, stale {s_stale:.4f} (gap {s_rebuilt - s_stale:+.4f})")
    if major:
        cls = "major"
    elif not changed_any:
        cls = "patch"
    elif worst_share <= 0.05 and worst_gap <= 0.005:
        cls = "minor"
    else:
        cls = "major"
    print(f"class: {cls} (largest share of changed document vectors {100 * worst_share:.2f} %, "
          f"largest stale gap {worst_gap:.4f})")
    if changed_any and info_a.get("fingerprint") == info_b.get("fingerprint"):
        print("the vectors changed but the fingerprint did not: the fingerprint misses an input")
        return 4
    if args.declared and CLASSES.index(cls) > CLASSES.index(args.declared):
        print(f"the change is {cls}, above the declared {args.declared}")
        return 3
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
    c.add_argument("--declared", choices=CLASSES, help="exit 3 when the change is above this class")
    args = parser.parse_args()
    sys.exit({"dump": dump, "compare": compare}[args.command](args) or 0)


if __name__ == "__main__":
    main()
