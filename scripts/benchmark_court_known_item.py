"""Court known-item retrieval: a decision's own summary section is the query and the rest of the
decision the target, among all the corpus sample's decisions cut into chunks.

    python scripts/benchmark_court_known_item.py [--sample DIR] [--json OUT] [--per-query OUT]

with akana importable, from the repository root (the sample is `.cache/morph_sample/eval`,
written by scripts/morph_corpus_sample.py from the court decisions dataset).

Queries, with their span removed from the decision before it is chunked:
- aym_bb: the paragraph under "BAŞVURUNUN KONUSU" (the application's subject);
- aym_norm: the text after "İTİRAZIN KONUSU" or "İPTAL DAVASININ KONUSU";
- danistay: the text after "İstemin Özeti";
- yargitay: the sentence that opens with "Dava, ".
Each query is cut at its paragraph's end and at 80 words, and kept with 8 words or more. The
emsal decisions, whose case headings are too general to be queries, are in the corpus only.

Documents are chunked as scripts/encoder_churn.py chunks court text (sentences packed into up
to 256 words); a query's target is any chunk of its decision. Reported: MRR@10 of BM25 over
the sparse vectors (as benchmark_rag_real_data.py's Morphological BM25 scores them), of their
dot product (as encoder_churn.py scores them) and of the hybrid distance, overall and per
subset. `--per-query` writes each query's reciprocal ranks, to compare two builds query by
query.
"""

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import akana

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from encoder_churn import chunk  # noqa: E402

SUBSETS = ["aym_bb", "aym_norm", "danistay", "yargitay", "emsal"]
HEADERS = {
    "aym_bb": re.compile(r"(?m)^\s*(?:[IVX]+\.\s*)?BAŞVURUNUN KONUSU\s*\n+(?:\d+\.\s*)?"),
    "aym_norm": re.compile(r"(?:İTİRAZIN|İPTAL DAVASININ) KONUSU\s*:?\s*"),
    "danistay": re.compile(r"İstemin Özeti\s*:\s*"),
    "yargitay": re.compile(r"(?m)^Dava, "),
}
# A paragraph ends at a blank line or at the next heading line (OLAY:, II. BAŞVURU SÜRECİ).
END = re.compile(r"\n\s*\n|\n(?=[A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜ .]{3,}:)|\n(?=[IVX]+\.\s)")
SENTENCE = re.compile(r"(?<=[.!?])\s")
MAX_WORDS, MIN_WORDS = 80, 8


def decisions(sample: Path, subset: str) -> list[str]:
    text = (sample / f"{subset}.txt").read_text(encoding="utf-8")
    return [d for d in text.split("\n\x1e\n") if d.strip()]


def summary(subset: str, decision: str) -> tuple[str, str] | None:
    """The query and the decision without it, or None when the decision has no summary."""
    header = HEADERS.get(subset)
    m = header.search(decision) if header else None
    if not m:
        return None
    rest = decision[m.end():]
    end = END.search(rest)
    span = rest[: end.start()] if end else rest
    if subset == "yargitay":
        # Only the opening sentence summarises the case.
        sentence = SENTENCE.search(span)
        span = span[: sentence.start()] if sentence else span
    words = span.split()
    if len(words) < MIN_WORDS:
        return None
    body = decision[: m.start()] + "\n" + decision[m.end() + len(span):]
    return " ".join(words[:MAX_WORDS]), body


def build(sample: Path) -> tuple[list[str], list[str], list[list[int]], list[str]]:
    docs, queries, gold, subsets = [], [], [], []
    for subset in SUBSETS:
        for decision in decisions(sample, subset):
            found = summary(subset, decision)
            body = found[1] if found else decision
            first = len(docs)
            docs.extend(chunk(body))
            if found and len(docs) > first:
                queries.append(found[0])
                gold.append(list(range(first, len(docs))))
                subsets.append(subset)
    return docs, queries, gold, subsets


def reciprocal(ranked: list[int], gold: set[int]) -> float:
    return next((1.0 / r for r, i in enumerate(ranked[:10], 1) if i in gold), 0.0)


def evaluate(docs, queries, gold) -> dict[str, list[float]]:
    sparse_docs = [akana.sparse_embed(d) for d in docs]
    sparse_queries = [akana.sparse_embed(q) for q in queries]
    postings = defaultdict(list)
    lengths = []
    for n, s in enumerate(sparse_docs):
        lengths.append(sum(s["values"]) or 1.0)
        for i, v in zip(s["indices"], s["values"]):
            postings[i].append((n, v))
    average = sum(lengths) / len(lengths)
    k1, b = 1.2, 0.75
    out = {"bm25": [], "sparse_dot": [], "hybrid": []}
    for q, g in zip(sparse_queries, gold):
        bm25, dot = defaultdict(float), defaultdict(float)
        for i, w in zip(q["indices"], q["values"]):
            plist = postings.get(i, [])
            if not plist:
                continue
            idf = max(0.1, math.log((len(docs) - len(plist) + 0.5) / (len(plist) + 0.5) + 1.0))
            for n, tf in plist:
                norm = tf * (k1 + 1.0) / (tf + k1 * (1.0 - b + b * lengths[n] / average))
                bm25[n] += idf * norm
                dot[n] += w * tf
        g = set(g)
        out["bm25"].append(reciprocal(sorted(bm25, key=lambda n: (-bm25[n], n)), g))
        out["sparse_dot"].append(reciprocal(sorted(dot, key=lambda n: (-dot[n], n)), g))
    emb = akana.Embeddings()
    hybrid_docs = [bytes(h) for h in akana.embed_batch_hybrid(docs)]
    hybrid_queries = [bytes(h) for h in akana.embed_batch_hybrid(queries)]
    for q, g in zip(hybrid_queries, gold):
        g = set(g)
        ranked = sorted(range(len(docs)), key=lambda n: (emb.distance_hybrid(q, hybrid_docs[n]), n))
        out["hybrid"].append(reciprocal(ranked[:10], g))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sample", default=".cache/morph_sample/eval")
    ap.add_argument("--json", help="write the figures here")
    ap.add_argument("--per-query", help="write each query's reciprocal ranks here")
    args = ap.parse_args()
    docs, queries, gold, subsets = build(Path(args.sample))
    print(f"{len(queries)} queries over {len(docs)} chunks", flush=True)
    rr = evaluate(docs, queries, gold)
    figures = {}
    for method, values in rr.items():
        by = defaultdict(list)
        for s, v in zip(subsets, values):
            by[s].append(v)
        figures[method] = {"all": sum(values) / len(values)} | {
            s: sum(v) / len(v) for s, v in by.items()
        }
        print(f"{method}: MRR@10 {figures[method]['all']:.4f} ("
              + ", ".join(f"{s} {figures[method][s]:.4f}" for s in by) + ")")
    if args.json:
        Path(args.json).write_text(json.dumps(figures, indent=1), encoding="utf-8")
    if args.per_query:
        Path(args.per_query).write_text(
            json.dumps({"subsets": subsets, "queries": queries, "rr": rr}, ensure_ascii=False),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
