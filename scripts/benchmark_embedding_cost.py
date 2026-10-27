"""Single-core embedding cost on court-decision chunks of up to 256 words (the chunks of
`encoder_churn.py`): hybrid, sparse and dense, the best of three passes over 300 chunks.

    # with akana importable, from the repository root (needs .cache/morph_sample/eval):
    python scripts/benchmark_embedding_cost.py
"""

import sys
import time
from pathlib import Path

import akana

sys.path.insert(0, str(Path(__file__).resolve().parent))
from encoder_churn import court_chunks  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    chunks = court_chunks(Path(".cache/morph_sample/eval"))[::16][:300]
    words = sum(len(c.split()) for c in chunks)
    print(f"{len(chunks)} chunks, {words / len(chunks):.0f} words each on average")
    for name, embed in (("hybrid", akana.embed_hybrid), ("sparse", akana.sparse_embed), ("dense", akana.embed)):
        best = float("inf")
        for _ in range(3):
            t = time.perf_counter()
            for c in chunks:
                embed(c)
            best = min(best, time.perf_counter() - t)
        print(f"{name:7} {len(chunks) / best:8.0f} chunks/s  {words / best:9.0f} words/s")


if __name__ == "__main__":
    main()
