"""The rule-based dependency parser on held-out UD Turkish: unlabelled and labelled attachment
(UAS, LAS) per treebank's test split, beside the baseline that attaches every word to the next
(review 3 PARSER3-01, PARSER3-07).

    python scripts/parser_ud_eval.py [--min-las N] [--json OUT.json]
    python scripts/parser_ud_eval.py --light-verbs crates/akana-core/src/parser/light_verbs.txt

with akana importable (the extension built from this checkout) and the pinned treebanks in
.cache/morph_sources/ud (scripts/fetch_morph_sources.py). The parser gets the treebank's word
forms (a multiword token's parts), so tokenization does not count. LAS compares the universal
relation (nsubj for nsubj:pass). A sentence whose parse has another number of words than the
treebank's is not scored, and its words count as wrong: the scores are over every word, so that
the parser cannot gain by failing on hard sentences (review 4 CONTRACT4-07). With --min-las it
exits 1 when the overall LAS is below N, as a ratchet for the gates. Only test splits are read;
IMST is evaluation-only data.

--light-verbs writes the parser's light verbs instead, from the train splits of the six
treebanks other than IMST: the verbs (by lemma) with at least 15 bare nouns right before them
(NOUN, nominative, no plural, no possessive) that are more often a compound with the verb than
its object or its subject. It needs no akana.
"""

import argparse
import collections
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UD = os.path.join(ROOT, ".cache", "morph_sources", "ud")


def sentences(path):
    words = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if words:
                    yield words
                words = []
                continue
            if line.startswith("#"):
                continue
            cols = line.split("\t")
            if "-" in cols[0] or "." in cols[0]:
                continue
            words.append((cols[1], int(cols[6]), cols[7].split(":")[0]))
    if words:
        yield words


def rows(path):
    """Each sentence's word rows (CoNLL-U columns), multiword tokens and empty nodes left out."""
    sentence = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if sentence:
                    yield sentence
                sentence = []
            elif not line.startswith("#"):
                cols = line.split("\t")
                if "-" not in cols[0] and "." not in cols[0]:
                    sentence.append(cols)
    if sentence:
        yield sentence


def light_verbs(out):
    by_verb = collections.defaultdict(collections.Counter)
    for path in sorted(glob.glob(os.path.join(UD, "*", "*-ud-train.conllu"))):
        if "IMST" in path:
            continue
        for sentence in rows(path):
            for noun, verb in zip(sentence, sentence[1:]):
                feats = noun[5]
                if (
                    noun[3] == "NOUN"
                    and verb[3] == "VERB"
                    and "Case=Nom" in feats
                    and "Number=Plur" not in feats
                    and "Number[psor]" not in feats
                ):
                    attached = int(noun[6]) == int(verb[0])
                    by_verb[verb[2]][noun[7].split(":")[0] if attached else "other"] += 1
    verbs = sorted(
        (
            v
            for v, c in by_verb.items()
            if sum(c.values()) >= 15 and c["compound"] > max(c["obj"], c["nsubj"])
        ),
        key=lambda v: (-by_verb[v]["compound"], v),
    )
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(
            "# The dependency parser's light verbs: a bare noun right before one is a compound\n"
            "# with it. Written by scripts/parser_ud_eval.py --light-verbs from the UD Turkish\n"
            "# train splits (CC BY-SA 4.0, see THIRD_PARTY_NOTICES). One verb root per line.\n"
        )
        f.writelines(v + "\n" for v in verbs)
    print(f"{len(verbs)} light verbs written to {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--min-las", type=float, help="fail below this overall LAS (%%)")
    parser.add_argument("--json", help="write the figures here")
    parser.add_argument("--light-verbs", metavar="OUT", help="write the parser's light verbs")
    args = parser.parse_args()
    if args.light_verbs:
        light_verbs(args.light_verbs)
        return
    import akana

    dep = akana.DependencyParser()
    totals = collections.Counter()
    by_bank = {}
    by_rel = collections.defaultdict(collections.Counter)
    for path in sorted(glob.glob(os.path.join(UD, "*", "*-ud-test.conllu"))):
        bank = os.path.basename(os.path.dirname(path)).removeprefix("UD_Turkish-")
        c = collections.Counter()
        for words in sentences(path):
            out = dep.parse_conllu([w for w, _, _ in words])
            rows = [
                line.split("\t") for line in out.splitlines() if line and not line.startswith("#")
            ]
            if len(rows) != len(words):
                c["skipped"] += 1
                scored = sum(rel != "punct" for _, _, rel in words)
                c["words"] += scored
                c["skipped_words"] += scored
                continue
            # The baseline attaches each word to the next word that is not punctuation, the
            # last to the root (review 4 DOWN4-04).
            content = [j for j, (w, _, _) in enumerate(words, 1) if any(ch.isalnum() for ch in w)]
            for k, ((_, head, rel), row) in enumerate(zip(words, rows)):
                if rel == "punct":
                    continue
                got_head, got_rel = int(row[6]), row[7].split(":")[0]
                c["words"] += 1
                c["uas"] += got_head == head
                c["las"] += got_head == head and got_rel == rel
                c["next"] += head == next((j for j in content if j > k + 1), 0)
                by_rel[rel]["gold"] += 1
                by_rel[rel]["right"] += got_head == head and got_rel == rel
        by_bank[bank] = c
        totals.update(c)
    pct = lambda c, k: 100.0 * c[k] / max(c["words"], 1)  # noqa: E731
    print(f"{'treebank':12} {'words':>7} {'UAS':>6} {'LAS':>6} {'next-word UAS':>14}  (next word: not punctuation)")
    for bank, c in [*by_bank.items(), ("all", totals)]:
        uas, las, nxt = pct(c, "uas"), pct(c, "las"), pct(c, "next")
        print(f"{bank:12} {c['words']:7d} {uas:6.1f} {las:6.1f} {nxt:14.1f}")
    print(f"sentences not scored: {totals['skipped']} ({totals['skipped_words']} words, counted wrong)")
    print("relations (gold count, labelled attachment %):")
    for rel, c in sorted(by_rel.items(), key=lambda kv: -kv[1]["gold"])[:16]:
        print(f"  {rel:10} {c['gold']:6d} {100.0 * c['right'] / c['gold']:5.1f}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"banks": by_bank, "all": totals, "relations": by_rel}, f, indent=1)
    las = pct(totals, "las")
    if args.min_las is not None and las < args.min_las:
        print(f"LAS {las:.1f} is below {args.min_las}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
