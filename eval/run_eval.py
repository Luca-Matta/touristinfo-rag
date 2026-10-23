"""Retrieval evaluation: does the right source land in the top-k?

    python -m eval.run_eval            # compares dense / keyword / hybrid
    python -m eval.run_eval -k 3 -v    # verbose: list the misses

A question counts as a hit if at least one of its expected sources (matched by record
name) is in the top-k. MRR uses the rank of the first expected source. This needs no LLM,
costs nothing to run, and is the first thing to check when an answer looks wrong:
if retrieval missed, no prompt will fix it.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from rag.config import Settings
from rag.corpus import load_corpus
from rag.retrieval import HybridIndex

QUESTIONS = Path(__file__).with_name("questions.json")


def evaluate(index: HybridIndex, questions: list[dict], k: int, mode: str):
    per_lang = defaultdict(lambda: [0, 0, 0.0])  # hits, total, reciprocal-rank sum
    misses = []
    for item in questions:
        names = [h.chunk.metadata["name"] for h in index.search(item["q"], k=k, mode=mode)]
        rank = next((i for i, n in enumerate(names, 1) if n in item["expected"]), None)
        stats = per_lang[item["lang"]]
        stats[1] += 1
        if rank:
            stats[0] += 1
            stats[2] += 1 / rank
        else:
            misses.append((item["q"], names))
    return per_lang, misses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    settings = Settings()
    index = HybridIndex(load_corpus(settings.data_dir), settings)
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    print(f"embeddings: {index.embedder.name} · {len(questions)} questions · k={args.k}\n")
    print(f"{'mode':<8} {'hit@k':>7} {'MRR':>6}   " + "   ".join(f"{l:>9}" for l in ("en", "de", "it")))

    for mode in ("dense", "keyword", "hybrid"):
        per_lang, misses = evaluate(index, questions, args.k, mode)
        hits = sum(s[0] for s in per_lang.values())
        total = sum(s[1] for s in per_lang.values())
        mrr = sum(s[2] for s in per_lang.values()) / total
        langs = "   ".join(f"{per_lang[l][0]:>4}/{per_lang[l][1]:<4}" for l in ("en", "de", "it"))
        print(f"{mode:<8} {hits / total:>7.0%} {mrr:>6.2f}   {langs}")
        if args.verbose:
            for q, names in misses:
                print(f"    miss: {q!r} -> {names}")


if __name__ == "__main__":
    main()
