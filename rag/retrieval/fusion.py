"""Reciprocal rank fusion (Cormack, Clarke & Büttcher, 2009).

    score(d) = Σ over lists  1 / (k + rank of d in that list)

It merges ranked lists from retrievers whose scores are not comparable (BM25 vs cosine)
using ranks only. k = 60 is the constant from the paper; it flattens the difference
between rank 1 and rank 2 so that agreement across lists matters more than one list's top.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

RRF_K = 60


@dataclass
class Fused:
    id: str
    score: float = 0.0
    ranks: dict[str, int] = field(default_factory=dict)  # list name -> 1-based rank


def reciprocal_rank_fusion(ranked_lists: Mapping[str, Sequence[str]], k: int = RRF_K) -> list[Fused]:
    fused: dict[str, Fused] = {}
    for list_name, ids in ranked_lists.items():
        for rank, item in enumerate(ids, start=1):
            entry = fused.setdefault(item, Fused(item))
            entry.score += 1.0 / (k + rank)
            entry.ranks[list_name] = rank
    return sorted(fused.values(), key=lambda f: (-f.score, f.id))
