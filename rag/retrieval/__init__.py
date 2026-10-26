"""Hybrid retrieval: dense (Chroma) + keyword (BM25), fused per *record* with RRF.

Search can take several phrasings of one question (the original and the planner's
English rewrite) and an optional category filter. Every (phrasing, retriever) pair yields
one ranked list. Chunks are collapsed to their record inside each list, so a long record
split in two chunks cannot take two of the k slots, and the lists are fused with RRF.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from ..chunking import chunk_records
from ..config import Settings
from ..ingest import Corpus
from ..models import Category, Record
from .embeddings import Embedder, make_embedder
from .fusion import reciprocal_rank_fusion
from .keyword import KeywordIndex
from .vector_store import VectorStore

Mode = Literal["hybrid", "dense", "keyword"]


@dataclass
class RecordHit:
    record: Record
    score: float
    ranks: dict[str, int] = field(default_factory=dict)  # "dense:0" -> rank, for --debug


class HybridRetriever:
    def __init__(self, corpus: Corpus, settings: Settings, embedder: Embedder | None = None) -> None:
        self.corpus = corpus
        self.settings = settings
        self.chunks = chunk_records(corpus)
        self._record_of = {c.id: c.record_id for c in self.chunks}
        self.keyword = KeywordIndex(self.chunks)
        self.vectors = VectorStore(settings.index_dir, embedder or make_embedder(settings), self.chunks, corpus.records)

    def search(
        self,
        queries: str | Sequence[str],
        k: int | None = None,
        categories: Iterable[Category] = (),
        mode: Mode = "hybrid",
    ) -> list[RecordHit]:
        queries = [queries] if isinstance(queries, str) else [q for q in queries if q.strip()]
        k = k or self.settings.top_k
        categories = list(categories)

        hits = self._search(queries, k, categories, mode)
        if categories and len(hits) < k:  # a narrow filter must not starve the answer
            seen = {h.record.id for h in hits}
            hits += [h for h in self._search(queries, k, [], mode) if h.record.id not in seen]
        return hits[:k]

    def _search(self, queries: Sequence[str], k: int, categories: list[Category], mode: Mode) -> list[RecordHit]:
        n = self.settings.candidates
        allowed = {c.id for c in self.chunks if self.corpus[c.record_id].category in categories} if categories else None

        ranked: dict[str, list[str]] = {}
        for i, query in enumerate(queries):
            if mode in ("hybrid", "dense"):
                ranked[f"dense:{i}"] = self._records(self.vectors.search(query, n, categories))
            if mode in ("hybrid", "keyword"):
                ranked[f"keyword:{i}"] = self._records(self.keyword.search(query, n, allowed))

        return [RecordHit(self.corpus[f.id], f.score, f.ranks) for f in reciprocal_rank_fusion(ranked)[:k]]

    def _records(self, chunk_ids: Iterable[str]) -> list[str]:
        """Chunk ids -> record ids, keeping each record at the rank of its best chunk."""
        return list(dict.fromkeys(self._record_of[c] for c in chunk_ids))
