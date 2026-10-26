"""Chroma as a persistent vector store.

Chroma only stores what similarity search needs: chunk id, vector, the chunk text (handy
when inspecting the index) and two filter fields. Records themselves live in memory, in
the :class:`~rag.ingest.Corpus`, so nothing has to be squeezed into flat metadata.

The collection is rebuilt only when its content hash changes (corpus or chunking edits)
and each embedding model gets its own collection, so switching models is a config change.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.errors import NotFoundError

from ..models import Category, Chunk, Record
from .embeddings import Embedder

log = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, path: Path, embedder: Embedder, chunks: Sequence[Chunk], records: dict[str, Record]) -> None:
        self.embedder = embedder
        client = chromadb.PersistentClient(path=str(path))
        name = "kiens_" + re.sub(r"[^a-zA-Z0-9]+", "_", embedder.name)
        digest = hashlib.sha256("\x00".join(f"{c.id}\x01{c.text}" for c in chunks).encode()).hexdigest()[:16]

        try:
            collection = client.get_collection(name)
            if (collection.metadata or {}).get("content_hash") != digest:
                log.info("index %s is stale, rebuilding", name)
                client.delete_collection(name)
                collection = self._build(client, name, digest, chunks, records)
        except NotFoundError:
            log.info("building index %s (%d chunks)", name, len(chunks))
            collection = self._build(client, name, digest, chunks, records)
        self._collection = collection

    def _build(
        self, client: ClientAPI, name: str, digest: str, chunks: Sequence[Chunk], records: dict[str, Record]
    ) -> Collection:
        collection = client.create_collection(
            name, metadata={"content_hash": digest}, configuration={"hnsw": {"space": "cosine"}}
        )
        collection.add(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[{"record_id": c.record_id, "category": str(records[c.record_id].category)} for c in chunks],
            embeddings=self.embedder.embed([c.text for c in chunks]),  # type: ignore[arg-type]
        )
        return collection

    def count(self) -> int:
        return self._collection.count()

    def search(self, query: str, n: int, categories: Iterable[Category] = ()) -> list[str]:
        """Chunk ids by cosine similarity, optionally restricted to some categories."""
        cats = [str(c) for c in categories]
        where = {"category": {"$in": cats}} if cats else None
        res = self._collection.query(
            query_embeddings=self.embedder.embed([query]),  # type: ignore[arg-type]
            n_results=min(n, self.count()),
            where=where,  # type: ignore[arg-type]
        )
        return res["ids"][0]
