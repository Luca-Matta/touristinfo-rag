"""Vector store + keyword index, combined with reciprocal rank fusion.

Why hybrid: the questions tourists ask contain proper nouns ("Falkenau", "Ciastel de
Tor", "Schöneck") and exact terms ("Bäckerei", "pizza") that BM25 matches perfectly,
while paraphrased or cross-lingual questions ("Wo kann ich mit Kindern wandern?") need
embeddings. Reciprocal rank fusion (RRF) merges the two ranked lists without having to
calibrate BM25 scores against cosine similarities.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field

import chromadb
from rank_bm25 import BM25Okapi

from .config import Settings
from .corpus import Chunk

RRF_K = 60  # standard constant from Cormack et al. (2009); dampens the weight of top ranks


# --------------------------------------------------------------------------- embeddings


class Embedder:
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class OpenAIEmbedder(Embedder):
    def __init__(self, model: str):
        from openai import OpenAI

        self.client = OpenAI()
        self.model = self.name = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]


class LocalEmbedder(Embedder):
    """Chroma's default all-MiniLM-L6-v2 (ONNX, ~80 MB, downloaded on first use).

    Good enough to develop offline, but it is an English model: German and Italian
    questions rely much more on the BM25 side and on the DE/IT names in the chunks.
    """

    name = "all-MiniLM-L6-v2"

    def __init__(self):
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        self.fn = DefaultEmbeddingFunction()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self.fn(texts)]


def make_embedder(settings: Settings) -> Embedder:
    provider = settings.resolved_embedding_provider()
    if provider == "openai":
        return OpenAIEmbedder(settings.openai_embedding_model)
    if provider == "local":
        return LocalEmbedder()
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {provider}")


# --------------------------------------------------------------------------- keyword search


# Function words in EN/DE/IT. Without this, question words ("where can I find ...") match
# the example questions inside every overview chunk and drown out the content words.
STOPWORDS = set(
    """a an and are at be can do does find for from has have here how i in is it me my near
    nearby of on or the there this to what when where which who with you your any some
    am an auf bei bis das dem den der die ein eine einen es gibt hier ich im ist kann man
    mit nach oder sind und von was welche welcher wie wo zu
    a al alla c che ci con da del della di dove e il in la le lo mi per posso qui si sono
    su un una uno vicino""".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase, strip accents (so 'Schoneck' matches 'Schöneck'), drop stopwords."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return [t for t in re.findall(r"\w+", text) if t not in STOPWORDS]


# --------------------------------------------------------------------------- index


def record_id(chunk_id: str) -> str:
    """'castles-0.1' -> 'castles-0' (chunks of a split record share the part before the dot)."""
    return chunk_id.rsplit(".", 1)[0]


@dataclass
class Hit:
    chunk: Chunk
    score: float
    ranks: dict[str, int] = field(default_factory=dict)  # retriever -> 1-based rank (for debugging)


class HybridIndex:
    def __init__(self, chunks: list[Chunk], settings: Settings, embedder: Embedder | None = None):
        self.settings = settings
        self.chunks = {c.id: c for c in chunks}
        self.embedder = embedder or make_embedder(settings)
        self.bm25_ids = [c.id for c in chunks]
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])
        self.collection = self._load_or_build_collection(chunks)

    def _load_or_build_collection(self, chunks: list[Chunk]):
        """Persist embeddings in Chroma; rebuild only when the corpus or embedding model changes."""
        corpus_hash = hashlib.sha256(
            "".join(c.id + c.text for c in chunks).encode()
        ).hexdigest()[:16]
        client = chromadb.PersistentClient(path=str(self.settings.chroma_dir))
        name = "kiens_" + re.sub(r"[^a-zA-Z0-9]+", "_", self.embedder.name)
        try:
            col = client.get_collection(name)
            if (col.metadata or {}).get("corpus_hash") == corpus_hash:
                return col
            client.delete_collection(name)
        except Exception:  # collection does not exist yet (exception type differs across Chroma versions)
            pass

        col = client.create_collection(
            name,
            metadata={"corpus_hash": corpus_hash},
            configuration={"hnsw": {"space": "cosine"}},
        )
        col.add(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks],
            embeddings=self.embedder.embed([c.text for c in chunks]),
        )
        return col

    def dense_search(self, query: str, n: int) -> list[str]:
        res = self.collection.query(query_embeddings=self.embedder.embed([query]), n_results=n)
        return res["ids"][0]

    def keyword_search(self, query: str, n: int) -> list[str]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.bm25_ids[i] for i in ranked[:n] if scores[i] > 0]

    def search(self, query: str, k: int | None = None, mode: str = "hybrid") -> list[Hit]:
        """mode: 'hybrid' (default), 'dense' or 'keyword' (the latter two for evaluation/debugging)."""
        k = k or self.settings.top_k
        n = self.settings.candidates
        ranked_lists = {}
        if mode in {"hybrid", "dense"}:
            ranked_lists["dense"] = self.dense_search(query, n)
        if mode in {"hybrid", "keyword"}:
            ranked_lists["keyword"] = self.keyword_search(query, n)

        fused: dict[str, Hit] = {}
        for retriever, ids in ranked_lists.items():
            seen_records = set()
            for cid in ids:
                # Parts of one split record count as one result, ranked by their best part,
                # so a long record cannot take several of the k slots.
                record = record_id(cid)
                if record in seen_records:
                    continue
                seen_records.add(record)
                rank = len(seen_records)
                hit = fused.setdefault(record, Hit(self.chunks[cid], 0.0))
                hit.score += 1.0 / (RRF_K + rank)
                hit.ranks[retriever] = rank
        return sorted(fused.values(), key=lambda h: h.score, reverse=True)[:k]

    def siblings(self, chunk: Chunk) -> list[Chunk]:
        """All parts of the record a chunk belongs to, in order ('castles-0.1' -> [castles-0.0, castles-0.1])."""
        record = record_id(chunk.id)
        return [c for cid, c in self.chunks.items() if record_id(cid) == record]
