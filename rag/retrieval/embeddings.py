"""Embedding models behind one small protocol."""

from __future__ import annotations

from typing import Protocol

from ..config import Settings
from ..errors import ConfigError, provider_errors

OPENAI_BATCH = 256


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbedder:
    """``text-embedding-3-small`` by default: multilingual, cheap ($0.02 / 1M tokens)."""

    def __init__(self, model: str) -> None:
        from openai import OpenAI

        self.client = OpenAI()
        self.name = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        with provider_errors("OpenAI embeddings"):
            for i in range(0, len(texts), OPENAI_BATCH):
                resp = self.client.embeddings.create(model=self.name, input=texts[i : i + OPENAI_BATCH])
                vectors += [d.embedding for d in resp.data]
        return vectors


class LocalEmbedder:
    """Chroma's bundled all-MiniLM-L6-v2 (ONNX, ~80 MB, downloaded once).

    Free and offline, which makes it the right default for tests and CI, but it is an
    English model: German and Italian questions then lean on keyword search and on the
    query planner's English rewrite.
    """

    name = "all-MiniLM-L6-v2"

    def __init__(self) -> None:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        self._fn = DefaultEmbeddingFunction()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(x) for x in v] for v in self._fn(texts)]


def make_embedder(settings: Settings) -> Embedder:
    match settings.embedding_provider:
        case "openai":
            return OpenAIEmbedder(settings.openai_embedding_model)
        case "local":
            return LocalEmbedder()
    raise ConfigError(f"Unknown EMBEDDING_PROVIDER '{settings.embedding_provider}' (use openai or local)")
