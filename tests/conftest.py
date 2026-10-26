"""Shared fixtures. Nothing here touches the network: the embedder and LLM are fakes."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from rag.config import ROOT, Settings
from rag.generation.llm import Completion, Message
from rag.ingest import Corpus, load_corpus
from rag.retrieval import HybridRetriever
from rag.retrieval.keyword import tokenize

TZ = ZoneInfo("Europe/Rome")
MONDAY_10AM = datetime(2026, 10, 26, 10, 0, tzinfo=TZ)


class HashEmbedder:
    """Deterministic bag-of-words vectors: similar wording -> similar vectors."""

    name = "test-hash"
    dims = 256

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            v = [0.0] * self.dims
            for token in tokenize(text):
                v[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dims] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            vectors.append([x / norm for x in v])
        return vectors


class FakeChat:
    """Returns canned replies in order and records every call."""

    name = "fake"

    def __init__(self, *replies: str | dict[str, Any]) -> None:
        self.replies = [r if isinstance(r, str) else json.dumps(r) for r in replies]
        self.calls: list[dict[str, Any]] = []

    def complete(
        self, system: str, messages: list[Message], *, schema: dict[str, Any] | None = None, max_tokens: int = 2000
    ) -> Completion:
        self.calls.append({"system": system, "messages": messages, "schema": schema})
        return Completion(self.replies.pop(0), self.name, 10, 5)


@pytest.fixture(scope="session")
def corpus() -> Corpus:
    return load_corpus(ROOT / "data")


@pytest.fixture(scope="session")
def settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    return Settings(index_dir=Path(tmp_path_factory.mktemp("index")))


@pytest.fixture(scope="session")
def retriever(corpus: Corpus, settings: Settings) -> HybridRetriever:
    return HybridRetriever(corpus, settings, embedder=HashEmbedder())
