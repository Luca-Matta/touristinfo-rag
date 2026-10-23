"""The RAG pipeline: load corpus -> index -> retrieve -> generate."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from .config import Settings
from .corpus import load_corpus
from .generation import SYSTEM_PROMPT, LLM, build_user_message, format_context, make_llm
from .retrieval import Hit, HybridIndex


@dataclass
class Answer:
    text: str
    hits: list[Hit]
    prompt: str  # the exact user message sent to the LLM, for debugging


class RagPipeline:
    def __init__(self, settings: Settings | None = None, llm: LLM | None = None):
        self.settings = settings or Settings()
        self.chunks = load_corpus(self.settings.data_dir)
        self.index = HybridIndex(self.chunks, self.settings)
        self._llm = llm

    @property
    def llm(self) -> LLM:
        if self._llm is None:  # created lazily so retrieval works without an API key
            self._llm = make_llm(self.settings)
        return self._llm

    def now(self) -> datetime:
        return datetime.now(ZoneInfo(self.settings.timezone)).replace(tzinfo=None)

    def retrieve(self, question: str, k: int | None = None) -> list[Hit]:
        return self.index.search(question, k=k)

    def answer(self, question: str, k: int | None = None, now: datetime | None = None) -> Answer:
        now = now or self.now()
        hits = self.retrieve(question, k)
        prompt = build_user_message(question, format_context(hits, self.index, now), now)
        return Answer(self.llm.complete(SYSTEM_PROMPT, prompt), hits, prompt)
