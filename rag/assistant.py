"""The question-answering pipeline: plan -> retrieve -> select sources -> answer.

    guest message + history
        │
        ▼
    LLMPlanner ── standalone question, English query, language, categories,
        │         date/time, required services, "list all"
        ▼
    HybridRetriever ── dense + BM25 over both phrasings, fused with RRF
        │
        ▼
    source selection ── structured matches first (places offering every requested
        │               service, every place of a category for "list all"), then the
        │               retrieved records, without duplicates, at most MAX_SOURCES
        ▼
    answer LLM ── system prompt + history + sources (with opening status computed
                  in code for the moment asked about)

Similarity search is good at "where can I get bread?" and bad at "which B&Bs have a
sauna?" (every B&B looks alike) or "list all churches" (k is too small). Those two
shapes are answered from the structured fields instead, which is why the planner
extracts them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

from .config import Settings
from .generation import SYSTEM_PROMPT, ChatModel, Completion, Message, build_user_message, make_chat_model
from .ingest import Corpus, load_corpus
from .models import Record
from .planner import LLMPlanner, QueryPlan, trivial_plan
from .retrieval import HybridRetriever, Mode, RecordHit

MAX_SOURCES = 12
HISTORY_MESSAGES = 6
ANSWER_MAX_TOKENS = 4000


@dataclass
class Answer:
    question: str
    plan: QueryPlan
    sources: list[Record]  # in prompt order: source [n] is sources[n - 1]
    hits: list[RecordHit] = field(default_factory=list)
    text: str | None = None  # None in retrieval-only mode
    completion: Completion | None = None


class TouristAssistant:
    def __init__(
        self,
        corpus: Corpus,
        retriever: HybridRetriever,
        settings: Settings,
        llm: ChatModel | None = None,
        planner: LLMPlanner | None = None,
    ) -> None:
        self.corpus = corpus
        self.retriever = retriever
        self.settings = settings
        self.llm = llm
        self.planner = planner
        self.tz = ZoneInfo(settings.timezone)

    @classmethod
    def from_settings(cls, settings: Settings, *, retrieve_only: bool = False) -> TouristAssistant:
        corpus = load_corpus(settings.data_dir)
        retriever = HybridRetriever(corpus, settings)
        if retrieve_only or settings.llm_provider is None:
            return cls(corpus, retriever, settings)
        llm = make_chat_model(settings)
        planner = LLMPlanner(make_chat_model(settings, planner=True))
        return cls(corpus, retriever, settings, llm, planner)

    def now(self) -> datetime:
        return datetime.now(self.tz)

    def plan(self, question: str, history: Sequence[Message] = (), now: datetime | None = None) -> QueryPlan:
        if self.planner is None:
            return trivial_plan(question)
        return self.planner.plan(question, list(history), now or self.now())

    def retrieve(self, plan: QueryPlan, mode: Mode = "hybrid") -> tuple[list[Record], list[RecordHit]]:
        """Pick the sources for a plan: structured matches first, then search results."""
        hits = self.retriever.search(
            plan.queries, categories=plan.categories, mode=mode, keyword_queries=plan.keyword_queries
        )
        structured: list[Record] = []
        if plan.services:
            structured += self.corpus.with_services(plan.services, plan.categories)
        if plan.list_all and plan.categories:
            structured += self.corpus.places(plan.categories)
        records = list({r.id: r for r in [*structured, *(h.record for h in hits)]}.values())
        return records[:MAX_SOURCES], hits

    def ask(
        self,
        question: str,
        history: Sequence[Message] = (),
        *,
        now: datetime | None = None,
        mode: Mode = "hybrid",
    ) -> Answer:
        now = now or self.now()
        plan = self.plan(question, history, now)
        sources, hits = self.retrieve(plan, mode)
        answer = Answer(question, plan, sources, hits)
        if self.llm is None:
            return answer

        messages: list[Message] = [
            *list(history)[-HISTORY_MESSAGES:],
            {"role": "user", "content": build_user_message(question, sources, plan, now)},
        ]
        answer.completion = self.llm.complete(SYSTEM_PROMPT, messages, max_tokens=ANSWER_MAX_TOKENS)
        answer.text = answer.completion.text.strip()
        return answer
