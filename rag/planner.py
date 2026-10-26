"""Query planning: turn a guest message (plus chat history) into a structured search plan.

One cheap LLM call with a JSON schema extracts what similarity search cannot express:

* a **standalone question** that resolves follow-ups ("and on Sunday?" -> "Is the Ladin
  museum open on Sunday?"),
* an **English search query** (the corpus is English; this is what makes German and
  Italian questions work even with an English-only embedding model),
* the guest's **language** (so the right detail-page link is chosen in code),
* **categories** to filter on, the **date/time** the question is about, required
  **services** ("sauna", "dogs allowed") and whether the guest wants a **complete list**.

Without an LLM the :func:`trivial_plan` keeps the pipeline working (retrieval only).
The LLM output is validated with pydantic; an invalid plan falls back to the trivial one.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

from .errors import ProviderError
from .generation.llm import ChatModel, Message
from .models import CATEGORY_LABELS, LANGUAGES, Category, Language

log = logging.getLogger(__name__)

PLACE_CATEGORIES = [c for c in Category if c is not Category.WEB_PAGE]


class QueryPlan(BaseModel):
    standalone_question: str
    english_query: str
    language: Language | None = None
    categories: list[Category] = Field(default_factory=list)
    date: dt.date | None = None
    time: dt.time | None = None
    services: list[str] = Field(default_factory=list)
    list_all: bool = False

    @field_validator("categories", mode="before")
    @classmethod
    def _drop_unknown_categories(cls, v: Any) -> Any:
        return [c for c in v if c in set(Category)] if isinstance(v, list) else v

    @property
    def queries(self) -> list[str]:
        """The phrasings to search with, original first, without duplicates."""
        return list(dict.fromkeys(q for q in (self.standalone_question, self.english_query) if q.strip()))

    def moment(self, now: dt.datetime) -> dt.datetime:
        """The point in time the question is about: the asked date/time, else now."""
        if self.date is None and self.time is None:
            return now
        return dt.datetime.combine(self.date or now.date(), self.time or dt.time(12, 0), tzinfo=now.tzinfo)

    @property
    def asks_about_other_time(self) -> bool:
        return self.date is not None or self.time is not None


def trivial_plan(question: str) -> QueryPlan:
    return QueryPlan(standalone_question=question, english_query=question)


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": [schema, {"type": "null"}]}


PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "standalone_question": {"type": "string"},
        "english_query": {"type": "string"},
        "language": {"type": "string", "enum": [*LANGUAGES, "other"]},
        "categories": {"type": "array", "items": {"type": "string", "enum": [str(c) for c in PLACE_CATEGORIES]}},
        "date": _nullable({"type": "string", "description": "YYYY-MM-DD"}),
        "time": _nullable({"type": "string", "description": "HH:MM, 24h"}),
        "services": {"type": "array", "items": {"type": "string"}},
        "list_all": {"type": "boolean"},
    },
    "required": [
        "standalone_question",
        "english_query",
        "language",
        "categories",
        "date",
        "time",
        "services",
        "list_all",
    ],
    "additionalProperties": False,
}

PLANNER_SYSTEM = """\
You prepare searches for a tourist information chatbot about Kiens/Chienes (South Tyrol).
Given the conversation and the guest's latest message, return JSON:
- standalone_question: the latest message rewritten so it is understandable without the \
conversation, in the guest's language. Only resolve references; add nothing else.
- english_query: a concise English search query for the same need (names of places unchanged).
  Never add the destination name (Kiens/Chienes) or other places the guest did not mention: \
every source is about the area, so it only dilutes the search.
- language: the language of the latest message (en, de, it, or other).
- categories: place categories the guest asks about; empty if none or unsure. Options:
{categories}
- date / time: only if the guest asks about a specific day or time (resolve "tomorrow", \
"on Sunday", "tonight" relative to the current time below); otherwise null.
- services: facilities the guest requires, as short English terms matching typical \
accommodation services (e.g. "sauna", "dogs allowed", "indoor pool"); empty if none \
or if the guest is not looking for a place to stay or eat.
- list_all: true if the guest wants all places of a category (e.g. "list all churches").

Current local time: {now}"""


class LLMPlanner:
    def __init__(self, llm: ChatModel) -> None:
        self.llm = llm

    def plan(self, question: str, history: list[Message], now: dt.datetime) -> QueryPlan:
        categories = "\n".join(f"  {c}: {CATEGORY_LABELS[c]['en']}" for c in PLACE_CATEGORIES)
        system = PLANNER_SYSTEM.format(categories=categories, now=f"{now:%A %Y-%m-%d %H:%M}")
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
        prompt = (f"Conversation so far:\n{transcript}\n\n" if transcript else "") + f"Latest message: {question}"
        try:
            data = self.llm.complete(system, [{"role": "user", "content": prompt}], schema=PLAN_SCHEMA).json()
            if data.get("language") == "other":
                data["language"] = None
            plan = QueryPlan.model_validate(data)
        except (ProviderError, ValidationError) as exc:
            log.warning("query planning failed, searching with the raw question: %s", exc)
            return trivial_plan(question)
        log.info("plan: %s", plan.model_dump_json(exclude_defaults=True))
        return plan
