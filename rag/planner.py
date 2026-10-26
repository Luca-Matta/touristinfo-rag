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
import re
from typing import Any, Literal, get_args

from pydantic import BaseModel, Field, ValidationError, field_validator

from .errors import ProviderError
from .generation.llm import ChatModel, Message
from .models import CATEGORY_LABELS, LANGUAGES, Category, Language

log = logging.getLogger(__name__)

PLACE_CATEGORIES = [c for c in Category if c is not Category.WEB_PAGE]
KNOWN_CATEGORIES = frozenset(Category)

RelativeDay = Literal["today", "tomorrow", "day_after_tomorrow"]
Weekday = Literal["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
DayRef = RelativeDay | Weekday
RELATIVE_DAYS: tuple[RelativeDay, ...] = get_args(RelativeDay)
WEEKDAYS: tuple[Weekday, ...] = get_args(Weekday)


def resolve_day(ref: DayRef, today: dt.date) -> dt.date:
    """'tomorrow' -> today + 1; a weekday -> its next occurrence, today included."""
    if ref in RELATIVE_DAYS:
        return today + dt.timedelta(days=RELATIVE_DAYS.index(ref))
    return today + dt.timedelta(days=(WEEKDAYS.index(ref) - today.weekday()) % 7)


def resolve_month_day(month_day: str, today: dt.date) -> dt.date | None:
    """'12-24' -> the next 24 December, today included; an impossible date -> None."""
    month, day = (int(x) for x in month_day.split("-"))
    for year in range(today.year, today.year + 5):  # 02-29 can be up to four years away
        try:
            d = dt.date(year, month, day)
        except ValueError:
            continue
        if d >= today:
            return d
    return None


class QueryPlan(BaseModel):
    standalone_question: str
    english_query: str
    language: Language | None = None
    categories: list[Category] = Field(default_factory=list)
    day: DayRef | None = None
    month_day: str | None = Field(default=None, pattern=r"^(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])$")
    time: dt.time | None = None
    services: list[str] = Field(default_factory=list)
    list_all: bool = False

    @field_validator("categories", mode="before")
    @classmethod
    def _drop_unknown_categories(cls, v: Any) -> Any:
        return [c for c in v if c in KNOWN_CATEGORIES] if isinstance(v, list) else v

    @property
    def queries(self) -> list[str]:
        """The phrasings to search with, original first, without duplicates."""
        return list(dict.fromkeys(q for q in (self.standalone_question, self.english_query) if q.strip()))

    @property
    def keyword_queries(self) -> list[str]:
        """BM25 only matches words of the (English) corpus: use the English query alone.

        Proper nouns survive the rewrite ("names of places unchanged"), so nothing is lost.
        """
        return [self.english_query] if self.english_query.strip() else self.queries

    def date(self, today: dt.date) -> dt.date | None:
        """The calendar day the question is about, if it names one."""
        if self.month_day is not None:
            return resolve_month_day(self.month_day, today)
        return resolve_day(self.day, today) if self.day is not None else None

    def moment(self, now: dt.datetime) -> dt.datetime:
        """The point in time the question is about: the asked day/time, else now."""
        day = self.date(now.date())
        if day is None and self.time is None:
            return now
        return dt.datetime.combine(day or now.date(), self.time or dt.time(12, 0), tzinfo=now.tzinfo)

    @property
    def asks_about_other_time(self) -> bool:
        return self.day is not None or self.month_day is not None or self.time is not None


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
        "day": _nullable({"type": "string", "enum": [*RELATIVE_DAYS, *WEEKDAYS]}),
        "month_day": _nullable({"type": "string", "description": "MM-DD"}),
        "time": _nullable({"type": "string", "description": "HH:MM, 24h"}),
        "services": {"type": "array", "items": {"type": "string"}},
        "list_all": {"type": "boolean"},
    },
    "required": [
        "standalone_question",
        "english_query",
        "language",
        "categories",
        "day",
        "month_day",
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
- day: the day the guest asks about, as said: "today" (also "tonight"), "tomorrow", \
"day_after_tomorrow" or a weekday ("monday" ... "sunday"); otherwise null. Do not convert \
it to a date.
- month_day: only if the guest names a calendar date ("on 24 December" -> "12-24"); otherwise null.
- time: only if the guest asks about a time of day, as "HH:MM" 24h ("tonight" -> "20:00"); \
otherwise null.
- services: facilities the guest requires of a place to stay, as short English terms \
matching typical accommodation services (e.g. "sauna", "dogs allowed", "indoor pool"); \
empty if none or if the guest is not looking for accommodation.
- list_all: true if the guest wants all places of a category (e.g. "list all churches").

Current local time: {now}"""

DESTINATION = re.compile(
    r"(?:,?\s+(?:in|at|near|around|im|bei|um|a|ad|di|presso|nei dintorni di))?\s*\b(?:Kiens|Chienes)"
    r"(?:\s*/\s*(?:Kiens|Chienes))?\b",
    re.IGNORECASE,
)


def strip_destination(text: str) -> str:
    """'Bakery in Kiens/Chienes open Sunday' -> 'Bakery open Sunday'.

    Every source is about the destination, so its name only dilutes the search. The prompt
    says so too, but models add it anyway, so it is removed here unless the guest wrote it.
    """
    stripped = re.sub(r"\s+([?.!,])", r"\1", DESTINATION.sub("", text)).strip()
    return stripped or text


class LLMPlanner:
    def __init__(self, llm: ChatModel) -> None:
        self.llm = llm

    def plan(self, question: str, history: list[Message], now: dt.datetime) -> QueryPlan:
        """``history`` is used as given: the caller decides how much of it is relevant."""
        categories = "\n".join(f"  {c}: {CATEGORY_LABELS[c]['en']}" for c in PLACE_CATEGORIES)
        system = PLANNER_SYSTEM.format(categories=categories, now=f"{now:%A %Y-%m-%d %H:%M}")
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in history)
        prompt = (f"Conversation so far:\n{transcript}\n\n" if transcript else "") + f"Latest message: {question}"
        try:
            data = self.llm.complete(system, [{"role": "user", "content": prompt}], schema=PLAN_SCHEMA).json()
            if data.get("language") == "other":
                data["language"] = None
            plan = QueryPlan.model_validate(data)
            if not DESTINATION.search(" ".join([question, transcript])):
                plan.standalone_question = strip_destination(plan.standalone_question)
                plan.english_query = strip_destination(plan.english_query)
        except (ProviderError, ValidationError) as exc:
            log.warning("query planning failed, searching with the raw question: %s", exc)
            return trivial_plan(question)
        log.info("plan: %s", plan.model_dump_json(exclude_defaults=True))
        return plan
