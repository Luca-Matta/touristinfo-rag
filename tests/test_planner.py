from datetime import date, time

import pytest

from rag.models import Category
from rag.planner import (
    PLAN_SCHEMA,
    LLMPlanner,
    QueryPlan,
    resolve_day,
    resolve_month_day,
    strip_destination,
    trivial_plan,
)

from .conftest import MONDAY_10AM, FakeChat

PLAN = {
    "standalone_question": "Ist das Museum Ladin am Sonntag geöffnet?",
    "english_query": "Museum Ladin opening hours Sunday",
    "language": "de",
    "categories": ["castle", "spaceship"],
    "day": "sunday",
    "month_day": None,
    "time": None,
    "services": [],
    "list_all": False,
}


def test_plan_is_parsed_and_validated():
    llm = FakeChat(PLAN)
    history = [
        {"role": "user", "content": "Wo ist das Museum Ladin?"},
        {"role": "assistant", "content": "In San Martino."},
    ]
    plan = LLMPlanner(llm).plan("Und am Sonntag?", history, MONDAY_10AM)

    assert plan.language == "de"
    assert plan.categories == [Category.CASTLE]  # the unknown category is dropped
    assert plan.date(MONDAY_10AM.date()) == date(2026, 11, 1) and plan.time is None
    assert plan.queries == [PLAN["standalone_question"], PLAN["english_query"]]
    call = llm.calls[0]
    assert call["schema"] is PLAN_SCHEMA
    assert "Monday 2026-10-26 10:00" in call["system"]
    assert "Wo ist das Museum Ladin?" in call["messages"][0]["content"]


def test_other_language_becomes_none():
    plan = LLMPlanner(FakeChat({**PLAN, "language": "other"})).plan("Hola", [], MONDAY_10AM)
    assert plan.language is None


def test_invalid_output_falls_back_to_the_raw_question():
    assert LLMPlanner(FakeChat("not json")).plan("pizza?", [], MONDAY_10AM) == trivial_plan("pizza?")
    assert LLMPlanner(FakeChat({**PLAN, "day": "next week"})).plan("pizza?", [], MONDAY_10AM) == trivial_plan("pizza?")


def test_moment():
    assert trivial_plan("x").moment(MONDAY_10AM) == MONDAY_10AM
    plan = QueryPlan(standalone_question="x", english_query="x", time=time(20, 0))
    assert plan.moment(MONDAY_10AM) == MONDAY_10AM.replace(hour=20)
    assert plan.asks_about_other_time


def test_schema_is_strict():
    # OpenAI strict mode: every property required, no extra properties.
    assert set(PLAN_SCHEMA["required"]) == set(PLAN_SCHEMA["properties"])
    assert PLAN_SCHEMA["additionalProperties"] is False


MONDAY = date(2026, 10, 26)


@pytest.mark.parametrize(
    ("ref", "today", "expected"),
    [
        ("today", MONDAY, MONDAY),
        ("tomorrow", MONDAY, date(2026, 10, 27)),
        ("day_after_tomorrow", MONDAY, date(2026, 10, 28)),
        ("monday", MONDAY, MONDAY),  # asked on that weekday: today
        ("sunday", MONDAY, date(2026, 11, 1)),  # the coming Sunday, not yesterday
        ("sunday", date(2026, 11, 5), date(2026, 11, 8)),
        ("monday", date(2026, 12, 31), date(2027, 1, 4)),  # across the year
    ],
)
def test_resolve_day(ref, today, expected):
    assert resolve_day(ref, today) == expected


@pytest.mark.parametrize(
    ("month_day", "today", "expected"),
    [
        ("12-24", MONDAY, date(2026, 12, 24)),
        ("10-26", MONDAY, MONDAY),
        ("01-06", MONDAY, date(2027, 1, 6)),  # already past this year
        ("02-29", MONDAY, date(2028, 2, 29)),
        ("02-30", MONDAY, None),
    ],
)
def test_resolve_month_day(month_day, today, expected):
    assert resolve_month_day(month_day, today) == expected


def test_explicit_date_wins_and_invalid_month_day_is_rejected():
    plan = QueryPlan(standalone_question="x", english_query="x", day="sunday", month_day="12-24")
    assert plan.date(MONDAY) == date(2026, 12, 24)
    assert LLMPlanner(FakeChat({**PLAN, "month_day": "2026-12-24"})).plan("x", [], MONDAY_10AM) == trivial_plan("x")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Bakery in Kiens/Chienes open on Sunday", "Bakery open on Sunday"),
        ("Hiking trails near Kiens", "Hiking trails"),
        ("Dove posso comprare il pane a Chienes?", "Dove posso comprare il pane?"),
        ("Gibt es in Kiens eine Pizzeria?", "Gibt es eine Pizzeria?"),
        ("Kiens", "Kiens"),  # nothing would be left
        ("Museum Ladin opening hours", "Museum Ladin opening hours"),
    ],
)
def test_strip_destination(text, expected):
    assert strip_destination(text) == expected


def test_destination_is_removed_unless_the_guest_wrote_it():
    plan = {**PLAN, "standalone_question": "Pizzeria in Kiens?", "english_query": "pizzeria in Kiens/Chienes"}
    assert LLMPlanner(FakeChat(plan)).plan("Pizzeria?", [], MONDAY_10AM).english_query == "pizzeria"
    kept = LLMPlanner(FakeChat(plan)).plan("Pizzeria in Kiens?", [], MONDAY_10AM)
    assert kept.english_query == "pizzeria in Kiens/Chienes"
