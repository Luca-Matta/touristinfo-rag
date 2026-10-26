from datetime import date, time

from rag.models import Category
from rag.planner import PLAN_SCHEMA, LLMPlanner, QueryPlan, trivial_plan

from .conftest import MONDAY_10AM, FakeChat

PLAN = {
    "standalone_question": "Ist das Museum Ladin am Sonntag geöffnet?",
    "english_query": "Museum Ladin opening hours Sunday",
    "language": "de",
    "categories": ["castle", "spaceship"],
    "date": "2026-11-01",
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
    assert plan.date == date(2026, 11, 1) and plan.time is None
    assert plan.queries == [PLAN["standalone_question"], PLAN["english_query"]]
    call = llm.calls[0]
    assert call["schema"] is PLAN_SCHEMA
    assert "Monday 2026-10-26 10:00" in call["system"]
    assert "Wo ist das Museum Ladin?" in call["messages"][0]["content"]  # history is passed on


def test_other_language_becomes_none():
    plan = LLMPlanner(FakeChat({**PLAN, "language": "other"})).plan("Hola", [], MONDAY_10AM)
    assert plan.language is None


def test_invalid_output_falls_back_to_the_raw_question():
    assert LLMPlanner(FakeChat("not json")).plan("pizza?", [], MONDAY_10AM) == trivial_plan("pizza?")
    assert LLMPlanner(FakeChat({"date": "tomorrow"})).plan("pizza?", [], MONDAY_10AM) == trivial_plan("pizza?")


def test_moment():
    assert trivial_plan("x").moment(MONDAY_10AM) == MONDAY_10AM
    plan = QueryPlan(standalone_question="x", english_query="x", time=time(20, 0))
    assert plan.moment(MONDAY_10AM) == MONDAY_10AM.replace(hour=20)
    assert plan.asks_about_other_time


def test_schema_is_strict():
    # OpenAI strict mode: every property required, no extra properties.
    assert set(PLAN_SCHEMA["required"]) == set(PLAN_SCHEMA["properties"])
    assert PLAN_SCHEMA["additionalProperties"] is False
