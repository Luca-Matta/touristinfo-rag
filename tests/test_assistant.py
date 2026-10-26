from rag.assistant import MAX_SOURCES, TouristAssistant
from rag.generation.prompts import SYSTEM_PROMPT
from rag.models import Category
from rag.planner import LLMPlanner, QueryPlan

from .conftest import MONDAY_10AM, FakeChat


def plan(**kw):
    return {
        "standalone_question": "q",
        "english_query": "q",
        "language": "en",
        "categories": [],
        "day": None,
        "month_day": None,
        "time": None,
        "services": [],
        "list_all": False,
    } | kw


def make(corpus, retriever, settings, *replies):
    llm = FakeChat(*replies[1:])
    return TouristAssistant(corpus, retriever, settings, llm, LLMPlanner(FakeChat(replies[0]))), llm


def test_retrieval_only_mode_returns_sources_without_text(corpus, retriever, settings):
    answer = TouristAssistant(corpus, retriever, settings).ask("Falkenau", now=MONDAY_10AM)
    assert answer.text is None
    assert answer.sources[0].id == "butcher-2625"


def test_end_to_end_with_history(corpus, retriever, settings):
    assistant, llm = make(
        corpus,
        retriever,
        settings,
        plan(
            standalone_question="Is Gatterer open on Sunday?",
            english_query="Gatterer bakery Sunday",
            categories=["bakery"],
            day="sunday",
        ),
        "No hours are listed for Sunday [1].",
    )
    history = [{"role": "user", "content": "Tell me about Gatterer"}, {"role": "assistant", "content": "A bakery."}]
    answer = assistant.ask("And on Sunday?", history, now=MONDAY_10AM)

    assert answer.text == "No hours are listed for Sunday [1]."
    assert answer.sources[0].id == "bakery-3383"
    call = llm.calls[0]
    assert call["system"] == SYSTEM_PROMPT
    assert call["messages"][:2] == history
    assert "Status on Sunday 01.11.2026" in call["messages"][2]["content"]


def test_service_matches_come_first(corpus, retriever, settings):
    assistant, _ = make(
        corpus,
        retriever,
        settings,
        plan(english_query="accommodation with dogs allowed", services=["dogs allowed"]),
        "ok",
    )
    answer = assistant.ask("Posso portare il cane?", now=MONDAY_10AM)
    assert answer.sources[0].id == "bed_and_breakfast-12399"


def test_list_all_includes_every_place_of_the_category(corpus, retriever, settings):
    sources, _ = TouristAssistant(corpus, retriever, settings).retrieve(
        QueryPlan(
            standalone_question="list churches", english_query="churches", categories=[Category.CHURCH], list_all=True
        )
    )
    assert {r.id for r in corpus.places([Category.CHURCH])} <= {r.id for r in sources}
    assert len(sources) <= MAX_SOURCES
