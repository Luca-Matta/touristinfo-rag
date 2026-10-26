"""Evaluation: retrieval quality (free, offline) and answer quality (LLM calls).

    python -m eval.run_eval retrieval                 # dense / keyword / hybrid, dev split
    python -m eval.run_eval retrieval --planner       # with LLM query planning (cheap calls)
    python -m eval.run_eval retrieval --split test    # the held-out split: report, don't tune
    python -m eval.run_eval answers --judge -v        # full pipeline, fact checks + LLM judge

Each question in ``questions.json`` names the record ids that can answer it (``expected``,
any one is enough), facts the answer must contain (``must_include``: a list of groups,
each group a list of acceptable spellings), optional chat ``history`` and a fixed ``now``
so that opening-hour answers are reproducible. ``answerable: false`` marks questions the
data cannot answer: the right answer is to say so.

The questions are split into ``dev`` (look at the misses, tune) and ``test`` (only
report). With 69 questions the numbers are a regression check, not a benchmark: one
question is 1.5-4 points, so small differences between configurations are noise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Literal
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from rag.assistant import TouristAssistant
from rag.config import Settings
from rag.errors import RagError
from rag.generation import Message, make_chat_model
from rag.ingest.text import fold
from rag.models import Language
from rag.planner import PLANNER_SYSTEM, QueryPlan, trivial_plan

HERE = Path(__file__).parent
QUESTIONS = HERE / "questions.json"
PLAN_CACHE = HERE / ".cache" / "plans.json"
DEFAULT_NOW = "2026-10-26T10:00"  # a Monday in season: several places are closed on Mondays
Split = Literal["dev", "test", "all"]


class EvalItem(BaseModel):
    id: str
    split: Literal["dev", "test"]
    lang: Language
    question: str
    expected: list[str]
    must_include: list[list[str]] = Field(default_factory=list)
    history: list[Message] = Field(default_factory=list)
    now: str = DEFAULT_NOW
    answerable: bool = True

    def moment(self, tz: str) -> datetime:
        return datetime.fromisoformat(self.now).replace(tzinfo=ZoneInfo(tz))


def load_items(split: Split) -> list[EvalItem]:
    items = [EvalItem.model_validate(x) for x in json.loads(QUESTIONS.read_text(encoding="utf-8"))]
    return [i for i in items if split == "all" or i.split == split]


# ---------------------------------------------------------------------- planning


class PlanCache:
    """Plans keyed by model and input, so repeated eval runs cost nothing."""

    def __init__(self, assistant: TouristAssistant, enabled: bool) -> None:
        self.assistant = assistant
        self.enabled = enabled and assistant.planner is not None
        self.plans: dict[str, Any] = json.loads(PLAN_CACHE.read_text()) if PLAN_CACHE.exists() else {}

    def plan(self, item: EvalItem, tz: str) -> QueryPlan:
        if not self.enabled:
            return trivial_plan(item.question)
        assert self.assistant.planner is not None
        key_src = json.dumps([PLANNER_SYSTEM, self.assistant.planner.llm.name, item.question, item.history, item.now])
        key = hashlib.sha256(key_src.encode()).hexdigest()[:16]
        if key not in self.plans:
            plan = self.assistant.plan(item.question, item.history, item.moment(tz))
            self.plans[key] = plan.model_dump(mode="json")
        return QueryPlan.model_validate(self.plans[key])

    def save(self) -> None:
        if self.enabled:
            PLAN_CACHE.parent.mkdir(exist_ok=True)
            PLAN_CACHE.write_text(json.dumps(self.plans, indent=1, ensure_ascii=False))


# ---------------------------------------------------------------------- retrieval


def retrieval_eval(assistant: TouristAssistant, items: list[EvalItem], plans: PlanCache, k: int, verbose: bool) -> None:
    tz = assistant.settings.timezone
    items = [i for i in items if i.expected]
    planned = {i.id: plans.plan(i, tz) for i in items}
    plans.save()

    print(f"{len(items)} questions · k={k} · planner {'on' if plans.enabled else 'off'}\n")
    print(f"{'mode':<8} {'hit@k':>6} {'MRR':>5}   {'en':>7} {'de':>7} {'it':>7}")
    for mode in ("dense", "keyword", "hybrid"):
        ranks: dict[str, int | None] = {}
        misses = []
        for item in items:
            sources, _ = assistant.retrieve(planned[item.id], mode)
            ids = [r.id for r in sources[:k]]
            ranks[item.id] = next((n for n, rid in enumerate(ids, 1) if rid in item.expected), None)
            if ranks[item.id] is None:
                misses.append(f"    {item.id} {item.question!r} -> {ids}")
        by_lang = _group(items, lambda i: i.lang)
        hit = mean(ranks[i.id] is not None for i in items)
        mrr = mean(1 / r if (r := ranks[i.id]) else 0 for i in items)
        langs = " ".join(
            f"{sum(ranks[i.id] is not None for i in group):>3}/{len(group):<3}"
            for lang in ("en", "de", "it")
            if (group := by_lang.get(lang, []))
        )
        print(f"{mode:<8} {hit:>6.0%} {mrr:>5.2f}   {langs}")
        if verbose and misses:
            print("\n".join(misses))


# ---------------------------------------------------------------------- answers


JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "grounded": {"type": "boolean"},
        "correct": {"type": "boolean"},
        "language_ok": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["grounded", "correct", "language_ok", "reason"],
    "additionalProperties": False,
}

JUDGE_SYSTEM = """\
You grade answers of a tourist information chatbot. You get the guest's question, the \
sources the bot was given, and its answer. Return JSON:
- grounded: every factual claim in the answer is supported by the sources.
- correct: the answer addresses the question correctly. If the sources do not contain the \
answer, the correct behaviour is to say so (and not to invent one).
- language_ok: the answer is written in {language}.
- reason: one short sentence, mainly when something is false."""


def fact_check(answer: str, groups: Iterable[list[str]]) -> list[str]:
    """Return the groups none of whose spellings occurs in the answer (spacing and accents ignored)."""

    def norm(s: str) -> str:
        return "".join(fold(s).split())

    text = norm(answer)
    return [" | ".join(g) for g in groups if not any(norm(alt) in text for alt in g)]


def answers_eval(assistant: TouristAssistant, items: list[EvalItem], judge: bool, verbose: bool) -> None:
    from rag.generation.prompts import LANGUAGE_NAMES, format_sources

    tz = assistant.settings.timezone
    judge_llm = make_chat_model(assistant.settings) if judge else None
    rows = []
    for item in items:
        now = item.moment(tz)
        answer = assistant.ask(item.question, item.history, now=now)
        if answer.text is None:
            raise RagError("answers mode needs an LLM: set an API key in .env")
        row: dict[str, Any] = {
            "id": item.id,
            "lang": item.lang,
            "question": item.question,
            "answer": answer.text,
            "sources": [r.id for r in answer.sources],
            "retrieved": not item.expected or any(r.id in item.expected for r in answer.sources),
            "missing_facts": fact_check(answer.text, item.must_include),
        }
        if judge_llm is not None:
            prompt = (
                f"Question: {item.question}\n\n<sources>\n{format_sources(answer.sources, answer.plan, now)}\n"
                f"</sources>\n\nAnswer:\n{answer.text}"
            )
            system = JUDGE_SYSTEM.format(language=LANGUAGE_NAMES[item.lang])
            row["judge"] = judge_llm.complete(system, [{"role": "user", "content": prompt}], schema=JUDGE_SCHEMA).json()
        rows.append(row)
        if verbose:
            _print_row(row)

    out = HERE / ".cache" / f"answers-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, indent=1, ensure_ascii=False))

    print(f"\n{len(rows)} answers · details in {out.relative_to(HERE.parent)}")
    print(f"  source retrieved      {mean(r['retrieved'] for r in rows):.0%}")
    print(f"  all facts present     {mean(not r['missing_facts'] for r in rows):.0%}")
    if judge:
        for key in ("grounded", "correct", "language_ok"):
            print(f"  judge: {key:<14} {mean(r['judge'][key] for r in rows):.0%}")


def _print_row(row: dict[str, Any]) -> None:
    flags = []
    if not row["retrieved"]:
        flags.append("source missed")
    if row["missing_facts"]:
        flags.append("missing " + "; ".join(row["missing_facts"]))
    if (j := row.get("judge")) and not (j["grounded"] and j["correct"] and j["language_ok"]):
        flags.append(f"judge: {j['reason']}")
    status = "ok" if not flags else " · ".join(flags)
    print(f"{row['id']} [{row['lang']}] {row['question']}\n    {status}")


def _group(items: list[EvalItem], key: Any) -> dict[str, list[EvalItem]]:
    groups: dict[str, list[EvalItem]] = defaultdict(list)
    for item in items:
        groups[key(item)].append(item)
    return groups


# ---------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=["retrieval", "answers"])
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    parser.add_argument("-k", type=int, default=5, help="retrieval: cut-off for hit@k")
    parser.add_argument("--planner", action="store_true", help="retrieval: plan queries with the LLM")
    parser.add_argument("--judge", action="store_true", help="answers: grade each answer with an LLM judge")
    parser.add_argument("-v", "--verbose", action="store_true", help="list misses / per-question results")
    args = parser.parse_args(argv)

    load_dotenv()
    try:
        settings = Settings.from_env()
        offline = args.what == "retrieval" and not args.planner
        assistant = TouristAssistant.from_settings(settings, retrieve_only=offline)
        if args.planner and assistant.planner is None:
            raise RagError("--planner needs an LLM: set an API key in .env")
        print(f"embeddings: {assistant.retriever.vectors.embedder.name} · split: {args.split}")
        items = load_items(args.split)
        if args.what == "retrieval":
            retrieval_eval(assistant, items, PlanCache(assistant, args.planner), args.k, args.verbose)
        else:
            answers_eval(assistant, items, args.judge, args.verbose)
    except RagError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
