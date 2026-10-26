"""Command line chat.

python app.py "Where can I buy bread on Saturday?"     # one question
python app.py                                         # interactive chat with history
python app.py --debug "Ist das Ladinische Museum am Montag offen?"
python app.py --retrieve-only "pizza"                 # no LLM, no API key needed
"""

from __future__ import annotations

import argparse
import logging
import sys

from dotenv import load_dotenv

from .assistant import Answer, TouristAssistant
from .config import Settings
from .errors import RagError
from .generation import Message


def print_debug(answer: Answer) -> None:
    plan = answer.plan.model_dump_json(exclude_defaults=True, indent=2)
    print(f"\n--- plan ---\n{plan}\n\n--- sources (fused score · rank per list) ---", file=sys.stderr)
    scores = {h.record.id: h for h in answer.hits}
    for n, record in enumerate(answer.sources, 1):
        hit = scores.get(record.id)
        why = "structured match"
        if hit:
            why = f"{hit.score:.4f}  " + ", ".join(f"{k} #{v}" for k, v in hit.ranks.items())
        print(f"[{n}] {record.id:<42} {why}", file=sys.stderr)
    print("---\n", file=sys.stderr)


def run(assistant: TouristAssistant, question: str, history: list[Message], *, debug: bool) -> None:
    answer = assistant.ask(question, history)
    if debug or answer.text is None:
        print_debug(answer)
    if answer.text is not None:
        print(answer.text)
        history += [{"role": "user", "content": question}, {"role": "assistant", "content": answer.text}]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tourist information chatbot for Kiens/Chienes")
    parser.add_argument("question", nargs="*", help="question to ask (omit for interactive mode)")
    parser.add_argument("--debug", action="store_true", help="show the query plan and the selected sources")
    parser.add_argument("--retrieve-only", action="store_true", help="skip the LLM and show retrieval results")
    parser.add_argument("-v", "--verbose", action="store_true", help="log LLM calls, latency and token usage")
    args = parser.parse_args(argv)

    load_dotenv()
    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "chromadb", "openai", "anthropic"):  # their INFO logs drown ours
        logging.getLogger(noisy).setLevel(logging.WARNING)

    try:
        settings = Settings.from_env()
        if settings.llm_provider is None and not args.retrieve_only:
            print("No API key found: running retrieval only (see .env.example).", file=sys.stderr)
        assistant = TouristAssistant.from_settings(settings, retrieve_only=args.retrieve_only)
    except RagError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    history: list[Message] = []
    if args.question:
        try:
            run(assistant, " ".join(args.question), history, debug=args.debug)
        except RagError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0

    print("Kiens/Chienes tourist assistant. Ask in English, Deutsch or Italiano (Ctrl+D to quit).")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if question:
            try:
                run(assistant, question, history, debug=args.debug)
            except RagError as exc:
                print(f"error: {exc}", file=sys.stderr)
