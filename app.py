"""Tourist chatbot CLI.

    python app.py "Where can I buy bread on Saturday?"      # one question
    python app.py                                          # interactive chat
    python app.py --debug "Ist das Ladinische Museum am Montag offen?"
    python app.py --retrieve-only "pizza"                  # no LLM / API key needed
"""

from __future__ import annotations

import argparse
import sys

from rag.pipeline import RagPipeline


def print_hits(hits) -> None:
    print("\n--- retrieved chunks (RRF score · rank per retriever) ---")
    for i, h in enumerate(hits, 1):
        ranks = ", ".join(f"{name} #{rank}" for name, rank in h.ranks.items())
        print(f"{i}. {h.chunk.id:<45} {h.score:.4f}  ({ranks})")
        print("   " + h.chunk.text.split("\n", 1)[0][:110])
    print("---------------------------------------------------------\n")


def ask(rag: RagPipeline, question: str, args) -> None:
    if args.retrieve_only:
        print_hits(rag.retrieve(question, args.k))
        return
    ans = rag.answer(question, k=args.k)
    if args.debug:
        print_hits(ans.hits)
        print(ans.prompt + "\n---------------------------------------------------------\n")
    print(ans.text)


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG tourist chatbot for Kiens/Chienes")
    parser.add_argument("question", nargs="*", help="question to ask (omit for interactive mode)")
    parser.add_argument("-k", type=int, default=None, help="number of chunks to retrieve")
    parser.add_argument("--debug", action="store_true", help="show retrieved chunks and the prompt")
    parser.add_argument("--retrieve-only", action="store_true", help="skip the LLM, show retrieval results")
    args = parser.parse_args()

    rag = RagPipeline()
    if args.question:
        try:
            ask(rag, " ".join(args.question), args)
        except RuntimeError as e:  # no API key configured
            print(e, file=sys.stderr)
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
                ask(rag, question, args)
            except RuntimeError as e:
                print(e, file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
