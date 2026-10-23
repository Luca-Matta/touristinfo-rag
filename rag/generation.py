"""Prompt construction and LLM clients (OpenAI or Anthropic, chosen by config)."""

from __future__ import annotations

import json
from datetime import datetime

from .config import Settings
from .hours import HoursRule, status_at
from .retrieval import Hit, HybridIndex

SYSTEM_PROMPT = """\
You are the digital tourist information assistant for Kiens/Chienes and the surrounding \
Kronplatz area in South Tyrol, Italy.

Answer the guest's question using only the numbered sources provided with it.
- Reply in the language the guest writes in (German, Italian or English).
- If the sources do not contain the answer, say so plainly and suggest contacting the \
local tourist office. Never invent names, opening hours, prices, phone numbers or links.
- For "is it open" questions, rely on the "Status right now" line of a source; it has \
already been computed for the current local time. Opening hours not listed means unknown, \
not closed.
- When you mention a place, give its useful details (address, hours, phone) and at most \
one link, preferring the link in the guest's language.
- Cite the sources you used as [1], [2], ... after the relevant sentence.
- Be concise and friendly: a short paragraph or a short list."""


def format_context(hits: list[Hit], index: HybridIndex, now: datetime) -> str:
    """Numbered source blocks. Split records are re-assembled so the LLM sees the whole record."""
    blocks, seen_records = [], set()
    for hit in hits:
        parts = index.siblings(hit.chunk)
        if parts[0].id in seen_records:
            continue  # another part of this record was already included
        seen_records.add(parts[0].id)

        meta = hit.chunk.metadata
        header = [f"[{len(blocks) + 1}] {meta.get('name', '')} (source: {meta.get('source', '')})"]
        if hours_json := meta.get("hours_json"):
            rules = [HoursRule.from_dict(d) for d in json.loads(hours_json)]
            header.append(f"Status right now: {status_at(rules, now)}")
        links = [f"{lang.upper()} {meta[f'url_{lang}']}" for lang in ("en", "de", "it") if meta.get(f"url_{lang}")]
        if links:
            header.append("Links: " + " · ".join(links))
        if meta.get("url_outdooractive"):
            header.append(f"Route map: {meta['url_outdooractive']}")

        # every part repeats the title line; keep it once
        body = "\n".join([parts[0].text] + [p.text.split("\n", 1)[1] for p in parts[1:]])
        blocks.append("\n".join(header) + "\n" + body)
    return "\n\n".join(blocks)


def build_user_message(question: str, context: str, now: datetime) -> str:
    # The timestamp goes in the user turn, not the system prompt, so the system prompt
    # stays byte-identical across requests (cacheable).
    return (
        f"Current local time: {now:%A %d %B %Y, %H:%M}\n\n"
        f"Sources:\n{context or '(no relevant sources found)'}\n\n"
        f"Guest question: {question}"
    )


class LLM:
    def complete(self, system: str, user: str) -> str:
        raise NotImplementedError


class OpenAILLM(LLM):
    def __init__(self, model: str):
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model

    def complete(self, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content or ""


class AnthropicLLM(LLM):
    def __init__(self, model: str):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model

    def complete(self, system: str, user: str) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": "low"},  # grounded short answers do not need deep reasoning
        )
        if resp.stop_reason == "refusal":
            return "Sorry, I can't help with that request."
        return "".join(b.text for b in resp.content if b.type == "text")


def make_llm(settings: Settings) -> LLM:
    provider = settings.resolved_llm_provider()
    if provider == "openai":
        return OpenAILLM(settings.openai_chat_model)
    if provider == "anthropic":
        return AnthropicLLM(settings.anthropic_model)
    raise RuntimeError(
        "No LLM configured: set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env "
        "(or run with --retrieve-only)."
    )
