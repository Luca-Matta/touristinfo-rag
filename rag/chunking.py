"""Split records' search text into chunks for indexing.

Most records fit in one chunk (they are a few hundred words at most), and keeping a record
whole keeps every fact next to the entity it belongs to. Longer ones are packed by
paragraph (and a paragraph that is too long by sentence); each piece repeats the record
header so it is still self-describing.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator

from .models import Chunk, Record

MAX_CHUNK_CHARS = 1500


def chunk_record(record: Record, max_chars: int = MAX_CHUNK_CHARS) -> list[Chunk]:
    text = record.search_text()
    if len(text) <= max_chars:
        return [Chunk(record.id, record.id, text)]

    header, _, body = text.partition("\n")
    budget = max_chars - len(header)
    pieces: list[str] = []
    current = ""
    for paragraph in _units(body, budget):
        if current and len(current) + len(paragraph) > budget:
            pieces.append(current)
            current = ""
        current = f"{current}\n{paragraph}" if current else paragraph
    if current:
        pieces.append(current)
    return [Chunk(f"{record.id}#{n}", record.id, f"{header}\n{piece}") for n, piece in enumerate(pieces)]


def _units(body: str, budget: int) -> Iterator[str]:
    """Paragraphs, with any paragraph longer than the budget broken into sentences."""
    for paragraph in (p.strip() for p in body.split("\n") if p.strip()):
        if len(paragraph) <= budget:
            yield paragraph
        else:
            yield from re.split(r"(?<=[.!?])\s+", paragraph)


def chunk_records(records: Iterable[Record], max_chars: int = MAX_CHUNK_CHARS) -> list[Chunk]:
    return [chunk for record in records for chunk in chunk_record(record, max_chars)]
