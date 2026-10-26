"""Load ``data/`` into a :class:`Corpus` of records, plus a report of anything odd."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from ..models import Category, Record, RecordKind
from .exports import parse_export_file
from .text import fold
from .web import parse_web_file


@dataclass
class Corpus:
    records: dict[str, Record]
    warnings: list[str] = field(default_factory=list)

    def __iter__(self) -> Iterator[Record]:
        return iter(self.records.values())

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, record_id: str) -> Record:
        return self.records[record_id]

    def places(self, categories: Iterable[Category] = ()) -> list[Record]:
        wanted = set(categories)
        return [r for r in self if r.kind is RecordKind.PLACE and (not wanted or r.category in wanted)]

    def with_services(self, terms: Iterable[str], categories: Iterable[Category] = ()) -> list[Record]:
        """Places offering *all* the given services (accent/case-insensitive substring match)."""
        folded = [fold(t) for t in terms if t.strip()]
        if not folded:
            return []
        return [r for r in self.places(categories) if all(any(t in fold(s) for s in r.services) for t in folded)]


def load_corpus(data_dir: str | Path) -> Corpus:
    corpus = Corpus({})
    by_category: dict[Category, list[Record]] = defaultdict(list)
    descriptions: dict[Category, str] = {}

    for path in sorted(Path(data_dir).glob("ti_export_*.txt")):
        result = parse_web_file(path) if "_website_" in path.name else parse_export_file(path)
        corpus.warnings += result.warnings
        for record in result.records:
            if record.id in corpus.records:
                corpus.warnings.append(f"duplicate record id {record.id} in {path.name}; skipped")
                continue
            corpus.records[record.id] = record
            if record.kind is RecordKind.PLACE:
                by_category[record.category].append(record)
                descriptions[record.category] = result.description

    for category, records in by_category.items():
        overview = _overview(category, records, descriptions[category])
        corpus.records[overview.id] = overview
    return corpus


def _overview(category: Category, records: list[Record], description: str) -> Record:
    entries = "; ".join(f"{r.name} ({r.town})" if r.town else r.name for r in records)
    return Record(
        id=f"{category}-overview",
        kind=RecordKind.OVERVIEW,
        category=category,
        name=f"All {category.label.split(',')[0]} entries",
        source_file=records[0].source_file,
        description=f"{description}\n{_count(len(records))}: {entries}.",
    )


def _count(n: int) -> str:
    return "The only entry" if n == 1 else f"All {n} entries"
