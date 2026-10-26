"""Parser for the structured exports (``ti_export_<town>_<type>_<Category>.txt``).

File layout::

    # Bakeries
    This is a list of bakeries.
    The items in this file are helpful in answering these questions: ...

    ### BEGIN POI
    This is a bakery in Kiens.
    ** Gatterer bakery-confectionery with café **
    **Also known as:** German: ... | Italian: ...
    **Opening Hours for ...:**
    {'period_start': '01.01', ...}
    ... is located at: Kienerdorfweg 2, 39030.0, Kiens
    ...
    ** About Gatterer bakery-confectionery with café**
    <free text until the next record>

Each line of a record's head is matched against a table of patterns. A line that no
pattern recognises is kept as a fact *and* reported, so a format change in the exports
shows up in the ingest report instead of silently degrading the chunks.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..hours import OpeningHours
from ..models import FILE_CATEGORIES, Category, Language, Record, RecordKind
from .text import clean, normalise_phone, slugify

RECORD_SEPARATOR = re.compile(r"^### BEGIN \w+\s*$", re.M)
ABOUT_HEADING = re.compile(r"^\*\* About .+\*\*\s*$", re.M)
MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
ALIAS_LANGUAGES: dict[str, Language] = {"German": "de", "Italian": "it", "English": "en"}
LINK_LANGUAGES: dict[str, Language] = {"English": "en", "German": "de", "Italian": "it"}
UNKNOWN_TOWN = "this area"


@dataclass
class ParseResult:
    records: list[Record]
    warnings: list[str] = field(default_factory=list)
    description: str = ""  # the file header's description and example questions


Fields = dict[str, Any]


def _intro(m: re.Match[str], f: Fields) -> None:
    if m["town"] != UNKNOWN_TOWN:
        f["town"] = clean(m["town"])


def _name(m: re.Match[str], f: Fields) -> None:
    f.setdefault("name", clean(m["name"]))


def _hours(m: re.Match[str], f: Fields) -> None:
    f["hours"] = OpeningHours.parse(m[0])
    if f["hours"] is None:
        f["_warnings"].append(f"unparsable opening hours: {m[0][:80]}")


def _address(m: re.Match[str], f: Fields) -> None:
    f["address"] = clean(m["address"])


def _coordinates(m: re.Match[str], f: Fields) -> None:
    f["coordinates"] = (float(m["lat"]), float(m["lon"]))


def _altitude(m: re.Match[str], f: Fields) -> None:
    f["altitude_m"] = round(float(m["altitude"]))


def _services(m: re.Match[str], f: Fields) -> None:
    f["services"] = tuple(s.strip() for s in clean(m["services"]).split(",") if s.strip())


def _field(m: re.Match[str], f: Fields) -> None:
    """'**Key:** value' lines."""
    key, value = m["key"].strip(), m["value"].strip()
    match key:
        case "Also known as":
            for part in value.split("|"):
                lang, _, alias = part.partition(":")
                if lang.strip() in ALIAS_LANGUAGES and alias.strip():
                    f["aliases"][ALIAS_LANGUAGES[lang.strip()]] = clean(alias)
        case "Phone" | "Mobile":
            if phone := normalise_phone(value):
                f[key.lower()] = phone
        case "Website":
            f["website"] = value
        case "CIN Code":
            pass  # national accommodation id: no value to a guest
        case "Detail Page links":
            for label, url in MARKDOWN_LINK.findall(value):
                if label in LINK_LANGUAGES:
                    f["links"][LINK_LANGUAGES[label]] = url
        case "Route links":
            for _, url in MARKDOWN_LINK.findall(value):
                f["links"]["route"] = url
        case "Summary" | "Overview":
            f["facts"].append(clean(value))
        case _ if key == f.get("name"):  # hikes: '**<name>:** elevation ... · best months ...'
            f["facts"].append(clean(value))
        case _:
            f["facts"].append(f"{key}: {clean(value)}")
            f["_warnings"].append(f"unknown field '{key}'")


def _ignore(m: re.Match[str], f: Fields) -> None:
    pass


LINE_HANDLERS: list[tuple[re.Pattern[str], Callable[[re.Match[str], Fields], None]]] = [
    (re.compile(r"^This is an? .+? in (?P<town>.+)\.$"), _intro),
    (re.compile(r"^\*\* Contact Information for .+\*\*$"), _ignore),
    (re.compile(r"^\*\*Opening Hours for .+:\*\*$"), _ignore),
    (re.compile(r"^\*\* (?P<name>.+?) \*\*$"), _name),
    (re.compile(r"^\{'period_start'.*\}$"), _hours),
    (re.compile(r"^\*\*(?P<key>[^*:]+):\*\*\s*(?P<value>.*)$"), _field),
    (re.compile(r"^.+? is located at: (?P<address>.+)$"), _address),
    (re.compile(r"^\[Google Maps\]\(.*?query=(?P<lat>-?[\d.]+),(?P<lon>-?[\d.]+)"), _coordinates),
    (re.compile(r"^Altitude: (?P<altitude>[\d.]+)"), _altitude),
    (re.compile(r"^.+? offers the following services: (?P<services>.+)$"), _services),
]


def parse_record(block: str, category: Category, source_file: str) -> tuple[Record | None, list[str]]:
    head, about = _split_about(block)
    fields: Fields = {"aliases": {}, "facts": [], "links": {}, "_warnings": []}
    for line in (ln.strip() for ln in head.splitlines()):
        if not line:
            continue
        for pattern, handler in LINE_HANDLERS:
            if m := pattern.match(line):
                handler(m, fields)
                break
        else:
            fields["facts"].append(clean(line))
            fields["_warnings"].append(f"unrecognised line: {line[:80]}")

    warnings = fields.pop("_warnings")
    if "name" not in fields:
        return None, [*warnings, f"record without a name in {source_file}"]

    description = "\n".join(clean(ln) for ln in about.splitlines() if ln.strip())
    record = Record(
        id=_record_id(category, fields["name"], fields["links"]),
        kind=RecordKind.PLACE,
        category=category,
        source_file=source_file,
        description=description,
        facts=tuple(fields.pop("facts")),
        **fields,
    )
    return record, [f"{record.id}: {w}" for w in warnings]


def _split_about(block: str) -> tuple[str, str]:
    """(head with the structured lines, free-text description after '** About ...**')."""
    parts = ABOUT_HEADING.split(block, maxsplit=1)
    return (parts[0], parts[1]) if len(parts) == 2 else (block, "")


def _record_id(category: Category, name: str, links: dict[str, str]) -> str:
    """Stable id: the numeric id of the detail page if there is one, else the name slug."""
    for url in links.values():
        if m := re.search(r"[_#](\d+)$", url):
            return f"{category}-{m.group(1)}"
    return f"{category}-{slugify(name)}"


def parse_export_file(path: Path) -> ParseResult:
    suffix = path.stem.rsplit("_", 1)[-1]
    category = FILE_CATEGORIES.get(suffix)
    if category is None:
        return ParseResult([], [f"{path.name}: unknown category '{suffix}'"])

    header, *blocks = RECORD_SEPARATOR.split(path.read_text(encoding="utf-8"))
    result = ParseResult([])
    result.description = " ".join(
        ln.strip() for ln in header.splitlines() if ln.strip() and not ln.startswith("#")
    ).replace("{location}", "Kiens")
    for block in blocks:
        record, warnings = parse_record(block, category, path.name)
        result.warnings += warnings
        if record:
            result.records.append(record)
    return result
