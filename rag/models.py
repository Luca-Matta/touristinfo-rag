"""Domain model: a Record is one thing a guest can ask about.

Records are the unit of everything downstream. Retrieval works on chunks of a record's
*search text*, but the LLM always receives the whole record's *context text*. The two
views are deliberately different:

* ``search_text`` is what gets embedded and keyword-indexed: name, aliases, category
  labels, town, services and description. No URLs, coordinates or phone numbers, which
  only add noise to an embedding.
* ``context_text`` is what the LLM reads: every field, rendered consistently, with
  opening hours already turned into prose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

from .hours import OpeningHours

Language = Literal["en", "de", "it"]
LANGUAGES: tuple[Language, ...] = ("en", "de", "it")


class Category(StrEnum):
    BED_AND_BREAKFAST = "bed_and_breakfast"
    EASY_HIKE = "easy_hike"
    BAKERY = "bakery"
    BUTCHER = "butcher"
    CASTLE = "castle"
    CHURCH = "church"
    PIZZERIA = "pizzeria"
    WEB_PAGE = "web_page"

    @property
    def label(self) -> str:
        return CATEGORY_LABELS[self]["en"]

    def labels(self) -> str:
        """All three languages, e.g. 'bakery / Bäckerei / panificio, panetteria'."""
        return " / ".join(CATEGORY_LABELS[self][lang] for lang in LANGUAGES)


# Category names in the three languages guests use. This is domain vocabulary (a tourist
# office keeps exactly this kind of taxonomy) and it lets keyword search match "Bäckerei"
# or "macelleria" against an English corpus.
CATEGORY_LABELS: dict[Category, dict[Language, str]] = {
    Category.BED_AND_BREAKFAST: {
        "en": "bed & breakfast, guesthouse",
        "de": "Frühstückspension, Unterkunft",
        "it": "B&B, pensione, alloggio",
    },
    Category.EASY_HIKE: {
        "en": "easy hike, family walk",
        "de": "leichte Wanderung, Familienwanderung",
        "it": "escursione facile, passeggiata",
    },
    Category.BAKERY: {
        "en": "bakery",
        "de": "Bäckerei, Konditorei",
        "it": "panificio, panetteria, pasticceria",
    },
    Category.BUTCHER: {
        "en": "butcher, delicatessen",
        "de": "Metzgerei, Feinkost",
        "it": "macelleria, salumeria",
    },
    Category.CASTLE: {
        "en": "castle, palace, museum",
        "de": "Burg, Schloss, Museum",
        "it": "castello, museo",
    },
    Category.CHURCH: {
        "en": "church, chapel",
        "de": "Kirche, Kapelle",
        "it": "chiesa, cappella",
    },
    Category.PIZZERIA: {
        "en": "pizzeria, restaurant",
        "de": "Pizzeria, Restaurant",
        "it": "pizzeria, ristorante",
    },
    Category.WEB_PAGE: {
        "en": "information page",
        "de": "Informationsseite",
        "it": "pagina informativa",
    },
}

# Export file name suffix -> category (the file name 'Bucheries' is a typo in the source).
FILE_CATEGORIES: dict[str, Category] = {
    "BedAndBreakfasts": Category.BED_AND_BREAKFAST,
    "EasyHikes": Category.EASY_HIKE,
    "Bakeries": Category.BAKERY,
    "Bucheries": Category.BUTCHER,
    "Castles": Category.CASTLE,
    "Churches": Category.CHURCH,
    "Pizzerias": Category.PIZZERIA,
}


class RecordKind(StrEnum):
    PLACE = "place"
    OVERVIEW = "overview"  # "all bakeries": answers list questions in one piece
    WEB_SECTION = "web_section"


@dataclass(frozen=True)
class Record:
    id: str
    kind: RecordKind
    category: Category
    name: str
    source_file: str
    town: str | None = None
    aliases: dict[Language, str] = field(default_factory=dict)
    facts: tuple[str, ...] = ()  # summary lines such as "covers 3.2km · 1h 30min · difficulty Easy"
    hours: OpeningHours | None = None
    address: str | None = None
    altitude_m: int | None = None
    coordinates: tuple[float, float] | None = None
    phone: str | None = None
    mobile: str | None = None
    website: str | None = None
    services: tuple[str, ...] = ()
    links: dict[str, str] = field(default_factory=dict)  # "en"/"de"/"it" detail pages, "route" map
    description: str = ""

    @property
    def header(self) -> str:
        where = f" — {self.town}" if self.town else ""
        return f"{self.name} — {self.category.labels()}{where}"

    def search_text(self) -> str:
        lines = [self.header]
        if self.aliases:
            lines.append("Also known as: " + "; ".join(self.aliases.values()))
        lines += self.facts
        if self.services:
            lines.append("Services: " + ", ".join(self.services))
        if self.description:
            lines.append(self.description)
        return "\n".join(lines)

    def context_text(self, language: Language | None = None) -> str:
        """Everything the LLM may need, with the detail link in the guest's language."""
        lines = [f"{self.name} ({self.category.label})" + (f", {self.town}" if self.town else "")]
        if self.aliases:
            lines.append("Also known as: " + "; ".join(f"{k.upper()}: {v}" for k, v in self.aliases.items()))
        lines += self.facts
        if self.hours:
            lines.append(f"Opening hours: {self.hours.describe()}")
        elif self.kind is RecordKind.PLACE and self.category is not Category.EASY_HIKE:
            lines.append("Opening hours: not listed")
        if self.address:
            alt = f" (altitude {self.altitude_m} m)" if self.altitude_m is not None else ""
            lines.append(f"Address: {self.address}{alt}")
        contact = [
            f"{label}: {value}"
            for label, value in (("Phone", self.phone), ("Mobile", self.mobile), ("Website", self.website))
            if value
        ]
        if contact:
            lines.append(" · ".join(contact))
        if self.services:
            lines.append("Services: " + ", ".join(self.services))
        if link := self.link_for(language):
            lines.append(f"More information: {link}")
        if route := self.links.get("route"):
            lines.append(f"Route map: {route}")
        if self.description:
            lines.append(self.description)
        return "\n".join(lines)

    def link_for(self, language: Language | None) -> str | None:
        """Detail page in the guest's language, falling back to English, then any."""
        for lang in (language, "en", "de", "it"):
            if lang and lang in self.links:
                return self.links[lang]
        return None


@dataclass(frozen=True)
class Chunk:
    """A piece of a record's search text; ``record_id`` links it back to the record."""

    id: str
    record_id: str
    text: str
