"""Load the export files in data/ and turn them into retrieval chunks.

There are two kinds of files, and they need different treatment:

* Structured exports (``ti_export_<town>_<type>_<Category>.txt``): a file header followed
  by records separated by ``### BEGIN <TYPE>``. Every record describes exactly one
  entity (a bakery, a church, ...). We parse the fields, normalise them, and emit **one
  chunk per record**: records are small (< 3 kB) and splitting one would separate facts
  like opening hours from the entity name they belong to.
* Website scrapes (``ti_export_<town>_website_<Title>.txt``): free text with markdown
  headings, cookie banners, "Read more" links and duplicated paragraphs. We clean them
  and chunk by section, packing paragraphs up to ``MAX_CHUNK_CHARS``.

On top of that we add one *overview* chunk per category ("all bakeries"), so that list
questions ("show me all B&Bs") retrieve a single complete answer instead of depending on
top-k happening to include every record.

Content is mostly English, but users ask in English, German and Italian. Each record
chunk therefore carries the DE/IT names from the export plus the category label in all
three languages, which makes both keyword and embedding search work across languages.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from .hours import format_hours, parse_hours

MAX_CHUNK_CHARS = 1500

# Category label in EN / DE / IT. Injected into chunk text for cross-lingual matching.
CATEGORY_LABELS = {
    "BedAndBreakfasts": "Bed & Breakfast / guesthouse · Frühstückspension / Unterkunft · B&B / pensione / alloggio",
    "EasyHikes": "Easy hike / family walk · leichte Wanderung / Familienwanderung · escursione facile / passeggiata",
    "Bakeries": "Bakery · Bäckerei / Konditorei · panificio / panetteria / pasticceria",
    "Bucheries": "Butcher / delicatessen · Metzgerei / Feinkost · macelleria / salumeria",
    "Castles": "Castle / palace · Burg / Schloss · castello",
    "Churches": "Church / chapel · Kirche / Kapelle · chiesa / cappella",
    "Pizzerias": "Pizzeria / restaurant · Pizzeria / Restaurant · pizzeria / ristorante",
}

LANG_KEYS = {"English": "en", "German": "de", "Italian": "it"}


@dataclass
class Chunk:
    id: str
    text: str  # what gets embedded, keyword-indexed and shown to the LLM
    metadata: dict = field(default_factory=dict)  # flat str/int/float values (Chroma requirement)


# --------------------------------------------------------------------------- helpers


def _clean(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"\b(\d+)\.0\b", r"\1", s)  # '39030.0' -> '39030', 'Dorfweg 22.0' -> 'Dorfweg 22'
    return re.sub(r"[ \t]+", " ", s).strip()


def _phone(s: str) -> str:
    s = s.strip()
    if s in {"", "-"}:
        return ""
    if s.isdigit() and s.startswith("39") and len(s) >= 11:  # '390474565238' -> '+39 0474565238'
        return f"+39 {s[2:]}"
    return s


def _links(s: str) -> dict[str, str]:
    """'[English](url) · [German](url)' -> {'English': url, 'German': url}."""
    return dict(re.findall(r"\[([^\]]+)\]\(([^)]+)\)", s))


def _parse_file_name(path: Path) -> tuple[str, str, str]:
    """'ti_export_kiens_pois_Bakeries.txt' -> ('kiens', 'pois', 'Bakeries')."""
    m = re.match(r"ti_export_([a-z]+)_([a-z]+)_(.+)\.txt$", path.name)
    if not m:
        raise ValueError(f"Unexpected file name: {path.name}")
    return m.group(1), m.group(2), m.group(3)


# --------------------------------------------------------------------------- records

FIELD_RE = re.compile(r"^\*\*(?P<key>[^*:]+):\*\*\s*(?P<value>.*)$")  # **Phone:** ...
HEADING_RE = re.compile(r"^\*\*\s*(?P<title>[^*]+?)\s*\*\*$")  # ** Name **


def parse_record(block: str, category: str) -> dict:
    """Parse one '### BEGIN ...' block into a dict of normalised fields."""
    rec: dict = {"category": category, "facts": [], "description": [], "links": {}}
    lines = block.splitlines()
    in_about = False
    pending_hours = False

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if in_about:
            rec["description"].append(_clean(line))
            continue
        if pending_hours:
            rec["hours_raw"] = line
            pending_hours = False
            continue

        if m := re.match(r"^This is an? (.+?) in (.+)\.$", line):
            rec["type_label"] = m.group(1)
            if m.group(2) != "this area":
                rec["town"] = _clean(m.group(2))
        elif line.startswith("This is "):
            rec["type_label"] = line.removeprefix("This is ").rstrip(".")
        elif line.startswith("**Opening Hours"):
            pending_hours = True  # the raw hours are on the next non-empty line
        elif m := HEADING_RE.match(line):
            title = m.group("title")
            if title.startswith("About "):
                in_about = True
            elif title.startswith("Contact Information"):
                pass
            elif "name" not in rec:
                rec["name"] = _clean(title)
        elif m := FIELD_RE.match(line):
            key, value = m.group("key").strip(), m.group("value").strip()
            if key == "Also known as":
                rec["aliases"] = _clean(value.replace(" | ", "; "))
            elif key in {"Phone", "Mobile"}:
                if phone := _phone(value):
                    rec[key.lower()] = phone
            elif key == "Website":
                rec["website"] = value
            elif key == "CIN Code":
                rec["cin"] = value
            elif key.endswith("links"):
                rec["links"].update(_links(value))
            else:  # Summary, Overview, '<name>' (hike elevation line), ...
                label = "Details" if key == rec.get("name") else key
                rec["facts"].append(f"{label}: {_clean(value)}")
        elif " is located at: " in line:
            rec["address"] = _clean(line.split(" is located at: ", 1)[1])
        elif line.startswith("[Google Maps]"):
            if m := re.search(r"query=([\d.]+),([\d.]+)", line):
                rec["lat"], rec["lon"] = float(m.group(1)), float(m.group(2))
        elif line.startswith("Altitude:"):
            if m := re.search(r"([\d.]+)", line):
                rec["altitude_m"] = int(float(m.group(1)))
        elif " offers the following services: " in line:
            rec["services"] = line.split(" offers the following services: ", 1)[1]
        else:
            rec["facts"].append(_clean(line))

    if raw := rec.pop("hours_raw", None):
        rec["hours"] = parse_hours(raw)
    return rec


def render_record(rec: dict) -> str:
    """Compact, consistent text for one record (embedded *and* shown to the LLM)."""
    label = CATEGORY_LABELS.get(rec["category"], rec["category"])
    town = rec.get("town", "the Kiens/Chienes area")
    lines = [f"{rec['name']} — {label} — in {town}"]
    if aliases := rec.get("aliases"):
        lines.append(f"Also known as: {aliases}")
    lines += rec["facts"]
    if rec.get("hours"):
        lines.append(f"Opening hours: {format_hours(rec['hours'])}")
    elif rec["category"] not in {"EasyHikes"}:
        lines.append("Opening hours: not listed")
    if addr := rec.get("address"):
        alt = f" (altitude {rec['altitude_m']} m)" if "altitude_m" in rec else ""
        lines.append(f"Address: {addr}{alt}")
    contact = [f"{k.capitalize()}: {rec[k]}" for k in ("phone", "mobile", "website") if rec.get(k)]
    if contact:
        lines.append(" · ".join(contact))
    if services := rec.get("services"):
        lines.append(f"Services: {services}")
    if rec["description"]:
        lines.append("About: " + "\n".join(rec["description"]))
    return "\n".join(lines)


def record_metadata(rec: dict, source: str) -> dict:
    meta = {
        "kind": "record",
        "source": source,
        "category": rec["category"],
        "name": rec["name"],
        "town": rec.get("town", ""),
    }
    for lang, url in rec["links"].items():
        if lang in LANG_KEYS:
            meta[f"url_{LANG_KEYS[lang]}"] = url
        else:
            meta[f"url_{lang.lower()}"] = url  # e.g. url_outdooractive
    if rec.get("hours"):
        meta["hours_json"] = json.dumps([r.to_dict() for r in rec["hours"]])
    for key in ("lat", "lon", "altitude_m"):
        if key in rec:
            meta[key] = rec[key]
    return meta


def parse_structured_file(path: Path) -> list[Chunk]:
    _, _, category = _parse_file_name(path)
    text = path.read_text(encoding="utf-8")
    header, *blocks = re.split(r"^### BEGIN \w+\s*$", text, flags=re.M)

    chunks = []
    names = []
    for i, block in enumerate(blocks):
        rec = parse_record(block, category)
        if "name" not in rec:
            continue
        names.append(f"{rec['name']} ({rec.get('town', 'Kiens area')})")
        chunks += _split(f"{category.lower()}-{i}", render_record(rec), record_metadata(rec, path.name))

    # Overview chunk: answers "list all X" in one piece. Uses the file's own description
    # and example questions, which are written exactly like user questions.
    desc_lines = [ln.strip() for ln in header.splitlines() if ln.strip() and not ln.startswith("#")]
    desc = " ".join(desc_lines).replace("{location}", "Kiens")
    overview = (
        f"Overview — {CATEGORY_LABELS.get(category, category)} — Kiens/Chienes area\n"
        f"{desc}\n"
        f"All {len(names)} entries in this category: " + "; ".join(names) + "."
    )
    chunks.append(
        Chunk(f"{category.lower()}-overview", overview,
              {"kind": "overview", "source": path.name, "category": category, "name": f"All {category}"})
    )
    return chunks


# --------------------------------------------------------------------------- web pages

BOILERPLATE_LINES = {"read more", "1 / 1", "your feedback"}


def clean_web_text(text: str) -> tuple[str, str]:
    """Return (source_url, cleaned_text). Drops cookie banners, nav crumbs and repeated sentences."""
    lines = text.splitlines()
    url = ""
    if lines and (m := re.match(r"^\[(https?://[^\]]+)\]", lines[0].strip())):
        url, lines = m.group(1), lines[1:]

    out: list[str] = []
    seen: set[str] = set()
    for line in lines:
        s = line.strip()
        if s.lower().startswith("# privacy settings"):
            break  # consent banner until the end of file
        if s.lower() in BOILERPLATE_LINES:
            continue
        s = re.sub(r"^à\s+", "", s)  # stray arrow glyph from the scrape
        if not s or s.startswith("#"):
            out.append(s)
            continue
        s = _drop_repetition(s)
        key = re.sub(r"\W+", " ", s.lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()
    return url, cleaned


def _drop_repetition(paragraph: str) -> str:
    """The scrape sometimes doubles a paragraph's body: '**Title** A. B. A. B.' -> '**Title** A. B.'."""
    title = ""
    if m := re.match(r"^(\*\*[^*]+\*\*)\s*(.*)$", paragraph):
        title, paragraph = m.group(1) + " ", m.group(2)
    body = re.sub(r"\s+", " ", paragraph).strip()
    half = len(body) // 2
    for i in range(max(half - 3, 1), min(half + 4, len(body))):
        if body[:i].strip() == body[i:].strip():
            return title + body[:i].strip()
    return title + paragraph


def parse_web_file(path: Path) -> list[Chunk]:
    _, _, raw_title = _parse_file_name(path)
    title = re.sub(r"\s+", " ", raw_title).strip().removesuffix(" Kiens - Chienes")
    url, text = clean_web_text(path.read_text(encoding="utf-8"))
    base_meta = {"kind": "web", "source": path.name, "category": "Website", "name": title, "url_en": url}

    # Split on markdown headings; each section keeps its heading as context.
    sections = re.split(r"\n(?=#+ )", "\n" + text)
    chunks: list[Chunk] = []
    for i, section in enumerate(s.strip() for s in sections):
        if not section:
            continue
        heading, _, body = section.partition("\n")
        heading = heading.lstrip("# ").strip() if heading.startswith("#") else ""
        body = body if heading else section
        if len(body.strip()) < 40:
            continue  # heading-only fragments
        prefix = f"{title}" + (f" — {heading}" if heading else "")
        chunks += _split(f"web-{_slug(title)}-{i}", f"{prefix}\n{body.strip()}", base_meta)
    return chunks


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40]


def _split(chunk_id: str, text: str, meta: dict) -> list[Chunk]:
    """Keep text whole if it fits, else pack paragraphs; every piece repeats the first line (its title)."""
    if len(text) <= MAX_CHUNK_CHARS:
        return [Chunk(chunk_id, text, dict(meta))]
    title, _, body = text.partition("\n")
    paragraphs = [p for p in re.split(r"\n+", body) if p.strip()]
    pieces, cur = [], ""
    for p in paragraphs:
        if cur and len(cur) + len(p) > MAX_CHUNK_CHARS:
            pieces.append(cur)
            cur = ""
        cur = f"{cur}\n{p}" if cur else p
    if cur:
        pieces.append(cur)
    return [Chunk(f"{chunk_id}.{n}", f"{title}\n{p}", dict(meta)) for n, p in enumerate(pieces)]


# --------------------------------------------------------------------------- entry point


def load_corpus(data_dir: str | Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(Path(data_dir).glob("ti_export_*.txt")):
        _, kind, _ = _parse_file_name(path)
        chunks += parse_web_file(path) if kind == "website" else parse_structured_file(path)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids)), "duplicate chunk ids"
    return chunks
