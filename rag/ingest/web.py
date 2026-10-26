"""Parser for the scraped web pages (``ti_export_<town>_website_<Title>.txt``).

The scrapes are noisy: a cookie banner at the end, "Read more" crumbs, a stray arrow
glyph, and on some pages each paragraph body appears twice ("A. B. A. B."). They are
cleaned and split into one record per section (markdown heading), so a question about
wheelchairs on buses retrieves the "Buses" section rather than the whole page.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..models import Category, Record, RecordKind
from .exports import ParseResult
from .text import clean, fold, slugify

BOILERPLATE_LINES = {"read more", "1 / 1", "your feedback"}
END_OF_CONTENT = "# privacy settings"  # the consent banner runs from here to the end of file
BOLD_LEAD = re.compile(r"^(\*\*[^*]+\*\*)\s*(.*)$")
MIN_SECTION_CHARS = 40


def clean_page(text: str) -> tuple[str, str]:
    """Return (source_url, cleaned markdown)."""
    lines = text.splitlines()
    url = ""
    if lines and (m := re.match(r"^\[(https?://[^\]]+)\]", lines[0].strip())):
        url, lines = m.group(1), lines[1:]

    out: list[str] = []
    seen: set[str] = set()
    for raw in lines:
        line = raw.strip()
        if line.lower().startswith(END_OF_CONTENT):
            break
        if line.lower() in BOILERPLATE_LINES:
            continue
        line = re.sub(r"^à\s+", "", line)
        if line and not line.startswith("#"):
            line = drop_doubled_body(line)
            key = re.sub(r"\W+", " ", fold(line)).strip()
            if key in seen:
                continue
            seen.add(key)
        out.append(line)
    return url, re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def drop_doubled_body(paragraph: str) -> str:
    """'**Title** A. B. A. B.' -> '**Title** A. B.' (the scrape repeats paragraph bodies)."""
    title = ""
    if m := BOLD_LEAD.match(paragraph):
        title, paragraph = m.group(1) + " ", m.group(2)
    body = re.sub(r"\s+", " ", paragraph).strip()
    half = len(body) // 2
    for cut in range(max(half - 3, 1), min(half + 4, len(body))):  # allow for a stray space
        if body[:cut].strip() == body[cut:].strip():
            return title + body[:cut].strip()
    return title + paragraph


def page_title(path: Path) -> str:
    raw = path.stem.split("_website_", 1)[-1]
    return re.sub(r"\s+", " ", raw).strip().removesuffix(" Kiens - Chienes")


def parse_web_file(path: Path) -> ParseResult:
    title = page_title(path)
    url, text = clean_page(path.read_text(encoding="utf-8"))
    result = ParseResult([])
    for n, section in enumerate(re.split(r"\n(?=#+ )", "\n" + text)):
        section = section.strip()
        heading, _, body = section.partition("\n")
        if not heading.startswith("#"):
            heading, body = "", section
        body = body.strip()
        if len(body) < MIN_SECTION_CHARS:
            continue
        heading = heading.lstrip("# ").strip()
        result.records.append(
            Record(
                id=f"web-{slugify(title, 32)}-{n}",
                kind=RecordKind.WEB_SECTION,
                category=Category.WEB_PAGE,
                name=f"{title} — {heading}" if heading else title,
                source_file=path.name,
                links={"en": url} if url else {},
                description="\n".join(clean(ln) for ln in body.splitlines() if ln.strip()),
            )
        )
    if not result.records:
        result.warnings.append(f"{path.name}: no content after cleaning")
    return result
