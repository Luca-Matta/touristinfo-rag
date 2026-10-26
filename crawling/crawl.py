"""Extract structured data from a valgardena.it address page.

    python crawl.py                          # default URL (Bar Oswald), JSON to stdout
    python crawl.py <url>
    python crawl.py --html page.html         # parse a saved page (offline / tests)
    python crawl.py --format export          # same block format as data/ti_export_*.txt
    python crawl.py --schema                 # print the JSON schema of the output
    python crawl.py --no-llm                 # skip the optional LLM description

Approach: deterministic HTML parsing for every *fact* (name, phone, email, address,
coordinates, opening hours), and an LLM only for the one thing that is genuinely
generative: the 2-3 sentence description. The LLM therefore cannot invent a phone number
or an opening time, and the script works without an API key (the description then falls
back to a template built from the page's own text). The output is validated against a
pydantic schema, so a broken selector shows up as an error rather than as bad data.

Selectors are structural rather than tied to CSS class names (tel:/mailto: links, tables
whose header row is Mo..Su, section anchors), so small redesigns don't break the scraper.
The crawler checks robots.txt and identifies itself; it fetches one page per run.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import date
from itertools import pairwise
from typing import Literal
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup, Tag
from pydantic import BaseModel, Field

DEFAULT_URL = (
    "https://www.valgardena.it/en/addresses/detail/base/company/"
    "selva-di-val-gardena/bar-oswald/4206CF9CFA75D1664ECC5822E183F31B/"
)
USER_AGENT = "touristinfo-rag-crawler/1.0 (interview exercise; one page per run)"
HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "en;q=0.9,de;q=0.8,it;q=0.7"}
TIMEOUT_S = 20

DayCode = Literal["mo", "tu", "we", "th", "fr", "sa", "su"]
DAY_CODES: list[DayCode] = ["mo", "tu", "we", "th", "fr", "sa", "su"]
DAY_HEADER = re.compile(r"^(mo|tu|we|th|fr|sa|su|mon|tue|wed|thu|fri|sat|sun)\.?$", re.I)
DATE_RANGE = re.compile(r"(\d{2})\.(\d{2})\.(\d{4})\s*[-–]\s*(\d{2})\.(\d{2})\.(\d{4})")
TIME_RANGE = re.compile(r"(\d{1,2}[:.]\d{2})\s*[-–]\s*(\d{1,2}[:.]\d{2})")
POSTAL_LINE = re.compile(r"^(?P<postal_code>\d{5})\s+(?P<locality>[^-–,]+?)(?:\s*[-–,]\s*(?P<region>.+))?$")
HHMM = r"^\d{2}:\d{2}$"


class OpeningPeriod(BaseModel):
    valid_from: date | None = Field(description="first day of the validity period")
    valid_to: date | None = Field(description="last day of the validity period")
    days: list[DayCode] = Field(description="weekdays on which these hours apply")
    opens: str = Field(pattern=HHMM)
    closes: str = Field(pattern=HHMM)


class Address(BaseModel):
    street: str | None
    postal_code: str
    locality: str
    region: str | None
    country: str = "IT"


class Coordinates(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


class Contact(BaseModel):
    phone: str | None = None
    email: str | None = None
    website: str | None = None
    address: Address | None = None
    coordinates: Coordinates | None = None


class Place(BaseModel):
    name: str
    category: str | None = Field(None, description="the page's own label, e.g. 'Après Ski'")
    opening_hours: list[OpeningPeriod]
    contact: Contact
    description: str
    source_url: str
    notes: list[str] = Field(default_factory=list, description="assumptions and data problems found on the page")


class CrawlError(Exception):
    pass


def fetch_html(url: str) -> str:
    """Plain HTTP first; fall back to a headless browser if the page needs JavaScript."""
    if not allowed_by_robots(url):
        raise CrawlError(f"robots.txt disallows fetching {url}")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT_S)
        resp.raise_for_status()
        if BeautifulSoup(resp.text, "html.parser").find("h1"):
            return resp.text
        print("[crawl] page has no <h1> (rendered by JavaScript?); trying a headless browser", file=sys.stderr)
    except requests.RequestException as e:
        print(f"[crawl] plain request failed ({e}); trying a headless browser", file=sys.stderr)
    return fetch_with_browser(url)


def allowed_by_robots(url: str) -> bool:
    """True unless robots.txt explicitly disallows the URL (an unreachable robots.txt allows)."""
    parser = RobotFileParser()
    try:
        resp = requests.get(urljoin(url, "/robots.txt"), headers=HEADERS, timeout=TIMEOUT_S)
    except requests.RequestException:
        return True
    if resp.status_code >= 400:
        return True
    parser.parse(resp.text.splitlines())
    return parser.can_fetch(USER_AGENT, url)


def fetch_with_browser(url: str) -> str:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        raise CrawlError("the page needs a browser: pip install playwright && playwright install chromium") from e

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(user_agent=USER_AGENT)
        page.goto(url, wait_until="networkidle", timeout=45_000)
        html: str = page.content()
        browser.close()
    return html


def page_title(soup: BeautifulSoup) -> str:
    og = soup.find("meta", property="og:title")
    if isinstance(og, Tag) and og.get("content"):
        return str(og["content"])
    return soup.title.get_text(strip=True) if soup.title else ""


def extract_name(soup: BeautifulSoup) -> str | None:
    if h1 := soup.find("h1"):
        return h1.get_text(" ", strip=True)
    # "Après Ski: Bar Oswald - Selva di Val Gardena" -> "Bar Oswald"
    return re.sub(r"^[^:]+:\s*|\s+-\s+.*$", "", page_title(soup)).strip() or None


def extract_category(soup: BeautifulSoup) -> str | None:
    """The label before the colon in the title: 'Après Ski: Bar Oswald - ...' -> 'Après Ski'."""
    label, colon, _ = page_title(soup).partition(":")
    if not colon:
        return None
    return label.strip() or None


def decode_cfemail(encoded: str) -> str:
    """Cloudflare email obfuscation: first byte is the XOR key for the remaining bytes."""
    key = int(encoded[:2], 16)
    return "".join(chr(int(encoded[i : i + 2], 16) ^ key) for i in range(2, len(encoded), 2))


def extract_email(soup: BeautifulSoup) -> str | None:
    if a := soup.select_one('a[href^="mailto:"]'):
        return str(a["href"]).removeprefix("mailto:").split("?")[0]
    if el := soup.select_one("[data-cfemail]"):
        return decode_cfemail(str(el["data-cfemail"]))
    if a := soup.select_one('a[href*="/cdn-cgi/l/email-protection#"]'):
        return decode_cfemail(str(a["href"]).split("#", 1)[1])
    return None


def extract_phone(soup: BeautifulSoup) -> str | None:
    if a := soup.select_one('a[href^="tel:"]'):
        return a.get_text(strip=True) or str(a["href"]).removeprefix("tel:")
    return None


def _strip_tracking(url: str) -> str:
    parts = urlparse(url)
    query = {k: v for k, v in parse_qs(parts.query).items() if not k.startswith("utm_")}
    return urlunparse(parts._replace(query=urlencode(query, doseq=True)))


def contact_block(soup: BeautifulSoup) -> Tag | None:
    """Smallest ancestor of the phone (or email) link that also contains a postcode line."""
    node = soup.select_one('a[href^="tel:"]') or soup.select_one('a[href*="email-protection"], a[href^="mailto:"]')
    while node is not None and node.name != "body":
        if re.search(r"\b\d{5}\s+\w", node.get_text("\n")):
            return node
        node = node.parent
    return None


def extract_website(block: Tag | None, page_host: str) -> str | None:
    if block is None:
        return None
    for a in block.select('a[href^="http"]'):
        host = urlparse(str(a["href"])).netloc
        if host and page_host not in host and "google." not in host:
            return _strip_tracking(str(a["href"]))
    return None


def extract_address(block: Tag | None) -> Address | None:
    if block is None:
        return None
    lines = [ln.strip() for ln in block.get_text("\n").splitlines() if ln.strip()]
    for i, line in enumerate(lines):
        if m := POSTAL_LINE.match(line):
            street = lines[i - 1] if i > 0 and not TIME_RANGE.search(lines[i - 1]) else None
            return Address(
                street=street,
                postal_code=m["postal_code"],
                locality=m["locality"].strip(),
                region=(m["region"] or "").strip() or None,
            )
    return None


def extract_coordinates(soup: BeautifulSoup) -> Coordinates | None:
    """Route-planner / maps links carry the coordinates as query parameters."""
    for a in soup.select("a[href]"):
        href = str(a["href"])
        qs = parse_qs(urlparse(href).query)
        for lat_key, lon_key in (("toLat", "toLon"), ("lat", "lon"), ("lat", "lng")):
            if lat_key in qs and lon_key in qs:
                try:
                    return Coordinates(lat=float(qs[lat_key][0]), lon=float(qs[lon_key][0]))
                except ValueError:
                    pass
        if m := re.search(r"[?&](?:q|query|destination)=(-?\d+\.\d+),(-?\d+\.\d+)", href):
            return Coordinates(lat=float(m[1]), lon=float(m[2]))
    return None


def _cell_marked(cell: Tag) -> bool:
    """A weekday cell counts as 'open' if it holds any text, icon/element or a state class."""
    if cell.get_text(strip=True) or cell.find(True):  # text, <i class="icon-check">, <svg>, <img> ...
        return True
    return any(re.search(r"\b(open|active|checked?|yes)\b", c, re.I) for c in cell.get("class") or [])


def _hhmm(t: str) -> str:
    return t.replace(".", ":").zfill(5)


def extract_opening_hours(soup: BeautifulSoup) -> tuple[list[OpeningPeriod], list[str]]:
    """Parse every table whose header row lists weekdays.

    Expected layout (one table per validity period):
        | 01.06.2026 - 30.09.2026 | Mo | Tu | We | Th | Fr | Sa | Su |
        | 06:00 - 22:00           |  ✓ |  ✓ |    |  ✓ | ...
    Returns (periods, notes) where notes flag anything that had to be assumed.
    """
    scope = soup.select_one("#c_opening_times_container") or soup
    periods: list[OpeningPeriod] = []
    notes: list[str] = []
    for table in scope.find_all("table"):
        rows = table.find_all("tr")
        if not rows:
            continue
        header = [c.get_text(" ", strip=True) for c in rows[0].find_all(["th", "td"])]
        day_cols = [i for i, h in enumerate(header) if DAY_HEADER.match(h)]
        if len(day_cols) != 7:
            continue
        caption = table.find("caption")
        period_text = " ".join(header) + " " + (caption.get_text(" ") if caption else "")
        prev = table.find_previous(string=DATE_RANGE)
        m = DATE_RANGE.search(period_text) or (DATE_RANGE.search(prev) if prev else None)
        valid_from = date(int(m[3]), int(m[2]), int(m[1])) if m else None
        valid_to = date(int(m[6]), int(m[5]), int(m[4])) if m else None

        for row in rows[1:]:
            cells = row.find_all(["th", "td"])
            if not (t := TIME_RANGE.search(row.get_text(" "))):
                continue
            marked = [DAY_CODES[k] for k, col in enumerate(day_cols) if col < len(cells) and _cell_marked(cells[col])]
            if not marked:
                marked = list(DAY_CODES)
                note = "weekday columns carry no markers; assumed open every day"
                if note not in notes:
                    notes.append(note)
            periods.append(
                OpeningPeriod(
                    valid_from=valid_from, valid_to=valid_to, days=marked, opens=_hhmm(t[1]), closes=_hhmm(t[2])
                )
            )
    return periods, notes + _period_notes(periods)


def _period_notes(periods: list[OpeningPeriod]) -> list[str]:
    """Flag overlapping or implausibly long periods: usually data-entry errors on the source site."""
    spans = sorted({(p.valid_from, p.valid_to) for p in periods if p.valid_from and p.valid_to})
    notes = [
        f"period {a}..{b} is longer than a year (likely a typo on the source page)"
        for a, b in spans
        if (b - a).days > 366
    ]
    for (a_from, a_to), (b_from, b_to) in pairwise(spans):
        if b_from <= a_to:
            notes.append(f"periods {a_from}..{a_to} and {b_from}..{b_to} overlap")
    return notes


def extract_description_text(soup: BeautifulSoup) -> str:
    """Raw description text from the page (description section, else meta description)."""
    section: Tag | None = soup.select_one("#c_description_container")
    if section is None:
        heading = soup.find(lambda t: t.name in {"h2", "h3"} and t.get_text(strip=True).lower() == "description")
        section = heading.parent if heading else None
    if section is not None:
        text = re.sub(r"^description\s*", "", section.get_text(" ", strip=True), flags=re.I)
        if text:
            return text
    meta = soup.find("meta", attrs={"name": "description"})
    return str(meta["content"]).strip() if isinstance(meta, Tag) and meta.get("content") else ""


def template_description(name: str, category: str | None, locality: str | None, raw: str) -> str:
    """Offline fallback: two factual sentences built only from extracted fields."""
    kind = f"{category.lower()} venue" if category else "venue"
    article = "an" if kind[:1] in "aeiouà" else "a"
    first = f"{name} is {article} {kind} in {locality or 'Val Gardena'}."
    second = f"It offers {raw.rstrip('.').lower()}." if raw else ""
    return " ".join(s for s in (first, second) if s)


def llm_description(name: str, raw: str, page_text: str) -> str | None:
    if not os.getenv("OPENAI_API_KEY"):
        return None
    from openai import OpenAI

    prompt = (
        "Write a 2-3 sentence description of this place for a tourist information chatbot. "
        "Use only facts from the page text below; do not invent anything. Plain text, English.\n\n"
        f"Name: {name}\nDescription field: {raw}\n\nPage text (truncated):\n{page_text[:6000]}"
    )
    resp = OpenAI(timeout=60).chat.completions.create(
        model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini"),
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=300,
    )
    return (resp.choices[0].message.content or "").strip() or None


def extract(html: str, url: str, use_llm: bool = True) -> Place:
    soup = BeautifulSoup(html, "html.parser")
    name = extract_name(soup)
    if not name:
        raise CrawlError("no place name found: is this an address detail page?")
    block = contact_block(soup)
    hours, notes = extract_opening_hours(soup)
    if not hours:
        note = "no opening-hours table with Mo..Su columns found: opening_hours is empty, check the page"
        notes.append(note)
        print(f"[crawl] {note}", file=sys.stderr)
    contact = Contact(
        phone=extract_phone(soup),
        email=extract_email(soup),
        website=extract_website(block, urlparse(url).netloc.removeprefix("www.")),
        address=extract_address(block),
        coordinates=extract_coordinates(soup),
    )
    category = extract_category(soup)
    raw = extract_description_text(soup)

    description = None
    if use_llm:
        try:
            description = llm_description(name, raw, soup.get_text(" ", strip=True))
        except Exception as e:  # noqa: BLE001 - a failed summary must not lose the extracted facts
            print(f"[crawl] LLM description failed ({e}); using the template", file=sys.stderr)
    if description is None:
        description = template_description(name, category, contact.address.locality if contact.address else None, raw)

    return Place(
        name=name,
        category=category,
        opening_hours=hours,
        contact=contact,
        description=description,
        source_url=url,
        notes=notes,
    )


def to_export_block(place: Place) -> str:
    """Render in the format of data/ti_export_*.txt, so the RAG can ingest crawled pages.

    The export format has year-less periods (dd.mm) and no email field: both are lost.
    """
    lines = ["### BEGIN POI", ""]
    locality = place.contact.address.locality if place.contact.address else "this area"
    kind = (place.category or "place").lower()
    lines += [f"This is {'an' if kind[:1] in 'aeiouà' else 'a'} {kind} in {locality}.", "", f"** {place.name} **", ""]
    if place.opening_hours:
        entries = [
            str(
                {
                    "period_start": f"{p.valid_from:%d.%m}" if p.valid_from else "01.01",
                    "period_end": f"{p.valid_to:%d.%m}" if p.valid_to else "31.12",
                    "service_type": "General",
                    "hours": f"{p.opens} - {p.closes}",
                    "days": p.days,
                }
            )
            for p in place.opening_hours
        ]
        lines += [f"**Opening Hours for {place.name}:**", "", ", ".join(entries), ""]
    if a := place.contact.address:
        parts = [x for x in (a.street, a.postal_code, a.locality) if x]
        lines += [f"{place.name} is located at: {', '.join(parts)}", ""]
    if c := place.contact.coordinates:
        lines += [f"[Google Maps](https://www.google.com/maps/search/?api=1&query={c.lat},{c.lon})", ""]
    contact = [
        f"**{label}:** {value}"
        for label, value in (("Phone", place.contact.phone), ("Website", place.contact.website))
        if value
    ]
    if contact:
        lines += [f"** Contact Information for {place.name}**", ""] + [x for line in contact for x in (line, "")]
    lines += [f"** About {place.name}**", "", place.description, ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("url", nargs="?", default=DEFAULT_URL)
    parser.add_argument("--html", help="parse a saved HTML file instead of fetching")
    parser.add_argument("--format", choices=["json", "export"], default="json")
    parser.add_argument("--schema", action="store_true", help="print the output JSON schema and exit")
    parser.add_argument("--no-llm", action="store_true", help="never call the LLM")
    args = parser.parse_args(argv)

    if args.schema:
        print(json.dumps(Place.model_json_schema(), indent=2))
        return 0
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    try:
        if args.html:
            with open(args.html, encoding="utf-8") as f:
                html = f.read()
        else:
            html = fetch_html(args.url)
        place = extract(html, args.url, use_llm=not args.no_llm)
    except (CrawlError, requests.RequestException, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(place.model_dump_json(indent=2) if args.format == "json" else to_export_block(place))
    return 0


if __name__ == "__main__":
    sys.exit(main())
