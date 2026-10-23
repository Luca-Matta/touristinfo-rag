"""Extract structured data from a valgardena.it address page and print it as JSON.

    python crawl.py                      # default URL (Bar Oswald)
    python crawl.py <url>
    python crawl.py --html page.html     # parse a saved page (offline / tests)
    python crawl.py --no-llm             # skip the optional LLM summary

Approach: deterministic HTML parsing for every *fact* (name, phone, email, address,
coordinates, opening hours), and an LLM only for the one thing that is genuinely
generative: the 2-3 sentence description. That way the LLM can't hallucinate a phone
number or opening time, and the script still works without an API key (the description
then falls back to a template built from the page's own text).

The selectors are deliberately structural rather than tied to CSS class names
(tel:/mailto: links, tables whose header row is Mo..Su, section anchors), so small
redesigns of the site don't break the scraper.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import date
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup, Tag

DEFAULT_URL = (
    "https://www.valgardena.it/en/addresses/detail/base/company/"
    "selva-di-val-gardena/bar-oswald/4206CF9CFA75D1664ECC5822E183F31B/"
)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36",
    "Accept-Language": "en;q=0.9,de;q=0.8,it;q=0.7",
}
DAY_CODES = ["mo", "tu", "we", "th", "fr", "sa", "su"]
DAY_HEADER = re.compile(r"^(mo|tu|we|th|fr|sa|su|mon|tue|wed|thu|fri|sat|sun)\.?$", re.I)
DATE_RANGE = re.compile(r"(\d{2})\.(\d{2})\.(\d{4})\s*[-–]\s*(\d{2})\.(\d{2})\.(\d{4})")
TIME_RANGE = re.compile(r"(\d{1,2}[:.]\d{2})\s*[-–]\s*(\d{1,2}[:.]\d{2})")
POSTAL_LINE = re.compile(r"^(?P<postal_code>\d{5})\s+(?P<locality>[^-–,]+?)(?:\s*[-–,]\s*(?P<region>.+))?$")


# --------------------------------------------------------------------------- fetching


def fetch_html(url: str) -> str:
    """Plain HTTP first; fall back to a headless browser if the page needs JavaScript."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        if BeautifulSoup(resp.text, "html.parser").find("h1"):
            return resp.text
    except requests.RequestException as e:
        print(f"[crawl] plain request failed ({e}); trying a headless browser", file=sys.stderr)
    return fetch_with_browser(url)


def fetch_with_browser(url: str) -> str:
    from playwright.sync_api import sync_playwright  # optional heavy dependency, imported lazily

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(user_agent=HEADERS["User-Agent"])
        page.goto(url, wait_until="networkidle", timeout=45_000)
        html = page.content()
        browser.close()
    return html


# --------------------------------------------------------------------------- field extractors


def extract_name(soup: BeautifulSoup) -> str | None:
    if h1 := soup.find("h1"):
        return h1.get_text(" ", strip=True)
    og = soup.find("meta", property="og:title")
    title = og["content"] if og else (soup.title.string if soup.title else "")
    # "Après Ski: Bar Oswald - Selva di Val Gardena" -> "Bar Oswald"
    return re.sub(r"^[^:]+:\s*|\s+-\s+.*$", "", title).strip() or None


def decode_cfemail(encoded: str) -> str:
    """Cloudflare email obfuscation: first byte is the XOR key for the remaining bytes."""
    key = int(encoded[:2], 16)
    return "".join(chr(int(encoded[i : i + 2], 16) ^ key) for i in range(2, len(encoded), 2))


def extract_email(soup: BeautifulSoup) -> str | None:
    if a := soup.select_one('a[href^="mailto:"]'):
        return a["href"].removeprefix("mailto:").split("?")[0]
    if el := soup.select_one("[data-cfemail]"):
        return decode_cfemail(el["data-cfemail"])
    if a := soup.select_one('a[href*="/cdn-cgi/l/email-protection#"]'):
        return decode_cfemail(a["href"].split("#", 1)[1])
    return None


def extract_phone(soup: BeautifulSoup) -> str | None:
    if a := soup.select_one('a[href^="tel:"]'):
        return a.get_text(strip=True) or a["href"].removeprefix("tel:")
    return None


def _strip_tracking(url: str) -> str:
    parts = urlparse(url)
    query = {k: v for k, v in parse_qs(parts.query).items() if not k.startswith("utm_")}
    return urlunparse(parts._replace(query=urlencode(query, doseq=True)))


def contact_block(soup: BeautifulSoup) -> Tag | None:
    """Smallest ancestor of the phone link that also contains a postal code line."""
    anchor = soup.select_one('a[href^="tel:"]') or soup.select_one('a[href*="email-protection"], a[href^="mailto:"]')
    node = anchor
    while node is not None and node.name != "body":
        if re.search(r"\b\d{5}\s+\w", node.get_text("\n")):
            return node
        node = node.parent
    return None


def extract_website(block: Tag | None, page_host: str) -> str | None:
    if block is None:
        return None
    for a in block.select('a[href^="http"]'):
        host = urlparse(a["href"]).netloc
        if host and page_host not in host and "google." not in host:
            return _strip_tracking(a["href"])
    return None


def extract_address(block: Tag | None) -> dict | None:
    if block is None:
        return None
    lines = [ln.strip() for ln in block.get_text("\n").splitlines() if ln.strip()]
    for i, line in enumerate(lines):
        if m := POSTAL_LINE.match(line):
            street = lines[i - 1] if i > 0 and not TIME_RANGE.search(lines[i - 1]) else None
            return {
                "street": street,
                "postal_code": m["postal_code"],
                "locality": m["locality"].strip(),
                "region": (m["region"] or "").strip() or None,
                "country": "IT",
            }
    return None


def extract_coordinates(soup: BeautifulSoup) -> dict | None:
    """Route-planner / maps links carry the coordinates as query parameters."""
    for a in soup.select("a[href]"):
        qs = parse_qs(urlparse(a["href"]).query)
        for lat_key, lon_key in (("toLat", "toLon"), ("lat", "lon"), ("lat", "lng")):
            if lat_key in qs and lon_key in qs:
                try:
                    return {"lat": float(qs[lat_key][0]), "lon": float(qs[lon_key][0])}
                except ValueError:
                    pass
        if m := re.search(r"[?&](?:q|query|destination)=(-?\d+\.\d+),(-?\d+\.\d+)", a["href"]):
            return {"lat": float(m[1]), "lon": float(m[2])}
    return None


def _cell_marked(cell: Tag) -> bool:
    """A weekday cell counts as 'open' if it holds any text, icon/element or a state class."""
    if cell.get_text(strip=True):
        return True
    if cell.find(True):  # <i class="icon-check">, <svg>, <img> ...
        return True
    return any(re.search(r"\b(open|active|checked?|yes)\b", c, re.I) for c in cell.get("class", []))


def extract_opening_hours(soup: BeautifulSoup) -> tuple[list[dict], list[str]]:
    """Parse every table whose header row lists weekdays.

    Expected layout (one table per validity period):
        | 01.06.2026 - 30.09.2026 | Mo | Tu | We | Th | Fr | Sa | Su |
        | 06:00 - 22:00           |  ✓ |  ✓ |    |  ✓ | ...
    Returns (periods, notes) where notes flag anything we had to assume.
    """
    scope = soup.select_one("#c_opening_times_container") or soup
    periods, notes = [], []
    for table in scope.find_all("table"):
        rows = table.find_all("tr")
        if not rows:
            continue
        header = [c.get_text(" ", strip=True) for c in rows[0].find_all(["th", "td"])]
        day_cols = [i for i, h in enumerate(header) if DAY_HEADER.match(h)]
        if len(day_cols) != 7:
            continue
        period_text = " ".join(header) + " " + (table.find("caption").get_text(" ") if table.find("caption") else "")
        prev = table.find_previous(string=DATE_RANGE)
        m = DATE_RANGE.search(period_text) or (DATE_RANGE.search(prev) if prev else None)
        valid_from = f"{m[3]}-{m[2]}-{m[1]}" if m else None
        valid_to = f"{m[6]}-{m[5]}-{m[4]}" if m else None

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
                {
                    "valid_from": valid_from,
                    "valid_to": valid_to,
                    "days": marked,
                    "opens": t[1].replace(".", ":").zfill(5),
                    "closes": t[2].replace(".", ":").zfill(5),
                }
            )
    notes += _overlap_notes(periods)
    return periods, notes


def _overlap_notes(periods: list[dict]) -> list[str]:
    """Flag overlapping validity periods; usually a data-entry error on the source site."""
    spans = sorted(
        {(p["valid_from"], p["valid_to"]) for p in periods if p["valid_from"] and p["valid_to"]}
    )
    notes = []
    for (a_from, a_to), (b_from, b_to) in zip(spans, spans[1:]):
        if date.fromisoformat(b_from) <= date.fromisoformat(a_to):
            notes.append(f"periods {a_from}..{a_to} and {b_from}..{b_to} overlap (likely a typo on the source page)")
    return notes


def extract_description_text(soup: BeautifulSoup) -> str:
    """Raw description text from the page (description section, else meta description)."""
    section = soup.select_one("#c_description_container")
    if section is None:
        heading = soup.find(lambda t: t.name in {"h2", "h3"} and t.get_text(strip=True).lower() == "description")
        section = heading.parent if heading else None
    if section is not None:
        text = section.get_text(" ", strip=True)
        text = re.sub(r"^description\s*", "", text, flags=re.I)
        if text:
            return text
    meta = soup.find("meta", attrs={"name": "description"})
    return meta["content"].strip() if meta and meta.get("content") else ""


# --------------------------------------------------------------------------- description


def template_description(data: dict, raw: str, title: str) -> str:
    """Offline fallback: two factual sentences built only from extracted fields."""
    kind = title.split(":", 1)[0].strip() if ":" in title else "venue"
    where = (data["contact"].get("address") or {}).get("locality") or "Val Gardena"
    first = f"{data['name']} is {'an' if kind[:1].lower() in 'aeiou' else 'a'} {kind.lower()} venue in {where}."
    second = f"It offers {raw.rstrip('.').lower()}." if raw else ""
    return " ".join(s for s in (first, second) if s)


def llm_description(data: dict, raw: str, page_text: str) -> str | None:
    if not os.getenv("OPENAI_API_KEY"):
        return None
    from openai import OpenAI

    prompt = (
        "Write a 2-3 sentence description of this place for a tourist information chatbot. "
        "Use only facts from the page text below; do not invent anything. Plain text, English.\n\n"
        f"Name: {data['name']}\nDescription field: {raw}\n\nPage text (truncated):\n{page_text[:6000]}"
    )
    resp = OpenAI().chat.completions.create(
        model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini"),
        messages=[{"role": "user", "content": prompt}],
    )
    return (resp.choices[0].message.content or "").strip() or None


# --------------------------------------------------------------------------- main


def extract(html: str, url: str, use_llm: bool = True) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""
    block = contact_block(soup)
    hours, notes = extract_opening_hours(soup)
    data = {
        "name": extract_name(soup),
        "opening_hours": hours,
        "contact": {
            "phone": extract_phone(soup),
            "email": extract_email(soup),
            "website": extract_website(block, urlparse(url).netloc.removeprefix("www.")),
            "address": extract_address(block),
            "coordinates": extract_coordinates(soup),
        },
        "description": "",
        "source_url": url,
        "notes": notes,
    }
    raw = extract_description_text(soup)
    description = None
    if use_llm:
        try:
            description = llm_description(data, raw, soup.get_text(" ", strip=True))
        except Exception as e:  # network/auth problems should not lose the extracted facts
            print(f"[crawl] LLM summary failed ({e}); using template", file=sys.stderr)
    data["description"] = description or template_description(data, raw, title)
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("url", nargs="?", default=DEFAULT_URL)
    parser.add_argument("--html", help="parse a saved HTML file instead of fetching")
    parser.add_argument("--no-llm", action="store_true", help="never call the LLM")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    html = open(args.html, encoding="utf-8").read() if args.html else fetch_html(args.url)
    print(json.dumps(extract(html, args.url, use_llm=not args.no_llm), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
