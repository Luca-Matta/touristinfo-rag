"""Bonus crawler: parsing tests against a synthetic page (no network)."""

import importlib.util
import sys
from datetime import date
from pathlib import Path

from rag.ingest.exports import parse_record
from rag.models import Category

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("crawl", ROOT / "crawling" / "crawl.py")
assert spec and spec.loader
crawl = importlib.util.module_from_spec(spec)
sys.modules["crawl"] = crawl  # pydantic resolves the models' annotations through the module
spec.loader.exec_module(crawl)

HTML = (ROOT / "tests" / "fixtures" / "address_page_synthetic.html").read_text(encoding="utf-8")
PLACE = crawl.extract(HTML, crawl.DEFAULT_URL, use_llm=False)


def test_name_category_and_contact():
    assert PLACE.name == "Example Bar"
    assert PLACE.category == "Après Ski"
    c = PLACE.contact
    assert c.phone == "+39 000 0000000"
    assert c.email == "info@example.com"  # decoded from Cloudflare obfuscation
    assert c.website == "https://www.example-bar.it/?lang=en"  # utm_* stripped
    assert c.address.model_dump() == {
        "street": "Streda Example 1",
        "postal_code": "39048",
        "locality": "Selva di Val Gardena",
        "region": "South Tyrol",
        "country": "IT",
    }
    assert (c.coordinates.lat, c.coordinates.lon) == (46.555, 11.76)


def test_opening_hours_periods_and_day_markers():
    hours = PLACE.opening_hours
    assert [(h.valid_from, h.opens, h.closes) for h in hours] == [
        (date(2026, 6, 1), "06:00", "22:00"),
        (date(2026, 10, 1), "06:00", "15:00"),
        (date(2026, 12, 1), "06:00", "22:00"),
    ]
    assert hours[1].days == ["mo", "tu", "th", "fr"]  # icons mark the open days
    assert hours[0].days == crawl.DAY_CODES  # no markers -> assumed every day, and flagged
    assert any("assumed open every day" in n for n in PLACE.notes)
    assert any("overlap" in n for n in PLACE.notes)  # 01.10-03.12 overlaps 01.12-30.04


def test_implausible_period_is_flagged():
    typo = HTML.replace("01.06.2026 - 30.09.2026", "01.06.2026 - 30.09.2027")
    notes = crawl.extract(typo, crawl.DEFAULT_URL, use_llm=False).notes
    assert any("longer than a year" in n for n in notes)


def test_template_description_without_llm():
    assert PLACE.description == (
        "Example Bar is an après ski venue in Selva di Val Gardena. "
        "It offers pub, cocktails, beer, sandwiches, sun terrace, music."
    )


def test_cfemail_decoder():
    assert crawl.decode_cfemail("5a33343c351a3f223b372a363f74393537") == "info@example.com"


def test_export_format_round_trips_through_the_rag_parser():
    block = crawl.to_export_block(PLACE).removeprefix("### BEGIN POI")
    record, warnings = parse_record(block, Category.PIZZERIA, "crawled.txt")
    assert warnings == []
    assert record.name == "Example Bar" and record.town == "Selva di Val Gardena"
    assert record.phone == "+39 000 0000000"
    assert record.hours is not None and record.coordinates == (46.555, 11.76)


def test_schema_and_cli(capsys):
    assert crawl.main(["--schema"]) == 0
    assert "opening_hours" in capsys.readouterr().out
    assert crawl.main(["--html", str(ROOT / "tests/fixtures/address_page_synthetic.html"), "--no-llm"]) == 0
    assert '"name": "Example Bar"' in capsys.readouterr().out


def test_missing_hours_table_is_reported_not_silent():
    without_header = HTML.replace("<th>Mo</th>", "<th>Montag</th>")
    place = crawl.extract(without_header, crawl.DEFAULT_URL, use_llm=False)
    assert place.opening_hours == []
    assert any("no opening-hours table" in n for n in place.notes)


def test_http_error_stops_instead_of_parsing_the_error_page(monkeypatch):
    class Blocked:
        status_code = 503
        text = "<html><h1>Service Unavailable</h1></html>"

    monkeypatch.setattr(crawl, "allowed_by_robots", lambda url: True)
    monkeypatch.setattr(crawl.requests, "get", lambda *a, **kw: Blocked())
    assert crawl.main([crawl.DEFAULT_URL, "--no-llm"]) == 1
