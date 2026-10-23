"""Bonus crawler: parsing tests against a synthetic page (no network)."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("crawl", ROOT / "crawling" / "crawl.py")
crawl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(crawl)

HTML = (ROOT / "tests" / "fixtures" / "address_page_synthetic.html").read_text(encoding="utf-8")
DATA = crawl.extract(HTML, crawl.DEFAULT_URL, use_llm=False)


def test_name_and_contact():
    assert DATA["name"] == "Example Bar"
    c = DATA["contact"]
    assert c["phone"] == "+39 000 0000000"
    assert c["email"] == "info@example.com"  # decoded from Cloudflare obfuscation
    assert c["website"] == "https://www.example-bar.it/?lang=en"  # utm_* stripped
    assert c["address"] == {
        "street": "Streda Example 1",
        "postal_code": "39048",
        "locality": "Selva di Val Gardena",
        "region": "South Tyrol",
        "country": "IT",
    }
    assert c["coordinates"] == {"lat": 46.555, "lon": 11.76}


def test_opening_hours_periods_and_day_markers():
    hours = DATA["opening_hours"]
    assert [(h["valid_from"], h["opens"], h["closes"]) for h in hours] == [
        ("2026-06-01", "06:00", "22:00"),
        ("2026-10-01", "06:00", "15:00"),
        ("2026-12-01", "06:00", "22:00"),
    ]
    assert hours[1]["days"] == ["mo", "tu", "th", "fr"]  # icons mark the open days
    assert hours[0]["days"] == crawl.DAY_CODES  # no markers -> assumed every day, and flagged
    assert any("assumed open every day" in n for n in DATA["notes"])
    assert any("overlap" in n for n in DATA["notes"])  # 01.10-03.12 overlaps 01.12-30.04


def test_template_description_without_llm():
    assert DATA["description"].startswith("Example Bar is an après ski venue in Selva di Val Gardena.")


def test_cfemail_decoder():
    assert crawl.decode_cfemail("5a33343c351a3f223b372a363f74393537") == "info@example.com"
