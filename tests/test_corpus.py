from collections import Counter

from rag.config import Settings
from rag.corpus import clean_web_text, load_corpus
from rag.retrieval import record_id

CHUNKS = load_corpus(Settings().data_dir)
BY_NAME = {c.metadata["name"]: c for c in CHUNKS}


def test_every_record_is_parsed():
    records = Counter(
        {record_id(c.id): c.metadata["category"] for c in CHUNKS if c.metadata["kind"] == "record"}.values()
    )
    assert records == {"BedAndBreakfasts": 2, "EasyHikes": 1, "Bakeries": 2, "Bucheries": 1, "Castles": 3, "Churches": 5, "Pizzerias": 1}


def test_record_fields_are_normalised():
    text = BY_NAME["Gatterer bakery-confectionery with café"].text
    assert "Opening hours: Mon-Fri 06:30-19:00; Sat 06:30-18:00; closed Sun" in text
    assert "39030," in text and "39030.0" not in text
    assert "period_start" not in text
    assert "Bäckerei" in text  # cross-lingual labels


def test_metadata_has_links_in_three_languages():
    meta = BY_NAME["Castle Ehrenburg/Casteldarne"].metadata
    assert {"url_en", "url_de", "url_it"} <= meta.keys()


def test_overview_lists_all_entries():
    assert "All 5 entries" in BY_NAME["All Churches"].text


def test_web_cleanup_drops_boilerplate_and_duplicates():
    raw = (
        "[https://x.org](https://x.org) \n\n**Title** Alpha one two three. Beta four five six. "
        "Alpha one two three. Beta four five six.\n\nRead more \n\n# Privacy Settings\n\ncookies"
    )
    url, text = clean_web_text(raw)
    assert url == "https://x.org"
    assert text == "**Title** Alpha one two three. Beta four five six."


def test_chunks_stay_small():
    assert max(len(c.text) for c in CHUNKS) < 2000
