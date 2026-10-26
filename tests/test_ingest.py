from rag.chunking import chunk_record
from rag.ingest.exports import parse_record
from rag.models import Category, RecordKind


def test_whole_corpus_parses_without_warnings(corpus):
    assert corpus.warnings == []
    kinds = [r.kind for r in corpus]
    assert kinds.count(RecordKind.PLACE) == 15
    assert kinds.count(RecordKind.OVERVIEW) == 7
    assert kinds.count(RecordKind.WEB_SECTION) == 11


def test_record_ids_are_stable(corpus):
    # Eval questions reference these ids: renaming one must be a conscious change.
    for rid in (
        "bakery-3383",
        "butcher-2625",
        "castle-2767",
        "church-3191",
        "bed_and_breakfast-12399",
        "easy_hike-5070",
        "pizzeria-gasthof-pizzeria-petra",
        "web-accessible-travel-1",
        "church-overview",
    ):
        assert rid in corpus.records


def test_place_fields(corpus):
    bakery = corpus["bakery-3383"]
    assert bakery.category is Category.BAKERY
    assert bakery.phone == "+39 0474 565312"
    assert bakery.address == "Kienerdorfweg 2, 39030, Kiens"  # no '39030.0' float artefact
    assert bakery.altitude_m == 780
    assert bakery.aliases["de"] == "Gatterer Bäckerei- Konditorei mit Cafè"
    assert bakery.hours is not None
    assert set(bakery.links) == {"en", "de", "it"}

    hike = corpus["easy_hike-5070"]
    assert any("3.2km" in f for f in hike.facts)
    assert "route" in hike.links


def test_link_follows_language_with_english_fallback(corpus):
    bakery = corpus["bakery-3383"]
    assert "/it/" in bakery.link_for("it")
    assert "/en/" in bakery.link_for(None)
    assert "/it/" in bakery.context_text("it") and "/de/" not in bakery.context_text("it")


def test_search_text_has_no_urls_but_has_multilingual_labels(corpus):
    text = corpus["butcher-2625"].search_text()
    assert "http" not in text
    assert "Metzgerei" in text and "macelleria" in text


def test_services_filter(corpus):
    assert {r.id for r in corpus.with_services(["sauna"])} == {"bed_and_breakfast-36275", "bed_and_breakfast-12399"}
    assert [r.id for r in corpus.with_services(["Dogs allowed", "sauna"])] == ["bed_and_breakfast-12399"]
    assert corpus.with_services(["helipad"]) == []


def test_services_synonyms_and_negations(corpus):
    # Vanea lists "No pets or domestic animals": a negated service never matches.
    assert [r.id for r in corpus.with_services(["pets"])] == ["bed_and_breakfast-12399"]
    assert [r.id for r in corpus.with_services(["pet friendly"])] == ["bed_and_breakfast-12399"]
    assert [r.id for r in corpus.with_services(["gluten-free"])] == ["bed_and_breakfast-12399"]


def test_overview_lists_every_place(corpus):
    overview = corpus["church-overview"].description
    assert "All 5 entries" in overview
    assert "The only entry" in corpus["pizzeria-overview"].description


def test_web_pages_drop_boilerplate(corpus):
    web = " ".join(r.description for r in corpus if r.kind is RecordKind.WEB_SECTION)
    assert "cookie" not in web.lower()


def test_unknown_field_is_reported_not_dropped_silently():
    block = "This is a bakery in Kiens.\n** Test Place **\n**Shoe size:** 42\nTest Place is located at: Somewhere 1"
    record, warnings = parse_record(block, Category.BAKERY, "test.txt")
    assert record is not None and record.name == "Test Place"
    assert any("Shoe size" in w for w in warnings)


def test_long_records_are_split_with_repeated_header(corpus):
    record = corpus["castle-2767"]
    chunks = chunk_record(record, max_chars=400)
    assert len(chunks) > 1
    assert all(c.record_id == record.id and c.text.startswith(record.header) for c in chunks)
    assert chunk_record(record)[0].id == record.id  # fits in one chunk by default
