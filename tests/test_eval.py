from eval.run_eval import fact_check, forbidden_phrases, load_items


def test_questions_reference_existing_records(corpus):
    items = load_items("all")
    assert len({i.id for i in items}) == len(items)
    assert {i.split for i in items} == {"dev", "test"}
    assert all(rid in corpus.records for i in items for rid in i.expected)


def test_fact_check_ignores_spacing_and_accents():
    answer = "Call +39 0474 565312 — the Bäckerei opens at 6:30."
    assert fact_check(answer, [["0474565312"], ["06:30", "6:30"], ["backerei"]]) == []
    assert fact_check(answer, [["Sunday", "Sonntag"]]) == ["Sunday | Sonntag"]


def test_numbers_match_whole_and_words_from_their_start():
    answer = "Haus Lerchner (+39 3664474827) has 5 rooms, open Tue-Sun 10-18, on the Romans' trail."
    assert fact_check(answer, [["5"], ["10"], ["18"], ["Roman"]]) == []
    assert fact_check(answer, [["4"]]) == ["4"]  # only inside the phone number
    assert fact_check(answer, [["oman"]]) == ["oman"]


def test_forbidden_phrases():
    assert forbidden_phrases("Both include breakfast.", ["both include breakfast", "pets"]) == [
        "both include breakfast"
    ]
