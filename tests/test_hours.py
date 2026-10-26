from datetime import date, datetime

from rag.hours import OpeningHours, StatusKind

BAKERY = OpeningHours.parse(
    "{'period_start': '01.01', 'period_end': '31.12', 'service_type': 'General', 'hours': '07:45 - 12:00', 'days': ['mo', 'tu', 'we', 'th', 'fr']}, "
    "{'period_start': '01.01', 'period_end': '31.12', 'service_type': 'General', 'hours': '07:45 - 12:00', 'days': ['sa']}, "
    "{'period_start': '01.01', 'period_end': '31.12', 'service_type': 'General', 'hours': '15:00 - 18:30', 'days': ['mo', 'tu', 'we', 'th', 'fr']}"
)
MUSEUM = OpeningHours.parse(
    "{'period_start': '01.05', 'period_end': '05.11', 'service_type': 'General', 'hours': '10:00 - 18:00', 'days': ['tu', 'we', 'th', 'fr', 'sa', 'su']}"
)
assert BAKERY and MUSEUM


def test_describe_groups_days_and_says_what_is_missing():
    assert BAKERY.describe() == "Mon-Fri 07:45-12:00, 15:00-18:30; Sat 07:45-12:00; other days not listed"
    assert MUSEUM.describe() == "Tue-Sun 10:00-18:00; other days not listed (season 01.05 - 05.11)"


def test_status_kinds():
    cases = [
        (BAKERY, datetime(2026, 10, 23, 9, 0), StatusKind.OPEN),  # Friday morning
        (BAKERY, datetime(2026, 10, 23, 13, 0), StatusKind.CLOSED),  # lunch break
        (BAKERY, datetime(2026, 10, 23, 18, 30), StatusKind.CLOSED),  # closing time is exclusive
        (BAKERY, datetime(2026, 10, 25, 9, 0), StatusKind.NOT_LISTED),  # Sunday: no hours given
        (MUSEUM, datetime(2026, 7, 14, 11, 0), StatusKind.OPEN),  # Tuesday in season
        (MUSEUM, datetime(2026, 7, 13, 11, 0), StatusKind.NOT_LISTED),  # Monday
        (MUSEUM, datetime(2026, 12, 15, 11, 0), StatusKind.OUT_OF_SEASON),
    ]
    for hours, at, kind in cases:
        assert hours.status_at(at).kind is kind, at


def test_status_text_points_to_the_next_opening():
    assert BAKERY.status_at(datetime(2026, 10, 23, 16, 0)).text == "open (until 18:30)"
    assert BAKERY.status_at(datetime(2026, 10, 23, 13, 0)).text == (
        "closed at this time (hours that day: 07:45-12:00, 15:00-18:30); next opening same day at 15:00"
    )
    assert BAKERY.status_at(datetime(2026, 10, 24, 20, 0)).text.endswith("next opening Mon 26.10 at 07:45")


def test_on_day():
    assert BAKERY.on_day(date(2026, 10, 26)) == "open 07:45-12:00, 15:00-18:30"
    assert BAKERY.on_day(date(2026, 10, 25)) == "no opening hours listed for Sun (possibly closed)"
    assert MUSEUM.on_day(date(2026, 12, 1)) == "outside the opening season"


def test_season_wrapping_around_new_year():
    winter = OpeningHours.parse(
        "{'period_start': '01.12', 'period_end': '31.03', 'hours': '09:00 - 17:00', 'days': ['sa']}"
    )
    assert winter
    assert winter.status_at(datetime(2027, 1, 2, 10, 0)).kind is StatusKind.OPEN
    assert winter.status_at(datetime(2026, 7, 4, 10, 0)).kind is StatusKind.OUT_OF_SEASON


def test_garbage_input():
    assert OpeningHours.parse("not a dict") is None
    assert OpeningHours.parse("__import__('os').system('true')") is None  # literal_eval, not eval
    assert OpeningHours.parse("{'hours': 'whenever'}") is None
