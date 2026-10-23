from datetime import datetime

from rag.hours import format_hours, is_open, parse_hours, status_at

BAKERY = (
    "{'period_start': '01.01', 'period_end': '31.12', 'service_type': 'General', 'hours': '07:45 - 12:00', 'days': ['mo', 'tu', 'we', 'th', 'fr']}, "
    "{'period_start': '01.01', 'period_end': '31.12', 'service_type': 'General', 'hours': '07:45 - 12:00', 'days': ['sa']}, "
    "{'period_start': '01.01', 'period_end': '31.12', 'service_type': 'General', 'hours': '15:00 - 18:30', 'days': ['mo', 'tu', 'we', 'th', 'fr']}"
)
MUSEUM = "{'period_start': '01.05', 'period_end': '05.11', 'service_type': 'General', 'hours': '10:00 - 18:00', 'days': ['tu', 'we', 'th', 'fr', 'sa', 'su']}"


def test_format_groups_days_with_identical_intervals():
    assert format_hours(parse_hours(BAKERY)) == "Mon-Fri 07:45-12:00, 15:00-18:30; Sat 07:45-12:00; closed Sun"


def test_format_mentions_seasonal_period():
    assert format_hours(parse_hours(MUSEUM)) == "Tue-Sun 10:00-18:00; closed Mon (from 01.05 to 05.11)"


def test_is_open_respects_lunch_break_weekday_and_season():
    bakery, museum = parse_hours(BAKERY), parse_hours(MUSEUM)
    assert is_open(bakery, datetime(2026, 10, 23, 9, 0))  # Friday morning
    assert not is_open(bakery, datetime(2026, 10, 23, 13, 0))  # lunch break
    assert not is_open(bakery, datetime(2026, 10, 25, 9, 0))  # Sunday
    assert is_open(museum, datetime(2026, 7, 14, 11, 0))  # in season, Tuesday
    assert not is_open(museum, datetime(2026, 7, 13, 11, 0))  # Monday
    assert not is_open(museum, datetime(2026, 12, 15, 11, 0))  # off season


def test_status_points_to_next_opening():
    bakery = parse_hours(BAKERY)
    assert status_at(bakery, datetime(2026, 10, 23, 13, 0)) == "closed now; opens today at 15:00"
    assert status_at(bakery, datetime(2026, 10, 24, 20, 0)) == "closed now; opens Mon 26.10 at 07:45"
    assert status_at(bakery, datetime(2026, 10, 23, 16, 0)) == "open now (until 18:30)"


def test_wrapping_period_and_garbage_input():
    winter = parse_hours("{'period_start': '01.12', 'period_end': '31.03', 'hours': '09:00 - 17:00', 'days': ['sa']}")
    assert is_open(winter, datetime(2027, 1, 2, 10, 0))
    assert not is_open(winter, datetime(2026, 7, 4, 10, 0))
    assert parse_hours("not a dict") == []
