"""Opening hours: parse the export format, describe it, and evaluate it at a given moment.

The exports store opening hours as a comma-separated list of Python dict literals:

    {'period_start': '01.01', 'period_end': '31.12', 'service_type': 'General',
     'hours': '06:30 - 19:00', 'days': ['mo', 'tu', 'we', 'th', 'fr']}, {...}

Embedding that string is useless and LLMs are unreliable at weekday and date arithmetic,
so the string is parsed once and every "is it open?" question is answered here, in code.

One semantic point matters: the exports list the days a place *is* open. A weekday that
does not appear is "not listed", which usually means closed but may just be missing data.
The status therefore distinguishes ``CLOSED`` (the day has hours, but not at this time)
from ``NOT_LISTED`` (no hours for that day) and ``OUT_OF_SEASON``.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from enum import StrEnum

DAY_CODES = ("mo", "tu", "we", "th", "fr", "sa", "su")
DAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
WHOLE_YEAR = ((1, 1), (31, 12))
LOOKAHEAD_DAYS = 14

DayMonth = tuple[int, int]  # (day, month)


@dataclass(frozen=True)
class HoursRule:
    """One opening interval on some weekdays, valid every year between two dates."""

    valid_from: DayMonth
    valid_to: DayMonth
    opens: time
    closes: time
    weekdays: frozenset[int]  # 0 = Monday

    def in_season(self, d: date) -> bool:
        start, end, cur = self.valid_from[::-1], self.valid_to[::-1], (d.month, d.day)
        if start <= end:
            return start <= cur <= end
        return cur >= start or cur <= end  # season wraps around new year, e.g. 01.12 - 31.03

    def applies_on(self, d: date) -> bool:
        return d.weekday() in self.weekdays and self.in_season(d)


class StatusKind(StrEnum):
    OPEN = "open"
    CLOSED = "closed"  # the day has opening hours, but not at this time
    NOT_LISTED = "not_listed"  # no opening hours listed for this weekday
    OUT_OF_SEASON = "out_of_season"


@dataclass(frozen=True)
class HoursStatus:
    kind: StatusKind
    text: str


@dataclass(frozen=True)
class OpeningHours:
    rules: tuple[HoursRule, ...]

    @classmethod
    def parse(cls, raw: str) -> OpeningHours | None:
        """Parse the export string; malformed entries are skipped, nothing parsable -> None."""
        try:
            entries = ast.literal_eval(f"[{raw.strip()}]")  # literals only: no code execution
        except (ValueError, SyntaxError):
            return None
        rules = []
        for entry in entries:
            try:
                opens, closes = (time.fromisoformat(t.strip()) for t in entry["hours"].split("-"))
                rules.append(
                    HoursRule(
                        valid_from=_day_month(entry["period_start"]),
                        valid_to=_day_month(entry["period_end"]),
                        opens=opens,
                        closes=closes,
                        weekdays=frozenset(DAY_CODES.index(d) for d in entry["days"]),
                    )
                )
            except (KeyError, TypeError, ValueError, AttributeError):
                continue
        return cls(tuple(rules)) if rules else None

    # ------------------------------------------------------------------ description

    def describe(self) -> str:
        """E.g. 'Mon-Fri 07:45-12:00, 15:00-18:30; Sat 07:45-12:00; other days not listed'."""
        by_season: dict[tuple[DayMonth, DayMonth], list[HoursRule]] = defaultdict(list)
        for rule in self.rules:
            by_season[(rule.valid_from, rule.valid_to)].append(rule)

        parts = []
        for (start, end), rules in by_season.items():
            intervals: dict[int, list[tuple[time, time]]] = defaultdict(list)
            for rule in rules:
                for day in rule.weekdays:
                    intervals[day].append((rule.opens, rule.closes))
            days_by_intervals: dict[tuple[tuple[time, time], ...], list[int]] = defaultdict(list)
            for day, ivs in intervals.items():
                days_by_intervals[tuple(sorted(ivs))].append(day)

            groups = sorted(days_by_intervals.items(), key=lambda kv: min(kv[1]))
            text = "; ".join(
                f"{_day_ranges(days)} " + ", ".join(f"{o:%H:%M}-{c:%H:%M}" for o, c in ivs) for ivs, days in groups
            )
            if len(intervals) < 7:
                text += "; other days not listed"
            if (start, end) != WHOLE_YEAR:
                text += f" (season {start[0]:02d}.{start[1]:02d} - {end[0]:02d}.{end[1]:02d})"
            parts.append(text)
        return " | ".join(parts)

    # ------------------------------------------------------------------ evaluation

    def status_at(self, at: datetime) -> HoursStatus:
        today = [r for r in self.rules if r.applies_on(at.date())]
        for rule in today:
            if rule.opens <= at.time() < rule.closes:
                return HoursStatus(StatusKind.OPEN, f"open (until {rule.closes:%H:%M})")

        next_opening = self._next_opening(at)
        then = f"; next opening {next_opening}" if next_opening else ""
        if today:
            hours = ", ".join(f"{r.opens:%H:%M}-{r.closes:%H:%M}" for r in sorted(today, key=lambda r: r.opens))
            return HoursStatus(StatusKind.CLOSED, f"closed at this time (hours that day: {hours}){then}")
        if not any(r.in_season(at.date()) for r in self.rules):
            return HoursStatus(StatusKind.OUT_OF_SEASON, f"outside the opening season{then}")
        day = DAY_NAMES[at.weekday()]
        return HoursStatus(StatusKind.NOT_LISTED, f"no opening hours listed for {day} (possibly closed){then}")

    def on_day(self, d: date) -> str:
        """Hours for a whole day, for questions like 'is it open on Sunday?'."""
        rules = sorted((r for r in self.rules if r.applies_on(d)), key=lambda r: r.opens)
        if rules:
            return "open " + ", ".join(f"{r.opens:%H:%M}-{r.closes:%H:%M}" for r in rules)
        if not any(r.in_season(d) for r in self.rules):
            return "outside the opening season"
        return f"no opening hours listed for {DAY_NAMES[d.weekday()]} (possibly closed)"

    def _next_opening(self, at: datetime) -> str | None:
        for offset in range(LOOKAHEAD_DAYS + 1):
            d = at.date() + timedelta(days=offset)
            starts = sorted(r.opens for r in self.rules if r.applies_on(d) and (offset > 0 or r.opens > at.time()))
            if starts:
                when = "same day" if offset == 0 else f"{DAY_NAMES[d.weekday()]} {d:%d.%m}"
                return f"{when} at {starts[0]:%H:%M}"
        return None


def _day_month(s: str) -> DayMonth:
    day, month = s.strip().split(".")
    return int(day), int(month)


def _day_ranges(days: list[int]) -> str:
    """[0, 1, 2, 3, 4] -> 'Mon-Fri'; [0, 2, 5] -> 'Mon, Wed, Sat'."""
    groups: list[list[int]] = []
    for d in sorted(days):
        if groups and d == groups[-1][-1] + 1:
            groups[-1].append(d)
        else:
            groups.append([d])
    names: list[str] = []
    for g in groups:
        if len(g) >= 3:
            names.append(f"{DAY_NAMES[g[0]]}-{DAY_NAMES[g[-1]]}")
        else:
            names.extend(DAY_NAMES[d] for d in g)
    return ", ".join(names)
