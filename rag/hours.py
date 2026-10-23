"""Opening hours: parse the export format, render it for humans, evaluate it in code.

The exports contain opening hours as a comma-separated list of Python dict literals:

    {'period_start': '01.01', 'period_end': '31.12', 'service_type': 'General',
     'hours': '06:30 - 19:00', 'days': ['mo', 'tu', 'we', 'th', 'fr']}, {...}

Embedding that raw string is useless (it is mostly punctuation) and asking an LLM to
work out "is it open right now?" from it is unreliable. So we parse it once, render a
compact human-readable form for the chunk text, and compute open/closed status in code
at query time.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

DAY_CODES = ["mo", "tu", "we", "th", "fr", "sa", "su"]
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


@dataclass(frozen=True)
class HoursRule:
    """One opening interval, valid on some weekdays within a yearly date period."""

    period_start: tuple[int, int]  # (day, month)
    period_end: tuple[int, int]
    opens: time
    closes: time
    days: tuple[int, ...]  # 0 = Monday

    def in_period(self, d: date) -> bool:
        start = (self.period_start[1], self.period_start[0])  # compare as (month, day)
        end = (self.period_end[1], self.period_end[0])
        cur = (d.month, d.day)
        if start <= end:
            return start <= cur <= end
        return cur >= start or cur <= end  # period wraps around new year, e.g. 01.11 - 30.04

    def applies(self, d: date) -> bool:
        return d.weekday() in self.days and self.in_period(d)

    def to_dict(self) -> dict:
        return {
            "period_start": list(self.period_start),
            "period_end": list(self.period_end),
            "opens": self.opens.strftime("%H:%M"),
            "closes": self.closes.strftime("%H:%M"),
            "days": list(self.days),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HoursRule":
        return cls(
            period_start=tuple(d["period_start"]),
            period_end=tuple(d["period_end"]),
            opens=time.fromisoformat(d["opens"]),
            closes=time.fromisoformat(d["closes"]),
            days=tuple(d["days"]),
        )


def _parse_day_month(s: str) -> tuple[int, int]:
    day, month = s.strip().split(".")
    return int(day), int(month)


def parse_hours(raw: str) -> list[HoursRule]:
    """Parse the raw export string. Returns [] if it cannot be parsed (never raises)."""
    try:
        entries = ast.literal_eval(f"[{raw.strip()}]")  # safe: literals only, no code execution
    except (ValueError, SyntaxError):
        return []
    rules = []
    for e in entries:
        try:
            opens, closes = (time.fromisoformat(t.strip()) for t in e["hours"].split("-"))
            rules.append(
                HoursRule(
                    period_start=_parse_day_month(e["period_start"]),
                    period_end=_parse_day_month(e["period_end"]),
                    opens=opens,
                    closes=closes,
                    days=tuple(DAY_CODES.index(day) for day in e["days"]),
                )
            )
        except (KeyError, ValueError, AttributeError):
            continue  # skip a malformed entry rather than dropping the whole record
    return rules


def _day_ranges(days: list[int]) -> str:
    """[0,1,2,3,4] -> 'Mon-Fri', [0,2,5] -> 'Mon, Wed, Sat'."""
    days = sorted(days)
    groups: list[list[int]] = []
    for d in days:
        if groups and d == groups[-1][-1] + 1:
            groups[-1].append(d)
        else:
            groups.append([d])
    parts = []
    for g in groups:
        if len(g) >= 3:
            parts.append(f"{DAY_NAMES[g[0]]}-{DAY_NAMES[g[-1]]}")
        else:
            parts.extend(DAY_NAMES[d] for d in g)
    return ", ".join(parts)


def format_hours(rules: list[HoursRule]) -> str:
    """Human-readable summary, e.g. 'Mon-Fri 07:45-12:00, 15:00-18:30; Sat 07:45-12:00; closed Sun'.

    Days that share the same set of intervals are grouped together. The validity period
    is only mentioned when it is not the whole year (e.g. seasonal museums).
    """
    if not rules:
        return ""
    by_period: dict[tuple, list[HoursRule]] = {}
    for r in rules:
        by_period.setdefault((r.period_start, r.period_end), []).append(r)

    lines = []
    for (start, end), period_rules in by_period.items():
        intervals_per_day: dict[int, list[tuple[time, time]]] = {}
        for r in period_rules:
            for d in r.days:
                intervals_per_day.setdefault(d, []).append((r.opens, r.closes))
        # group days with identical intervals
        by_intervals: dict[tuple, list[int]] = {}
        for d, ivs in intervals_per_day.items():
            by_intervals.setdefault(tuple(sorted(ivs)), []).append(d)
        parts = []
        for ivs, days in sorted(by_intervals.items(), key=lambda kv: min(kv[1])):
            times = ", ".join(f"{o:%H:%M}-{c:%H:%M}" for o, c in ivs)
            parts.append(f"{_day_ranges(days)} {times}")
        closed = [d for d in range(7) if d not in intervals_per_day]
        if closed:
            parts.append(f"closed {_day_ranges(closed)}")
        text = "; ".join(parts)
        if (start, end) != ((1, 1), (31, 12)):
            text += f" (from {start[0]:02d}.{start[1]:02d} to {end[0]:02d}.{end[1]:02d})"
        lines.append(text)
    return " | ".join(lines)


def is_open(rules: list[HoursRule], at: datetime) -> bool:
    return any(r.applies(at.date()) and r.opens <= at.time() < r.closes for r in rules)


def status_at(rules: list[HoursRule], at: datetime) -> str:
    """One-line status relative to `at`, e.g. 'closed now; opens Sat 06:30'."""
    if not rules:
        return ""
    for r in rules:
        if r.applies(at.date()) and r.opens <= at.time() < r.closes:
            return f"open now (until {r.closes:%H:%M})"
    # next opening within the coming two weeks
    for offset in range(15):
        d = at.date() + timedelta(days=offset)
        starts = sorted(
            r.opens for r in rules if r.applies(d) and (offset > 0 or r.opens > at.time())
        )
        if starts:
            when = "today" if offset == 0 else ("tomorrow" if offset == 1 else DAY_NAMES[d.weekday()] + f" {d:%d.%m}")
            return f"closed now; opens {when} at {starts[0]:%H:%M}"
    return "closed now; no opening in the next two weeks (outside the season)"
