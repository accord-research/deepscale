"""Dekad math.

A dekad is a 10-day calendar slice: day 1–10, day 11–20, day 21–end-of-month.
The third dekad's length is variable (8–11 days). We identify a dekad by its
start date.
"""

from __future__ import annotations

from calendar import monthrange
from datetime import date, timedelta


def _dekad_start_for(d: date) -> date:
    """Return the start date of the dekad containing d."""
    if d.day <= 10:
        return d.replace(day=1)
    if d.day <= 20:
        return d.replace(day=11)
    return d.replace(day=21)


def dekad_window(start: date) -> tuple[date, date]:
    """Return [start, end) for the dekad beginning at `start`.

    end is exclusive — it's the first day of the *next* dekad.
    """
    if start.day == 1:
        return start, start.replace(day=11)
    if start.day == 11:
        return start, start.replace(day=21)
    # Third dekad: end = first of next month.
    days_in_month = monthrange(start.year, start.month)[1]
    next_month_first = (start.replace(day=days_in_month) + timedelta(days=1))
    return start, next_month_first


def _next_dekad_start(start: date) -> date:
    """Given a dekad start, return the next dekad's start."""
    _, end = dekad_window(start)
    return end


def dekads_for_issuance(issuance: date, lead_days: tuple[int, int]) -> list[date]:
    """Return the list of dekad start dates covered by an issuance.

    The window is [issuance + lead_min, issuance + lead_max] (both inclusive).
    A dekad is included if its start date falls inside that window.
    """
    lead_min, lead_max = lead_days
    window_start = issuance + timedelta(days=lead_min)
    window_end = issuance + timedelta(days=lead_max)

    cur = _dekad_start_for(window_start)
    out: list[date] = []
    while cur <= window_end:
        out.append(cur)
        cur = _next_dekad_start(cur)
    return out
