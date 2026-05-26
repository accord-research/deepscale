"""Unit tests for scripts.s2s.dekads.dekads_for_issuance and dekad_window."""

from datetime import date

import pytest


def test_dekads_for_issuance_mid_dekad_start():
    """Issued 2026-05-15 (mid-dekad-2 of May), lead 0–46 → covers May-d2 through end-of-June."""
    from scripts.s2s.dekads import dekads_for_issuance
    result = dekads_for_issuance(date(2026, 5, 15), lead_days=(0, 46))
    # First dekad: the one containing the issuance day (May 11–20 → start May 11).
    assert result[0] == date(2026, 5, 11)
    # Last dekad: the one whose start is <= issuance + 46 days = 2026-06-30 → start Jun 21.
    assert result[-1] == date(2026, 6, 21)
    # Five dekads inclusive: May 11, May 21, Jun 1, Jun 11, Jun 21.
    assert result == [
        date(2026, 5, 11), date(2026, 5, 21),
        date(2026, 6, 1),  date(2026, 6, 11), date(2026, 6, 21),
    ]


def test_dekads_for_issuance_at_dekad_boundary():
    """Issued on the 1st of a month maps to dekad-1 starting on the issuance day."""
    from scripts.s2s.dekads import dekads_for_issuance
    result = dekads_for_issuance(date(2026, 5, 1), lead_days=(0, 30))
    assert result[0] == date(2026, 5, 1)


def test_dekad_window_returns_inclusive_start_exclusive_end():
    """A May-dekad-1 start (May 1) returns (May 1, May 11)."""
    from scripts.s2s.dekads import dekad_window
    start, end = dekad_window(date(2026, 5, 1))
    assert start == date(2026, 5, 1)
    assert end == date(2026, 5, 11)


def test_dekad_window_handles_third_dekad_variable_length():
    """A May-dekad-3 start (May 21) returns (May 21, Jun 1) — May has 31 days so dekad-3 is 11 days."""
    from scripts.s2s.dekads import dekad_window
    start, end = dekad_window(date(2026, 5, 21))
    assert start == date(2026, 5, 21)
    assert end == date(2026, 6, 1)


def test_dekad_window_february_third_dekad_non_leap():
    """Feb 21 in a non-leap year returns (Feb 21, Mar 1) — 8 days long."""
    from scripts.s2s.dekads import dekad_window
    start, end = dekad_window(date(2026, 2, 21))
    assert start == date(2026, 2, 21)
    assert end == date(2026, 3, 1)
