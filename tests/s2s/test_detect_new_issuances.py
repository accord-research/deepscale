"""Unit tests for scripts.s2s.detect_new_issuances."""

from datetime import date


def test_issuances_in_window_returns_only_mon_and_thu():
    """Within a 7-day window, only Mondays and Thursdays are valid issuance dates."""
    from scripts.s2s.detect_new_issuances import issuances_in_window
    # 2026-05-11 = Monday; 14 = Thursday; 18 = Monday; 21 = Thursday.
    out = issuances_in_window(date(2026, 5, 11), date(2026, 5, 21))
    assert out == [date(2026, 5, 11), date(2026, 5, 14), date(2026, 5, 18), date(2026, 5, 21)]


def test_issuances_in_window_excludes_dates_outside_inclusive_window():
    from scripts.s2s.detect_new_issuances import issuances_in_window
    out = issuances_in_window(date(2026, 5, 12), date(2026, 5, 17))
    assert out == [date(2026, 5, 14)]


def test_new_since_filters_out_already_present(tmp_path):
    """new_since returns issuances after `since` that are NOT already on disk."""
    from datetime import date
    from scripts.s2s.detect_new_issuances import new_since
    store = tmp_path / "issuances"
    (store / "kenya" / "2026-05-14").mkdir(parents=True)  # already fetched
    out = new_since(
        store_root=store, country="kenya",
        since=date(2026, 5, 11), until=date(2026, 5, 21),
    )
    assert date(2026, 5, 14) not in out
    assert date(2026, 5, 11) in out  # before since=11? no — since-inclusive
    assert out == [date(2026, 5, 11), date(2026, 5, 18), date(2026, 5, 21)]
