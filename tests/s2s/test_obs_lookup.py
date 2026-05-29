"""Unit tests for the S2S obs-dekad lookup helpers in scripts.s2s.verify.

These pin down three bugs found 2026-05-29 that left the dashboard with zero
verification scores (the finalized 2026 obs cache was empty, so every pair fell
through to a broken fallback):

  B. ``_obs_at_dekad_rolling`` / ``_obs_climatology_for_dekad`` queried
     ``target + 9``, but sheerwater's ``roll_and_agg`` LEFT-aligns the 10-day
     window (label T = mean of [T, T+9]), so the correct label is ``target``.
  C. ``_obs_at_dekad_daily`` built ``have`` as a set of ``datetime.date``
     (``datetime64[D].tolist()``) but tested membership with ``np.datetime64``
     scalars — always False, so it always returned None.

Pure-logic, no network — these run in CI (no ``integration`` marker).
"""

from datetime import date

import numpy as np
import pandas as pd
import xarray as xr


def _doy_obs(start: str, end: str, nlat: int = 3, nlon: int = 4) -> xr.DataArray:
    """Daily (time, lat, lon) obs whose value at every cell on day T is T's
    day-of-year. Lets a test identify *which* timestamp a lookup returned."""
    times = pd.date_range(start, end, freq="D")
    doy = times.dayofyear.to_numpy().astype("float32")
    data = np.broadcast_to(doy[:, None, None], (len(times), nlat, nlon)).astype("float32")
    return xr.DataArray(
        data,
        dims=["time", "lat", "lon"],
        coords={"time": times, "lat": np.linspace(-2, 2, nlat), "lon": np.linspace(33, 36, nlon)},
        name="precip",
    )


def test_obs_at_dekad_rolling_uses_left_aligned_label():
    """The dekad starting at ``target`` maps to the rolling label ``target``
    (left-aligned), NOT ``target + 9``."""
    from scripts.s2s.verify import _obs_at_dekad_rolling

    obs = _doy_obs("2026-01-01", "2026-12-31")
    target = date(2026, 3, 2)  # 2026 is non-leap → doy 61, target+9 → doy 70
    field = _obs_at_dekad_rolling(obs, target)

    assert field is not None
    assert float(field.values.flat[0]) == target.timetuple().tm_yday  # 61, not 70


def test_obs_at_dekad_daily_returns_window_mean_not_none():
    """The daily fallback must actually find the [target, target+9] window and
    return its mean — regression for the datetime64-vs-date membership bug."""
    from scripts.s2s.verify import _obs_at_dekad_daily

    obs = _doy_obs("2026-01-01", "2026-12-31")
    target = date(2026, 3, 2)
    field = _obs_at_dekad_daily(obs, target)

    assert field is not None  # was always None due to the type-mismatch bug
    base = target.timetuple().tm_yday
    expected = float(np.mean([base + i for i in range(10)]))  # mean(61..70) = 65.5
    assert abs(float(field.values.flat[0]) - expected) < 1e-4


def test_obs_at_dekad_daily_none_when_window_incomplete():
    """If the 10-day window runs past the available data, return None (pending)."""
    from scripts.s2s.verify import _obs_at_dekad_daily

    obs = _doy_obs("2026-01-01", "2026-03-05")  # ends mid-window for a Mar 2 target
    assert _obs_at_dekad_daily(obs, date(2026, 3, 2)) is None


def test_obs_climatology_for_dekad_selects_target_doy():
    """Climatology samples the rolling label at ``target``'s day-of-year across
    years, not ``target + 9``'s."""
    from scripts.s2s.verify import _obs_climatology_for_dekad

    obs = _doy_obs("1991-01-01", "2021-12-31")
    target = date(2010, 2, 1)  # doy 32 (leap-safe); target+9 → Feb 10 → doy 41
    clim = _obs_climatology_for_dekad(obs, target, (1991, 2020))

    assert "year" in clim.dims
    vals = np.unique(np.round(clim.values[np.isfinite(clim.values)]).astype(int))
    assert set(vals.tolist()) == {target.timetuple().tm_yday}  # {32}, not {41}
