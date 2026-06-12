"""End-to-end integration test for scripts.s2s.run_issuance with mocked rosetta."""

from datetime import date
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr
import yaml

pytestmark = pytest.mark.integration


# ---- Synthetic data builders shared across tests ----

def _coarse_grid():
    return np.linspace(-5, 5, 8), np.linspace(33, 42, 12)


def _fine_grid():
    return np.linspace(-5, 5, 24), np.linspace(33, 42, 36)


def _synthetic_forecast(issuance: date):
    """Synthetic ECMWF S2S forecast: lead_time × member × lat × lon.

    lead_time is hours from issuance (24, 48, ..., 1104) to match what
    real rosetta produces from cfgrib's `step` dim.
    """
    lat, lon = _coarse_grid()
    members = np.arange(5)
    leads = np.arange(24, 24 * 47, 24, dtype="float64")  # hours: 24..1104 step 24
    rng = np.random.default_rng(int(issuance.toordinal()))
    data = rng.gamma(2.0, 1.0, size=(len(leads), len(members), len(lat), len(lon))).astype("float32")
    return xr.DataArray(
        data,
        dims=["lead_time", "member", "lat", "lon"],
        coords={"lead_time": leads, "member": members, "lat": lat, "lon": lon},
        name="precip",
    )


def _synthetic_reforecast(issuance: date):
    """Synthetic reforecast: year × member × lead_time × lat × lon (hours)."""
    lat, lon = _coarse_grid()
    members = np.arange(5)
    leads = np.arange(24, 24 * 47, 24, dtype="float64")
    years = np.arange(issuance.year - 20, issuance.year)
    rng = np.random.default_rng(int(issuance.toordinal()) + 1)
    data = rng.gamma(2.0, 1.0, size=(len(years), len(members), len(leads), len(lat), len(lon))).astype("float32")
    return xr.DataArray(
        data,
        dims=["year", "member", "lead_time", "lat", "lon"],
        coords={"year": years, "member": members, "lead_time": leads, "lat": lat, "lon": lon},
        name="precip",
    )


def _synthetic_chirps_climatology():
    """Synthetic CHIRPS daily series matching real sheerwater chirps_v2(agg_days=10).

    Shape (time, lat, lon); each timestamp is daily and each value represents
    the 10-day rolling mean ending on that day. _obs_climatology_for_dekad
    picks the target_offset day from each year.
    """
    import pandas as pd
    lat, lon = _fine_grid()
    times = pd.date_range("1991-01-01", "2020-12-31", freq="D")
    rng = np.random.default_rng(0)
    data = rng.gamma(2.0, 1.5, size=(len(times), len(lat), len(lon))).astype("float32")
    return xr.DataArray(
        data,
        dims=["time", "lat", "lon"],
        coords={"time": times, "lat": lat, "lon": lon},
        name="precip",
    )


@pytest.fixture
def s2s_config_path(tmp_path):
    cfg = {
        "countries": {
            "kenya": {
                "bbox": {"min_lat": -5.0, "max_lat": 5.5, "min_lon": 33.5, "max_lon": 42.0},
                "methods": ["raw", "climatology", "bcsd", "rank-analog"],
                "obs": "obs/chirps-v2-dekadal-rhiza",
                "forecast": "c3s/ecmwf-s2s",
                "variable": "precip",
            },
        },
        "lead_days": {"min": 0, "max": 46},
        "climatology_years": [1991, 2020],
        "store_root": str(tmp_path / "issuances"),
    }
    path = tmp_path / "s2s.yml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def _patched_fetch(*args, **kwargs):
    """Stand-in for rosetta.fetch dispatched by (product, reforecast) kwargs."""
    product = kwargs.get("product") or (args[0] if args else None)
    reforecast = kwargs.get("reforecast", False)
    init_raw = kwargs.get("init")
    init = date.fromisoformat(init_raw) if isinstance(init_raw, str) else init_raw

    if product == "c3s/ecmwf-s2s" and not reforecast:
        return _synthetic_forecast(init).to_dataset()
    if product == "c3s/ecmwf-s2s" and reforecast:
        return _synthetic_reforecast(init).to_dataset()
    if product == "obs/chirps-v2-dekadal-rhiza":
        return _synthetic_chirps_climatology().to_dataset()
    raise AssertionError(f"unexpected fetch call: {product=} {reforecast=}")


def test_run_issuance_writes_one_file_per_method_per_dekad(s2s_config_path, tmp_path):
    from scripts.s2s.run_issuance import run_issuance
    issuance = date(2026, 5, 15)
    with patch("scripts.s2s.run_issuance.rosetta_fetch", side_effect=_patched_fetch):
        run_issuance(country="kenya", issuance=issuance, config_path=s2s_config_path)

    store = tmp_path / "issuances" / "kenya" / issuance.isoformat()
    methods_dirs = sorted(p.name for p in store.iterdir())
    assert methods_dirs == ["bcsd", "climatology", "rank-analog", "raw"]
    # Each method has a file per target dekad. For an issuance on 2026-05-15
    # with lead 0–46, that's 5 dekad starts (May 11, May 21, Jun 1, Jun 11, Jun 21).
    for m in methods_dirs:
        assert len(list((store / m).glob("dekad_*.nc"))) == 5


def test_run_issuance_ensemble_methods_carry_tercile_probs(s2s_config_path, tmp_path):
    from scripts.s2s.run_issuance import run_issuance
    issuance = date(2026, 5, 15)
    with patch("scripts.s2s.run_issuance.rosetta_fetch", side_effect=_patched_fetch):
        run_issuance(country="kenya", issuance=issuance, config_path=s2s_config_path)

    bcsd_file = next((tmp_path / "issuances" / "kenya" / issuance.isoformat() / "bcsd").glob("dekad_*.nc"))
    ds = xr.open_dataset(bcsd_file)
    assert set(ds.data_vars) == {"mean", "tercile_probs"}
    assert ds["tercile_probs"].dims == ("category", "lat", "lon")
    # Probabilities sum to ~1 along the category axis.
    sums = ds["tercile_probs"].sum("category")
    assert float(sums.min()) > 0.99
    assert float(sums.max()) < 1.01


def test_run_issuance_climatology_file_has_only_mean(s2s_config_path, tmp_path):
    from scripts.s2s.run_issuance import run_issuance
    issuance = date(2026, 5, 15)
    with patch("scripts.s2s.run_issuance.rosetta_fetch", side_effect=_patched_fetch):
        run_issuance(country="kenya", issuance=issuance, config_path=s2s_config_path)

    clim_file = next((tmp_path / "issuances" / "kenya" / issuance.isoformat() / "climatology").glob("dekad_*.nc"))
    ds = xr.open_dataset(clim_file)
    assert set(ds.data_vars) == {"mean"}


def test_run_issuance_raw_carries_tercile_probs(s2s_config_path, tmp_path):
    """raw is ensemble-producing (10+ members from ECMWF perturbed forecast),
    so it writes tercile_probs alongside mean — Plan C's RPSS needs it to score
    raw as a baseline probabilistic forecast."""
    from scripts.s2s.run_issuance import run_issuance
    issuance = date(2026, 5, 15)
    with patch("scripts.s2s.run_issuance.rosetta_fetch", side_effect=_patched_fetch):
        run_issuance(country="kenya", issuance=issuance, config_path=s2s_config_path)

    raw_file = next((tmp_path / "issuances" / "kenya" / issuance.isoformat() / "raw").glob("dekad_*.nc"))
    ds = xr.open_dataset(raw_file)
    assert set(ds.data_vars) == {"mean", "tercile_probs"}


def test_run_issuance_degrades_gracefully_on_missing_reforecast(s2s_config_path, tmp_path):
    """When the reforecast suite isn't published yet (MarsNoDataError from
    ECDS), the shard should still produce raw + climatology and skip the
    methods that need reforecast training (bcsd, cca, rank-analog)."""
    from scripts.s2s.run_issuance import run_issuance
    issuance = date(2026, 5, 15)

    def _fetch_without_reforecast(*args, **kwargs):
        if kwargs.get("reforecast"):
            # Mirrors the exception bubbled up by cdsapi from ECDS.
            raise RuntimeError(
                "400 Client Error: Bad Request. The job failed with: "
                "MarsNoDataError. MARS returned no data, please check your selection."
            )
        return _patched_fetch(*args, **kwargs)

    with patch("scripts.s2s.run_issuance.rosetta_fetch",
               side_effect=_fetch_without_reforecast):
        run_issuance(country="kenya", issuance=issuance, config_path=s2s_config_path)

    store = tmp_path / "issuances" / "kenya" / issuance.isoformat()
    methods_dirs = sorted(p.name for p in store.iterdir())
    # Only the reforecast-independent methods are written.
    assert methods_dirs == ["climatology", "raw"]
    # Each still gets a file per target dekad (5 dekads for lead 0–46).
    for m in methods_dirs:
        assert len(list((store / m).glob("dekad_*.nc"))) == 5


def test_run_issuance_skips_shard_when_realtime_forecast_embargoed(
    s2s_config_path, tmp_path
):
    """ECDS embargoes very recent S2S forecasts with 'Restricted access to
    S2S data'. With no realtime forecast there's nothing to produce, so
    the shard should exit cleanly without writing anything."""
    from scripts.s2s.run_issuance import run_issuance
    issuance = date(2026, 5, 15)

    def _fetch_with_embargo(*args, **kwargs):
        # First (forecast) call hits the embargo; reforecast call shouldn't
        # even be reached.
        raise RuntimeError(
            "400 Client Error: Bad Request. MarsRuntimeError: AccessError: "
            "Restricted access to S2S data."
        )

    with patch("scripts.s2s.run_issuance.rosetta_fetch",
               side_effect=_fetch_with_embargo):
        # Should not raise — graceful exit.
        run_issuance(country="kenya", issuance=issuance, config_path=s2s_config_path)

    # Nothing written for this issuance.
    assert not (tmp_path / "issuances" / "kenya" / issuance.isoformat()).exists()


def test_run_issuance_reraises_unexpected_reforecast_errors(s2s_config_path, tmp_path):
    """Errors during reforecast fetch that aren't 'no data published' should
    propagate — we don't want to silently swallow auth failures, network
    timeouts, or other genuine bugs."""
    from scripts.s2s.run_issuance import run_issuance
    issuance = date(2026, 5, 15)

    def _fetch_with_real_error(*args, **kwargs):
        if kwargs.get("reforecast"):
            raise RuntimeError("401 Unauthorized: invalid API key")
        return _patched_fetch(*args, **kwargs)

    with patch("scripts.s2s.run_issuance.rosetta_fetch",
               side_effect=_fetch_with_real_error):
        with pytest.raises(RuntimeError, match="401 Unauthorized"):
            run_issuance(country="kenya", issuance=issuance, config_path=s2s_config_path)


def test_run_issuance_outputs_on_obs_grid_with_no_nans(s2s_config_path, tmp_path):
    from scripts.s2s.run_issuance import run_issuance
    issuance = date(2026, 5, 15)
    with patch("scripts.s2s.run_issuance.rosetta_fetch", side_effect=_patched_fetch):
        run_issuance(country="kenya", issuance=issuance, config_path=s2s_config_path)

    fine_lat, fine_lon = _fine_grid()
    for m in ["raw", "climatology", "bcsd", "rank-analog"]:
        f = next((tmp_path / "issuances" / "kenya" / issuance.isoformat() / m).glob("dekad_*.nc"))
        ds = xr.open_dataset(f)
        assert len(ds["lat"]) == len(fine_lat)
        assert len(ds["lon"]) == len(fine_lon)
        assert not np.any(np.isnan(ds["mean"].values))
