"""End-to-end integration test for scripts.s2s.verify."""

import json
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

pytestmark = pytest.mark.integration


def _fine_grid():
    return np.linspace(-5, 5, 24), np.linspace(33, 42, 36)


def _make_obs_daily():
    """Sheerwater-shape obs: (time, lat, lon) with daily timestamps.

    Range covers 1991..2027 so the climatology window and the 2026 target
    fall inside the same array.
    """
    lat, lon = _fine_grid()
    times = pd.date_range("1991-01-01", "2027-12-31", freq="D")
    rng = np.random.default_rng(1)
    data = rng.gamma(2.0, 1.5, size=(len(times), len(lat), len(lon))).astype("float32")
    return xr.DataArray(
        data, dims=["time", "lat", "lon"],
        coords={"time": times, "lat": lat, "lon": lon},
        name="precip",
    )


def _write_synthetic_store(store_root: Path, country: str, issuance: date, target: date):
    """Write a handful of method outputs into the store layout."""
    from scripts.s2s.issuance_store import write_issuance
    lat, lon = _fine_grid()
    rng = np.random.default_rng(42)

    for method, with_probs in [("raw", True), ("climatology", False), ("bcsd", True)]:
        mean = xr.DataArray(
            rng.gamma(2, 1.5, (len(lat), len(lon))).astype("float32"),
            dims=["lat", "lon"], coords={"lat": lat, "lon": lon},
        )
        vars_ = {"mean": mean}
        if with_probs:
            probs = rng.dirichlet([1, 1, 1], (len(lat), len(lon))).transpose(2, 0, 1).astype("float32")
            vars_["tercile_probs"] = xr.DataArray(
                probs, dims=["category", "lat", "lon"],
                coords={"category": ["below", "normal", "above"], "lat": lat, "lon": lon},
            )
        write_issuance(store_root, country, issuance, method, target, xr.Dataset(vars_))


def _patched_fetch(*args, **kwargs):
    product = kwargs.get("product") or (args[0] if args else None)
    if product == "obs/chirps-dekadal":
        return _make_obs_daily().to_dataset()
    raise AssertionError(f"unexpected fetch call: {product=}")


@pytest.fixture
def s2s_cfg(tmp_path):
    """Build an S2SConfig pointing the store_root at tmp_path."""
    from scripts.s2s.config import load_config
    cfg_dict = {
        "countries": {
            "kenya": {
                "bbox": {"min_lat": -5.0, "max_lat": 5.0, "min_lon": 33.0, "max_lon": 42.0},
                "methods": ["raw", "climatology", "bcsd"],
                "obs": "obs/chirps-dekadal",
                "forecast": "c3s/ecmwf-s2s",
                "variable": "precip",
            },
        },
        "lead_days": {"min": 0, "max": 46},
        "climatology_years": [1991, 2020],
        "store_root": str(tmp_path / "issuances"),
    }
    path = tmp_path / "s2s.yml"
    path.write_text(yaml.safe_dump(cfg_dict))
    return load_config(path)


def test_verify_writes_records_for_pending_pairs(s2s_cfg, tmp_path):
    from scripts.s2s.verify import verify
    store = tmp_path / "issuances"
    verif = tmp_path / "verification"
    issuance = date(2026, 5, 15)
    target = date(2026, 5, 21)
    _write_synthetic_store(store, "kenya", issuance, target)

    with patch("scripts.s2s.verify.rosetta_fetch", side_effect=_patched_fetch):
        verify(store_root=store, verification_root=verif, cfg=s2s_cfg)

    lines = (verif / "kenya" / "scores.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in lines]
    methods_seen = sorted({r["method"] for r in records})
    assert methods_seen == ["bcsd", "climatology", "raw"]


def test_verify_idempotent_on_second_run(s2s_cfg, tmp_path):
    from scripts.s2s.verify import verify
    store = tmp_path / "issuances"
    verif = tmp_path / "verification"
    issuance = date(2026, 5, 15)
    target = date(2026, 5, 21)
    _write_synthetic_store(store, "kenya", issuance, target)

    with patch("scripts.s2s.verify.rosetta_fetch", side_effect=_patched_fetch):
        verify(store_root=store, verification_root=verif, cfg=s2s_cfg)
        first = (verif / "kenya" / "scores.jsonl").read_text().splitlines()
        verify(store_root=store, verification_root=verif, cfg=s2s_cfg)
        second = (verif / "kenya" / "scores.jsonl").read_text().splitlines()

    assert len(first) == len(second)


def test_verify_rpss_only_when_tercile_probs_present(s2s_cfg, tmp_path):
    from scripts.s2s.verify import verify
    store = tmp_path / "issuances"
    verif = tmp_path / "verification"
    issuance = date(2026, 5, 15)
    target = date(2026, 5, 21)
    _write_synthetic_store(store, "kenya", issuance, target)

    with patch("scripts.s2s.verify.rosetta_fetch", side_effect=_patched_fetch):
        verify(store_root=store, verification_root=verif, cfg=s2s_cfg)

    records = [json.loads(line) for line in (verif / "kenya" / "scores.jsonl").read_text().splitlines()]
    by_method = {r["method"]: r for r in records}
    assert "rpss" in by_method["bcsd"]
    assert "rpss" in by_method["raw"]
    assert "rpss" not in by_method["climatology"]


def _make_obs_clim_only():
    """Finalized rolled obs covering ONLY the climatology window (1991-2020).

    The recent (2026) entries are absent — mimicking the stale/empty current-year
    sheerwater cache that produced zero scores. Forces verify onto the daily feed.
    """
    lat, lon = _fine_grid()
    times = pd.date_range("1991-01-01", "2020-12-31", freq="D")
    rng = np.random.default_rng(7)
    data = rng.gamma(2.0, 1.5, size=(len(times), len(lat), len(lon))).astype("float32")
    return xr.DataArray(
        data, dims=["time", "lat", "lon"],
        coords={"time": times, "lat": lat, "lon": lon}, name="precip",
    )


def _make_obs_live_fine():
    """2026 daily live feed on a FINER grid than the climatology — exercises the
    regrid-onto-climatology-grid step (a missing regrid would crash score_pair)."""
    lat = np.linspace(-5, 5, 48)
    lon = np.linspace(33, 42, 72)
    times = pd.date_range("2026-01-01", "2026-06-30", freq="D")
    rng = np.random.default_rng(8)
    data = rng.gamma(2.0, 1.5, size=(len(times), len(lat), len(lon))).astype("float32")
    return xr.DataArray(
        data, dims=["time", "lat", "lon"],
        coords={"time": times, "lat": lat, "lon": lon}, name="precip",
    )


def _patched_fetch_with_live(*args, **kwargs):
    product = kwargs.get("product") or (args[0] if args else None)
    if product == "obs/chirps-dekadal":
        return _make_obs_clim_only().to_dataset()
    if product == "obs/chirps-live":
        return _make_obs_live_fine().to_dataset()
    raise AssertionError(f"unexpected fetch call: {product=}")


@pytest.fixture
def s2s_cfg_with_live(tmp_path):
    """Config that, unlike s2s_cfg, declares obs_live so the daily fallback runs."""
    from scripts.s2s.config import load_config
    cfg_dict = {
        "countries": {
            "kenya": {
                "bbox": {"min_lat": -5.0, "max_lat": 5.0, "min_lon": 33.0, "max_lon": 42.0},
                "methods": ["raw", "climatology", "bcsd"],
                "obs": "obs/chirps-dekadal",
                "obs_live": "obs/chirps-live",
                "forecast": "c3s/ecmwf-s2s",
                "variable": "precip",
            },
        },
        "lead_days": {"min": 0, "max": 46},
        "climatology_years": [1991, 2020],
        "store_root": str(tmp_path / "issuances"),
    }
    path = tmp_path / "s2s.yml"
    path.write_text(yaml.safe_dump(cfg_dict))
    return load_config(path)


def test_verify_scores_via_daily_fallback_when_finalized_obs_missing(s2s_cfg_with_live, tmp_path):
    """The 2026 regression: finalized rolled obs has no entry for the recent
    target (empty cache), so verify falls back to the daily live feed, regrids it
    onto the climatology grid, and still writes a full set of scores.

    Pre-fix this wrote ZERO records (dead daily fallback + grid mismatch)."""
    from scripts.s2s.verify import verify
    store = tmp_path / "issuances"
    verif = tmp_path / "verification"
    issuance = date(2026, 5, 15)
    target = date(2026, 5, 21)
    _write_synthetic_store(store, "kenya", issuance, target)

    with patch("scripts.s2s.verify.rosetta_fetch", side_effect=_patched_fetch_with_live):
        verify(store_root=store, verification_root=verif, cfg=s2s_cfg_with_live)

    records = [json.loads(line) for line in (verif / "kenya" / "scores.jsonl").read_text().splitlines()]
    methods_seen = sorted({r["method"] for r in records})
    assert methods_seen == ["bcsd", "climatology", "raw"]
    # Obs came off a 48x72 grid; without the regrid-onto-clim-grid step
    # score_pair would have raised on the shape mismatch.
    assert all("acc" in r and "rmse" in r for r in records)
