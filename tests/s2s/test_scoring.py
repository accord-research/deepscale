"""Unit tests for scripts.s2s.scoring."""

from datetime import date

import numpy as np
import pytest
import xarray as xr


def _make_pred_obs(seed: int = 0, with_tercile_probs: bool = True):
    rng = np.random.default_rng(seed)
    lat = np.linspace(-2, 2, 6)
    lon = np.linspace(30, 35, 8)

    # Build a per-cell obs climatology so RPSS has a comparator.
    years = np.arange(1991, 2021)
    obs_clim = xr.DataArray(
        rng.gamma(2.0, 1.0, size=(len(years), len(lat), len(lon))).astype("float32"),
        dims=["year", "lat", "lon"],
        coords={"year": years, "lat": lat, "lon": lon},
    )
    obs = xr.DataArray(
        rng.gamma(2.0, 1.0, size=(len(lat), len(lon))).astype("float32"),
        dims=["lat", "lon"],
        coords={"lat": lat, "lon": lon},
    )

    mean = obs + rng.normal(0, 0.5, size=obs.shape).astype("float32")
    ds_vars = {"mean": mean}
    if with_tercile_probs:
        probs = rng.dirichlet([1, 1, 1], size=(len(lat), len(lon))).transpose(2, 0, 1).astype("float32")
        ds_vars["tercile_probs"] = xr.DataArray(
            probs,
            dims=["category", "lat", "lon"],
            coords={"category": ["below", "normal", "above"], "lat": lat, "lon": lon},
        )
    pred = xr.Dataset(ds_vars)
    return pred, obs, obs_clim


def test_score_pair_deterministic_keys_present():
    """ACC, RMSE, bias keys always present when pred has a mean field."""
    from scripts.s2s.scoring import score_pair
    pred, obs, clim = _make_pred_obs(with_tercile_probs=False)
    record = score_pair(pred, obs, clim)
    assert set(["acc", "rmse", "bias"]).issubset(record.keys())
    assert "rpss" not in record


def test_score_pair_probabilistic_key_present_when_probs_carried():
    """RPSS is added when pred carries tercile_probs."""
    from scripts.s2s.scoring import score_pair
    pred, obs, clim = _make_pred_obs(with_tercile_probs=True)
    record = score_pair(pred, obs, clim)
    assert "rpss" in record


def test_score_pair_perfect_forecast_gets_near_perfect_acc():
    """When pred.mean == obs, ACC ≈ 1."""
    from scripts.s2s.scoring import score_pair
    pred, obs, clim = _make_pred_obs(with_tercile_probs=False)
    pred = pred.assign(mean=obs)
    record = score_pair(pred, obs, clim)
    assert record["acc"] > 0.999


def test_append_jsonl_creates_file_and_appends(tmp_path):
    """First append creates the file; subsequent appends add lines."""
    import json
    from scripts.s2s.scoring import append_score_record
    path = tmp_path / "scores.jsonl"
    append_score_record(path, {"country": "kenya", "method": "raw", "acc": 0.1})
    append_score_record(path, {"country": "kenya", "method": "bcsd", "acc": 0.2})
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["method"] == "raw"
    assert json.loads(lines[1])["method"] == "bcsd"


def test_load_scored_keys_reads_idempotency_set(tmp_path):
    """load_scored_keys returns the set of (country, issuance, method, dekad) tuples
    that have already been scored so the verifier skips them on re-run."""
    from scripts.s2s.scoring import append_score_record, load_scored_keys
    path = tmp_path / "scores.jsonl"
    append_score_record(path, {
        "country": "kenya", "issuance": "2026-05-15", "method": "raw",
        "target_dekad": "2026-05-21", "acc": 0.1, "rmse": 1.0, "bias": 0.0,
    })
    keys = load_scored_keys(path)
    assert (("kenya", date(2026, 5, 15), "raw", date(2026, 5, 21))) in keys
