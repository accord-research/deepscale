"""Unit tests for scripts.s2s.plotting."""

import numpy as np
import pytest
import xarray as xr


def _make_method_dataset(rng_seed: int = 0):
    rng = np.random.default_rng(rng_seed)
    lat = np.linspace(-5, 5, 12)
    lon = np.linspace(33, 42, 18)
    mean = xr.DataArray(
        rng.gamma(2, 1.5, (len(lat), len(lon))).astype("float32"),
        dims=["lat", "lon"], coords={"lat": lat, "lon": lon},
    )
    return xr.Dataset({"mean": mean})


def test_comparison_grid_triptych_with_obs():
    from scripts.s2s.plotting import comparison_grid
    obs = _make_method_dataset(9)
    methods = {
        "raw": _make_method_dataset(0),
        "bcsd": _make_method_dataset(1),
        "cca": _make_method_dataset(2),
    }
    fig = comparison_grid(obs, methods, dekad_label="2026-05-21")
    # 3 methods x 3 columns (obs | forecast | difference) = 9 data axes (plus colorbars).
    assert len(fig.axes) >= 3 * len(methods)
    assert "2026-05-21" in (fig._suptitle.get_text() if fig._suptitle else "")


def test_comparison_grid_without_obs_shows_forecasts_only():
    from scripts.s2s.plotting import comparison_grid
    methods = {"raw": _make_method_dataset(0), "bcsd": _make_method_dataset(1)}
    fig = comparison_grid(None, methods, dekad_label="2026-05-25")
    # No obs -> one forecast panel per method (plus a colorbar).
    assert len(fig.axes) >= len(methods)
    assert "2026-05-25" in (fig._suptitle.get_text() if fig._suptitle else "")


def test_metrics_panel_handles_empty_scores(tmp_path):
    """Empty scores list still returns a valid figure (a placeholder)."""
    from scripts.s2s.plotting import metrics_panel
    fig = metrics_panel(scores=[], country="kenya")
    assert fig is not None


def test_metrics_panel_with_real_scores():
    from scripts.s2s.plotting import metrics_panel
    scores = [
        {"method": "raw", "target_dekad": "2026-05-21", "acc": 0.1, "rmse": 1.0, "bias": 0.0, "rpss": 0.0},
        {"method": "bcsd", "target_dekad": "2026-05-21", "acc": 0.3, "rmse": 0.8, "bias": -0.1, "rpss": 0.05},
        {"method": "raw", "target_dekad": "2026-06-01", "acc": 0.2, "rmse": 0.9, "bias": 0.05, "rpss": 0.02},
        {"method": "bcsd", "target_dekad": "2026-06-01", "acc": 0.4, "rmse": 0.7, "bias": -0.05, "rpss": 0.08},
    ]
    fig = metrics_panel(scores=scores, country="kenya")
    assert len(fig.axes) >= 1  # at least one metric panel
