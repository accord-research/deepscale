import numpy as np
import pytest
import xarray as xr

import deepscale
from deepscale.registry import get_calibrator

SEASONS = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ",
           "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]


def _cube(n_year=25, n_member=6, nlat=3, nlon=4, seed=0):
    rng = np.random.default_rng(seed)
    ns = len(SEASONS)
    signal = rng.standard_normal((ns, n_year, nlat, nlon))
    members = signal[:, :, None] + 0.3 * rng.standard_normal((ns, n_year, n_member, nlat, nlon))
    obs = 0.7 * signal + 0.1 * rng.standard_normal((ns, n_year, nlat, nlon))
    coords = {"season": SEASONS, "year": np.arange(1991, 1991 + n_year),
              "member": np.arange(n_member), "lat": np.linspace(50, 60, nlat),
              "lon": np.linspace(-110, -100, nlon)}
    fc = xr.DataArray(members, dims=("season", "year", "member", "lat", "lon"), coords=coords)
    ob = xr.DataArray(obs, dims=("season", "year", "lat", "lon"),
                      coords={k: coords[k] for k in ("season", "year", "lat", "lon")})
    return fc, ob


def test_smoothed_regression_registered_and_deterministic():
    fn = get_calibrator("smoothed_regression")
    assert getattr(fn, "supports_deterministic", False) is True


def test_calibrate_deterministic_returns_season_map():
    fc, ob = _cube()
    out = deepscale.calibrate(fc, ob, method="smoothed_regression",
                              output_type="deterministic", temporal_sigma="constant",
                              forecast_year=2010)
    assert out.dims == ("season", "lat", "lon")
    assert bool(np.isfinite(out).all())


def test_calibrate_deterministic_rejected_for_ereg():
    fc, ob = _cube()
    with pytest.raises(ValueError, match="output_type"):
        deepscale.calibrate({"m": (fc.isel(season=0), fc.isel(season=0, year=-1))},
                            ob.isel(season=0), method="ereg", output_type="deterministic")


def test_smoothed_regression_tercile_output_supported():
    # Round 2 filled in the previously-stubbed tercile path; it now returns category
    # probabilities (below/near/above) summing to 1, with the season axis this method owns.
    fc, ob = _cube()
    out = deepscale.calibrate(fc, ob, method="smoothed_regression", output_type="tercile",
                              temporal_sigma="constant")
    assert out.dims == ("season", "tercile", "lat", "lon")
    np.testing.assert_allclose(out.sum("tercile").values, 1.0, atol=1e-9)


def test_smoothed_regression_out_of_sample_year_raises_clear_error():
    fc, ob = _cube()  # years 1991..2015
    with pytest.raises(ValueError, match="not in the hindcast years"):
        deepscale.calibrate(fc, ob, method="smoothed_regression",
                            output_type="deterministic", forecast_year=2099)


def test_smoothed_regression_separate_forecast_field_rejected():
    fc, ob = _cube()
    with pytest.raises(NotImplementedError, match="separate out-of-sample"):
        deepscale.calibrate(fc, ob, method="smoothed_regression",
                            output_type="deterministic", forecast=fc.isel(year=-1))
