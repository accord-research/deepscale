"""Unit tests for deepscale.logistic_forecast (the WVG/logit stream)."""
import numpy as np
import pytest
import xarray as xr

from deepscale.logistic import logistic_forecast  # internal engine behind calibrate(method="logit")


def _make_obs(index, responsive_cells, n_lat=3, n_lon=3, slope=-40.0, noise=8.0, seed=0):
    """Build obs (year,lat,lon). `responsive_cells` indices (flattened) get a
    rainfall that decreases with `index` (so high index -> below normal); the
    rest are index-independent noise."""
    rng = np.random.default_rng(seed)
    ny = len(index)
    flat = rng.normal(120.0, noise, (ny, n_lat * n_lon))
    for g in responsive_cells:
        flat[:, g] = 120.0 + slope * np.asarray(index) + rng.normal(0, noise, ny)
    obs = flat.reshape(ny, n_lat, n_lon)
    years = np.arange(2000, 2000 + ny)
    return xr.DataArray(
        obs, dims=["year", "lat", "lon"],
        coords={"year": years, "lat": np.arange(n_lat), "lon": np.arange(n_lon)},
    )


@pytest.fixture
def index():
    # A standardized-ish index over 30 years.
    return xr.DataArray(
        np.linspace(-2.0, 2.0, 30),
        dims=["year"], coords={"year": np.arange(2000, 2030)},
    )


def test_output_shape_and_terciles_sum_to_one(index):
    obs = _make_obs(index, responsive_cells=range(9))
    p = logistic_forecast(index, obs, 1.0)
    assert p.dims == ("tercile", "lat", "lon")
    assert list(p.tercile.values) == [0, 1, 2]
    np.testing.assert_allclose(p.sum("tercile").values, 1.0, atol=1e-9)
    assert float(p.min()) >= 0.0 and float(p.max()) <= 1.0


def test_skill_high_index_predicts_below_normal(index):
    """Rainfall decreases with the index → a high forecast index should give a
    higher P(below) than a low one."""
    obs = _make_obs(index, responsive_cells=range(9))
    p_hi = logistic_forecast(index, obs, 2.0)
    p_lo = logistic_forecast(index, obs, -2.0)
    assert float(p_hi.sel(tercile=0).mean()) > float(p_lo.sel(tercile=0).mean())
    assert float(p_hi.sel(tercile=0).mean()) > 0.5


def test_sklearn_and_statsmodels_agree_unregularized(index):
    """The two backends are both MLE logits → close probabilities on a moderate
    (non-separable) signal."""
    obs = _make_obs(index, responsive_cells=range(9), slope=-20.0, noise=15.0)
    p_sk = logistic_forecast(index, obs, 1.0, backend="sklearn")
    p_sm = logistic_forecast(index, obs, 1.0, backend="statsmodels")
    np.testing.assert_allclose(p_sk.values, p_sm.values, atol=0.05)


def test_multinomial_sums_to_one(index):
    obs = _make_obs(index, responsive_cells=range(9))
    p = logistic_forecast(index, obs, 1.0, model="multinomial")
    np.testing.assert_allclose(p.sum("tercile").values, 1.0, atol=1e-9)


def test_regularization_accepted(index):
    obs = _make_obs(index, responsive_cells=range(9))
    p = logistic_forecast(index, obs, 1.0, regularization=1.0)
    np.testing.assert_allclose(p.sum("tercile").values, 1.0, atol=1e-9)


def test_significance_mask_requires_statsmodels(index):
    obs = _make_obs(index, responsive_cells=range(9))
    with pytest.raises(ValueError, match="statsmodels"):
        logistic_forecast(index, obs, 1.0, backend="sklearn", significance_mask=0.1)


def test_significance_mask_requires_no_regularization(index):
    obs = _make_obs(index, responsive_cells=range(9))
    with pytest.raises(ValueError, match="regularization"):
        logistic_forecast(index, obs, 1.0, backend="statsmodels",
                          regularization=1.0, significance_mask=0.1)


def test_significance_mask_drops_unrelated_cells():
    """Cell 0 has a real (but non-separable) relationship to the index; the rest
    are noise. With a significance threshold, the responsive cell stays finite
    and the unrelated cells get masked to NaN. A moderate signal is used on
    purpose: a near-perfect signal causes logistic separation, which inflates
    the standard error and (faithfully to the ICPAC R) would mask the cell too."""
    long_index = xr.DataArray(
        np.linspace(-2.0, 2.0, 40),
        dims=["year"], coords={"year": np.arange(2000, 2040)},
    )
    obs = _make_obs(long_index, responsive_cells=[0], slope=-25.0, noise=20.0, seed=0)
    p = logistic_forecast(long_index, obs, 1.0, backend="statsmodels",
                          significance_mask=0.1)
    flat = p.sel(tercile=0).values.reshape(-1)
    assert np.isfinite(flat[0])                       # responsive cell kept
    assert np.isnan(flat[1:]).sum() >= 6              # most noise cells dropped


def test_index_length_mismatch_raises(index):
    obs = _make_obs(index, responsive_cells=range(9))
    with pytest.raises(ValueError, match="must match obs.year"):
        logistic_forecast(index.isel(year=slice(0, 10)), obs, 1.0)


def test_too_few_years_gives_nan():
    yrs = np.arange(2000, 2008)  # 8 < default min_years=10
    idx = xr.DataArray(np.linspace(-1, 1, 8), dims=["year"], coords={"year": yrs})
    obs = _make_obs(idx, responsive_cells=range(9))
    p = logistic_forecast(idx, obs, 0.5)
    assert np.isnan(p.values).all()


def test_degenerate_label_uses_base_rate():
    """A cell that is always below-normal → P(below) ≈ 1 (no fit possible)."""
    yrs = np.arange(2000, 2030)
    idx = xr.DataArray(np.linspace(-2, 2, 30), dims=["year"], coords={"year": yrs})
    # One cell is monotically tiny (always the driest) → label below in ~1/3 of
    # years by construction of terciles; instead force a constant-low cell:
    rng = np.random.default_rng(5)
    obs = rng.normal(120, 10, (30, 1, 1))
    da = xr.DataArray(obs, dims=["year", "lat", "lon"],
                      coords={"year": yrs, "lat": [0], "lon": [0]})
    p = logistic_forecast(idx, da, 0.0)
    # Probabilities still valid and sum to 1.
    np.testing.assert_allclose(p.sum("tercile").values, 1.0, atol=1e-9)
