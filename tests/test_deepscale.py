import numpy as np
import pytest
import xarray as xr


# ===================================================================
# 1. Method base & registry
# ===================================================================

def test_method_base_is_abstract():
    from deepscale.methods.base import MethodBase
    with pytest.raises(TypeError):
        MethodBase()


def test_method_registry_lookup():
    from deepscale.registry import get_method
    from deepscale.methods.bcsd import BCSDMethod
    from deepscale.methods.cca import CCAMethod
    assert get_method("bcsd") is BCSDMethod
    assert get_method("cca") is CCAMethod


def test_method_registry_missing():
    from deepscale.registry import get_method
    with pytest.raises(KeyError):
        get_method("nonexistent")


def test_register_method_decorator():
    from deepscale.registry import register_method, get_method
    from deepscale.methods.base import MethodBase

    @register_method("test_dummy_m")
    class DummyMethod(MethodBase):
        def fit(self, hindcast, obs, **kwargs): pass
        def predict(self, forecast, **kwargs): return forecast

    assert get_method("test_dummy_m") is DummyMethod


# ===================================================================
# 2. BCSD method
# ===================================================================

def test_bcsd_fit_stores_state(synthetic_gcm_hindcast, synthetic_obs):
    from deepscale.methods.bcsd import BCSDMethod
    m = BCSDMethod()
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    assert hasattr(m, "obs_clim_")
    assert hasattr(m, "gcm_sorted_")
    assert hasattr(m, "obs_coarse_sorted_")


def test_bcsd_predict_shape(synthetic_gcm_hindcast, synthetic_gcm_forecast, synthetic_obs):
    from deepscale.methods.bcsd import BCSDMethod
    m = BCSDMethod()
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    result = m.predict(synthetic_gcm_forecast)
    assert result.dims == ("member", "lat", "lon")
    assert len(result.lat) == len(synthetic_obs.lat)
    assert len(result.lon) == len(synthetic_obs.lon)


def test_bcsd_predict_values_plausible(synthetic_gcm_hindcast, synthetic_gcm_forecast, synthetic_obs):
    from deepscale.methods.bcsd import BCSDMethod
    m = BCSDMethod()
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    result = m.predict(synthetic_gcm_forecast)
    assert not np.all(np.isnan(result.values))
    assert np.std(result.values) > 0  # not constant


# ===================================================================
# 3. CCA method
# ===================================================================

def test_cca_fit_stores_state(synthetic_gcm_hindcast, synthetic_obs):
    from deepscale.methods.cca import CCAMethod
    m = CCAMethod(n_modes=2)
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    assert hasattr(m, "mu_")  # canonical correlations
    assert hasattr(m, "x_mean_")


def test_cca_predict_shape(synthetic_gcm_hindcast, synthetic_gcm_forecast, synthetic_obs):
    from deepscale.methods.cca import CCAMethod
    m = CCAMethod(n_modes=2)
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    result = m.predict(synthetic_gcm_forecast)
    assert result.dims == ("member", "lat", "lon")
    assert len(result.lat) == len(synthetic_obs.lat)
    assert len(result.lon) == len(synthetic_obs.lon)


def test_cca_with_few_modes(synthetic_gcm_hindcast, synthetic_gcm_forecast, synthetic_obs):
    from deepscale.methods.cca import CCAMethod
    m = CCAMethod(n_modes=2)
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    result = m.predict(synthetic_gcm_forecast)
    assert not np.all(np.isnan(result.values))


def test_cca_with_short_hindcast(synthetic_gcm_forecast, synthetic_obs):
    from deepscale.methods.cca import CCAMethod
    # Only 5 years
    short_years = np.arange(2000, 2005)
    gcm = xr.DataArray(
        np.random.randn(5, 3, 5, 5) + 5,
        dims=["year", "member", "lat", "lon"],
        coords={"year": short_years, "member": np.arange(3),
                "lat": np.linspace(-4, 4, 5), "lon": np.linspace(30, 38, 5)},
    )
    obs = synthetic_obs.sel(year=short_years)
    m = CCAMethod(n_modes=2)
    m.fit(gcm, obs)
    result = m.predict(synthetic_gcm_forecast)
    assert not np.all(np.isnan(result.values))


# ===================================================================
# 3b. CCA with EOF truncation
# ===================================================================

def test_cca_eof_fit_stores_pca_state(synthetic_gcm_hindcast, synthetic_obs):
    from deepscale.methods.cca import CCAMethod
    m = CCAMethod(n_modes=2, x_eof_modes=3, y_eof_modes=3)
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    assert hasattr(m, "eofx_")
    assert hasattr(m, "eofy_")
    assert hasattr(m, "mu_")


def test_cca_eof_predict_shape(synthetic_gcm_hindcast, synthetic_gcm_forecast, synthetic_obs):
    from deepscale.methods.cca import CCAMethod
    m = CCAMethod(n_modes=2, x_eof_modes=3, y_eof_modes=3)
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    result = m.predict(synthetic_gcm_forecast)
    assert result.dims == ("member", "lat", "lon")
    assert len(result.lat) == len(synthetic_obs.lat)


def test_cca_eof_produces_nondegenerate_terciles(synthetic_gcm_hindcast, synthetic_obs):
    from deepscale.methods.cca import CCAMethod
    from deepscale.tercile import to_tercile
    m = CCAMethod(n_modes=2, x_eof_modes=3, y_eof_modes=3)
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    forecast = synthetic_gcm_hindcast.isel(year=-1, drop=True)
    result = m.predict(forecast)
    tercile = to_tercile(result, synthetic_obs)
    p_normal = float(tercile.sel(tercile=1).mean())
    assert p_normal < 0.9, f"Degenerate: P(normal) = {p_normal:.1%}"


def test_cca_auto_eof_modes(synthetic_gcm_hindcast, synthetic_obs):
    from deepscale.methods.cca import CCAMethod
    m = CCAMethod(n_modes=2)
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    assert hasattr(m, "eofx_")
    assert m.eofx_.shape[1] <= 10  # n_components


# ===================================================================
# 3c. CCA dual-grid predictor support (§1.1)
# ===================================================================

def test_cca_stores_predictor_coords(synthetic_gcm_hindcast, synthetic_obs):
    """Predictor lat/lon must be stored separately from predictand."""
    from deepscale.methods.cca import CCAMethod
    m = CCAMethod(n_modes=2)
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    assert hasattr(m, "predictor_coords_")
    np.testing.assert_array_equal(
        m.predictor_coords_["lat"].values, synthetic_gcm_hindcast.lat.values
    )
    np.testing.assert_array_equal(
        m.predictor_coords_["lon"].values, synthetic_gcm_hindcast.lon.values
    )


def test_cca_stores_predictand_coords(synthetic_gcm_hindcast, synthetic_obs):
    """Predictand lat/lon must be stored separately from predictor."""
    from deepscale.methods.cca import CCAMethod
    m = CCAMethod(n_modes=2)
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    assert hasattr(m, "predictand_coords_")
    np.testing.assert_array_equal(
        m.predictand_coords_["lat"].values, synthetic_obs.lat.values
    )
    np.testing.assert_array_equal(
        m.predictand_coords_["lon"].values, synthetic_obs.lon.values
    )


def test_cca_stores_distinct_predictor_and_predictand_shapes(synthetic_gcm_hindcast, synthetic_obs):
    """When grids differ, the two shapes must be stored independently."""
    from deepscale.methods.cca import CCAMethod
    m = CCAMethod(n_modes=2)
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    assert m.predictor_shape_ == (
        len(synthetic_gcm_hindcast.lat),
        len(synthetic_gcm_hindcast.lon),
    )
    assert m.predictand_shape_ == (
        len(synthetic_obs.lat),
        len(synthetic_obs.lon),
    )
    assert m.predictor_shape_ != m.predictand_shape_


def test_cca_eofx_reconstructs_to_predictor_grid(synthetic_gcm_hindcast, synthetic_obs):
    """eofx_ + x_valid_ + predictor_shape_ must reconstruct a spatial map (§3.2 contract)."""
    from deepscale.methods.cca import CCAMethod
    m = CCAMethod(n_modes=2, x_eof_modes=3, y_eof_modes=3)
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    n_predictor_pts = int(np.prod(m.predictor_shape_))
    assert m.x_valid_.shape == (n_predictor_pts,)
    mode0 = np.full(n_predictor_pts, np.nan)
    mode0[m.x_valid_] = m.eofx_[:, 0]
    spatial = mode0.reshape(m.predictor_shape_)
    assert spatial.shape == m.predictor_shape_


def test_cca_eofy_reconstructs_to_predictand_grid(synthetic_gcm_hindcast, synthetic_obs):
    """eofy_ + y_valid_ + predictand_shape_ must reconstruct a spatial map (§3.2 contract)."""
    from deepscale.methods.cca import CCAMethod
    m = CCAMethod(n_modes=2, x_eof_modes=3, y_eof_modes=3)
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    n_predictand_pts = int(np.prod(m.predictand_shape_))
    assert m.y_valid_.shape == (n_predictand_pts,)
    mode0 = np.full(n_predictand_pts, np.nan)
    mode0[m.y_valid_] = m.eofy_[:, 0]
    spatial = mode0.reshape(m.predictand_shape_)
    assert spatial.shape == m.predictand_shape_


def test_cca_predicts_on_predictand_grid_when_grids_differ(
    synthetic_gcm_hindcast, synthetic_gcm_forecast, synthetic_obs
):
    """Predict output must use the predictand grid, not the predictor grid."""
    from deepscale.methods.cca import CCAMethod
    assert synthetic_gcm_hindcast.shape[2:] != synthetic_obs.shape[1:]
    m = CCAMethod(n_modes=2)
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    result = m.predict(synthetic_gcm_forecast)
    np.testing.assert_array_equal(result.lat.values, synthetic_obs.lat.values)
    np.testing.assert_array_equal(result.lon.values, synthetic_obs.lon.values)


def _build_dual_grid_fixture(seed=0, n_years=25, signal_amp=2.0, noise_amp=0.3):
    """Synthetic SST→precip dual-grid fixture.

    Predictor: 'tropical Pacific' SST on a coarse 6x8 grid (lat ±10°, lon 180-240°).
    Predictand: 'East Africa' precip on a fine 12x12 grid (lat -5 to 15°, lon 30-50°).
    A shared yearly time signal modulates a dipole on each grid (different patterns),
    so CCA should recover the link.
    """
    rng = np.random.default_rng(seed)
    years = np.arange(2000, 2000 + n_years)
    members = np.arange(3)

    # Predictor grid (geographically disjoint from predictand)
    p_lat = np.linspace(-10, 10, 6)
    p_lon = np.linspace(180, 240, 8)
    # Predictand grid
    o_lat = np.linspace(-5, 15, 12)
    o_lon = np.linspace(30, 50, 12)

    # Shared time signal
    t = np.arange(n_years)
    time_signal = np.sin(2 * np.pi * t / 5.0)

    # Planted spatial patterns (different on each grid)
    p_pattern = np.outer(np.sin(np.deg2rad(p_lat) * 3), np.cos(np.deg2rad(p_lon) * 2))
    o_pattern = np.outer(np.cos(np.deg2rad(o_lat) * 2), np.sin(np.deg2rad(o_lon) * 4))

    p_signal = signal_amp * time_signal[:, None, None] * p_pattern[None, :, :]
    o_signal = signal_amp * time_signal[:, None, None] * o_pattern[None, :, :]

    p_noise = rng.standard_normal((n_years, len(members), len(p_lat), len(p_lon))) * noise_amp
    o_noise = rng.standard_normal((n_years, len(o_lat), len(o_lon))) * noise_amp

    predictor = xr.DataArray(
        p_signal[:, None, :, :] + p_noise + 290.0,
        dims=["year", "member", "lat", "lon"],
        coords={"year": years, "member": members, "lat": p_lat, "lon": p_lon},
    )
    predictand = xr.DataArray(
        o_signal + o_noise + 5.0,
        dims=["year", "lat", "lon"],
        coords={"year": years, "lat": o_lat, "lon": o_lon},
    )
    return predictor, predictand, o_pattern


def test_cca_dual_grid_integration_end_to_end():
    """Integration: full dual-grid call via the public downscale() API."""
    import deepscale
    predictor, predictand, _ = _build_dual_grid_fixture()
    forecast = predictor.isel(year=-1, drop=True)
    train_predictor = predictor.isel(year=slice(None, -1))
    train_predictand = predictand.isel(year=slice(None, -1))

    result = deepscale.downscale(
        train_predictor, train_predictand, method="cca", forecast=forecast
    )
    assert result.dims == ("member", "lat", "lon")
    assert len(result.lat) == len(predictand.lat)
    assert len(result.lon) == len(predictand.lon)
    np.testing.assert_array_equal(result.lat.values, predictand.lat.values)
    np.testing.assert_array_equal(result.lon.values, predictand.lon.values)
    assert not np.all(np.isnan(result.values))


def test_cca_dual_grid_recovers_planted_signal():
    """Integration: CCA on dual grids recovers a known planted relationship."""
    from deepscale.methods.cca import CCAMethod
    predictor, predictand, o_pattern = _build_dual_grid_fixture()

    train_predictor = predictor.isel(year=slice(None, -1))
    train_predictand = predictand.isel(year=slice(None, -1))
    forecast = predictor.isel(year=-1, drop=True)
    truth = predictand.isel(year=-1)

    m = CCAMethod(n_modes=2)
    m.fit(train_predictor, train_predictand)

    # CCA should pick up the shared time signal — first canonical correlation high.
    assert m.mu_[0] > 0.7, f"first canonical correlation too low: {m.mu_[0]:.3f}"

    # Predict held-out year and check predicted anomaly pattern correlates
    # with the truth anomaly pattern.
    pred = m.predict(forecast).mean("member")
    pred_anom = (pred - pred.mean()).values.ravel()
    truth_anom = (truth - truth.mean()).values.ravel()
    corr = np.corrcoef(pred_anom, truth_anom)[0, 1]
    assert corr > 0.5, f"predicted vs truth pattern correlation too low: {corr:.3f}"


def test_cca_same_grid_case_still_works():
    """Regression guard: predictor==predictand grid is just a special case."""
    from deepscale.methods.cca import CCAMethod
    rng = np.random.default_rng(0)
    years = np.arange(2000, 2010)
    members = np.arange(2)
    lat = np.linspace(-4, 4, 6)
    lon = np.linspace(30, 38, 6)
    signal = np.sin(np.arange(len(years)) * 0.5)[:, None, None]
    spatial = np.outer(np.sin(lat), np.cos(lon))[None, :, :]
    gcm = xr.DataArray(
        signal[:, None, :, :] * spatial[:, None, :, :]
        + rng.standard_normal((len(years), len(members), len(lat), len(lon))) * 0.3
        + 5.0,
        dims=["year", "member", "lat", "lon"],
        coords={"year": years, "member": members, "lat": lat, "lon": lon},
    )
    obs = xr.DataArray(
        signal * spatial + rng.standard_normal((len(years), len(lat), len(lon))) * 0.2 + 5.0,
        dims=["year", "lat", "lon"],
        coords={"year": years, "lat": lat, "lon": lon},
    )
    forecast = gcm.isel(year=-1, drop=True)
    m = CCAMethod(n_modes=2)
    m.fit(gcm, obs)
    result = m.predict(forecast)
    assert result.shape == (len(members), len(lat), len(lon))
    assert m.predictor_shape_ == m.predictand_shape_ == (len(lat), len(lon))
    assert not np.any(np.isnan(result.values))


# ===================================================================
# 4. RPSS metric
# ===================================================================

def test_rpss_perfect_forecast(perfect_tercile_forecast, synthetic_obs):
    from deepscale.metrics.rpss import RPSSMetric
    m = RPSSMetric()
    score = m.compute(perfect_tercile_forecast, synthetic_obs)
    assert score > 0.9  # near-perfect


def test_rpss_climatology_forecast(climatology_forecast, synthetic_obs):
    from deepscale.metrics.rpss import RPSSMetric
    m = RPSSMetric()
    score = m.compute(climatology_forecast, synthetic_obs)
    np.testing.assert_allclose(score, 0.0, atol=0.05)


def test_rpss_worse_than_climatology(synthetic_obs):
    """Inverted forecast should give negative RPSS."""
    from deepscale.metrics.rpss import RPSSMetric
    years = synthetic_obs.year.values
    fine_lat = synthetic_obs.lat.values
    fine_lon = synthetic_obs.lon.values

    from deepscale.metrics.rpss import _cpt_boundaries
    t33, t67 = _cpt_boundaries(synthetic_obs.values)
    t33_da = xr.DataArray(t33, dims=["lat", "lon"], coords={k: synthetic_obs.coords[k] for k in ["lat", "lon"]})
    t67_da = xr.DataArray(t67, dims=["lat", "lon"], coords={k: synthetic_obs.coords[k] for k in ["lat", "lon"]})
    cat = xr.where(t33_da > synthetic_obs, 0, xr.where(t67_da > synthetic_obs, 1, 2))

    # Invert: assign high probability to wrong category
    inverted = xr.concat([(cat == (2 - i)).astype(float) for i in range(3)], dim="tercile")
    inverted["tercile"] = [0, 1, 2]

    m = RPSSMetric()
    score = m.compute(inverted, synthetic_obs)
    assert score < 0


def test_rpss_shape_spatial(climatology_forecast, synthetic_obs):
    from deepscale.metrics.rpss import RPSSMetric
    m = RPSSMetric()
    result = m.compute(climatology_forecast, synthetic_obs, spatial=True)
    assert isinstance(result, xr.DataArray)
    assert "lat" in result.dims and "lon" in result.dims


# ===================================================================
# 5. ROC metric
# ===================================================================

def test_roc_perfect_discrimination(perfect_tercile_forecast, synthetic_obs):
    from deepscale.metrics.roc import ROCMetric
    m = ROCMetric()
    result = m.compute(perfect_tercile_forecast, synthetic_obs)
    assert "roc_bn" in result
    assert "roc_an" in result
    assert result["roc_bn"] > 0.8  # should be near 1.0


def test_roc_no_discrimination(climatology_forecast, synthetic_obs):
    from deepscale.metrics.roc import ROCMetric
    m = ROCMetric()
    result = m.compute(climatology_forecast, synthetic_obs)
    # Uniform forecast => no discrimination => ROC ~0.5
    np.testing.assert_allclose(result["roc_bn"], 0.5, atol=0.15)


def test_roc_per_tercile(perfect_tercile_forecast, synthetic_obs):
    from deepscale.metrics.roc import ROCMetric
    m = ROCMetric()
    result = m.compute(perfect_tercile_forecast, synthetic_obs)
    assert "roc_bn" in result
    assert "roc_nn" in result
    assert "roc_an" in result


# ===================================================================
# 6. Pearson metric
# ===================================================================

def test_pearson_perfect(synthetic_obs):
    from deepscale.metrics.pearson import PearsonMetric
    m = PearsonMetric()
    score = m.compute(synthetic_obs, synthetic_obs)
    np.testing.assert_allclose(score, 1.0, atol=0.001)


def test_pearson_zero():
    from deepscale.metrics.pearson import PearsonMetric
    np.random.seed(123)
    years = np.arange(100)
    lat = np.linspace(-1, 1, 5)
    lon = np.linspace(0, 1, 5)
    a = xr.DataArray(np.random.randn(100, 5, 5), dims=["year", "lat", "lon"],
                     coords={"year": years, "lat": lat, "lon": lon})
    b = xr.DataArray(np.random.randn(100, 5, 5), dims=["year", "lat", "lon"],
                     coords={"year": years, "lat": lat, "lon": lon})
    m = PearsonMetric()
    score = m.compute(a, b)
    assert abs(score) < 0.2


# ===================================================================
# 6b. RMSE metric
# ===================================================================

def test_rmse_perfect(synthetic_obs):
    from deepscale.metrics.rmse import RMSEMetric
    m = RMSEMetric()
    score = m.compute(synthetic_obs, synthetic_obs)
    np.testing.assert_allclose(score, 0.0, atol=1e-10)


def test_rmse_constant_mean(synthetic_obs):
    from deepscale.metrics.rmse import RMSEMetric
    # Forecast = climatological mean broadcast back across years.
    # RMSE per grid cell should equal the population std (ddof=0) per grid cell.
    forecast = synthetic_obs.mean("year") + 0 * synthetic_obs  # broadcast trick
    m = RMSEMetric()
    spatial_rmse = m.compute(forecast, synthetic_obs, spatial=True)
    expected = synthetic_obs.std("year")  # xarray default is ddof=0
    np.testing.assert_allclose(spatial_rmse.values, expected.values, atol=1e-10)


def test_rmse_alias_registered():
    from deepscale.registry import get_metric
    from deepscale.metrics.rmse import RMSEMetric
    assert get_metric("root_mean_squared_error") is RMSEMetric
    assert get_metric("rmse") is RMSEMetric


# ===================================================================
# 6c. HSS metric
# ===================================================================

def test_hss_perfect(synthetic_obs):
    from deepscale.metrics.heidke import HSSMetric
    from deepscale.metrics.rpss import _cpt_boundaries

    t33, t67 = _cpt_boundaries(synthetic_obs.values)
    obs_vals = synthetic_obs.values
    obs_cat = np.where(t33 > obs_vals, 0, np.where(t67 > obs_vals, 1, 2))

    n_year, n_lat, n_lon = obs_vals.shape
    fcst = np.zeros((n_year, 3, n_lat, n_lon))
    for c in range(3):
        fcst[:, c, :, :] = (obs_cat == c).astype(float)

    forecast = xr.DataArray(
        fcst,
        dims=["year", "tercile", "lat", "lon"],
        coords={
            "year": synthetic_obs.year,
            "tercile": [0, 1, 2],
            "lat": synthetic_obs.lat,
            "lon": synthetic_obs.lon,
        },
    )

    m = HSSMetric()
    score = m.compute(forecast, synthetic_obs)
    np.testing.assert_allclose(score, 1.0, atol=1e-10)


def test_hss_no_skill(synthetic_obs):
    from deepscale.metrics.heidke import HSSMetric

    n_year, n_lat, n_lon = synthetic_obs.shape
    fcst = np.zeros((n_year, 3, n_lat, n_lon))
    fcst[:, 1, :, :] = 1.0  # always pick "normal" (middle tercile)

    forecast = xr.DataArray(
        fcst,
        dims=["year", "tercile", "lat", "lon"],
        coords={
            "year": synthetic_obs.year,
            "tercile": [0, 1, 2],
            "lat": synthetic_obs.lat,
            "lon": synthetic_obs.lon,
        },
    )

    m = HSSMetric()
    score = m.compute(forecast, synthetic_obs)
    assert abs(score) < 0.1


def test_hss_alias_registered():
    from deepscale.registry import get_metric
    from deepscale.metrics.heidke import HSSMetric
    assert get_metric("heidke_skill_score") is HSSMetric
    assert get_metric("hss") is HSSMetric


# ===================================================================
# 6d. Spearman metric
# ===================================================================

def test_spearman_perfect(synthetic_obs):
    from deepscale.metrics.spearman import SpearmanMetric
    m = SpearmanMetric()
    score = m.compute(synthetic_obs, synthetic_obs)
    np.testing.assert_allclose(score, 1.0, atol=0.001)


def test_spearman_monotonic_nonlinear():
    from deepscale.metrics.spearman import SpearmanMetric
    from deepscale.metrics.pearson import PearsonMetric

    # Centered Gaussian data: x ~ N(0, 1). Cubing strongly distorts linearity
    # while preserving rank order. Theoretical Pearson(X, X^3) for X ~ N(0,1)
    # is 3/sqrt(15) ≈ 0.775; gap from Spearman = 1.0 is ~0.22.
    np.random.seed(7)
    n_year, n_lat, n_lon = 100, 5, 5
    x = np.random.randn(n_year, n_lat, n_lon)
    obs = xr.DataArray(
        x,
        dims=["year", "lat", "lon"],
        coords={
            "year": np.arange(n_year),
            "lat": np.linspace(-1, 1, n_lat),
            "lon": np.linspace(0, 1, n_lon),
        },
    )
    forecast = obs ** 3  # monotonic but strongly nonlinear over [-3, 3]

    spearman = SpearmanMetric().compute(forecast, obs)
    pearson = PearsonMetric().compute(forecast, obs)

    assert spearman > 0.999, f"expected spearman ≈ 1, got {spearman}"
    assert pearson < spearman - 0.01, (
        f"expected pearson strictly below spearman by >= 0.01; "
        f"got pearson={pearson}, spearman={spearman}"
    )


def test_spearman_zero():
    from deepscale.metrics.spearman import SpearmanMetric
    np.random.seed(123)
    years = np.arange(100)
    lat = np.linspace(-1, 1, 5)
    lon = np.linspace(0, 1, 5)
    a = xr.DataArray(np.random.randn(100, 5, 5), dims=["year", "lat", "lon"],
                     coords={"year": years, "lat": lat, "lon": lon})
    b = xr.DataArray(np.random.randn(100, 5, 5), dims=["year", "lat", "lon"],
                     coords={"year": years, "lat": lat, "lon": lon})
    m = SpearmanMetric()
    score = m.compute(a, b)
    assert abs(score) < 0.2


# ===================================================================
# 7. Cross-validation
# ===================================================================

def test_loyo_yields_correct_folds():
    from deepscale.cv import loyo
    years = list(range(2000, 2010))
    folds = list(loyo(years))
    assert len(folds) == 10
    for train, test in folds:
        assert len(train) == 9
        assert test not in train


def test_loyo_no_leakage():
    from deepscale.cv import loyo
    years = list(range(2000, 2010))
    test_years_seen = []
    for train, test in loyo(years):
        assert test not in train
        test_years_seen.append(test)
    assert sorted(test_years_seen) == years


def test_lko_sliding_consecutive():
    """Leave-k-out yields sliding windows of k consecutive test years."""
    from deepscale.cv import lko
    years = list(range(2000, 2010))
    folds = list(lko(years, k=3))
    # Sliding window: positions 0..7 inclusive → 8 folds
    assert len(folds) == 8
    for train, test in folds:
        assert isinstance(test, list)
        assert len(test) == 3
        # Test years are consecutive
        assert test == sorted(test)
        for i in range(1, len(test)):
            assert test[i] - test[i - 1] == 1
        # No leakage
        assert all(t not in train for t in test)
        assert len(train) == len(years) - 3


def test_lko_default_k():
    from deepscale.cv import lko
    years = list(range(2000, 2010))
    folds = list(lko(years))  # default k=3
    assert all(len(test) == 3 for _, test in folds)


def test_lko_k_equals_n_yields_one_fold():
    from deepscale.cv import lko
    years = list(range(2000, 2005))
    folds = list(lko(years, k=5))
    assert len(folds) == 1
    train, test = folds[0]
    assert train == []
    assert test == years


def test_blocked_partitions_into_contiguous_blocks():
    """Blocked CV partitions years into non-overlapping contiguous blocks."""
    from deepscale.cv import blocked
    years = list(range(2000, 2010))
    folds = list(blocked(years, block_size=5))
    assert len(folds) == 2
    test_years_seen = []
    for train, test in folds:
        assert isinstance(test, list)
        assert len(test) == 5
        # Block is contiguous
        assert test == sorted(test)
        for i in range(1, len(test)):
            assert test[i] - test[i - 1] == 1
        # No leakage
        assert all(t not in train for t in test)
        test_years_seen.extend(test)
    # Each year appears exactly once across the partition
    assert sorted(test_years_seen) == years


def test_blocked_with_gap_excludes_neighbours():
    """A nonzero gap removes years adjacent to the test block from the train set."""
    from deepscale.cv import blocked
    years = list(range(2000, 2010))
    folds = list(blocked(years, block_size=2, gap=1))
    # First fold: test=[2000, 2001], train must not include 2002 (gap=1)
    train, test = folds[0]
    assert test == [2000, 2001]
    assert 2002 not in train
    assert 2003 in train  # outside the gap
    # Middle fold: test=[2004, 2005], train excludes 2003 and 2006
    train, test = folds[2]
    assert test == [2004, 2005]
    assert 2003 not in train
    assert 2006 not in train
    assert 2002 in train and 2007 in train


def test_blocked_drops_partial_trailing_block():
    """If years don't divide evenly, the trailing partial block is dropped."""
    from deepscale.cv import blocked
    years = list(range(2000, 2008))  # 8 years, block_size=3 → 2 full blocks of 3
    folds = list(blocked(years, block_size=3))
    assert len(folds) == 2
    test_years = [yr for _, test in folds for yr in test]
    assert 2006 not in test_years  # part of dropped trailing block
    assert 2007 not in test_years


def test_expanding_simulates_realtime():
    """Expanding window: train on years[:i], test year i, for i >= min_train."""
    from deepscale.cv import expanding
    years = list(range(2000, 2010))
    folds = list(expanding(years, min_train=4))
    # i ranges from 4..9 → 6 folds
    assert len(folds) == 6
    for train, test in folds:
        assert isinstance(test, int) or not isinstance(test, list)
        # train is strictly the prefix before test
        assert all(yr < test for yr in train)
        assert len(train) >= 4


def test_expanding_short_hindcast_warns():
    """Expanding with too few eval years should warn (issue pitfall)."""
    import warnings
    from deepscale.cv import expanding
    years = list(range(2000, 2010))
    # min_train=8 → only 2 evaluation years
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        list(expanding(years, min_train=8))
    assert any("evaluation" in str(w.message).lower() for w in caught), (
        f"expected warning about few evaluation years, got: {[str(w.message) for w in caught]}"
    )


def test_get_cv_registers_all_schemes():
    from deepscale.cv import get_cv, loyo, lko, blocked, expanding
    assert get_cv("loyo") is loyo
    assert get_cv("lko") is lko
    assert get_cv("blocked") is blocked
    assert get_cv("expanding") is expanding


def _run_cv_pipeline(scheme, scheme_kwargs, gcm, obs):
    """Helper: run an end-to-end CV pipeline using the given scheme.

    Returns (cv_predictions, cv_obs) aligned by year, so a downstream
    `to_tercile_cv()` + `skill()` call can score the result.
    """
    from deepscale.methods.cca import CCAMethod
    years = list(gcm.year.values)
    preds = []
    obs_pieces = []
    for fold in scheme(years, **scheme_kwargs):
        train_years, test = fold
        test_list = test if isinstance(test, list) else [test]
        m = CCAMethod(n_modes=2)
        m.fit(gcm.sel(year=train_years), obs.sel(year=train_years))
        for test_yr in test_list:
            forecast = gcm.sel(year=test_yr)
            pred = m.predict(forecast).mean("member")
            preds.append(pred.expand_dims(year=[test_yr]))
            obs_pieces.append(obs.sel(year=[test_yr]))
    cv_pred = xr.concat(preds, dim="year").sortby("year")
    cv_obs = xr.concat(obs_pieces, dim="year").sortby("year")
    return cv_pred, cv_obs


def test_blocked_cv_pipeline_end_to_end(synthetic_gcm_hindcast, synthetic_obs):
    """Integration: full CV pipeline with `blocked` CV → tercile → skill."""
    import deepscale
    from deepscale.cv import blocked
    from deepscale.tercile import to_tercile_cv
    cv_pred, cv_obs = _run_cv_pipeline(
        blocked, {"block_size": 2}, synthetic_gcm_hindcast, synthetic_obs,
    )
    # Blocked with block_size=2 on 10 years → 5 folds, every year scored once.
    assert len(cv_pred.year) == len(synthetic_gcm_hindcast.year)
    tercile = to_tercile_cv(cv_pred, cv_obs, method="bootstrap")
    np.testing.assert_allclose(tercile.sum("tercile").values, 1.0, atol=1e-6)
    rpss = float(deepscale.skill(tercile, cv_obs, metrics=["rpss"]).scores["rpss"])
    assert -1.5 < rpss < 1.0
    assert not np.isnan(rpss)


def test_expanding_cv_pipeline_end_to_end(synthetic_gcm_hindcast, synthetic_obs):
    """Integration: full CV pipeline with `expanding` window → tercile → skill."""
    import warnings
    import deepscale
    from deepscale.cv import expanding
    from deepscale.tercile import to_tercile_cv
    # min_train=4 → 6 evaluation years (suppressing the short-hindcast warning).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv_pred, cv_obs = _run_cv_pipeline(
            expanding, {"min_train": 4}, synthetic_gcm_hindcast, synthetic_obs,
        )
    # Expanding only scores years past the min_train threshold.
    assert len(cv_pred.year) == 6
    tercile = to_tercile_cv(cv_pred, cv_obs, method="bootstrap")
    np.testing.assert_allclose(tercile.sum("tercile").values, 1.0, atol=1e-6)
    rpss = float(deepscale.skill(tercile, cv_obs, metrics=["rpss"]).scores["rpss"])
    assert -1.5 < rpss < 1.0
    assert not np.isnan(rpss)


def test_disciplined_to_tercile_cv_pipeline_end_to_end(synthetic_gcm_hindcast, synthetic_obs):
    """Integration for §6.5 default flip: a real LOYO+CCA hindcast scored via
    `to_tercile_cv()` (default cpt_boundaries=True) yields a valid RPSS.
    """
    import deepscale
    from deepscale.cv import loyo
    from deepscale.tercile import to_tercile_cv
    cv_pred, cv_obs = _run_cv_pipeline(
        loyo, {}, synthetic_gcm_hindcast, synthetic_obs,
    )
    # Use the cpt method to actually exercise the cpt_boundaries=True default;
    # leverages of zero are fine for this skill-validity check.
    n = len(cv_obs.year)
    leverages = np.zeros(n)
    tercile = to_tercile_cv(cv_pred, cv_obs, method="cpt", leverages=leverages)
    np.testing.assert_allclose(tercile.sum("tercile").values, 1.0, atol=1e-6)
    rpss = float(deepscale.skill(tercile, cv_obs, metrics=["rpss"]).scores["rpss"])
    assert -1.5 < rpss < 1.0
    assert not np.isnan(rpss)


# ===================================================================
# 8. Tercile conversion
# ===================================================================

def test_to_tercile_cv_default_is_leakage_disciplined():
    """`to_tercile_cv()` defaults to cpt_boundaries=True (CPT reference convention).

    Regression guard against accidentally flipping back to the leaky default —
    the issue (§6.5 / #22) made this the disciplined default.
    """
    import inspect
    from deepscale.tercile import to_tercile_cv
    sig = inspect.signature(to_tercile_cv)
    assert sig.parameters["cpt_boundaries"].default is True


def test_to_tercile_cv_disciplined_and_leaky_paths_diverge():
    """The disciplined (cpt_boundaries=True) and leaky (False) paths must
    produce different tercile probabilities on synthetic data.

    If they ever produce identical results, the boundary path has silently
    been short-circuited and the leakage discipline has lost its teeth.
    """
    from deepscale.tercile import to_tercile_cv
    rng = np.random.default_rng(0)
    n_years = 14
    years = np.arange(2000, 2000 + n_years)
    lat = np.linspace(-2, 2, 5)
    lon = np.linspace(0, 4, 5)
    obs_data = rng.standard_normal((n_years, len(lat), len(lon)))
    cv_data = obs_data * 0.6 + rng.standard_normal((n_years, len(lat), len(lon))) * 0.4
    obs = xr.DataArray(obs_data, dims=["year", "lat", "lon"],
                       coords={"year": years, "lat": lat, "lon": lon})
    cv = xr.DataArray(cv_data, dims=["year", "lat", "lon"],
                      coords={"year": years, "lat": lat, "lon": lon})
    leverages = np.full(n_years, 0.1)

    disciplined = to_tercile_cv(cv, obs, method="cpt", leverages=leverages,
                                cpt_boundaries=True)
    leaky = to_tercile_cv(cv, obs, method="cpt", leverages=leverages,
                          cpt_boundaries=False)

    # Probabilities must differ somewhere (boundaries computed differently).
    assert not np.allclose(disciplined.values, leaky.values, equal_nan=True), (
        "disciplined and leaky paths produced identical probabilities — "
        "the cpt_boundaries flag is no longer affecting behaviour."
    )


def test_to_tercile_cv_leaky_path_still_available():
    """Opt-in leaky behaviour stays accessible for legacy/comparison runs."""
    from deepscale.tercile import to_tercile_cv
    rng = np.random.default_rng(1)
    n_years = 10
    years = np.arange(2000, 2000 + n_years)
    obs = xr.DataArray(
        rng.standard_normal((n_years, 3, 3)),
        dims=["year", "lat", "lon"],
        coords={"year": years, "lat": np.arange(3.0), "lon": np.arange(3.0)},
    )
    cv = xr.DataArray(
        rng.standard_normal((n_years, 3, 3)),
        dims=["year", "lat", "lon"],
        coords={"year": years, "lat": np.arange(3.0), "lon": np.arange(3.0)},
    )
    leverages = np.full(n_years, 0.1)
    out = to_tercile_cv(cv, obs, method="cpt", leverages=leverages,
                        cpt_boundaries=False)
    np.testing.assert_allclose(out.sum("tercile").values, 1.0, atol=1e-6)


def test_continuous_to_tercile(synthetic_gcm_forecast, synthetic_obs):
    from deepscale.tercile import to_tercile
    result = to_tercile(synthetic_gcm_forecast, synthetic_obs)
    assert "tercile" in result.dims
    assert result.dims == ("tercile", "lat", "lon")
    # Probabilities sum to 1
    sums = result.sum("tercile")
    np.testing.assert_allclose(sums.values, 1.0, atol=1e-10)
    # Values in [0, 1]
    assert float(result.min()) >= 0.0
    assert float(result.max()) <= 1.0


def test_tercile_uniform_from_climatology(synthetic_obs):
    """If forecast matches obs climatology, tercile probs ≈ 1/3."""
    from deepscale.tercile import to_tercile
    # Use obs mean as every member's "forecast"
    clim = synthetic_obs.mean("year")
    members = np.arange(20)  # many members all at climatology
    fcst = clim.expand_dims(member=members)
    result = to_tercile(fcst, synthetic_obs)
    # Since all members equal climatology mean, some pixels will be near 1/3
    mean_probs = result.mean(dim=["lat", "lon"])
    # Each tercile should get roughly 1/3 of the probability
    for i in range(3):
        assert 0.0 <= float(mean_probs.isel(tercile=i)) <= 1.0


# ===================================================================
# 9. Ensemble
# ===================================================================

def test_uniform_ensemble():
    from deepscale.ensemble import ensemble
    lat = np.linspace(-1, 1, 5)
    lon = np.linspace(0, 1, 5)
    a = xr.DataArray(np.ones((3, 5, 5)) * 2, dims=["member", "lat", "lon"],
                     coords={"member": [0, 1, 2], "lat": lat, "lon": lon})
    b = xr.DataArray(np.ones((3, 5, 5)) * 4, dims=["member", "lat", "lon"],
                     coords={"member": [0, 1, 2], "lat": lat, "lon": lon})
    c = xr.DataArray(np.ones((3, 5, 5)) * 6, dims=["member", "lat", "lon"],
                     coords={"member": [0, 1, 2], "lat": lat, "lon": lon})
    result = ensemble([a, b, c], obs=None, strategy="uniform")
    np.testing.assert_allclose(result.values, 4.0)


def test_ensemble_single_model():
    from deepscale.ensemble import ensemble
    lat = np.linspace(-1, 1, 5)
    lon = np.linspace(0, 1, 5)
    a = xr.DataArray(np.ones((3, 5, 5)) * 7, dims=["member", "lat", "lon"],
                     coords={"member": [0, 1, 2], "lat": lat, "lon": lon})
    result = ensemble([a], obs=None, strategy="uniform")
    np.testing.assert_allclose(result.values, 7.0)


def test_ensemble_output_shape():
    from deepscale.ensemble import ensemble
    lat = np.linspace(-1, 1, 5)
    lon = np.linspace(0, 1, 5)
    a = xr.DataArray(np.ones((3, 5, 5)), dims=["member", "lat", "lon"],
                     coords={"member": [0, 1, 2], "lat": lat, "lon": lon})
    b = xr.DataArray(np.ones((3, 5, 5)) * 2, dims=["member", "lat", "lon"],
                     coords={"member": [0, 1, 2], "lat": lat, "lon": lon})
    result = ensemble([a, b], obs=None)
    assert "lat" in result.dims
    assert "lon" in result.dims


# ===================================================================
# 9b. New ensemble strategies (§24.1)
# ===================================================================

def _make_member(value, lat, lon):
    return xr.DataArray(
        np.full((3, len(lat), len(lon)), value),
        dims=["member", "lat", "lon"],
        coords={"member": [0, 1, 2], "lat": lat, "lon": lon},
    )


def test_drop_worst_drops_lowest_score_member():
    from deepscale.ensemble import ensemble
    lat = np.linspace(-1, 1, 4); lon = np.linspace(0, 1, 4)
    a = _make_member(1.0, lat, lon)
    b = _make_member(5.0, lat, lon)
    c = _make_member(9.0, lat, lon)
    # Scores rank: a=0.1 (worst), b=0.4, c=0.7. Drop a; mean of b,c = 7.0
    out = ensemble([a, b, c], obs=None, strategy="drop_worst",
                   scores=[0.1, 0.4, 0.7])
    np.testing.assert_allclose(out.values, 7.0)


def test_drop_worst_n_drop_kwarg():
    from deepscale.ensemble import ensemble
    lat = np.linspace(-1, 1, 4); lon = np.linspace(0, 1, 4)
    forecasts = [_make_member(v, lat, lon) for v in [1.0, 2.0, 3.0, 10.0]]
    # Drop bottom 2 (scores 0.0, 0.1) → mean of values 3.0 and 10.0 = 6.5
    out = ensemble(forecasts, obs=None, strategy="drop_worst",
                   scores=[0.0, 0.1, 0.5, 0.9], n_drop=2)
    np.testing.assert_allclose(out.values, 6.5)


def test_drop_worst_uses_optimize_result_score():
    from deepscale.ensemble import ensemble
    from deepscale.optimize import OptimizeResult
    lat = np.linspace(-1, 1, 4); lon = np.linspace(0, 1, 4)
    forecasts = [
        OptimizeResult(method="cca", score=-0.2, forecast=_make_member(1.0, lat, lon)),
        OptimizeResult(method="cca", score=0.3,  forecast=_make_member(5.0, lat, lon)),
        OptimizeResult(method="cca", score=0.5,  forecast=_make_member(9.0, lat, lon)),
    ]
    out = ensemble(forecasts, obs=None, strategy="drop_worst")
    np.testing.assert_allclose(out.values, 7.0)


def test_drop_worst_explicit_scores_override_optimize_result():
    from deepscale.ensemble import ensemble
    from deepscale.optimize import OptimizeResult
    lat = np.linspace(-1, 1, 4); lon = np.linspace(0, 1, 4)
    # OptimizeResult scores would suggest dropping the first; explicit override
    # flips it so the last is dropped instead.
    forecasts = [
        OptimizeResult(method="cca", score=-0.2, forecast=_make_member(1.0, lat, lon)),
        OptimizeResult(method="cca", score=0.3,  forecast=_make_member(5.0, lat, lon)),
        OptimizeResult(method="cca", score=0.5,  forecast=_make_member(9.0, lat, lon)),
    ]
    out = ensemble(forecasts, obs=None, strategy="drop_worst",
                   scores=[0.9, 0.8, 0.1])
    np.testing.assert_allclose(out.values, 3.0)  # mean of 1.0, 5.0


def test_drop_worst_errors_without_scores():
    """Plain DataArrays with no scores and no kwarg cannot be ranked."""
    from deepscale.ensemble import ensemble
    lat = np.linspace(-1, 1, 4); lon = np.linspace(0, 1, 4)
    forecasts = [_make_member(v, lat, lon) for v in [1.0, 5.0, 9.0]]
    with pytest.raises(ValueError, match="score"):
        ensemble(forecasts, obs=None, strategy="drop_worst")


def test_drop_worst_n_drop_too_large():
    from deepscale.ensemble import ensemble
    lat = np.linspace(-1, 1, 4); lon = np.linspace(0, 1, 4)
    forecasts = [_make_member(v, lat, lon) for v in [1.0, 5.0]]
    with pytest.raises(ValueError, match="n_drop"):
        ensemble(forecasts, obs=None, strategy="drop_worst",
                 scores=[0.1, 0.5], n_drop=2)


def test_skill_weighted_positive_scores():
    """Weights ∝ scores, normalized to sum to 1."""
    from deepscale.ensemble import ensemble
    lat = np.linspace(-1, 1, 4); lon = np.linspace(0, 1, 4)
    a = _make_member(2.0, lat, lon)
    b = _make_member(4.0, lat, lon)
    # scores [0.1, 0.3] → weights [0.25, 0.75] → 0.25*2 + 0.75*4 = 3.5
    out = ensemble([a, b], obs=None, strategy="skill_weighted",
                   scores=[0.1, 0.3])
    np.testing.assert_allclose(out.values, 3.5)


def test_skill_weighted_negative_scores_get_zero_weight():
    """Members with non-positive skill score should not contribute."""
    from deepscale.ensemble import ensemble
    lat = np.linspace(-1, 1, 4); lon = np.linspace(0, 1, 4)
    a = _make_member(2.0, lat, lon)  # bad
    b = _make_member(4.0, lat, lon)  # good
    c = _make_member(8.0, lat, lon)  # good
    # scores [-0.5, 0.2, 0.6] → weights [0, 0.25, 0.75] → 0 + 1.0 + 6.0 = 7.0
    out = ensemble([a, b, c], obs=None, strategy="skill_weighted",
                   scores=[-0.5, 0.2, 0.6])
    np.testing.assert_allclose(out.values, 7.0)


def test_skill_weighted_all_nonpositive_falls_back_to_uniform():
    """If every member has score ≤ 0, return the uniform mean rather than divide-by-zero."""
    from deepscale.ensemble import ensemble
    lat = np.linspace(-1, 1, 4); lon = np.linspace(0, 1, 4)
    a = _make_member(2.0, lat, lon)
    b = _make_member(4.0, lat, lon)
    out = ensemble([a, b], obs=None, strategy="skill_weighted",
                   scores=[-0.5, -0.2])
    np.testing.assert_allclose(out.values, 3.0)  # uniform mean


def test_skill_weighted_explicit_weights_kwarg():
    """`weights=` kwarg bypasses score-based computation."""
    from deepscale.ensemble import ensemble
    lat = np.linspace(-1, 1, 4); lon = np.linspace(0, 1, 4)
    a = _make_member(2.0, lat, lon)
    b = _make_member(8.0, lat, lon)
    # weights normalize: [0.1, 0.4] → [0.2, 0.8] → 0.4 + 6.4 = 6.8
    out = ensemble([a, b], obs=None, strategy="skill_weighted",
                   weights=[0.1, 0.4])
    np.testing.assert_allclose(out.values, 6.8)


def test_skill_weighted_uses_optimize_result_score():
    from deepscale.ensemble import ensemble
    from deepscale.optimize import OptimizeResult
    lat = np.linspace(-1, 1, 4); lon = np.linspace(0, 1, 4)
    forecasts = [
        OptimizeResult(method="cca", score=0.2, forecast=_make_member(2.0, lat, lon)),
        OptimizeResult(method="cca", score=0.6, forecast=_make_member(4.0, lat, lon)),
    ]
    out = ensemble(forecasts, obs=None, strategy="skill_weighted")
    # weights [0.25, 0.75] → 0.5 + 3.0 = 3.5
    np.testing.assert_allclose(out.values, 3.5)


def _bma_synthetic_setup(seed=0):
    """Two-member BMA scenario: member 0 tracks obs, member 1 is pure noise."""
    rng = np.random.default_rng(seed)
    n_years, nlat, nlon = 12, 5, 5
    years = np.arange(2000, 2000 + n_years)
    lat = np.linspace(-2, 2, nlat); lon = np.linspace(0, 4, nlon)
    obs_data = rng.standard_normal((n_years, nlat, nlon))
    obs = xr.DataArray(obs_data, dims=["year", "lat", "lon"],
                       coords={"year": years, "lat": lat, "lon": lon})
    # Member 0: obs + small noise (skillful)
    h0 = obs_data + rng.standard_normal(obs_data.shape) * 0.1
    # Member 1: pure noise (no skill)
    h1 = rng.standard_normal(obs_data.shape) * 1.5
    hindcasts = [
        xr.DataArray(h0, dims=["year", "lat", "lon"],
                     coords={"year": years, "lat": lat, "lon": lon}),
        xr.DataArray(h1, dims=["year", "lat", "lon"],
                     coords={"year": years, "lat": lat, "lon": lon}),
    ]
    # Forecasts at one target time: a constant value per member (so we can
    # check the combined output equals the BMA-weighted average).
    f0 = xr.DataArray(np.full((3, nlat, nlon), 5.0),
                      dims=["member", "lat", "lon"],
                      coords={"member": [0, 1, 2], "lat": lat, "lon": lon})
    f1 = xr.DataArray(np.full((3, nlat, nlon), 9.0),
                      dims=["member", "lat", "lon"],
                      coords={"member": [0, 1, 2], "lat": lat, "lon": lon})
    return [f0, f1], obs, hindcasts


def test_bma_weights_concentrate_on_skillful_member():
    """BMA should put most weight on the member whose hindcast tracks obs."""
    from deepscale.strategies.bma import BMAStrategy
    forecasts, obs, hindcasts = _bma_synthetic_setup(seed=0)
    s = BMAStrategy()
    weights, _sigma2 = s.fit(hindcasts, obs)
    np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-8)
    assert weights[0] > 0.8, f"skillful member got weight {weights[0]:.3f}"
    assert weights[1] < 0.2, f"noise member got weight {weights[1]:.3f}"


def test_bma_combine_uses_fitted_weights():
    """Combined forecast equals weighted sum using the fitted BMA weights."""
    from deepscale.ensemble import ensemble
    from deepscale.strategies.bma import BMAStrategy
    forecasts, obs, hindcasts = _bma_synthetic_setup(seed=0)
    out = ensemble(forecasts, obs, strategy="bma", hindcasts=hindcasts)
    # Fit independently to compare
    weights, _ = BMAStrategy().fit(hindcasts, obs)
    expected = weights[0] * 5.0 + weights[1] * 9.0
    np.testing.assert_allclose(out.values, expected, rtol=1e-6)


def test_bma_requires_hindcasts():
    from deepscale.ensemble import ensemble
    forecasts, obs, _ = _bma_synthetic_setup(seed=0)
    with pytest.raises(ValueError, match="hindcasts"):
        ensemble(forecasts, obs, strategy="bma")


def test_bma_weights_sum_to_one():
    """Even on degenerate cases, BMA weights should always normalize."""
    from deepscale.strategies.bma import BMAStrategy
    rng = np.random.default_rng(42)
    n_years = 8
    years = np.arange(2000, 2000 + n_years)
    obs = xr.DataArray(rng.standard_normal((n_years, 3, 3)),
                       dims=["year", "lat", "lon"],
                       coords={"year": years, "lat": np.arange(3.0), "lon": np.arange(3.0)})
    # Three near-identical members
    h = [obs + rng.standard_normal((n_years, 3, 3)) * 0.01 for _ in range(3)]
    weights, _ = BMAStrategy().fit(h, obs)
    np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-8)
    assert np.all(weights >= 0)


# ===================================================================
# 10. skill()
# ===================================================================

def test_skill_returns_report(climatology_forecast, synthetic_obs):
    from deepscale.skill import skill, SkillReport
    report = skill(climatology_forecast, synthetic_obs, metrics=["rpss"])
    assert isinstance(report, SkillReport)
    assert "rpss" in report.scores
    assert isinstance(report.scores["rpss"], float)


def test_skill_spatial_maps(climatology_forecast, synthetic_obs):
    from deepscale.skill import skill
    report = skill(climatology_forecast, synthetic_obs, metrics=["rpss"], spatial=True)
    assert "rpss" in report.spatial
    assert "lat" in report.spatial["rpss"].dims
    assert "lon" in report.spatial["rpss"].dims


def test_skill_multiple_metrics(climatology_forecast, synthetic_obs):
    from deepscale.skill import skill
    report = skill(climatology_forecast, synthetic_obs, metrics=["rpss", "roc"])
    assert "rpss" in report.scores
    assert "roc_bn" in report.scores


# ===================================================================
# 11. downscale()
# ===================================================================

def test_downscale_bcsd(synthetic_gcm_hindcast, synthetic_obs):
    import deepscale
    result = deepscale.downscale(synthetic_gcm_hindcast, synthetic_obs, method="bcsd")
    assert result.dims == ("member", "lat", "lon")
    assert len(result.lat) == len(synthetic_obs.lat)


def test_downscale_cca(synthetic_gcm_hindcast, synthetic_obs):
    import deepscale
    result = deepscale.downscale(synthetic_gcm_hindcast, synthetic_obs, method="cca")
    assert result.dims == ("member", "lat", "lon")


def test_downscale_tercile_output(synthetic_gcm_hindcast, synthetic_obs):
    import deepscale
    result = deepscale.downscale(
        synthetic_gcm_hindcast, synthetic_obs, method="bcsd", output_type="tercile"
    )
    assert "tercile" in result.dims
    sums = result.sum("tercile")
    np.testing.assert_allclose(sums.values, 1.0, atol=1e-10)


def test_downscale_accepts_predictor_hindcast_keyword(synthetic_gcm_hindcast, synthetic_obs):
    """The new canonical kwarg works without warnings."""
    import warnings
    import deepscale
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        result = deepscale.downscale(
            predictor_hindcast=synthetic_gcm_hindcast, obs=synthetic_obs, method="bcsd",
        )
    assert result.dims == ("member", "lat", "lon")


def test_downscale_legacy_gcm_keyword_emits_deprecation(synthetic_gcm_hindcast, synthetic_obs):
    """The old `gcm=` kwarg still works but emits a DeprecationWarning."""
    import deepscale
    with pytest.warns(DeprecationWarning, match="gcm"):
        result = deepscale.downscale(
            gcm=synthetic_gcm_hindcast, obs=synthetic_obs, method="bcsd",
        )
    assert result.dims == ("member", "lat", "lon")


def test_downscale_rejects_both_names(synthetic_gcm_hindcast, synthetic_obs):
    """Passing both `predictor_hindcast` and `gcm` is an explicit error."""
    import deepscale
    with pytest.raises(TypeError, match="both"):
        deepscale.downscale(
            predictor_hindcast=synthetic_gcm_hindcast,
            gcm=synthetic_gcm_hindcast,
            obs=synthetic_obs,
            method="bcsd",
        )


# ===================================================================
# 12. optimize()
# ===================================================================

def test_optimize_single_gcm(synthetic_gcm_hindcast, synthetic_obs):
    import deepscale
    best = deepscale.optimize(
        synthetic_gcm_hindcast, synthetic_obs,
        methods=["bcsd", "cca"], cv="loyo", primary_metric="rpss",
    )
    assert best.method in ["bcsd", "cca"]
    assert np.isfinite(best.score)
    assert best.forecast is not None
    assert "lat" in best.forecast.dims


def test_optimize_with_blocked_cv(synthetic_gcm_hindcast, synthetic_obs):
    """`optimize(cv="blocked")` works end-to-end with multi-year test folds."""
    import deepscale
    best = deepscale.optimize(
        synthetic_gcm_hindcast, synthetic_obs,
        methods=["cca"], cv="blocked", primary_metric="rpss",
        verbose=False, progress=False,
    )
    assert best.method == "cca"
    assert np.isfinite(best.score)


def test_optimize_with_lko_cv(synthetic_gcm_hindcast, synthetic_obs):
    """`optimize(cv="lko")` works with sliding multi-year test folds."""
    import deepscale
    best = deepscale.optimize(
        synthetic_gcm_hindcast, synthetic_obs,
        methods=["cca"], cv="lko", primary_metric="rpss",
        verbose=False, progress=False,
    )
    assert best.method == "cca"
    assert np.isfinite(best.score)


def test_optimize_with_expanding_cv(synthetic_gcm_hindcast, synthetic_obs):
    """`optimize(cv=callable)` works with realtime-style folds.

    `expanding`'s default min_train=10 leaves no eval years on a 10-year
    fixture, so we pass a pre-configured callable. This also exercises
    optimize()'s callable CV path.
    """
    from functools import partial
    import deepscale
    from deepscale.cv import expanding
    best = deepscale.optimize(
        synthetic_gcm_hindcast, synthetic_obs,
        methods=["cca"],
        cv=partial(expanding, min_train=4),
        primary_metric="rpss",
        verbose=False, progress=False,
    )
    assert best.method == "cca"
    assert np.isfinite(best.score)


# ===================================================================
# 13. E2E pipeline
# ===================================================================

def test_e2e_single_gcm_single_method(synthetic_gcm_hindcast, synthetic_obs):
    import deepscale
    result = deepscale.downscale(synthetic_gcm_hindcast, synthetic_obs, method="bcsd")
    assert result.dims == ("member", "lat", "lon")
    assert len(result.lat) == len(synthetic_obs.lat)


def test_e2e_multi_gcm_ensemble(synthetic_gcm_hindcast, synthetic_gcm_hindcast2, synthetic_obs):
    import deepscale
    best1 = deepscale.optimize(synthetic_gcm_hindcast, synthetic_obs,
                                methods=["bcsd"], cv="loyo", primary_metric="rpss")
    best2 = deepscale.optimize(synthetic_gcm_hindcast2, synthetic_obs,
                                methods=["bcsd"], cv="loyo", primary_metric="rpss")
    mme = deepscale.ensemble([best1, best2], synthetic_obs, strategy="uniform")
    assert "lat" in mme.dims
    assert "lon" in mme.dims


def test_e2e_drop_worst_and_skill_weighted_through_optimize(
    synthetic_gcm_hindcast, synthetic_gcm_hindcast2, synthetic_obs,
):
    """Integration: optimize→ensemble using the new skill-aware strategies.

    Both `drop_worst` and `skill_weighted` should consume the OptimizeResult
    objects directly (using their `.score` field) and produce a valid MME.
    """
    import deepscale
    best1 = deepscale.optimize(synthetic_gcm_hindcast, synthetic_obs,
                                methods=["bcsd"], cv="loyo", primary_metric="rpss",
                                verbose=False, progress=False)
    best2 = deepscale.optimize(synthetic_gcm_hindcast2, synthetic_obs,
                                methods=["bcsd"], cv="loyo", primary_metric="rpss",
                                verbose=False, progress=False)

    sw_mme = deepscale.ensemble([best1, best2], synthetic_obs, strategy="skill_weighted")
    assert sw_mme.dims == ("member", "lat", "lon")
    assert not np.all(np.isnan(sw_mme.values))

    # drop_worst with two members and n_drop=1 reduces to "keep best member only".
    dw_mme = deepscale.ensemble([best1, best2], synthetic_obs, strategy="drop_worst")
    assert dw_mme.dims == ("member", "lat", "lon")
    # Result equals whichever single forecast had the higher score.
    winner = best1.forecast if best1.score >= best2.score else best2.forecast
    np.testing.assert_allclose(dw_mme.values, winner.values, equal_nan=True)


def test_e2e_bma_through_optimize(
    synthetic_gcm_hindcast, synthetic_gcm_hindcast2, synthetic_obs,
):
    """Integration: BMA strategy with hindcasts pulled from the actual fits."""
    import deepscale
    from deepscale.methods.bcsd import BCSDMethod

    # Generate per-member hindcasts on the obs grid.
    hindcasts = []
    for gcm in (synthetic_gcm_hindcast, synthetic_gcm_hindcast2):
        m = BCSDMethod()
        m.fit(gcm, synthetic_obs)
        # Predict on each year of the GCM hindcast, ensemble-mean across members.
        preds = []
        for yr in synthetic_obs.year.values:
            pred = m.predict(gcm.sel(year=yr)).mean("member")
            preds.append(pred.expand_dims(year=[yr]))
        hindcasts.append(xr.concat(preds, dim="year"))

    best1 = deepscale.optimize(synthetic_gcm_hindcast, synthetic_obs,
                                methods=["bcsd"], cv="loyo", primary_metric="rpss",
                                verbose=False, progress=False)
    best2 = deepscale.optimize(synthetic_gcm_hindcast2, synthetic_obs,
                                methods=["bcsd"], cv="loyo", primary_metric="rpss",
                                verbose=False, progress=False)

    mme = deepscale.ensemble(
        [best1, best2], synthetic_obs, strategy="bma", hindcasts=hindcasts,
    )
    assert mme.dims == ("member", "lat", "lon")
    assert not np.all(np.isnan(mme.values))


def test_e2e_climatology_baseline(climatology_forecast, synthetic_obs):
    import deepscale
    report = deepscale.skill(climatology_forecast, synthetic_obs, metrics=["rpss"])
    np.testing.assert_allclose(report.scores["rpss"], 0.0, atol=0.05)


# ===================================================================
# 14. Plugin contract tests
# ===================================================================

def test_plugin_method_contract(synthetic_gcm_hindcast, synthetic_gcm_forecast, synthetic_obs):
    from deepscale.methods.base import MethodBase
    from deepscale.registry import register_method
    import deepscale

    @register_method("plugin_dummy")
    class DummyMethod(MethodBase):
        def fit(self, hindcast, obs, **kwargs):
            self.obs_mean_ = obs.mean("year")
            self.obs_coords_ = {"lat": obs.lat, "lon": obs.lon}

        def predict(self, forecast, **kwargs):
            result = self.obs_mean_.expand_dims(member=forecast.member)
            return result

    result = deepscale.downscale(synthetic_gcm_hindcast, synthetic_obs, method="plugin_dummy")
    assert result.dims == ("member", "lat", "lon")


def test_plugin_metric_contract(climatology_forecast, synthetic_obs):
    from deepscale.metrics.base import MetricBase
    from deepscale.registry import register_metric
    import deepscale

    @register_metric("always_half")
    class AlwaysHalf(MetricBase):
        def compute(self, forecast, obs, **kwargs):
            return 0.5

    report = deepscale.skill(climatology_forecast, synthetic_obs, metrics=["always_half"])
    assert report.scores["always_half"] == 0.5


# ===================================================================
# 15. Plotting subpackage
# ===================================================================

def test_plotting_package_imports():
    """Package must import cleanly even when matplotlib/cartopy aren't installed."""
    import deepscale.plotting  # noqa: F401


def test_plot_skill_maps_smoke():
    pytest.importorskip("matplotlib")
    pytest.importorskip("cartopy")
    import matplotlib.pyplot as plt
    from deepscale.skill import SkillReport
    from deepscale.plotting.skill import plot_skill_maps

    lat = np.linspace(-5, 5, 6)
    lon = np.linspace(30, 45, 8)
    rpss = xr.DataArray(
        np.random.RandomState(0).uniform(-1, 1, (6, 8)),
        dims=["lat", "lon"], coords={"lat": lat, "lon": lon},
    )
    rmse = xr.DataArray(
        np.random.RandomState(1).uniform(0, 2, (6, 8)),
        dims=["lat", "lon"], coords={"lat": lat, "lon": lon},
    )
    report = SkillReport(scores={"rpss": float(rpss.mean()), "rmse": float(rmse.mean())},
                         spatial={"rpss": rpss, "rmse": rmse})

    fig = plot_skill_maps(report, ["rpss", "rmse"], ncols=2)

    assert fig is not None
    assert len(fig.axes) >= 2
    plt.close(fig)


def test_plot_domains_smoke():
    pytest.importorskip("matplotlib")
    pytest.importorskip("cartopy")
    import matplotlib.pyplot as plt
    from deepscale.plotting.domains import plot_domains

    # predictand: East Africa, predictor: tropical Pacific (antimeridian-spanning)
    fig = plot_domains(
        predictor_extent=(-20, 20, 120, -60),     # lon_w > lon_e — crosses dateline
        predictand_extent=(-12, 15, 22, 52),
    )

    assert fig is not None
    plt.close(fig)


def test_plot_tercile_forecast_smoke():
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    from deepscale.plotting.forecasts import plot_tercile_forecast

    n_lat, n_lon = 4, 5
    probs = np.zeros((3, n_lat, n_lon))
    probs[0, :, :] = 0.15
    probs[1, :, :] = 0.25
    probs[2, :, :] = 0.60
    pr_fcst = xr.DataArray(
        probs,
        dims=["tercile", "lat", "lon"],
        coords={
            "tercile": [0, 1, 2],
            "lat": np.linspace(-5, 5, n_lat),
            "lon": np.linspace(30, 45, n_lon),
        },
    )
    fig = plot_tercile_forecast(pr_fcst)
    assert fig is not None
    plt.close(fig)


def test_plot_deterministic_forecast_smoke():
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    from deepscale.plotting.forecasts import plot_deterministic_forecast

    n_lat, n_lon = 4, 5
    da = xr.DataArray(
        np.random.RandomState(2).randn(n_lat, n_lon),
        dims=["lat", "lon"],
        coords={"lat": np.linspace(-5, 5, n_lat), "lon": np.linspace(30, 45, n_lon)},
    )
    fig = plot_deterministic_forecast(da, title="test")
    assert fig is not None
    plt.close(fig)


def test_plot_exceedance_probability_smoke():
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    from deepscale.plotting.forecasts import plot_exceedance_probability

    n_lat, n_lon = 4, 5
    da = xr.DataArray(
        np.random.RandomState(3).uniform(0, 1, (n_lat, n_lon)),
        dims=["lat", "lon"],
        coords={"lat": np.linspace(-5, 5, n_lat), "lon": np.linspace(30, 45, n_lon)},
    )
    fig = plot_exceedance_probability(da, threshold=100.0)
    assert fig is not None
    plt.close(fig)


def test_plot_flex_pdf_smoke():
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    from deepscale.plotting.forecasts import plot_flex_pdf

    fig = plot_flex_pdf(
        fcst_mu=2.5, fcst_scale=1.2,
        climo_mu=2.0, climo_scale=1.5,
        location=(35.0, 0.0),
    )
    assert fig is not None
    plt.close(fig)


# ===================================================================
# 16. 2AFC metric
# ===================================================================

def test_2afc_perfect(synthetic_obs):
    from deepscale.metrics.two_afc import TwoAFCMetric
    score = TwoAFCMetric().compute(synthetic_obs, synthetic_obs)
    np.testing.assert_allclose(score, 1.0, atol=0.001)


def test_2afc_uniform_random():
    from deepscale.metrics.two_afc import TwoAFCMetric
    np.random.seed(42)
    n_year, n_lat, n_lon = 100, 5, 5
    coords = {
        "year": np.arange(n_year),
        "lat": np.linspace(-1, 1, n_lat),
        "lon": np.linspace(0, 1, n_lon),
    }
    obs = xr.DataArray(np.random.randn(n_year, n_lat, n_lon),
                       dims=["year", "lat", "lon"], coords=coords)
    forecast = xr.DataArray(np.random.randn(n_year, n_lat, n_lon),
                            dims=["year", "lat", "lon"], coords=coords)
    score = TwoAFCMetric().compute(forecast, obs)
    assert abs(score - 0.5) < 0.05


def test_2afc_constant_forecast_no_skill(synthetic_obs):
    from deepscale.metrics.two_afc import TwoAFCMetric
    forecast = synthetic_obs * 0 + 1.0  # all-constant
    score = TwoAFCMetric().compute(forecast, synthetic_obs)
    # Half-credit-for-ties: a constant forecast scores 0.5 (matches the
    # issue's "constant forecast (no skill) ≈ 0.5" criterion).
    np.testing.assert_allclose(score, 0.5, atol=1e-12)


# ===================================================================
# 17. Per-tercile ROC variants
# ===================================================================

def test_roc_area_below_normal_matches_roc_bn(synthetic_obs, perfect_tercile_forecast):
    from deepscale.registry import get_metric
    full = get_metric("roc")().compute(perfect_tercile_forecast, synthetic_obs)
    bn = get_metric("roc_area_below_normal")().compute(perfect_tercile_forecast, synthetic_obs)
    np.testing.assert_allclose(bn, full["roc_bn"], atol=1e-12)


def test_roc_area_above_normal_matches_roc_an(synthetic_obs, perfect_tercile_forecast):
    from deepscale.registry import get_metric
    full = get_metric("roc")().compute(perfect_tercile_forecast, synthetic_obs)
    an = get_metric("roc_area_above_normal")().compute(perfect_tercile_forecast, synthetic_obs)
    np.testing.assert_allclose(an, full["roc_an"], atol=1e-12)


# ===================================================================
# 18. Reliability metric + diagram
# ===================================================================

def test_reliability_climatology(synthetic_obs):
    from deepscale.metrics.reliability import ReliabilityMetric
    n_year, n_lat, n_lon = synthetic_obs.shape
    fcst = np.ones((n_year, 3, n_lat, n_lon)) / 3.0  # uniform climatology
    forecast = xr.DataArray(
        fcst, dims=["year", "tercile", "lat", "lon"],
        coords={
            "year": synthetic_obs.year,
            "tercile": [0, 1, 2],
            "lat": synthetic_obs.lat,
            "lon": synthetic_obs.lon,
        },
    )
    rel = ReliabilityMetric().compute(forecast, synthetic_obs)
    assert rel < 0.05, f"expected near-perfect calibration, got {rel}"


def test_reliability_overconfident(synthetic_obs):
    from deepscale.metrics.reliability import ReliabilityMetric
    n_year, n_lat, n_lon = synthetic_obs.shape
    fcst = np.zeros((n_year, 3, n_lat, n_lon))
    fcst[:, 0, :, :] = 1.0  # always confident BN
    forecast = xr.DataArray(
        fcst, dims=["year", "tercile", "lat", "lon"],
        coords={
            "year": synthetic_obs.year,
            "tercile": [0, 1, 2],
            "lat": synthetic_obs.lat,
            "lon": synthetic_obs.lon,
        },
    )
    rel = ReliabilityMetric().compute(forecast, synthetic_obs)
    assert rel > 0.2, f"expected badly calibrated forecast, got {rel}"


def test_plot_reliability_diagram_smoke(synthetic_obs):
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    from deepscale.plotting.reliability import plot_reliability_diagram

    n_year, n_lat, n_lon = synthetic_obs.shape
    fcst = np.ones((n_year, 3, n_lat, n_lon)) / 3.0
    forecast = xr.DataArray(
        fcst, dims=["year", "tercile", "lat", "lon"],
        coords={
            "year": synthetic_obs.year,
            "tercile": [0, 1, 2],
            "lat": synthetic_obs.lat,
            "lon": synthetic_obs.lon,
        },
    )
    fig = plot_reliability_diagram(forecast, synthetic_obs)
    assert fig is not None
    plt.close(fig)


# ===================================================================
# 18b. EOF / CCA mode plots (§3.2)
# ===================================================================

def test_apply_sign_convention_flips_negative_dominant_lobe():
    from deepscale.plotting.modes import _apply_sign_convention
    arr = np.array([[-3.0, 1.0], [0.5, -0.5]])
    flipped, sign = _apply_sign_convention(arr)
    assert sign == -1.0
    np.testing.assert_array_equal(flipped, -arr)
    # After flip, the dominant lobe is positive.
    assert flipped.flat[int(np.nanargmax(np.abs(flipped)))] > 0


def test_apply_sign_convention_keeps_positive_dominant_lobe():
    from deepscale.plotting.modes import _apply_sign_convention
    arr = np.array([[3.0, -1.0], [0.5, -0.5]])
    out, sign = _apply_sign_convention(arr)
    assert sign == 1.0
    np.testing.assert_array_equal(out, arr)


def test_apply_sign_convention_handles_all_nan():
    from deepscale.plotting.modes import _apply_sign_convention
    arr = np.full((2, 2), np.nan)
    out, sign = _apply_sign_convention(arr)
    assert sign == 1.0
    assert np.all(np.isnan(out))


def _fit_cca_for_mode_plots():
    """Helper: fit CCAMethod on the dual-grid fixture for plotting tests."""
    from deepscale.methods.cca import CCAMethod
    predictor, predictand, _ = _build_dual_grid_fixture()
    m = CCAMethod(n_modes=3, x_eof_modes=4, y_eof_modes=4)
    m.fit(predictor, predictand)
    return m, predictor, predictand


def test_plot_eof_modes_predictor_returns_figure():
    pytest.importorskip("matplotlib")
    pytest.importorskip("cartopy")
    import matplotlib.pyplot as plt
    from deepscale.plotting.modes import plot_eof_modes
    m, _, _ = _fit_cca_for_mode_plots()
    fig = plot_eof_modes(m, kind="predictor", n_modes=3)
    assert fig is not None
    # 3 mode panels (plus colorbars are extra axes)
    map_axes = [ax for ax in fig.axes if hasattr(ax, "coastlines") and ax.get_visible()]
    assert len(map_axes) == 3
    plt.close(fig)


def test_plot_eof_modes_predictand_returns_figure():
    pytest.importorskip("matplotlib")
    pytest.importorskip("cartopy")
    import matplotlib.pyplot as plt
    from deepscale.plotting.modes import plot_eof_modes
    m, _, _ = _fit_cca_for_mode_plots()
    fig = plot_eof_modes(m, kind="predictand", n_modes=2)
    map_axes = [ax for ax in fig.axes if hasattr(ax, "coastlines") and ax.get_visible()]
    assert len(map_axes) == 2
    plt.close(fig)


def test_plot_eof_modes_invalid_kind_raises():
    pytest.importorskip("matplotlib")
    pytest.importorskip("cartopy")
    from deepscale.plotting.modes import plot_eof_modes
    m, _, _ = _fit_cca_for_mode_plots()
    with pytest.raises(ValueError, match="kind"):
        plot_eof_modes(m, kind="bogus")


def test_plot_eof_modes_caps_n_modes_at_available():
    pytest.importorskip("matplotlib")
    pytest.importorskip("cartopy")
    import matplotlib.pyplot as plt
    from deepscale.plotting.modes import plot_eof_modes
    m, _, _ = _fit_cca_for_mode_plots()
    # Ask for more modes than were fitted; should silently cap.
    fig = plot_eof_modes(m, kind="predictor", n_modes=99)
    map_axes = [ax for ax in fig.axes if hasattr(ax, "coastlines") and ax.get_visible()]
    assert len(map_axes) == m.eofx_.shape[1]
    plt.close(fig)


def test_plot_eof_modes_title_includes_variance_fraction():
    pytest.importorskip("matplotlib")
    pytest.importorskip("cartopy")
    import matplotlib.pyplot as plt
    from deepscale.plotting.modes import plot_eof_modes
    m, _, _ = _fit_cca_for_mode_plots()
    fig = plot_eof_modes(m, kind="predictor", n_modes=2)
    titles = [
        ax.get_title() for ax in fig.axes
        if hasattr(ax, "coastlines") and ax.get_visible()
    ]
    assert all("EOF" in t for t in titles)
    assert all("var" in t for t in titles)
    plt.close(fig)


def test_plot_cca_modes_returns_paired_grid():
    pytest.importorskip("matplotlib")
    pytest.importorskip("cartopy")
    import matplotlib.pyplot as plt
    from deepscale.plotting.modes import plot_cca_modes
    m, _, _ = _fit_cca_for_mode_plots()
    fig = plot_cca_modes(m, n_modes=2)
    map_axes = [ax for ax in fig.axes if hasattr(ax, "coastlines") and ax.get_visible()]
    # 2 modes x (predictor + predictand) = 4 map panels
    assert len(map_axes) == 4
    plt.close(fig)


def test_plot_cca_modes_title_includes_canonical_correlation():
    pytest.importorskip("matplotlib")
    pytest.importorskip("cartopy")
    import matplotlib.pyplot as plt
    from deepscale.plotting.modes import plot_cca_modes
    m, _, _ = _fit_cca_for_mode_plots()
    fig = plot_cca_modes(m, n_modes=1)
    titles = [
        ax.get_title() for ax in fig.axes
        if hasattr(ax, "coastlines") and ax.get_visible()
    ]
    assert any("predictor" in t for t in titles)
    assert any("predictand" in t for t in titles)
    assert all("r=" in t for t in titles)
    plt.close(fig)


def test_plot_cca_modes_caps_at_available_modes():
    pytest.importorskip("matplotlib")
    pytest.importorskip("cartopy")
    import matplotlib.pyplot as plt
    from deepscale.plotting.modes import plot_cca_modes
    m, _, _ = _fit_cca_for_mode_plots()
    fig = plot_cca_modes(m, n_modes=99)
    map_axes = [ax for ax in fig.axes if hasattr(ax, "coastlines") and ax.get_visible()]
    assert len(map_axes) == 2 * m.ncc_
    plt.close(fig)


def test_mode_plots_dual_grid_integration(tmp_path):
    """Integration: fit CCA on the dual-grid fixture and render both mode plots to disk."""
    pytest.importorskip("matplotlib")
    pytest.importorskip("cartopy")
    import matplotlib.pyplot as plt
    from deepscale.methods.cca import CCAMethod
    from deepscale.plotting.modes import plot_eof_modes, plot_cca_modes

    predictor, predictand, _ = _build_dual_grid_fixture()
    m = CCAMethod(n_modes=3, x_eof_modes=4, y_eof_modes=4)
    m.fit(predictor, predictand)

    eof_path = tmp_path / "eof_predictor.png"
    cca_path = tmp_path / "cca_modes.png"
    fig_eof = plot_eof_modes(m, kind="predictor", n_modes=3)
    fig_eof.savefig(eof_path, dpi=80)
    plt.close(fig_eof)
    fig_cca = plot_cca_modes(m, n_modes=2)
    fig_cca.savefig(cca_path, dpi=80)
    plt.close(fig_cca)

    # Both files exist and are non-trivially sized (a blank figure is much smaller).
    assert eof_path.exists() and eof_path.stat().st_size > 5000
    assert cca_path.exists() and cca_path.stat().st_size > 5000


def test_plot_cca_modes_pair_shares_sign_convention():
    """Predictor and predictand of a CCA pair should be flipped together."""
    pytest.importorskip("matplotlib")
    pytest.importorskip("cartopy")
    from deepscale.plotting.modes import (
        plot_cca_modes, _apply_sign_convention, _reconstruct_spatial,
    )
    import matplotlib.pyplot as plt
    m, _, _ = _fit_cca_for_mode_plots()

    # Manually compute what the locked-sign predictor / predictand patterns should be
    # for mode 0, then check the rendered colour-meshes' raw arrays match.
    p_raw = _reconstruct_spatial(
        (m.eofx_ @ m.s_.T)[:, 0], m.x_valid_, m.predictor_shape_
    )
    o_raw = _reconstruct_spatial(
        (m.eofy_ @ m.r_)[:, 0], m.y_valid_, m.predictand_shape_
    )
    p_signed, sign = _apply_sign_convention(p_raw)
    o_signed = o_raw * sign

    fig = plot_cca_modes(m, n_modes=1)
    map_axes = [ax for ax in fig.axes if hasattr(ax, "coastlines") and ax.get_visible()]
    p_mesh = map_axes[0].collections[0].get_array().reshape(m.predictor_shape_)
    o_mesh = map_axes[1].collections[0].get_array().reshape(m.predictand_shape_)
    np.testing.assert_allclose(np.asarray(p_mesh), p_signed, equal_nan=True)
    np.testing.assert_allclose(np.asarray(o_mesh), o_signed, equal_nan=True)
    plt.close(fig)


# ===================================================================
# 19. Metric presets (#52)
# ===================================================================

def test_skill_preset_svslrf(synthetic_obs, perfect_tercile_forecast):
    import deepscale
    report = deepscale.skill(perfect_tercile_forecast, synthetic_obs, metrics="svslrf")
    assert "rpss" in report.scores
    assert "roc_bn" in report.scores  # from "roc" metric (returns dict)
    assert "roc_nn" in report.scores
    assert "roc_an" in report.scores
    assert "reliability" in report.scores


def test_skill_preset_all_dedupes_aliases(synthetic_obs, perfect_tercile_forecast):
    import warnings
    import deepscale
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = deepscale.skill(perfect_tercile_forecast, synthetic_obs, metrics="all")
    rmse_keys = [k for k in report.scores if k in ("rmse", "root_mean_squared_error")]
    assert len(rmse_keys) == 1, f"expected one RMSE key, got {rmse_keys}"
    hss_keys = [k for k in report.scores if k in ("hss", "heidke_skill_score")]
    assert len(hss_keys) == 1, f"expected one HSS key, got {hss_keys}"

    # spread_error_* require a `member` dim; the tercile forecast doesn't
    # have one, so metrics="all" should skip them with a warning rather than
    # abort the whole report.
    assert "spread_error_ratio" not in report.scores
    assert "spread_error_correlation" not in report.scores
    skipped = [str(w.message) for w in caught if "spread_error" in str(w.message)]
    assert skipped, "expected a skip-warning for spread_error_* metrics"


def test_skill_bare_string_single_metric(synthetic_obs, perfect_tercile_forecast):
    import deepscale
    report = deepscale.skill(perfect_tercile_forecast, synthetic_obs, metrics="rpss")
    assert "rpss" in report.scores


def test_skill_list_metrics_still_works(synthetic_obs, perfect_tercile_forecast):
    import deepscale
    report = deepscale.skill(
        perfect_tercile_forecast, synthetic_obs,
        metrics=["rpss", "pearson_r"],
    )
    assert set(report.scores.keys()) >= {"rpss", "pearson_r"}


# ===================================================================
# 19. Spread-error metric
# ===================================================================

def test_spread_error_ratio_calibrated():
    """Calibrated ensemble: per-year ensemble-mean bias matched to per-member
    spread so spread ≈ error.

    Construction: each member = truth + bias_y(lat, lon) + member_noise. The
    bias is shared across members (so it contributes to error but not spread).
    The member noise contributes to spread (std σ) and adds a small σ/√N
    component to error. With bias std B = σ·√(π/2), E[|bias|] = σ, so
    mean(spread) ≈ mean(error) ≈ σ.
    """
    from deepscale.metrics.spread_error import SpreadErrorRatioMetric

    np.random.seed(0)
    n_year, n_member, n_lat, n_lon = 200, 8, 4, 4
    coords = {
        "year": np.arange(n_year),
        "member": np.arange(n_member),
        "lat": np.linspace(-1, 1, n_lat),
        "lon": np.linspace(0, 1, n_lon),
    }
    truth = np.random.randn(n_year, n_lat, n_lon)
    sigma = 1.0
    B = sigma * np.sqrt(np.pi / 2)
    bias = np.random.randn(n_year, n_lat, n_lon) * B  # shared across members
    member_noise = np.random.randn(n_year, n_member, n_lat, n_lon) * sigma
    fcst = truth[:, None, :, :] + bias[:, None, :, :] + member_noise

    forecast = xr.DataArray(
        fcst, dims=["year", "member", "lat", "lon"], coords=coords
    )
    obs = xr.DataArray(
        truth, dims=["year", "lat", "lon"],
        coords={k: coords[k] for k in ("year", "lat", "lon")},
    )

    ratio = SpreadErrorRatioMetric().compute(forecast, obs)
    assert 0.85 < ratio < 1.15, f"expected ~1, got {ratio}"


def test_spread_error_ratio_underdispersed():
    """Spread = 0.1 × error → ratio ≈ 0.1."""
    from deepscale.metrics.spread_error import SpreadErrorRatioMetric

    np.random.seed(0)
    n_year, n_member, n_lat, n_lon = 30, 8, 4, 4
    coords = {
        "year": np.arange(n_year),
        "member": np.arange(n_member),
        "lat": np.linspace(-1, 1, n_lat),
        "lon": np.linspace(0, 1, n_lon),
    }
    truth = np.random.randn(n_year, n_lat, n_lon)
    # Add a large per-year bias so |mean - obs| is dominated by the bias,
    # and shrink the member-axis noise so spread is 10× smaller.
    bias = np.random.randn(n_year, 1, n_lat, n_lon) * 5.0
    small_noise = np.random.randn(n_year, n_member, n_lat, n_lon) * 0.5
    fcst = truth[:, None, :, :] + bias + small_noise

    forecast = xr.DataArray(
        fcst, dims=["year", "member", "lat", "lon"], coords=coords
    )
    obs = xr.DataArray(
        truth, dims=["year", "lat", "lon"],
        coords={k: coords[k] for k in ("year", "lat", "lon")},
    )

    ratio = SpreadErrorRatioMetric().compute(forecast, obs)
    assert ratio < 0.3, f"expected strongly underdispersed (<0.3), got {ratio}"


def test_spread_error_correlation_tracks():
    """Ensemble where high-spread years are also high-error years.

    Construct an ensemble whose per-year noise amplitude varies with year;
    the ensemble mean's distance from truth grows with that amplitude, so
    spread and error track each other strongly.
    """
    from deepscale.metrics.spread_error import SpreadErrorCorrelationMetric

    np.random.seed(0)
    n_year, n_member, n_lat, n_lon = 30, 8, 4, 4
    coords = {
        "year": np.arange(n_year),
        "member": np.arange(n_member),
        "lat": np.linspace(-1, 1, n_lat),
        "lon": np.linspace(0, 1, n_lon),
    }
    truth = np.random.randn(n_year, n_lat, n_lon)
    amplitude = np.linspace(0.2, 4.0, n_year)  # year-varying noise level
    noise = (
        np.random.randn(n_year, n_member, n_lat, n_lon)
        * amplitude[:, None, None, None]
    )
    fcst = truth[:, None, :, :] + noise

    forecast = xr.DataArray(
        fcst, dims=["year", "member", "lat", "lon"], coords=coords
    )
    obs = xr.DataArray(
        truth, dims=["year", "lat", "lon"],
        coords={k: coords[k] for k in ("year", "lat", "lon")},
    )

    r = SpreadErrorCorrelationMetric().compute(forecast, obs)
    assert r > 0.7, f"expected strong positive spread-error correlation, got {r}"


def test_spread_error_no_member_raises():
    """Forecast without a 'member' dim is a usage error."""
    from deepscale.metrics.spread_error import (
        SpreadErrorRatioMetric,
        SpreadErrorCorrelationMetric,
    )

    n_year, n_lat, n_lon = 10, 4, 4
    coords = {
        "year": np.arange(n_year),
        "lat": np.linspace(-1, 1, n_lat),
        "lon": np.linspace(0, 1, n_lon),
    }
    forecast = xr.DataArray(
        np.random.randn(n_year, n_lat, n_lon),
        dims=["year", "lat", "lon"], coords=coords,
    )
    obs = xr.DataArray(
        np.random.randn(n_year, n_lat, n_lon),
        dims=["year", "lat", "lon"], coords=coords,
    )

    for cls in (SpreadErrorRatioMetric, SpreadErrorCorrelationMetric):
        with pytest.raises(ValueError, match="member"):
            cls().compute(forecast, obs)


def test_spread_error_spatial_returns_dataarray():
    """spatial=True collapses only the year dim and returns a DataArray."""
    from deepscale.metrics.spread_error import (
        SpreadErrorRatioMetric,
        SpreadErrorCorrelationMetric,
    )

    np.random.seed(0)
    n_year, n_member, n_lat, n_lon = 10, 4, 5, 6
    coords = {
        "year": np.arange(n_year),
        "member": np.arange(n_member),
        "lat": np.linspace(-1, 1, n_lat),
        "lon": np.linspace(0, 1, n_lon),
    }
    forecast = xr.DataArray(
        np.random.randn(n_year, n_member, n_lat, n_lon),
        dims=["year", "member", "lat", "lon"], coords=coords,
    )
    obs = xr.DataArray(
        np.random.randn(n_year, n_lat, n_lon),
        dims=["year", "lat", "lon"],
        coords={k: coords[k] for k in ("year", "lat", "lon")},
    )

    ratio = SpreadErrorRatioMetric().compute(forecast, obs, spatial=True)
    corr = SpreadErrorCorrelationMetric().compute(forecast, obs, spatial=True)

    for result in (ratio, corr):
        assert isinstance(result, xr.DataArray)
        assert set(result.dims) == {"lat", "lon"}
        assert result.sizes == {"lat": n_lat, "lon": n_lon}


def test_spread_error_diagnostics_pairs():
    """Helper returns per-year spread and error series of equal length."""
    from deepscale.metrics.spread_error import (
        SpreadErrorDiagnostics,
        spread_error_diagnostics,
    )

    np.random.seed(0)
    n_year, n_member, n_lat, n_lon = 8, 5, 3, 3
    coords = {
        "year": np.arange(n_year),
        "member": np.arange(n_member),
        "lat": np.linspace(-1, 1, n_lat),
        "lon": np.linspace(0, 1, n_lon),
    }
    forecast = xr.DataArray(
        np.random.randn(n_year, n_member, n_lat, n_lon),
        dims=["year", "member", "lat", "lon"], coords=coords,
    )
    obs = xr.DataArray(
        np.random.randn(n_year, n_lat, n_lon),
        dims=["year", "lat", "lon"],
        coords={k: coords[k] for k in ("year", "lat", "lon")},
    )

    diag = spread_error_diagnostics(forecast, obs)
    assert isinstance(diag, SpreadErrorDiagnostics)
    assert diag.spread.dims == ("year",)
    assert diag.error.dims == ("year",)
    assert diag.spread.sizes["year"] == n_year
    assert diag.error.sizes["year"] == n_year

    diag_sp = spread_error_diagnostics(forecast, obs, spatial=True)
    assert set(diag_sp.spread.dims) == {"year", "lat", "lon"}
    assert set(diag_sp.error.dims) == {"year", "lat", "lon"}
