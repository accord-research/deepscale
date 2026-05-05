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


# ===================================================================
# 8. Tercile conversion
# ===================================================================

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
    import deepscale
    report = deepscale.skill(perfect_tercile_forecast, synthetic_obs, metrics="all")
    rmse_keys = [k for k in report.scores if k in ("rmse", "root_mean_squared_error")]
    assert len(rmse_keys) == 1, f"expected one RMSE key, got {rmse_keys}"
    hss_keys = [k for k in report.scores if k in ("hss", "heidke_skill_score")]
    assert len(hss_keys) == 1, f"expected one HSS key, got {hss_keys}"


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
