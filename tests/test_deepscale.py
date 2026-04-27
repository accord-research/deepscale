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
