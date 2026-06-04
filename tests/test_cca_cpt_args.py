"""§7 CPT_ARGS parity tests (issue #23).

Each of the six knobs has a test that toggling it changes the output in the
expected way:
  - drymask_threshold  : removes low-climatology predictand cells before the fit
  - transform_predictand="Empirical" : rank-transform round-trip (bounded output)
  - transform_predictand="Gamma"     : deferred -> NotImplementedError
  - tailoring="Anomaly": output is the anomaly (climatological mean not added back)
  - synchronous_predictors : accepted (documented; no behaviour change)
  - skillmask_threshold: low-CV-skill cells replaced with climatology (post-CV)
  - crossvalidation_window: threaded into the CV scheme's window
"""
import numpy as np
import pytest
import xarray as xr

from deepscale.methods.cca import CCAMethod
from deepscale.pipelines.seasonal import seasonal_mme, _METHOD_PARAMS


def _tracks(hcst):
    return {"prcp": {"m1": (hcst, None)}}


# ---- C1: whitelist ----
def test_method_params_whitelist_includes_cca_cpt_args():
    for k in ("transform_predictand", "tailoring", "drymask_threshold"):
        assert k in _METHOD_PARAMS


# ---- C2: drymask_threshold (before the fit) ----
def test_drymask_excludes_low_climatology_cells(synthetic_gcm_hindcast, synthetic_obs):
    obs = synthetic_obs.copy()
    obs[:, 0, 0] = 1e-3                      # near-zero climatology corner cell
    m = CCAMethod(drymask_threshold=0.1)
    m.fit(synthetic_gcm_hindcast, obs)
    pred = m.predict(synthetic_gcm_hindcast.isel(year=0))   # forecast keeps member dim
    assert np.isnan(pred.isel(lat=0, lon=0).values).all()
    assert np.isfinite(pred.isel(lat=10, lon=10).values).any()


def test_no_drymask_keeps_all_cells(synthetic_gcm_hindcast, synthetic_obs):
    obs = synthetic_obs.copy()
    obs[:, 0, 0] = 1e-3
    m = CCAMethod(drymask_threshold=None)
    m.fit(synthetic_gcm_hindcast, obs)
    pred = m.predict(synthetic_gcm_hindcast.isel(year=0))
    assert np.isfinite(pred.isel(lat=0, lon=0).values).any()


# ---- C3: transform_predictand ----
def test_transform_empirical_predictions_bounded(synthetic_gcm_hindcast, synthetic_obs):
    m = CCAMethod(transform_predictand="Empirical")
    m.fit(synthetic_gcm_hindcast, synthetic_obs)
    pred = m.predict(synthetic_gcm_hindcast.isel(year=0)).mean("member")
    vals = pred.values[np.isfinite(pred.values)]
    # empirical inverse-CDF maps back within the observed range (no blow-ups)
    assert vals.min() >= float(synthetic_obs.min()) - 1e-6
    assert vals.max() <= float(synthetic_obs.max()) + 1e-6


def test_transform_gamma_not_implemented(synthetic_gcm_hindcast, synthetic_obs):
    m = CCAMethod(transform_predictand="Gamma")
    with pytest.raises(NotImplementedError):
        m.fit(synthetic_gcm_hindcast, synthetic_obs)


# ---- C4: tailoring + synchronous_predictors ----
def test_tailoring_anomaly_changes_output(synthetic_gcm_hindcast, synthetic_obs):
    full = CCAMethod(tailoring=None)
    full.fit(synthetic_gcm_hindcast, synthetic_obs)
    pred_full = full.predict(synthetic_gcm_hindcast.isel(year=0)).mean("member")

    anom = CCAMethod(tailoring="Anomaly")
    anom.fit(synthetic_gcm_hindcast, synthetic_obs)
    pred_anom = anom.predict(synthetic_gcm_hindcast.isel(year=0)).mean("member")

    # full field sits near the obs climatology (~5); the anomaly sits near 0.
    assert abs(float(pred_anom.mean())) < 1.0 < abs(float(pred_full.mean()))


def test_synchronous_predictors_accepted():
    m = CCAMethod(synchronous_predictors=False)
    assert m.synchronous_predictors is False


# ---- C5: skillmask_threshold (post-CV, orchestrator) ----
def test_skillmask_replaces_low_skill_with_climatology(synthetic_gcm_hindcast, synthetic_obs):
    years = synthetic_obs.year.values
    climo = synthetic_obs.sel(year=years).mean("year")
    # threshold 2.0 > max possible Pearson (1.0) => every cell replaced by climo
    masked = seasonal_mme(_tracks(synthetic_gcm_hindcast), synthetic_obs,
                          cpt_args={"skillmask_threshold": 2.0}, verbose=False)
    np.testing.assert_allclose(masked.forecast.values, climo.values, atol=1e-6)

    plain = seasonal_mme(_tracks(synthetic_gcm_hindcast), synthetic_obs, verbose=False)
    assert not np.allclose(plain.forecast.values, climo.values, atol=1e-6)


# ---- C6: crossvalidation_window threading ----
def test_crossvalidation_window_threaded(monkeypatch, synthetic_gcm_hindcast, synthetic_obs):
    import deepscale.pipelines.seasonal as S
    captured = {}

    def fake_cv(years, window=1):
        captured["window"] = window
        for y in years:
            yield [yy for yy in years if yy != y], y

    monkeypatch.setattr(S, "get_cv", lambda name: fake_cv)
    seasonal_mme(_tracks(synthetic_gcm_hindcast), synthetic_obs,
                 cpt_args={"crossvalidation_window": 3}, verbose=False)
    assert captured["window"] == 3
