"""Probabilistic calibration to tercile probabilities: eReg and logistic/WVG.

calibrate() is Model Output Statistics — it does NOT change resolution and
does NOT regrid. Put every gridded predictor on the obs grid first:
    hcst = hcst.interp(lat=obs.lat, lon=obs.lon)
"""

import deepscale as ds

# =========================================================================
# 1. Ensemble regression (eReg) — gridded multi-model calibration
# =========================================================================
# Per grid cell: OLS of obs on ensemble-mean hindcast -> Gaussian terciles
# with leverage-inflated prediction-error variance (Wilks 2006 eq 6.22).
# Predictor form: {model: (hindcast, forecast_or_None)}; models are
# calibrated independently, averaged, and renormalized to sum to 1.

models = {
    "ECMWF": (ecmwf_hcst, ecmwf_fcst),   # each hcst (year, member, lat, lon), on obs grid
    "CFSv2": (cfsv2_hcst, None),          # None -> forecast_year picks the year
}
probs = ds.calibrate(
    models, obs,
    method="ereg",
    forecast_year=2024,
    clip_negative=True,        # precip: clamp negative regressed values
    threshold_source="obs",    # tercile boundaries from obs (or "fitted")
    native_years=False,        # True: per-model obs-overlap calibration windows
)
assert probs.sizes["tercile"] == 3   # (tercile, lat, lon), sums to 1 per valid cell

# =========================================================================
# 2. Logistic calibration on a teleconnection index (scalar predictor)
# =========================================================================
# Named indices: "wvg" (Western-V Gradient, 3-box), "wvg2", "nino34", "nino4"
wvg = ds.Index.named("wvg")

# Or a custom index from bbox regions ([lat_s, lat_n, lon_w, lon_e], 0-360 lon ok):
wvg_custom = ds.Index.custom(
    name="wvg",
    regions={
        "nino34": [-5, 5, 190, 240],
        "wnp": [20, 35, 160, 210],
        "wep": [-15, 20, 120, 160],
        "wsp": [-30, -15, 155, 210],
    },
    combine=lambda z: z["nino34"] - (z["wnp"] + z["wep"] + z["wsp"]) / 3,
)

# Gridded SST in, LogitConfig reduces it through the index automatically.
# Pass hindcast SST as the climatology when reducing forecasts so both
# share the same standardization scale (Index.reduce(sst, climatology=...)).
probs = ds.calibrate(
    predictor_hindcast=sst_hindcast,     # (year, member, lat, lon) SST
    obs=obs,                             # (year, lat, lon) predictand
    predictor_forecast=sst_forecast,     # (member, lat, lon) SST
    method=ds.LogitConfig(
        index=wvg_custom,
        model="independent_binomial",    # or "multinomial"
        predictor_level="model_mean",
        detrend=False,
        significance=None,               # set e.g. 0.1 -> statsmodels backend + mask
        regularization=None,
    ),
)

# Already-reduced scalar index series work too:
probs = ds.calibrate(index_series, obs, method="logit", forecast=1.7, min_years=10)

ds.plot_terciles(probs, title="WVG-calibrated MAM precip")
