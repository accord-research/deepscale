---
name: deepscale
description: Downscale, calibrate, ensemble, and verify seasonal climate forecasts with the accord-deepscale Python package. Use when bias-correcting or downscaling GCM hindcasts (BCSD, CCA, quantile mapping), producing tercile (below/normal/above) probability forecasts, calibrating with ensemble regression or logistic/WVG indices, building multi-model ensembles, computing skill metrics (RPSS, ROC, reliability), running the seasonal_mme PyCPT-style pipeline, or generating SVSLRF verification reports and forecast maps.
license: MIT
metadata:
  author: accord-research
  package: accord-deepscale
compatibility: Requires Python 3.10+. Real-data examples need accord-rosetta and CDS credentials; CorrDiff needs an NVIDIA GPU with torch/earth2studio installed manually.
---

# DeepScale — seasonal forecast downscaling, calibration, and verification

DeepScale turns coarse GCM seasonal hindcasts/forecasts plus fine-resolution observations into calibrated high-resolution forecasts (continuous fields or below/normal/above tercile probabilities) and scores them with cross-validated skill metrics. It operates purely on xarray and is source-agnostic — it pairs naturally with **rosetta** for data acquisition but does not require it.

- Install: `pip install accord-deepscale` — **import name is `deepscale`** (conventionally `import deepscale as ds`).
- Methods, metrics, ensemble strategies, and CV schemes are **registries looked up by name strings** — you select behavior with plain strings like `method="cca"`, `metrics="rpss"`, `strategy="bma"`, `cv="loyo"`.
- CCA is validated against CPT Fortran 17.8.3 / PyCPT to r≈0.9996 on predictions.

## Data conventions (get these right first)

| Object | Dims | Notes |
|---|---|---|
| GCM hindcast | `(year, member, lat, lon)` | `year` = consecutive integers |
| GCM forecast | `(member, lat, lon)` | year squeezed out |
| Observations (predictand) | `(year, lat, lon)` | fine grid |
| Tercile forecast | `(tercile, lat, lon)` | `tercile=[0,1,2]` = below/normal/above, sums to 1 per valid cell |
| CV tercile hindcasts | `(year, tercile, lat, lon)` | |

`lat`/`latitude`/`Y` and `lon`/`longitude`/`X` aliases are accepted, but prefer `lat`/`lon`. **CV years must be consecutive integers** (gap of 1) or CV schemes raise. NaN cells (ocean/dry) propagate cleanly. Rosetta's `fetch(..., year_index=True)` / `assemble()` produce exactly these shapes.

## Quick start — pick a method, forecast, verify

```python
import deepscale as ds
from deepscale.tercile import to_tercile, to_tercile_cv
from deepscale.cv import loyo

# gcm: (year, member, lat, lon) hindcast; obs: (year, lat, lon)
best = ds.optimize(gcm, obs, methods=["bcsd", "cca"], cv="loyo", primary_metric="rpss")
print(best.method, best.score)          # winning method + CV RPSS
tercile_fc = to_tercile(best.forecast, obs)   # (tercile, lat, lon)

# Honest cross-validated verification (see leakage warning below)
cv_preds = []
for train_years, test_year in loyo(gcm.year.values):
    model = ds.train(best.method, gcm.sel(year=train_years), obs.sel(year=train_years))
    cv_preds.append(model.predict(gcm.sel(year=test_year)).expand_dims(year=[test_year]))
import xarray as xr
cv_fcst = xr.concat(cv_preds, dim="year")
cv_terc = to_tercile_cv(cv_fcst, obs, method="bootstrap")

report = ds.skill(cv_terc, obs, metrics=["rpss", "roc", "reliability", "hss"], spatial=True)
report.to_pdf("skill.pdf")               # WMO-SVSLRF verification report
ds.write_terciles(tercile_fc, "forecast.nc", title="MAM precip")
ds.plot_terciles(tercile_fc, title="MAM 2024")
```

Or run the whole PyCPT-style pipeline in one call:

```python
result = ds.seasonal_mme({"prcp": {"ECMWF": (gcm, None)}}, obs, method="cca", cv="loyo")
result.tercile_forecast; result.skill_report.scores; result.metadata
```

## The core verbs

| Verb | Purpose | Key output |
|---|---|---|
| `ds.downscale(predictor_hindcast, obs, method="bcsd", output_type="continuous")` | One method, one GCM → downscaled field or terciles | `(member,lat,lon)` or `(tercile,lat,lon)` |
| `ds.optimize(gcm, obs, methods=[...], cv="loyo")` | Try methods under CV, keep the best | `OptimizeResult(method, score, forecast)` |
| `ds.train(name, hindcast, obs, save_to=path)` | Fit once, checkpoint; required for `requires_training` (DL) methods | fitted `MethodBase` |
| `ds.calibrate(predictor, obs, method="ereg"\|"logit"\|"smoothed_regression"\|LogitConfig)` | MOS calibration, **no regridding** | `(tercile,lat,lon)`; `smoothed_regression` is season-aware → `(season,lat,lon)` deterministic or `(season,tercile,lat,lon)` |
| `ds.ensemble(forecasts, obs, strategy="uniform", optimize_ensemble=...)` | Combine forecasts | `EnsembleResult(forecast, weights, ...)` |
| `ds.skill(forecast, obs, metrics=..., spatial=...)` | Score a forecast | `SkillReport` (`.scores`, `.spatial`, `.to_pdf`) |
| `ds.skill_compare({name: fc}, obs)` | Score several forecasts side by side | `ComparisonReport` (`.to_table`, `.to_heatmap`) |
| `ds.seasonal_mme(tracks, obs, method="cca")` | Full multi-track MME pipeline | `SeasonalMMEResult` |
| `ds.prediction_error_variance(cv_preds, obs)` | Per-cell PEV from CV residuals | `(lat,lon)` |
| `ds.flex_forecast(det_fcst, pev, obs, threshold)` | Exceedance probability P(Y > threshold) | `FlexForecastResult` |

Downscale methods: `bcsd`, `cca`, `qm`, `dqm`, `delta`, `climatology`, `rank-analog`, `corrdiff` (GPU). Ensemble strategies: `uniform`, `skill_weighted`, `bma`, `drop_worst`. CV schemes: `loyo`, `lko`, `blocked`, `expanding`. Metrics: `rpss`, `roc`, `groc`, `reliability`, `hss`, `pearson_r`, `spearman`, `2afc`, `rmse`, `msss`, `crpss`, `spread_error_ratio`, `spread_error_correlation` (+ presets `"svslrf"`, `"all"`). Full parameter tables: [references/methods.md](references/methods.md), [references/metrics-and-terciles.md](references/metrics-and-terciles.md).

## Critical discipline rules

1. **Tercile leakage:** never score CV hindcasts with `to_tercile(pred, obs)` — full obs leaks the held-out year. Use `to_tercile_cv(cv_predictions, obs, method="bootstrap"|"cpt"|...)`. `to_tercile` is for the production forecast only.
2. **Grid rule:** `calibrate()` and `skill_compare()` do **not** regrid — put the GCM on the obs grid first (`gcm.interp(lat=obs.lat, lon=obs.lon)`). `downscale()` methods map coarse→fine themselves.
3. **Metric/forecast pairing:** probabilistic metrics (`rpss`, `roc`, `groc`, `reliability`, `hss`) need a `tercile` dim of size 3; continuous metrics (`pearson_r`, `spearman`, `2afc`, `rmse`, `spread_error_*`) need the deterministic ensemble and reject tercile input. Score each family in its own `skill()` call.
4. **Don't nest `optimize()` inside a CV loop** — double CV plus non-consecutive inner years breaks the CV schemes. Use `train()` + `.predict()` in manual loops (see quick start).
5. **`primary_metric` must be a leaf metric** — `roc_an`, not `roc` (which expands to a dict).
6. **DL methods** (`requires_training=True`) refuse inline fitting: `ds.train(name, ..., save_to=path)` then `ds.downscale(..., weights_path=path)`.

## Getting data in (rosetta)

Any xarray source works; rosetta produces the exact shapes deepscale expects:

```python
import rosetta
gcm = rosetta.fetch("c3s/ecmwf-monthly", "precip", init="2024-02", target="MAM",
                    region=[-5, 15, 33, 48], hindcast=(1993, 2016),
                    year_index=True)["precip"]                    # (year, member, lat, lon)
obs = rosetta.fetch("obs/era5", "precip", region=[-5, 15, 33, 48],
                    hindcast=(1993, 2016), target="MAM",
                    seasonal="mean")["precip"]                    # (year, lat, lon)
```

`rosetta.assemble(roster, ...)` returns `{label: (hindcast, forecast)}` already shaped `(year, member, lat, lon)` with a guaranteed `member` dim. Rosetta is optional at runtime — only shapefile/geometry `Index` regions import it.

## Common pitfalls (errors + environment: [references/troubleshooting.md](references/troubleshooting.md))

- `downscale(gcm=...)` is deprecated → `predictor_hindcast=`.
- `ensemble(optimize_ensemble=True)` requires `obs` and can silently fall back to uniform weights (gate; `RuntimeWarning`).
- Degenerate/dry cells get NaN tercile boundaries and are excluded from skill — expected, not a bug.
- Plotting/reporting/GeoTIFF need the `plotting` extra (`pip install accord-deepscale[plotting]`); errors say so.
- CorrDiff GPU deps (`torch`, `earth2studio`, `nvidia-physicsnemo`) are not on PyPI — install manually; its `save`/`load` raise `NotImplementedError`.

## Runnable examples (also see the repo's `examples/` directory)

- [examples/end_to_end_forecast.py](examples/end_to_end_forecast.py) — optimize → forecast → honest CV verification → PDF report
- [examples/seasonal_mme_pipeline.py](examples/seasonal_mme_pipeline.py) — one-call multi-model MME with `seasonal_mme`
- [examples/calibration.py](examples/calibration.py) — eReg multi-model calibration, logistic/WVG index calibration, and season-aware smoothed_regression (deterministic + tercile)
- [examples/ensemble_and_reporting.py](examples/ensemble_and_reporting.py) — strategies, safeguards, skill comparison, plots

## Reference files

- [references/api.md](references/api.md) — full signatures for every public function/dataclass
- [references/methods.md](references/methods.md) — downscale methods, calibrators, ensemble strategies, CV schemes, registries
- [references/metrics-and-terciles.md](references/metrics-and-terciles.md) — every metric's semantics + tercile conversion discipline
- [references/plotting-reporting.md](references/plotting-reporting.md) — maps, diagrams, SVSLRF PDFs, GeoTIFF/NetCDF export
- [references/troubleshooting.md](references/troubleshooting.md) — error → cause table, environment/install setup, test markers, operational scripts
