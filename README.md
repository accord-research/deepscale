# DeepScale

**Modular downscaling, calibration, and verification for seasonal forecasts.**

DeepScale turns coarse GCM forecasts and fine-resolution observations into calibrated, high-resolution forecast products with reproducible skill evaluation.

## Why this architecture

- **Method-agnostic core:** downscaling methods plug into a shared interface.
- **Composable workflow:** optimize, ensemble, and skill reuse common building blocks.
- **Extensible registry model:** add methods/metrics/strategies incrementally.
- **xarray-native contract:** interoperates cleanly with Rosetta outputs and downstream tools.

## Quickstart

```bash
git clone https://github.com/jataware/deepscale.git
cd deepscale
uv sync
```

To run the end-to-end example (which uses Rosetta for data acquisition), you also need Rosetta:

```bash
cd ..
git clone https://github.com/jataware/rosetta.git
cd rosetta
uv sync
```

Then from the parent directory:

```bash
python deepscale/examples/demo_forecast.py
```

See [Example workflow](#example-workflow) below for details on what this demo does and CDS prerequisites.

## Core API

```python
import deepscale

result = deepscale.downscale(gcm, obs, method="bcsd", ...)
probs = deepscale.calibrate(predictor, obs, method="ereg", ...)
best = deepscale.optimize(gcm, obs, methods=["bcsd", "cca"], ...)
mme = deepscale.ensemble([best_gcm1, best_gcm2], obs, strategy="uniform")
report = deepscale.skill(forecast, obs, metrics=["rpss", "roc"])
```

### Minimal contract

- **Input:** xarray arrays with CF-style coordinates.
- **Output:** continuous or tercile forecast products plus skill summaries/maps.
- **Core dims:** GCM hindcast `(year, member, lat, lon)`, obs `(year, lat, lon)`.

## Current scope (v0)

- downscaling methods: BCSD, CCA
- calibration methods: ensemble regression (`ereg`), logistic index calibration (`logit`)
- metrics: RPSS, ROC area, Pearson correlation
- cross-validation: LOYO
- ensemble strategy: uniform weights
- outputs: continuous and tercile forecasts

v0 proves the framework. Most growth should come from adding methods/metrics, not rewriting orchestration logic. The full roadmap — including PyCPT parity, ML methods, and spec-compliance work — is tracked on [GitHub Issues](https://github.com/accord-research/deepscale/issues?q=is%3Aopen+label%3Av1-roadmap) under the `v1-roadmap` label.

## Example workflow

Run the end-to-end demo:

```bash
python deepscale/examples/demo_forecast.py
```

How this uses Rosetta:

- The demo imports Rosetta and calls `rosetta.fetch(...)` twice in `deepscale/examples/demo_forecast.py`.
- First call fetches ERA5 temperature observations (`product="obs/era5"`) for the training period and region.
- Second call fetches ECMWF seasonal hindcasts (`product="c3s/ecmwf-monthly"`) for the same target setup.
- Those Rosetta outputs are normalized xarray datasets; the demo reshapes them into DeepScale input dims and then runs optimize -> tercile -> skill.

In short: Rosetta handles remote retrieval + normalization, and DeepScale starts from those prepared xarray datasets.

The demo also requires CDS credentials configured in `~/.cdsapirc` with accepted dataset licenses — see the Rosetta README for setup.

See `deepscale/examples/README.md` for full prerequisites and output details.

## Calibration API

Use `deepscale.calibrate()` when the predictor is already on the target grid, or
when a scalar predictor index is being converted directly into tercile
probabilities. Calibration methods return `(tercile, lat, lon)` probabilities,
where `tercile=[0, 1, 2]` means below-normal, normal, and above-normal.

### Ensemble regression (`method="ereg"`)

`ereg` fits each model independently with per-grid-cell ordinary least squares:
the ensemble-mean hindcast predicts the observed field, then the selected
forecast year is converted to parametric tercile probabilities. Multi-model
inputs are averaged after each model produces its own probability map.

```python
probs = deepscale.calibrate(
    {
        "ecmwf": (ecmwf_hindcast_on_obs_grid, ecmwf_forecast_on_obs_grid),
        "ukmo": (ukmo_hindcast_on_obs_grid, ukmo_forecast_on_obs_grid),
    },
    obs,
    method="ereg",
    forecast_year=2026,
)
```

Each hindcast should have `year`, optional `member`, and spatial dimensions
named `lat/lon`, `latitude/longitude`, `Y/X`, or `y/x`. eReg is a calibration
method, not a regridding method, so put model fields on the obs grid before
calling it.

If every provided forecast contains exactly one `year`, `forecast_year` can be
omitted and is inferred. If no forecast is provided, eReg falls back to the
requested year from the hindcast; with no requested year, it uses the maximum
obs year.

### Logistic index calibration (`method="logit"`)

`logit` fits a gridded logistic relationship between a scalar predictor index
and observed tercile occurrence. Pass the hindcast index series as `predictor`
and the forecast-year index value as `forecast`.

```python
index = deepscale.Index.named("wvg")

hindcast_index = index.reduce(sst_hindcast)
forecast_index = index.reduce(sst_forecast, climatology=sst_hindcast)

probs = deepscale.calibrate(
    hindcast_index,
    obs,
    method="logit",
    forecast=forecast_index,
)
```

For gridded SST predictors, `LogitConfig` reduces hindcast and forecast fields
through an `Index` before calling the same logit engine:

```python
probs = deepscale.calibrate(
    predictor_hindcast=sst_hindcast,
    predictor_forecast=sst_forecast,
    obs=obs,
    method=deepscale.LogitConfig(
        index=deepscale.Index.named("wvg"),
        detrend=True,
        significance=0.1,
    ),
)
```

`logit` aligns index and observation years by the `year` coordinate and requires
the forecast index to contain exactly one value. A `significance` mask uses the
`statsmodels` backend automatically; otherwise the default backend is
`sklearn`.

Runnable examples:

- [`examples/demo_ensemble_regression.py`](examples/demo_ensemble_regression.py)
  for eReg.
- [`examples/demo_logistic_wvg.py`](examples/demo_logistic_wvg.py) for WVG/logit.

## Relationship to Rosetta

Rosetta handles acquisition and normalization. DeepScale handles forecasting logic and verification. The boundary is standardized xarray data, so DeepScale remains source-agnostic.

## Seasonal pipeline (currently unscheduled)

DeepScale shipped a nightly GitHub Actions workflow that ran the seasonal forecast pipeline for Kenya, Ethiopia, and Nigeria and published results to a static dashboard at [https://accord-research.github.io/deepscale/](https://accord-research.github.io/deepscale/). **That workflow has been retired** — SEAS5 only republishes monthly and CHIRPS observations lag by ~2 months, so daily runs produced ~95% redundant output for the cost of CI time and dashboard churn.

The seasonal pipeline code remains intact for on-demand local execution:

```bash
uv run python -m scripts.nightly.run_country \
  --country ethiopia \
  --today 2026-05-19 \
  --output-root output/
```

- Per-country parameters: [`scripts/nightly/nightly.yml`](scripts/nightly/nightly.yml).
- Original design + plan: [`docs/superpowers/specs/2026-05-16-nightly-forecast-workflow-design.md`](docs/superpowers/specs/2026-05-16-nightly-forecast-workflow-design.md), [`docs/superpowers/plans/2026-05-16-nightly-forecast-workflow.md`](docs/superpowers/plans/2026-05-16-nightly-forecast-workflow.md).
- The `gh-pages` branch is preserved as a frozen snapshot of the last published seasonal output.

The replacement, a sub-seasonal downscaling testbed with daily cadence justified by faster-arriving ground truth, is designed in [`docs/superpowers/specs/2026-05-19-s2s-downscaling-testbed-design.md`](docs/superpowers/specs/2026-05-19-s2s-downscaling-testbed-design.md) and will land in subsequent plans.

## Repository hygiene

`deepscale/.gitignore` excludes local-only artifacts including virtualenvs, caches, and generated example outputs (`deepscale/examples/output/`, `*.png`, `*.nc`, `*.zarr/`).

Do not commit machine-local caches, generated artifacts, or credential files.
