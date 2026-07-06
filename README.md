# DeepScale

Modular downscaling, calibration, and verification for seasonal climate forecasts.

DeepScale turns coarse global-model (GCM) forecasts and fine-resolution observations into calibrated, high-resolution forecast products, and scores them with cross-validated skill metrics. It operates on xarray arrays and is agnostic to where the data came from, so it pairs naturally with a data layer like [Rosetta](https://github.com/accord-research/rosetta) but does not require it.

Downscaling methods, skill metrics, and ensemble strategies are looked up by name from a registry, so you select them with plain strings and can add new ones without changing the orchestration code.

## Installation

```bash
pip install accord-deepscale
```

The distribution is published as `accord-deepscale`; the import name is `deepscale`:

```python
import deepscale
```

The shapefile and region-clipping helpers additionally require Rosetta:

```bash
pip install accord-rosetta
```

DeepScale requires Python 3.10 or newer.

## Core API

```python
import deepscale

# Bias-correct and downscale one model against observations.
result = deepscale.downscale(gcm, obs, method="bcsd")

# Turn a predictor into below/normal/above tercile probabilities.
probs = deepscale.calibrate(predictor, obs, method="ereg")

# Try several methods and keep the most skillful.
best = deepscale.optimize(gcm, obs, methods=["bcsd", "cca"])

# Combine multiple models into one forecast.
mme = deepscale.ensemble([model_a, model_b], obs, strategy="uniform")

# Score a forecast against observations.
report = deepscale.skill(forecast, obs, metrics=["rpss", "roc"])
```

Inputs are xarray arrays with CF-style coordinates. A GCM hindcast has dimensions `(year, member, lat, lon)` and observations have `(year, lat, lon)`. Outputs are continuous or tercile forecast products plus skill summaries and maps. Terciles are ordered `[0, 1, 2]` for below-normal, normal, and above-normal.

## What is included

Everything below is selected by name.

Downscaling and bias-correction methods, passed as `method=` to `downscale()` and `optimize()`:

| Method | Description |
|---|---|
| `bcsd` | Bias correction with spatial disaggregation |
| `cca` | Canonical correlation analysis |
| `qm` | Quantile mapping |
| `dqm` | Detrended quantile mapping |
| `delta` | Delta-change |
| `climatology` | Climatological baseline |
| `rank-analog` | Rank-based quantile matching |
| `corrdiff` | NVIDIA CorrDiff diffusion downscaling; needs GPU dependencies that are not on PyPI (see `src/deepscale/methods/corrdiff.py`) |

Calibration methods, passed as `method=` to `calibrate()`:

| Method | Description |
|---|---|
| `ereg` | Ensemble regression |
| `logit` | Logistic index calibration |

Ensemble strategies, passed as `strategy=` to `ensemble()`: `uniform`, `skill_weighted`, `bma`, `drop_worst`.

Skill metrics, passed as `metrics=` to `skill()`: `rpss`, `roc`, `roc_area_below_normal`, `roc_area_above_normal`, `generalized_roc`, `pearson_r`, `spearman`, `2afc`, `root_mean_squared_error`, `heidke_skill_score`, `reliability`, `spread_error_ratio`, `spread_error_correlation`.

Cross-validation schemes: `loyo` (leave-one-year-out), `lko` (leave-k-out), `blocked`, `expanding`.

## Example workflow

The repository ships a runnable end-to-end demo:

```bash
python examples/demo_forecast.py
```

It uses Rosetta to fetch ERA5 temperature observations (`obs/era5`) and ECMWF seasonal hindcasts (`c3s/ecmwf-monthly`), reshapes them into DeepScale inputs, then runs optimize, tercile conversion, and skill scoring. Rosetta handles the remote retrieval and normalization; DeepScale starts from the prepared xarray datasets.

The demo needs CDS credentials in `~/.cdsapirc` with the relevant dataset licences accepted (see the Rosetta README for setup). `examples/README.md` lists all demos and their prerequisites.

## Calibration

`deepscale.calibrate()` produces tercile probabilities with dims `(tercile, lat, lon)` directly. Use it when the predictor is already on the target grid, or when a scalar index drives the forecast.

### Ensemble regression (`method="ereg"`)

eReg fits each model independently with per-grid-cell ordinary least squares: the ensemble-mean hindcast predicts the observed field, and the chosen forecast year is converted to parametric tercile probabilities. Multiple models are averaged after each produces its own probability map.

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

Each hindcast needs a `year` dimension, an optional `member` dimension, and spatial dimensions named `lat/lon`, `latitude/longitude`, `Y/X`, or `y/x`. eReg calibrates, it does not regrid, so put model fields on the observation grid first. If every forecast contains exactly one year, `forecast_year` is inferred.

### Logistic index calibration (`method="logit"`)

logit fits a gridded logistic relationship between a scalar predictor index and observed tercile occurrence. Pass the hindcast index series as `predictor` and the forecast-year value as `forecast`.

```python
index = deepscale.Index.named("wvg")
hindcast_index = index.reduce(sst_hindcast)
forecast_index = index.reduce(sst_forecast, climatology=sst_hindcast)

probs = deepscale.calibrate(
    hindcast_index, obs, method="logit", forecast=forecast_index,
)
```

For gridded SST predictors, `LogitConfig` reduces the fields through an `Index` before calibration:

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

Runnable examples: `examples/demo_ensemble_regression.py` (eReg) and `examples/demo_logistic_wvg.py` (logit).

## Relationship to Rosetta

Rosetta handles data acquisition and normalization; DeepScale handles forecasting and verification. The interface between them is standardized xarray, so DeepScale stays source-agnostic and works with any data prepared the same way.

## Development setup

```bash
git clone https://github.com/accord-research/deepscale.git
cd deepscale
uv sync
```

Some examples also use Rosetta for data acquisition:

```bash
cd ..
git clone https://github.com/accord-research/rosetta.git
cd rosetta
uv sync
```

The roadmap (PyCPT parity, additional methods, and machine-learning tiers) is tracked on [GitHub Issues](https://github.com/accord-research/deepscale/issues?q=is%3Aopen+label%3Av1-roadmap) under the `v1-roadmap` label.
