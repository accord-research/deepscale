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

## Core API (four functions)

```python
import deepscale

result = deepscale.downscale(gcm, obs, method="bcsd", ...)
best = deepscale.optimize(gcm, obs, methods=["bcsd", "cca"], ...)
mme = deepscale.ensemble([best_gcm1, best_gcm2], obs, strategy="uniform")
report = deepscale.skill(forecast, obs, metrics=["rpss", "roc"])
```

### Minimal contract

- **Input:** xarray arrays with CF-style coordinates.
- **Output:** continuous or tercile forecast products plus skill summaries/maps.
- **Core dims:** GCM hindcast `(year, member, lat, lon)`, obs `(year, lat, lon)`.

## Current scope (v0)

- methods: BCSD, CCA
- metrics: RPSS, ROC area, Pearson correlation
- cross-validation: LOYO
- ensemble strategy: uniform weights
- outputs: continuous and tercile forecasts

v0 proves the framework. Most growth should come from adding methods/metrics, not rewriting orchestration logic. See [TODO.md](TODO.md) for the full roadmap, including PyCPT-equivalent capabilities to replicate.

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

## Relationship to Rosetta

Rosetta handles acquisition and normalization. DeepScale handles forecasting logic and verification. The boundary is standardized xarray data, so DeepScale remains source-agnostic.

## Repository hygiene

`deepscale/.gitignore` excludes local-only artifacts including virtualenvs, caches, and generated example outputs (`deepscale/examples/output/`, `*.png`, `*.nc`, `*.zarr/`).

Do not commit machine-local caches, generated artifacts, or credential files.
