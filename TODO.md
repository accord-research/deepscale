# DeepScale — TODO

Tracks planned work beyond v0. The primary goal is to replicate and extend the PyCPT set of capabilities around calibration, bias correction, skill computation, and multi-model ensembling — but in a modular, method-agnostic framework.

## Downscaling / calibration methods

Current: BCSD, CCA.

### PyCPT-equivalent methods to add

- [ ] **PCR** — Principal Component Regression (the other core PyCPT method alongside CCA)
- [ ] **ELR** — Extended Logistic Regression (PyCPT's probabilistic calibration method)
- [ ] **Quantile mapping** — standard bias correction used widely in climate services
- [ ] **Variance inflation** — correct ensemble underdispersion (common PyCPT post-processing step)
- [ ] **Ensemble dress** — kernel dressing for continuous-to-probabilistic conversion

### Beyond PyCPT

- [ ] **Random forest regression** — non-linear alternative to CCA/PCR
- [ ] **XGBoost / gradient boosting** — another ML baseline
- [ ] **Neural network methods** — CNN or U-Net for spatial downscaling
- [ ] **Analog method** — match current forecast to historical analogs

## Verification metrics

Current: RPSS, ROC area, Pearson correlation.

### PyCPT-equivalent metrics to add

- [ ] **2AFC** — Two-Alternative Forced Choice (PyCPT's primary deterministic skill score)
- [ ] **Generalized ROC** — multi-category ROC (PyCPT computes this for BN/NN/AN)
- [ ] **Reliability diagrams** — calibration assessment (framework exists but no implementation)
- [ ] **Hit rate / false alarm rate** — per-category contingency table metrics
- [ ] **Brier Skill Score** — probabilistic counterpart to RPSS at individual category level
- [ ] **RMSE / MAE** — basic continuous error metrics
- [ ] **Anomaly correlation** — standard WMO verification metric

### Spatial and summary outputs

- [ ] **Goodness index** — composite skill score across multiple metrics (PyCPT uses this for optimization)
- [ ] Spatial significance masks (field significance testing)
- [ ] Summary skill tables for reports (e.g., domain-mean, sub-region breakdowns)

## Cross-validation schemes

Current: LOYO only.

- [ ] **Blocked CV** — preserves temporal autocorrelation (important for climate data)
- [ ] **Expanding window** — simulates real-time operations (train on all data up to year N, predict N+1)
- [ ] **k-fold temporal** — standard k-fold respecting time ordering
- [ ] Configurable embargo period between train/test to prevent leakage

## Ensemble strategies

Current: uniform weights only.

- [ ] **Skill-weighted** — weight GCMs by their individual cross-validated skill
- [ ] **Drop-worst** — exclude GCMs below a skill threshold before combining
- [ ] **BMA (Bayesian Model Averaging)** — probabilistic model combination
- [ ] **Regression-based** — learn optimal combination weights from cross-validated forecasts

## Calibration pipeline

PyCPT bundles calibration tightly with downscaling. DeepScale should support calibration as a separable step:

- [ ] **MOS (Model Output Statistics)** style calibration — fit statistical relationship between raw GCM output and observations
- [ ] **Tercile recalibration** — adjust tercile probabilities so they are reliable (observed frequencies match predicted probabilities)
- [ ] **Exceedance probability calibration** — extend beyond terciles to arbitrary thresholds
- [ ] Calibration should be pluggable into the pipeline: `downscale -> calibrate -> skill`

## Multi-model ensemble (MME) workflow

The full PyCPT MME pipeline that DeepScale should replicate:

- [ ] Per-GCM optimization (select best method/predictor per model) — this exists
- [ ] Per-GCM cross-validated skill — this exists
- [ ] Flexible combination strategies — partially exists (uniform only)
- [ ] MME skill assessment — compare MME skill to individual GCM skill
- [ ] Visualization: MME skill maps, individual GCM contribution diagnostics

## Predictor handling

PyCPT allows custom predictor domains (X domain != Y domain). DeepScale should support:

- [ ] Separate predictor/predictand domain specification
- [ ] Multiple predictor fields (e.g., SST + SLP as joint predictors)
- [ ] Predictor selection/search as part of `optimize()`

## Output and visualization

- [ ] **Flexible forecast maps** — standard tercile probability maps matching GHACOF/RCOF style
- [ ] **Forecast plume plots** — ensemble member trajectories
- [ ] **Skill summary reports** — exportable HTML/PDF report with maps, tables, and diagnostics
- [ ] **Flexible plot customization** — region masks, custom colorbars, country boundaries

## Code quality / testing

- [ ] Full unit test coverage for each method (fit/predict round-trip with synthetic data)
- [ ] Regression tests against PyCPT outputs for CCA and BCSD on a reference dataset
- [ ] Integration tests with real data (gated behind `@pytest.mark.integration`)
- [ ] CI pipeline
- [ ] Benchmarks: runtime and memory for typical operational workloads

## Additional examples

- [ ] Multi-GCM ensemble example (several C3S models through optimize -> ensemble -> skill)
- [ ] Precipitation example (CHIRPS obs, different variable dynamics than temperature)
- [ ] Custom method example (show how to register a new method with the decorator pattern)
- [ ] Notebook-based tutorial for interactive exploration
