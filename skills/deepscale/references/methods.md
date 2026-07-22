# Methods, strategies, and cross-validation

Everything here is selected by name string via registries (`deepscale/registry.py`). Importing `deepscale` registers all built-ins.

## Downscale methods (`method=` to `downscale()` / `optimize()` / `train()`)

| Name | Class | Key params | What it does |
|---|---|---|---|
| `bcsd` | `BCSDMethod` | — | Bias Correction + Spatial Disaggregation: per-cell quantile match GCM↔coarsened obs, interpolate to fine grid, add the fine-scale detail term `obs_clim − interp(obs_clim_coarse)` |
| `cca` | `CCAMethod` | `n_modes=3`, `x_eof_modes`, `y_eof_modes`, `cca_modes`, `standardize=False`, `transform_predictand` (`None`\|`"Empirical"`; `"Gamma"` → NotImplementedError), `tailoring` (`"Anomaly"` returns anomalies), `drymask_threshold`, `synchronous_predictors=True` | SVD-based canonical correlation analysis matching CPT Fortran 17.8.3 semantics. Extras: `.leverage(forecast)` for CPT-style tercile PEV inflation; `select_modes(...)` for CV Kendall-tau auto mode selection |
| `qm` | `QuantileMappingMethod` | `variant="empirical"` \| `"parametric"` | Per-cell CDF matching, then interpolation to the obs grid; no disaggregation detail. Empirical variant clamps out-of-range values (no extrapolation) |
| `dqm` | `DetrendedQuantileMappingMethod` | `variant="empirical"` | Detrended QM (Cannon 2015): remove per-cell linear trend, QM the anomalies, re-add the GCM trend at the latest hindcast year |
| `delta` | `DeltaScalingMethod` | — | `obs_clim + interp(forecast − gcm_hist_clim)`. Sanity-check baseline |
| `climatology` | `ClimatologyMethod` | — | No-skill reference: obs climatology tiled across members |
| `rank-analog` | `RankAnalogMethod` | `closing_size=50`, `gaussian_sigma=1.5`, `upscale_factor=None` | Rank within hindcast climatology → nearest-neighbor upscale → grey-closing + Gaussian smoothing → index into sorted obs climatology |
| `corrdiff` | `CorrDiffMethod` | `device="cuda"`, `n_samples=10`, `target_variable="t2m"` | NVIDIA CorrDiff diffusion downscaler (CMIP6→ERA5), `is_pretrained=True`. Needs GPU deps installed manually (`torch`, `earth2studio`, `nvidia-physicsnemo` — not on PyPI); registered only if imports succeed. Input via `corrdiff_input=(tensor, coords)` from `prepare_corrdiff_input(dataset, target_time, model)`; `save`/`load` raise NotImplementedError |

### CCA / CPT parity

CCA numerics intentionally match CPT Fortran 17.8.3: standardize before SVD, empirical tercile boundaries with `rndx = n*p + 0.5`, leverage `= 1/n + Σ prjc²`, prediction-error variance `= s2_cv·(1+h)`, Student-t terciles with `dofr = n − n_modes − 1`. `scripts/reproduce.py` reproduces PyCPT step by step (r ≈ 0.9996 on predictions); the `agreement` pytest marker gates the parity suite. Changes to CCA must preserve this parity.

Method base classes (`deepscale.methods.base`): `MethodBase` (`fit`, `predict`, `save`/`load`, `is_trained`, `requires_training`) and `ProbabilisticMethodBase` (adds `predict_distribution()`, consumed by `downscale(output_type="tercile")` via counting). Register your own with `deepscale.registry.register_method("name")`.

## Calibrators (`method=` to `calibrate()`)

- `ereg` — ensemble regression (per-cell OLS + Gaussian terciles with leverage-inflated PEV, Wilks 2006 eq 6.22). Gridded predictors on the obs grid.
- `logit` — logistic regression on a scalar teleconnection index (`model="independent_binomial"|"multinomial"`, `backend="sklearn"|"statsmodels"`; `significance_mask` requires statsmodels and `regularization=None`; `tercile_edges="exclusive"|"inclusive"`). Use a `LogitConfig(index=...)` to reduce gridded SST predictors automatically.
- `smoothed_regression` — Kharin et al. (2017) smoothed-coefficient postprocessing. Season-aware: `predictor` `(season, year, member, lat, lon)`, `obs` `(season, year, lat, lon)` on the same grid; it owns the `season` dim (up to 12 rolling seasons). Rescales the ensemble-mean anomaly with a per-cell regression coefficient smoothed *across the seasonal cycle* to suppress the per-season sampling error a ~30-year record leaves behind. `temporal_sigma`: `None` (per-season), a `float` (cyclic Gaussian smoothing across the calendar), or `"constant"` (one year-round coefficient — a pooled regression for `output_type="deterministic"`, the mean of the per-season `a`/`b` for `output_type="tercile"`; the two `"constant"` paths differ deliberately). `output_type="deterministic"` (default) returns the rescaled forecast `(season, lat, lon)`; `output_type="tercile"` also scales the spread and returns below/near/above probabilities `(season, tercile, lat, lon)` summing to 1, via a Gaussian predictive distribution — `distribution="normal"` (temperature) or `"gamma"` (precipitation: members and obs are mapped through the gamma CDF into normal space first, so probabilities never fall on negative rainfall), with `constrained=True` (analytic spread) or `False` (numerically CRPS-minimizing). Round 1 is fit-and-apply on the hindcast: `forecast_year` must be a year present in `predictor`, and a separate out-of-sample `forecast=` field is not yet accepted (raises `NotImplementedError`); tercile output additionally requires a `member` dim (raises `ValueError` otherwise). CV scoring is the caller's concern, as with `ereg`. `deepscale.seasonal_coefficients(hindcast, obs, temporal_sigma=...)` exposes the fitted coefficient field for inspection. Score deterministic output with `msss`, tercile output with `crpss` / `reliability`.

`ereg` and `logit` are multi-model: dict predictors are calibrated per model, averaged (skipna), and renormalized to the probability simplex. `smoothed_regression` takes a single ensemble hindcast (not a dict).

## Ensemble strategies (`strategy=` to `ensemble()`)

| Name | Class | Notes |
|---|---|---|
| `uniform` | `UniformStrategy` | Simple average / weighted sum |
| `skill_weighted` | `SkillWeightedStrategy` | Weights ∝ per-member skill (`scores=` kwarg), negatives clipped to 0; all-zero falls back to uniform. Krishnamurti et al. 1999 |
| `bma` | `BMAStrategy(max_iter=200, tol=1e-7)` | Bayesian Model Averaging via EM, spatially uniform weights; `combine` needs `hindcasts=` + `obs`; self-shrinking. Raftery et al. 2005 |
| `drop_worst` | `DropWorstStrategy` | Discard the `n_drop=1` lowest-skill members, average the rest. Weigel et al. 2008 |

`regime_dependent` is planned but not implemented.

### Ensemble safeguards (with `optimize_ensemble=True`)

Defaults `{"nested_cv": True, "shrinkage": 0.5, "min_effective_n": 3, "gate": True}`:

- `nested_cv` — weights chosen in an inner CV so reported skill stays honest (`False` emits an optimistic-skill `RuntimeWarning`).
- `shrinkage` — λ-shrink optimized weights toward uniform.
- `min_effective_n` — floor on effective member count (`1/Σw²`).
- `gate` — accept optimized weights only if they beat uniform under CV; otherwise fall back to uniform with a `RuntimeWarning` (`EnsembleResult.gate_passed=False`).

## Cross-validation schemes (`cv=` to `optimize()` / `ensemble()` / `seasonal_mme()`)

All take/return integer year arrays and **require consecutive years** (gap of 1; `ValueError` otherwise). `get_cv(name)` resolves names; a callable is accepted anywhere a name is (e.g. `functools.partial(expanding, min_train=4)`).

| Name | Signature | Yields |
|---|---|---|
| `loyo` | `loyo(years, window=1)` | `(train_years, test_year)` — leave-one(-window)-year-out |
| `lko` | `lko(years, k=3)` | `(train_years, [test_years])` — leave-k-consecutive-out sliding |
| `blocked` | `blocked(years, block_size=5, gap=0)` | non-overlapping blocks; trailing partial block dropped |
| `expanding` | `expanding(years, min_train=10)` | growing training prefix; warns if < 5 eval years |

## Registries (extension points)

```python
from deepscale.registry import (register_method, register_calibrator,
                                register_metric, register_strategy,
                                get_method, get_metric, get_strategy)

@register_method("my-method")
class MyMethod(MethodBase):
    def fit(self, hindcast, obs): ...
    def predict(self, gcm_field): ...
```

Metrics register with optional aliases: `register_metric("root_mean_squared_error", aliases=("rmse",))`.
