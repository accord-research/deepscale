# Skill metrics and tercile conversion

## Tercile conversion — the leakage discipline

Two distinct paths in `deepscale.tercile`. Choosing the wrong one silently inflates skill.

### `to_tercile(forecast, obs_climatology, method="counting")` — production path

- `forecast` `(member, lat, lon)`; boundaries = 1/3 and 2/3 empirical quantiles of the supplied climatology over `year`.
- `method`: `"counting"` (fraction of members per category) or `"gaussian"`.
- Returns `(tercile, lat, lon)`.
- **Leakage hazard:** for CV hindcasts, passing the full obs leaks the held-out year into the boundaries. Inside a manual CV loop use `to_tercile(pred, obs.sel(year=train_years))` at most — or better, use `to_tercile_cv`.

### `to_tercile_cv(cv_predictions, obs, method="bootstrap", leverages=None, n_modes=3, cpt_boundaries=True)` — scoring path

- `cv_predictions` `(year, [member,] lat, lon)` from held-out folds; returns `(year, tercile, lat, lon)` with held-out discipline.
- `method`: `"cpt"` (Student-t with leverage-inflated PEV, `dofr = n − n_modes − 1`; requires per-year `leverages` from a CCA fit), `"bootstrap"`, `"gaussian_loo"`, `"t"`, `"gaussian_pev"`.

### `cpt_tercile_forecast(forecast, t33, t67, s2, dofr, leverage=0.0)`

Single-map CPT Student-t tercile kernel (the same math CPT Fortran uses for its probabilistic forecast page).

### Boundaries

Default is CPT-compatible empirical quantiles (`rndx = n*p + 0.5` convention). Cells where t33 == t67 (degenerate/dry) get NaN boundaries and drop out of skill computations — expected behavior.

## Metrics (`metrics=` to `skill()`)

All metrics are classes with `compute(forecast, obs, spatial=False, **kwargs)`; `spatial=False` → scalar (pooled over cells and years), `spatial=True` → `(lat, lon)` map. Some add `compute_diagram(...)`, captured in `SkillReport.diagrams`.

### Probabilistic (require `tercile` dim of size 3)

| Name (aliases) | Range / no-skill | Semantics |
|---|---|---|
| `rpss` | (−∞, 1]; 0 = climatology | Ranked Probability Skill Score vs climatology `[1/3, 2/3, 1]`; CPT categorization (strict `<`). kwargs: `loo_boundaries=False`, `bounded=False`, `cv_window=1`. Raises `ValueError` (pointing to `to_tercile_cv`) if no tercile dim |
| `roc` | [0,1] per category; 0.5 = no skill | Returns a dict `{"roc_bn", "roc_nn", "roc_an"}` (AUC per category). Diagram: `{bn/nn/an: {fpr, tpr, area}}`. **Not usable as `primary_metric`** — use the leaf metrics below |
| `roc_area_below_normal`, `roc_area_above_normal` | [0,1]; 0.5 | Scalar per-category AUCs (leaf metrics) |
| `generalized_roc` (`groc`) | [0,1]; 0.5 | Multi-category discrimination. `loo_boundaries` kwarg; NaN + warning if < 2 distinct obs categories |
| `reliability` | ≥ 0; 0 = perfect | Brier reliability decomposition term (lower is better), `n_bins=5`. Diagram: per-category bins |
| `heidke_skill_score` (`hss`) | ≤ 1; 0 = chance | Categorical: tercile collapsed via argmax, pooled contingency table (or per-cell when spatial) |

### Continuous / deterministic (reject a `tercile` dim; need the ensemble or its mean)

| Name (aliases) | Notes |
|---|---|
| `pearson_r` | Anomaly correlation over `year`; averages `member` first; raises `ValueError` on tercile input |
| `spearman` | Rank correlation, NaN-aware |
| `2afc` | P(forecast correctly ranks two random non-tied obs years); [0,1], 0.5 = no skill |
| `root_mean_squared_error` (`rmse`) | RMSE |
| `spread_error_ratio` | `mean(spread)/mean(error)`; ≈1 = well calibrated. Needs `member` dim |
| `spread_error_correlation` | Pearson r between per-year spread and error; > 0 desirable; NaN + warning if < 3 years. Needs `member` dim |

Helper: `deepscale.metrics.spread_error.spread_error_diagnostics(forecast, obs, *, spatial=False) -> SpreadErrorDiagnostics(spread, error)`.

### Presets

- `metrics="svslrf"` → `["rpss", "roc", "reliability"]` (the WMO-SVSLRF mandatory triplet).
- `metrics="all"` → every registered metric; ones that raise `ValueError` on incompatible shapes are skipped with a `RuntimeWarning`.

### Pairing rule

Score tercile metrics on the tercile forecast and continuous metrics on the deterministic ensemble — two separate `skill()` calls:

```python
report_prob = ds.skill(cv_terciles, obs,
                       metrics=["rpss", "hss", "roc", "generalized_roc", "reliability"],
                       spatial=True)
report_det = ds.skill(cv_continuous, obs,
                      metrics=["pearson_r", "spearman", "2afc", "rmse",
                               "spread_error_ratio", "spread_error_correlation"],
                      spatial=True)
```
