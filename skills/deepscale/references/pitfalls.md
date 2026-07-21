# Pitfalls, errors, and environment setup

## Environment

- `pip install accord-deepscale`; `import deepscale as ds`. Python ≥ 3.10.
- Dev setup: `git clone`, `uv sync`. For real-data demos also clone rosetta alongside and `uv pip install -e ../rosetta` — the `[tool.uv] override-dependencies = ["zarr>=3.1.0"]` in pyproject exists precisely so rosetta (sheerwater pins `zarr==2.18.3`) and icechunk (`zarr>=3`) co-resolve.
- Extras: `plotting` (cartopy, matplotlib, rioxarray — needed for maps, PDFs, GeoTIFF), `dev` (pytest etc.), `validation` (reference downscaling libs). CorrDiff GPU deps (`torch`, `earth2studio`, `nvidia-physicsnemo`) are **not on PyPI extras** — install manually.
- Real-data examples need CDS credentials (`~/.cdsapirc`) with ERA5/C3S licences accepted — see the rosetta skill/README for setup.
- Tests: bare `pytest` runs the unit suite (< 30 s, coverage gate > 85% on `src/deepscale`); markers `integration` (real data), `agreement` (vs PyCPT reference), `gpu` are skipped unless requested: `pytest -m integration`.

## Error → cause table

| Symptom | Cause / fix |
|---|---|
| `ValueError` about grid mismatch from `calibrate`/`skill_compare` | These never regrid. `gcm.interp(lat=obs.lat, lon=obs.lon)` first |
| `ValueError` pointing to `to_tercile_cv()` from RPSS | You passed a continuous forecast to a probabilistic metric — convert to terciles (with CV discipline) first |
| `ValueError` from `pearson_r` on tercile input | Continuous metrics reject tercile forecasts — score the deterministic ensemble |
| `ValueError`: years not consecutive | CV schemes require integer years with gap 1. Usually caused by nesting `optimize()` inside an outer CV loop — use `train()` + `.predict()` instead |
| `ValueError` from `primary_metric="roc"` | `roc` expands to a dict; use a leaf metric (`roc_an`, `roc_area_below_normal`, ...) |
| `RuntimeError`: method requires training | `requires_training=True` (DL) method — `ds.train(name, ..., save_to=p)` then `ds.downscale(..., weights_path=p)` |
| `TypeError` passing both `gcm=` and `predictor_hindcast=` | `gcm=` is a deprecated alias; use `predictor_hindcast=` only |
| `RuntimeWarning`: ensemble gate fell back to uniform | Optimized weights didn't beat uniform under CV — working as designed; check `EnsembleResult.gate_passed` |
| `RuntimeWarning`: optimistic skill | You set `safeguards={"nested_cv": False}` — reported skill is no longer honest |
| `ValueError`: unknown safeguard key | `safeguards` accepts only `nested_cv`, `shrinkage`, `min_effective_n`, `gate` |
| `ImportError` naming `accord-deepscale[plotting]` | Install the plotting extra for maps/PDF/GeoTIFF |
| `NotImplementedError` from CorrDiff `save`/`load` | Torch model isn't picklable — re-instantiate instead of checkpointing |
| `NotImplementedError` from `flex_forecast(distribution="gamma")` | V1 is Gaussian-only |
| `NotImplementedError`: `transform_predictand="Gamma"` | CCA supports `None` or `"Empirical"` |
| `ValueError` from `prediction_error_variance` | `cv_predictions` and `obs` must cover the same set of years |
| `seasonal_mme` raises about years | Needs ≥ 5 intersection years across obs and all hindcasts |
| Whole regions NaN in tercile output | Degenerate boundaries (t33 == t67, e.g. dry cells) or uncalibratable cells (eReg < 3 finite years; logit < `min_years`) — masked by design |

## Statistical discipline (worth restating)

- **Leakage:** `to_tercile(pred, obs)` on CV hindcasts leaks the held-out year through the boundaries. Production forecast → `to_tercile`; CV hindcasts → `to_tercile_cv`. `optimize()` handles this internally (uses train-fold obs).
- **Two skill calls:** probabilistic metrics on `(year, tercile, lat, lon)` CV terciles; continuous metrics on the `(year, member, lat, lon)` CV ensemble.
- **Honest ensembling:** keep the default safeguards; `member_cv_skill`, `effective_n`, `shrinkage_lambda` on `EnsembleResult` document what happened.
- CCA numerics intentionally match CPT Fortran (standardize before SVD, empirical `rndx = n*p + 0.5` boundaries, leverage `= 1/n + Σ prjc²`, PEV `= s2_cv·(1+h)`, Student-t terciles with `dofr = n − n_modes − 1`). `scripts/reproduce.py` reproduces PyCPT step by step (r ≈ 0.9996).

## Rosetta integration

Rosetta output feeds deepscale directly:

```python
import rosetta
gcm_ds = rosetta.fetch("c3s/ecmwf-monthly", "precip", init="2024-02", target="MAM",
                       region=[-5, 15, 33, 48], hindcast=(1993, 2016), year_index=True)
obs_ds = rosetta.fetch("obs/era5", "precip", region=[-5, 15, 33, 48],
                       hindcast=(1993, 2016), target="MAM", seasonal="mean")
gcm = gcm_ds["precip"]   # (year, member, lat, lon)
obs = obs_ds["precip"]   # (year, lat, lon)
```

`rosetta.assemble(roster, ...)` returns `{label: (hindcast, forecast)}` already shaped `(year, member, lat, lon)` with a guaranteed `member` dim (deepscale calls `hindcast.mean("member")`). Rosetta is also a soft dependency of `Index`: shapefile/geometry index regions import `rosetta.region.resolve_region` — bbox regions never need it.

Larger operational drivers in the repo: `scripts/nightly/run_country.py` (per-country nightly pipeline), `scripts/s2s/run_issuance.py` (sub-seasonal testbed), `scripts/reproduce.py` (PyCPT parity), `notebooks/experimentB1.ipynb` (PyCPT vs DeepScale side-by-side).
