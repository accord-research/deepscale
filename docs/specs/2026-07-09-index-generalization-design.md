# Index generalization: transforms, weighting, baselines

**Status:** implemented (`deepscale.indices`, `deepscale.series`)
**Date:** 2026-07-09

## The problem

`Index.reduce()` did two things unconditionally that looked like the only way to do them:

1. **It z-scored every box.** `reduce()` returned `(series - ref.mean()) / ref.std()`, always.
2. **It took an unweighted spatial mean.** `sub.mean(["lat", "lon"])`, always.

Each was a reasonable default for the Western-V Gradient family the module was written for, and each made an ordinary index inexpressible.

- **RONI** (Relative Oceanic Niño Index) is the Niño3.4 SST *anomaly* minus the 20°S–20°N mean SST *anomaly*, in °C. A difference of z-scores is a different quantity with no units.
- **The 29 °C western-Indian-Ocean threshold** used to diagnose extreme East African short rains is a statement about absolute temperature. A z-score cannot be compared to 29.
- **RONI's tropical band is 40° tall.** An unweighted mean over it over-counts its poleward cells, whose true area falls off as `cos(lat)`. Harmless for a 10°-tall Niño box; wrong here.

## The reframing

An index is three declared choices: **which boxes**, **how each box is transformed**, and **how the transformed boxes combine**. Name all three and the operational indices fall out of one machinery:

| Index | Combine | Transform | Weighting |
|---|---|---|---|
| Niño3.4 | `z(nino34)` | standardize | none |
| WVG | `z(nino34) − mean(z(wnp), z(wep), z(wsp))` | standardize | none |
| DMI / IOD | `a(wtio) − a(setio)` | anomaly | cos-lat |
| RONI | `a(nino34) − a(tropics)` | anomaly | cos-lat |
| WIO | `raw(wtio)` | raw | cos-lat |

So:

```python
transform  "standardize" | "anomaly" | "raw"   # globally, or per region
weights    None | "cos_lat" | DataArray
baseline   restricts the transform's reference period, e.g. (1991, 2020)
```

## Back-compatibility is the constraint

The WVG family's values are pinned against operational ICPAC / WASS2S reference implementations, which are unweighted. **The defaults therefore stay `transform="standardize"`, `weights=None`**, and `tests/test_indices.py` passes unchanged. The new named indices declare cos-lat weighting explicitly; the legacy ones do not.

This is the whole reason weighting is a per-index declaration rather than a global default. A "sensible" global default would have silently changed every WVG value in the library.

## Nothing here is SST-specific

The module was SST-*branded*, never SST-specific. The same reduction over a precipitation field is the Walker-circulation indicator that CHC uses to validate its Ethiopia forecast:

```python
Index.named("wpac").reduce(era5_precip)   # 9°S–4°N, 103°E–140°E
```

`reduce()`'s first argument is renamed `sst` → `field` to say so. The old keyword still works, with a `DeprecationWarning`.

## Baselines

`baseline=(1991, 2020)` restricts the transform's reference to the WMO baseline without the caller slicing the climatology by hand. It works on a `year`, `time` or `init_time` dim, and raises if the selection is empty rather than silently standardizing against nothing.

`transform="raw"` never consults the reference at all, so an absolute threshold works on a single forecast map with no time axis.

## A latent bug this surfaced

RONI's tropical band spans `west: 0, east: 360`. Both bounds are `0` after `% 360`, and the old mask (`lon >= 0 & lon <= 0`) selected **a single meridian** rather than the whole band. No existing index had a full-sweep box, so nothing caught it. `_box_series` now distinguishes the three cases: a normal box, a box wrapping the prime meridian, and a full sweep.

## Companion: `deepscale.series`

An index forecast is a scalar per year, and needs the same two things a gridded forecast does.

- **`quantile_map(x, source, target)`** bias-corrects a series against observations. It calls the same transfer function as the gridded `qm` downscaler — extracted into `methods/_qm_kernel.py` — so the two cannot drift.
- **`error_bounds(hindcast_pred, hindcast_obs, forecast, level=0.8)`** turns realised hindcast errors into a confidence interval. Because an observed value is `prediction − error`, the interval is the forecast minus the error distribution's tail quantiles, which removes the model's mean bias in the same step.

The extraction bought two generalizations for both callers:

- **Unequal sample sizes.** Each side is reduced to its own plotting positions before matching, so a 25-year model record maps onto a 45-year observed record. Same-length inputs are numerically identical to the old paired-sort form.
- **A tail policy.** `numpy.interp` clamps out-of-support inputs. That is right for a bias corrector — it never invents a value the observations have never shown — and wrong for forecasting an extreme: a record-strength El Niño mapped through a clamped transfer function comes back as merely the strongest event in the training record. `extrapolate="linear"` continues the end slope. The gridded path keeps clamping, as before.

## Validation

Against real ERSSTv5 (1991–2025, 1991–2020 baseline), the named indices reproduce the published record: 1997 and 2015 as the strongest El Niños by RONI; 2019 as a strong positive dipole with near-neutral ENSO; 2015 and 2023 exceeding 29 °C in the western Indian Ocean. See `analyses/chc_ethiopia/recreate.py ocean-state`.
