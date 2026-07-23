# Analog selection and scenario completion

**Status:** implemented (`deepscale.analog`, `deepscale.completion`, `deepscale.climate`, `deepscale.time`)
**Date:** 2026-07-09
**Motivating case:** CHC's Seasonal Monitoring Probability Generator (SMPG), as described in `analyses/chc_ethiopia/METHODOLOGY.md`

## The problem

Part-way through a rainy season, a forecaster knows three things and needs a fourth:

1. what has fallen so far (observations),
2. what a dynamic model says about the next 15–30 days (a forecast),
3. what happened over the rest of the season in comparable past years (analogs),

and wants an end-of-season total, expressed as a percentile of the historical record.

Splicing these three together is the SMPG's core mechanic and it had no counterpart in deepscale. Every ingredient existed — climatology, quantile machinery, region masking — but nothing composed them.

The trap is that the obvious implementation is a function that takes "nine El Niño years", a CHIRPS dekad array, and an Ethiopia shapefile, and is useless for the next country, season, or driver. This document records the three axes along which the naive version is a special case, and how each became a parameter.

## Axis 1 — what makes two years comparable

The deck says "nine moderate-to-very-strong El Niño years". That is one point in a space of selection rules. `deepscale.analog` spans the space:

| Selector | Comparability is… |
|---|---|
| `analogs_from_years` | asserted by an expert |
| `analogs_from_index` | closeness in any scalar index |
| `analogs_from_field` | closeness in any gridded pattern, over any region |
| `analogs_where` | a threshold on any series |

All four return `AnalogSet`. All four score **every** candidate year, not just the chosen ones, so the margin between the ninth analog and the tenth is inspectable — an analog ensemble's credibility rests entirely on how "comparable" was decided, and that decision should be auditable.

Every metric is distance-like (the correlation metrics return `1 - r`), so `top(n)` means one thing throughout.

`AnalogSet` composes with `&` and `|`. A compound criterion is an expression, not a new function:

```python
strong = analogs_where(nino34 >= 0.5)
rapid  = analogs_where(nino34 - nino34.shift(year=1) >= 0.5)
analogs = (strong & rapid).top(9)
```

This is not hypothetical generality. The companion OND deck's four standout analogs (1997, 2015, 2019, 2023) are not expressible as one threshold — 2015's dipole was weak, 2019's western Indian Ocean pole was below 29 °C. They are the *union* of two criteria, and `analogs_where(dmi > 1.1) | analogs_where(wio > 29.0)` reproduces the published set exactly from ERSSTv5.

### Weighting

`AnalogSet.weights()` offers `uniform`, `inverse_distance` and `gaussian`. Uniform is the default: the scores are distances in an arbitrary index space, not calibrated likelihoods, and pretending otherwise would put a spurious precision on the consensus. Explicit-year selections have no distances at all, so distance weighting degenerates to uniform rather than dividing by zero.

## Axis 2 — the time axis

Calendar timestamps from two different years never align. 1997-08-01 is not 2026-08-01 in any sense the splice cares about, and a leap day shifts every subsequent day-of-year.

What two years *do* share is an **ordinal position within the season**: the k-th dekad of Kiremt is the k-th dekad of Kiremt in every year. `deepscale.time.season_step` assigns that ordinal, and it is the coordinate every cross-year operation joins on.

- Dekad, pentad and monthly cadences have a leap-invariant number of steps per year (36, 72, 12), so their steps are ordinal differences.
- Daily steps are measured as **elapsed days from the season start**, not as a day-of-year difference, which would be off by one across a leap year.
- The cadence is inferred from the median spacing of the time coordinate. The caller does not declare it and cannot get it wrong.

`climate.seasonal_stack` reshapes a continuous archive into `(year, step, ...)`; `time.season_times` enumerates a season's steps *before* any data is placed on them, so a half-observed season still knows how long it is.

## Axis 3 — the spatial dimensions

`complete()` reduces along `step` and touches nothing else.

That single constraint means `(time, lat, lon)` yields per-pixel maps, `(time, region)` from `rosetta.zonal` yields per-district curves, and `(time,)` yields a single series — with no admin-unit code path anywhere in deepscale. It is the highest-leverage decision in the design, and a cross-library test in Rosetta pins it.

## The engine

```python
complete(observed, analogs, *, climatology, season, year=None, forecast=None,
         analog_source=None, reduce="median", weights=None,
         percentile_reference=None, overlap="observed", ...) -> CompletionResult
```

The result carries the full spliced ensemble (`scenarios`), not just its summary. Totals, consensus, percentile and the accumulation curves are thin functions over it, and `segments` records where each step's value came from — so a plot cannot disagree with the numbers.

`forecast` is optional. Running with and without it isolates exactly what the dynamic forecast contributed, which is the deck's two configurations expressed as one parameter rather than two functions.

`analog_source` may differ from `climatology`, so a bias-corrected archive can supply the remainders while the percentile is still taken against the raw record.

### Choices that fail loudly

- **The consensus reducer propagates NaN.** Skipping a missing scenario silently shrinks the ensemble, and a quantile of a smaller ensemble is a different quantile. Same for `accumulate`, which by default requires every step to be present rather than under-counting a partial season.
- **`percentile_of` propagates NaN.** The naive `(climatology < values).mean()` returns 0.0 for a NaN input, which renders missing data as "driest on record" — a plausible-looking, catastrophic answer.
- **Output dim order is pinned** to `(scenario, step, ...)`. `xr.where` broadcasts in whatever order it likes; the result must not.
- **`overlap`** must be stated when observations and forecast cover the same step. The default is `"observed"` (fact beats prediction); `"error"` refuses to guess.

## What is deliberately not here

- **No skill assessment.** `deepscale.skill` already scores forecasts; a completion is a forecast like any other.
- **No plotting logic.** `plotting/scenarios.py` reads `segments` and draws; it holds no knowledge of what a segment is.
- **No data acquisition.** Everything arrives as aligned xarray from Rosetta.

## Related

- `2026-07-09-index-generalization-design.md` — the transform/weighting axes that made RONI and the absolute-SST threshold expressible, which analog selection then consumes.
- Rosetta: `2026-07-09-issuance-keyed-forecasts-design.md` — where the forecast segment comes from.
