"""Calibrate family: probabilistic post-processing to tercile probabilities.

``calibrate()`` is the verb for methods that turn a predictor into below/normal/
above tercile probabilities *without changing resolution* — Model Output
Statistics / calibration, as distinct from ``downscale()`` (CCA, BCSD, CorrDiff)
which maps a coarse predictor onto a finer grid.

    p = calibrate(predictor, obs, method="ereg", forecast_year=2024)
    p = calibrate(index_per_model, obs, method="logit", forecast=fcst_index)

Both registered methods are multi-model: the predictor is a ``{model: ...}``
dict, each model is calibrated independently to its own tercile probabilities,
and the per-model maps are combined (default: cross-model average). The result
is an ``(tercile, lat, lon)`` DataArray, ``tercile=[0,1,2]`` = below/normal/above.

Registered methods:

- ``ereg`` — ensemble regression. Per model, per-grid-cell OLS of obs on the
  ensemble-mean hindcast, then parametric (Gaussian, prediction-error-variance)
  tercile probabilities; averaged across models.
- ``logit`` — logistic calibration. Per model, per-grid-cell logistic of tercile
  occurrence on a scalar predictor index (e.g. the WVG SST index); averaged
  across models. Supports independent-binomial and multinomial formulations,
  detrend (via the index), and a significance mask.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from .indices import Index
from .registry import get_calibrator, register_calibrator


@dataclass(frozen=True)
class LogitConfig:
    """Configuration for gridded-predictor logistic calibration.

    ``index`` reduces the gridded predictor hindcast/forecast to scalar index
    values. The resulting index series is passed to the same registered
    ``method="logit"`` engine used by the lower-level precomputed-index API.
    """

    index: Index
    model: str = "icpac_independent"
    predictor_level: str = "model_mean"
    detrend: bool = False
    significance: float | None = None
    regularization: float | None = None
    backend: str | None = None
    min_years: int = 10


def calibrate(predictor=None, obs=None, *, method, forecast=None,
              forecast_year=None, predictor_hindcast=None,
              predictor_forecast=None, combine="mean", verbose=False,
              **method_kwargs):
    """Calibrate a predictor to tercile probabilities (no resolution change).

    Parameters
    ----------
    predictor
        Method-specific. ``ereg``: ``{model: (hindcast, forecast)}`` (gridded,
        on the obs grid) or a single ``(hindcast, forecast)``. ``logit``:
        ``{model: index_series}`` or a single index series.
    obs : xr.DataArray
        Predictand ``(year, lat, lon)``; sets the tercile boundaries.
    method : {"ereg", "logit"}
        Calibrate-family method name.
    forecast, forecast_year
        Method-specific forecast selector (see the registered methods).
    combine : str
        Cross-model combination of the per-model tercile maps ("mean").

    Returns
    -------
    xr.DataArray
        ``(tercile, lat, lon)`` probabilities, ``tercile=[0,1,2]``, summing to 1
        per cell where defined.
    """
    if predictor is None and predictor_hindcast is not None:
        predictor = predictor_hindcast
    if forecast is None and predictor_forecast is not None:
        forecast = predictor_forecast

    if isinstance(method, LogitConfig):
        return _calibrate_logit_config(
            predictor, obs, forecast,
            config=method, forecast_year=forecast_year, combine=combine,
            verbose=verbose, **method_kwargs,
        )

    fn = get_calibrator(method)
    return fn(predictor, obs, forecast=forecast, forecast_year=forecast_year,
              combine=combine, verbose=verbose, **method_kwargs)


def _as_model_dict(predictor):
    """Normalize predictor into a ``{model: value}`` dict."""
    if isinstance(predictor, dict):
        return predictor
    return {"model": predictor}


def _combine_models(maps: dict, combine: str) -> xr.DataArray:
    """Combine per-model ``(tercile, lat, lon)`` maps. Default: cross-model mean.

    Each per-model map sums to 1 (or is all-NaN where that model can't
    calibrate); a skipna mean therefore also sums to 1 wherever at least one
    model is defined.
    """
    items = [m for m in maps.values() if m is not None]
    if not items:
        raise ValueError("calibrate: no model produced a tercile forecast.")
    if combine != "mean":
        raise ValueError(f"calibrate: unknown combine={combine!r} (expected 'mean').")
    if len(items) == 1:
        out = items[0]
    else:
        out = xr.concat(items, dim="model").mean("model")  # skipna
    out = out.transpose("tercile", ...)
    out = _probability_simplex_or_nan(out)
    out = out.transpose("tercile", ...)
    out.attrs.update(combine=combine, n_models=len(items))
    return out


def _probability_simplex_or_nan(probs: xr.DataArray) -> xr.DataArray:
    """Keep each cell as a complete finite tercile simplex, or mask it out."""
    finite = np.isfinite(probs)
    complete = finite.all("tercile")
    total = probs.sum("tercile", skipna=False)
    valid = complete & np.isfinite(total) & (total > 0)
    return xr.where(valid, probs / total, np.nan)


def _detrend_index(index: xr.DataArray, forecast_index, forecast_year=None):
    """Remove a linear trend from hindcast and forecast index values.

    The trend is fitted over the hindcast index using the ``year`` coordinate
    when present, otherwise positional years. The forecast value is adjusted by
    the same fitted line at its forecast year, resolved in priority order:
    an explicit ``forecast_year``; else the ``forecast_index``'s own ``year``
    coordinate; else one step after the hindcast period. The explicit argument
    matters for a bare-scalar forecast index (no year coordinate), where the
    one-step fallback would otherwise silently evaluate the trend at the wrong
    year whenever the real forecast year is not ``years[-1] + 1``.
    """
    x = np.asarray(index, dtype=float).reshape(-1)
    if "year" in index.coords:
        years = np.asarray(index["year"].values, dtype=float).reshape(-1)
    else:
        years = np.arange(x.size, dtype=float)
    ok = np.isfinite(years) & np.isfinite(x)
    if ok.sum() < 2 or np.unique(years[ok]).size < 2:
        return index, forecast_index
    slope, intercept = np.polyfit(years[ok], x[ok], deg=1)
    trend = xr.DataArray(
        slope * years + intercept,
        dims=index.dims,
        coords=index.coords,
    )
    detrended = index - trend

    if forecast_year is not None:
        fyear = float(forecast_year)
    elif isinstance(forecast_index, xr.DataArray) and "year" in forecast_index.coords:
        fyear = float(np.asarray(forecast_index["year"].values).reshape(-1)[0])
    else:
        fyear = float(years[-1] + 1.0)
    ftrend = slope * fyear + intercept
    return detrended, forecast_index - ftrend


def _calibrate_logit_config(
    predictor_hindcast,
    obs,
    predictor_forecast,
    *,
    config: LogitConfig,
    forecast_year=None,
    combine="mean",
    verbose=False,
    **overrides,
):
    """Reduce gridded predictors through ``config.index`` and run logit."""
    if overrides:
        unknown = set(overrides) - set(LogitConfig.__dataclass_fields__)
        if unknown:
            raise TypeError(
                "calibrate(LogitConfig) got unexpected keyword argument(s): "
                f"{sorted(unknown)}."
            )
        values = {field: getattr(config, field) for field in LogitConfig.__dataclass_fields__}
        values.update(overrides)
        config = LogitConfig(**values)

    if predictor_hindcast is None:
        raise ValueError(
            "calibrate(LogitConfig) requires predictor_hindcast or predictor."
        )
    if predictor_forecast is None:
        raise ValueError(
            "calibrate(LogitConfig) requires predictor_forecast or forecast."
        )
    if obs is None:
        raise ValueError("calibrate(LogitConfig) requires obs.")
    if config.predictor_level != "model_mean":
        raise ValueError(
            "LogitConfig.predictor_level currently supports only 'model_mean'."
        )

    hindcasts = _as_model_dict(predictor_hindcast)
    forecasts = _as_model_dict(predictor_forecast)
    if set(hindcasts) != set(forecasts):
        raise ValueError(
            "calibrate(LogitConfig): predictor_forecast keys "
            f"{sorted(forecasts)} must match predictor_hindcast keys "
            f"{sorted(hindcasts)}."
        )

    idx = {}
    fc = {}
    for name, hcst in hindcasts.items():
        forecast_field = _select_single_forecast(
            forecasts[name], forecast_year, context=f"calibrate(LogitConfig) model {name!r}"
        )
        index = config.index.reduce(hcst)
        forecast_index = config.index.reduce(forecast_field, climatology=hcst)
        forecast_index = _require_single_value(
            forecast_index, f"calibrate(LogitConfig) forecast index for model {name!r}"
        )
        if config.detrend:
            index, forecast_index = _detrend_index(index, forecast_index)
        idx[name] = index
        fc[name] = forecast_index
        if verbose:
            print(f"[calibrate:logit] {name}: reduced predictor via {config.index.name}")

    backend = config.backend
    if backend is None:
        backend = "statsmodels" if config.significance is not None else "sklearn"

    return _calibrate_logit(
        idx, obs,
        forecast=fc,
        combine=combine,
        model=config.model,
        backend=backend,
        regularization=config.regularization,
        significance_mask=config.significance,
        min_years=config.min_years,
        verbose=verbose,
    )


def _require_single_value(value, label: str):
    arr = np.asarray(value)
    if arr.size != 1:
        raise ValueError(f"{label} must contain exactly one value; got {arr.size}.")
    return value


def _select_single_forecast(forecast, forecast_year, *, context: str):
    """Select one forecast year from a DataArray, or validate an already scalar field."""
    if not isinstance(forecast, xr.DataArray) or "year" not in forecast.dims:
        return forecast
    if forecast_year is not None:
        if forecast_year not in set(forecast.year.values.tolist()):
            raise ValueError(
                f"{context}: forecast_year={forecast_year} is not available in "
                "the provided forecast."
            )
        return forecast.sel(year=[forecast_year])
    if forecast.sizes["year"] != 1:
        raise ValueError(
            f"{context}: forecast_year=None requires the provided forecast to "
            "contain exactly one year, or pass forecast_year=..."
        )
    return forecast


def _select_forecast_year_slice(source, forecast_year):
    """Return the single-year forecast slice from ``source`` for ``forecast_year``,
    whether ``year`` is a dimension to index into or must be attached to a
    year-less field. Shared by the ereg forecast-input resolution paths."""
    if "year" in source.dims:
        return source.sel(year=[forecast_year])
    return source.expand_dims(year=[forecast_year])


def _split_ereg_predictor(predictor, forecast):
    """Return {model: (hindcast, forecast)} from pair or alias-style inputs."""
    predictors = _as_model_dict(predictor)
    forecasts = None if forecast is None else _as_model_dict(forecast)

    if forecasts is not None and set(predictors) != set(forecasts):
        raise ValueError(
            f"calibrate(method='ereg'): forecast keys {sorted(forecasts)} must "
            f"match predictor model keys {sorted(predictors)}."
        )

    pairs = {}
    for name, value in predictors.items():
        if isinstance(value, tuple) and len(value) == 2:
            hcst, embedded_fcst = value
        elif isinstance(value, tuple):
            raise ValueError(
                "calibrate(method='ereg') predictor values must be "
                "(hindcast, forecast) pairs."
            )
        else:
            hcst, embedded_fcst = value, None
        fcst = forecasts[name] if forecasts is not None else embedded_fcst
        pairs[name] = (hcst, fcst)
    return pairs


def _resolve_ereg_forecast_year(models, obs, forecast_year):
    if forecast_year is not None:
        missing = []
        for name, (hcst, fcst) in models.items():
            source = fcst if fcst is not None else hcst
            if "year" in source.dims and forecast_year not in set(source.year.values.tolist()):
                missing.append(name)
        if missing:
            raise ValueError(
                f"calibrate(method='ereg'): forecast_year={forecast_year} is "
                f"not available in models {missing}."
            )
        return int(forecast_year)

    provided = {name: fcst is not None for name, (_hcst, fcst) in models.items()}
    if all(provided.values()):
        years = {}
        for name, (_hcst, fcst) in models.items():
            if "year" not in fcst.dims:
                continue
            if fcst.sizes["year"] != 1:
                raise ValueError(
                    "calibrate(method='ereg'): forecast_year=None requires every "
                    "provided forecast to contain exactly one year. Pass "
                    "forecast_year=..."
                )
            years[name] = int(fcst.year.values[0])
        unique = set(years.values())
        if len(unique) == 1:
            return unique.pop()
        if len(unique) > 1:
            raise ValueError(
                "calibrate(method='ereg'): forecast_year=None and forecast "
                f"slices have different years: {years}. Pass forecast_year=..."
            )
        raise ValueError(
            "calibrate(method='ereg'): forecast_year=None cannot infer a year "
            "from forecasts without a year dimension. Pass forecast_year=..."
        )

    if any(provided.values()):
        raise ValueError(
            "calibrate(method='ereg'): forecast_year=None with mixed provided "
            "and missing forecasts. Pass forecast_year=..."
        )
    return int(np.max(obs.year.values))


def _common_obs_hindcast_years(hcst, obs, *, name):
    hcst_years = set(hcst.year.values.tolist())
    obs_years = set(obs.year.values.tolist())
    years = [int(y) for y in obs.year.values.tolist() if y in hcst_years]
    if not years:
        raise ValueError(
            f"calibrate(method='ereg') model {name!r}: hindcast and obs have no "
            "overlapping years."
        )
    if len(years) < len(obs_years):
        missing = sorted(obs_years - hcst_years)
        raise ValueError(
            f"calibrate(method='ereg') model {name!r}: hindcast is missing obs "
            f"years {missing}."
        )
    return years


_NATIVE_YEARS_MIN_OVERLAP = 3


def _native_obs_hindcast_years(hcst, obs, *, name):
    """native_years=True: trim to each model's own hcst ∩ obs overlap instead
    of requiring the hindcast to cover every obs year. Matches the consumer's
    ``calibrate_ereg_native_years`` (rosetta_deepscale.run_pipeline), which
    floors at 3 overlapping years."""
    obs_years = set(obs.year.values.tolist())
    years = [int(y) for y in hcst.year.values.tolist() if y in obs_years]
    if len(years) < _NATIVE_YEARS_MIN_OVERLAP:
        raise ValueError(
            f"calibrate(method='ereg') model {name!r}: fewer than "
            f"{_NATIVE_YEARS_MIN_OVERLAP} overlapping hindcast/obs years "
            f"(got {len(years)})."
        )
    return years


@register_calibrator("ereg")
def _calibrate_ereg(predictor, obs, *, forecast=None, forecast_year=None,
                    combine="mean", clip_negative=False, threshold_source="obs",
                    native_years: bool = False, verbose=False, **_):
    """eReg calibration: per-model OLS(obs ~ ens-mean) → parametric terciles →
    cross-model average. Each model's predictor is ``(hindcast, forecast)`` with
    the GCM already on the obs grid. ``forecast_year`` selects the year; with no
    provided forecast it defaults to the maximum obs year.

    ``native_years`` (opt-in, default False): when True, each model is
    calibrated on its OWN ``hcst.year ∩ obs.year`` overlap (floor 3 years)
    instead of requiring every model's hindcast to cover every obs year.
    Default False leaves this byte-for-byte unchanged (still raises on any
    missing obs year)."""
    from .methods.ensemble_regression import EnsembleRegressionMethod

    models = _split_ereg_predictor(predictor, forecast)
    forecast_year = _resolve_ereg_forecast_year(models, obs, forecast_year)

    maps = {}
    for name, pair in models.items():
        hcst, fcst = pair
        if native_years:
            years = _native_obs_hindcast_years(hcst, obs, name=name)
        else:
            years = _common_obs_hindcast_years(hcst, obs, name=name)
        m = EnsembleRegressionMethod(clip_negative=clip_negative)
        m.fit(hcst.sel(year=years), obs.sel(year=years))
        fc = _select_forecast_year_slice(
            fcst if fcst is not None else hcst, forecast_year)
        # native_years=False: years == every obs year (enforced by
        # _common_obs_hindcast_years), so obs.sel(year=years) == obs; pass
        # obs unchanged to keep this path byte-for-byte identical to before.
        # native_years=True: years is this model's own (possibly narrower)
        # overlap, so the climatology reference must be trimmed to match —
        # this is what the consumer (calibrate_ereg_native_years) does.
        obs_climatology = obs.sel(year=years) if native_years else obs
        maps[name] = m.predict_tercile(
            fc, obs_climatology, threshold_source=threshold_source)
        if verbose:
            print(f"[calibrate:ereg] {name}: calibrated")
    out = _combine_models(maps, combine)
    out.attrs.update(method="ereg", forecast_year=int(forecast_year))
    return out


@register_calibrator("logit")
def _calibrate_logit(predictor, obs, *, forecast=None, forecast_year=None,
                     combine="mean", model="icpac_independent", backend="sklearn",
                     regularization=None, significance_mask=None, min_years=10,
                     tercile_edges: str = "exclusive",
                     detrend: bool = False, verbose=False, **_):
    """Logistic calibration: per-model per-cell logistic of tercile occurrence on
    a scalar index → cross-model average. ``predictor`` is ``{model: index}`` and
    ``forecast`` the matching ``{model: index_value}`` (or single values).

    ``tercile_edges`` (opt-in, default "exclusive"): how boundary-tied obs
    values are classified into below/normal/above; see
    ``deepscale.logistic._labels_from_obs``. Default reproduces the standard
    tercile definition / legacy behavior; "inclusive" helps dry/tied cells."""
    from .logistic import logistic_forecast

    idx = _as_model_dict(predictor)
    fc = _as_model_dict(forecast) if forecast is not None else None
    if fc is None:
        raise ValueError(
            "calibrate(method='logit') requires forecast=<index value(s)>: the "
            "predictor index for the forecast year, per model."
        )

    if set(idx) != set(fc):
        raise ValueError(
            f"calibrate(method='logit'): forecast keys {sorted(fc)} must match "
            f"predictor model keys {sorted(idx)}."
        )
    maps = {}
    for name, index in idx.items():
        _require_single_value(fc[name], f"calibrate(method='logit') forecast for model {name!r}")
        fval = float(np.asarray(fc[name]).reshape(-1)[0])
        if detrend:
            index, fval = _detrend_index(index, fc[name], forecast_year=forecast_year)
            fval = float(np.asarray(fval).reshape(-1)[0])
        maps[name] = logistic_forecast(
            index, obs, fval, model=model, backend=backend,
            regularization=regularization, significance_mask=significance_mask,
            min_years=min_years, tercile_edges=tercile_edges,
        )
        if verbose:
            print(f"[calibrate:logit] {name}: fit on index")
    out = _combine_models(maps, combine)
    out.attrs.update(method="logit", model=model, backend=backend)
    return out
