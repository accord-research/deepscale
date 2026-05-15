"""Ensemble combination with optional weight optimisation under overfitting
safeguards (nested CV, shrinkage, effective-N floor, acceptance gate).

The public surface is :func:`ensemble`, returning an :class:`EnsembleResult`
that carries the combined forecast, the final per-member weights, per-member
CV skill, and diagnostics for each safeguard that fired.

Spec: docs/superpowers/specs/2026-05-13-ensemble-safeguards-design.md
"""

import warnings
from dataclasses import dataclass, field

import numpy as np
import xarray as xr

from .registry import get_strategy
from .cv import get_cv
from .skill import skill
from .pev import prediction_error_variance


_DEFAULT_SAFEGUARDS = {
    "nested_cv": True,
    "shrinkage": 0.5,
    "min_effective_n": 3,
    "gate": True,
}

_VALID_SAFEGUARD_KEYS = set(_DEFAULT_SAFEGUARDS)


@dataclass
class EnsembleResult:
    """Outcome of an ensemble combination.

    See docs/superpowers/specs/2026-05-13-ensemble-safeguards-design.md
    for the field contract.

    `pev` (added in the PEV design,
    `docs/superpowers/specs/2026-05-15-prediction-error-variance-design.md`)
    is the cross-validated prediction error variance per grid cell. It is
    populated automatically by `ensemble()` from honest CV predictions
    when those are available:

    - `optimize_ensemble=False` + year-dim forecasts + obs given → from the
      uniform-combined CV hindcasts.
    - `optimize_ensemble=True`, `nested_cv=True` + year-dim forecasts + obs
      → from the pooled outer-fold CV forecasts.
    - `optimize_ensemble=True`, `nested_cv=False` → `None` (no honest CV
      predictions exist on this path; matches the existing
      `nested_cv_warning`).
    - Year-less forecasts or `obs is None` → `None`.
    """
    forecast: xr.DataArray
    weights: np.ndarray
    member_names: list
    member_cv_skill: dict = field(default_factory=dict)
    effective_n: float = 0.0
    gate_passed: bool = True
    shrinkage_lambda: float = 0.0
    safeguards_applied: dict = field(default_factory=dict)
    pev: xr.DataArray | None = None


def _uniform_weights(n):
    return np.full(n, 1.0 / n)


def _apply_shrinkage(w, lambda_):
    w = np.asarray(w, dtype=float)
    if lambda_ == 0.0:
        return w.copy()
    n = len(w)
    return (1.0 - lambda_) * w + lambda_ / n


def _effective_n(w):
    w = np.asarray(w, dtype=float)
    return float(1.0 / (w ** 2).sum())


def _resolve_safeguards(user_safeguards):
    if user_safeguards is None:
        return dict(_DEFAULT_SAFEGUARDS)
    unknown = set(user_safeguards) - _VALID_SAFEGUARD_KEYS
    if unknown:
        raise ValueError(
            f"unknown safeguards keys: {sorted(unknown)}; "
            f"valid keys are {sorted(_VALID_SAFEGUARD_KEYS)}"
        )
    merged = dict(_DEFAULT_SAFEGUARDS)
    merged.update(user_safeguards)
    return merged


def _score(forecast, obs, primary_metric):
    """Score a single forecast against obs on the named metric. Returns float.

    Raises ``ValueError`` if ``primary_metric`` expands to multiple sub-metrics
    (e.g. ``roc`` → ``roc_bn``/``roc_nn``/``roc_an``). Pick a leaf metric to
    score against; silent default-selection would let the wrong AUC drive
    weight optimisation.
    """
    report = skill(forecast, obs, metrics=[primary_metric])
    if primary_metric in report.scores:
        return float(report.scores[primary_metric])
    raise ValueError(
        f"primary_metric={primary_metric!r} expands to {sorted(report.scores)}; "
        f"pick one of those leaf metrics instead."
    )


def _member_names(forecasts):
    names = []
    for i, f in enumerate(forecasts):
        if hasattr(f, "method"):
            names.append(f.method)
            continue
        arr = f.forecast if hasattr(f, "forecast") else f
        names.append(arr.name or f"member_{i}")
    return names


def ensemble(forecasts, obs, *, strategy="uniform", optimize_ensemble=False,
             primary_metric="rpss", safeguards=None, cv="loyo", **kwargs):
    """Combine multiple forecasts into a single forecast.

    See docs/superpowers/specs/2026-05-13-ensemble-safeguards-design.md
    for the full contract.

    ``**kwargs`` are forwarded to the strategy's ``combine()`` (and, where
    possible, ``fit()``) to preserve strategy-specific extras like
    ``hindcasts=`` (bma), ``n_drop=`` (drop_worst), and ``scores=``
    (skill_weighted, drop_worst).
    """
    safeguards = _resolve_safeguards(safeguards)
    strat = get_strategy(strategy)()
    member_names = _member_names(forecasts)

    if not optimize_ensemble:
        combined = strat.combine(forecasts, obs=obs, **kwargs)
        # Weights for the EnsembleResult report. Most strategies' fit() needs
        # obs (and year-dim forecasts) to evaluate per-member skill; when obs
        # is None or fit() can't consume the forwarded kwargs, fall back to
        # uniform weights for the report — the combined forecast is still
        # produced by combine(). Narrow except: (TypeError, ValueError) covers
        # the documented kwarg-mismatch and shape-mismatch cases; other
        # exceptions are real bugs and should propagate.
        if obs is None:
            w = _uniform_weights(len(forecasts))
        else:
            try:
                w = strat.fit(forecasts, obs, primary_metric=primary_metric, **kwargs)
            except (TypeError, ValueError):
                w = _uniform_weights(len(forecasts))
        pev = (
            prediction_error_variance(combined, obs)
            if obs is not None and "year" in combined.dims
            else None
        )
        return EnsembleResult(
            forecast=combined,
            weights=w,
            member_names=member_names,
            member_cv_skill={},
            effective_n=_effective_n(w),
            gate_passed=True,
            shrinkage_lambda=0.0,
            safeguards_applied={},
            pev=pev,
        )

    if obs is None:
        raise ValueError("ensemble(optimize_ensemble=True) requires obs")
    years = list(obs.year.values)
    n_members = len(forecasts)
    diag = {}

    nested_cv = safeguards["nested_cv"]
    diag["nested_cv"] = nested_cv

    use_shrinkage = (
        safeguards["shrinkage"] != 0.0
        and not strat.self_shrinks()
    )
    if strat.self_shrinks() and safeguards["shrinkage"] != 0.0:
        diag["shrinkage_skipped_for_self_shrinking"] = True
    lam = safeguards["shrinkage"] if use_shrinkage else 0.0
    diag["shrinkage"] = lam

    floor_fired = []

    if not nested_cv:
        diag["nested_cv_warning"] = (
            "running unsafeguarded optimisation; OOS skill estimate will be "
            "optimistic"
        )
        warnings.warn(diag["nested_cv_warning"], RuntimeWarning, stacklevel=2)
        w_raw = strat.fit(forecasts, obs, primary_metric=primary_metric)
        w_final = w_raw if not use_shrinkage else _apply_shrinkage(w_raw, lam)
        if _effective_n(w_final) < safeguards["min_effective_n"]:
            w_final = _uniform_weights(n_members)
            floor_fired.append("in_sample_final")
        final_fcst = strat.combine(forecasts, weights=w_final)
        if floor_fired:
            diag["min_effective_n_floor_fired_on"] = floor_fired
        return EnsembleResult(
            forecast=final_fcst,
            weights=w_final,
            member_names=member_names,
            member_cv_skill={
                name: _score(forecasts[i] if not hasattr(forecasts[i], "forecast")
                             else forecasts[i].forecast,
                             obs, primary_metric)
                for i, name in enumerate(member_names)
            },
            effective_n=_effective_n(w_final),
            gate_passed=True,
            shrinkage_lambda=lam,
            safeguards_applied=diag,
            pev=None,
        )

    # Honest nested-CV path.
    # Pool per-fold OOS predictions then score once per path, matching the
    # pattern in optimize.py. Per-fold scoring is undefined for variance-based
    # metrics like pearson_r when a fold has a single year (e.g. loyo), so
    # accumulating into pooled (year, ...) series and scoring once is the only
    # well-defined choice for arbitrary CV schemes and metrics.
    opt_cv_forecasts = []
    unif_cv_forecasts = []
    cv_obs_list = []
    cv_fn = cv if callable(cv) else get_cv(cv)
    for train_years, test in cv_fn(years):
        test_years = test if isinstance(test, list) else [test]

        # Inner fit on the outer-train years.
        w_raw = strat.fit(forecasts, obs, years=train_years, primary_metric=primary_metric)
        w_shrunk = w_raw if not use_shrinkage else _apply_shrinkage(w_raw, lam)
        if _effective_n(w_shrunk) < safeguards["min_effective_n"]:
            w_outer = _uniform_weights(n_members)
            floor_fired.extend(test_years)
        else:
            w_outer = w_shrunk

        fcst_opt = strat.combine(forecasts, weights=w_outer).sel(year=test_years)
        fcst_unif = strat.combine(
            forecasts, weights=_uniform_weights(n_members),
        ).sel(year=test_years)

        opt_cv_forecasts.append(fcst_opt)
        unif_cv_forecasts.append(fcst_unif)
        cv_obs_list.append(obs.sel(year=test_years))

    # Schemes that overlap (e.g. lko sliding) produce multiple predictions per
    # year; average them into one prediction per year before scoring. Schemes
    # that don't overlap (loyo, blocked) pass through unchanged.
    opt_pooled = xr.concat(opt_cv_forecasts, dim="year").groupby("year").mean("year")
    unif_pooled = xr.concat(unif_cv_forecasts, dim="year").groupby("year").mean("year")
    # obs is identical across overlapping folds — `.first()` is cheaper than
    # `.mean()` and avoids NaN propagation if any obs slice is partial.
    cv_obs_pooled = xr.concat(cv_obs_list, dim="year").groupby("year").first()

    opt_cv = _score(opt_pooled, cv_obs_pooled, primary_metric)
    unif_cv = _score(unif_pooled, cv_obs_pooled, primary_metric)
    diag["gate"] = safeguards["gate"]
    diag["gate_outer_cv"] = opt_cv
    diag["uniform_outer_cv"] = unif_cv
    if floor_fired:
        diag["min_effective_n_floor_fired_on"] = sorted(set(floor_fired))

    gate_passed = (
        opt_cv >= unif_cv if safeguards["gate"]
        else True  # gate disabled → accept optimised path unconditionally
    )

    if gate_passed:
        w_final_raw = strat.fit(forecasts, obs, primary_metric=primary_metric)
        w_final = w_final_raw if not use_shrinkage else _apply_shrinkage(w_final_raw, lam)
        if _effective_n(w_final) < safeguards["min_effective_n"]:
            w_final = _uniform_weights(n_members)
            diag["min_effective_n_floor_fired_on_final"] = True
        final_fcst = strat.combine(forecasts, weights=w_final)
    else:
        w_final = _uniform_weights(n_members)
        final_fcst = strat.combine(forecasts, weights=w_final)
        warnings.warn(
            f"ensemble(optimize_ensemble=True): gate failed "
            f"(optimised CV {opt_cv:.4f} < uniform CV {unif_cv:.4f}); "
            f"returning uniform.",
            RuntimeWarning,
            stacklevel=2,
        )

    pev = prediction_error_variance(opt_pooled, obs.sel(year=opt_pooled.year))

    return EnsembleResult(
        forecast=final_fcst,
        weights=w_final,
        member_names=member_names,
        member_cv_skill={
            name: _score(forecasts[i] if not hasattr(forecasts[i], "forecast")
                         else forecasts[i].forecast,
                         obs, primary_metric)
            for i, name in enumerate(member_names)
        },
        effective_n=_effective_n(w_final),
        gate_passed=gate_passed,
        shrinkage_lambda=lam,
        safeguards_applied=diag,
        pev=pev,
    )
