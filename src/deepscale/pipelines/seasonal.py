"""Seasonal multi-model ensemble (MME) orchestrator.

See docs/superpowers/specs/2026-05-15-seasonal-mme-orchestrator-design.md
for the full contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import xarray as xr

from ..cv import get_cv
from ..ensemble import EnsembleResult, ensemble
from ..registry import get_method
from ..skill import SkillReport, skill
from ..tercile import to_tercile_cv, to_tercile

_PROBABILISTIC_METHODS = frozenset({"corrdiff"})


@dataclass
class SeasonalMMEResult:
    """Outcome of `seasonal_mme()`. Field semantics: see the design doc."""
    forecast: xr.DataArray
    tercile_forecast: xr.DataArray
    tercile_cv: xr.DataArray
    skill_report: SkillReport
    ensemble_result: EnsembleResult
    pev: xr.DataArray | None
    per_model_methods: dict = field(default_factory=dict)
    per_model_cv_hindcasts: dict = field(default_factory=dict)
    per_model_forecasts: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


def seasonal_mme(
    predictor_tracks,
    obs,
    *,
    method: str = "cca",
    cv: str = "loyo",
    cpt_args: dict | None = None,
    skill_metrics=None,
    tercile_method: str | None = None,
    forecast_year: int | None = None,
    optimize_ensemble: bool = False,
    primary_metric: str = "rpss",
    verbose: bool = True,
) -> SeasonalMMEResult:
    """Run the full PyCPT-style multi-track seasonal pipeline end-to-end.

    See docs/superpowers/specs/2026-05-15-seasonal-mme-orchestrator-design.md
    for parameter semantics, the four-case `forecast_year` resolution rules,
    and the error-handling table.
    """
    if not predictor_tracks:
        raise ValueError(
            "seasonal_mme: predictor_tracks is empty; at least one track with "
            "one model is required."
        )
    empty_tracks = [name for name, models in predictor_tracks.items() if not models]
    if empty_tracks:
        raise ValueError(
            f"seasonal_mme: tracks {empty_tracks} contain zero models. "
            f"Each track must have at least one (model_name, (hcst, fcst)) entry."
        )
    if method in _PROBABILISTIC_METHODS:
        raise NotImplementedError(
            f"seasonal_mme V1 supports deterministic methods only; method "
            f"{method!r} produces samples. Track this in the V2 roadmap."
        )

    years = _intersect_years(predictor_tracks, obs)
    obs_sliced = obs.sel(year=years)
    resolved_forecast_year = _resolve_forecast_year(predictor_tracks, years, forecast_year)
    resolved_tercile_method = _resolve_tercile_method(method, tercile_method)

    per_model_cv_hindcasts: dict = {}
    per_model_forecasts: dict = {}
    per_model_methods: dict = {}
    per_model_leverages: dict = {}

    for track_name, models in predictor_tracks.items():
        for model_name, (hcst, fcst) in models.items():
            key = (track_name, model_name)
            cv_hindcast, forecast_pred, m_full, levs = _per_model_cv(
                hcst, fcst, obs_sliced,
                method=method, cv_scheme=cv, cpt_args=cpt_args,
                forecast_year=resolved_forecast_year, verbose=verbose,
            )
            per_model_cv_hindcasts[key] = cv_hindcast
            per_model_forecasts[key] = forecast_pred
            per_model_methods[key] = m_full
            if levs is not None:
                per_model_leverages[key] = levs

    pooled_cv = _pool_members(per_model_cv_hindcasts)
    pooled_fcst = _pool_members(per_model_forecasts)

    ensemble_result = ensemble(
        pooled_cv, obs_sliced,
        strategy="uniform",
        optimize_ensemble=optimize_ensemble,
        primary_metric=primary_metric,
        cv=cv,
    )

    tercile_kwargs = {}
    if resolved_tercile_method == "cpt":
        if not per_model_leverages:
            raise ValueError(
                "seasonal_mme: tercile_method='cpt' but no per-model leverages "
                "were collected; was a non-CCA method used?"
            )
        # Average leverages across (track, model) per year. Leverage is a
        # per-year scalar; averaging gives an MME-level leverage estimate.
        n_models = len(per_model_leverages)
        levs = [sum(vals) / n_models
                for vals in zip(*per_model_leverages.values())]
        tercile_kwargs["leverages"] = levs
        tercile_kwargs["n_modes"] = (cpt_args or {}).get("n_modes", 3)

    tercile_cv = to_tercile_cv(
        ensemble_result.forecast, obs_sliced,
        method=resolved_tercile_method,
        **tercile_kwargs,
    )

    # Skill is scored against two forecast shapes:
    #   - deterministic ensemble mean → deterministic metrics
    #     (pearson_r, rmse, spearman, 2afc): they detect the lack of a
    #     `tercile` dim and skip the mean-collapse step.
    #   - tercile-probability CV forecast → probabilistic metrics
    #     (rpss, roc, reliability, heidke, generalized_roc): they require a
    #     `tercile` dim of size 3.
    # We call skill() twice and merge the reports. Each `metrics="all"` call
    # uses the existing skip_incompatible path to filter metrics that don't
    # match the supplied forecast shape.
    # The dual skill() call is by design: each metric is computed against the
    # input shape it expects, and the irrelevant half is skipped. The skip
    # warnings are useful when a caller invokes skill() directly with a single
    # input but are pure noise here, so suppress them locally rather than
    # forwarding them to seasonal_mme()'s caller.
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message=r"^skill\(metrics='all'\): skipping ",
        )
        det_report = skill(
            ensemble_result.forecast, obs_sliced,
            metrics=skill_metrics or "all",
        )
        prob_report = skill(
            tercile_cv, obs_sliced,
            metrics=skill_metrics or "all",
        )
    # Each metric raises ValueError when handed an input shape it can't
    # interpret, so the two reports have disjoint key sets in practice:
    # deterministic metrics live in det_report, probabilistic in prob_report.
    # Merge order therefore doesn't matter for correctness.
    skill_report = SkillReport(
        scores={**det_report.scores, **prob_report.scores},
        spatial={**det_report.spatial, **prob_report.spatial},
        metadata={**det_report.metadata, **prob_report.metadata},
        diagrams={**det_report.diagrams, **prob_report.diagrams},
    )

    # Build a single-year MME forecast (member dim = each per-model forecast).
    forecast_members = xr.concat(
        list(per_model_forecasts.values()),
        dim=xr.DataArray(
            [f"{t}__{m}" for (t, m) in per_model_forecasts.keys()],
            dims="member",
            name="member",
        ),
    )

    # `to_tercile` produces (tercile, lat, lon) probabilities using member-
    # counting or Gaussian fit. We use Gaussian when the ensemble has fewer
    # than 10 members (counting becomes noisy); otherwise counting.
    tercile_forecast_method = "counting" if forecast_members.sizes.get("member", 0) >= 10 else "gaussian"
    tercile_forecast = to_tercile(
        forecast_members, obs_sliced, method=tercile_forecast_method,
    )

    # MME mean for the forecast year (consumer-facing 'forecast' scalar map).
    forecast_mean = forecast_members.mean("member")

    metadata = {
        "years_used": list(years),
        "cv": cv,
        "method": method,
        "tercile_method": resolved_tercile_method,
        "tracks": list(predictor_tracks.keys()),
        "n_members": len(per_model_forecasts),
        "forecast_year": int(resolved_forecast_year),
        "run_at": datetime.now(timezone.utc).isoformat(),
    }

    return SeasonalMMEResult(
        forecast=forecast_mean,
        tercile_forecast=tercile_forecast,
        tercile_cv=tercile_cv,
        skill_report=skill_report,
        ensemble_result=ensemble_result,
        pev=ensemble_result.pev,
        per_model_methods=per_model_methods,
        per_model_cv_hindcasts=per_model_cv_hindcasts,
        per_model_forecasts=per_model_forecasts,
        metadata=metadata,
    )


def _resolve_forecast_year(predictor_tracks, intersection_years, forecast_year):
    """Resolve forecast_year per the four-case rules in the design doc.

    Returns an int. Raises ValueError on mismatch/missing.
    """
    # Inventory forecast-slice years per (track, model).
    fcst_years_per_model = {}
    fcst_provided = {}
    for track_name, models in predictor_tracks.items():
        for model_name, (_hcst, fcst) in models.items():
            key = (track_name, model_name)
            fcst_provided[key] = fcst is not None
            if fcst is not None:
                fcst_years_per_model[key] = sorted(fcst.year.values.tolist())

    if forecast_year is not None:
        # Rule 1: explicit year must be available everywhere.
        missing = []
        for track_name, models in predictor_tracks.items():
            for model_name, (hcst, fcst) in models.items():
                hcst_years = set(hcst.year.values.tolist())
                fcst_years = set() if fcst is None else set(fcst.year.values.tolist())
                if forecast_year not in (hcst_years | fcst_years):
                    missing.append(f"{track_name}/{model_name}")
        if missing:
            raise ValueError(
                f"seasonal_mme: forecast_year={forecast_year} is not available "
                f"in models {missing} (neither in their hindcast nor in a "
                f"provided forecast slice)."
            )
        return forecast_year

    all_provided = all(fcst_provided.values())
    none_provided = not any(fcst_provided.values())

    if all_provided:
        # Rule 2: every model has a forecast slice — infer year.
        single_years = {k: ys for k, ys in fcst_years_per_model.items() if len(ys) == 1}
        if len(single_years) != len(fcst_years_per_model):
            multi = [k for k, ys in fcst_years_per_model.items() if len(ys) != 1]
            raise ValueError(
                f"seasonal_mme: forecast_year=None requires every forecast "
                f"slice to be a single year. Multi-year slices: {multi}. "
                f"Pass an explicit forecast_year=..."
            )
        years = {ys[0] for ys in single_years.values()}
        if len(years) != 1:
            raise ValueError(
                f"seasonal_mme: forecast_year=None and forecast slices have "
                f"different years: {fcst_years_per_model}. Pass an explicit "
                f"forecast_year=..."
            )
        return next(iter(years))

    if none_provided:
        # Rule 3: fall back to last year of the intersected hindcast range.
        return max(intersection_years)

    # Rule 4: mixed → require explicit year.
    raise ValueError(
        "seasonal_mme: forecast_year=None with mixed fcst=None / "
        "fcst=DataArray across models. Pass an explicit forecast_year=..."
    )


_METHOD_PARAMS = (
    "n_modes", "x_eof_modes", "y_eof_modes", "cca_modes",
    "device", "n_samples", "target_variable", "standardize",
)


def _per_model_cv(hcst, fcst, obs_sliced, *, method, cv_scheme, cpt_args,
                  forecast_year, verbose):
    """Run a CV-folded fit-and-predict for one (hcst, fcst) pair.

    Returns (cv_hindcast, forecast_pred, fitted_method_full, leverages).

    - `cv_hindcast`: (year, lat, lon) — held-out predictions, one per year in
      obs_sliced.
    - `forecast_pred`: (lat, lon) — single-year prediction for `forecast_year`.
    - `fitted_method_full`: method object trained on all years of (hcst, obs).
      Used by the orchestrator if leverages or other introspection are needed.
    - `leverages`: list[float] aligned to obs_sliced.year, or None if the
      method does not expose a `.leverage` attribute.
    """
    method_cls = get_method(method)
    method_kwargs = {k: v for k, v in (cpt_args or {}).items() if k in _METHOD_PARAMS}

    # Slice hcst to the intersection years used for fitting/scoring.
    # Use .values to avoid xarray carrying along any non-dimension scalar
    # coordinates (e.g. a stray 'member' scalar from CDS-fetched data) that
    # would conflict with dimension coordinates on hcst.
    hcst_sliced = hcst.sel(year=obs_sliced.year.values)
    all_years = list(obs_sliced.year.values)

    # ---- CV loop: held-out predictions for each obs year. ----
    cv_fn = get_cv(cv_scheme) if isinstance(cv_scheme, str) else cv_scheme
    fold_predictions = []
    leverages = [] if hasattr(method_cls, "leverage") else None
    can_leverage = leverages is not None

    for train_years, test in cv_fn(all_years):
        test_years = test if isinstance(test, list) else [test]
        m = method_cls(**method_kwargs)
        m.fit(hcst_sliced.sel(year=train_years), obs_sliced.sel(year=train_years),
              **method_kwargs)
        test_forecast = hcst_sliced.sel(year=test_years)
        pred = m.predict(test_forecast, **method_kwargs)
        # `predict` may return a result with the year dim or with year squeezed
        # out depending on the method. Normalize to (year, lat, lon).
        if "year" not in pred.dims:
            pred = pred.expand_dims(year=test_years)
        fold_predictions.append(pred)
        if can_leverage:
            # Leverage is computed on the same test_forecast for each test year.
            for yr in test_years:
                leverages.append(m.leverage(test_forecast.sel(year=[yr])))

    cv_hindcast = xr.concat(fold_predictions, dim="year").sortby("year")
    # Schemes that overlap (lko sliding) produce multiple predictions per year.
    cv_hindcast = cv_hindcast.groupby("year").mean("year")
    # Reduce over any member dim so cv_hindcast is (year, lat, lon), as
    # `ensemble()` expects a list of (year, lat, lon) DataArrays.
    if "member" in cv_hindcast.dims:
        cv_hindcast = cv_hindcast.mean("member")

    # ---- Full fit + forecast-year prediction. ----
    m_full = method_cls(**method_kwargs)
    m_full.fit(hcst_sliced, obs_sliced, **method_kwargs)

    if fcst is not None:
        # Caller-provided forecast slice.
        if "year" in fcst.dims:
            fcst_for_year = fcst.sel(year=[forecast_year])
        else:
            fcst_for_year = fcst.expand_dims(year=[forecast_year])
        forecast_pred = m_full.predict(fcst_for_year, **method_kwargs)
    elif forecast_year in [int(y) for y in hcst.year.values]:
        # Forecast year lives inside the original hindcast range.
        forecast_pred = m_full.predict(
            hcst.sel(year=[forecast_year]), **method_kwargs,
        )
    else:
        # Should not happen: _resolve_forecast_year should have caught this.
        raise ValueError(
            f"_per_model_cv: forecast_year={forecast_year} is unavailable in "
            f"both the provided forecast and the original hindcast."
        )

    if "year" in forecast_pred.dims and forecast_pred.sizes["year"] == 1:
        forecast_pred = forecast_pred.isel(year=0, drop=True)
    # Reduce over any member dim so forecast_pred is (lat, lon), as the spec
    # requires per_model_forecasts to be per-model (lat, lon) scalar maps.
    if "member" in forecast_pred.dims:
        forecast_pred = forecast_pred.mean("member")

    if verbose:
        print(f"[deepscale.seasonal_mme] per-model CV complete: "
              f"{len(all_years)} CV years, forecast_year={forecast_year}")

    return cv_hindcast, forecast_pred, m_full, leverages


def _pool_members(per_model_arrays):
    """Convert {(track, model): DataArray} into a list of DataArrays, each
    carrying a `method` attribute of `"{track}__{model}"` for downstream
    identification by `ensemble`'s `_member_names`.
    """
    pooled = []
    for (track, model), arr in per_model_arrays.items():
        tagged = arr.copy()
        tagged.attrs["method"] = f"{track}__{model}"
        pooled.append(tagged)
    return pooled


def _resolve_tercile_method(method, tercile_method):
    """Resolve the tercile-conversion method.

    Returns the resolved string. Raises ValueError if the caller asked for
    `cpt` with a non-CCA method (no leverages available).
    """
    if tercile_method is None:
        return "cpt" if method == "cca" else "bootstrap"
    if tercile_method == "cpt" and method != "cca":
        raise ValueError(
            f"seasonal_mme: tercile_method='cpt' requires method='cca' (which "
            f"exposes hat-matrix leverages); got method={method!r}. Use "
            f"tercile_method='bootstrap' or 'gaussian_loo' for non-CCA methods."
        )
    return tercile_method


def _intersect_years(predictor_tracks, obs):
    """Return the sorted list of years common to obs and every hindcast.

    Raises ValueError with diagnostic year ranges if the intersection has
    fewer than 5 years.
    """
    year_sets = {"obs": set(obs.year.values.tolist())}
    for track_name, models in predictor_tracks.items():
        for model_name, (hcst, _fcst) in models.items():
            year_sets[f"{track_name}/{model_name}"] = set(hcst.year.values.tolist())
    intersection = sorted(set.intersection(*year_sets.values()))
    if len(intersection) < 5:
        ranges = {
            name: (min(ys), max(ys)) if ys else (None, None)
            for name, ys in year_sets.items()
        }
        raise ValueError(
            f"seasonal_mme: year intersection across obs and all hindcasts "
            f"has only {len(intersection)} year(s); need at least 5. "
            f"Per-source year ranges: {ranges}. "
            f"Intersection: {intersection}."
        )
    return intersection
