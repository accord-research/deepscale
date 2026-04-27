from dataclasses import dataclass
import numpy as np
import xarray as xr
from .registry import get_method
from .cv import get_cv
from .skill import skill
from .tercile import to_tercile

_METHOD_PARAMS = ("n_modes", "x_eof_modes", "y_eof_modes", "cca_modes",
                  "device", "n_samples", "target_variable")


def _iter(items, desc, enabled=True):
    if not enabled:
        return items
    try:
        from tqdm.auto import tqdm
        return tqdm(items, desc=desc)
    except Exception:
        return items


def _log(verbose, msg):
    if verbose:
        print(f"[deepscale] {msg}")


@dataclass
class OptimizeResult:
    method: str
    score: float
    forecast: xr.DataArray

def optimize(gcm, obs, methods=None, cv="loyo", primary_metric="rpss", **kwargs):
    progress = kwargs.pop("progress", True)
    verbose = kwargs.pop("verbose", True)
    if methods is None:
        methods = ["bcsd", "cca"]

    best_score = -np.inf
    best_method = None
    best_forecast = None

    years = list(gcm.year.values)
    cv_fn = get_cv(cv)
    _log(verbose, f"optimize start: methods={len(methods)}, cv={cv}, years={len(years)}")

    for method_name in _iter(methods, "DeepScale methods", enabled=progress):
        _log(verbose, f"scoring method={method_name}")
        cv_forecasts = []
        cv_obs_list = []

        method_cls = get_method(method_name)
        is_pretrained = getattr(method_cls, "is_pretrained", False)

        # Pre-trained methods (e.g. CorrDiff) are not retrained per fold.
        # Create once and fit on the full dataset to store metadata.
        if is_pretrained:
            m = method_cls(**{k: v for k, v in kwargs.items() if k in _METHOD_PARAMS})
            m.fit(gcm, obs, **kwargs)

        for train_years, test_year in _iter(list(cv_fn(years)), f"{method_name} folds", enabled=progress):
            gcm_train = gcm.sel(year=train_years)
            obs_train = obs.sel(year=train_years)
            gcm_test = gcm.sel(year=test_year)
            obs_test = obs.sel(year=test_year)

            if not is_pretrained:
                m = method_cls(**{k: v for k, v in kwargs.items() if k in _METHOD_PARAMS})
                m.fit(gcm_train, obs_train, **kwargs)

            pred = m.predict(gcm_test, **kwargs)
            tercile_pred = to_tercile(pred, obs_train)
            cv_forecasts.append(tercile_pred)
            cv_obs_list.append(obs_test)

        cv_fcst = xr.concat(cv_forecasts, dim="year")
        cv_fcst["year"] = years
        cv_obs = xr.concat(cv_obs_list, dim="year")
        cv_obs["year"] = years

        report = skill(cv_fcst, cv_obs, metrics=[primary_metric])
        score = report.scores.get(
            primary_metric, report.scores.get(list(report.scores.keys())[0])
        )

        if score > best_score:
            best_score = score
            best_method = method_name
            method_cls = get_method(method_name)
            m = method_cls(**{k: v for k, v in kwargs.items() if k in _METHOD_PARAMS})
            m.fit(gcm, obs, **kwargs)
            final_fcst = kwargs.get("forecast", gcm.mean("year") if "year" in gcm.dims else gcm)
            best_forecast = m.predict(final_fcst, **kwargs)
            _log(verbose, f"new best: {best_method} ({best_score:.4f})")

    _log(verbose, f"optimize complete: best={best_method}, score={best_score:.4f}")
    return OptimizeResult(method=best_method, score=best_score, forecast=best_forecast)
