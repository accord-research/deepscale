import warnings
import xarray as xr
from .registry import get_method
from .tercile import to_tercile

_METHOD_PARAMS = ("n_modes", "x_eof_modes", "y_eof_modes", "cca_modes",
                  "device", "n_samples", "target_variable")


def _log(verbose, msg):
    if verbose:
        print(f"[deepscale] {msg}")


def downscale(predictor_hindcast=None, obs=None, method="bcsd",
              output_type="continuous", **kwargs):
    # Deprecation shim: `gcm=` was the old name for `predictor_hindcast`.
    legacy_gcm = kwargs.pop("gcm", None)
    if legacy_gcm is not None:
        if predictor_hindcast is not None:
            raise TypeError(
                "downscale() received both `predictor_hindcast` and the deprecated "
                "`gcm` keyword. Use `predictor_hindcast` only."
            )
        warnings.warn(
            "`gcm=` is deprecated; use `predictor_hindcast=` instead.",
            DeprecationWarning, stacklevel=2,
        )
        predictor_hindcast = legacy_gcm

    if predictor_hindcast is None or obs is None:
        raise TypeError("downscale() requires `predictor_hindcast` and `obs`.")

    verbose = kwargs.pop("verbose", True)
    forecast = kwargs.pop("forecast", None)
    fit_predictor, fit_obs = predictor_hindcast, obs
    if "year" in predictor_hindcast.dims and forecast is None:
        if len(predictor_hindcast.year) < 2:
            raise ValueError("Need at least 2 years when forecast is not provided.")
        fit_predictor = predictor_hindcast.isel(year=slice(None, -1))
        fit_obs = obs.sel(year=fit_predictor.year)
        forecast = predictor_hindcast.isel(year=-1, drop=True)

    method_cls = get_method(method)
    _log(verbose, f"downscale start: method={method}")
    m = method_cls(**{k: v for k, v in kwargs.items() if k in _METHOD_PARAMS})
    m.fit(fit_predictor, fit_obs, **kwargs)
    forecast = predictor_hindcast if forecast is None else forecast

    result = m.predict(forecast, **kwargs)

    if output_type == "tercile":
        result = to_tercile(result, obs)

    _log(verbose, "downscale complete")
    return result
