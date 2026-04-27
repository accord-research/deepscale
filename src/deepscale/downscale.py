import xarray as xr
from .registry import get_method
from .tercile import to_tercile

_METHOD_PARAMS = ("n_modes", "x_eof_modes", "y_eof_modes", "cca_modes",
                  "device", "n_samples", "target_variable")


def _log(verbose, msg):
    if verbose:
        print(f"[deepscale] {msg}")


def downscale(gcm, obs, method="bcsd", output_type="continuous", **kwargs):
    verbose = kwargs.pop("verbose", True)
    forecast = kwargs.pop("forecast", None)
    fit_gcm, fit_obs = gcm, obs
    if "year" in gcm.dims and forecast is None:
        if len(gcm.year) < 2:
            raise ValueError("Need at least 2 years when forecast is not provided.")
        fit_gcm = gcm.isel(year=slice(None, -1))
        fit_obs = obs.sel(year=fit_gcm.year)
        forecast = gcm.isel(year=-1, drop=True)

    method_cls = get_method(method)
    _log(verbose, f"downscale start: method={method}")
    m = method_cls(**{k: v for k, v in kwargs.items() if k in _METHOD_PARAMS})
    m.fit(fit_gcm, fit_obs, **kwargs)
    forecast = gcm if forecast is None else forecast

    result = m.predict(forecast, **kwargs)

    if output_type == "tercile":
        result = to_tercile(result, obs)

    _log(verbose, "downscale complete")
    return result
