import numpy as np
import xarray as xr
from .base import MetricBase
from ..registry import register_metric


def _q_empirical(sorted_v, p):
    """CPT-compatible empirical quantile (distribs.F95 q_empirical).

    Formula: rndx = n*p + 0.5, then linear interpolation at that 1-based index.
    """
    n = len(sorted_v)
    rndx = n * p + 0.5
    indx = int(rndx)
    d = rndx - indx
    if indx > 0 and indx < n:
        return sorted_v[indx - 1] * (1.0 - d) + sorted_v[indx] * d
    elif indx > 0:
        return sorted_v[n - 1] + (sorted_v[n - 1] - sorted_v[n - 2]) * d
    else:
        if n > 1:
            return sorted_v[0] - (sorted_v[1] - sorted_v[0]) * (1.0 - d)
        return sorted_v[0]


def _cpt_boundaries(obs_arr):
    """Compute per-gridpoint tercile boundaries using CPT's q_empirical.

    Parameters
    ----------
    obs_arr : np.ndarray, shape (n_years, ...) — observed values, may contain NaN.

    Returns
    -------
    t33, t67 : np.ndarray, shape (...) — lower and upper tercile boundaries.
    """
    spatial_shape = obs_arr.shape[1:]
    t33 = np.full(spatial_shape, np.nan)
    t67 = np.full(spatial_shape, np.nan)
    for idx in np.ndindex(spatial_shape):
        col = obs_arr[(slice(None),) + idx]
        valid = col[~np.isnan(col)]
        if len(valid) < 3:
            continue
        sv = np.sort(valid)
        t33[idx] = _q_empirical(sv, 1.0 / 3.0)
        t67[idx] = _q_empirical(sv, 2.0 / 3.0)
    return t33, t67


@register_metric("rpss")
class RPSSMetric(MetricBase):
    def compute(self, forecast, obs, spatial=False, loo_boundaries=False, bounded=False, cv_window=1, **kwargs):
        hcw = (cv_window - 1) // 2
        years = list(obs.year.values)
        n = len(years)
        obs_vals = obs.values  # (year, ...)
        spatial_dims = [d for d in obs.dims if d != "year"]
        spatial_coords = {k: v for k, v in obs.coords.items()
                          if k != "year" and set(obs[k].dims).issubset(set(spatial_dims))}

        results = []
        # Pre-compute full-sample boundaries if not using LOO
        if not loo_boundaries:
            t33_full, t67_full = _cpt_boundaries(obs_vals)

        for idx, yr in enumerate(years):
            if loo_boundaries:
                mask = np.array([abs(j - idx) > hcw for j in range(n)])
                t33, t67 = _cpt_boundaries(obs_vals[mask])
            else:
                t33, t67 = t33_full, t67_full

            obs_yr = obs_vals[idx]
            # CPT categorization: t > x means obs < threshold (strict <)
            obs_cat = np.where(t33 > obs_yr, 0, np.where(t67 > obs_yr, 1, 2))
            # NaN propagation: where obs or boundaries are NaN, category is meaningless
            nan_mask = np.isnan(obs_yr) | np.isnan(t33)
            obs_oh = np.stack([(obs_cat == i).astype(float) for i in range(3)], axis=0)
            if nan_mask.any():
                obs_oh[:, nan_mask] = np.nan

            fcst_yr = forecast.sel(year=yr).values  # (tercile, ...)
            fcst_cum = np.cumsum(fcst_yr, axis=0)
            obs_cum = np.cumsum(obs_oh, axis=0)
            rps = np.nansum((fcst_cum - obs_cum) ** 2, axis=0) / 2.0
            # Where forecast is NaN, RPS should be NaN
            fcst_nan = np.isnan(fcst_yr[0])
            if fcst_nan.any():
                rps[fcst_nan] = np.nan

            clim_cum = np.array([1 / 3, 2 / 3, 1.0]).reshape((-1,) + (1,) * len(obs_yr.shape))
            rps_clim = np.nansum((clim_cum - obs_cum) ** 2, axis=0) / 2.0
            if nan_mask.any():
                rps_clim[nan_mask] = np.nan

            rps_da = xr.DataArray(rps, dims=spatial_dims, coords=spatial_coords)
            rps_clim_da = xr.DataArray(rps_clim, dims=spatial_dims, coords=spatial_coords)
            results.append(xr.Dataset({"rps": rps_da, "rps_clim": rps_clim_da}))

        combined = xr.concat(results, dim="year")

        if spatial:
            rps_mean = combined["rps"].mean(dim="year")
            rps_clim_mean = combined["rps_clim"].mean(dim="year")
            if bounded:
                rpss = xr.where(
                    (rps_clim_mean > 0) & (rps_clim_mean > rps_mean),
                    1 - rps_mean / rps_clim_mean,
                    (rps_clim_mean - rps_mean) / (1 - rps_clim_mean),
                )
            else:
                rpss = 1 - rps_mean / rps_clim_mean
            return rpss

        # Global: pool all gridpoints and years, compute one RPSS
        rps_global = float(combined["rps"].mean())
        rps_clim_global = float(combined["rps_clim"].mean())
        if bounded and rps_clim_global > 0:
            if rps_clim_global > rps_global:
                return 1 - rps_global / rps_clim_global
            else:
                return (rps_clim_global - rps_global) / (1 - rps_clim_global)
        if rps_clim_global > 0:
            return 1 - rps_global / rps_clim_global
        return 0.0
