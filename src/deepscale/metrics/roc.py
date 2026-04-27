import numpy as np
import xarray as xr
from .base import MetricBase
from ..registry import register_metric
from .rpss import _cpt_boundaries

@register_metric("roc")
class ROCMetric(MetricBase):
    def compute(self, forecast, obs, spatial=False, **kwargs):
        t33, t67 = _cpt_boundaries(obs.values)
        spatial_dims = [d for d in obs.dims if d != "year"]
        spatial_coords = {k: v for k, v in obs.coords.items()
                          if k != "year" and set(obs[k].dims).issubset(set(spatial_dims))}
        t33_da = xr.DataArray(t33, dims=spatial_dims, coords=spatial_coords)
        t67_da = xr.DataArray(t67, dims=spatial_dims, coords=spatial_coords)
        obs_cat = xr.where(t33_da > obs, 0, xr.where(t67_da > obs, 1, 2))

        results = {}
        names = {0: "roc_bn", 1: "roc_nn", 2: "roc_an"}

        for cat_idx in range(3):
            binary_obs = (obs_cat == cat_idx).values.flatten()
            prob_fcst = forecast.isel(tercile=cat_idx).values.flatten()

            mask = ~(np.isnan(binary_obs) | np.isnan(prob_fcst))
            binary_obs = binary_obs[mask]
            prob_fcst = prob_fcst[mask]

            if len(binary_obs) == 0 or binary_obs.sum() == 0 or binary_obs.sum() == len(binary_obs):
                results[names[cat_idx]] = 0.5
                continue

            order = np.argsort(-prob_fcst)
            binary_obs = binary_obs[order]

            n_pos = binary_obs.sum()
            n_neg = len(binary_obs) - n_pos
            tp = np.cumsum(binary_obs)
            fp = np.cumsum(1 - binary_obs)
            tpr = tp / n_pos
            fpr = fp / n_neg

            auc = np.trapezoid(tpr, fpr)
            results[names[cat_idx]] = float(auc)

        return results
