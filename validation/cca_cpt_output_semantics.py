"""Diagnose CPT synthetic CCA crossvalidated output semantics.

The synthetic CCA parity fixture showed that DeepScale reconstructs the
low-rank control field exactly, while CPT's deterministic crossvalidated output
is opposite-signed and strongly damped. This script quantifies whether the gap
can be explained by simple output conventions such as anomaly sign, global
affine scaling, per-cell affine scaling, or per-year affine scaling.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr

from cca_synthetic_cpt_parity import _score


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation" / "results"
FIXTURE = OUT / "cca_synthetic_fixture.nc"
DEEPSCALE_OUT = OUT / "cca_synthetic_deepscale_loyo.nc"
CPT_OUT = OUT / "cca_synthetic_cptcore_crossvalidation.nc"
WORKSPACE = OUT / "cca_synthetic_cpt_workspace"
RESULT = OUT / "cca_cpt_output_semantics.json"


def _linear_fit(source: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(source) & np.isfinite(target)
    x = source[valid].ravel()
    y = target[valid].ravel()
    if len(x) < 2 or np.std(x) == 0:
        return 1.0, 0.0
    slope, intercept = np.polyfit(x, y, deg=1)
    return float(slope), float(intercept)


def _global_affine(source: xr.DataArray, target: xr.DataArray) -> tuple[xr.DataArray, dict]:
    slope, intercept = _linear_fit(source.values, target.values)
    return (source * slope + intercept).rename("cpt_global_affine_to_obs"), {
        "slope": slope,
        "intercept": intercept,
    }


def _per_cell_affine(source: xr.DataArray, target: xr.DataArray) -> tuple[xr.DataArray, dict]:
    corrected = xr.full_like(source, np.nan).rename("cpt_per_cell_affine_to_obs")
    slopes = []
    intercepts = []
    for i in range(source.sizes["lat"]):
        for j in range(source.sizes["lon"]):
            s = source.isel(lat=i, lon=j)
            t = target.isel(lat=i, lon=j)
            slope, intercept = _linear_fit(s.values, t.values)
            corrected.loc[dict(lat=s.lat, lon=s.lon)] = s * slope + intercept
            slopes.append(slope)
            intercepts.append(intercept)
    return corrected, {
        "slope_min": float(np.min(slopes)),
        "slope_median": float(np.median(slopes)),
        "slope_max": float(np.max(slopes)),
        "intercept_min": float(np.min(intercepts)),
        "intercept_median": float(np.median(intercepts)),
        "intercept_max": float(np.max(intercepts)),
    }


def _per_year_affine(source: xr.DataArray, target: xr.DataArray) -> tuple[xr.DataArray, dict]:
    corrected = xr.full_like(source, np.nan).rename("cpt_per_year_affine_to_obs")
    slopes = []
    intercepts = []
    for year in source.year.values:
        s = source.sel(year=year)
        t = target.sel(year=year)
        slope, intercept = _linear_fit(s.values, t.values)
        corrected.loc[dict(year=year)] = s * slope + intercept
        slopes.append(slope)
        intercepts.append(intercept)
    return corrected, {
        "slope_min": float(np.min(slopes)),
        "slope_median": float(np.median(slopes)),
        "slope_max": float(np.max(slopes)),
        "intercept_min": float(np.min(intercepts)),
        "intercept_median": float(np.median(intercepts)),
        "intercept_max": float(np.max(intercepts)),
    }


def _gridpoint_correlations(source: xr.DataArray, target: xr.DataArray) -> dict:
    corrs = []
    for i in range(source.sizes["lat"]):
        for j in range(source.sizes["lon"]):
            s = source.isel(lat=i, lon=j).values
            t = target.isel(lat=i, lon=j).values
            valid = np.isfinite(s) & np.isfinite(t)
            if valid.sum() > 2 and np.std(s[valid]) > 0 and np.std(t[valid]) > 0:
                corrs.append(float(np.corrcoef(s[valid], t[valid])[0, 1]))
    return {
        "min": float(np.min(corrs)),
        "median": float(np.median(corrs)),
        "max": float(np.max(corrs)),
    }


def _read_cpt_metric_grid(path: Path) -> xr.DataArray:
    lines = path.read_text().splitlines()
    lons = None
    lats = []
    rows = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("xmlns") or stripped.startswith("cpt:"):
            continue
        parts = stripped.split()
        if lons is None:
            lons = [float(x) for x in parts]
            continue
        lats.append(float(parts[0]))
        rows.append([float(x) for x in parts[1:]])
    return xr.DataArray(
        np.asarray(rows, dtype=float),
        dims=("lat", "lon"),
        coords={"lat": np.asarray(lats), "lon": np.asarray(lons)},
    ).sortby("lat")


def main() -> int:
    fixture = xr.open_dataset(FIXTURE)
    deep_ds = xr.open_dataset(DEEPSCALE_OUT)
    cpt_ds = xr.open_dataset(CPT_OUT)
    obs = fixture["obs"].transpose("year", "lat", "lon")
    deep = deep_ds["deepscale_cca"].transpose("year", "lat", "lon")
    cpt = cpt_ds["cpt_cca"].interp(lat=obs.lat, lon=obs.lon, method="nearest").transpose("year", "lat", "lon")
    obs_clim = obs.mean("year")
    deep_anom = deep - obs_clim
    obs_anom = obs - obs_clim
    cpt_anom = cpt - obs_clim
    cpt_sign_flipped = (obs_clim - cpt_anom).transpose("year", "lat", "lon").rename("cpt_anomaly_sign_flipped")
    cpt_global_affine, global_params = _global_affine(cpt, obs)
    cpt_cell_affine, cell_params = _per_cell_affine(cpt, obs)
    cpt_year_affine, year_params = _per_year_affine(cpt, obs)
    cpt_anom_global_affine, anom_global_params = _global_affine(cpt_anom, obs_anom)
    cpt_anom_global_affine = (cpt_anom_global_affine + obs_clim).transpose("year", "lat", "lon").rename("cpt_anomaly_global_affine")

    payload = {
        "files": {
            "fixture": str(FIXTURE),
            "deepscale": str(DEEPSCALE_OUT),
            "cpt": str(CPT_OUT),
            "cpt_workspace": str(WORKSPACE),
        },
        "standard_deviations": {
            "obs_full": float(obs.std()),
            "obs_anomaly": float(obs_anom.std()),
            "deepscale_full": float(deep.std()),
            "cpt_full": float(cpt.std()),
            "cpt_anomaly": float(cpt_anom.std()),
            "cpt_to_obs_full_std_ratio": float(cpt.std() / obs.std()),
            "cpt_to_obs_anomaly_std_ratio": float(cpt_anom.std() / obs_anom.std()),
        },
        "gridpoint_time_correlations": {
            "cpt_vs_obs": _gridpoint_correlations(cpt, obs),
            "deepscale_vs_obs": _gridpoint_correlations(deep, obs),
        },
        "scores_vs_obs": {
            "deepscale": _score(deep, obs),
            "cpt_raw": _score(cpt, obs),
            "cpt_anomaly_sign_flipped": _score(cpt_sign_flipped, obs),
            "cpt_global_affine_to_obs": _score(cpt_global_affine, obs),
            "cpt_per_cell_affine_to_obs": _score(cpt_cell_affine, obs),
            "cpt_per_year_affine_to_obs": _score(cpt_year_affine, obs),
            "cpt_anomaly_global_affine_to_obs": _score(cpt_anom_global_affine, obs),
        },
        "affine_parameters": {
            "global": global_params,
            "per_cell": cell_params,
            "per_year": year_params,
            "anomaly_global": anom_global_params,
        },
        "cpt_metric_files": {
            "pearson_min": float(_read_cpt_metric_grid(WORKSPACE / "pearson.txt").min()),
            "pearson_max": float(_read_cpt_metric_grid(WORKSPACE / "pearson.txt").max()),
            "rmse_min": float(_read_cpt_metric_grid(WORKSPACE / "root_mean_squared_error.txt").min()),
            "rmse_max": float(_read_cpt_metric_grid(WORKSPACE / "root_mean_squared_error.txt").max()),
        },
        "interpretation": [
            "CPT's own Pearson output is -1 at every grid cell on the synthetic fixture, matching the parsed hindcast values.",
            "CPT's raw full-field output has only about 4.3 percent of the expected anomaly standard deviation.",
            "After subtracting the observed climatology, a single global affine anomaly correction nearly reconstructs the fixture, with slope about -23 and near-zero intercept.",
            "The full-sample CCA decomposition diagnostic already agrees closely, so the remaining CCA parity target is CPT's crossvalidated anomaly scaling/sign convention rather than EOF/CCA fitting or spatial reconstruction.",
        ],
    }
    RESULT.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"Wrote {RESULT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
