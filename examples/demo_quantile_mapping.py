"""
Demo: Quantile Mapping bias correction + downscaling (#42, §19.1).

Shows both variants correcting a deliberately biased GCM (mean 10, wide) toward
the observed climatology (mean 5, narrow):
  - empirical : nonparametric CDF matching, F_obs^-1(F_gcm(x))
  - parametric: Gaussian z-score rescaling

Network-free — synthetic data on a shared grid. Run from the repo root:

    uv run python examples/demo_quantile_mapping.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deepscale.methods.qm import QuantileMappingMethod  # noqa: E402


def _data():
    rng = np.random.default_rng(1)
    years = np.arange(2000, 2030)
    members = np.arange(5)
    lat, lon = np.linspace(0, 4, 4), np.linspace(30, 34, 4)
    gcm = xr.DataArray(
        rng.normal(10.0, 3.0, (len(years), len(members), 4, 4)),
        dims=["year", "member", "lat", "lon"],
        coords={"year": years, "member": members, "lat": lat, "lon": lon},
    )
    obs = xr.DataArray(
        rng.normal(5.0, 1.0, (len(years), 4, 4)),
        dims=["year", "lat", "lon"],
        coords={"year": years, "lat": lat, "lon": lon},
    )
    return gcm, obs


def main() -> None:
    print("=" * 60)
    print("  QUANTILE MAPPING DEMO  (#42 §19.1)")
    print("=" * 60)

    gcm, obs = _data()
    forecast = gcm.isel(year=-1)
    obs_mean = float(obs.mean())

    print(f"\n    obs climatology mean   = {obs_mean:.2f}")
    print(f"    raw GCM forecast mean  = {float(forecast.mean()):.2f}  "
          f"(bias {float(forecast.mean()) - obs_mean:+.2f})")

    for variant in ("empirical", "parametric"):
        m = QuantileMappingMethod(variant=variant)
        m.fit(gcm, obs)
        out = m.predict(forecast)
        bias = float(out.mean()) - obs_mean
        print(f"    {variant:10s} QM mean    = {float(out.mean()):.2f}  "
              f"(bias {bias:+.2f})")

    print("\n    -> both variants pull the forecast onto the obs distribution.")
    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
