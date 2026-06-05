"""
Real cross-validated skill (RPSS) of downscaling methods — East Africa MAM precip.

The honest test: leave-one-year-out cross-validation on real CHIRPS + C3S/ECMWF
data, convert each method's held-out predictions to tercile probabilities, and
score with RPSS. RPSS is referenced to climatology, so:

    RPSS  > 0  ->  beats "just predict the average"  (real skill)
    RPSS == 0  ->  no better than climatology
    RPSS  < 0  ->  worse than climatology

Boundaries are computed leave-one-out (no leakage of the held-out year).

Run from the repo root:

    uv run python examples/demo_realdata_skill.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO.parent / "rosetta" / "src"))

import rosetta  # noqa: E402
from deepscale.cv import loyo  # noqa: E402
from deepscale.skill import skill  # noqa: E402
from deepscale.tercile import to_tercile_cv  # noqa: E402
from deepscale.methods.bcsd import BCSDMethod  # noqa: E402
from deepscale.methods.cca import CCAMethod  # noqa: E402
from deepscale.methods.climatology import ClimatologyMethod  # noqa: E402
from deepscale.methods.delta import DeltaScalingMethod  # noqa: E402
from deepscale.methods.dqm import DetrendedQuantileMappingMethod  # noqa: E402
from deepscale.methods.qm import QuantileMappingMethod  # noqa: E402

OBS_REGION = [-12, 15, 22, 52]
GCM_REGION = [-20, 20, 10, 75]
YEARS = (1993, 2016)
MAM = [3, 4, 5]


def load_obs(coarsen=8):
    ds = rosetta.fetch("obs/chirps-v2", "precip", hindcast=YEARS, region=OBS_REGION)
    da = ds["precip"].where(ds["precip"] >= 0)
    annual = da.sel(time=da.time.dt.month.isin(MAM)).groupby("time.year").mean("time")
    annual = annual.coarsen(lat=coarsen, lon=coarsen, boundary="trim").mean()
    return annual.sel(year=slice(YEARS[0], YEARS[1]))


def load_gcm():
    ds = rosetta.fetch("c3s/ecmwf-monthly", "precip", init="2025-02", target="MAM",
                       hindcast=YEARS, region=GCM_REGION)
    da = ds["precip"]
    for dim in ("lead_time", "forecastMonth"):
        if dim in da.dims:
            da = da.mean(dim)
    if "number" in da.dims:
        da = da.rename({"number": "member"})
    for tdim in ("init_time", "time", "forecast_reference_time"):
        if tdim in da.dims:
            da = da.assign_coords(year=(tdim, da[tdim].dt.year.values))
            da = da.swap_dims({tdim: "year"}).drop_vars(tdim)
            break
    return da.sel(lat=slice(OBS_REGION[0], OBS_REGION[1]),
                  lon=slice(OBS_REGION[2], OBS_REGION[3]))


METHODS = [
    ("Climatology", ClimatologyMethod, {}),
    ("Delta", DeltaScalingMethod, {}),
    ("QM (emp)", QuantileMappingMethod, {"variant": "empirical"}),
    ("QM (param)", QuantileMappingMethod, {"variant": "parametric"}),
    ("DQM", DetrendedQuantileMappingMethod, {"variant": "empirical"}),
    ("BCSD", BCSDMethod, {}),
    ("CCA", CCAMethod, {"n_modes": 3}),
]


def cv_rpss(cls, kwargs, gcm, obs, years):
    cv = []
    for train_years, test_year in loyo(years):
        m = cls(**kwargs)
        m.fit(gcm.sel(year=train_years), obs.sel(year=train_years))
        pred = m.predict(gcm.sel(year=test_year)).mean("member")
        cv.append(pred.expand_dims(year=[test_year]))
    cv = xr.concat(cv, dim="year")
    terc = to_tercile_cv(cv, obs, method="bootstrap")
    report = skill(terc, obs, metrics=["rpss"], loo_boundaries=True, cv_window=1)
    return float(report.scores["rpss"])


def main():
    print("=" * 64)
    print("  REAL CV SKILL (RPSS) — East Africa MAM precip, 1993-2016")
    print("=" * 64)

    obs = load_obs()
    gcm = load_gcm()
    years = [int(y) for y in obs.year.values]
    print(f"\n  obs {dict(obs.sizes)} | gcm {dict(gcm.sizes)} | {len(years)} LOYO folds\n")

    rows = []
    for name, cls, kw in METHODS:
        rpss = cv_rpss(cls, kw, gcm, obs, years)
        rows.append((name, rpss))
        print(f"    {name:12s}  RPSS = {rpss:+.3f}  "
              f"{'(beats climatology)' if rpss > 0.005 else '(~ climatology)' if rpss > -0.005 else '(worse)'}")

    print("\n  ranked:")
    for name, rpss in sorted(rows, key=lambda r: -r[1]):
        bar = "#" * max(0, int(round(rpss * 200)))
        print(f"    {name:12s} {rpss:+.3f} {bar}")

    print("\n" + "=" * 64)
    print("  DONE  (RPSS>0 = real skill beyond the long-term average)")
    print("=" * 64)


if __name__ == "__main__":
    main()
