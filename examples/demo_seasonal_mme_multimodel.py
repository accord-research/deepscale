"""
End-to-end demo: seasonal_mme() with a multi-track, multi-model MME.

Demonstrates the realistic PyCPT-style use case where the orchestrator pools
several per-(track, model) downscaling runs into one MME. This is the demo
that actually exercises member-contribution diagnostics — the simpler
`demo_seasonal_mme.py` runs with a single track and single model, which
makes the member-contributions page degenerate (one bar, one map).

Data construction note. For a real workflow you would fetch separate
predictor variables for each track (e.g. PRCP on a regional domain and SST
on a tropical-ocean domain). To keep this demo self-contained on a single
CDS fetch, we partition the SEAS5 ensemble's 25 members into four groups
and treat them as four pseudo-"models", split across two pseudo-"tracks".
The MME math is honest — four genuinely-different ensemble-mean predictors
go through CCA and get pooled — but the labels `prcp` / `sst` are
illustrative, not physical.

Run from the repository root:
  python examples/demo_seasonal_mme_multimodel.py

Prerequisites:
  1. Install Rosetta and DeepScale in local virtualenvs.
  2. Configure CDS credentials in ~/.cdsapirc (see rosetta/README.md).
  3. Accept CDS dataset licenses for ERA5 + C3S datasets.

If CDS credentials are absent the script fails at the fetch step.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _configure_import_paths() -> Path:
    """Allow running this example without requiring package installation."""
    repo_root = Path(__file__).resolve().parents[1]
    rosetta_src = repo_root / "rosetta" / "src"
    deepscale_src = repo_root / "src"
    sys.path.insert(0, str(rosetta_src))
    sys.path.insert(0, str(deepscale_src))
    return repo_root


REPO_ROOT = _configure_import_paths()

import xarray as xr
import deepscale

# ---------------------------------------------------------------------------
# Configuration — mirrors demo_seasonal_mme.py for data-layer parity.
# ---------------------------------------------------------------------------
REGION = [-5, 5, 33, 42]          # East Africa [lat_s, lat_n, lon_w, lon_e]
HINDCAST_YEARS = list(range(2000, 2015))
INIT_MONTH = "02"
TARGET = "MAM"
CACHE_DIR = REPO_ROOT / "examples" / "output" / "demo_cache"
OUTPUT_DIR = REPO_ROOT / "examples" / "output"
VERBOSE = True
PROGRESS = True

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Data helpers (identical to demo_seasonal_mme.py).
# ---------------------------------------------------------------------------

def _load_or_fetch(cache_path: Path, fetch_fn):
    if cache_path.exists():
        return xr.open_dataset(cache_path)
    ds = fetch_fn()
    ds.to_netcdf(cache_path)
    return ds


def _era5_to_obs(ds, target_months, years):
    da = ds["temp"]
    seasonal = da.sel(time=da.time.dt.month.isin(target_months))
    annual = seasonal.groupby("time.year").mean("time")
    return annual.sel(year=years)


def _seasonal_to_gcm(ds, years):
    da = ds["temp"]
    keep = {"lat", "lon", "time", "member", "year", "forecast_reference_time", "init_time"}
    for dim in list(da.dims):
        if dim not in keep:
            da = da.mean(dim=dim)
    for dim in ("forecast_reference_time", "init_time", "time"):
        if dim in da.dims:
            da = da.assign_coords(year=(dim, da[dim].dt.year.values))
            da = da.swap_dims({dim: "year"}).drop_vars(dim)
            break
    return da.sel(year=years)


def _partition_members(gcm, n_groups):
    """Split a (year, member, lat, lon) DataArray into `n_groups` along the
    `member` dim. Returns a list of DataArrays with disjoint member subsets.

    Used to fabricate multiple pseudo-"models" from one ensemble for demo
    purposes. In a real workflow each model would come from a different GCM.
    """
    n_members = gcm.sizes["member"]
    edges = [int(round(i * n_members / n_groups)) for i in range(n_groups + 1)]
    return [
        gcm.isel(member=slice(edges[i], edges[i + 1])).assign_coords(
            member=list(range(edges[i + 1] - edges[i]))
        )
        for i in range(n_groups)
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  SEASONAL MME DEMO (MULTI-TRACK, MULTI-MODEL)")
    print("  East Africa MAM Temperature, real CDS data via Rosetta")
    print("=" * 60)

    print("\n[1] ERA5 monthly temperature...")
    try:
        import rosetta
        era5_ds = _load_or_fetch(
            CACHE_DIR / "era5_temp_monthly.nc",
            lambda: rosetta.fetch(
                "obs/era5", "temp",
                hindcast=(HINDCAST_YEARS[0], HINDCAST_YEARS[-1]),
                region=REGION, verbose=VERBOSE, progress=PROGRESS,
            ),
        )
    except Exception as exc:
        print(f"\n[BLOCKED] Could not load/fetch ERA5 obs: {exc}")
        print("  Check ~/.cdsapirc credentials and network access.")
        return

    obs = _era5_to_obs(era5_ds, target_months=[3, 4, 5], years=HINDCAST_YEARS)
    print(f"    obs  {dict(obs.sizes)}  (0.25 deg ERA5)")

    print("\n[2] C3S/ECMWF seasonal hindcast (Feb init -> MAM)...")
    try:
        gcm_ds = _load_or_fetch(
            CACHE_DIR / "c3s_ecmwf_temp_feb_mam.nc",
            lambda: rosetta.fetch(
                "c3s/ecmwf-monthly", "temp",
                init=f"2025-{INIT_MONTH}", target=TARGET,
                hindcast=(HINDCAST_YEARS[0], HINDCAST_YEARS[-1]),
                region=REGION, verbose=VERBOSE, progress=PROGRESS,
            ),
        )
    except Exception as exc:
        print(f"\n[BLOCKED] Could not load/fetch GCM hindcast: {exc}")
        print("  Check ~/.cdsapirc credentials and network access.")
        return

    gcm = _seasonal_to_gcm(gcm_ds, years=HINDCAST_YEARS)
    print(f"    gcm  {dict(gcm.sizes)}  (~1 deg ECMWF SEAS5, 25 members)")

    # ------------------------------------------------------------------
    # 3. Build a 2-track × 2-model `predictor_tracks` dict
    # ------------------------------------------------------------------
    # Split the 25-member ensemble into 4 disjoint subsets, then label them
    # across two pseudo-"tracks" of two pseudo-"models" each. The MME math
    # is honest; the track labels are illustrative. See module docstring.
    sub_a, sub_b, sub_c, sub_d = _partition_members(gcm, n_groups=4)
    print(
        "\n[3] Partitioned 25 SEAS5 members into 4 pseudo-models: "
        f"A={sub_a.sizes['member']}, B={sub_b.sizes['member']}, "
        f"C={sub_c.sizes['member']}, D={sub_d.sizes['member']}"
    )

    predictor_tracks = {
        "prcp": {
            "ECMWF_A": (sub_a, None),
            "ECMWF_B": (sub_b, None),
        },
        "sst": {
            "ECMWF_C": (sub_c, None),
            "ECMWF_D": (sub_d, None),
        },
    }

    print("\n[4] Running seasonal_mme(method='cca', cv='loyo') ...")
    print("    (this runs a LOYO CV loop for each of the 4 (track, model)")
    print("    pairs — may take several minutes)")

    result = deepscale.seasonal_mme(
        predictor_tracks,
        obs,
        method="cca",
        cv="loyo",
        verbose=True,
    )

    # ------------------------------------------------------------------
    # 5. Print headline outputs
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)

    print(f"\n  Years used       : {result.metadata['years_used']}")
    print(f"  Forecast year    : {result.metadata['forecast_year']}")
    print(f"  CV scheme        : {result.metadata['cv']}")
    print(f"  Method           : {result.metadata['method']}")
    print(f"  Tercile method   : {result.metadata['tercile_method']}")
    print(f"  N members        : {result.metadata['n_members']}")

    print("\n  --- Skill scores (domain mean) ---")
    for metric, value in result.skill_report.scores.items():
        if isinstance(value, (int, float)):
            print(f"    {metric:25s}: {value:+.4f}")

    print("\n  --- Member contributions ---")
    mc = result.ensemble_result.member_contributions
    if mc is None:
        print("    (not populated)")
    else:
        for name, entry in mc.items():
            corr = entry["correlation_with_mme_mean"]
            sd_mean = float(entry["skill_delta"].mean())
            print(f"    {name:15s}: corr={corr:+.3f}   skill_delta_mean={sd_mean:+.3f}")

    print(f"\n  tercile_forecast shape : {dict(result.tercile_forecast.sizes)}")
    print(f"  tercile_cv shape       : {dict(result.tercile_cv.sizes)}")

    bn = float(result.tercile_forecast.sel(tercile=0).mean())
    nn = float(result.tercile_forecast.sel(tercile=1).mean())
    an = float(result.tercile_forecast.sel(tercile=2).mean())
    print(f"\n  Headline probabilities (domain mean):")
    print(f"    P(below normal) : {bn:.1%}")
    print(f"    P(normal)       : {nn:.1%}")
    print(f"    P(above normal) : {an:.1%}")

    # ------------------------------------------------------------------
    # 6. Save tercile_forecast + skill PDF
    # ------------------------------------------------------------------
    nc_out = OUTPUT_DIR / "demo_seasonal_mme_multimodel_tercile.nc"
    result.tercile_forecast.to_netcdf(nc_out)
    print(f"\n  Saved tercile_forecast -> {nc_out}")

    pdf_out = OUTPUT_DIR / "demo_seasonal_mme_multimodel_skill.pdf"
    result.skill_report.metadata = {
        "region": "East Africa (5°S-5°N, 33-42°E)",
        "target": TARGET,
        "init": "February",
        "predictand": "2m air temperature",
        "method": "CCA (seasonal_mme, 2 tracks × 2 models)",
    }
    try:
        result.skill_report.to_pdf(pdf_out)
        print(f"  Saved skill PDF     -> {pdf_out}")
    except Exception as exc:
        print(f"  (PDF export skipped: {exc})")

    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
