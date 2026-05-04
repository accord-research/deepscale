"""
End-to-end demo: seasonal temperature forecast for East Africa (MAM).

This example uses real CDS data via Rosetta:
  - ERA5 monthly means (observations)
  - C3S/ECMWF seasonal hindcasts (GCM)

Run from the repository root:
  python deepscale/examples/demo_forecast.py

Prerequisites:
  1. Install Rosetta and DeepScale in local virtualenvs.
  2. Configure CDS credentials in ~/.cdsapirc (see rosetta/README.md).
  3. Accept CDS dataset licenses for ERA5 + C3S datasets.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import xarray as xr


def _configure_import_paths() -> Path:
    """Allow running this example without requiring package installation."""
    repo_root = Path(__file__).resolve().parents[2]
    rosetta_src = repo_root / "rosetta" / "src"
    deepscale_src = repo_root / "deepscale" / "src"
    sys.path.insert(0, str(rosetta_src))
    sys.path.insert(0, str(deepscale_src))
    return repo_root


REPO_ROOT = _configure_import_paths()

import deepscale
import rosetta
from deepscale.cv import loyo
from deepscale.tercile import to_tercile

# Configuration
REGION = [-5, 5, 33, 42]  # East Africa [lat_s, lat_n, lon_w, lon_e]
HINDCAST_YEARS = list(range(2000, 2015))
INIT_MONTH = "02"
TARGET = "MAM"
CACHE_DIR = REPO_ROOT / "deepscale" / "examples" / "output" / "demo_cache"
PLOT_PATH = REPO_ROOT / "deepscale" / "examples" / "output" / "demo_forecast.png"
VERBOSE = True
PROGRESS = True

os.makedirs(CACHE_DIR, exist_ok=True)


def load_or_fetch(cache_path: Path, fetch_fn):
    """Load cached dataset when present; otherwise fetch and cache."""
    if cache_path.exists():
        return xr.open_dataset(cache_path)
    ds = fetch_fn()
    ds.to_netcdf(cache_path)
    return ds


def era5_to_obs(ds, target_months, years):
    """ERA5 monthly Dataset -> seasonal-mean obs DataArray (year, lat, lon)."""
    da = ds["temp"]
    seasonal = da.sel(time=da.time.dt.month.isin(target_months))
    annual = seasonal.groupby("time.year").mean("time")
    return annual.sel(year=years)


def seasonal_to_gcm(ds, years):
    """C3S seasonal-monthly Dataset -> GCM DataArray (year, member, lat, lon)."""
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


def main() -> None:
    print("=" * 60)
    print("  SEASONAL TEMPERATURE FORECAST - East Africa MAM")
    print("  (real CDS data via Rosetta + DeepScale)")
    print("=" * 60)

    print("\n[1] ERA5 monthly temperature...")
    era5_ds = load_or_fetch(
        CACHE_DIR / "era5_temp_monthly.nc",
        lambda: rosetta.fetch(
            "obs/era5",
            "temp",
            hindcast=(HINDCAST_YEARS[0], HINDCAST_YEARS[-1]),
            region=REGION,
            verbose=VERBOSE,
            progress=PROGRESS,
        ),
    )
    obs = era5_to_obs(era5_ds, target_months=[3, 4, 5], years=HINDCAST_YEARS)
    print(f"    obs  {dict(obs.sizes)}  (0.25 deg ERA5)")

    print("\n[2] C3S/ECMWF seasonal hindcast (Feb init -> MAM)...")
    gcm_ds = load_or_fetch(
        CACHE_DIR / "c3s_ecmwf_temp_feb_mam.nc",
        lambda: rosetta.fetch(
            "c3s/ecmwf-monthly",
            "temp",
            init=f"2025-{INIT_MONTH}",
            target=TARGET,
            hindcast=(HINDCAST_YEARS[0], HINDCAST_YEARS[-1]),
            region=REGION,
            verbose=VERBOSE,
            progress=PROGRESS,
        ),
    )
    gcm = seasonal_to_gcm(gcm_ds, years=HINDCAST_YEARS)
    print(f"    gcm  {dict(gcm.sizes)}  (~1 deg ECMWF SEAS5)")

    print("\n[3] Optimizing across methods (BCSD, CCA)...")
    best = deepscale.optimize(
        gcm,
        obs,
        methods=["bcsd", "cca"],
        primary_metric="rpss",
        verbose=VERBOSE,
        progress=PROGRESS,
    )
    print(f"    Best method: {best.method}  |  CV RPSS: {best.score:.3f}")

    print("\n[4] Computing tercile probabilities...")
    tercile_forecast = to_tercile(best.forecast, obs)

    bn = float(tercile_forecast.sel(tercile=0).mean())
    nn = float(tercile_forecast.sel(tercile=1).mean())
    an = float(tercile_forecast.sel(tercile=2).mean())
    print(f"    P(below normal):  {bn:.1%}")
    print(f"    P(normal):        {nn:.1%}")
    print(f"    P(above normal):  {an:.1%}")

    print("\n[5] LOYO cross-validated skill (this may take a minute)...")
    cv_forecasts_terc = []
    cv_forecasts_det = []
    for train_years, _test_year in loyo(HINDCAST_YEARS):
        fitted = deepscale.optimize(
            gcm.sel(year=train_years),
            obs.sel(year=train_years),
            methods=[best.method],
            verbose=VERBOSE,
            progress=PROGRESS,
        )
        cv_forecasts_terc.append(to_tercile(fitted.forecast, obs.sel(year=train_years)))
        cv_forecasts_det.append(fitted.forecast)

    cv_fcst = xr.concat(cv_forecasts_terc, dim="year")
    cv_fcst["year"] = HINDCAST_YEARS
    cv_fcst_det = xr.concat(cv_forecasts_det, dim="year")
    cv_fcst_det["year"] = HINDCAST_YEARS

    # Probabilistic metrics need the tercile forecast; RMSE needs the raw deterministic
    # ensemble (same units as obs). Two skill calls keep each metric on the right input.
    report = deepscale.skill(cv_fcst, obs, metrics=["rpss", "pearson_r", "hss"], spatial=True)
    report_det = deepscale.skill(cv_fcst_det, obs, metrics=["rmse"], spatial=True)

    print("\n" + "=" * 60)
    print(f"  SKILL REPORT - ECMWF SEAS5 ({best.method.upper()})")
    print("=" * 60)
    for metric, value in report.scores.items():
        print(f"    {metric:20s}: {value:+.3f}")
    for metric, value in report_det.scores.items():
        print(f"    {metric:20s}: {value:+.3f}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=(18, 18))
        gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.30)

        tercile_titles = ["P(Below Normal)", "P(Normal)", "P(Above Normal)"]
        for i, title in enumerate(tercile_titles):
            ax = fig.add_subplot(gs[0, i], aspect="equal")
            data = tercile_forecast.sel(tercile=i)
            image = ax.pcolormesh(obs.lon, obs.lat, data.values, vmin=0, vmax=1, cmap="RdYlGn")
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("Lon")
            if i == 0:
                ax.set_ylabel("Lat")
            plt.colorbar(image, ax=ax, fraction=0.046, label="Probability")

        skill_panels = []
        if "rpss" in report.spatial:
            skill_panels.append(("RPSS", report.spatial["rpss"], "RdBu", (-1, 1)))
        if "pearson_r" in report.spatial:
            skill_panels.append(("Pearson r", report.spatial["pearson_r"], "RdBu", (-1, 1)))

        for j, (label, spatial_da, cmap, (vlo, vhi)) in enumerate(skill_panels):
            ax = fig.add_subplot(gs[1, j], aspect="equal")
            norm = TwoSlopeNorm(vmin=vlo, vcenter=0, vmax=vhi)
            image = ax.pcolormesh(spatial_da.lon, spatial_da.lat, spatial_da.values, cmap=cmap, norm=norm)
            ax.set_title(f"Spatial {label} (LOYO CV)", fontsize=11)
            ax.set_xlabel("Lon")
            if j == 0:
                ax.set_ylabel("Lat")
            plt.colorbar(image, ax=ax, fraction=0.046, label=label)

        ax_clim = fig.add_subplot(gs[1, 2], aspect="equal")
        obs_clim = obs.mean("year")
        image = ax_clim.pcolormesh(obs.lon, obs.lat, obs_clim.values, cmap="inferno")
        ax_clim.set_title("Obs Climatology (ERA5 MAM mean)", fontsize=11)
        ax_clim.set_xlabel("Lon")
        plt.colorbar(image, ax=ax_clim, fraction=0.046, label="deg C")

        if "rmse" in report_det.spatial:
            ax_rmse = fig.add_subplot(gs[2, :])
            rmse_da = report_det.spatial["rmse"]
            image = ax_rmse.pcolormesh(rmse_da.lon, rmse_da.lat, rmse_da.values, cmap="viridis")
            ax_rmse.set_title("Spatial RMSE (LOYO CV) — deg C, lower is better", fontsize=11)
            ax_rmse.set_xlabel("Lon")
            ax_rmse.set_ylabel("Lat")
            plt.colorbar(image, ax=ax_rmse, fraction=0.025, label="RMSE (deg C)")

        ax_ts = fig.add_subplot(gs[3, :2])
        obs_ts = obs.mean(["lat", "lon"])
        gcm_ts_mean = gcm.mean(["member", "lat", "lon"])
        gcm_member_ts = gcm.mean(["lat", "lon"])
        member_min = gcm_member_ts.min("member").values
        member_max = gcm_member_ts.max("member").values

        ax_ts.plot(HINDCAST_YEARS, obs_ts.values, "k.-", label="ERA5 obs", linewidth=1.5)
        ax_ts.plot(
            HINDCAST_YEARS,
            gcm_ts_mean.values,
            "b.-",
            label="ECMWF ens mean",
            linewidth=1.5,
            alpha=0.8,
        )
        ax_ts.fill_between(HINDCAST_YEARS, member_min, member_max, alpha=0.15, color="blue", label="Ensemble spread")
        ax_ts.set_xlabel("Year")
        ax_ts.set_ylabel("Temperature (deg C)")
        ax_ts.set_title("Area-Mean MAM Temperature: Obs vs GCM Hindcast", fontsize=11)
        ax_ts.legend(fontsize=9, loc="best")
        ax_ts.grid(alpha=0.3)

        ax_bar = fig.add_subplot(gs[3, 2])
        # RMSE excluded from the signed-skill bar chart — different scale (deg C vs ±1).
        metrics_to_plot = {k: v for k, v in report.scores.items() if isinstance(v, (int, float)) and k != "rmse"}
        names = list(metrics_to_plot.keys())
        vals = list(metrics_to_plot.values())
        colors = ["#d9534f" if v < 0 else "#5cb85c" for v in vals]
        bars = ax_bar.barh(names, vals, color=colors, edgecolor="gray", height=0.5)
        ax_bar.axvline(0, color="black", linewidth=0.8)
        ax_bar.set_xlabel("Score")
        ax_bar.set_title("Domain-Mean Skill Scores", fontsize=11)
        ax_bar.set_xlim(-0.5, 0.5)
        for bar, value in zip(bars, vals):
            x_offset = 0.02 if value >= 0 else -0.06
            ax_bar.text(value + x_offset, bar.get_y() + bar.get_height() / 2, f"{value:+.3f}", va="center", fontsize=10)

        fig.suptitle(
            f"East Africa {TARGET} - Seasonal Temperature Forecast & Skill\n"
            f"ECMWF SEAS5 {best.method.upper()} | {HINDCAST_YEARS[0]}-{HINDCAST_YEARS[-1]} | Real CDS Data",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )
        PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
        print(f"\n    Saved -> {PLOT_PATH}")
    except ImportError:
        print("\n    (matplotlib not installed - skipping plot)")

    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
