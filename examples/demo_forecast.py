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
    report = deepscale.skill(
        cv_fcst, obs,
        metrics=[
            "rpss", "pearson_r", "spearman", "hss",
            "2afc", "roc_area_below_normal", "roc_area_above_normal",
            "generalized_roc", "reliability",
        ],
        spatial=True,
    )
    report_det = deepscale.skill(
        cv_fcst_det, obs,
        metrics=["rmse", "spread_error_ratio", "spread_error_correlation"],
        spatial=True,
    )

    print("\n" + "=" * 60)
    print(f"  SKILL REPORT - ECMWF SEAS5 ({best.method.upper()})")
    print("=" * 60)
    for metric, value in report.scores.items():
        print(f"    {metric:20s}: {value:+.3f}")
    for metric, value in report_det.scores.items():
        print(f"    {metric:20s}: {value:+.3f}")

    # ---- Plots --------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from deepscale.skill import SkillReport
        from deepscale.plotting import (
            plot_domains, plot_deterministic_forecast,
            plot_skill_maps, plot_tercile_forecast,
            plot_reliability_diagram,
        )

        OUTPUT_DIR = PLOT_PATH.parent
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # 1. Domain map
        fig = plot_domains(
            predictor_extent=tuple(REGION),
            predictand_extent=tuple(REGION),
            title=f"Domain ({TARGET})",
        )
        out = OUTPUT_DIR / "demo_domains.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved -> {out}")

        # 2. Climatology
        fig = plot_deterministic_forecast(
            obs.mean("year"),
            title="Obs Climatology (ERA5 MAM mean)",
            cmap="inferno",
        )
        out = OUTPUT_DIR / "demo_climatology.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved -> {out}")

        # 3. Skill maps grid (combine probabilistic + deterministic reports)
        combined_report = SkillReport(
            scores={**report.scores, **report_det.scores},
            spatial={**report.spatial, **report_det.spatial},
        )
        fig = plot_skill_maps(
            combined_report,
            ["rpss", "pearson_r", "spearman", "hss", "rmse"],
            ncols=3,
        )
        out = OUTPUT_DIR / "demo_skill_maps.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved -> {out}")

        # 4. Dominant-tercile probability map
        # tercile_forecast may have a year dim; squeeze to (tercile, lat, lon)
        tcst = tercile_forecast
        if "year" in tcst.dims:
            tcst = tcst.isel(year=-1)
        fig = plot_tercile_forecast(
            tcst,
            title=f"Dominant Tercile ({best.method.upper()}, {TARGET})",
            variable_kind="temp",
        )
        out = OUTPUT_DIR / "demo_tercile.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved -> {out}")

        # 4b. Reliability diagram
        fig = plot_reliability_diagram(
            cv_fcst, obs,
            title=f"Reliability ({best.method.upper()}, {TARGET})",
        )
        out = OUTPUT_DIR / "demo_reliability.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved -> {out}")

        # 5. Time-series + bar charts summary (no plotting helper covers this layout)
        from matplotlib.gridspec import GridSpec
        fig = plt.figure(figsize=(18, 5), constrained_layout=True)
        gs = GridSpec(1, 4, figure=fig, wspace=0.3)

        ax_ts = fig.add_subplot(gs[0, :2])
        obs_ts = obs.mean(["lat", "lon"])
        gcm_ts_mean = gcm.mean(["member", "lat", "lon"])
        gcm_member_ts = gcm.mean(["lat", "lon"])
        member_min = gcm_member_ts.min("member").values
        member_max = gcm_member_ts.max("member").values

        ax_ts.plot(HINDCAST_YEARS, obs_ts.values, "k.-", label="ERA5 obs", linewidth=1.5)
        ax_ts.plot(HINDCAST_YEARS, gcm_ts_mean.values, "b.-",
                   label="ECMWF ens mean", linewidth=1.5, alpha=0.8)
        ax_ts.fill_between(HINDCAST_YEARS, member_min, member_max,
                            alpha=0.15, color="blue", label="Ensemble spread")
        ax_ts.set_xlabel("Year")
        ax_ts.set_ylabel("Temperature (deg C)")
        ax_ts.set_title("Area-Mean MAM Temperature: Obs vs GCM Hindcast")
        ax_ts.legend(fontsize=9, loc="best")
        ax_ts.grid(alpha=0.3)

        ax_bar = fig.add_subplot(gs[0, 2])
        # Signed skill chart: only keep metrics on the [-1, 1] scale.
        # RMSE has a different scale (deg C); reliability is calibration error;
        # 2afc + roc_area_* are [0, 1] discrimination skills (next chart).
        signed_excludes = (
            "rmse", "root_mean_squared_error", "reliability",
            "2afc", "roc_area_below_normal", "roc_area_above_normal",
        )
        metrics_to_plot = {
            k: v for k, v in report.scores.items()
            if isinstance(v, (int, float)) and k not in signed_excludes
        }
        names = list(metrics_to_plot.keys())
        vals = list(metrics_to_plot.values())
        colors = ["#d9534f" if v < 0 else "#5cb85c" for v in vals]
        bars = ax_bar.barh(names, vals, color=colors, edgecolor="gray", height=0.5)
        ax_bar.axvline(0, color="black", linewidth=0.8)
        ax_bar.set_xlabel("Score")
        ax_bar.set_title("Domain-Mean Skill Scores")
        ax_bar.set_xlim(-0.5, 0.5)
        for bar, value in zip(bars, vals):
            x_offset = 0.02 if value >= 0 else -0.06
            ax_bar.text(value + x_offset, bar.get_y() + bar.get_height() / 2,
                        f"{value:+.3f}", va="center", fontsize=10)

        # Second bar chart: [0, 1] discrimination skills with 0.5 = no skill.
        ax_bar2 = fig.add_subplot(gs[0, 3])
        disc_metrics = {
            k: v for k, v in report.scores.items()
            if k in ("2afc", "roc_area_below_normal", "roc_area_above_normal")
        }
        if disc_metrics:
            names2 = list(disc_metrics.keys())
            vals2 = list(disc_metrics.values())
            colors2 = ["#5cb85c" if v >= 0.5 else "#d9534f" for v in vals2]
            bars2 = ax_bar2.barh(names2, vals2, color=colors2, edgecolor="gray", height=0.5)
            ax_bar2.axvline(0.5, color="black", linewidth=0.8, linestyle="--",
                            label="No skill (0.5)")
            ax_bar2.set_xlabel("Score")
            ax_bar2.set_title("Discrimination Skill")
            ax_bar2.set_xlim(0, 1)
            for bar, value in zip(bars2, vals2):
                ax_bar2.text(value + 0.02, bar.get_y() + bar.get_height() / 2,
                             f"{value:.3f}", va="center", fontsize=9)
            ax_bar2.legend(loc="lower right", fontsize=8)

        fig.suptitle(
            f"East Africa {TARGET} - Seasonal Temperature Forecast & Skill\n"
            f"ECMWF SEAS5 {best.method.upper()} | {HINDCAST_YEARS[0]}-{HINDCAST_YEARS[-1]} | Real CDS Data",
            fontsize=12, fontweight="bold",
        )
        out = PLOT_PATH  # demo_forecast.png — preserves existing entry point
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved -> {out}")

    except ImportError:
        print("\n    (plotting deps not installed - skipping plots; install with `pip install deepscale[plotting]`)")

    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
