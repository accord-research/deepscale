"""BCSD validation against scikit-downscale and a reference implementation.

This is a validation artifact. scikit-downscale's BCSD API is pointwise and
time-series oriented, while DeepScale's BCSD API predicts a held-out field.
The adapter below labels that difference explicitly.

The reference implementation is an independent, validation-only reconstruction of
the classic BCSD workflow: empirical quantile mapping at the coarse grid,
linear interpolation to the fine grid, and restoration of observed fine-grid
climatological spatial detail.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from downscaling_validation import PRESETS, fetch_chirps, make_hindcast, score_prediction

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "validation" / "results"
FIGURES = ROOT / "validation" / "figures"

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "validation" / ".matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class PairScore:
    comparison: str
    n_cells: int
    bias: float
    mae: float
    rmse: float
    corr: float
    max_abs_diff: float


def _make_time_index(years) -> pd.DatetimeIndex:
    return pd.to_datetime([f"{int(y)}-07-01" for y in years])


def _score_pair(name: str, a: xr.DataArray, b: xr.DataArray) -> PairScore:
    a, b = xr.align(a.transpose("year", "lat", "lon"), b.transpose("year", "lat", "lon"), join="inner")
    valid = np.isfinite(a.values) & np.isfinite(b.values)
    av = a.values[valid].ravel()
    bv = b.values[valid].ravel()
    diff = av - bv
    corr = np.corrcoef(av, bv)[0, 1] if np.std(av) > 0 and np.std(bv) > 0 else np.nan
    return PairScore(
        comparison=name,
        n_cells=int(valid.sum()),
        bias=float(diff.mean()),
        mae=float(np.abs(diff).mean()),
        rmse=float(np.sqrt((diff**2).mean())),
        corr=float(corr),
        max_abs_diff=float(np.abs(diff).max()),
    )


def _run_deepscale_bcsd(hindcast: xr.DataArray, obs: xr.DataArray) -> xr.DataArray:
    from deepscale.registry import get_method

    cls = get_method("bcsd")
    preds = []
    years = list(obs.year.values)
    for year in years:
        train_years = [y for y in years if y != year]
        model = cls()
        model.fit(hindcast.sel(year=train_years), obs.sel(year=train_years))
        fcst = hindcast.sel(year=[year]).isel(year=0, drop=True)
        pred = model.predict(fcst).mean("member")
        preds.append(pred.expand_dims(year=[year]))
    return xr.concat(preds, "year").rename("deepscale_bcsd")


def _fit_predict_scikit_series(hist: np.ndarray, ref: np.ndarray, sim_series: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    from skdownscale.pointwise_models import BcsdPrecipitation

    years = np.arange(sim_series.shape[0])
    index = _make_time_index(2000 + years)
    train_index = index[train_mask]
    x_train = pd.DataFrame(hist[train_mask], index=train_index)
    y_train = pd.DataFrame(ref[train_mask], index=train_index)
    x_all = pd.DataFrame(sim_series, index=index)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = BcsdPrecipitation(return_anoms=False).fit(x_train, y_train)
        pred = model.predict(x_all)
    return np.asarray(pred).reshape(-1)


def _run_scikit_bcsd_series_context(hindcast: xr.DataArray, obs: xr.DataArray) -> xr.DataArray:
    """Run scikit-downscale BCSD in a series-context LOYO mode.

    The model is fit with training target years only, but its pointwise API
    ranks the full simulation series during predict. We therefore pass the full
    pseudo-model series and select the held-out year. This uses no held-out obs,
    but it is not the same contract as DeepScale's single-field prediction.
    """
    hist = hindcast.mean("member")
    ref = obs.interp(lat=hist.lat, lon=hist.lon, method="linear")
    sim = hist
    years = np.asarray(obs.year.values)
    coarse_preds = np.full(sim.shape, np.nan, dtype=float)

    for i in range(sim.sizes["lat"]):
        for j in range(sim.sizes["lon"]):
            hist_col = hist.isel(lat=i, lon=j).values
            ref_col = ref.isel(lat=i, lon=j).values
            sim_col = sim.isel(lat=i, lon=j).values
            if not (np.all(np.isfinite(hist_col)) and np.all(np.isfinite(ref_col)) and np.all(np.isfinite(sim_col))):
                continue
            for yi, year in enumerate(years):
                train_mask = years != year
                try:
                    pred_series = _fit_predict_scikit_series(hist_col, ref_col, sim_col, train_mask)
                    coarse_preds[yi, i, j] = pred_series[yi]
                except Exception:
                    coarse_preds[yi, i, j] = np.nan

    coarse = xr.DataArray(
        coarse_preds,
        dims=("year", "lat", "lon"),
        coords={"year": obs.year, "lat": hist.lat, "lon": hist.lon},
        name="scikit_bcsd_series_context",
    )

    fine_preds = []
    for year in years:
        train_years = [y for y in years if y != year]
        obs_clim = obs.sel(year=train_years).mean("year")
        obs_clim_coarse = obs_clim.interp(lat=hist.lat, lon=hist.lon, method="linear")
        obs_clim_back = obs_clim_coarse.interp(lat=obs.lat, lon=obs.lon, method="linear")
        spatial_detail = obs_clim - obs_clim_back
        fine = coarse.sel(year=year).interp(lat=obs.lat, lon=obs.lon, method="linear") + spatial_detail
        fine_preds.append(fine.expand_dims(year=[year]))
    return xr.concat(fine_preds, "year").rename("scikit_bcsd_series_context")


def _empirical_qm_like_bcsd(forecast: np.ndarray, hist: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Validation-only empirical QM convention used by the BCSD reference implementation.

    This pins the otherwise-under-specified paper step to the same declared
    convention as the current method audit: sorted empirical distributions,
    searchsorted plotting positions, linear interpolation between reference
    order statistics, and bounded tails.
    """
    n_years, n_lat, n_lon = hist.shape
    hist_sorted = np.sort(hist, axis=0)
    ref_sorted = np.sort(ref, axis=0)
    corrected = np.full_like(forecast, np.nan, dtype=float)
    for i in range(n_lat):
        for j in range(n_lon):
            h = hist_sorted[:, i, j]
            r = ref_sorted[:, i, j]
            f = forecast[i, j]
            if not (np.all(np.isfinite(h)) and np.all(np.isfinite(r)) and np.isfinite(f)):
                continue
            quantile = np.searchsorted(h, f) / n_years
            quantile = np.clip(quantile, 0.01, 0.99)
            idx = quantile * (n_years - 1)
            lo = int(np.floor(idx))
            hi = min(lo + 1, n_years - 1)
            corrected[i, j] = r[lo] * (1.0 - (idx - lo)) + r[hi] * (idx - lo)
    return corrected


def _run_bcsd_rebuild(hindcast: xr.DataArray, obs: xr.DataArray) -> xr.DataArray:
    """Run a compact literature-style BCSD reconstruction in LOYO mode."""
    gcm_mean = hindcast.mean("member")
    years = np.asarray(obs.year.values)
    fine_preds = []
    for year in years:
        train_years = [y for y in years if y != year]
        hist = gcm_mean.sel(year=train_years)
        ref = obs.sel(year=train_years).interp(lat=hist.lat, lon=hist.lon, method="linear")
        sim = gcm_mean.sel(year=year)

        corrected = xr.DataArray(
            _empirical_qm_like_bcsd(sim.values, hist.values, ref.values),
            dims=("lat", "lon"),
            coords={"lat": hist.lat, "lon": hist.lon},
            name="bcsd_rebuild_coarse",
        )
        fine = corrected.interp(lat=obs.lat, lon=obs.lon, method="linear")

        obs_clim = obs.sel(year=train_years).mean("year")
        obs_clim_coarse = obs_clim.interp(lat=hist.lat, lon=hist.lon, method="linear")
        obs_clim_back = obs_clim_coarse.interp(lat=obs.lat, lon=obs.lon, method="linear")
        spatial_detail = obs_clim - obs_clim_back
        fine_preds.append((fine + spatial_detail).clip(min=0.0).expand_dims(year=[year]))
    return xr.concat(fine_preds, "year").rename("bcsd_rebuild")


def _plot_metrics(scores: list[dict], pair_scores: list[PairScore], suffix: str) -> None:
    label_by_reference = {
        "deepscale": "DeepScale",
        "scikit-downscale-series-context": "scikit\nseries",
        "literature-style-rebuild": "Reference\nBCSD",
    }
    labels = [label_by_reference.get(s["reference"], s["reference"]) for s in scores]
    colors = []
    for score in scores:
        if score["reference"] == "deepscale":
            colors.append("#2f6f9f")
        elif score["reference"] == "literature-style-rebuild":
            colors.append("#b45f06")
        else:
            colors.append("#6aa84f")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.5), constrained_layout=True)
    axes[0].bar(labels, [s["rmse"] for s in scores], color=colors)
    axes[0].set_title("BCSD Reference: RMSE vs Obs")
    axes[0].set_ylabel("RMSE (mm/day)")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(labels, [s["corr"] for s in scores], color=colors)
    axes[1].set_title("BCSD Reference: Correlation vs Obs")
    axes[1].set_ylabel("Pearson correlation")
    axes[1].set_ylim(0.9, 1.0)
    axes[1].grid(axis="y", alpha=0.25)
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"bcsd_reference_metric_bars_{suffix}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {path}")

    if pair_scores:
        pair_label_by_name = {
            "deepscale_vs_scikit_series_context": "DeepScale\nvs scikit",
            "deepscale_vs_bcsd_rebuild": "DeepScale\nvs reference",
            "scikit_series_context_vs_bcsd_rebuild": "scikit\nvs reference",
        }
        labels = [pair_label_by_name.get(p.comparison, p.comparison.replace("_", "\n")) for p in pair_scores]
        fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
        ax.bar(labels, [p.rmse for p in pair_scores], color="#b45f06")
        ax.set_title("BCSD Prediction Difference")
        ax.set_ylabel("RMSE (mm/day)")
        ax.grid(axis="y", alpha=0.25)
        path = FIGURES / f"bcsd_reference_difference_{suffix}.png"
        fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Wrote {path}")


def _plot_single_map(obs: xr.DataArray, pred: xr.DataArray, suffix: str, label: str, filename_label: str) -> None:
    year = int(obs.year.values[-1])
    if "ethiopia" in suffix:
        fixture_label = "Ethiopia"
    elif "texas" in suffix:
        fixture_label = "Texas"
    else:
        fixture_label = f"Example held-out year {year}"
    obs_y = obs.sel(year=year).transpose("lat", "lon")
    pred_y = pred.sel(year=year).transpose("lat", "lon")
    fields = [obs_y, pred_y]
    vmin = float(xr.concat(fields, "field", coords="minimal", compat="override").quantile(0.02))
    vmax = float(xr.concat(fields, "field", coords="minimal", compat="override").quantile(0.98))
    diff = pred_y - obs_y
    diff_max = float(abs(diff).quantile(0.98))
    diff_max = max(diff_max, 1e-6)

    fig = plt.figure(figsize=(14.2, 4.6))
    gs = fig.add_gridspec(1, 6, width_ratios=[1, 1, 0.055, 0.22, 1, 0.055], wspace=0.38)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 4])]
    precip_cax = fig.add_subplot(gs[0, 2])
    diff_cax = fig.add_subplot(gs[0, 5])
    for ax, da, title in zip(axes[:2], fields, [f"CHIRPS observed ({year})", label]):
        mesh = ax.pcolormesh(da.lon, da.lat, da, cmap="YlGnBu", vmin=vmin, vmax=vmax, shading="auto")
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    fig.colorbar(mesh, cax=precip_cax, label="Precipitation (mm/day)")

    dm = axes[2].pcolormesh(diff.lon, diff.lat, diff, cmap="RdBu_r", vmin=-diff_max, vmax=diff_max, shading="auto")
    axes[2].set_title(f"{label} minus CHIRPS")
    axes[2].set_xlabel("Longitude")
    axes[2].set_ylabel("Latitude")
    fig.colorbar(dm, cax=diff_cax, label="Difference (mm/day)")
    fig.suptitle(fixture_label, fontsize=13)
    fig.subplots_adjust(top=0.82)
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"bcsd_{filename_label}_maps_{suffix}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {path}")


def _plot_maps(obs: xr.DataArray, deep: xr.DataArray, sk: xr.DataArray, rebuild: xr.DataArray, suffix: str) -> None:
    _plot_single_map(obs, deep, suffix, "DeepScale BCSD", "deepscale")
    _plot_single_map(obs, sk, suffix, "scikit BCSD", "scikit")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=sorted(PRESETS), default="texas")
    parser.add_argument("--product")
    parser.add_argument("--start-year", type=int)
    parser.add_argument("--end-year", type=int)
    parser.add_argument("--bbox", nargs=4, type=float)
    parser.add_argument("--coarsen-factor", type=int, default=3)
    parser.add_argument("--members", type=int, default=3)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--out-json")
    parser.add_argument("--out-nc")
    args = parser.parse_args()

    preset = PRESETS[args.preset]
    product = args.product or preset["product"]
    start_year = args.start_year if args.start_year is not None else preset["start_year"]
    end_year = args.end_year if args.end_year is not None else preset["end_year"]
    bbox = args.bbox or preset["bbox"]
    months = preset.get("months")
    suffix = args.preset
    out_json = Path(args.out_json) if args.out_json else RESULTS / f"bcsd_reference_validation_{suffix}.json"
    out_nc = Path(args.out_nc) if args.out_nc else RESULTS / f"bcsd_reference_predictions_{suffix}.nc"

    obs = fetch_chirps(product, (start_year, end_year), bbox, cache=not args.no_cache, months=months)
    hindcast = make_hindcast(obs, args.coarsen_factor, args.members)
    deep = _run_deepscale_bcsd(hindcast, obs)
    sk = _run_scikit_bcsd_series_context(hindcast, obs)
    rebuild = _run_bcsd_rebuild(hindcast, obs)

    deep_score = asdict(score_prediction("bcsd", "deepscale", deep, obs))
    sk_score = asdict(score_prediction("bcsd", "scikit-downscale-series-context", sk, obs))
    rebuild_score = asdict(score_prediction("bcsd", "literature-style-rebuild", rebuild, obs))
    pair_scores = [
        _score_pair("deepscale_vs_scikit_series_context", deep, sk),
        _score_pair("deepscale_vs_bcsd_rebuild", deep, rebuild),
        _score_pair("scikit_series_context_vs_bcsd_rebuild", sk, rebuild),
    ]
    payload = {
        "preset": args.preset,
        "product": product,
        "requested_years": [start_year, end_year],
        "actual_years": [int(y) for y in obs.year.values],
        "bbox": bbox,
        "months": months,
        "prediction_file": str(out_nc),
        "scores": [deep_score, sk_score, rebuild_score],
        "pair_scores": [asdict(pair) for pair in pair_scores],
        "notes": [
            "scikit-downscale BCSD is pointwise and series-oriented; this adapter passes the full pseudo-model series during predict and selects the held-out year",
            "The adapter uses BcsdPrecipitation(return_anoms=False) at the coarse grid, then applies DeepScale's climatological fine-grid detail reconstruction",
            "The reference implementation reconstructs classic BCSD with coarse empirical quantile mapping, fine-grid interpolation, and observed climatological spatial-detail restoration",
            "The reference implementation pins under-specified paper conventions to sorted empirical distributions, searchsorted plotting positions, linear interpolation between order statistics, bounded tails, linear spatial interpolation, and non-negative precipitation clipping",
            "Run with scikit-learn<1.6 because current scikit-downscale 0.0.0 calls sklearn estimator internals removed from newer scikit-learn",
        ],
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    xr.Dataset(
        {
            "obs": obs,
            "deepscale_bcsd": deep,
            "scikit_bcsd_series_context": sk,
            "bcsd_rebuild": rebuild,
        }
    ).to_netcdf(out_nc)
    out_json.write_text(json.dumps(payload, indent=2))
    _plot_metrics(payload["scores"], pair_scores, suffix)
    _plot_maps(obs, deep, sk, rebuild, suffix)
    print(f"Wrote {out_json}")
    print(f"Wrote {out_nc}")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
