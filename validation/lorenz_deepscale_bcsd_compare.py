"""Compare DeepScale BCSD with the Lorenz et al. WDCC SEAS5-BCSD product.

This is a validation artifact, not production code. It uses:

* Rosetta C3S SEAS5 monthly precipitation as the coarse forecast input.
* Rosetta ERA5-Land monthly precipitation as the reference.
* WDCC SaWaM D03 SEAS5-BCSD as the published paper product.

The comparison is limited to the common Rosetta SEAS5 hindcast period
1993-2016 and to Rosetta's six monthly lead times.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "validation" / "results"
FIGURES = ROOT / "validation" / "figures"

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "validation" / ".matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _json_safe(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _score(name: str, pred: xr.DataArray, ref: xr.DataArray) -> dict:
    pred, ref = xr.align(pred, ref, join="inner")
    valid = np.isfinite(pred.values) & np.isfinite(ref.values)
    if int(valid.sum()) < 2:
        return {"name": name, "n": int(valid.sum()), "bias": None, "mae": None, "rmse": None, "corr": None}
    pv = pred.values[valid].ravel()
    rv = ref.values[valid].ravel()
    diff = pv - rv
    corr = np.corrcoef(pv, rv)[0, 1] if np.std(pv) > 0 and np.std(rv) > 0 else np.nan
    return {
        "name": name,
        "n": int(valid.sum()),
        "bias": float(diff.mean()),
        "mae": float(np.abs(diff).mean()),
        "rmse": float(np.sqrt((diff**2).mean())),
        "corr": float(corr) if np.isfinite(corr) else None,
    }


def _target_month(issue_time: np.datetime64, lead_time: int) -> np.datetime64:
    issue = pd.Timestamp(issue_time).to_period("M")
    return np.datetime64((issue + int(lead_time) - 1).strftime("%Y-%m-01"))


def _wdcc_issue_file(wdcc_dir: Path, issue_time: np.datetime64) -> Path:
    stamp = pd.Timestamp(issue_time).strftime("%Y%m")
    return wdcc_dir / f"SaWaM_D03_SEAS5_BCSD_monthly_pr_{stamp}.nc"


def _load_wdcc_latlon(wdcc_dir: Path) -> tuple[xr.DataArray, xr.DataArray]:
    first = sorted(wdcc_dir.glob("*.nc"))[0]
    with xr.open_dataset(first) as ds:
        return ds["lat"].load(), ds["lon"].load()


def _load_wdcc_prediction(wdcc_dir: Path, issue_times: np.ndarray, lead_time: int) -> xr.DataArray:
    fields = []
    years = []
    time_index = int(lead_time) - 1
    for issue in issue_times:
        path = _wdcc_issue_file(wdcc_dir, issue)
        ds = xr.open_dataset(path)
        field = ds["pr"].isel(time=time_index).mean("ens", skipna=True).load()
        fields.append(field)
        years.append(int(pd.Timestamp(issue).year))
        ds.close()
    return xr.concat(fields, dim=xr.IndexVariable("year", years)).transpose("year", "lat", "lon")


def _make_case(
    seas5: xr.DataArray,
    era5: xr.DataArray,
    wdcc_dir: Path,
    *,
    issue_month: int,
    lead_time: int,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    init = seas5["init_time"]
    mask = init.dt.month == issue_month
    fcst = seas5.sel(init_time=mask).sel(lead_time=lead_time)
    years = [int(pd.Timestamp(t).year) for t in fcst["init_time"].values]
    fcst = fcst.assign_coords(year=("init_time", years)).swap_dims({"init_time": "year"}).drop_vars("init_time")
    fcst = fcst.transpose("year", "member", "lat", "lon")

    wdcc_lat, wdcc_lon = _load_wdcc_latlon(wdcc_dir)
    target_fields = []
    for issue in fcst["year"].values:
        issue_time = np.datetime64(f"{int(issue):04d}-{issue_month:02d}-01")
        target_time = _target_month(issue_time, lead_time)
        target_fields.append(era5.sel(time=target_time).interp(lat=wdcc_lat, lon=wdcc_lon).load())
    obs = xr.concat(target_fields, dim=xr.IndexVariable("year", years)).transpose("year", "lat", "lon")

    raw = fcst.mean("member").interp(lat=wdcc_lat, lon=wdcc_lon).load()
    wdcc = _load_wdcc_prediction(wdcc_dir, [np.datetime64(f"{y:04d}-{issue_month:02d}-01") for y in years], lead_time)
    return fcst.load(), obs, raw, wdcc


def _predict_deepscale_bcsd(hindcast: xr.DataArray, obs: xr.DataArray) -> xr.DataArray:
    from deepscale.methods.bcsd import BCSDMethod

    model = BCSDMethod()
    model.fit(hindcast, obs)
    preds = []
    for year in hindcast["year"].values:
        pred = model.predict(hindcast.sel(year=[year]).isel(year=0, drop=True)).mean("member")
        preds.append(pred.expand_dims(year=[int(year)]))
    return xr.concat(preds, "year").transpose("year", "lat", "lon")


def _plot_summary(scores: list[dict], path: Path) -> None:
    methods = ["raw_seas5", "deepscale_bcsd", "wdcc_bcsd"]
    labels = ["Raw SEAS5", "DeepScale BCSD", "WDCC BCSD"]
    metrics = ["bias", "mae", "rmse", "corr"]
    grouped = {
        method: {
            metric: [row[metric] for row in scores if row["method"] == method and row[metric] is not None]
            for metric in metrics
        }
        for method in methods
    }
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), constrained_layout=True)
    colors = ["#5b6c8c", "#2f8f83", "#c56b47"]
    for ax, metric in zip(axes, metrics):
        vals = [float(np.nanmean(grouped[method][metric])) for method in methods]
        ax.bar(labels, vals, color=colors)
        ax.set_title(metric.upper())
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_heatmaps(scores: list[dict], path: Path) -> None:
    methods = ["raw_seas5", "deepscale_bcsd", "wdcc_bcsd"]
    labels = ["Raw SEAS5", "DeepScale BCSD", "WDCC BCSD"]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.0), constrained_layout=True)
    for col, (method, label) in enumerate(zip(methods, labels)):
        rows = [row for row in scores if row["method"] == method]
        for row_i, metric in enumerate(["rmse", "corr"]):
            arr = np.full((12, 6), np.nan)
            for row in rows:
                arr[int(row["issue_month"]) - 1, int(row["lead_time"]) - 1] = row[metric]
            ax = axes[row_i, col]
            cmap = "YlGnBu" if metric == "rmse" else "viridis"
            mesh = ax.imshow(arr, aspect="auto", origin="lower", cmap=cmap)
            ax.set_title(f"{label} {metric.upper()}")
            ax.set_xlabel("Lead")
            ax.set_ylabel("Issue month")
            ax.set_xticks(np.arange(6), labels=np.arange(1, 7))
            ax.set_yticks(np.arange(12), labels=np.arange(1, 13))
            fig.colorbar(mesh, ax=ax, shrink=0.8)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run(args: argparse.Namespace) -> dict:
    seas5 = xr.open_dataset(args.seas5_nc)["precip"]
    era5 = xr.open_dataset(args.era5_nc)["precip"]
    wdcc_dir = args.wdcc_dir
    years = pd.to_datetime(seas5["init_time"].values).year
    scores = []
    for issue_month in args.issue_months:
        for lead_time in args.leads:
            print(f"case issue_month={issue_month:02d} lead={lead_time}")
            hindcast, obs, raw, wdcc = _make_case(
                seas5,
                era5,
                wdcc_dir,
                issue_month=issue_month,
                lead_time=lead_time,
            )
            deepscale = _predict_deepscale_bcsd(hindcast, obs)
            for method, pred in [("raw_seas5", raw), ("deepscale_bcsd", deepscale), ("wdcc_bcsd", wdcc)]:
                row = _score(method, pred, obs)
                row.update({"method": method, "issue_month": issue_month, "lead_time": lead_time})
                scores.append(row)
            row = _score("deepscale_vs_wdcc", deepscale, wdcc)
            row.update({"method": "deepscale_vs_wdcc", "issue_month": issue_month, "lead_time": lead_time})
            scores.append(row)

    fig_path = FIGURES / f"lorenz_deepscale_bcsd_compare_{args.suffix}.png"
    heatmap_path = FIGURES / f"lorenz_deepscale_bcsd_compare_heatmaps_{args.suffix}.png"
    _plot_summary(scores, fig_path)
    _plot_heatmaps(scores, heatmap_path)
    summary = {}
    for method in ["raw_seas5", "deepscale_bcsd", "wdcc_bcsd", "deepscale_vs_wdcc"]:
        rows = [row for row in scores if row["method"] == method]
        summary[method] = {
            metric: float(np.nanmean([row[metric] for row in rows if row[metric] is not None]))
            for metric in ["bias", "mae", "rmse", "corr"]
        }

    return {
        "status": "ok",
        "role": "First common-period DeepScale BCSD vs WDCC SEAS5-BCSD comparison.",
        "period": [int(years.min()), int(years.max())],
        "issue_months": args.issue_months,
        "leads": args.leads,
        "lead_alignment": "Rosetta lead_time 1..6 matched to WDCC time index 0..5.",
        "inputs": {
            "seas5_nc": str(args.seas5_nc),
            "era5_nc": str(args.era5_nc),
            "wdcc_dir": str(args.wdcc_dir),
        },
        "summary_mean_across_issue_month_lead": summary,
        "scores": scores,
        "figures": [str(fig_path), str(heatmap_path)],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seas5-nc", type=Path, default=Path("/Users/david/data/seas5/seas5_monthly_pr_d03_1993_2016.nc"))
    parser.add_argument("--era5-nc", type=Path, default=Path("/Users/david/data/era5-land/era5_land_monthly_pr_d03_1981_2019.nc"))
    parser.add_argument("--wdcc-dir", type=Path, default=Path("/Users/david/data/wdcc/D03_BCSD_monthly_pr"))
    parser.add_argument("--issue-months", type=int, nargs="+", default=list(range(1, 13)))
    parser.add_argument("--leads", type=int, nargs="+", default=list(range(1, 7)))
    parser.add_argument("--suffix", default="d03_1993_2016")
    parser.add_argument("--out-json", type=Path, default=RESULTS / "lorenz_deepscale_bcsd_compare_d03_1993_2016.json")
    args = parser.parse_args()

    payload = run(args)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(_json_safe(payload), indent=2))
    print(json.dumps(_json_safe(payload["summary_mean_across_issue_month_lead"]), indent=2))
    print(f"Wrote {args.out_json}")
    for figure in payload["figures"]:
        print(f"Wrote {figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
