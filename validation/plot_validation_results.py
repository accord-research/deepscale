"""Generate plots for DeepScale downscaling validation artifacts.

This is validation/reporting code, not production package code. It reads the
JSON/NetCDF outputs produced by the validation harnesses and writes PNG figures
under ``validation/figures``.

Cartopy is optional. If installed, maps use a PlateCarree projection with
coastlines/borders; otherwise the script falls back to plain lon/lat heatmaps.
"""

from __future__ import annotations

import json
import os
import argparse
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "validation" / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULTS = ROOT / "validation" / "results"
FIGURES = ROOT / "validation" / "figures"


def _pycpt_cca_paths(tag: str) -> dict[str, Path]:
    stem = "pycpt_cca" if tag == "texas" else f"pycpt_cca_{tag}"
    return {
        "deepscale": RESULTS / f"{stem}_deepscale_loyo.nc",
        "cpt": RESULTS / f"{stem}_cptcore_crossvalidation.nc",
        "comparison": RESULTS / f"{stem}_comparison.json",
    }


def _try_cartopy():
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        return ccrs, cfeature
    except Exception:
        return None, None


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {path}")


def _plot_field(ax, da: xr.DataArray, *, title: str, cmap: str, vmin=None, vmax=None):
    ccrs, cfeature = _try_cartopy()
    lon = da["lon"].values
    lat = da["lat"].values
    if ccrs:
        mesh = ax.pcolormesh(lon, lat, da.values, transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax)
        ax.coastlines(resolution="50m", linewidth=0.7)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4)
        ax.set_extent([float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())], crs=ccrs.PlateCarree())
        gl = ax.gridlines(draw_labels=True, linewidth=0.2, color="0.5", alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
    else:
        mesh = ax.pcolormesh(lon, lat, da.values, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    ax.set_title(title, fontsize=10)
    return mesh


def plot_metric_bars(input_json: Path | None = None, figure_name: str = "chirps_method_metric_bars.png", title: str = "CHIRPS High/Low Benchmark") -> None:
    payload = json.loads((input_json or (RESULTS / "downscaling_validation_chirps.json")).read_text())
    rows = [r for r in payload["scores"] if r["status"] == "ok"]
    rows = sorted(rows, key=lambda r: (r["rmse"], -r["corr"]))
    labels = [f"{r['method']}\n{r['reference']}" for r in rows]
    rmse = [r["rmse"] for r in rows]
    corr = [r["corr"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    colors = ["#2f6f9f" if r["reference"] == "deepscale" else "#b45f06" for r in rows]
    axes[0].bar(labels, rmse, color=colors)
    axes[0].set_title(f"{title}: RMSE")
    axes[0].set_ylabel("RMSE (mm/day)")
    axes[0].tick_params(axis="x", rotation=45, labelsize=8)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(labels, corr, color=colors)
    axes[1].set_title(f"{title}: Correlation")
    axes[1].set_ylabel("Pearson correlation")
    axes[1].set_ylim(0.6, 1.0)
    axes[1].tick_params(axis="x", rotation=45, labelsize=8)
    axes[1].grid(axis="y", alpha=0.25)

    _save(fig, FIGURES / figure_name)


def plot_distribution_diagnostics(
    input_json: Path | None = None,
    figure_name: str = "chirps_distribution_diagnostics.png",
    title: str = "CHIRPS Distribution Diagnostics",
) -> None:
    payload = json.loads((input_json or (RESULTS / "downscaling_validation_chirps.json")).read_text())
    rows = [
        r
        for r in payload["scores"]
        if r["status"] == "ok" and r.get("distribution_bias") and r["reference"] in {"deepscale", "xsdba", "formula", "interpolation"}
    ]
    rows = sorted(rows, key=lambda r: (r["method"] != "bilinear", r["reference"], r["method"]))
    labels = [f"{r['method']}\n{r['reference']}" for r in rows]
    metrics = ["mean", "std", "p90", "p95"]
    colors = ["#2f6f9f", "#7a9a01", "#b45f06", "#6f4e7c"]

    fig, ax = plt.subplots(figsize=(max(10, 0.7 * len(rows)), 5.2), constrained_layout=True)
    x = np.arange(len(rows))
    width = 0.18
    for i, metric in enumerate(metrics):
        offset = (i - (len(metrics) - 1) / 2) * width
        values = [r["distribution_bias"].get(metric, np.nan) for r in rows]
        ax.bar(x + offset, values, width=width, label=f"{metric} bias", color=colors[i])

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel("Prediction - benchmark (mm/day)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncols=4, fontsize=8)
    _save(fig, FIGURES / figure_name)


def _report_rows(payload: dict) -> list[dict]:
    rows = [
        r
        for r in payload["scores"]
        if r["status"] == "ok"
        and r.get("pred_summary")
        and r.get("obs_summary")
        and r["reference"] in {"interpolation", "deepscale", "xsdba"}
    ]
    order = {"bilinear": 0, "delta": 1, "bcsd": 2, "qm": 3, "dqm": 4, "cca": 5, "climatology": 6}
    return sorted(rows, key=lambda r: (order.get(r["method"], 99), r["reference"]))


def _row_label(row: dict) -> str:
    if row["reference"] == "deepscale":
        return row["method"]
    if row["reference"] == "interpolation":
        return "bilinear"
    return f"{row['method']} ({row['reference']})"


def plot_taylor_diagram(
    input_json: Path | None = None,
    figure_name: str = "chirps_taylor_diagram.png",
    title: str = "Taylor Diagram",
) -> None:
    """Plot a compact Taylor diagram from saved correlation/std summaries."""
    payload = json.loads((input_json or (RESULTS / "downscaling_validation_chirps.json")).read_text())
    rows = _report_rows(payload)
    if not rows:
        return

    max_ratio = 1.25
    ratios = []
    for row in rows:
        obs_std = row["obs_summary"].get("std")
        pred_std = row["pred_summary"].get("std")
        if obs_std and pred_std:
            ratios.append(pred_std / obs_std)
    if ratios:
        max_ratio = max(max_ratio, min(1.8, max(ratios) * 1.12))

    fig = plt.figure(figsize=(7.2, 6.0), constrained_layout=True)
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_thetamin(0)
    ax.set_thetamax(90)
    ax.set_ylim(0, max_ratio)
    ax.set_title(title, pad=24)

    corr_ticks = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0])
    ax.set_thetagrids(np.degrees(np.arccos(corr_ticks)), labels=[f"{c:g}" for c in corr_ticks])
    ax.set_ylabel("Standard deviation ratio")
    ax.grid(alpha=0.35)
    ax.plot(0, 1, marker="*", markersize=14, color="black", linestyle="None", label="CHIRPS")

    cmap = plt.get_cmap("tab10")
    for idx, row in enumerate(rows):
        corr = row.get("corr")
        obs_std = row["obs_summary"].get("std")
        pred_std = row["pred_summary"].get("std")
        if corr is None or obs_std in (None, 0) or pred_std is None:
            continue
        theta = float(np.arccos(np.clip(corr, 0.0, 1.0)))
        ratio = float(pred_std / obs_std)
        ax.plot(theta, ratio, marker="o", markersize=7, linestyle="None", color=cmap(idx % 10), label=_row_label(row))

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.08), fontsize=8, frameon=False)
    _save(fig, FIGURES / figure_name)


def plot_quantile_profile(
    input_json: Path | None = None,
    figure_name: str = "chirps_quantile_profile.png",
    title: str = "Quantile-Quantile Summary",
) -> None:
    """Plot saved method quantiles against benchmark quantiles."""
    payload = json.loads((input_json or (RESULTS / "downscaling_validation_chirps.json")).read_text())
    rows = _report_rows(payload)
    quantile_keys = ["p50", "p70", "p90", "p95"]
    fig, ax = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)
    cmap = plt.get_cmap("tab10")

    all_values = []
    for idx, row in enumerate(rows):
        obs = [row["obs_summary"].get(k) for k in quantile_keys]
        pred = [row["pred_summary"].get(k) for k in quantile_keys]
        if any(v is None for v in obs + pred):
            continue
        all_values.extend(obs + pred)
        ax.plot(obs, pred, marker="o", linewidth=1.6, color=cmap(idx % 10), label=_row_label(row))

    if all_values:
        lo = float(np.nanmin(all_values))
        hi = float(np.nanmax(all_values))
        pad = (hi - lo) * 0.05 if hi > lo else 0.1
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", linewidth=1.0, label="1:1")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
    ax.set_title(title)
    ax.set_xlabel("CHIRPS benchmark quantile (mm/day)")
    ax.set_ylabel("Prediction quantile (mm/day)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False)
    _save(fig, FIGURES / figure_name)


def _fixture_label(path: Path) -> str:
    name = path.stem
    if "ethiopia" in name:
        return "Ethiopia FMA"
    if "texas" in name:
        return "Texas"
    return name.replace("downscaling_validation_", "").replace("_", " ").title()


def plot_delta_reference_diagnostics(
    input_jsons: list[Path] | None = None,
    figure_name: str = "delta_reference_diagnostics_1991_2020.png",
) -> None:
    """Plot Delta-only agreement across the built-in and external references."""
    input_jsons = input_jsons or [
        RESULTS / "downscaling_validation_chirps_monthly_texas_1991_2020.json",
        RESULTS / "downscaling_validation_chirps_monthly_ethiopia_fma_1991_2020.json",
    ]
    references = ["deepscale", "formula", "python-cmethods"]
    reference_labels = {
        "deepscale": "DeepScale",
        "formula": "Formula",
        "python-cmethods": "python-cmethods",
    }
    fixture_labels: list[str] = []
    rows_by_fixture: dict[str, dict[str, dict]] = {}
    for input_json in input_jsons:
        payload = json.loads(input_json.read_text())
        label = _fixture_label(input_json)
        fixture_labels.append(label)
        rows_by_fixture[label] = {
            row["reference"]: row
            for row in payload["scores"]
            if row["status"] == "ok" and row["method"] == "delta" and row["reference"] in references
        }

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), constrained_layout=True)
    panels = [
        ("RMSE", "rmse", "RMSE (mm/day)", None),
        ("Correlation", "corr", "Pearson correlation", (0.985, 0.995)),
        ("Standard Deviation Bias", ("distribution_bias", "std"), "Prediction - CHIRPS (mm/day)", None),
        ("P90 Bias", ("distribution_bias", "p90"), "Prediction - CHIRPS (mm/day)", None),
    ]
    colors = {
        "deepscale": "#2f6f9f",
        "formula": "#7f7f7f",
        "python-cmethods": "#b45f06",
    }
    x = np.arange(len(fixture_labels))
    width = 0.22
    for ax, (title, key, ylabel, ylim) in zip(axes.ravel(), panels):
        for idx, reference in enumerate(references):
            values = []
            for fixture in fixture_labels:
                row = rows_by_fixture[fixture].get(reference)
                if row is None:
                    values.append(np.nan)
                elif isinstance(key, tuple):
                    values.append(row[key[0]].get(key[1], np.nan))
                else:
                    values.append(row.get(key, np.nan))
            offset = (idx - (len(references) - 1) / 2) * width
            ax.bar(x + offset, values, width=width, color=colors[reference], label=reference_labels[reference])
        if title.endswith("Bias"):
            ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_xticks(x)
        ax.set_xticklabels(fixture_labels)
        ax.grid(axis="y", alpha=0.25)
    axes[0, 1].legend(ncols=3, fontsize=8, frameon=False, loc="upper center")
    fig.suptitle("Delta Reference Agreement", fontsize=14)
    _save(fig, FIGURES / figure_name)


def plot_quantile_reference_metric_bars(
    method: str,
    input_jsons: list[Path] | None = None,
    figure_name: str | None = None,
) -> None:
    method = method.lower()
    if method not in {"qm", "dqm"}:
        raise ValueError(f"unsupported quantile method {method!r}")
    method_label = method.upper()
    figure_name = figure_name or f"{method}_reference_metric_bars_1991_2020.png"
    input_jsons = input_jsons or [
        RESULTS / "downscaling_validation_chirps_monthly_texas_1991_2020.json",
        RESULTS / "downscaling_validation_chirps_monthly_ethiopia_fma_1991_2020.json",
    ]
    fixture_labels = [_fixture_label(path).replace(" FMA", "") for path in input_jsons]
    references = ["deepscale", "xsdba"]
    reference_labels = {"deepscale": "DeepScale", "xsdba": "xsdba"}
    colors = {"deepscale": "#2f6f9f", "xsdba": "#7f7f7f"}

    rows_by_fixture = []
    for path in input_jsons:
        payload = json.loads(path.read_text())
        rows = {
            row["reference"]: row
            for row in payload["scores"]
            if row["status"] == "ok" and row["method"] == method and row["reference"] in references
        }
        rows_by_fixture.append(rows)

    x = np.arange(len(input_jsons))
    width = 0.32
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.5), constrained_layout=True)
    for idx, reference in enumerate(references):
        offset = (idx - 0.5) * width
        axes[0].bar(
            x + offset,
            [rows[reference]["rmse"] for rows in rows_by_fixture],
            width=width,
            color=colors[reference],
            label=reference_labels[reference],
        )
        axes[1].bar(
            x + offset,
            [rows[reference]["corr"] for rows in rows_by_fixture],
            width=width,
            color=colors[reference],
            label=reference_labels[reference],
        )

    axes[0].set_title(f"{method_label} Reference: RMSE")
    axes[0].set_ylabel("RMSE (mm/day)")
    axes[1].set_title(f"{method_label} Reference: Correlation")
    axes[1].set_ylabel("Pearson correlation")
    axes[1].set_ylim(0.9, 1.0)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(fixture_labels)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8, frameon=False)
    _save(fig, FIGURES / figure_name)


def plot_qm_reference_metric_bars(
    input_jsons: list[Path] | None = None,
    figure_name: str = "qm_reference_metric_bars_1991_2020.png",
) -> None:
    plot_quantile_reference_metric_bars("qm", input_jsons, figure_name)


def plot_dqm_reference_metric_bars(
    input_jsons: list[Path] | None = None,
    figure_name: str = "dqm_reference_metric_bars_1991_2020.png",
) -> None:
    plot_quantile_reference_metric_bars("dqm", input_jsons, figure_name)


def plot_cca_maps(
    tag: str = "texas",
    figure_name: str = "pycpt_cca_maps_latest_year.png",
    label: str = "Texas",
    source: str = "deepscale",
) -> None:
    if source not in {"deepscale", "pycpt"}:
        raise ValueError(f"unsupported CCA map source {source!r}")
    paths = _pycpt_cca_paths(tag)
    deep_ds = xr.open_dataset(paths["deepscale"])
    cpt_ds = xr.open_dataset(paths["cpt"])
    obs = deep_ds["obs"]
    deep = deep_ds["deepscale_cca"]
    cpt = cpt_ds["cpt_cca"].interp(lat=obs.lat, lon=obs.lon, method="nearest")
    cpt = cpt.transpose("year", "lat", "lon")

    year = int(obs.year.values[-1])
    obs_y = obs.sel(year=year)
    deep_y = deep.sel(year=year)
    cpt_y = cpt.sel(year=year)
    pred_y = deep_y if source == "deepscale" else cpt_y
    source_label = "DeepScale CCA" if source == "deepscale" else "PyCPT CCA"
    fields = [obs_y, pred_y]
    fields_concat = xr.concat(fields, "field", coords="minimal", compat="override")
    vmin = float(fields_concat.quantile(0.02))
    vmax = float(fields_concat.quantile(0.98))
    diff = pred_y - obs_y
    diff_max = float(abs(diff).quantile(0.98))
    diff_max = max(diff_max, 1e-6)

    fig = plt.figure(figsize=(14.2, 4.6))
    gs = fig.add_gridspec(1, 6, width_ratios=[1, 1, 0.055, 0.22, 1, 0.055], wspace=0.38)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 4])]
    precip_cax = fig.add_subplot(gs[0, 2])
    diff_cax = fig.add_subplot(gs[0, 5])

    field_titles = [f"CHIRPS observed ({year})", source_label]
    for ax, da, title in zip(axes[:2], fields, field_titles):
        mesh = ax.pcolormesh(da.lon, da.lat, da, cmap="YlGnBu", vmin=vmin, vmax=vmax, shading="auto")
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    fig.colorbar(mesh, cax=precip_cax, label="Precipitation (mm/day)")

    dm = axes[2].pcolormesh(
        diff.lon,
        diff.lat,
        diff,
        cmap="RdBu_r",
        vmin=-diff_max,
        vmax=diff_max,
        shading="auto",
    )
    axes[2].set_title(f"{source_label} minus CHIRPS")
    axes[2].set_xlabel("Longitude")
    axes[2].set_ylabel("Latitude")
    fig.colorbar(dm, cax=diff_cax, label="Difference (mm/day)")
    fig.suptitle(label, fontsize=13)
    fig.subplots_adjust(top=0.82)
    _save(fig, FIGURES / figure_name)


def plot_cca_scatter() -> None:
    deep_ds = xr.open_dataset(RESULTS / "pycpt_cca_deepscale_loyo.nc")
    cpt_ds = xr.open_dataset(RESULTS / "pycpt_cca_cptcore_crossvalidation.nc")
    obs = deep_ds["obs"]
    deep = deep_ds["deepscale_cca"]
    cpt = cpt_ds["cpt_cca"].interp(lat=obs.lat, lon=obs.lon, method="nearest").transpose("year", "lat", "lon")

    pairs = [
        ("DeepScale CCA vs Obs", deep, obs),
        ("PyCPT vs Obs", cpt, obs),
        ("DeepScale CCA vs PyCPT", deep, cpt),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    for ax, (title, xda, yda) in zip(axes, pairs):
        x, y = xr.align(xda, yda, join="inner")
        valid = np.isfinite(x.values) & np.isfinite(y.values)
        xv = x.values[valid].ravel()
        yv = y.values[valid].ravel()
        if len(xv) > 2500:
            idx = np.linspace(0, len(xv) - 1, 2500).astype(int)
            xv = xv[idx]
            yv = yv[idx]
        lim = [float(min(xv.min(), yv.min())), float(max(xv.max(), yv.max()))]
        ax.scatter(xv, yv, s=5, alpha=0.22, color="#1f4e79", edgecolors="none")
        ax.plot(lim, lim, color="black", linewidth=1)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("First field (mm/day)")
        ax.set_ylabel("Second field (mm/day)")
        ax.grid(alpha=0.2)
    fig.suptitle("CCA Parity Scatter Checks", fontsize=14)
    _save(fig, FIGURES / "pycpt_cca_scatter.png")


def plot_cca_metric_bars() -> None:
    cases = [
        ("Texas", "texas"),
        ("Ethiopia", "ethiopia_fma_1991_2020"),
    ]
    payloads = [(label, json.loads(_pycpt_cca_paths(tag)["comparison"].read_text())) for label, tag in cases]
    labels = [label for label, _ in payloads]
    x = np.arange(len(labels))
    width = 0.32

    deep_rmse = [payload["deepscale_vs_obs"]["rmse"] for _, payload in payloads]
    pycpt_rmse = [payload["cpt_vs_obs"]["rmse"] for _, payload in payloads]
    deep_corr = [payload["deepscale_vs_obs"]["corr"] for _, payload in payloads]
    pycpt_corr = [payload["cpt_vs_obs"]["corr"] for _, payload in payloads]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.5), constrained_layout=True)
    axes[0].bar(x - width / 2, deep_rmse, width, color="#2f6f9f", label="DeepScale")
    axes[0].bar(x + width / 2, pycpt_rmse, width, color="#7f7f7f", label="PyCPT")
    axes[0].set_title("CCA Reference: RMSE")
    axes[0].set_ylabel("RMSE (mm/day)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x - width / 2, deep_corr, width, color="#2f6f9f", label="DeepScale")
    axes[1].bar(x + width / 2, pycpt_corr, width, color="#7f7f7f", label="PyCPT")
    axes[1].set_title("CCA Reference: Correlation")
    axes[1].set_ylabel("Pearson correlation")
    axes[1].set_ylim(0.9, 1.0)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].grid(axis="y", alpha=0.25)
    for ax in axes:
        ax.legend(fontsize=8, frameon=False)
    _save(fig, FIGURES / "pycpt_cca_metric_bars.png")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-json")
    parser.add_argument("--metrics-figure", default="chirps_method_metric_bars.png")
    parser.add_argument("--metrics-title", default="CHIRPS High/Low Benchmark")
    parser.add_argument("--distribution-figure")
    parser.add_argument("--distribution-title")
    parser.add_argument("--taylor-figure")
    parser.add_argument("--taylor-title")
    parser.add_argument("--quantile-figure")
    parser.add_argument("--quantile-title")
    parser.add_argument("--delta-reference-figure")
    parser.add_argument("--delta-reference-jsons", nargs="+")
    parser.add_argument("--qm-reference-figure")
    parser.add_argument("--qm-reference-jsons", nargs="+")
    parser.add_argument("--dqm-reference-figure")
    parser.add_argument("--dqm-reference-jsons", nargs="+")
    parser.add_argument("--skip-metrics", action="store_true")
    parser.add_argument("--metrics-only", action="store_true")
    args = parser.parse_args()

    FIGURES.mkdir(parents=True, exist_ok=True)
    if not args.skip_metrics:
        plot_metric_bars(
            Path(args.metrics_json) if args.metrics_json else None,
            args.metrics_figure,
            args.metrics_title,
        )
    if args.distribution_figure:
        plot_distribution_diagnostics(
            Path(args.metrics_json) if args.metrics_json else None,
            args.distribution_figure,
            args.distribution_title or args.metrics_title.replace("Benchmark", "Distribution Diagnostics"),
        )
    if args.taylor_figure:
        plot_taylor_diagram(
            Path(args.metrics_json) if args.metrics_json else None,
            args.taylor_figure,
            args.taylor_title or args.metrics_title.replace("Benchmark", "Taylor Diagram"),
        )
    if args.quantile_figure:
        plot_quantile_profile(
            Path(args.metrics_json) if args.metrics_json else None,
            args.quantile_figure,
            args.quantile_title or args.metrics_title.replace("Benchmark", "Quantile-Quantile Summary"),
        )
    if args.delta_reference_figure:
        plot_delta_reference_diagnostics(
            [Path(p) for p in args.delta_reference_jsons] if args.delta_reference_jsons else None,
            args.delta_reference_figure,
        )
    if args.qm_reference_figure:
        plot_qm_reference_metric_bars(
            [Path(p) for p in args.qm_reference_jsons] if args.qm_reference_jsons else None,
            args.qm_reference_figure,
        )
    if args.dqm_reference_figure:
        plot_dqm_reference_metric_bars(
            [Path(p) for p in args.dqm_reference_jsons] if args.dqm_reference_jsons else None,
            args.dqm_reference_figure,
        )
    if args.metrics_only:
        return 0
    plot_cca_maps()
    plot_cca_maps(source="pycpt", figure_name="pycpt_cca_pycpt_maps_latest_year.png")
    plot_cca_maps(
        tag="ethiopia_fma_1991_2020",
        figure_name="pycpt_cca_maps_ethiopia_fma_1991_2020.png",
        label="Ethiopia",
    )
    plot_cca_maps(
        tag="ethiopia_fma_1991_2020",
        figure_name="pycpt_cca_pycpt_maps_ethiopia_fma_1991_2020.png",
        label="Ethiopia",
        source="pycpt",
    )
    plot_cca_scatter()
    plot_cca_metric_bars()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
