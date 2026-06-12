"""Build method-specific real-hindcast figures for the validation report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


RESULTS = Path("validation/results")
FIGURES = Path("validation/figures")

CASES = [
    ("ECMWF MAM", "ecmwf_east_africa_mam_2001_2016"),
    ("ECMWF OND", "ecmwf_east_africa_ond_2001_2016"),
    ("DWD MAM", "dwd_east_africa_mam_2001_2016"),
    ("DWD OND", "dwd_east_africa_ond_2001_2016"),
]

METHODS = {
    "delta": {
        "label": "Delta",
        "var": "deepscale_delta",
        "aliases": [("delta", "deepscale"), ("delta", "formula")],
    },
    "bcsd": {
        "label": "BCSD",
        "var": "deepscale_bcsd",
        "aliases": [("bcsd", "deepscale")],
    },
    "rank_analog": {
        "label": "Rank-analog",
        "var": "deepscale_rank_analog",
        "aliases": [("rank-analog", "deepscale")],
    },
    "cca": {
        "label": "CCA",
        "var": "deepscale_cca",
        "aliases": [("cca", "deepscale")],
    },
    "qm": {
        "label": "QM",
        "var": "deepscale_qm",
        "aliases": [("qm", "deepscale"), ("qm", "xsdba")],
    },
    "dqm": {
        "label": "DQM",
        "var": "deepscale_dqm",
        "aliases": [("dqm", "deepscale"), ("dqm", "xsdba")],
    },
    "climatology": {
        "label": "Climatology",
        "var": "deepscale_climatology",
        "aliases": [("climatology", "deepscale")],
    },
}


def _json_for_case(case_id: str) -> dict:
    return json.loads((RESULTS / f"real_forecast_validation_{case_id}.json").read_text())


def _prediction_for_case(case_id: str) -> xr.Dataset:
    return xr.open_dataset(RESULTS / f"real_forecast_predictions_{case_id}.nc")


def _score_lookup(case_id: str) -> dict[tuple[str, str], dict]:
    rows = _json_for_case(case_id)["scores"]
    return {(row["method"], row["reference"]): row for row in rows if row["status"] == "ok"}


def _best_deepscale(case_id: str, metric: str) -> float:
    rows = [
        row
        for row in _json_for_case(case_id)["scores"]
        if row["status"] == "ok" and row["reference"] == "deepscale"
    ]
    if metric == "rmse":
        return min(row[metric] for row in rows)
    return max(row[metric] for row in rows)


def _case_title(case_label: str, ds: xr.Dataset) -> str:
    years = ds["year"].values
    latest = int(years[-1])
    return f"{case_label}\n{latest}"


def build_maps(method_key: str, spec: dict) -> None:
    fig, axes = plt.subplots(3, len(CASES), figsize=(14, 8.5), constrained_layout=True)
    cmap_field = "YlGnBu"
    cmap_error = "RdBu_r"

    for col, (case_label, case_id) in enumerate(CASES):
        ds = _prediction_for_case(case_id)
        obs = ds["obs"].isel(year=-1)
        pred = ds[spec["var"]].isel(year=-1)
        err = pred - obs

        field_values = np.concatenate(
            [np.ravel(obs.values[np.isfinite(obs.values)]), np.ravel(pred.values[np.isfinite(pred.values)])]
        )
        vmin = float(np.nanpercentile(field_values, 2))
        vmax = float(np.nanpercentile(field_values, 98))
        err_abs = float(np.nanpercentile(np.abs(err.values[np.isfinite(err.values)]), 98))
        err_abs = max(err_abs, 1e-6)

        extent = [
            float(ds["lon"].min()),
            float(ds["lon"].max()),
            float(ds["lat"].min()),
            float(ds["lat"].max()),
        ]

        panels = [
            (obs, "Observed CHIRPS", cmap_field, vmin, vmax),
            (pred, spec["label"], cmap_field, vmin, vmax),
            (err, "Prediction - observed", cmap_error, -err_abs, err_abs),
        ]
        for row, (arr, row_title, cmap, lo, hi) in enumerate(panels):
            ax = axes[row, col]
            im = ax.imshow(
                arr.values,
                origin="lower",
                extent=extent,
                cmap=cmap,
                vmin=lo,
                vmax=hi,
                aspect="auto",
            )
            if row == 0:
                ax.set_title(_case_title(case_label, ds), fontsize=10)
            if col == 0:
                ax.set_ylabel(row_title, fontsize=10)
            ax.set_xlabel("lon", fontsize=8)
            ax.set_ylabel(ax.get_ylabel() or "lat", fontsize=8)
            ax.tick_params(labelsize=8)
            fig.colorbar(im, ax=ax, shrink=0.75)

    fig.suptitle(f"{spec['label']}: latest-year real hindcast maps", fontsize=14)
    out = FIGURES / f"method_{method_key}_real_maps.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)


def build_metric_bars(method_key: str, spec: dict) -> None:
    labels = [label for label, _ in CASES]
    x = np.arange(len(CASES))

    primary_rmse = []
    primary_corr = []
    clim_rmse = []
    clim_corr = []
    best_rmse = []
    best_corr = []
    alt_rmse = []
    alt_corr = []
    alt_label = None

    for _, case_id in CASES:
        scores = _score_lookup(case_id)
        primary = scores[spec["aliases"][0]]
        clim = scores[("climatology", "deepscale")]
        primary_rmse.append(primary["rmse"])
        primary_corr.append(primary["corr"])
        clim_rmse.append(clim["rmse"])
        clim_corr.append(clim["corr"])
        best_rmse.append(_best_deepscale(case_id, "rmse"))
        best_corr.append(_best_deepscale(case_id, "corr"))

        if len(spec["aliases"]) > 1:
            alt = scores[spec["aliases"][1]]
            alt_rmse.append(alt["rmse"])
            alt_corr.append(alt["corr"])
            alt_label = f"{spec['aliases'][1][1]} {spec['label']}"

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    width = 0.22 if alt_label else 0.26

    axes[0].bar(x - width, primary_rmse, width, label=spec["label"])
    axes[0].bar(x, clim_rmse, width, label="Climatology")
    axes[0].bar(x + width, best_rmse, width, label="Best DeepScale")
    if alt_label:
        axes[0].bar(x + 2 * width, alt_rmse, width, label=alt_label)
    axes[0].set_ylabel("RMSE lower is better")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].set_title("Real hindcast RMSE")

    axes[1].bar(x - width, primary_corr, width, label=spec["label"])
    axes[1].bar(x, clim_corr, width, label="Climatology")
    axes[1].bar(x + width, best_corr, width, label="Best DeepScale")
    if alt_label:
        axes[1].bar(x + 2 * width, alt_corr, width, label=alt_label)
    axes[1].set_ylabel("Correlation higher is better")
    axes[1].set_ylim(0, 1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].set_title("Real hindcast correlation")

    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle(f"{spec['label']}: real hindcast numeric comparison", fontsize=14)
    out = FIGURES / f"method_{method_key}_real_metrics.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for method_key, spec in METHODS.items():
        build_maps(method_key, spec)
        build_metric_bars(method_key, spec)


if __name__ == "__main__":
    main()
