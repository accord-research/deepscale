"""Probabilistic tercile verification for saved real-forecast validations.

Reads deterministic crossvalidated prediction files from
``real_forecast_validation.py``, converts each method to tercile probabilities
with leakage-disciplined ``to_tercile_cv`` methods, and scores RPSS, ROC,
reliability, generalized ROC, HSS, and tercile accuracy.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import xarray as xr

from deepscale.metrics.rpss import _cpt_boundaries
from deepscale.skill import skill
from deepscale.tercile import to_tercile_cv


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "validation" / "results"
FIGURES = ROOT / "validation" / "figures"

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "validation" / ".matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_INPUTS = {
    "ecmwf_east_africa_mam_2001_2016": RESULTS / "real_forecast_predictions_ecmwf_east_africa_mam_2001_2016.nc",
    "ecmwf_east_africa_ond_2001_2016": RESULTS / "real_forecast_predictions_ecmwf_east_africa_ond_2001_2016.nc",
    "dwd_east_africa_mam_2001_2016": RESULTS / "real_forecast_predictions_dwd_east_africa_mam_2001_2016.nc",
    "dwd_east_africa_ond_2001_2016": RESULTS / "real_forecast_predictions_dwd_east_africa_ond_2001_2016.nc",
}

TERCILE_METHODS = ("bootstrap", "gaussian_loo")
SKILL_METRICS = ["rpss", "roc", "reliability", "generalized_roc", "heidke_skill_score"]


def _clean_name(name: str) -> str:
    return name.replace("deepscale_", "").replace("formula_", "formula-").replace("xsdba_", "xsdba-")


def _finite_float(value) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _obs_categories(obs: xr.DataArray) -> xr.DataArray:
    obs = obs.transpose("year", ...)
    t33, t67 = _cpt_boundaries(obs.values)
    cats = np.where(t33 > obs.values, 0, np.where(t67 > obs.values, 1, 2)).astype(float)
    cats[np.isnan(obs.values) | np.isnan(t33)[None, ...]] = np.nan
    return xr.DataArray(cats, dims=obs.dims, coords=obs.coords, name="obs_tercile")


def _tercile_accuracy(probs: xr.DataArray, obs: xr.DataArray) -> float | None:
    cats = _obs_categories(obs)
    spatial_dims = [d for d in obs.dims if d != "year"]
    pred_cat = probs.transpose("year", "tercile", *spatial_dims).argmax("tercile")
    pred_cat, cats = xr.align(pred_cat, cats, join="inner")
    valid = np.isfinite(cats.values) & np.isfinite(pred_cat.values)
    if int(valid.sum()) == 0:
        return None
    return float((pred_cat.values[valid] == cats.values[valid]).mean())


def _score_probs(dataset: str, source_name: str, tercile_method: str, probs: xr.DataArray, obs: xr.DataArray) -> dict:
    report = skill(probs, obs, metrics=SKILL_METRICS, n_bins=5)
    row = {
        "dataset": dataset,
        "source": source_name,
        "tercile_method": tercile_method,
        "tercile_accuracy": _tercile_accuracy(probs, obs),
    }
    for key, value in report.scores.items():
        row[key] = _finite_float(value)
    return row


def _finite_point_inputs(pred: xr.DataArray, obs: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    pred, obs = xr.align(pred.transpose("year", "lat", "lon"), obs.transpose("year", "lat", "lon"), join="inner")
    pred_pt = pred.stack(point=("lat", "lon"))
    obs_pt = obs.stack(point=("lat", "lon"))
    valid = np.isfinite(pred_pt).all("year") & np.isfinite(obs_pt).all("year")
    pred_valid = pred_pt.sel(point=valid).transpose("year", "point")
    obs_valid = obs_pt.sel(point=valid).transpose("year", "point")
    if pred_valid.sizes["point"] == 0:
        raise ValueError("no spatial points are finite across all years")
    return pred_valid, obs_valid


def _run_one(dataset: str, path: Path) -> tuple[list[dict], dict[str, xr.DataArray]]:
    ds = xr.open_dataset(path)
    obs = ds["obs"].transpose("year", "lat", "lon")
    rows = []
    probs_out = {}
    for var in sorted(ds.data_vars):
        if var == "obs":
            continue
        pred = ds[var].transpose("year", "lat", "lon")
        source_name = _clean_name(var)
        for tercile_method in TERCILE_METHODS:
            try:
                pred_valid, obs_valid = _finite_point_inputs(pred, obs)
                probs = to_tercile_cv(pred_valid, obs_valid, method=tercile_method).transpose("year", "tercile", "point")
                point_coord = np.arange(probs.sizes["point"])
                probs = probs.reset_index("point", drop=True).assign_coords(point=point_coord)
                obs_score = obs_valid.reset_index("point", drop=True).assign_coords(point=point_coord)
                probs_name = f"{dataset}__{source_name}__{tercile_method}".replace("-", "_")
                probs_out[probs_name] = probs.rename(probs_name)
                row = _score_probs(dataset, source_name, tercile_method, probs, obs_score)
                row["n_points"] = int(obs_score.sizes["point"])
                rows.append(row)
                print(f"ok {dataset} {source_name} {tercile_method}")
            except Exception as exc:
                rows.append(
                    {
                        "dataset": dataset,
                        "source": source_name,
                        "tercile_method": tercile_method,
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(f"failed {dataset} {source_name} {tercile_method}: {type(exc).__name__}: {exc}")
    return rows, probs_out


def _plot_metric(rows: list[dict], metric: str, suffix: str, *, higher_is_better: bool = True) -> None:
    ok = [r for r in rows if r.get(metric) is not None]
    if not ok:
        return
    ok = sorted(ok, key=lambda r: r[metric], reverse=higher_is_better)[:24]
    labels = [f"{r['dataset'].replace('_east_africa_', '\\n')}\n{r['source']}\n{r['tercile_method']}" for r in ok]
    colors = ["#2f6f9f" if r["tercile_method"] == "bootstrap" else "#6aa84f" for r in ok]
    fig, ax = plt.subplots(figsize=(15, 6.5), constrained_layout=True)
    ax.bar(labels, [r[metric] for r in ok], color=colors)
    ax.set_title(f"Probabilistic Verification: {metric}")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=55, labelsize=7)
    ax.grid(axis="y", alpha=0.25)
    path = FIGURES / f"probabilistic_{metric}_{suffix}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DEFAULT_INPUTS), action="append")
    parser.add_argument("--out-json", default=str(RESULTS / "probabilistic_real_forecast_verification.json"))
    parser.add_argument("--out-nc", default=str(RESULTS / "probabilistic_real_forecast_terciles.nc"))
    args = parser.parse_args()

    datasets = args.dataset or list(DEFAULT_INPUTS)
    rows = []
    probs = {}
    for dataset in datasets:
        path = DEFAULT_INPUTS[dataset]
        if not path.exists():
            raise FileNotFoundError(path)
        ds_rows, ds_probs = _run_one(dataset, path)
        rows.extend(ds_rows)
        probs.update(ds_probs)

    payload = {
        "inputs": {name: str(DEFAULT_INPUTS[name]) for name in datasets},
        "tercile_methods": list(TERCILE_METHODS),
        "metrics": SKILL_METRICS + ["tercile_accuracy"],
        "scores": rows,
        "notes": [
            "Deterministic crossvalidated predictions are converted with to_tercile_cv, avoiding held-out-year leakage.",
            "bootstrap uses leave-one-year-out residual bootstrap probabilities.",
            "gaussian_loo uses leave-one-year-out prediction error variance and leave-one-out tercile boundaries.",
            "Scores are pooled over all valid years and grid cells.",
        ],
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(payload, indent=2))
    xr.Dataset(probs).to_netcdf(args.out_nc)
    _plot_metric(rows, "rpss", "rpss", higher_is_better=True)
    _plot_metric(rows, "generalized_roc", "groc", higher_is_better=True)
    _plot_metric(rows, "tercile_accuracy", "accuracy", higher_is_better=True)
    _plot_metric(rows, "reliability", "reliability", higher_is_better=False)
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_nc}")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
