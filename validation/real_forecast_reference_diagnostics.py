"""Method-specific reference diagnostics for real seasonal hindcast runs.

This validation artifact complements ``real_forecast_validation.py``. It reads
the saved DeepScale/xsdba prediction files from the real ECMWF benchmark, fetches
the matching coarse real hindcast through the existing Rosetta path, and adds
functional reference diagnostics for BCSD and analog methods.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import xarray as xr

from bcsd_reference_validation import _run_scikit_bcsd_series_context
from rank_analog_reference_validation import _run_scikit_pure_analog_baselines
from real_forecast_validation import PRESETS, _align_years, fetch_chirps_season, fetch_real_hindcast, score_prediction


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


def _suffix_from_preset(preset_name: str) -> str:
    return preset_name


def _prediction_path(suffix: str) -> Path:
    return RESULTS / f"real_forecast_predictions_{suffix}.nc"


def _transpose_year_lat_lon(da: xr.DataArray) -> xr.DataArray:
    return da.transpose("year", "lat", "lon")


def _plot_metric_bars(scores: list[dict], suffix: str) -> None:
    rows = [r for r in scores if r["status"] == "ok"]
    rows = sorted(rows, key=lambda r: (r["rmse"], -r["corr"]))
    labels = [f"{r['method']}\n{r['reference']}" for r in rows]
    colors = ["#2f6f9f" if r["reference"] == "deepscale" else "#6aa84f" for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), constrained_layout=True)
    axes[0].bar(labels, [r["rmse"] for r in rows], color=colors)
    axes[0].set_title("Real ECMWF References: RMSE vs Obs")
    axes[0].set_ylabel("RMSE (mm/day)")
    axes[0].tick_params(axis="x", rotation=45, labelsize=8)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(labels, [r["corr"] for r in rows], color=colors)
    axes[1].set_title("Real ECMWF References: Correlation vs Obs")
    axes[1].set_ylabel("Pearson correlation")
    axes[1].set_ylim(max(0.0, min(r["corr"] for r in rows) - 0.05), 1.0)
    axes[1].tick_params(axis="x", rotation=45, labelsize=8)
    axes[1].grid(axis="y", alpha=0.25)

    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"real_forecast_reference_metric_bars_{suffix}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {path}")


def _plot_pair_bars(pair_scores: list[PairScore], suffix: str) -> None:
    labels = [p.comparison.replace("_", "\n") for p in pair_scores]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), constrained_layout=True)
    axes[0].bar(labels, [p.rmse for p in pair_scores], color="#b45f06")
    axes[0].set_title("Reference Pair Differences: RMSE")
    axes[0].set_ylabel("RMSE (mm/day)")
    axes[0].tick_params(axis="x", rotation=35, labelsize=8)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(labels, [p.corr for p in pair_scores], color="#7f7f7f")
    axes[1].set_title("Reference Pair Differences: Correlation")
    axes[1].set_ylabel("Pearson correlation")
    axes[1].set_ylim(max(0.0, min(p.corr for p in pair_scores) - 0.05), 1.0)
    axes[1].tick_params(axis="x", rotation=35, labelsize=8)
    axes[1].grid(axis="y", alpha=0.25)

    path = FIGURES / f"real_forecast_reference_pair_bars_{suffix}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {path}")


def run_preset(preset_name: str, *, cache: bool, n_analogs: int) -> dict:
    preset = PRESETS[preset_name]
    suffix = _suffix_from_preset(preset_name)
    predictions_path = _prediction_path(suffix)
    if not predictions_path.exists():
        raise FileNotFoundError(f"missing saved real-forecast predictions: {predictions_path}")

    saved = xr.open_dataset(predictions_path)
    forecast_product = preset["forecast_product"]
    obs_product = preset["obs_product"]
    start_year = preset["start_year"]
    end_year = preset["end_year"]
    bbox = preset["bbox"]
    init = preset["init"]
    target = preset["target"]

    hindcast = fetch_real_hindcast(forecast_product, (start_year, end_year), bbox, init, target, cache=cache)
    obs = fetch_chirps_season(obs_product, (start_year, end_year), bbox, target, cache=cache)
    hindcast, obs = _align_years(hindcast, obs)

    deep_bcsd = _transpose_year_lat_lon(saved["deepscale_bcsd"])
    deep_rank = _transpose_year_lat_lon(saved["deepscale_rank_analog"])
    deep_qm = _transpose_year_lat_lon(saved["deepscale_qm"])
    deep_dqm = _transpose_year_lat_lon(saved["deepscale_dqm"])
    xsdba_qm = _transpose_year_lat_lon(saved["xsdba_qm"])
    xsdba_dqm = _transpose_year_lat_lon(saved["xsdba_dqm"])

    scikit_bcsd = _run_scikit_bcsd_series_context(hindcast, obs)
    analogs = _run_scikit_pure_analog_baselines(hindcast, obs, n_analogs=n_analogs)

    scores = [
        asdict(score_prediction("bcsd", "deepscale", deep_bcsd, obs)),
        asdict(score_prediction("bcsd", "scikit-downscale-series-context", scikit_bcsd, obs)),
        asdict(score_prediction("rank-analog", "deepscale", deep_rank, obs)),
        asdict(score_prediction("qm", "deepscale", deep_qm, obs)),
        asdict(score_prediction("qm", "xsdba", xsdba_qm, obs)),
        asdict(score_prediction("dqm", "deepscale", deep_dqm, obs)),
        asdict(score_prediction("dqm", "xsdba", xsdba_dqm, obs)),
    ]
    for name, da in analogs.items():
        scores.append(asdict(score_prediction("analog", name.replace("scikit_", "scikit-downscale-"), da, obs)))

    pair_scores = [
        _score_pair("deepscale_bcsd_vs_scikit_bcsd_series_context", deep_bcsd, scikit_bcsd),
        _score_pair("deepscale_rank_analog_vs_scikit_pure_analog_weighted", deep_rank, analogs["scikit_pure_analog_weighted"]),
        _score_pair("deepscale_qm_vs_xsdba_qm", deep_qm, xsdba_qm),
        _score_pair("deepscale_dqm_vs_xsdba_dqm", deep_dqm, xsdba_dqm),
    ]

    out_nc = RESULTS / f"real_forecast_reference_predictions_{suffix}.nc"
    out_json = RESULTS / f"real_forecast_reference_diagnostics_{suffix}.json"
    payload = {
        "dataset": {
            "preset": preset_name,
            "forecast_product": forecast_product,
            "obs_product": obs_product,
            "init": init,
            "target": target,
            "requested_years": [start_year, end_year],
            "actual_years": [int(y) for y in obs.year.values],
            "bbox": bbox,
            "source_prediction_file": str(predictions_path),
            "prediction_file": str(out_nc),
            "n_analogs": n_analogs,
        },
        "scores": scores,
        "pair_scores": [asdict(p) for p in pair_scores],
        "notes": [
            "BCSD scikit reference uses the same pointwise series-context adapter as the controlled CHIRPS diagnostics",
            "PureAnalog baselines use the full coarse forecast field as a feature vector to select historical CHIRPS years",
            "QM/DQM xsdba outputs are read from the saved real_forecast_validation prediction file",
            "These diagnostics compare functional reference behavior on real seasonal hindcasts; they are not new production methods",
        ],
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    xr.Dataset(
        {
            "obs": obs,
            "deepscale_bcsd": deep_bcsd,
            "scikit_bcsd_series_context": scikit_bcsd,
            "deepscale_rank_analog": deep_rank,
            **analogs,
            "deepscale_qm": deep_qm,
            "xsdba_qm": xsdba_qm,
            "deepscale_dqm": deep_dqm,
            "xsdba_dqm": xsdba_dqm,
        }
    ).to_netcdf(out_nc)
    out_json.write_text(json.dumps(payload, indent=2))
    _plot_metric_bars(scores, suffix)
    _plot_pair_bars(pair_scores, suffix)
    print(f"Wrote {out_json}")
    print(f"Wrote {out_nc}")
    print(json.dumps(payload, indent=2))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        action="append",
        help="Preset to run. May be supplied more than once. Defaults to full ECMWF MAM and OND.",
    )
    parser.add_argument("--n-analogs", type=int, default=5)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    presets = args.preset or [
        "ecmwf_east_africa_mam_2001_2016",
        "ecmwf_east_africa_ond_2001_2016",
    ]
    for preset_name in presets:
        run_preset(preset_name, cache=not args.no_cache, n_analogs=args.n_analogs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
