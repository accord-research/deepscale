"""Rank-analog validation against ECMWF-S2S4AFRICA.

This is a validation artifact, not production code. It compares DeepScale's
rank-analog method against the upstream `rank_upscale_and_align` function from
`alecjong-lab/ECMWF-S2S4AFRICA`.

The upstream file imports operational ECMWF/GRIB/plotting dependencies at module
import time. To avoid making those unrelated tools validation dependencies, this
script loads only the `rank_upscale_and_align` function from a local copy of
`get_ECMWF_functions.py` and executes it with the scientific dependencies it
actually uses.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter, grey_closing

from downscaling_validation import PRESETS, fetch_chirps, make_hindcast, score_prediction

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "validation" / "results"
FIGURES = ROOT / "validation" / "figures"

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "validation" / ".matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

UPSTREAM_URL = "https://github.com/alecjong-lab/ECMWF-S2S4AFRICA"
UPSTREAM_FUNCTION = "get_ECMWF_functions.rank_upscale_and_align"


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


def _load_s2s4africa_function(source_path: Path):
    """Load rank_upscale_and_align from upstream source without module imports."""
    tree = ast.parse(source_path.read_text())
    func = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "rank_upscale_and_align"),
        None,
    )
    if func is None:
        raise ValueError(f"rank_upscale_and_align not found in {source_path}")
    module = ast.Module(body=[func], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "np": np,
        "xr": xr,
        "grey_closing": grey_closing,
        "gaussian_filter": gaussian_filter,
    }
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["rank_upscale_and_align"]


def _run_deepscale_rank_analog(hindcast: xr.DataArray, obs: xr.DataArray) -> xr.DataArray:
    from deepscale.registry import get_method

    cls = get_method("rank-analog")
    preds = []
    years = list(obs.year.values)
    for year in years:
        train_years = [y for y in years if y != year]
        model = cls()
        model.fit(hindcast.sel(year=train_years), obs.sel(year=train_years))
        fcst = hindcast.sel(year=[year]).isel(year=0, drop=True)
        pred = model.predict(fcst).mean("member")
        preds.append(pred.expand_dims(year=[year]))
    return xr.concat(preds, "year").rename("deepscale_rank_analog")


def _as_s2s_source(train: xr.DataArray, forecast: xr.DataArray) -> xr.DataArray:
    source = xr.concat(
        [
            train,
            forecast.expand_dims(year=[int(train.year.max()) + 1]),
        ],
        dim="year",
    )
    return (
        source.rename({"lat": "latitude", "lon": "longitude"})
        .expand_dims(step=[0])
        .transpose("step", "year", "latitude", "longitude")
    )


def _as_s2s_target(obs_train: xr.DataArray) -> xr.DataArray:
    sorted_obs = np.sort(obs_train.values, axis=obs_train.dims.index("year"))
    return xr.DataArray(
        sorted_obs,
        dims=("rank", "latitude", "longitude"),
        coords={
            "rank": np.arange(obs_train.sizes["year"]),
            "latitude": obs_train.lat.values,
            "longitude": obs_train.lon.values,
        },
    )


def _normalize_s2s_output(out: xr.DataArray, year, obs: xr.DataArray) -> xr.DataArray:
    rename = {}
    if "latitude" in out.dims:
        rename["latitude"] = "lat"
    if "longitude" in out.dims:
        rename["longitude"] = "lon"
    out = out.rename(rename)
    if "step" in out.dims:
        out = out.isel(step=0, drop=True)
    if "year" in out.dims:
        out = out.isel(year=0, drop=True)
    out = out.transpose("lat", "lon")
    return out.assign_coords(lat=obs.lat, lon=obs.lon).expand_dims(year=[year])


def _run_s2s4africa_rank_analog(hindcast: xr.DataArray, obs: xr.DataArray, source_path: Path) -> xr.DataArray:
    rank_upscale_and_align = _load_s2s4africa_function(source_path)
    gcm = hindcast.mean("member")
    years = list(obs.year.values)
    preds = []
    upscale_factor = max(
        int(np.ceil(obs.sizes["lat"] / gcm.sizes["lat"])),
        int(np.ceil(obs.sizes["lon"] / gcm.sizes["lon"])),
        1,
    )
    for year in years:
        train_years = [y for y in years if y != year]
        source = _as_s2s_source(gcm.sel(year=train_years), gcm.sel(year=year))
        target = _as_s2s_target(obs.sel(year=train_years))
        pred = rank_upscale_and_align(
            source,
            target,
            rank_dim="year",
            upscale_factor=upscale_factor,
            rank_threshold=1,
            lat_name="latitude",
            lon_name="longitude",
        )
        preds.append(_normalize_s2s_output(pred, year, obs))
    return xr.concat(preds, "year").rename("s2s4africa_rank_upscale_and_align")


def _plot_metrics(scores: list[dict], pair_scores: list[PairScore], suffix: str) -> None:
    labels = ["DeepScale", "S2S4Africa"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.5), constrained_layout=True)
    axes[0].bar(labels, [s["rmse"] for s in scores], color=["#2f6f9f", "#6aa84f"])
    axes[0].set_title("Rank-Analog Reference: RMSE vs Obs")
    axes[0].set_ylabel("RMSE (mm/day)")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(labels, [s["corr"] for s in scores], color=["#2f6f9f", "#6aa84f"])
    axes[1].set_title("Rank-Analog Reference: Correlation vs Obs")
    axes[1].set_ylabel("Pearson correlation")
    axes[1].set_ylim(0.6, 1.0)
    axes[1].grid(axis="y", alpha=0.25)
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"rank_analog_reference_metric_bars_{suffix}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {path}")

    fig, ax = plt.subplots(figsize=(5.5, 4.2), constrained_layout=True)
    ax.bar(["DeepScale\nvs S2S4Africa"], [pair_scores[0].rmse], color="#b45f06")
    ax.set_title("Rank-Analog Prediction Difference")
    ax.set_ylabel("RMSE (mm/day)")
    ax.grid(axis="y", alpha=0.25)
    path = FIGURES / f"rank_analog_reference_difference_{suffix}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {path}")


def _plot_maps(obs: xr.DataArray, deep: xr.DataArray, s2s: xr.DataArray, suffix: str) -> None:
    year = int(obs.year.values[-1])
    obs_y = obs.sel(year=year)
    deep_y = deep.sel(year=year)
    s2s_y = s2s.sel(year=year)
    fields = [obs_y, deep_y, s2s_y]
    vmin = float(xr.concat(fields, "field").quantile(0.02))
    vmax = float(xr.concat(fields, "field").quantile(0.98))
    diff_max = float(max(abs(deep_y - obs_y).max(skipna=True), abs(s2s_y - obs_y).max(skipna=True), abs(deep_y - s2s_y).max(skipna=True)))
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), constrained_layout=True)
    for ax, da, title in zip(axes[0], fields, [f"Observed ({year})", "DeepScale rank-analog", "S2S4Africa direct"]):
        mesh = ax.pcolormesh(da.lon, da.lat, da, cmap="YlGnBu", vmin=vmin, vmax=vmax, shading="auto")
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    fig.colorbar(mesh, ax=axes[0, :], shrink=0.8, label="Precipitation (mm/day)")
    for ax, da, title in zip(axes[1], [deep_y - obs_y, s2s_y - obs_y, deep_y - s2s_y], ["DeepScale - Obs", "S2S4Africa - Obs", "DeepScale - S2S4Africa"]):
        dm = ax.pcolormesh(da.lon, da.lat, da, cmap="RdBu_r", vmin=-diff_max, vmax=diff_max, shading="auto")
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    fig.colorbar(dm, ax=axes[1, :], shrink=0.8, label="Difference (mm/day)")
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"rank_analog_reference_maps_{suffix}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=sorted(PRESETS), default="texas")
    parser.add_argument("--product")
    parser.add_argument("--start-year", type=int)
    parser.add_argument("--end-year", type=int)
    parser.add_argument("--bbox", nargs=4, type=float)
    parser.add_argument("--coarsen-factor", type=int, default=3)
    parser.add_argument("--members", type=int, default=3)
    parser.add_argument("--s2s4africa-source", type=Path, default=os.environ.get("S2S4AFRICA_SOURCE"))
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
    out_json = Path(args.out_json) if args.out_json else RESULTS / f"rank_analog_reference_validation_{suffix}.json"
    out_nc = Path(args.out_nc) if args.out_nc else RESULTS / f"rank_analog_reference_predictions_{suffix}.nc"

    obs = fetch_chirps(product, (start_year, end_year), bbox, cache=not args.no_cache, months=months)
    hindcast = make_hindcast(obs, args.coarsen_factor, args.members)
    deep = _run_deepscale_rank_analog(hindcast, obs)

    if not args.s2s4africa_source or not Path(args.s2s4africa_source).exists():
        payload = {
            "preset": args.preset,
            "product": product,
            "requested_years": [start_year, end_year],
            "actual_years": [int(y) for y in obs.year.values],
            "bbox": bbox,
            "months": months,
            "upstream": {"repository": UPSTREAM_URL, "function": UPSTREAM_FUNCTION},
            "scores": [asdict(score_prediction("rank-analog", "deepscale", deep, obs))],
            "status": "s2s4africa_source_missing",
            "notes": [
                "PureAnalog baselines were removed because they are not rank-upscale-index implementations.",
                "Provide --s2s4africa-source /path/to/ECMWF-S2S4AFRICA/get_ECMWF_functions.py to run the direct upstream function comparison.",
            ],
        }
        RESULTS.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, indent=2))
        print(json.dumps(payload, indent=2))
        return 0

    s2s = _run_s2s4africa_rank_analog(hindcast, obs, Path(args.s2s4africa_source))
    predictions = {"obs": obs, "deepscale_rank_analog": deep, "s2s4africa_rank_upscale_and_align": s2s}
    scores = [
        asdict(score_prediction("rank-analog", "deepscale", deep, obs)),
        asdict(score_prediction("rank-analog", "s2s4africa-rank_upscale_and_align", s2s, obs)),
    ]
    pair_scores = [_score_pair("deepscale_vs_s2s4africa_rank_upscale_and_align", deep, s2s)]
    payload = {
        "preset": args.preset,
        "product": product,
        "requested_years": [start_year, end_year],
        "actual_years": [int(y) for y in obs.year.values],
        "bbox": bbox,
        "months": months,
        "prediction_file": str(out_nc),
        "upstream": {
            "repository": UPSTREAM_URL,
            "function": UPSTREAM_FUNCTION,
            "source_path": str(args.s2s4africa_source),
        },
        "scores": scores,
        "pair_scores": [asdict(p) for p in pair_scores],
        "notes": [
            "PureAnalog baselines were removed because they are not rank-upscale-index implementations.",
            "The S2S4Africa reference is loaded directly from upstream get_ECMWF_functions.py by extracting only rank_upscale_and_align.",
            "Direct full-module import is intentionally avoided because unrelated ECMWF/GRIB/plotting imports are not needed for this function.",
        ],
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    xr.Dataset(predictions).to_netcdf(out_nc)
    out_json.write_text(json.dumps(payload, indent=2))
    _plot_metrics(scores, pair_scores, suffix)
    _plot_maps(obs, deep, s2s, suffix)
    print(f"Wrote {out_json}")
    print(f"Wrote {out_nc}")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
