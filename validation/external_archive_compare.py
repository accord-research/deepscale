"""Compare a DeepScale product against an external downscaled forecast archive.

This is a validation utility for local NetCDF artifacts. It is intentionally
provider-agnostic: pass one DeepScale file, one external archive file, and
optionally a benchmark file. The script aligns common years/grid cells and
writes JSON diagnostics with the same summary statistics used by the controlled
CHIRPS high/low harness.

Example:

    uv run python validation/external_archive_compare.py \
      --deepscale validation/results/my_deepscale_predictions.nc \
      --external /path/to/downscaled_nmme_archive.nc \
      --benchmark /path/to/chirps_benchmark.nc \
      --deepscale-var deepscale_bcsd \
      --external-var precip \
      --benchmark-var precip \
      --out validation/results/external_archive_compare.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import xarray as xr

from downscaling_validation import _summary_bias, _summary_stats


@dataclass
class PairScore:
    comparison: str
    status: str
    n_cells: int
    bias: float | None = None
    mae: float | None = None
    rmse: float | None = None
    corr: float | None = None
    first_summary: dict | None = None
    second_summary: dict | None = None
    distribution_bias: dict | None = None
    error: str | None = None


def _pick(ds: xr.Dataset, name: str | None) -> xr.DataArray:
    if name:
        return ds[name]
    data_vars = list(ds.data_vars)
    if len(data_vars) != 1:
        raise ValueError(f"file has multiple data variables; pass an explicit variable name: {data_vars}")
    return ds[data_vars[0]]


def _standardize_dims(da: xr.DataArray) -> xr.DataArray:
    rename = {}
    for src, dst in {
        "latitude": "lat",
        "longitude": "lon",
        "init_time": "year",
        "time": "year",
    }.items():
        if src in da.dims and dst not in da.dims:
            rename[src] = dst
    out = da.rename(rename)
    if "year" in out.dims and np.issubdtype(out["year"].dtype, np.datetime64):
        out = out.assign_coords(year=out["year"].dt.year.values)
    if "member" in out.dims:
        out = out.mean("member", skipna=True)
    wanted = [d for d in ("year", "lat", "lon") if d in out.dims]
    return out.transpose(*wanted, ...)


def _score_pair(name: str, first: xr.DataArray, second: xr.DataArray) -> PairScore:
    try:
        if {"lat", "lon"}.issubset(first.dims) and {"lat", "lon"}.issubset(second.dims):
            second = second.interp(lat=first.lat, lon=first.lon, method="linear")
        first, second = xr.align(first, second, join="inner")
        valid = np.isfinite(first.values) & np.isfinite(second.values)
        n = int(valid.sum())
        if n < 3:
            return PairScore(name, "failed", n, error="too few finite cells")
        fv = first.values[valid].ravel()
        sv = second.values[valid].ravel()
        diff = fv - sv
        corr = np.corrcoef(fv, sv)[0, 1] if np.std(fv) > 0 and np.std(sv) > 0 else np.nan
        first_masked = first.where(np.isfinite(first) & np.isfinite(second))
        second_masked = second.where(np.isfinite(first) & np.isfinite(second))
        first_summary = _summary_stats(first_masked)
        second_summary = _summary_stats(second_masked)
        return PairScore(
            comparison=name,
            status="ok",
            n_cells=n,
            bias=float(diff.mean()),
            mae=float(np.abs(diff).mean()),
            rmse=float(np.sqrt((diff**2).mean())),
            corr=float(corr),
            first_summary=first_summary,
            second_summary=second_summary,
            distribution_bias=_summary_bias(first_summary, second_summary),
        )
    except Exception as exc:
        return PairScore(name, "failed", 0, error=f"{type(exc).__name__}: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepscale", required=True, type=Path)
    parser.add_argument("--external", required=True, type=Path)
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--deepscale-var")
    parser.add_argument("--external-var")
    parser.add_argument("--benchmark-var")
    parser.add_argument("--out", type=Path, default=Path("validation/results/external_archive_compare.json"))
    args = parser.parse_args()

    deepscale = _standardize_dims(_pick(xr.open_dataset(args.deepscale), args.deepscale_var))
    external = _standardize_dims(_pick(xr.open_dataset(args.external), args.external_var))

    scores = [_score_pair("deepscale_vs_external", deepscale, external)]
    benchmark_payload = None
    if args.benchmark:
        benchmark = _standardize_dims(_pick(xr.open_dataset(args.benchmark), args.benchmark_var))
        scores.append(_score_pair("deepscale_vs_benchmark", deepscale, benchmark))
        scores.append(_score_pair("external_vs_benchmark", external, benchmark))
        benchmark_payload = {
            "path": str(args.benchmark),
            "variable": args.benchmark_var,
            "summary": _summary_stats(benchmark),
        }

    payload = {
        "inputs": {
            "deepscale": {"path": str(args.deepscale), "variable": args.deepscale_var, "summary": _summary_stats(deepscale)},
            "external": {"path": str(args.external), "variable": args.external_var, "summary": _summary_stats(external)},
            "benchmark": benchmark_payload,
        },
        "scores": [asdict(score) for score in scores],
        "notes": [
            "Use this for local external downscaled NMME archives once file paths are available.",
            "The script aligns common coordinates and reports min/mean/max/std and p50/p70/p90/p95 distribution diagnostics.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
