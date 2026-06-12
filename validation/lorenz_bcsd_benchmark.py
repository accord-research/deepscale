"""Lorenz et al. SEAS5-BCSD benchmark helper.

This validation artifact is intentionally separate from the production method
code. It documents the public WDCC D03 monthly precipitation dataset from
Lorenz et al. (2021) and, when local NetCDF files are available, creates
paper-style BCSD diagnostics.

The WDCC landing page marks the dataset as available to registered users. The
default metadata-only mode therefore records the benchmark target without
attempting a gated download.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re

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


DATASET_METADATA = {
    "paper": {
        "citation": "Lorenz et al. (2021), Earth System Science Data, 13, 2701-2724",
        "url": "https://essd.copernicus.org/articles/13/2701/2021/",
        "title": "Bias-corrected and spatially disaggregated seasonal forecasts: a long-term reference forecast product for the water sector in semi-arid regions",
    },
    "wdcc_group": {
        "name": "SaWaM D03 SEAS5 BCSD",
        "doi": "10.26050/WDCC/SaWaM_D03_SEAS5_BCSD",
        "url": "https://doi.org/10.26050/WDCC/SaWaM_D03_SEAS5_BCSD",
        "domain": "Tekeze-Atbara and Blue Nile Basins (Ethiopia/Eritrea/Sudan)",
        "spatial_coverage": {"lat_min": 6.1, "lat_max": 21.0, "lon_min": 31.0, "lon_max": 41.0},
        "temporal_coverage": "1981-01-01/2020-07-02",
        "aggregate_size_bytes": 213745684267,
    },
    "monthly_precipitation_dataset": {
        "entry_acronym": "D03_BCSD_monthly_pr",
        "handle": "hdl:21.14106/dbfbc0bc5c65f0be4b7548885ad3b7b6d0845de8",
        "url": "https://www.wdc-climate.de/ui/entry?acronym=D03_BCSD_monthly_pr",
        "description": "Monthly forecasts for precipitation from the issue date for the next 6 months.",
        "variable": "precipitation_rate",
        "unit": "mm d-1",
        "format": "NetCDF",
        "size_bytes": 1053358635,
        "temporal_coverage": "1981-01-01/2020-06-30",
        "access_constraint": "registered users",
    },
    "paper_method_summary": {
        "forecast": "ECMWF SEAS5",
        "reference": "ERA5-Land",
        "resolution_change": "approximately 36 km to 0.1 degrees",
        "product_period": "1981-2019 hindcast/reforecast period",
        "monthly_leads": 6,
        "diagnostic_role": "external paper-backed BCSD benchmark, not a DeepScale unit-test oracle",
    },
}


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


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2))
    print(f"Wrote {path}")


def _issue_from_filename(path: Path) -> np.datetime64 | None:
    match = re.search(r"_(\d{6})\.nc$", path.name)
    if not match:
        return None
    yyyymm = match.group(1)
    return np.datetime64(f"{yyyymm[:4]}-{yyyymm[4:]}-01")


def _open_issue_file(path: Path) -> xr.Dataset:
    ds = xr.open_dataset(path)
    issue = _issue_from_filename(path)
    if issue is None:
        return ds
    if "time" in ds.dims:
        ds = ds.rename({"time": "lead"})
        ds = ds.assign_coords(lead=np.arange(ds.sizes["lead"], dtype=int))
    return ds.expand_dims(issue=[issue])


def _open_dataset_or_directory(path: Path) -> xr.Dataset:
    if path.is_dir():
        files = sorted(path.glob("*.nc"))
        if not files:
            raise ValueError(f"no .nc files found in {path}")
        if _issue_from_filename(files[0]) is None:
            return xr.open_mfdataset(files, combine="by_coords")
        return xr.concat([_open_issue_file(file) for file in files], dim="issue", join="outer")
    return xr.open_dataset(path)


def _guess_precip_var(ds: xr.Dataset, requested: str | None) -> str:
    if requested:
        if requested not in ds:
            raise ValueError(f"requested variable {requested!r} not found; available variables: {list(ds.data_vars)}")
        return requested
    preferred = ["precipitation_rate", "pr", "precip", "PREC"]
    for name in preferred:
        if name in ds:
            return name
    numeric = [name for name, da in ds.data_vars.items() if np.issubdtype(da.dtype, np.number)]
    if not numeric:
        raise ValueError("no numeric data variable found")
    return numeric[0]


def _guess_dim(da: xr.DataArray, candidates: list[str]) -> str | None:
    lowered = {dim.lower(): dim for dim in da.dims}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    for dim in da.dims:
        low = dim.lower()
        if any(candidate in low for candidate in candidates):
            return dim
    return None


def _spatial_dims(da: xr.DataArray) -> tuple[str, str]:
    lat = _guess_dim(da, ["lat", "latitude"])
    lon = _guess_dim(da, ["lon", "longitude"])
    if not lat or not lon:
        raise ValueError(f"could not infer spatial dims from {da.dims}")
    return lat, lon


def _time_dim(da: xr.DataArray) -> str | None:
    return _guess_dim(da, ["time", "issue", "date", "forecast_reference_time"])


def _lead_dim(da: xr.DataArray) -> str | None:
    return _guess_dim(da, ["lead", "step", "forecast", "leadtime", "time2"])


def _ensemble_mean(da: xr.DataArray) -> xr.DataArray:
    lat, lon = _spatial_dims(da)
    protected = {lat, lon}
    time = _time_dim(da)
    lead = _lead_dim(da)
    if time:
        protected.add(time)
    if lead:
        protected.add(lead)
    member_dims = [
        dim
        for dim in da.dims
        if dim not in protected and any(token in dim.lower() for token in ("ens", "member", "ensemble", "realization", "number"))
    ]
    return da.mean(member_dims, skipna=True) if member_dims else da


def _month_labels(values: np.ndarray) -> np.ndarray:
    return pd.to_datetime(values).to_period("M").astype(str).to_numpy()


def _add_months(values: np.ndarray, offset: int) -> np.ndarray:
    periods = pd.to_datetime(values).to_period("M")
    return (periods + int(offset)).astype(str).to_numpy()


def _lead_offsets(da: xr.DataArray, lead_dim: str, base: str) -> list[int]:
    if base == "index" or lead_dim not in da.coords:
        return list(range(da.sizes[lead_dim]))
    coord = da[lead_dim].values
    if np.issubdtype(coord.dtype, np.timedelta64):
        days = coord.astype("timedelta64[D]").astype(float)
        offsets = np.rint(days / 30.4375).astype(int)
    elif np.issubdtype(coord.dtype, np.number):
        offsets = np.asarray(coord, dtype=int)
    else:
        return list(range(da.sizes[lead_dim]))
    if base == "one":
        return [int(v) - 1 for v in offsets]
    if base == "zero":
        return [int(v) for v in offsets]
    if offsets.size and int(np.nanmin(offsets)) == 1:
        return [int(v) - 1 for v in offsets]
    return [int(v) for v in offsets]


def _mean_field(da: xr.DataArray, *, lead_index: int | None = None, issue_month: int | None = None) -> xr.DataArray:
    lat, lon = _spatial_dims(da)
    subset = da
    time_dim = _guess_dim(da, ["time", "issue", "date"])
    lead_dim = _guess_dim(da, ["lead", "step", "forecast"])
    if lead_dim and lead_index is not None:
        subset = subset.isel({lead_dim: lead_index})
    if time_dim and issue_month is not None and np.issubdtype(subset[time_dim].dtype, np.datetime64):
        subset = subset.where(subset[time_dim].dt.month == issue_month, drop=True)
    reduce_dims = [dim for dim in subset.dims if dim not in (lat, lon)]
    if reduce_dims:
        subset = subset.mean(reduce_dims, skipna=True)
    return subset.transpose(lat, lon)


def _score_against_reference(pred: xr.DataArray, ref: xr.DataArray) -> dict:
    pred, ref = xr.align(pred, ref, join="inner")
    valid = np.isfinite(pred.values) & np.isfinite(ref.values)
    if int(valid.sum()) < 2:
        return {"n_cells": int(valid.sum()), "bias": None, "mae": None, "rmse": None, "corr": None}
    diff = pred.values[valid] - ref.values[valid]
    pv = pred.values[valid].ravel()
    rv = ref.values[valid].ravel()
    corr = np.corrcoef(pv, rv)[0, 1] if np.std(pv) > 0 and np.std(rv) > 0 else np.nan
    return {
        "n_cells": int(valid.sum()),
        "bias": float(diff.mean()),
        "mae": float(np.abs(diff).mean()),
        "rmse": float(np.sqrt((diff**2).mean())),
        "corr": float(corr) if np.isfinite(corr) else None,
    }


def _interp_to_field(source: xr.DataArray, target: xr.DataArray) -> xr.DataArray:
    source_lat, source_lon = _spatial_dims(source)
    target_lat, target_lon = _spatial_dims(target)
    renamed = source.rename({source_lat: target_lat, source_lon: target_lon})
    return renamed.interp({target_lat: target[target_lat], target_lon: target[target_lon]})


def _plot_fields(fields: list[tuple[str, xr.DataArray]], path: Path) -> None:
    lat, lon = _spatial_dims(fields[0][1])
    values = xr.concat([field for _, field in fields], "field")
    vmin = float(values.quantile(0.02))
    vmax = float(values.quantile(0.98))
    fig, axes = plt.subplots(1, len(fields), figsize=(4.6 * len(fields), 4.4), constrained_layout=True)
    if len(fields) == 1:
        axes = [axes]
    mesh = None
    for ax, (title, field) in zip(axes, fields):
        mesh = ax.pcolormesh(field[lon], field[lat], field, cmap="YlGnBu", vmin=vmin, vmax=vmax, shading="auto")
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    fig.colorbar(mesh, ax=axes, shrink=0.82, label="Precipitation (mm/day)")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {path}")


def _plot_lead_means(da: xr.DataArray, path: Path) -> dict:
    lead_dim = _guess_dim(da, ["lead", "step", "forecast"])
    if not lead_dim:
        return {"status": "skipped", "reason": "no lead dimension found"}
    reduce_dims = [dim for dim in da.dims if dim != lead_dim]
    lead_mean = da.mean(reduce_dims, skipna=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    ax.plot(np.arange(1, lead_mean.sizes[lead_dim] + 1), lead_mean.values, marker="o", color="#2f6f9f")
    ax.set_title("Lorenz D03 SEAS5-BCSD Monthly Precipitation by Lead")
    ax.set_xlabel("Lead month")
    ax.set_ylabel("Mean precipitation (mm/day)")
    ax.grid(alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {path}")
    return {"status": "ok", "lead_means": lead_mean.values.tolist()}


def _paper_metric_table(
    forecast: xr.DataArray,
    reference: xr.DataArray,
    *,
    lead_base: str,
) -> dict:
    """Reproduce the paper's monthly lead-dependent bias/RMSE diagnostic shape."""
    forecast = _ensemble_mean(forecast)
    forecast_time = _time_dim(forecast)
    reference_time = _time_dim(reference)
    lead = _lead_dim(forecast)
    if not forecast_time or not reference_time or not lead:
        return {
            "status": "skipped",
            "reason": f"required dims missing: forecast_time={forecast_time}, reference_time={reference_time}, lead={lead}",
        }

    reference_on_grid = _interp_to_field(reference, forecast.isel({forecast_time: 0, lead: 0}, drop=True))
    reference_months = _month_labels(reference_on_grid[reference_time].values)
    reference_lookup = {month: i for i, month in enumerate(reference_months)}
    issue_times = forecast[forecast_time].values
    issue_months = pd.to_datetime(issue_times).month.to_numpy()
    offsets = _lead_offsets(forecast, lead, lead_base)

    bias = np.full((forecast.sizes[lead], 12), np.nan, dtype=float)
    rmse = np.full_like(bias, np.nan)
    n_samples = np.zeros_like(bias, dtype=int)
    for lead_index, offset in enumerate(offsets):
        target_months = _add_months(issue_times, offset)
        lead_field = forecast.isel({lead: lead_index})
        for month in range(1, 13):
            pairs = [
                (i, reference_lookup[target])
                for i, target in enumerate(target_months)
                if issue_months[i] == month and target in reference_lookup
            ]
            if not pairs:
                continue
            pred = xr.concat([lead_field.isel({forecast_time: i}, drop=True) for i, _ in pairs], "sample")
            ref = xr.concat([reference_on_grid.isel({reference_time: j}, drop=True) for _, j in pairs], "sample")
            score = _score_against_reference(pred, ref)
            bias[lead_index, month - 1] = score["bias"] if score["bias"] is not None else np.nan
            rmse[lead_index, month - 1] = score["rmse"] if score["rmse"] is not None else np.nan
            n_samples[lead_index, month - 1] = len(pairs)

    return {
        "status": "ok",
        "lead_offsets_months": offsets,
        "issue_months": list(range(1, 13)),
        "bias_by_lead_issue_month": bias,
        "rmse_by_lead_issue_month": rmse,
        "n_samples_by_lead_issue_month": n_samples,
    }


def _plot_paper_metric_tables(tables: dict[str, dict], path: Path) -> None:
    ok_tables = {name: table for name, table in tables.items() if table.get("status") == "ok"}
    if not ok_tables:
        return
    months = np.arange(1, 13)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, max(table["bias_by_lead_issue_month"].shape[0] for table in ok_tables.values())))
    styles = {"raw": "--", "bcsd": "-"}
    labels = {"raw": "Raw SEAS5", "bcsd": "SEAS5-BCSD"}
    for table_name, table in ok_tables.items():
        style = styles.get(table_name, "-")
        label_prefix = labels.get(table_name, table_name)
        bias = np.asarray(table["bias_by_lead_issue_month"], dtype=float)
        rmse = np.asarray(table["rmse_by_lead_issue_month"], dtype=float)
        for lead_index in range(bias.shape[0]):
            label = f"{label_prefix} L{lead_index + 1}"
            axes[0].plot(months, bias[lead_index], linestyle=style, color=colors[lead_index], alpha=0.9, label=label)
            axes[1].plot(months, rmse[lead_index], linestyle=style, color=colors[lead_index], alpha=0.9, label=label)
    axes[0].axhline(0, color="#555555", linewidth=0.8)
    axes[0].set_title("Monthly Bias by Lead")
    axes[0].set_ylabel("Forecast - ERA5-Land (mm/day)")
    axes[1].set_title("Monthly RMSE by Lead")
    axes[1].set_ylabel("RMSE (mm/day)")
    for ax in axes:
        ax.set_xlabel("Issue month")
        ax.set_xticks(months)
        ax.grid(alpha=0.25)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {path}")


def build_benchmark(args: argparse.Namespace) -> dict:
    if not args.forecast_nc:
        return {
            **DATASET_METADATA,
            "status": "metadata_only",
            "notes": [
                "WDCC metadata is public, but D03_BCSD_monthly_pr download options require a registered-user session.",
                "Run this script with --forecast-nc after downloading the NetCDF from WDCC.",
                "Optional --reference-nc and --raw-nc inputs enable bias/RMSE comparisons against ERA5-Land and raw SEAS5-style fields.",
            ],
        }

    ds = _open_dataset_or_directory(args.forecast_nc)
    var = _guess_precip_var(ds, args.variable)
    bcsd = ds[var]
    bcsd_field = _mean_field(bcsd, lead_index=args.lead_index, issue_month=args.issue_month)
    fields = [("SEAS5-BCSD", bcsd_field)]
    scores = {}

    raw_path = Path(args.raw_nc) if args.raw_nc else None
    if raw_path:
        raw_ds = _open_dataset_or_directory(raw_path)
        raw_var = _guess_precip_var(raw_ds, args.raw_variable)
        raw_field = _mean_field(raw_ds[raw_var], lead_index=args.lead_index, issue_month=args.issue_month)
        raw_field = _interp_to_field(raw_field, bcsd_field)
        fields.insert(0, ("Raw forecast", raw_field))

    ref_path = Path(args.reference_nc) if args.reference_nc else None
    if ref_path:
        ref_ds = _open_dataset_or_directory(ref_path)
        ref_var = _guess_precip_var(ref_ds, args.reference_variable)
        ref_field = _mean_field(ref_ds[ref_var], issue_month=args.issue_month)
        ref_field = _interp_to_field(ref_field, bcsd_field)
        fields.append(("Reference", ref_field))
        scores["bcsd_vs_reference"] = _score_against_reference(bcsd_field, ref_field)
        if raw_path:
            scores["raw_vs_reference"] = _score_against_reference(fields[0][1], ref_field)

    suffix = args.suffix
    _plot_fields(fields, FIGURES / f"lorenz_bcsd_benchmark_maps_{suffix}.png")
    lead_summary = _plot_lead_means(bcsd, FIGURES / f"lorenz_bcsd_benchmark_leads_{suffix}.png")
    paper_metrics = {"status": "skipped", "reason": "--reference-nc is required for paper-style bias/RMSE diagnostics"}
    paper_metric_figure = None
    if ref_path:
        paper_metric_tables = {"bcsd": _paper_metric_table(bcsd, ref_ds[ref_var], lead_base=args.lead_base)}
        if raw_path:
            paper_metric_tables["raw"] = _paper_metric_table(raw_ds[raw_var], ref_ds[ref_var], lead_base=args.lead_base)
        paper_metric_figure = FIGURES / f"lorenz_bcsd_benchmark_paper_metrics_{suffix}.png"
        _plot_paper_metric_tables(paper_metric_tables, paper_metric_figure)
        paper_metrics = {
            "status": "ok" if any(table.get("status") == "ok" for table in paper_metric_tables.values()) else "skipped",
            "role": "paper Figure 2/Figure 3 style: monthly bias and RMSE by lead against ERA5-Land",
            "tables": paper_metric_tables,
            "figure": str(paper_metric_figure),
        }

    return {
        **DATASET_METADATA,
        "status": "ok",
        "input_files": {
            "forecast_nc": str(args.forecast_nc),
            "raw_nc": str(args.raw_nc) if args.raw_nc else None,
            "reference_nc": str(args.reference_nc) if args.reference_nc else None,
        },
        "selected_variable": var,
        "forecast_dims": dict(bcsd.sizes),
        "lead_index": args.lead_index,
        "issue_month": args.issue_month,
        "scores": scores,
        "lead_summary": lead_summary,
        "paper_metrics": paper_metrics,
        "figures": [
            str(FIGURES / f"lorenz_bcsd_benchmark_maps_{suffix}.png"),
            str(FIGURES / f"lorenz_bcsd_benchmark_leads_{suffix}.png"),
        ]
        + ([str(paper_metric_figure)] if paper_metric_figure else []),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecast-nc", type=Path, help="Local WDCC D03_BCSD_monthly_pr NetCDF file or extracted issue-file directory.")
    parser.add_argument("--raw-nc", type=Path, help="Optional raw SEAS5 monthly precipitation NetCDF aligned to the benchmark.")
    parser.add_argument("--reference-nc", type=Path, help="Optional ERA5-Land/reference monthly precipitation NetCDF.")
    parser.add_argument("--variable", help="Variable name in --forecast-nc.")
    parser.add_argument("--raw-variable", help="Variable name in --raw-nc.")
    parser.add_argument("--reference-variable", help="Variable name in --reference-nc.")
    parser.add_argument("--lead-index", type=int, default=0, help="Zero-based lead index for map diagnostics.")
    parser.add_argument(
        "--lead-base",
        choices=["auto", "index", "zero", "one"],
        default="auto",
        help="How to interpret numeric lead coordinates when matching forecasts to reference months.",
    )
    parser.add_argument("--issue-month", type=int, help="Optional issue month filter, 1-12.")
    parser.add_argument("--suffix", default="d03_monthly_pr")
    parser.add_argument("--out-json", type=Path, default=RESULTS / "lorenz_bcsd_benchmark_d03_monthly_pr.json")
    args = parser.parse_args()

    payload = build_benchmark(args)
    _write_json(args.out_json, payload)
    print(json.dumps(_json_safe(payload), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
