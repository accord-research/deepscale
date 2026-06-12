"""Synthetic no-interpolation CCA parity fixture for DeepScale vs CPT.

This fixture is intentionally small and deterministic. Predictor and predictand
use the same 3x3 grid, no missing values, one ensemble member, and fixed CCA
mode counts. That removes the interpolation and data-shape ambiguity from the
CHIRPS parity fixture so remaining differences point to CCA conventions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation" / "results"
FIGURES = ROOT / "validation" / "figures"
FIXTURE = OUT / "cca_synthetic_fixture.nc"
DEEPSCALE_OUT = OUT / "cca_synthetic_deepscale_loyo.nc"
CPT_OUT = OUT / "cca_synthetic_cptcore_crossvalidation.nc"
COMPARE_OUT = OUT / "cca_synthetic_cpt_comparison.json"
METADATA_OUT = OUT / "cca_synthetic_cpt_metadata.json"

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "validation" / ".matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _score(a: xr.DataArray, b: xr.DataArray) -> dict:
    a, b = xr.align(a, b, join="inner")
    valid = np.isfinite(a.values) & np.isfinite(b.values)
    if int(valid.sum()) < 2:
        return {"status": "failed", "error": "too few finite overlapping cells"}
    av = a.values[valid].ravel()
    bv = b.values[valid].ravel()
    diff = av - bv
    corr = np.corrcoef(av, bv)[0, 1] if np.std(av) > 0 and np.std(bv) > 0 else np.nan
    return {
        "status": "ok",
        "n_cells": int(valid.sum()),
        "bias": float(np.mean(diff)),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "corr": float(corr),
        "max_abs_diff": float(np.max(np.abs(diff))),
    }


def _make_synthetic_fixture(start_year: int, n_years: int) -> xr.Dataset:
    years = np.arange(start_year, start_year + n_years)
    lat = np.array([-1.0, 0.0, 1.0])
    lon = np.array([35.0, 36.0, 37.0])
    yy, xx = np.meshgrid(lat, lon - lon.mean(), indexing="ij")
    t = np.arange(n_years, dtype=float)

    latent1 = np.sin(2.0 * np.pi * t / 7.0)
    latent2 = np.cos(2.0 * np.pi * (t + 1.5) / 5.0)

    x_pattern1 = 1.0 + 0.20 * yy - 0.10 * xx
    x_pattern2 = -0.4 + 0.15 * xx + 0.08 * yy
    y_pattern1 = 0.9 + 0.10 * yy + 0.12 * xx
    y_pattern2 = -0.2 + 0.20 * yy - 0.06 * xx

    x = (
        10.0
        + 1.4 * latent1[:, None, None] * x_pattern1[None, :, :]
        + 0.9 * latent2[:, None, None] * x_pattern2[None, :, :]
    )
    y = (
        20.0
        + 1.2 * latent1[:, None, None] * y_pattern1[None, :, :]
        + 0.7 * latent2[:, None, None] * y_pattern2[None, :, :]
    )

    hindcast = xr.DataArray(
        x[:, None, :, :],
        dims=("year", "member", "lat", "lon"),
        coords={"year": years, "member": [0], "lat": lat, "lon": lon},
        name="hindcast",
        attrs={"units": "synthetic"},
    )
    obs = xr.DataArray(
        y,
        dims=("year", "lat", "lon"),
        coords={"year": years, "lat": lat, "lon": lon},
        name="obs",
        attrs={"units": "synthetic"},
    )
    return xr.Dataset({"hindcast": hindcast.astype("float64"), "obs": obs.astype("float64")})


def prepare(args) -> None:
    from deepscale.methods.cca import CCAMethod

    OUT.mkdir(parents=True, exist_ok=True)
    fixture = _make_synthetic_fixture(args.start_year, args.n_years)
    fixture.to_netcdf(FIXTURE)
    hindcast = fixture["hindcast"]
    obs = fixture["obs"]

    preds = []
    years = list(obs.year.values)
    for year in years:
        train_years = [y for y in years if y != year]
        model = CCAMethod(n_modes=args.cca_modes, x_eof_modes=args.x_eof_modes, y_eof_modes=args.y_eof_modes)
        model.fit(hindcast.sel(year=train_years), obs.sel(year=train_years))
        pred = model.predict(hindcast.sel(year=[year]).isel(year=0, drop=True)).mean("member")
        preds.append(pred.expand_dims(year=[year]))
    deep = xr.concat(preds, "year").rename("deepscale_cca")
    xr.Dataset({"deepscale_cca": deep, "obs": obs}).to_netcdf(DEEPSCALE_OUT)
    print(f"Wrote {FIXTURE}")
    print(f"Wrote {DEEPSCALE_OUT}")
    print(json.dumps({"deepscale_vs_obs": _score(deep, obs)}, indent=2))


def _to_cpt_da(da: xr.DataArray, *, name: str) -> xr.DataArray:
    out = da.rename({"lon": "X", "lat": "Y", "year": "T"}).transpose("T", "Y", "X")
    years = out["T"].values.astype(int)
    out = out.assign_coords(
        T=np.array([f"{y}-07-01" for y in years], dtype="datetime64[D]"),
        Ti=("T", np.array([f"{y}-01-01" for y in years], dtype="datetime64[D]")),
        Tf=("T", np.array([f"{y}-12-31" for y in years], dtype="datetime64[D]")),
    )
    out.name = name
    out.attrs["missing"] = -999.0
    out.attrs["units"] = da.attrs.get("units", "synthetic")
    return out.fillna(out.attrs["missing"])


def run_cpt(args) -> None:
    import cptcore
    import cptio

    fixture = xr.open_dataset(FIXTURE)
    x = fixture["hindcast"].mean("member")
    y = fixture["obs"]
    workspace = OUT / "cca_synthetic_cpt_workspace"
    metadata = {
        "reference": "PyCPT/CPT",
        "docker_image": "pycpt:2.10.4",
        "pycpt_version": _safe_version("pycpt"),
        "cptio_version": getattr(cptio, "__version__", "unknown"),
        "cptcore_version": getattr(cptcore, "__version__", "unknown"),
        "mode_settings": {
            "x_eof_modes": [args.x_eof_modes, args.x_eof_modes],
            "y_eof_modes": [args.y_eof_modes, args.y_eof_modes],
            "cca_modes": [args.cca_modes, args.cca_modes],
            "crossvalidation_window": args.crossvalidation_window,
            "validation": "crossvalidation",
            "synchronous_predictors": True,
        },
        "fixture": str(FIXTURE),
        "workspace": str(workspace),
        "raw_cptv10_parser_scope": "CPTv10 gridded deterministic hindcast_values.txt only",
    }
    try:
        cptcore.canonical_correlation_analysis(
            _to_cpt_da(x, name="X"),
            _to_cpt_da(y, name="Y"),
            F=None,
            transform_predictand=None,
            tailoring=None,
            cca_modes=(args.cca_modes, args.cca_modes),
            x_eof_modes=(args.x_eof_modes, args.x_eof_modes),
            y_eof_modes=(args.y_eof_modes, args.y_eof_modes),
            crossvalidation_window=args.crossvalidation_window,
            validation="crossvalidation",
            synchronous_predictors=True,
            cpt_kwargs={"outputdir": workspace},
        )
        metadata["cptcore_status"] = "returned_cleanly"
        metadata["cptio_reader_status"] = "ok"
    except Exception as exc:
        metadata["cptcore_status"] = "raised_after_writing_outputs"
        metadata["cptio_reader_status"] = "failed"
        metadata["cptio_reader_error"] = f"{type(exc).__name__}: {exc}"
        print(f"cptcore returned after writing files but parsing failed: {type(exc).__name__}: {exc}")
    det = _parse_cptv10_hindcast(workspace / "hindcast_values.txt")
    xr.Dataset({"cpt_cca": det}).to_netcdf(CPT_OUT)
    metadata["output"] = str(CPT_OUT)
    metadata["parsed_hindcast_shape"] = dict(det.sizes)
    METADATA_OUT.write_text(json.dumps(metadata, indent=2))
    print(f"Wrote {CPT_OUT}")
    print(f"Wrote {METADATA_OUT}")


def _safe_version(module_name: str) -> str:
    try:
        module = __import__(module_name)
        return getattr(module, "__version__", "unknown")
    except Exception:
        return "unavailable"


def _parse_cptv10_hindcast(path: Path) -> xr.DataArray:
    lines = path.read_text().splitlines()
    blocks = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if "cpt:T=" not in line:
            i += 1
            continue
        year = int(line.split("cpt:T=", 1)[1].split(",", 1)[0].strip()[:4])
        i += 1
        while i < len(lines) and not lines[i].strip():
            i += 1
        lons = [float(v) for v in lines[i].strip().split()]
        i += 1
        lats = []
        rows = []
        while i < len(lines):
            row = lines[i].strip()
            if not row:
                i += 1
                continue
            if row.startswith("cpt:"):
                break
            parts = row.split()
            lats.append(float(parts[0]))
            rows.append([float(v) for v in parts[1:]])
            i += 1
        blocks.append((year, np.asarray(lats), np.asarray(lons), np.asarray(rows, dtype=float)))
    if not blocks:
        raise RuntimeError(f"No CPT hindcast blocks parsed from {path}")
    da = xr.DataArray(
        np.stack([b[3] for b in blocks]),
        dims=("year", "lat", "lon"),
        coords={"year": [b[0] for b in blocks], "lat": blocks[0][1], "lon": blocks[0][2]},
        name="cpt_cca",
        attrs={"units": "synthetic"},
    ).sortby("lat")
    return da.where(da != -999.0)


def _plot_compare(deep: xr.DataArray, cpt: xr.DataArray, obs: xr.DataArray, payload: dict) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = [
        ("DeepScale vs obs", payload["deepscale_vs_obs"]["rmse"], payload["deepscale_vs_obs"]["corr"]),
        ("CPT vs obs", payload["cpt_vs_obs"]["rmse"], payload["cpt_vs_obs"]["corr"]),
        ("CPT sign-flip vs obs", payload["cpt_anomaly_sign_flipped_vs_obs"]["rmse"], payload["cpt_anomaly_sign_flipped_vs_obs"]["corr"]),
        ("DeepScale vs CPT", payload["deepscale_vs_cpt"]["rmse"], payload["deepscale_vs_cpt"]["corr"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    labels = [r[0] for r in rows]
    axes[0].bar(labels, [r[1] for r in rows], color=["#2f6f9f", "#d9902f", "#6aa84f", "#756bb1"])
    axes[0].set_title("Synthetic CCA Parity: RMSE")
    axes[0].set_ylabel("RMSE")
    axes[0].tick_params(axis="x", rotation=20, labelsize=8)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(labels, [r[2] for r in rows], color=["#2f6f9f", "#d9902f", "#6aa84f", "#756bb1"])
    axes[1].set_title("Synthetic CCA Parity: Correlation")
    axes[1].set_ylabel("Pearson correlation")
    ymin = -1.0 if min(r[2] for r in rows) < 0 else 0.9
    axes[1].set_ylim(ymin, 1.0)
    axes[1].tick_params(axis="x", rotation=20, labelsize=8)
    axes[1].grid(axis="y", alpha=0.25)
    fig.savefig(FIGURES / "cca_synthetic_metric_bars.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    latest = int(obs.year.values[-1])
    obs_y = obs.sel(year=latest)
    deep_y = deep.sel(year=latest)
    cpt_y = cpt.sel(year=latest)
    fields = [obs_y, deep_y, cpt_y]
    vmin = float(xr.concat(fields, "field").min())
    vmax = float(xr.concat(fields, "field").max())
    diff_max = float(max(abs(deep_y - obs_y).max(), abs(cpt_y - obs_y).max(), abs(deep_y - cpt_y).max()))
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2), constrained_layout=True)
    for ax, da, title in zip(axes[0], fields, [f"Observed ({latest})", "DeepScale CCA", "CPT CCA"]):
        mesh = ax.pcolormesh(da.lon, da.lat, da, cmap="viridis", vmin=vmin, vmax=vmax, shading="auto")
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    fig.colorbar(mesh, ax=axes[0, :], shrink=0.8, label="Synthetic value")
    for ax, da, title in zip(axes[1], [deep_y - obs_y, cpt_y - obs_y, deep_y - cpt_y], ["DeepScale - obs", "CPT - obs", "DeepScale - CPT"]):
        dm = ax.pcolormesh(da.lon, da.lat, da, cmap="RdBu_r", vmin=-diff_max, vmax=diff_max, shading="auto")
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    fig.colorbar(dm, ax=axes[1, :], shrink=0.8, label="Difference")
    fig.savefig(FIGURES / "cca_synthetic_maps_latest_year.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def compare(_args) -> None:
    ds_deep = xr.open_dataset(DEEPSCALE_OUT)
    ds_cpt = xr.open_dataset(CPT_OUT)
    deep = ds_deep["deepscale_cca"].transpose("year", "lat", "lon")
    obs = ds_deep["obs"].transpose("year", "lat", "lon")
    cpt = ds_cpt["cpt_cca"].interp(lat=deep.lat, lon=deep.lon, method="nearest").transpose("year", "lat", "lon")
    cpt_anomaly_sign_flipped = (2 * obs.mean("year") - cpt).transpose("year", "lat", "lon")
    payload = {
        "deepscale_vs_obs": _score(deep, obs),
        "cpt_vs_obs": _score(cpt, obs),
        "cpt_anomaly_sign_flipped_vs_obs": _score(cpt_anomaly_sign_flipped, obs),
        "deepscale_vs_cpt": _score(deep, cpt),
        "deepscale_vs_cpt_anomaly_sign_flipped": _score(deep, cpt_anomaly_sign_flipped),
        "files": {
            "fixture": str(FIXTURE),
            "deepscale": str(DEEPSCALE_OUT),
            "cpt": str(CPT_OUT),
            "metadata": str(METADATA_OUT),
            "metric_figure": str(FIGURES / "cca_synthetic_metric_bars.png"),
            "map_figure": str(FIGURES / "cca_synthetic_maps_latest_year.png"),
        },
        "notes": [
            "Synthetic 3x3 fixture with matching predictor/predictand grids, no missing values, and one member",
            "This removes interpolation and ensemble averaging from the CCA parity check",
            "CPT output is also scored after flipping anomalies around the observed climatology to diagnose sign/orientation conventions",
        ],
    }
    COMPARE_OUT.write_text(json.dumps(payload, indent=2))
    _plot_compare(deep, cpt, obs, payload)
    print(json.dumps(payload, indent=2))
    print(f"Wrote {COMPARE_OUT}")


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("--start-year", type=int, default=1991)
    p.add_argument("--n-years", type=int, default=24)
    p.add_argument("--x-eof-modes", type=int, default=2)
    p.add_argument("--y-eof-modes", type=int, default=2)
    p.add_argument("--cca-modes", type=int, default=2)
    p.set_defaults(func=prepare)

    p = sub.add_parser("run-cpt")
    p.add_argument("--x-eof-modes", type=int, default=2)
    p.add_argument("--y-eof-modes", type=int, default=2)
    p.add_argument("--cca-modes", type=int, default=2)
    p.add_argument("--crossvalidation-window", type=int, default=1)
    p.set_defaults(func=run_cpt)

    p = sub.add_parser("compare")
    p.set_defaults(func=compare)

    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
