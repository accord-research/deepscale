"""PyCPT/CPT parity fixture for DeepScale CCA.

This script has two execution modes:

1. Host mode, from the DeepScale repo:

   PYTHONPATH=/Users/david/rosetta/src .venv/bin/python \
     validation/pycpt_cca_parity.py prepare

   This fetches/prepares a small CHIRPS high/low fixture and writes DeepScale
   LOYO CCA predictions.

2. Container mode, using the existing PyCPT image:

   docker run --rm -v "$PWD/validation:/work/validation" -w /work \
     pycpt:2.10.4 bash -lc \
     'source /opt/conda/etc/profile.d/conda.sh && conda activate pycpt && \
      python validation/pycpt_cca_parity.py run-cpt'

   This runs cptcore.canonical_correlation_analysis against the same fixture.

Finally compare on the host:

   .venv/bin/python validation/pycpt_cca_parity.py compare
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation" / "results"


def _paths(tag: str) -> dict[str, Path]:
    stem = "pycpt_cca" if tag == "texas" else f"pycpt_cca_{tag}"
    workspace = "pycpt_workspace" if tag == "texas" else f"pycpt_workspace_{tag}"
    return {
        "fixture": OUT / f"{stem}_fixture.nc",
        "deepscale": OUT / f"{stem}_deepscale_loyo.nc",
        "cpt": OUT / f"{stem}_cptcore_crossvalidation.nc",
        "compare": OUT / f"{stem}_comparison.json",
        "metadata": OUT / f"{stem}_metadata.json",
        "workspace": OUT / workspace,
    }


def _add_rosetta_path() -> None:
    rosetta_src = Path.home() / "rosetta" / "src"
    if rosetta_src.exists() and str(rosetta_src) not in sys.path:
        sys.path.insert(0, str(rosetta_src))


def _score(a: xr.DataArray, b: xr.DataArray) -> dict:
    a, b = xr.align(a, b, join="inner")
    valid = np.isfinite(a.values) & np.isfinite(b.values)
    if int(valid.sum()) < 2:
        return {"status": "failed", "error": "too few finite overlapping cells"}
    av = a.values[valid]
    bv = b.values[valid]
    diff = av - bv
    corr = np.corrcoef(av.ravel(), bv.ravel())[0, 1]
    return {
        "status": "ok",
        "n_cells": int(valid.sum()),
        "bias": float(np.mean(diff)),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "corr": float(corr),
        "max_abs_diff": float(np.max(np.abs(diff))),
    }


def prepare(args) -> None:
    _add_rosetta_path()
    import rosetta
    from deepscale.methods.cca import CCAMethod

    OUT.mkdir(parents=True, exist_ok=True)
    paths = _paths(args.tag)
    ds = rosetta.fetch(
        args.product,
        "precip",
        hindcast=(args.start_year, args.end_year),
        region=args.bbox,
        cache=not args.no_cache,
        verbose=True,
        progress=False,
    )
    precip = ds["precip"]
    if args.months:
        precip = precip.sel(time=precip.time.dt.month.isin(args.months))
    obs = precip.groupby("time.year").mean("time", skipna=True).astype("float64")
    obs.attrs["units"] = "mm/day"
    coarse = obs.coarsen(lat=args.coarsen_factor, lon=args.coarsen_factor, boundary="trim").mean(skipna=True)
    hindcast = xr.concat(
        [coarse * (1.0 + (i - 1) * 0.01) for i in range(3)],
        dim=xr.DataArray([0, 1, 2], dims="member", name="member"),
    ).transpose("year", "member", "lat", "lon")

    xr.Dataset({"hindcast": hindcast, "obs": obs}).to_netcdf(paths["fixture"])

    preds = []
    truth = []
    years = list(obs.year.values)
    for year in years:
        train_years = [y for y in years if y != year]
        model = CCAMethod(n_modes=args.cca_modes, x_eof_modes=args.x_eof_modes, y_eof_modes=args.y_eof_modes)
        model.fit(hindcast.sel(year=train_years), obs.sel(year=train_years))
        pred = model.predict(hindcast.sel(year=[year]).isel(year=0, drop=True)).mean("member")
        preds.append(pred.expand_dims(year=[year]))
        truth.append(obs.sel(year=[year]))
    pred = xr.concat(preds, "year").rename("deepscale_cca")
    truth = xr.concat(truth, "year").rename("obs")
    xr.Dataset({"deepscale_cca": pred, "obs": truth}).to_netcdf(paths["deepscale"])
    print(f"Wrote {paths['fixture']}")
    print(f"Wrote {paths['deepscale']}")
    print(json.dumps({"deepscale_vs_obs": _score(pred, truth)}, indent=2))


def _to_cpt_da(da: xr.DataArray, *, name: str) -> xr.DataArray:
    da = da.reset_coords(drop=True)
    out = da.rename({"lon": "X", "lat": "Y", "year": "T"}).transpose("T", "Y", "X")
    years = out["T"].values.astype(int)
    out = out.assign_coords(
        T=np.array([f"{y}-07-01" for y in years], dtype="datetime64[D]"),
        Ti=("T", np.array([f"{y}-01-01" for y in years], dtype="datetime64[D]")),
        Tf=("T", np.array([f"{y}-12-31" for y in years], dtype="datetime64[D]")),
    )
    out.name = name
    out.attrs["missing"] = -999.0
    out.attrs["units"] = "mm/day"
    return out.fillna(out.attrs["missing"])


def run_cpt(args) -> None:
    import cptcore
    import cptio

    paths = _paths(args.tag)
    fixture = xr.open_dataset(paths["fixture"])
    x = fixture["hindcast"].mean("member")
    y = fixture["obs"]
    X = _to_cpt_da(x, name="X")
    Y = _to_cpt_da(y, name="Y")
    workspace = paths["workspace"]
    metadata = {
        "reference": "PyCPT/CPT",
        "docker_image": "pycpt:2.10.4",
        "pycpt_version": _safe_version("pycpt"),
        "cptio_version": getattr(cptio, "__version__", "unknown"),
        "cptcore_version": getattr(cptcore, "__version__", "unknown"),
        "cpt_bin_dir": _safe_env("CPT_BIN_DIR"),
        "mode_settings": {
            "x_eof_modes": [args.x_eof_modes, args.x_eof_modes],
            "y_eof_modes": [args.y_eof_modes, args.y_eof_modes],
            "cca_modes": [args.cca_modes, args.cca_modes],
            "crossvalidation_window": args.crossvalidation_window,
            "validation": "crossvalidation",
            "synchronous_predictors": True,
        },
        "fixture": str(paths["fixture"]),
        "workspace": str(workspace),
        "cptio_reader_status": "not_run",
        "raw_cptv10_parser_used": False,
        "raw_cptv10_parser_scope": "CPTv10 gridded deterministic hindcast_values.txt only",
    }
    try:
        cptcore.canonical_correlation_analysis(
            X,
            Y,
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
        # In this fixture CPT successfully writes hindcast_values.txt, but
        # PyCPT/cptio 1.4.0 can fail while parsing that file back into xarray.
        # This fallback is deliberately narrow: it parses only CPTv10 gridded
        # deterministic hindcast blocks so the CCA oracle remains auditable.
        print(f"cptcore returned after writing files but parsing failed: {type(exc).__name__}: {exc}")
    metadata["raw_cptv10_parser_used"] = True
    det = _parse_cptv10_hindcast(workspace / "hindcast_values.txt")
    out = xr.Dataset({"cpt_cca": det})
    out.to_netcdf(paths["cpt"])
    metadata["output"] = str(paths["cpt"])
    metadata["parsed_hindcast_shape"] = dict(det.sizes)
    paths["metadata"].write_text(json.dumps(metadata, indent=2))
    print(f"Wrote {paths['cpt']}")
    print(f"Wrote {paths['metadata']}")


def _safe_version(module_name: str) -> str:
    try:
        module = __import__(module_name)
        return getattr(module, "__version__", "unknown")
    except Exception:
        return "unavailable"


def _safe_env(name: str) -> str | None:
    try:
        import os

        return os.environ.get(name)
    except Exception:
        return None


def _parse_cptv10_hindcast(path: Path) -> xr.DataArray:
    lines = Path(path).read_text().splitlines()
    blocks = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line.startswith("cpt:field=") and not line.startswith("cpt:T="):
            i += 1
            continue
        if "cpt:T=" not in line:
            i += 1
            continue
        t_part = line.split("cpt:T=", 1)[1].split(",", 1)[0].strip()
        year = int(t_part[:4])
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
        arr = np.array(rows, dtype=float)
        blocks.append((year, np.array(lats), np.array(lons), arr))
    if not blocks:
        raise RuntimeError(f"No CPT hindcast blocks parsed from {path}")
    years = [b[0] for b in blocks]
    lats = blocks[0][1]
    lons = blocks[0][2]
    data = np.stack([b[3] for b in blocks])
    da = xr.DataArray(
        data,
        dims=("year", "lat", "lon"),
        coords={"year": years, "lat": lats, "lon": lons},
        name="cpt_cca",
    )
    da = da.sortby("lat")
    da.attrs["units"] = "mm/day"
    return da.where(da != -999.0)


def compare(_args) -> None:
    paths = _paths(_args.tag)
    ds_deep = xr.open_dataset(paths["deepscale"])
    ds_cpt = xr.open_dataset(paths["cpt"])
    deep = ds_deep["deepscale_cca"]
    obs = ds_deep["obs"]
    cpt = ds_cpt["cpt_cca"]
    # Snap CPT to DeepScale's coordinate names and ordering.
    cpt = cpt.interp(lat=deep.lat, lon=deep.lon, method="nearest").transpose("year", "lat", "lon")
    payload = {
        "deepscale_vs_obs": _score(deep, obs),
        "cpt_vs_obs": _score(cpt, obs),
        "deepscale_vs_cpt": _score(deep, cpt),
        "files": {
            "fixture": str(paths["fixture"]),
            "deepscale": str(paths["deepscale"]),
            "cpt": str(paths["cpt"]),
            "metadata": str(paths["metadata"]),
        },
    }
    paths["compare"].write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"Wrote {paths['compare']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("--tag", default="texas")
    p.add_argument("--product", default="obs/chirps-v2-dekadal-rhiza")
    p.add_argument("--start-year", type=int, default=2010)
    p.add_argument("--end-year", type=int, default=2021)
    p.add_argument("--bbox", nargs=4, type=float, default=[30.0, 35.0, -100.0, -95.0])
    p.add_argument("--months", nargs="+", type=int)
    p.add_argument("--coarsen-factor", type=int, default=3)
    p.add_argument("--x-eof-modes", type=int, default=3)
    p.add_argument("--y-eof-modes", type=int, default=3)
    p.add_argument("--cca-modes", type=int, default=2)
    p.add_argument("--no-cache", action="store_true")
    p.set_defaults(func=prepare)

    p = sub.add_parser("run-cpt")
    p.add_argument("--tag", default="texas")
    p.add_argument("--x-eof-modes", type=int, default=3)
    p.add_argument("--y-eof-modes", type=int, default=3)
    p.add_argument("--cca-modes", type=int, default=2)
    p.add_argument("--crossvalidation-window", type=int, default=1)
    p.set_defaults(func=run_cpt)

    p = sub.add_parser("compare")
    p.add_argument("--tag", default="texas")
    p.set_defaults(func=compare)

    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
