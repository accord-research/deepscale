"""Convention diagnostics for empirical QM against xsdba.

This isolates quantile-node and interpolation choices from the full DeepScale
method classes. It uses the same CHIRPS high/low fixture as the rest of the
validation suite.
"""

from __future__ import annotations

import json
import os
import warnings
import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import xarray as xr

from downscaling_validation import PRESETS, fetch_chirps, make_hindcast

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "validation" / "results"
FIGURES = ROOT / "validation" / "figures"
FIXTURE = RESULTS / "pycpt_cca_fixture.nc"

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "validation" / ".matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class Score:
    method: str
    reference: str
    variant: str
    status: str
    n_cells: int
    bias: float | None = None
    mae: float | None = None
    rmse: float | None = None
    corr: float | None = None
    error: str | None = None


def _score(method: str, reference: str, variant: str, pred: xr.DataArray, obs: xr.DataArray) -> Score:
    pred, obs = xr.align(pred.transpose("year", "lat", "lon"), obs.transpose("year", "lat", "lon"), join="inner")
    valid = np.isfinite(pred.values) & np.isfinite(obs.values)
    if int(valid.sum()) < 2:
        return Score(method, reference, variant, "failed", int(valid.sum()), error="too few finite cells")
    p = pred.values[valid].ravel()
    o = obs.values[valid].ravel()
    d = p - o
    corr = np.corrcoef(p, o)[0, 1] if np.std(p) > 0 and np.std(o) > 0 else np.nan
    return Score(method, reference, variant, "ok", int(valid.sum()), float(d.mean()), float(np.abs(d).mean()), float(np.sqrt((d**2).mean())), float(corr))


def _make_dense_hindcast(obs: xr.DataArray, factor: int = 3, members: int = 3) -> xr.DataArray:
    coarse = obs.coarsen(lat=factor, lon=factor, boundary="trim").mean(skipna=True)
    member_fields = []
    for i in range(members):
        scale = 1.0 + (i - (members - 1) / 2.0) * 0.01
        member_fields.append(coarse * scale)
    hindcast = xr.concat(member_fields, dim=xr.DataArray(np.arange(members), dims="member", name="member"))
    return hindcast.transpose("year", "member", "lat", "lon")


def _year_to_time(da: xr.DataArray) -> xr.DataArray:
    years = da.year.values.astype(int)
    out = da.rename({"year": "time"})
    out = out.assign_coords(time=np.array([f"{y}-07-01" for y in years], dtype="datetime64[D]"))
    out.attrs["units"] = da.attrs.get("units", "mm/day")
    return out


def _nodes(kind: str, n: int) -> np.ndarray:
    if kind == "midpoint":
        return (np.arange(n) + 0.5) / n
    if kind == "xsdba":
        return np.linspace(1 / n / 2, 1 - 1 / n / 2, n)
    if kind == "weibull":
        return (np.arange(n) + 1.0) / (n + 1.0)
    raise ValueError(f"unknown quantile node kind {kind!r}")


def _nearest_interp(x: float, xp: np.ndarray, fp: np.ndarray) -> float:
    if x <= xp[0]:
        return float(fp[0])
    if x >= xp[-1]:
        return float(fp[-1])
    return float(fp[int(np.argmin(np.abs(xp - x)))])


def _map_direct(x: np.ndarray, hist: np.ndarray, ref: np.ndarray, nodes: np.ndarray, interp: str) -> np.ndarray:
    hist_q = np.nanquantile(hist, nodes, axis=0)
    ref_q = np.nanquantile(ref, nodes, axis=0)
    out = np.full(x.shape, np.nan, dtype=float)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            h = hist_q[:, i, j]
            r = ref_q[:, i, j]
            if not np.isfinite(x[i, j]) or not np.all(np.isfinite(h)) or not np.all(np.isfinite(r)):
                continue
            out[i, j] = _nearest_interp(float(x[i, j]), h, r) if interp == "nearest" else float(np.interp(x[i, j], h, r))
    return out


def _map_factor(x: np.ndarray, hist: np.ndarray, ref: np.ndarray, nodes: np.ndarray, interp: str) -> np.ndarray:
    hist_q = np.nanquantile(hist, nodes, axis=0)
    ref_q = np.nanquantile(ref, nodes, axis=0)
    af_q = ref_q - hist_q
    out = np.full(x.shape, np.nan, dtype=float)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            h = hist_q[:, i, j]
            af = af_q[:, i, j]
            if not np.isfinite(x[i, j]) or not np.all(np.isfinite(h)) or not np.all(np.isfinite(af)):
                continue
            delta = _nearest_interp(float(x[i, j]), h, af) if interp == "nearest" else float(np.interp(x[i, j], h, af))
            out[i, j] = float(x[i, j]) + delta
    return out


def _map_deepscale_sorted(x: np.ndarray, hist: np.ndarray, ref: np.ndarray) -> np.ndarray:
    hist_sorted = np.sort(hist, axis=0)
    ref_sorted = np.sort(ref, axis=0)
    nodes = (np.arange(hist.shape[0]) + 0.5) / hist.shape[0]
    out = np.full(x.shape, np.nan, dtype=float)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            h = hist_sorted[:, i, j]
            r = ref_sorted[:, i, j]
            if not np.isfinite(x[i, j]) or not np.all(np.isfinite(h)) or not np.all(np.isfinite(r)):
                continue
            q = np.interp(float(x[i, j]), h, nodes)
            out[i, j] = float(np.interp(q, nodes, r))
    return out


def _run_numpy_eqm(
    hindcast: xr.DataArray,
    obs: xr.DataArray,
    *,
    mapping: str,
    node_kind: str,
    interp: str,
    nquantiles: int | None,
) -> xr.DataArray:
    preds = []
    years = list(obs.year.values)
    for year in years:
        train_years = [y for y in years if y != year]
        hist = hindcast.sel(year=train_years).mean("member")
        ref = obs.sel(year=train_years).interp(lat=hist.lat, lon=hist.lon, method="linear")
        sim = hindcast.sel(year=[year]).isel(year=0, drop=True).mean("member")
        nodes = _nodes(node_kind, hist.sizes["year"] if nquantiles is None else nquantiles)
        if mapping == "deepscale_sorted":
            corrected = _map_deepscale_sorted(sim.values, hist.values, ref.values)
        elif mapping == "direct":
            corrected = _map_direct(sim.values, hist.values, ref.values, nodes, interp)
        elif mapping == "factor":
            corrected = _map_factor(sim.values, hist.values, ref.values, nodes, interp)
        else:
            raise ValueError(f"unknown mapping {mapping!r}")
        coarse = xr.DataArray(corrected, dims=("lat", "lon"), coords={"lat": hist.lat, "lon": hist.lon})
        fine = coarse.interp(lat=obs.lat, lon=obs.lon, method="linear")
        preds.append(fine.expand_dims(year=[year]))
    return xr.concat(preds, "year")


def _run_deepscale(method: str, hindcast: xr.DataArray, obs: xr.DataArray) -> xr.DataArray:
    from deepscale.registry import get_method

    cls = get_method(method)
    preds = []
    years = list(obs.year.values)
    for year in years:
        train_years = [y for y in years if y != year]
        model = cls(variant="empirical")
        model.fit(hindcast.sel(year=train_years), obs.sel(year=train_years))
        pred = model.predict(hindcast.sel(year=[year]).isel(year=0, drop=True)).mean("member")
        preds.append(pred.expand_dims(year=[year]))
    return xr.concat(preds, "year")


def _run_xsdba(kind: str, interp: str, nquantiles: int, hindcast: xr.DataArray, obs: xr.DataArray) -> xr.DataArray:
    from xsdba import adjustment as adj

    cls = adj.EmpiricalQuantileMapping if kind == "qm" else adj.DetrendedQuantileMapping
    preds = []
    years = list(obs.year.values)
    for year in years:
        train_years = [y for y in years if y != year]
        hist = hindcast.sel(year=train_years).mean("member")
        ref = obs.sel(year=train_years).interp(lat=hist.lat, lon=hist.lon, method="linear")
        sim = hindcast.sel(year=[year]).mean("member")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trained = cls.train(_year_to_time(ref), _year_to_time(hist), nquantiles=nquantiles, group="time", kind="+")
            corrected = trained.adjust(_year_to_time(sim), interp=interp, extrapolation="constant")
        fine = corrected.isel(time=0, drop=True).interp(lat=obs.lat, lon=obs.lon, method="linear")
        preds.append(fine.expand_dims(year=[year]))
    return xr.concat(preds, "year")


def _plot(scores: list[Score], suffix: str) -> None:
    rows = sorted([s for s in scores if s.status == "ok"], key=lambda s: s.rmse or 999)
    labels = [
        f"{s.reference}\n{s.method.upper()} {s.variant.replace('_', ' ')}"
        for s in rows
    ]
    colors = ["#2f6f9f" if s.reference == "deepscale" else "#6aa84f" if s.reference == "numpy" else "#b45f06" for s in rows]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.2), constrained_layout=True)
    axes[0].bar(labels, [s.rmse for s in rows], color=colors)
    axes[0].set_title("Empirical QM Convention Diagnostic: RMSE")
    axes[0].set_ylabel("RMSE (mm/day)")
    axes[0].tick_params(axis="x", rotation=35, labelsize=7)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(labels, [s.corr for s in rows], color=colors)
    axes[1].set_title("Empirical QM Convention Diagnostic: Correlation")
    axes[1].set_ylabel("Pearson correlation")
    axes[1].set_ylim(0.93, 1.0)
    axes[1].tick_params(axis="x", rotation=35, labelsize=7)
    axes[1].grid(axis="y", alpha=0.25)
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"empirical_qm_convention_bars_{suffix}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=sorted(PRESETS), default="texas")
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--product")
    parser.add_argument("--start-year", type=int)
    parser.add_argument("--end-year", type=int)
    parser.add_argument("--bbox", nargs=4, type=float)
    parser.add_argument("--coarsen-factor", type=int, default=3)
    parser.add_argument("--members", type=int, default=3)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--out-json")
    parser.add_argument("--out-nc")
    args = parser.parse_args()

    preset = PRESETS[args.preset]
    suffix = args.preset
    out_json = Path(args.out_json) if args.out_json else RESULTS / f"empirical_qm_conventions_{suffix}.json"
    out_nc = Path(args.out_nc) if args.out_nc else RESULTS / f"empirical_qm_conventions_predictions_{suffix}.nc"

    fixture = args.fixture
    if fixture:
        obs = xr.open_dataset(fixture)["obs"]
        hindcast = _make_dense_hindcast(obs, args.coarsen_factor, args.members)
        source = {"fixture": str(fixture)}
    else:
        product = args.product or preset["product"]
        start_year = args.start_year if args.start_year is not None else preset["start_year"]
        end_year = args.end_year if args.end_year is not None else preset["end_year"]
        bbox = args.bbox or preset["bbox"]
        months = preset.get("months")
        obs = fetch_chirps(product, (start_year, end_year), bbox, cache=not args.no_cache, months=months)
        hindcast = make_hindcast(obs, args.coarsen_factor, args.members)
        source = {"product": product, "years": [start_year, end_year], "bbox": bbox, "months": months}

    jobs = [
        ("qm", "deepscale", "empirical_current", lambda: _run_deepscale("qm", hindcast, obs)),
        ("dqm", "deepscale", "empirical_current", lambda: _run_deepscale("dqm", hindcast, obs)),
        ("qm", "xsdba", "nq8_nearest", lambda: _run_xsdba("qm", "nearest", 8, hindcast, obs)),
        ("qm", "xsdba", "nq8_linear", lambda: _run_xsdba("qm", "linear", 8, hindcast, obs)),
        ("dqm", "xsdba", "nq8_nearest", lambda: _run_xsdba("dqm", "nearest", 8, hindcast, obs)),
        ("dqm", "xsdba", "nq8_linear", lambda: _run_xsdba("dqm", "linear", 8, hindcast, obs)),
        ("qm", "numpy", "direct_midpoint_linear", lambda: _run_numpy_eqm(hindcast, obs, mapping="direct", node_kind="midpoint", interp="linear", nquantiles=None)),
        ("qm", "numpy", "deepscale_sorted", lambda: _run_numpy_eqm(hindcast, obs, mapping="deepscale_sorted", node_kind="midpoint", interp="linear", nquantiles=None)),
        ("qm", "numpy", "direct_xsdba_nearest", lambda: _run_numpy_eqm(hindcast, obs, mapping="direct", node_kind="xsdba", interp="nearest", nquantiles=8)),
        ("qm", "numpy", "direct_xsdba_linear", lambda: _run_numpy_eqm(hindcast, obs, mapping="direct", node_kind="xsdba", interp="linear", nquantiles=8)),
        ("qm", "numpy", "factor_xsdba_nearest", lambda: _run_numpy_eqm(hindcast, obs, mapping="factor", node_kind="xsdba", interp="nearest", nquantiles=8)),
        ("qm", "numpy", "factor_xsdba_linear", lambda: _run_numpy_eqm(hindcast, obs, mapping="factor", node_kind="xsdba", interp="linear", nquantiles=8)),
        ("qm", "numpy", "direct_weibull_linear", lambda: _run_numpy_eqm(hindcast, obs, mapping="direct", node_kind="weibull", interp="linear", nquantiles=None)),
    ]

    predictions: dict[str, xr.DataArray] = {}
    scores: list[Score] = []
    for method, reference, variant, fn in jobs:
        key = f"{method}_{reference}_{variant}"
        try:
            pred = fn().rename(key)
            predictions[key] = pred
            scores.append(_score(method, reference, variant, pred, obs))
            print(f"ok {key}")
        except Exception as exc:
            scores.append(Score(method, reference, variant, "failed", 0, error=f"{type(exc).__name__}: {exc}"))
            print(f"failed {key}: {type(exc).__name__}: {exc}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    xr.Dataset(predictions).to_netcdf(out_nc)
    payload = {
        "preset": args.preset,
        **source,
        "prediction_file": str(out_nc),
        "scores": [asdict(s) for s in scores],
        "notes": [
            "numpy/deepscale_sorted mirrors DeepScale's empirical QM mapping convention",
            "numpy/direct_midpoint_linear uses numpy nanquantile at midpoint nodes and is not the exact DeepScale sorted-column path",
            "numpy/factor_xsdba_nearest mirrors xsdba EQM's additive adjustment-factor convention",
            "xsdba adjust defaults to interp='nearest' and extrapolation='constant'",
        ],
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))
    _plot(scores, suffix)
    print(f"Wrote {out_json}")
    print(f"Wrote {out_nc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
