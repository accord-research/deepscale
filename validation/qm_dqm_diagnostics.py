"""Diagnostics for DeepScale QM/DQM vs xsdba on the CHIRPS high/low fixture."""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
import argparse

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
    pred, obs = xr.align(pred, obs, join="inner")
    pred = pred.transpose("year", "lat", "lon")
    obs = obs.transpose("year", "lat", "lon")
    valid = np.isfinite(pred.values) & np.isfinite(obs.values)
    if int(valid.sum()) < 2:
        return Score(method, reference, variant, "failed", int(valid.sum()), error="too few finite cells")
    p = pred.values[valid].ravel()
    o = obs.values[valid].ravel()
    d = p - o
    corr = np.corrcoef(p, o)[0, 1] if np.std(p) > 0 and np.std(o) > 0 else np.nan
    return Score(
        method=method,
        reference=reference,
        variant=variant,
        status="ok",
        n_cells=int(valid.sum()),
        bias=float(np.mean(d)),
        mae=float(np.mean(np.abs(d))),
        rmse=float(np.sqrt(np.mean(d**2))),
        corr=float(corr),
    )


def _year_to_time(da: xr.DataArray) -> xr.DataArray:
    years = da["year"].values.astype(int)
    out = da.rename({"year": "time"})
    out = out.assign_coords(time=np.array([f"{y}-07-01" for y in years], dtype="datetime64[D]"))
    out.attrs["units"] = da.attrs.get("units", "mm/day")
    return out


def _make_dense_hindcast(obs: xr.DataArray, factor: int = 3, members: int = 3) -> xr.DataArray:
    coarse = obs.coarsen(lat=factor, lon=factor, boundary="trim").mean(skipna=True)
    member_fields = []
    for i in range(members):
        scale = 1.0 + (i - (members - 1) / 2.0) * 0.01
        member_fields.append(coarse * scale)
    hindcast = xr.concat(member_fields, dim=xr.DataArray(np.arange(members), dims="member", name="member"))
    return hindcast.transpose("year", "member", "lat", "lon")


def _run_deepscale(method: str, variant: str, hindcast: xr.DataArray, obs: xr.DataArray) -> xr.DataArray:
    from deepscale.registry import get_method

    cls = get_method(method)
    preds = []
    years = list(obs.year.values)
    for year in years:
        train_years = [y for y in years if y != year]
        model = cls(variant=variant) if method in {"qm", "dqm"} else cls()
        model.fit(hindcast.sel(year=train_years), obs.sel(year=train_years))
        pred = model.predict(hindcast.sel(year=[year]).isel(year=0, drop=True)).mean("member")
        preds.append(pred.expand_dims(year=[year]))
    return xr.concat(preds, "year")


def _run_deepscale_single_member(method: str, hindcast: xr.DataArray, obs: xr.DataArray) -> xr.DataArray:
    """Run DeepScale with a single mean member to test member-averaging effects."""
    single = hindcast.mean("member").expand_dims(member=[0]).transpose("year", "member", "lat", "lon")
    return _run_deepscale(method, "empirical", single, obs)


def _run_xsdba(kind: str, variant: str, hindcast: xr.DataArray, obs: xr.DataArray) -> xr.DataArray:
    from xsdba import adjustment as adj

    cls = adj.EmpiricalQuantileMapping if kind == "qm" else adj.DetrendedQuantileMapping
    n_train = obs.sizes["year"] - 1
    if variant == "nq8":
        kwargs = {"nquantiles": min(8, n_train), "group": "time", "kind": "+"}
    elif variant == "nq_train":
        kwargs = {"nquantiles": n_train, "group": "time", "kind": "+"}
    elif variant == "default":
        kwargs = {"group": "time", "kind": "+"}
    else:
        raise ValueError(f"unknown xsdba variant {variant!r}")

    preds = []
    years = list(obs.year.values)
    for year in years:
        train_years = [y for y in years if y != year]
        hist = hindcast.sel(year=train_years).mean("member")
        ref = obs.sel(year=train_years).interp(lat=hist.lat, lon=hist.lon, method="linear")
        sim = hindcast.sel(year=[year]).mean("member")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trained = cls.train(_year_to_time(ref), _year_to_time(hist), **kwargs)
            corrected = trained.adjust(_year_to_time(sim))
        fine = corrected.isel(time=0, drop=True).interp(lat=obs.lat, lon=obs.lon, method="linear")
        preds.append(fine.expand_dims(year=[year]))
    return xr.concat(preds, "year")


def _save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {path}")


def _plot_metric_bars(scores: list[Score], suffix: str) -> None:
    rows = sorted([s for s in scores if s.status == "ok"], key=lambda s: s.rmse or 999)
    labels = [
        f"{'DS' if s.reference == 'deepscale' else 'xsdba'} {s.method.upper()}\n{s.variant.replace('_', ' ')}"
        for s in rows
    ]
    colors = ["#2f6f9f" if s.reference == "deepscale" else "#b45f06" for s in rows]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), constrained_layout=True)
    axes[0].bar(labels, [s.rmse for s in rows], color=colors)
    axes[0].set_title("QM/DQM Diagnostics: RMSE")
    axes[0].set_ylabel("RMSE (mm/day)")
    axes[0].tick_params(axis="x", rotation=35, labelsize=8)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(labels, [s.corr for s in rows], color=colors)
    axes[1].set_title("QM/DQM Diagnostics: Correlation")
    axes[1].set_ylabel("Pearson correlation")
    min_corr = min(s.corr for s in rows if s.corr is not None)
    axes[1].set_ylim(max(0.0, min_corr - 0.02), 1.0)
    axes[1].tick_params(axis="x", rotation=35, labelsize=8)
    axes[1].grid(axis="y", alpha=0.25)
    _save(fig, f"qm_dqm_metric_bars_{suffix}.png")


def _plot_error_maps(preds: dict[str, xr.DataArray], obs: xr.DataArray, suffix: str) -> None:
    selected = [
        ("DeepScale QM", preds["qm_deepscale_empirical"]),
        ("xsdba EQM", preds["qm_xsdba_nq8"]),
        ("DeepScale DQM", preds["dqm_deepscale_empirical"]),
        ("xsdba DQM", preds["dqm_xsdba_nq8"]),
    ]
    mae_maps = [(title, abs(pred - obs).mean("year", skipna=True)) for title, pred in selected]
    vmax = float(xr.concat([m for _, m in mae_maps], "field").quantile(0.98))
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
    for ax, (title, da) in zip(axes.ravel(), mae_maps):
        da = da.transpose("lat", "lon")
        mesh = ax.pcolormesh(da.lon, da.lat, da, cmap="magma", vmin=0.0, vmax=vmax, shading="auto")
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    fig.colorbar(mesh, ax=axes.ravel().tolist(), shrink=0.8, label="Mean absolute error (mm/day)")
    fig.suptitle("QM/DQM Mean Absolute Error Maps")
    _save(fig, f"qm_dqm_mae_maps_{suffix}.png")


def _plot_reference_maps(
    preds: dict[str, xr.DataArray],
    obs: xr.DataArray,
    suffix: str,
    method: str,
    source: str,
) -> None:
    if source not in {"deepscale", "xsdba"}:
        raise ValueError(f"unsupported source {source!r}")
    year = int(obs.year.values[-1])
    if "ethiopia" in suffix:
        label = "Ethiopia"
    elif "texas" in suffix:
        label = "Texas"
    else:
        label = f"Example held-out year {year}"

    obs_y = obs.sel(year=year)
    deep_y = preds[f"{method}_deepscale_empirical"].sel(year=year).transpose("lat", "lon")
    xsdba_y = preds[f"{method}_xsdba_nq8"].sel(year=year).transpose("lat", "lon")
    pred_y = deep_y if source == "deepscale" else xsdba_y
    obs_y = obs_y.transpose("lat", "lon")
    valid_domain = np.isfinite(deep_y) & np.isfinite(xsdba_y)
    valid_lat = valid_domain.any("lon")
    valid_lon = valid_domain.any("lat")
    obs_y = obs_y.sel(lat=obs_y.lat[valid_lat], lon=obs_y.lon[valid_lon])
    deep_y = deep_y.sel(lat=deep_y.lat[valid_lat], lon=deep_y.lon[valid_lon])
    xsdba_y = xsdba_y.sel(lat=xsdba_y.lat[valid_lat], lon=xsdba_y.lon[valid_lon])
    pred_y = pred_y.sel(lat=pred_y.lat[valid_lat], lon=pred_y.lon[valid_lon])
    fields = [obs_y, pred_y]
    vmin = float(xr.concat(fields, "field", coords="minimal", compat="override").quantile(0.02))
    vmax = float(xr.concat(fields, "field", coords="minimal", compat="override").quantile(0.98))
    diff = pred_y - obs_y
    diff_max = float(abs(diff).quantile(0.98))
    diff_max = max(diff_max, 1e-6)

    fig = plt.figure(figsize=(14.2, 4.6))
    gs = fig.add_gridspec(1, 6, width_ratios=[1, 1, 0.055, 0.22, 1, 0.055], wspace=0.38)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 4])]
    precip_cax = fig.add_subplot(gs[0, 2])
    diff_cax = fig.add_subplot(gs[0, 5])

    method_label = method.upper()
    source_label = f"DeepScale {method_label}" if source == "deepscale" else f"xsdba {method_label}"
    for ax, da, title in zip(axes[:2], fields, [f"CHIRPS observed ({year})", source_label]):
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
    filename_source = "reference" if source == "deepscale" else "xsdba"
    _save(fig, f"{method}_{filename_source}_maps_{suffix}.png")


def _plot_qm_reference_maps(preds: dict[str, xr.DataArray], obs: xr.DataArray, suffix: str) -> None:
    _plot_reference_maps(preds, obs, suffix, "qm", "deepscale")


def _plot_dqm_reference_maps(preds: dict[str, xr.DataArray], obs: xr.DataArray, suffix: str) -> None:
    _plot_reference_maps(preds, obs, suffix, "dqm", "deepscale")


def _plot_qm_xsdba_maps(preds: dict[str, xr.DataArray], obs: xr.DataArray, suffix: str) -> None:
    _plot_reference_maps(preds, obs, suffix, "qm", "xsdba")


def _plot_dqm_xsdba_maps(preds: dict[str, xr.DataArray], obs: xr.DataArray, suffix: str) -> None:
    _plot_reference_maps(preds, obs, suffix, "dqm", "xsdba")


def _plot_cdf_example(hindcast: xr.DataArray, obs: xr.DataArray, preds: dict[str, xr.DataArray], suffix: str) -> None:
    lat = float(obs.lat.sel(lat=float(obs.lat.mean()), method="nearest"))
    lon = float(obs.lon.sel(lon=float(obs.lon.mean()), method="nearest"))
    coarse = hindcast.mean("member")
    c_lat = float(coarse.lat.sel(lat=lat, method="nearest"))
    c_lon = float(coarse.lon.sel(lon=lon, method="nearest"))
    year = int(obs.year.values[-1])
    train = [y for y in obs.year.values if y != year]
    hist = coarse.sel(year=train, lat=c_lat, lon=c_lon)
    ref = obs.sel(year=train).interp(lat=[c_lat], lon=[c_lon]).squeeze()
    sim = coarse.sel(year=year, lat=c_lat, lon=c_lon)

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    q = (np.arange(len(train)) + 0.5) / len(train)
    ax.plot(np.sort(hist.values), q, marker="o", label="coarse hist CDF")
    ax.plot(np.sort(ref.values), q, marker="o", label="obs-on-coarse ref CDF")
    ax.axvline(float(sim), color="black", linestyle="--", label=f"held-out coarse sim {year}")
    for key, label, color in [
        ("qm_deepscale_empirical", "DeepScale QM prediction", "#2f6f9f"),
        ("qm_xsdba_nq8", "xsdba EQM prediction", "#b45f06"),
        ("dqm_deepscale_empirical", "DeepScale DQM prediction", "#5b9bd5"),
        ("dqm_xsdba_nq8", "xsdba DQM prediction", "#ed7d31"),
    ]:
        val = preds[key].sel(year=year, lat=lat, lon=lon, method="nearest")
        ax.axvline(float(val), color=color, alpha=0.85, label=label)
    ax.set_title(f"Example CDF Diagnostic near lat={lat:.2f}, lon={lon:.2f}")
    ax.set_xlabel("Precipitation (mm/day)")
    ax.set_ylabel("Empirical probability")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    _save(fig, f"qm_dqm_cdf_example_{suffix}.png")


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
    out_json = Path(args.out_json) if args.out_json else RESULTS / f"qm_dqm_diagnostics_{suffix}.json"
    out_nc = Path(args.out_nc) if args.out_nc else RESULTS / f"qm_dqm_diagnostics_predictions_{suffix}.nc"

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

    predictions: dict[str, xr.DataArray] = {}
    scores: list[Score] = []

    jobs = [
        ("qm", "deepscale", "empirical", lambda: _run_deepscale("qm", "empirical", hindcast, obs)),
        ("qm", "deepscale", "parametric", lambda: _run_deepscale("qm", "parametric", hindcast, obs)),
        ("qm", "deepscale", "single_member", lambda: _run_deepscale_single_member("qm", hindcast, obs)),
        ("dqm", "deepscale", "empirical", lambda: _run_deepscale("dqm", "empirical", hindcast, obs)),
        ("dqm", "deepscale", "parametric", lambda: _run_deepscale("dqm", "parametric", hindcast, obs)),
        ("qm", "xsdba", "nq8", lambda: _run_xsdba("qm", "nq8", hindcast, obs)),
        ("qm", "xsdba", "nq_train", lambda: _run_xsdba("qm", "nq_train", hindcast, obs)),
        ("qm", "xsdba", "default", lambda: _run_xsdba("qm", "default", hindcast, obs)),
        ("dqm", "xsdba", "nq8", lambda: _run_xsdba("dqm", "nq8", hindcast, obs)),
        ("dqm", "xsdba", "nq_train", lambda: _run_xsdba("dqm", "nq_train", hindcast, obs)),
        ("dqm", "xsdba", "default", lambda: _run_xsdba("dqm", "default", hindcast, obs)),
    ]

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
            "single_member tests whether DeepScale member averaging explains the gap",
            "xsdba nq8 matches the first report's nquantiles setting",
            "xsdba default uses package defaults for quantile count",
        ],
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))
    _plot_metric_bars(scores, suffix)
    _plot_error_maps(predictions, obs, suffix)
    _plot_qm_reference_maps(predictions, obs, suffix)
    _plot_qm_xsdba_maps(predictions, obs, suffix)
    _plot_dqm_reference_maps(predictions, obs, suffix)
    _plot_dqm_xsdba_maps(predictions, obs, suffix)
    _plot_cdf_example(hindcast, obs, predictions, suffix)
    print(f"Wrote {out_json}")
    print(f"Wrote {out_nc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
