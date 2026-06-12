"""Real seasonal-forecast downscaling validation harness.

This is a validation artifact, separate from ``downscaling_validation.py``.
That harness uses CHIRPS coarsened into a pseudo-model. This one uses real
seasonal hindcasts from Rosetta as the predictor and CHIRPS as the target.

Default first track:

    C3S ECMWF SEAS5 monthly precipitation -> CHIRPS East Africa MAM/OND

The script writes JSON metrics, NetCDF predictions, and PNG figures under
``validation/results`` and ``validation/figures``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "validation" / "results"
FIGURES = ROOT / "validation" / "figures"
ROSETTA_SRC = Path.home() / "rosetta" / "src"
if ROSETTA_SRC.exists() and str(ROSETTA_SRC) not in sys.path:
    sys.path.insert(0, str(ROSETTA_SRC))

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "validation" / ".matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TARGET_MONTHS = {
    "MAM": [3, 4, 5],
    "OND": [10, 11, 12],
}

PRESETS = {
    "ecmwf_east_africa_mam_smoke_2001_2005": {
        "forecast_product": "c3s/ecmwf-monthly",
        "obs_product": "obs/chirps-v2-dekadal-rhiza",
        "init": "2001-02",
        "target": "MAM",
        "start_year": 2001,
        "end_year": 2005,
        "bbox": [-5.0, 0.0, 35.0, 40.0],
        "out": "validation/results/real_forecast_validation_ecmwf_east_africa_mam_smoke_2001_2005.json",
    },
    "ecmwf_east_africa_mam_2001_2016": {
        "forecast_product": "c3s/ecmwf-monthly",
        "obs_product": "obs/chirps-v2-dekadal-rhiza",
        "init": "2001-02",
        "target": "MAM",
        "start_year": 2001,
        "end_year": 2016,
        "bbox": [-5.0, 0.0, 35.0, 40.0],
        "out": "validation/results/real_forecast_validation_ecmwf_east_africa_mam_2001_2016.json",
    },
    "ecmwf_east_africa_ond_2001_2016": {
        "forecast_product": "c3s/ecmwf-monthly",
        "obs_product": "obs/chirps-v2-dekadal-rhiza",
        "init": "2001-09",
        "target": "OND",
        "start_year": 2001,
        "end_year": 2016,
        "bbox": [-5.0, 0.0, 35.0, 40.0],
        "out": "validation/results/real_forecast_validation_ecmwf_east_africa_ond_2001_2016.json",
    },
    "dwd_east_africa_mam_smoke_2001_2005": {
        "forecast_product": "c3s/dwd",
        "obs_product": "obs/chirps-v2-dekadal-rhiza",
        "init": "2001-02",
        "target": "MAM",
        "start_year": 2001,
        "end_year": 2005,
        "bbox": [-5.0, 0.0, 35.0, 40.0],
        "out": "validation/results/real_forecast_validation_dwd_east_africa_mam_smoke_2001_2005.json",
    },
    "dwd_east_africa_mam_2001_2016": {
        "forecast_product": "c3s/dwd",
        "obs_product": "obs/chirps-v2-dekadal-rhiza",
        "init": "2001-02",
        "target": "MAM",
        "start_year": 2001,
        "end_year": 2016,
        "bbox": [-5.0, 0.0, 35.0, 40.0],
        "out": "validation/results/real_forecast_validation_dwd_east_africa_mam_2001_2016.json",
    },
    "dwd_east_africa_ond_2001_2016": {
        "forecast_product": "c3s/dwd",
        "obs_product": "obs/chirps-v2-dekadal-rhiza",
        "init": "2001-09",
        "target": "OND",
        "start_year": 2001,
        "end_year": 2016,
        "bbox": [-5.0, 0.0, 35.0, 40.0],
        "out": "validation/results/real_forecast_validation_dwd_east_africa_ond_2001_2016.json",
    },
}


@dataclass
class Score:
    method: str
    reference: str
    status: str
    n_years: int
    n_cells: int
    bias: float | None = None
    mae: float | None = None
    rmse: float | None = None
    corr: float | None = None
    error: str | None = None


def _finite_float(value) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def score_prediction(method: str, reference: str, pred: xr.DataArray, obs: xr.DataArray) -> Score:
    pred, obs = xr.align(pred, obs, join="inner")
    if {"year", "lat", "lon"}.issubset(pred.dims) and {"year", "lat", "lon"}.issubset(obs.dims):
        pred = pred.transpose("year", "lat", "lon")
        obs = obs.transpose("year", "lat", "lon")
    diff = pred - obs
    valid = np.isfinite(pred.values) & np.isfinite(obs.values)
    n_cells = int(valid.sum())
    if n_cells < 2:
        return Score(method, reference, "failed", int(pred.sizes.get("year", 0)), n_cells, error="too few finite cells")
    p = pred.values[valid].ravel()
    o = obs.values[valid].ravel()
    corr = np.corrcoef(p, o)[0, 1] if len(p) > 2 and np.std(p) > 0 and np.std(o) > 0 else np.nan
    return Score(
        method=method,
        reference=reference,
        status="ok",
        n_years=int(pred.sizes.get("year", 0)),
        n_cells=n_cells,
        bias=_finite_float(diff.mean(skipna=True)),
        mae=_finite_float(abs(diff).mean(skipna=True)),
        rmse=_finite_float(np.sqrt((diff**2).mean(skipna=True))),
        corr=_finite_float(corr),
    )


def fetch_chirps_season(product: str, years: tuple[int, int], bbox: list[float], target: str, cache: bool) -> xr.DataArray:
    import rosetta

    months = TARGET_MONTHS[target.upper()]
    ds = rosetta.fetch(
        product=product,
        variable="precip",
        hindcast=years,
        region=bbox,
        cache=cache,
        verbose=True,
        progress=False,
    )
    da = ds["precip"]
    if "time" not in da.dims:
        raise ValueError(f"expected CHIRPS with time dim, got {da.dims}")
    da = da.where(da.time.dt.month.isin(months), drop=True)
    out = da.groupby("time.year").mean("time", skipna=True).astype("float64")
    out.attrs["units"] = "mm/day"
    return out


def _seasonalize_forecast(da: xr.DataArray) -> xr.DataArray:
    """Convert Rosetta forecast output to year/member/lat/lon seasonal fields."""
    if "init_time" in da.dims:
        years = da["init_time"].dt.year.values.astype(int)
        da = da.assign_coords(init_time=years).rename({"init_time": "year"})
    drop_dims = [d for d in ("time",) if d in da.dims and da.sizes[d] == 1]
    if drop_dims:
        da = da.isel({d: 0 for d in drop_dims}, drop=True)
    if "lead_time" in da.dims:
        da = da.mean("lead_time", skipna=True)
    elif "time" in da.dims and "year" not in da.dims:
        da = da.groupby("time.year").mean("time", skipna=True)
    if "year" not in da.dims:
        raise ValueError(f"expected forecast with year dim after seasonalization, got {da.dims}")
    if "member" not in da.dims:
        da = da.expand_dims(member=[0])
    out = da.transpose("year", "member", "lat", "lon").astype("float64")
    out.attrs["units"] = da.attrs.get("units", "mm/day")
    return out


def fetch_real_hindcast(product: str, years: tuple[int, int], bbox: list[float], init: str, target: str, cache: bool) -> xr.DataArray:
    import rosetta

    ds = rosetta.fetch(
        product=product,
        variable="precip",
        init=init,
        target=target,
        hindcast=years,
        region=bbox,
        cache=cache,
        verbose=True,
        progress=False,
    )
    return _seasonalize_forecast(ds["precip"])


def _align_years(hindcast: xr.DataArray, obs: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    years = sorted(set(int(y) for y in hindcast.year.values) & set(int(y) for y in obs.year.values))
    if len(years) < 3:
        raise ValueError(f"need at least 3 overlapping years, got {years}")
    return hindcast.sel(year=years), obs.sel(year=years)


def run_deepscale_predictions(method_name: str, hindcast: xr.DataArray, obs: xr.DataArray) -> tuple[Score, xr.DataArray | None]:
    from deepscale.registry import get_method

    kwargs_by_method = {
        "cca": {"n_modes": 2, "x_eof_modes": 3, "y_eof_modes": 3},
        "rank-analog": {"closing_size": 2, "gaussian_sigma": 0.75},
    }
    years = list(obs.year.values)
    preds = []
    truth = []
    try:
        cls = get_method(method_name)
        for year in years:
            train_years = [y for y in years if y != year]
            model = cls(**kwargs_by_method.get(method_name, {}))
            model.fit(hindcast.sel(year=train_years), obs.sel(year=train_years))
            fcst = hindcast.sel(year=[year]).isel(year=0, drop=True)
            pred = model.predict(fcst).mean("member").expand_dims(year=[year])
            preds.append(pred)
            truth.append(obs.sel(year=[year]))
        pred_all = xr.concat(preds, "year").rename(f"deepscale_{method_name.replace('-', '_')}")
        return score_prediction(method_name, "deepscale", pred_all, xr.concat(truth, "year")), pred_all
    except Exception as exc:
        return Score(method_name, "deepscale", "failed", len(years), 0, error=f"{type(exc).__name__}: {exc}"), None


def run_formula_delta_reference(hindcast: xr.DataArray, obs: xr.DataArray) -> tuple[Score, xr.DataArray | None]:
    preds = []
    truth = []
    years = list(obs.year.values)
    try:
        for year in years:
            train_years = [y for y in years if y != year]
            h_train = hindcast.sel(year=train_years)
            o_train = obs.sel(year=train_years)
            h_clim = h_train.mean(["year", "member"])
            o_clim = o_train.mean("year")
            sim = hindcast.sel(year=[year]).isel(year=0, drop=True).mean("member")
            anomaly = sim - h_clim
            pred = anomaly.interp(lat=obs.lat, lon=obs.lon, method="linear") + o_clim
            preds.append(pred.expand_dims(year=[year]))
            truth.append(obs.sel(year=[year]))
        pred_all = xr.concat(preds, "year").rename("formula_delta")
        return score_prediction("delta", "formula", pred_all, xr.concat(truth, "year")), pred_all
    except Exception as exc:
        return Score("delta", "formula", "failed", len(years), 0, error=f"{type(exc).__name__}: {exc}"), None


def _year_to_time(da: xr.DataArray) -> xr.DataArray:
    years = da["year"].values.astype(int)
    out = da.rename({"year": "time"})
    out = out.assign_coords(time=np.array([f"{y}-07-01" for y in years], dtype="datetime64[D]"))
    out.attrs["units"] = da.attrs.get("units", "mm/day")
    return out


def _coarse_reference_inputs(hindcast: xr.DataArray, obs: xr.DataArray, year):
    train_years = [y for y in obs.year.values if y != year]
    hist = hindcast.sel(year=train_years).mean("member")
    ref = obs.sel(year=train_years).interp(lat=hist.lat, lon=hist.lon, method="linear")
    sim = hindcast.sel(year=[year]).mean("member")
    for da in (hist, ref, sim):
        da.attrs["units"] = "mm/day"
    return hist, ref, sim


def run_xsdba_reference(kind: str, hindcast: xr.DataArray, obs: xr.DataArray) -> tuple[Score, xr.DataArray | None]:
    try:
        from xsdba import adjustment as adj
    except Exception as exc:
        return Score(kind, "xsdba", "skipped", int(obs.sizes["year"]), 0, error=f"xsdba unavailable: {exc}"), None

    cls_by_kind = {
        "qm": adj.EmpiricalQuantileMapping,
        "dqm": adj.DetrendedQuantileMapping,
    }
    preds = []
    truth = []
    try:
        for year in obs.year.values:
            hist, ref, sim = _coarse_reference_inputs(hindcast, obs, year)
            trained = cls_by_kind[kind].train(
                _year_to_time(ref),
                _year_to_time(hist),
                group="time",
                nquantiles=min(8, hist.sizes["year"]),
            )
            corrected = trained.adjust(_year_to_time(sim))
            fine = corrected.isel(time=0, drop=True).interp(lat=obs.lat, lon=obs.lon, method="linear")
            preds.append(fine.expand_dims(year=[year]))
            truth.append(obs.sel(year=[year]))
        pred_all = xr.concat(preds, "year").rename(f"xsdba_{kind}")
        return score_prediction(kind, "xsdba", pred_all, xr.concat(truth, "year")), pred_all
    except Exception as exc:
        return Score(kind, "xsdba", "failed", int(obs.sizes["year"]), 0, error=f"{type(exc).__name__}: {exc}"), None


def _try_cartopy():
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        return ccrs, cfeature
    except Exception:
        return None, None


def _plot_field(ax, da: xr.DataArray, *, title: str, cmap: str, vmin=None, vmax=None):
    ccrs, cfeature = _try_cartopy()
    lon = da["lon"].values
    lat = da["lat"].values
    if ccrs:
        mesh = ax.pcolormesh(lon, lat, da.values, transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax)
        ax.coastlines(resolution="50m", linewidth=0.7)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4)
        ax.set_extent([float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())], crs=ccrs.PlateCarree())
        gl = ax.gridlines(draw_labels=True, linewidth=0.2, color="0.5", alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
    else:
        mesh = ax.pcolormesh(lon, lat, da.values, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    ax.set_title(title, fontsize=10)
    return mesh


def plot_metric_bars(payload: dict, suffix: str) -> None:
    rows = [r for r in payload["scores"] if r["status"] == "ok"]
    rows = sorted(rows, key=lambda r: (r["rmse"], -r["corr"]))
    labels = [f"{r['method']}\n{r['reference']}" for r in rows]
    rmse = [r["rmse"] for r in rows]
    corr = [r["corr"] for r in rows]
    colors = ["#2f6f9f" if r["reference"] == "deepscale" else "#b45f06" for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].bar(labels, rmse, color=colors)
    axes[0].set_title("Real ECMWF -> CHIRPS: RMSE")
    axes[0].set_ylabel("RMSE (mm/day)")
    axes[0].tick_params(axis="x", rotation=45, labelsize=8)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(labels, corr, color=colors)
    axes[1].set_title("Real ECMWF -> CHIRPS: Correlation")
    axes[1].set_ylabel("Pearson correlation")
    axes[1].set_ylim(max(0.0, min(corr) - 0.05), 1.0)
    axes[1].tick_params(axis="x", rotation=45, labelsize=8)
    axes[1].grid(axis="y", alpha=0.25)
    path = FIGURES / f"real_forecast_metric_bars_{suffix}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {path}")


def plot_latest_maps(obs: xr.DataArray, predictions: dict[str, xr.DataArray], suffix: str) -> None:
    ok_preds = list(predictions.items())[:5]
    if not ok_preds:
        return
    year = int(obs.year.values[-1])
    obs_y = obs.sel(year=year)
    fields = [obs_y] + [da.sel(year=year) for _, da in ok_preds]
    vmin = float(xr.concat(fields, "field").quantile(0.02))
    vmax = float(xr.concat(fields, "field").quantile(0.98))
    diffs = [abs(da.sel(year=year) - obs_y).max(skipna=True) for _, da in ok_preds]
    diff_max = float(max(diffs)) if diffs else 1.0
    ccrs, _ = _try_cartopy()
    proj = {"projection": ccrs.PlateCarree()} if ccrs else {}
    ncols = len(ok_preds) + 1
    fig, axes = plt.subplots(2, ncols, figsize=(3.2 * ncols, 6.5), subplot_kw=proj, constrained_layout=True)
    if ncols == 1:
        axes = np.asarray(axes).reshape(2, 1)
    m = _plot_field(axes[0, 0], obs_y, title=f"CHIRPS ({year})", cmap="YlGnBu", vmin=vmin, vmax=vmax)
    axes[1, 0].axis("off")
    for col, (name, da) in enumerate(ok_preds, start=1):
        pred_y = da.sel(year=year)
        _plot_field(axes[0, col], pred_y, title=name, cmap="YlGnBu", vmin=vmin, vmax=vmax)
        dm = _plot_field(axes[1, col], pred_y - obs_y, title=f"{name} - CHIRPS", cmap="RdBu_r", vmin=-diff_max, vmax=diff_max)
    fig.colorbar(m, ax=axes[0, :], shrink=0.8, label="Precipitation (mm/day)")
    fig.colorbar(dm, ax=axes[1, 1:], shrink=0.8, label="Difference (mm/day)")
    path = FIGURES / f"real_forecast_maps_latest_year_{suffix}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {path}")


def _suffix_from_preset(args, preset_name: str) -> str:
    if preset_name != "custom":
        return preset_name
    target = args.target.lower()
    return f"{args.forecast_product.replace('/', '_')}_{target}_{args.start_year}_{args.end_year}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=sorted(PRESETS) + ["custom"], default="ecmwf_east_africa_mam_smoke_2001_2005")
    parser.add_argument("--forecast-product")
    parser.add_argument("--obs-product")
    parser.add_argument("--init")
    parser.add_argument("--target", choices=sorted(TARGET_MONTHS))
    parser.add_argument("--start-year", type=int)
    parser.add_argument("--end-year", type=int)
    parser.add_argument("--bbox", nargs=4, type=float)
    parser.add_argument("--methods", nargs="+", default=["bcsd", "cca", "climatology", "delta", "dqm", "qm", "rank-analog"])
    parser.add_argument("--skip-xsdba", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--out")
    parser.add_argument("--predictions-out")
    args = parser.parse_args()

    preset = PRESETS.get(args.preset, {})
    forecast_product = args.forecast_product or preset.get("forecast_product")
    obs_product = args.obs_product or preset.get("obs_product", "obs/chirps-v2-dekadal-rhiza")
    init = args.init or preset.get("init")
    target = args.target or preset.get("target")
    start_year = args.start_year if args.start_year is not None else preset.get("start_year")
    end_year = args.end_year if args.end_year is not None else preset.get("end_year")
    bbox = args.bbox or preset.get("bbox")
    if not all([forecast_product, obs_product, init, target, start_year, end_year, bbox]):
        raise SystemExit("custom runs require --forecast-product, --obs-product, --init, --target, --start-year, --end-year, and --bbox")

    suffix = _suffix_from_preset(args, args.preset)
    out_json = Path(args.out or preset.get("out") or RESULTS / f"real_forecast_validation_{suffix}.json")
    out_nc = Path(args.predictions_out or RESULTS / f"real_forecast_predictions_{suffix}.nc")

    hindcast = fetch_real_hindcast(forecast_product, (start_year, end_year), bbox, init, target, cache=not args.no_cache)
    obs = fetch_chirps_season(obs_product, (start_year, end_year), bbox, target, cache=not args.no_cache)
    hindcast, obs = _align_years(hindcast, obs)

    scores: list[Score] = []
    predictions: dict[str, xr.DataArray] = {"obs": obs}
    for method in args.methods:
        score, pred = run_deepscale_predictions(method, hindcast, obs)
        scores.append(score)
        if pred is not None:
            predictions[pred.name] = pred
        print(f"{score.status} {method} deepscale")

    score, pred = run_formula_delta_reference(hindcast, obs)
    scores.append(score)
    if pred is not None:
        predictions[pred.name] = pred
    print(f"{score.status} delta formula")

    if not args.skip_xsdba:
        for kind in ("qm", "dqm"):
            score, pred = run_xsdba_reference(kind, hindcast, obs)
            scores.append(score)
            if pred is not None:
                predictions[pred.name] = pred
            print(f"{score.status} {kind} xsdba")

    payload = {
        "dataset": {
            "preset": args.preset,
            "forecast_product": forecast_product,
            "obs_product": obs_product,
            "init": init,
            "target": target,
            "target_months": TARGET_MONTHS[target],
            "requested_years": [start_year, end_year],
            "actual_years": [int(y) for y in obs.year.values],
            "bbox": bbox,
            "obs_shape": dict(obs.sizes),
            "hindcast_shape": dict(hindcast.sizes),
            "prediction_file": str(out_nc),
            "notes": [
                "Real forecast benchmark: predictor is not derived from CHIRPS",
                "Rosetta forecast lead_time is averaged to one seasonal field per year/member before downscaling",
                "Validation is leave-one-year-out over overlapping hindcast and CHIRPS years",
            ],
        },
        "scores": [asdict(s) for s in scores],
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    xr.Dataset(predictions).to_netcdf(out_nc)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))
    plot_metric_bars(payload, suffix)
    plot_latest_maps(obs, {k: v for k, v in predictions.items() if k != "obs"}, suffix)
    print(f"Wrote {out_json}")
    print(f"Wrote {out_nc}")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
