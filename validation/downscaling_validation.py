"""Controlled CHIRPS high/low downscaling validation harness.

This script is intentionally outside src/: it is a validation artifact, not a
production feature. It builds a controlled high/low CHIRPS benchmark:

1. Fetch high-resolution CHIRPS through Rosetta.
2. Aggregate it to a coarse grid to create a pseudo-model predictor.
3. Run leave-one-year-out downscaling methods against held-out high-res truth.
4. Optionally run reference-library analogues when installed.

Run from the DeepScale repo root, for example:

    PYTHONPATH=/Users/david/rosetta/src uv run \
      --extra validation \
      python validation/downscaling_validation.py

The default "report" method suite excludes rank-analog so it is not presented
as a headline validation method. Use ``--method-suite nightly`` in scheduled or
regression runs when rank-analog coverage is desired.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import xarray as xr


ROSETTA_SRC = Path.home() / "rosetta" / "src"
if ROSETTA_SRC.exists() and str(ROSETTA_SRC) not in sys.path:
    sys.path.insert(0, str(ROSETTA_SRC))

REPORT_METHODS = ["bcsd", "cca", "climatology", "delta", "dqm", "qm"]
NIGHTLY_METHODS = [*REPORT_METHODS, "rank-analog"]


PRESETS = {
    "texas": {
        "product": "obs/chirps-v3-monthly",
        "start_year": 1991,
        "end_year": 2020,
        "bbox": [30.0, 35.0, -100.0, -95.0],
        "months": None,
        "coarsen_factor": 10,  # native 0.05 degree CHIRPS -> ~0.5 degree pseudo-model
        "out": "validation/results/downscaling_validation_chirps_monthly_texas_1991_2020.json",
    },
    # Backward-compatible alias for old command lines. The default Texas
    # fixture now uses native UCSB CHIRPS directly.
    "native_texas_2001_2020": {
        "product": "obs/chirps-v3-monthly",
        "start_year": 2001,
        "end_year": 2020,
        "bbox": [30.0, 35.0, -100.0, -95.0],
        "months": None,
        "coarsen_factor": 10,  # native 0.05 degree CHIRPS -> ~0.5 degree pseudo-model
        "out": "validation/results/downscaling_validation_chirps_native_texas_2001_2020.json",
    },
    "long_texas_2001_2020": {
        "product": "obs/chirps-v3-monthly",
        "start_year": 1991,
        "end_year": 2020,
        "bbox": [30.0, 35.0, -100.0, -95.0],
        "months": None,
        "coarsen_factor": 10,
        "out": "validation/results/downscaling_validation_chirps_monthly_long_texas_1991_2020.json",
    },
    "long_texas_1991_2020": {
        "product": "obs/chirps-v3-monthly",
        "start_year": 1991,
        "end_year": 2020,
        "bbox": [30.0, 35.0, -100.0, -95.0],
        "months": None,
        "coarsen_factor": 10,
        "out": "validation/results/downscaling_validation_chirps_monthly_long_texas_1991_2020.json",
    },
    "ethiopia_fma_1991_2020": {
        "product": "obs/chirps-v3-monthly",
        "start_year": 1991,
        "end_year": 2020,
        "bbox": [3.0, 15.0, 33.0, 48.0],
        "months": [2, 3, 4],
        "coarsen_factor": 10,
        "out": "validation/results/downscaling_validation_chirps_monthly_ethiopia_fma_1991_2020.json",
    },
    # Backward-compatible alias for old command lines.
    "ethiopia_fma_2001_2020": {
        "product": "obs/chirps-v3-monthly",
        "start_year": 1991,
        "end_year": 2020,
        "bbox": [3.0, 15.0, 33.0, 48.0],
        "months": [2, 3, 4],
        "coarsen_factor": 10,
        "out": "validation/results/downscaling_validation_chirps_monthly_ethiopia_fma_1991_2020.json",
    },
    "ethiopia_jjas_2001_2020": {
        "product": "obs/chirps-v3-monthly",
        "start_year": 2001,
        "end_year": 2020,
        "bbox": [3.0, 15.0, 33.0, 48.0],
        "months": [6, 7, 8, 9],
        "coarsen_factor": 10,
        "out": "validation/results/downscaling_validation_chirps_monthly_ethiopia_jjas_2001_2020.json",
    },
    "ethiopia_ond_2001_2020": {
        "product": "obs/chirps-v3-monthly",
        "start_year": 2001,
        "end_year": 2020,
        "bbox": [3.0, 15.0, 33.0, 48.0],
        "months": [10, 11, 12],
        "coarsen_factor": 10,
        "out": "validation/results/downscaling_validation_chirps_monthly_ethiopia_ond_2001_2020.json",
    },
    "east_africa_2001_2020": {
        "product": "obs/chirps-v3-monthly",
        "start_year": 2001,
        "end_year": 2020,
        "bbox": [-5.0, 0.0, 35.0, 40.0],
        "months": None,
        "coarsen_factor": 10,
        "out": "validation/results/downscaling_validation_chirps_monthly_east_africa_2001_2020.json",
    },
    "east_africa_mam_2001_2020": {
        "product": "obs/chirps-v3-monthly",
        "start_year": 2001,
        "end_year": 2020,
        "bbox": [-5.0, 0.0, 35.0, 40.0],
        "months": [3, 4, 5],
        "coarsen_factor": 10,
        "out": "validation/results/downscaling_validation_chirps_monthly_east_africa_mam_2001_2020.json",
    },
    "east_africa_ond_2001_2020": {
        "product": "obs/chirps-v3-monthly",
        "start_year": 2001,
        "end_year": 2020,
        "bbox": [-5.0, 0.0, 35.0, 40.0],
        "months": [10, 11, 12],
        "coarsen_factor": 10,
        "out": "validation/results/downscaling_validation_chirps_monthly_east_africa_ond_2001_2020.json",
    },
    # Historical presets used for the first validation report. Kept so the
    # existing dekadal artifacts remain reproducible while new primary runs use
    # Rosetta's default CHIRPS product above.
    "dekadal_long_texas_1991_2020": {
        "product": "obs/chirps-v2-dekadal-rhiza",
        "start_year": 1991,
        "end_year": 2020,
        "bbox": [30.0, 35.0, -100.0, -95.0],
        "months": None,
        "out": "validation/results/downscaling_validation_chirps_long_texas_1991_2020.json",
    },
    "dekadal_east_africa_2001_2020": {
        "product": "obs/chirps-v2-dekadal-rhiza",
        "start_year": 2001,
        "end_year": 2020,
        "bbox": [-5.0, 0.0, 35.0, 40.0],
        "months": None,
        "out": "validation/results/downscaling_validation_chirps_east_africa_2001_2020.json",
    },
    "dekadal_east_africa_mam_2001_2020": {
        "product": "obs/chirps-v2-dekadal-rhiza",
        "start_year": 2001,
        "end_year": 2020,
        "bbox": [-5.0, 0.0, 35.0, 40.0],
        "months": [3, 4, 5],
        "out": "validation/results/downscaling_validation_chirps_east_africa_mam_2001_2020.json",
    },
    "dekadal_east_africa_ond_2001_2020": {
        "product": "obs/chirps-v2-dekadal-rhiza",
        "start_year": 2001,
        "end_year": 2020,
        "bbox": [-5.0, 0.0, 35.0, 40.0],
        "months": [10, 11, 12],
        "out": "validation/results/downscaling_validation_chirps_east_africa_ond_2001_2020.json",
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
    pred_summary: dict | None = None
    obs_summary: dict | None = None
    distribution_bias: dict | None = None


@dataclass
class SignalDiagnostic:
    method: str
    reference: str
    status: str
    coarse_corr: float | None = None
    downscaled_corr: float | None = None
    corr_delta: float | None = None
    coarse_n_cells: int = 0
    downscaled_n_cells: int = 0
    error: str | None = None


def _finite_float(value) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _summary_stats(da: xr.DataArray) -> dict[str, float | None]:
    q = da.quantile([0.5, 0.7, 0.9, 0.95], skipna=True)
    return {
        "min": _finite_float(da.min(skipna=True)),
        "mean": _finite_float(da.mean(skipna=True)),
        "max": _finite_float(da.max(skipna=True)),
        "std": _finite_float(da.std(skipna=True)),
        "p50": _finite_float(q.sel(quantile=0.5)),
        "p70": _finite_float(q.sel(quantile=0.7)),
        "p90": _finite_float(q.sel(quantile=0.9)),
        "p95": _finite_float(q.sel(quantile=0.95)),
    }


def _summary_bias(pred_summary: dict, obs_summary: dict) -> dict[str, float | None]:
    keys = ("min", "mean", "max", "std", "p50", "p70", "p90", "p95")
    return {
        key: (
            None
            if pred_summary.get(key) is None or obs_summary.get(key) is None
            else pred_summary[key] - obs_summary[key]
        )
        for key in keys
    }


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
    pred_masked = pred.where(np.isfinite(pred) & np.isfinite(obs))
    obs_masked = obs.where(np.isfinite(pred) & np.isfinite(obs))
    pred_summary = _summary_stats(pred_masked)
    obs_summary = _summary_stats(obs_masked)
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
        pred_summary=pred_summary,
        obs_summary=obs_summary,
        distribution_bias=_summary_bias(pred_summary, obs_summary),
    )


def fetch_chirps(product: str, years: tuple[int, int], bbox: list[float], cache: bool, months: list[int] | None = None) -> xr.DataArray:
    import rosetta

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
    if months:
        da = da.where(da.time.dt.month.isin(months), drop=True)
    annual = da.groupby("time.year").mean("time", skipna=True)
    annual = annual.rename({"year": "year"}).astype("float64")
    annual.attrs["units"] = "mm/day"
    return annual


def make_hindcast(obs: xr.DataArray, factor: int, members: int) -> xr.DataArray:
    coarse = obs.coarsen(lat=factor, lon=factor, boundary="trim").mean(skipna=True)
    member_fields = []
    for i in range(members):
        # Deterministic, tiny perturbation so member-handling code is exercised
        # without changing the signal being validated.
        scale = 1.0 + (i - (members - 1) / 2.0) * 0.01
        member_fields.append(coarse * scale)
    hindcast = xr.concat(member_fields, dim=xr.DataArray(np.arange(members), dims="member", name="member"))
    return hindcast.transpose("year", "member", "lat", "lon")


def _linear_baseline(hindcast: xr.DataArray, obs: xr.DataArray) -> xr.DataArray:
    pred = hindcast.mean("member").interp(lat=obs.lat, lon=obs.lon, method="linear")
    return pred.transpose("year", "lat", "lon").rename("bilinear")


def _corr_with_n(a: xr.DataArray, b: xr.DataArray) -> tuple[float | None, int]:
    a, b = xr.align(a, b, join="inner")
    valid = np.isfinite(a.values) & np.isfinite(b.values)
    n = int(valid.sum())
    if n < 3:
        return None, n
    av = a.values[valid].ravel()
    bv = b.values[valid].ravel()
    if np.std(av) == 0 or np.std(bv) == 0:
        return None, n
    return _finite_float(np.corrcoef(av, bv)[0, 1]), n


def signal_preservation(method: str, reference: str, pred: xr.DataArray, hindcast: xr.DataArray, obs: xr.DataArray) -> SignalDiagnostic:
    try:
        coarse = hindcast.mean("member")
        coarse_obs = obs.interp(lat=coarse.lat, lon=coarse.lon, method="linear")
        coarse_corr, coarse_n = _corr_with_n(coarse.transpose("year", "lat", "lon"), coarse_obs.transpose("year", "lat", "lon"))
        downscaled_corr, downscaled_n = _corr_with_n(pred.transpose("year", "lat", "lon"), obs.transpose("year", "lat", "lon"))
        return SignalDiagnostic(
            method=method,
            reference=reference,
            status="ok" if coarse_corr is not None and downscaled_corr is not None else "failed",
            coarse_corr=coarse_corr,
            downscaled_corr=downscaled_corr,
            corr_delta=None if coarse_corr is None or downscaled_corr is None else downscaled_corr - coarse_corr,
            coarse_n_cells=coarse_n,
            downscaled_n_cells=downscaled_n,
        )
    except Exception as exc:
        return SignalDiagnostic(method, reference, "failed", error=f"{type(exc).__name__}: {exc}")


def run_deepscale_prediction(method_name: str, hindcast: xr.DataArray, obs: xr.DataArray) -> tuple[Score, xr.DataArray | None]:
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
        pred_all = xr.concat(preds, "year").rename(method_name.replace("-", "_"))
        return score_prediction(method_name, "deepscale", pred_all, xr.concat(truth, "year")), pred_all
    except Exception as exc:
        return Score(method_name, "deepscale", "failed", len(years), 0, error=f"{type(exc).__name__}: {exc}"), None


def run_deepscale_method(method_name: str, hindcast: xr.DataArray, obs: xr.DataArray) -> Score:
    score, _ = run_deepscale_prediction(method_name, hindcast, obs)
    return score


def _coarse_reference_inputs(hindcast: xr.DataArray, obs: xr.DataArray, year):
    train_years = [y for y in obs.year.values if y != year]
    hist = hindcast.sel(year=train_years).mean("member")
    ref = obs.sel(year=train_years).interp(lat=hist.lat, lon=hist.lon, method="linear")
    sim = hindcast.sel(year=[year]).mean("member")
    for da in (hist, ref, sim):
        da.attrs["units"] = "mm/day"
    return train_years, hist, ref, sim


def _year_to_time(da: xr.DataArray) -> xr.DataArray:
    """xsdba expects a time axis; DeepScale's method contract uses year."""
    years = da["year"].values.astype(int)
    out = da.rename({"year": "time"})
    out = out.assign_coords(time=np.array([f"{y}-07-01" for y in years], dtype="datetime64[D]"))
    out.attrs["units"] = da.attrs.get("units", "mm/day")
    return out


def _as_single_time(da: xr.DataArray, date: str = "2001-01-01") -> xr.DataArray:
    out = da.expand_dims(time=[np.datetime64(date)])
    out.attrs["units"] = da.attrs.get("units", "mm/day")
    return out


def run_xsdba_reference(kind: str, hindcast: xr.DataArray, obs: xr.DataArray) -> Score:
    try:
        from xsdba import adjustment as adj
    except Exception as exc:
        return Score(kind, "xsdba", "skipped", int(obs.sizes["year"]), 0, error=f"xsdba unavailable: {exc}")

    cls_by_kind = {
        "qm": adj.EmpiricalQuantileMapping,
        "dqm": adj.DetrendedQuantileMapping,
    }
    cls = cls_by_kind[kind]
    preds = []
    truth = []
    try:
        for year in obs.year.values:
            _, hist, ref, sim = _coarse_reference_inputs(hindcast, obs, year)
            hist_t = _year_to_time(hist)
            ref_t = _year_to_time(ref)
            sim_t = _year_to_time(sim)
            trained = cls.train(ref_t, hist_t, group="time", nquantiles=min(8, hist.sizes["year"]))
            corrected = trained.adjust(sim_t)
            fine = corrected.isel(time=0, drop=True).interp(lat=obs.lat, lon=obs.lon, method="linear")
            preds.append(fine.expand_dims(year=[year]))
            truth.append(obs.sel(year=[year]))
        return score_prediction(kind, "xsdba", xr.concat(preds, "year"), xr.concat(truth, "year"))
    except Exception as exc:
        return Score(kind, "xsdba", "failed", int(obs.sizes["year"]), 0, error=f"{type(exc).__name__}: {exc}")


def run_xsdba_prediction(kind: str, hindcast: xr.DataArray, obs: xr.DataArray) -> tuple[Score, xr.DataArray | None]:
    try:
        from xsdba import adjustment as adj
    except Exception as exc:
        return Score(kind, "xsdba", "skipped", int(obs.sizes["year"]), 0, error=f"xsdba unavailable: {exc}"), None

    cls_by_kind = {
        "qm": adj.EmpiricalQuantileMapping,
        "dqm": adj.DetrendedQuantileMapping,
    }
    cls = cls_by_kind[kind]
    preds = []
    truth = []
    try:
        for year in obs.year.values:
            _, hist, ref, sim = _coarse_reference_inputs(hindcast, obs, year)
            hist_t = _year_to_time(hist)
            ref_t = _year_to_time(ref)
            sim_t = _year_to_time(sim)
            trained = cls.train(ref_t, hist_t, group="time", nquantiles=min(8, hist.sizes["year"]))
            corrected = trained.adjust(sim_t)
            fine = corrected.isel(time=0, drop=True).interp(lat=obs.lat, lon=obs.lon, method="linear")
            preds.append(fine.expand_dims(year=[year]))
            truth.append(obs.sel(year=[year]))
        pred_all = xr.concat(preds, "year").rename(f"xsdba_{kind}")
        return score_prediction(kind, "xsdba", pred_all, xr.concat(truth, "year")), pred_all
    except Exception as exc:
        return Score(kind, "xsdba", "failed", int(obs.sizes["year"]), 0, error=f"{type(exc).__name__}: {exc}"), None


def run_formula_delta_reference(hindcast: xr.DataArray, obs: xr.DataArray) -> Score:
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
        return score_prediction("delta", "formula", xr.concat(preds, "year"), xr.concat(truth, "year"))
    except Exception as exc:
        return Score("delta", "formula", "failed", len(years), 0, error=f"{type(exc).__name__}: {exc}")


def run_formula_delta_prediction(hindcast: xr.DataArray, obs: xr.DataArray) -> tuple[Score, xr.DataArray | None]:
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


def run_cmethods_delta_reference(hindcast: xr.DataArray, obs: xr.DataArray) -> Score:
    try:
        from cmethods import adjust
    except Exception as exc:
        return Score("delta", "python-cmethods", "skipped", int(obs.sizes["year"]), 0, error=f"python-cmethods unavailable: {exc}")

    preds = []
    truth = []
    years = list(obs.year.values)
    try:
        for year in years:
            train_years = [y for y in years if y != year]
            h_train = hindcast.sel(year=train_years)
            o_train = obs.sel(year=train_years)
            h_clim = h_train.mean(["year", "member"]).interp(lat=obs.lat, lon=obs.lon, method="linear")
            o_clim = o_train.mean("year")
            sim = hindcast.sel(year=[year]).isel(year=0, drop=True).mean("member").interp(lat=obs.lat, lon=obs.lon, method="linear")

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
                corrected = adjust(
                    method="delta_method",
                    obs=_as_single_time(o_clim),
                    simh=_as_single_time(h_clim),
                    simp=_as_single_time(sim),
                    kind="+",
                    group="time",
                    input_core_dims={"obs": "time", "simh": "time", "simp": "time"},
                ).isel(time=0, drop=True)
            preds.append(corrected.expand_dims(year=[year]))
            truth.append(obs.sel(year=[year]))
        return score_prediction("delta", "python-cmethods", xr.concat(preds, "year"), xr.concat(truth, "year"))
    except Exception as exc:
        return Score("delta", "python-cmethods", "failed", len(years), 0, error=f"{type(exc).__name__}: {exc}")


def methods_for_suite(suite: str) -> list[str]:
    if suite == "report":
        return list(REPORT_METHODS)
    if suite == "nightly":
        return list(NIGHTLY_METHODS)
    raise ValueError(f"unknown method suite: {suite}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=sorted(PRESETS), default="texas")
    parser.add_argument(
        "--method-suite",
        choices=("report", "nightly"),
        default="report",
        help="Method set to run. 'report' excludes rank-analog; 'nightly' keeps rank-analog covered.",
    )
    parser.add_argument("--product")
    parser.add_argument("--start-year", type=int)
    parser.add_argument("--end-year", type=int)
    parser.add_argument("--bbox", nargs=4, type=float)
    parser.add_argument("--months", nargs="+", type=int, help="Optional 1-12 month numbers to average within each year")
    parser.add_argument("--coarsen-factor", type=int)
    parser.add_argument("--members", type=int, default=3)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--out")
    args = parser.parse_args()
    preset = PRESETS[args.preset]
    product = args.product or preset["product"]
    start_year = args.start_year if args.start_year is not None else preset["start_year"]
    end_year = args.end_year if args.end_year is not None else preset["end_year"]
    bbox = args.bbox or preset["bbox"]
    months = args.months if args.months is not None else preset.get("months")
    out_path = args.out or preset["out"]

    coarsen_factor = args.coarsen_factor if args.coarsen_factor is not None else preset.get("coarsen_factor", 3)

    obs = fetch_chirps(product, (start_year, end_year), bbox, cache=not args.no_cache, months=months)
    hindcast = make_hindcast(obs, coarsen_factor, args.members)

    scores: list[Score] = []
    predictions: list[tuple[str, str, xr.DataArray]] = []

    bilinear = _linear_baseline(hindcast, obs)
    scores.append(score_prediction("bilinear", "interpolation", bilinear, obs))
    predictions.append(("bilinear", "interpolation", bilinear))

    for method in methods_for_suite(args.method_suite):
        score, pred = run_deepscale_prediction(method, hindcast, obs)
        scores.append(score)
        if pred is not None:
            predictions.append((method, "deepscale", pred))

    formula_score, formula_pred = run_formula_delta_prediction(hindcast, obs)
    scores.append(formula_score)
    if formula_pred is not None:
        predictions.append(("delta", "formula", formula_pred))
    scores.append(run_cmethods_delta_reference(hindcast, obs))
    for kind in ("qm", "dqm"):
        xsdba_score, xsdba_pred = run_xsdba_prediction(kind, hindcast, obs)
        scores.append(xsdba_score)
        if xsdba_pred is not None:
            predictions.append((kind, "xsdba", xsdba_pred))

    signal_diagnostics = [
        signal_preservation(method, reference, pred, hindcast, obs)
        for method, reference, pred in predictions
    ]

    payload = {
        "dataset": {
            "preset": args.preset,
            "product": product,
            "years": [start_year, end_year],
            "actual_years": [int(y) for y in obs.year.values],
            "bbox": bbox,
            "months": months,
            "obs_shape": dict(obs.sizes),
            "hindcast_shape": dict(hindcast.sizes),
            "coarsen_factor": coarsen_factor,
            "members": args.members,
            "method_suite": args.method_suite,
            "methods": methods_for_suite(args.method_suite),
            "obs_summary": _summary_stats(obs),
            "coarse_input_summary": _summary_stats(hindcast.mean("member")),
        },
        "scores": [asdict(s) for s in scores],
        "signal_preservation": [asdict(s) for s in signal_diagnostics],
        "diagnostic_notes": [
            "pred_summary/obs_summary/distribution_bias compare min, mean, max, std, and p50/p70/p90/p95 on each scored prediction.",
            "bilinear/interpolation is the no-training spatial baseline from the coarsened CHIRPS pseudo-model to the benchmark grid.",
            "signal_preservation compares coarse pseudo-model vs coarse benchmark correlation with downscaled forecast vs high-resolution benchmark correlation.",
            "For quantile-mapping-style corrections, large negative corr_delta suggests the downscaling damaged the coarse forecast signal.",
        ],
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
