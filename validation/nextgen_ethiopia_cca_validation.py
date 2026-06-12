"""NextGen-style Ethiopia CCA validation.

This is a validation artifact, not a production feature. It follows the
reproducible pieces of Acharya et al. (2021), Climate Services 24, 100272:

* NMME precipitation forecasts calibrated against CHIRPS with CCA.
* Ethiopia predictand domain: 3N-15N, 33E-48E.
* Larger predictor domain: 0N-20N, 30E-55E.
* Lead-1 seasonal forecasts, with EOF pre-processing and leave-5-out CV.
* Deterministic and probabilistic skill diagnostics.

The defaults are intentionally conservative: a single-model JJAS smoke run over
the common CFSv2 hindcast period. Expand with --products and --season once the
data path is confirmed.

Example:

    PYTHONPATH=/Users/david/rosetta/src uv run python \
      validation/nextgen_ethiopia_cca_validation.py \
      --season JJAS --products nmme/cfsv2 --start-year 1991 --end-year 2010

Paper-window stitched products can be written as:

    CFSv2=nmme/cfsv2:1982-2010+nmme/cfsv2-forecast:2011-2018
"""

from __future__ import annotations

import argparse
import calendar
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "validation" / "results"
FIGURES = ROOT / "validation" / "figures"
ROSETTA_SRC = Path.home() / "rosetta" / "src"
if ROSETTA_SRC.exists() and str(ROSETTA_SRC) not in sys.path:
    sys.path.insert(0, str(ROSETTA_SRC))


SEASONS = {
    "FMAM": {"init_month": 1, "months": [2, 3, 4, 5], "paper_name": "Belg"},
    "JJAS": {"init_month": 5, "months": [6, 7, 8, 9], "paper_name": "Kiremt"},
    "ONDJ": {"init_month": 9, "months": [10, 11, 12, 1], "paper_name": "Bega"},
}

PREDICTOR_BBOX = [0.0, 20.0, 30.0, 55.0]
PREDICTAND_BBOX = [3.0, 15.0, 33.0, 48.0]

PAPER_AVAILABLE_PRODUCTS = [
    # Exact / near-exact Rosetta-available members of the Acharya et al. setup.
    # Several paper models (CanSIPSv2 and GFDL FLOR variants) are not currently
    # exposed in the local Rosetta catalog, so this preset is a documented
    # available-subset reproduction rather than the full seven-model paper set.
    "CFSv2=nmme/cfsv2:1982-2010+nmme/cfsv2-forecast:2011-2018",
    "CCSM4=nmme/ccsm4-iri:1982-2018",
    "GEOSS2S=nmme/geoss2s-hindcast:1982-2018",
]


@dataclass
class SkillRow:
    source: str
    kind: str
    status: str
    n_years: int
    n_cells: int
    rmse: float | None = None
    spearman: float | None = None
    two_afc: float | None = None
    rpss: float | None = None
    roc_above: float | None = None
    roc_below: float | None = None
    mean_bias: float | None = None
    error: str | None = None


def _finite(value) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _target_tuple(months: list[int]) -> tuple[datetime, datetime]:
    """Return a month-only seasonal target tuple for Rosetta's fetch helper."""
    start_month = months[0]
    end_month = months[-1]
    start_year = 2000
    end_year = 2001 if end_month < start_month else 2000
    last_day = calendar.monthrange(end_year, end_month)[1]
    return datetime(start_year, start_month, 1), datetime(end_year, end_month, last_day)


def _year_coord_from_init_time(da: xr.DataArray) -> xr.DataArray:
    if "year" in da.dims:
        return da
    if "init_time" not in da.dims:
        raise ValueError(f"expected 'year' or 'init_time' dimension; got {da.dims}")
    years = da["init_time"].dt.year.values.astype(int)
    return da.assign_coords(init_time=years).rename({"init_time": "year"})


def _fetch_obs_chunk(
    product: str,
    years: tuple[int, int],
    *,
    request_interval: float | None,
) -> xr.DataArray:
    import rosetta

    kwargs = {}
    if request_interval is not None:
        kwargs["request_interval"] = request_interval
    ds = rosetta.fetch(
        product,
        "precip",
        hindcast=years,
        region=PREDICTAND_BBOX,
        cache=True,
        verbose=True,
        progress=False,
        **kwargs,
    )
    return ds["precip"].astype("float64")


def _seasonal_obs(
    product: str,
    years: tuple[int, int],
    months: list[int],
    coarsen: int,
    *,
    year_chunk_size: int,
    request_interval: float | None,
) -> xr.DataArray:
    if year_chunk_size > 0:
        chunks = []
        for start in range(years[0], years[1] + 1, year_chunk_size):
            end = min(start + year_chunk_size - 1, years[1])
            print(f"fetch obs {product}: {start}-{end}")
            chunks.append(_fetch_obs_chunk(product, (start, end), request_interval=request_interval))
        precip = xr.concat(chunks, dim="time").sortby("time")
    else:
        precip = _fetch_obs_chunk(product, years, request_interval=request_interval)

    selected = precip.sel(time=precip.time.dt.month.isin(months))
    obs = selected.groupby("time.year").mean("time", skipna=True)
    if coarsen > 1:
        obs = obs.coarsen(lat=coarsen, lon=coarsen, boundary="trim").mean(skipna=True)
    obs = obs.sel(year=slice(years[0], years[1]))
    obs.name = "obs"
    obs.attrs["units"] = precip.attrs.get("units", "unknown")
    return obs


def _seasonal_forecast(
    product: str,
    years: tuple[int, int],
    season: str,
    region: list[float],
) -> xr.DataArray:
    import rosetta

    spec = SEASONS[season]
    # Any year works when hindcast is explicitly supplied; Rosetta uses the
    # month from init and the hindcast tuple for the actual year range.
    init = f"{years[1]}-{spec['init_month']:02d}"
    ds = rosetta.fetch(
        product,
        "precip",
        init=init,
        target=_target_tuple(spec["months"]),
        hindcast=years,
        region=region,
        cache=True,
        verbose=True,
        progress=False,
    )
    da = _year_coord_from_init_time(ds["precip"].astype("float64"))
    if "lead_time" in da.dims:
        da = da.mean("lead_time", skipna=True)
    if "member" not in da.dims:
        da = da.expand_dims(member=[0])
    da = da.transpose("year", "member", "lat", "lon")
    da.name = product.replace("/", "_")
    return da


def _parse_product_spec(spec: str, default_years: tuple[int, int]) -> tuple[str, list[tuple[str, tuple[int, int]]]]:
    """Parse NAME=product:y0-y1+product2:y0-y1 product specifications."""
    if "=" in spec:
        label, body = spec.split("=", 1)
    else:
        label, body = spec, spec
    segments = []
    for part in body.split("+"):
        if ":" in part:
            product, year_text = part.rsplit(":", 1)
            y0_text, y1_text = year_text.split("-", 1)
            years = (int(y0_text), int(y1_text))
        else:
            product = part
            years = default_years
        segments.append((product, years))
    return label, segments


def _seasonal_forecast_spec(spec: str, default_years: tuple[int, int], season: str) -> tuple[str, xr.DataArray, list[dict]]:
    label, segments = _parse_product_spec(spec, default_years)
    arrays = []
    segment_meta = []
    for product, years in segments:
        da = _seasonal_forecast(product, years, season, PREDICTOR_BBOX)
        arrays.append(da)
        segment_meta.append({"product": product, "years": list(years), "n_years": int(da.sizes.get("year", 0))})
    if len(arrays) == 1:
        out = arrays[0]
    else:
        out = xr.concat(arrays, dim="year").sortby("year")
        _, unique_idx = np.unique(out.year.values.astype(int), return_index=True)
        out = out.isel(year=np.sort(unique_idx))
    out.name = label
    return label, out, segment_meta


def _interp_to_obs_grid(da: xr.DataArray, obs: xr.DataArray) -> xr.DataArray:
    da = da.sel(lat=slice(PREDICTAND_BBOX[0], PREDICTAND_BBOX[1]), lon=slice(PREDICTAND_BBOX[2], PREDICTAND_BBOX[3]))
    return da.interp(lat=obs.lat, lon=obs.lon, method="linear")


def _select_common_years(*arrays: xr.DataArray) -> list[int]:
    common = set(int(y) for y in arrays[0].year.values)
    for arr in arrays[1:]:
        common &= set(int(y) for y in arr.year.values)
    return sorted(common)


def _cv_cca(
    hindcast: xr.DataArray,
    obs: xr.DataArray,
    *,
    cv_window: int,
    x_eof_modes: int,
    y_eof_modes: int,
    cca_modes: int,
) -> tuple[xr.DataArray, list[float]]:
    from deepscale.cv import loyo
    from deepscale.methods.cca import CCAMethod

    years = _select_common_years(hindcast, obs)
    hindcast = hindcast.sel(year=years)
    obs = obs.sel(year=years)
    preds = []
    leverages = []
    for train_years, test_year in loyo(years, window=cv_window):
        model = CCAMethod(
            n_modes=cca_modes,
            x_eof_modes=x_eof_modes,
            y_eof_modes=y_eof_modes,
            cca_modes=cca_modes,
            standardize=True,
        )
        model.fit(hindcast.sel(year=train_years), obs.sel(year=train_years))
        forecast = hindcast.sel(year=[test_year]).isel(year=0, drop=True)
        pred = model.predict(forecast).mean("member")
        preds.append(pred.expand_dims(year=[test_year]))
        leverages.append(float(model.leverage(forecast)))
    out = xr.concat(preds, "year").sortby("year")
    out.name = "cca"
    return out, leverages


def _score(source: str, kind: str, pred: xr.DataArray, obs: xr.DataArray, cv_window: int) -> tuple[SkillRow, xr.Dataset]:
    from deepscale.metrics.rmse import RMSEMetric
    from deepscale.metrics.roc import ROCAboveNormalMetric, ROCBelowNormalMetric
    from deepscale.metrics.rpss import RPSSMetric
    from deepscale.metrics.spearman import SpearmanMetric
    from deepscale.metrics.two_afc import TwoAFCMetric
    from deepscale.tercile import to_tercile_cv

    years = _select_common_years(pred, obs)
    pred = pred.sel(year=years)
    obs = obs.sel(year=years)
    pred, obs = xr.align(pred, obs, join="inner")
    pred = pred.transpose("year", "lat", "lon")
    obs = obs.transpose("year", "lat", "lon")
    valid = np.isfinite(pred.values) & np.isfinite(obs.values)
    n_cells = int(valid.sum())
    if n_cells < 2:
        row = SkillRow(source, kind, "failed", int(obs.sizes.get("year", 0)), n_cells, error="too few finite cells")
        return row, xr.Dataset()

    probs = to_tercile_cv(pred, obs, method="gaussian_loo").transpose("year", "tercile", "lat", "lon")
    row = SkillRow(
        source=source,
        kind=kind,
        status="ok",
        n_years=int(obs.sizes["year"]),
        n_cells=n_cells,
        rmse=_finite(RMSEMetric().compute(pred, obs)),
        spearman=_finite(SpearmanMetric().compute(pred, obs)),
        two_afc=_finite(TwoAFCMetric().compute(pred, obs)),
        rpss=_finite(RPSSMetric().compute(probs, obs, cv_window=cv_window)),
        roc_above=_finite(ROCAboveNormalMetric().compute(probs, obs)),
        roc_below=_finite(ROCBelowNormalMetric().compute(probs, obs)),
        mean_bias=_finite((pred - obs).mean()),
    )
    spatial = xr.Dataset(
        {
            "rmse": RMSEMetric().compute(pred, obs, spatial=True),
            "spearman": SpearmanMetric().compute(pred, obs, spatial=True),
            "two_afc": TwoAFCMetric().compute(pred, obs, spatial=True),
            "rpss": RPSSMetric().compute(probs, obs, spatial=True, cv_window=cv_window),
        }
    )
    return row, spatial


def _plot_spatial(summary: xr.Dataset, out: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    metrics = ["spearman", "two_afc", "rpss", "rmse"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax, metric in zip(axes.ravel(), metrics):
        da = summary[metric]
        robust = metric == "rmse"
        da.plot(ax=ax, cmap="viridis" if metric == "rmse" else "RdBu_r", robust=robust)
        ax.set_title(metric)
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
    fig.suptitle(title)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    years = (args.start_year, args.end_year)
    season = args.season.upper()
    if season not in SEASONS:
        raise ValueError(f"unknown season {args.season!r}; expected one of {sorted(SEASONS)}")

    obs = _seasonal_obs(
        args.obs_product,
        years,
        SEASONS[season]["months"],
        args.obs_coarsen,
        year_chunk_size=args.obs_year_chunk_size,
        request_interval=args.request_interval,
    )
    rows: list[SkillRow] = []
    spatial_outputs = []
    spatial_labels = []
    calibrated_predictions = []
    raw_predictions = []
    failed_products = []
    product_segments = {}

    for product_spec in args.products:
        try:
            product, hindcast, segment_meta = _seasonal_forecast_spec(product_spec, years, season)
            product_segments[product] = segment_meta
            common_years = _select_common_years(hindcast, obs)
            hindcast = hindcast.sel(year=common_years)
            obs_model = obs.sel(year=common_years)
            if obs_model.sizes["year"] < args.min_years:
                raise ValueError(f"only {obs_model.sizes['year']} overlapping years")

            cca_pred, leverages = _cv_cca(
                hindcast,
                obs_model,
                cv_window=args.cv_window,
                x_eof_modes=args.x_eof_modes,
                y_eof_modes=args.y_eof_modes,
                cca_modes=args.cca_modes,
            )
            raw = _interp_to_obs_grid(hindcast.mean("member"), obs_model)
            raw = raw.sel(year=cca_pred.year)
            obs_score = obs_model.sel(year=cca_pred.year)

            cca_row, cca_spatial = _score(product, "cca_calibrated", cca_pred, obs_score, args.cv_window)
            raw_row, raw_spatial = _score(product, "raw_ensemble_mean", raw, obs_score, args.cv_window)
            rows.extend([cca_row, raw_row])

            spatial_outputs.append(cca_spatial)
            spatial_labels.append((product, "cca_calibrated"))
            spatial_outputs.append(raw_spatial)
            spatial_labels.append((product, "raw_ensemble_mean"))
            calibrated_predictions.append(cca_pred.expand_dims(source=[product]))
            raw_predictions.append(raw.expand_dims(source=[product]))
            print(f"ok {product}: CCA spearman={cca_row.spearman}, raw spearman={raw_row.spearman}")
        except Exception as exc:
            label, _segments = _parse_product_spec(product_spec, years)
            failed_products.append({"product": label, "spec": product_spec, "error": f"{type(exc).__name__}: {exc}"})
            rows.append(SkillRow(label, "cca_calibrated", "failed", 0, 0, error=f"{type(exc).__name__}: {exc}"))
            print(f"failed {label}: {type(exc).__name__}: {exc}")

    if calibrated_predictions:
        cmme = xr.concat(calibrated_predictions, "source").mean("source", skipna=True)
        raw_mme = xr.concat(raw_predictions, "source").mean("source", skipna=True)
        obs_score = obs.sel(year=cmme.year)
        cmme_row, cmme_spatial = _score("equal_weight_products", "cca_cmme", cmme, obs_score, args.cv_window)
        raw_mme_row, raw_mme_spatial = _score("equal_weight_products", "raw_mme", raw_mme, obs_score, args.cv_window)
        rows.extend([cmme_row, raw_mme_row])
        spatial_outputs.append(cmme_spatial)
        spatial_labels.append(("equal_weight_products", "cca_cmme"))
        spatial_outputs.append(raw_mme_spatial)
        spatial_labels.append(("equal_weight_products", "raw_mme"))

    spatial_ds = xr.concat(spatial_outputs, dim="entry", join="override") if spatial_outputs else xr.Dataset()
    if "entry" in spatial_ds.dims:
        # Keep source/kind as coordinates along entry; this avoids xarray trying
        # to create a rectangular source x kind product with missing combinations.
        spatial_ds = spatial_ds.assign_coords(
            source=("entry", [source for source, _kind in spatial_labels]),
            kind=("entry", [kind for _source, kind in spatial_labels]),
        )

    tag = args.tag or f"nextgen_ethiopia_cca_{season.lower()}_{args.start_year}_{args.end_year}"
    json_path = RESULTS / f"{tag}.json"
    nc_path = RESULTS / f"{tag}_spatial.nc"
    fig_path = FIGURES / f"{tag}_skill_maps.png"

    metadata = {
        "paper": {
            "citation": "Acharya et al. 2021, Climate Services 24, 100272",
            "doi": "10.1016/j.cliser.2021.100272",
            "replication_scope": "NextGen-style CCA validation using Rosetta-available NMME/CHIRPS products; not an exact CPT/PyCPT reproduction.",
        },
        "configuration": {
            "season": season,
            "paper_season_name": SEASONS[season]["paper_name"],
            "years": list(years),
            "obs_product": args.obs_product,
            "obs_coarsen": args.obs_coarsen,
            "obs_year_chunk_size": args.obs_year_chunk_size,
            "request_interval": args.request_interval,
            "products": args.products,
            "product_segments": product_segments,
            "predictor_bbox": PREDICTOR_BBOX,
            "predictand_bbox": PREDICTAND_BBOX,
            "cv": f"leave-{args.cv_window}-out centered window",
            "x_eof_modes": args.x_eof_modes,
            "y_eof_modes": args.y_eof_modes,
            "cca_modes": args.cca_modes,
            "standardize": True,
            "probabilistic_conversion": "gaussian_loo from deterministic CV predictions",
        },
        "scores": [asdict(row) for row in rows],
        "failed_products": failed_products,
        "outputs": {"json": str(json_path), "spatial_nc": str(nc_path), "figure": str(fig_path)},
    }
    json_path.write_text(json.dumps(metadata, indent=2))
    if spatial_outputs:
        spatial_ds.to_netcdf(nc_path)
        # Plot CMME if present, otherwise the first calibrated model.
        plot_entry = None
        for i in range(spatial_ds.sizes["entry"]):
            if str(spatial_ds.kind.values[i]) == "cca_cmme":
                plot_entry = i
                break
        if plot_entry is None:
            for i in range(spatial_ds.sizes["entry"]):
                if str(spatial_ds.kind.values[i]) == "cca_calibrated":
                    plot_entry = i
                    break
        if plot_entry is not None:
            _plot_spatial(
                spatial_ds.isel(entry=plot_entry),
                fig_path,
                f"NextGen-style Ethiopia CCA {season} ({years[0]}-{years[1]})",
            )

    print(json.dumps(metadata, indent=2))
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default="JJAS", choices=sorted(SEASONS))
    parser.add_argument("--start-year", type=int, default=1991)
    parser.add_argument("--end-year", type=int, default=2010)
    parser.add_argument("--obs-product", default="obs/chirps-v2-monthly")
    parser.add_argument(
        "--obs-coarsen",
        type=int,
        default=5,
        help="Coarsen native CHIRPS before validation. Default 5 gives 0.25 degree from 0.05 degree.",
    )
    parser.add_argument(
        "--obs-year-chunk-size",
        type=int,
        default=1,
        help="Fetch observations in N-year chunks. Default 1 is slow but gentle on CHIRPS.",
    )
    parser.add_argument(
        "--request-interval",
        type=float,
        default=3.0,
        help="Seconds between Rosetta HTTP requests for sources that support throttling.",
    )
    parser.add_argument(
        "--products",
        nargs="+",
        default=["nmme/cfsv2"],
        help="Rosetta NMME products/specs to calibrate. Supports NAME=product:y0-y1+product2:y0-y1.",
    )
    parser.add_argument(
        "--paper-available-products",
        action="store_true",
        help="Use the Rosetta-available subset of the Acharya et al. 1982-2018 model setup.",
    )
    parser.add_argument("--cv-window", type=int, default=5)
    parser.add_argument("--x-eof-modes", type=int, default=5)
    parser.add_argument("--y-eof-modes", type=int, default=5)
    parser.add_argument("--cca-modes", type=int, default=5)
    parser.add_argument("--min-years", type=int, default=12)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()
    if args.paper_available_products:
        args.products = PAPER_AVAILABLE_PRODUCTS
        args.start_year = 1982
        args.end_year = 2018
    return args


if __name__ == "__main__":
    run(parse_args())
