"""
End-to-end demo: the WVG / logistic seasonal-forecast stream on real data.

Real-data path (default): fetches ERA5 sea-surface temperature over the tropical
Pacific (for the Western-V Gradient index) and ERA5 precipitation over East
Africa (the predictand) via Rosetta, builds a custom WVG index with
`deepscale.Index.custom(...)`, and runs `deepscale.calibrate(...)` with
`deepscale.LogitConfig` on gridded SST hindcast/forecast inputs. Always writes
a dominant-tercile PNG and NetCDF to examples/output/.

Run from the repository root:
  python examples/demo_logistic_wvg.py             # real data (needs CDS)
  python examples/demo_logistic_wvg.py --synthetic # offline, deterministic, self-checking

Prerequisites for the real-data path: Rosetta + DeepScale importable, CDS
credentials in ~/.cdsapirc, accepted ERA5 licences. Missing credentials/network
fail at the fetch step (a pre-existing requirement, not a bug). Use --synthetic
offline.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _configure_import_paths() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    for rel in ("rosetta/src", "../rosetta/src", "src"):
        p = (repo_root / rel).resolve()
        if p.exists():
            sys.path.insert(0, str(p))
    return repo_root


REPO_ROOT = _configure_import_paths()

import numpy as np
import xarray as xr
import deepscale as ds

PACIFIC = [-20, 40, 120, 240]            # WVG boxes live here (0-360 lon)
EAST_AFRICA = [-5, 5, 33, 48]
HINDCAST_YEARS = list(range(2000, 2021))
OUTPUT_DIR = REPO_ROOT / "examples" / "output"
CACHE_DIR = OUTPUT_DIR / "demo_cache"
PNG = OUTPUT_DIR / "logistic_wvg_tercile.png"
NC = OUTPUT_DIR / "logistic_wvg_tercile.nc"


def _load_or_fetch(cache_path, fetch_fn):
    if cache_path.exists():
        return xr.open_dataset(cache_path)
    d = fetch_fn()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    d.to_netcdf(cache_path)
    return d


def build_real():
    import rosetta
    span = f"{HINDCAST_YEARS[0]}-{HINDCAST_YEARS[-1]}"
    sst_ds = _load_or_fetch(
        CACHE_DIR / f"era5_sst_monthly_pacific_{span}.nc",
        lambda: rosetta.fetch("obs/era5", "sst",
                              hindcast=(HINDCAST_YEARS[0], HINDCAST_YEARS[-1]),
                              region=PACIFIC, verbose=True),
    )
    pr_ds = _load_or_fetch(
        CACHE_DIR / f"era5_precip_monthly_ea_{span}.nc",
        lambda: rosetta.fetch("obs/era5", "precip",
                              hindcast=(HINDCAST_YEARS[0], HINDCAST_YEARS[-1]),
                              region=EAST_AFRICA, verbose=True),
    )
    # MAM-seasonal SST field (year, lat, lon) for the index. Drop any stray
    # scalar member/number coordinate CDS attaches.
    sst = sst_ds["sst"]
    sst = sst.sel(time=sst.time.dt.month.isin([3, 4, 5])).groupby("time.year").mean("time")
    sst = sst.sel(year=HINDCAST_YEARS)
    sst = sst.drop_vars([c for c in ("member", "number") if c in sst.coords], errors="ignore")
    # MAM precipitation predictand, coarsened to ~1deg to keep the per-cell
    # logistic snappy for a demo.
    pr = pr_ds["precip"]
    pr = pr.sel(time=pr.time.dt.month.isin([3, 4, 5])).groupby("time.year").mean("time")
    pr = pr.sel(year=HINDCAST_YEARS).interp(
        lat=np.arange(-5, 5.01, 1.0), lon=np.arange(33, 48.01, 1.0))
    pr = pr.drop_vars([c for c in ("member", "number") if c in pr.coords], errors="ignore")
    return sst, pr


def build_synthetic():
    rng = np.random.default_rng(0)
    years = np.arange(1991, 2021)
    lat, lon = np.arange(-40, 41, 5.0), np.arange(0, 360, 5.0)
    driver = rng.normal(0, 1, len(years))
    box = (((xr.DataArray(lat, dims="lat") >= -5) & (xr.DataArray(lat, dims="lat") <= 5))
           & ((xr.DataArray(lon, dims="lon") >= 190) & (xr.DataArray(lon, dims="lon") <= 240))).astype(float)
    sst = xr.DataArray(27.0 + driver[:, None, None] * box.values[None]
                       + rng.normal(0, 0.3, (len(years), len(lat), len(lon))),
                       dims=["year", "lat", "lon"], coords={"year": years, "lat": lat, "lon": lon})
    nlat, nlon = 6, 8
    olat, olon = np.linspace(-5, 12, nlat), np.linspace(33, 48, nlon)
    responsive = np.zeros((nlat, nlon)); responsive[:3, :] = 1.0   # low-lat half responds
    rain = 120.0 - 30.0 * driver[:, None, None] * responsive[None] + rng.normal(0, 12.0, (len(years), nlat, nlon))
    obs = xr.DataArray(rain, dims=["year", "lat", "lon"],
                       coords={"year": years, "lat": olat, "lon": olon})
    return sst, obs


def save_png(tercile, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        fig = _plot_tercile_map(tercile, title)
        renderer = "cartopy"
    except ImportError:
        try:
            fig = _plot_tercile_geopandas_map(tercile, title)
            renderer = "geopandas"
        except ImportError:
            from deepscale.plotting import plot_tercile_forecast
            fig, ax = plt.subplots(figsize=(7, 5))
            plot_tercile_forecast(tercile, ax=ax, title=title)
            renderer = "plain"
    fig.savefig(PNG, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {PNG} ({renderer} renderer)")


def _tercile_rgb_and_extent(tercile):
    probs = tercile.transpose("tercile", "lat", "lon").values
    dom_cat = probs.argmax(axis=0)
    dom_prob = probs.max(axis=0)
    intensity = np.clip((dom_prob - 1 / 3) / 0.37, 0.0, 1.0)

    rgb = np.ones(dom_cat.shape + (3,))
    below = dom_cat == 0
    normal = dom_cat == 1
    above = dom_cat == 2
    rgb[below] = np.stack([
        np.ones(below.sum()),
        1 - intensity[below],
        1 - intensity[below],
    ], axis=-1)
    rgb[normal] = np.stack([
        1 - 0.4 * intensity[normal],
        1 - 0.4 * intensity[normal],
        1 - 0.4 * intensity[normal],
    ], axis=-1)
    rgb[above] = np.stack([
        1 - intensity[above],
        1 - intensity[above],
        np.ones(above.sum()),
    ], axis=-1)

    lon = tercile.lon.values
    lat = tercile.lat.values
    extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]
    return rgb, extent


def _legend_handles():
    from matplotlib.patches import Patch

    return [
        Patch(facecolor="red", edgecolor="black", linewidth=0.5,
              label="Below normal (drier)"),
        Patch(facecolor="#999999", edgecolor="black", linewidth=0.5,
              label="Normal"),
        Patch(facecolor="blue", edgecolor="black", linewidth=0.5,
              label="Above normal (wetter)"),
    ]


def _plot_tercile_map(tercile, title):
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    rgb, extent = _tercile_rgb_and_extent(tercile)

    fig = plt.figure(figsize=(8, 5.5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.imshow(rgb, extent=extent, origin="lower", transform=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="#f6f4ee", edgecolor="none", zorder=-2)
    ax.add_feature(cfeature.OCEAN, facecolor="#eef6fb", edgecolor="none", zorder=-3)
    ax.coastlines(resolution="50m", linewidth=0.8, color="#333333")
    ax.add_feature(cfeature.BORDERS, linewidth=0.6, edgecolor="#555555")
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="#777777", alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    ax.set_title(title)
    ax.legend(
        handles=_legend_handles(),
        loc="lower right",
        framealpha=0.92,
        fontsize=8,
        title="Dominant tercile",
        title_fontsize=8,
    )
    return fig


def _plot_tercile_geopandas_map(tercile, title):
    import matplotlib.pyplot as plt
    import geopandas as gpd

    ne_root = Path.home() / ".local/share/cartopy/shapefiles/natural_earth"
    coast_path = ne_root / "physical/ne_50m_coastline.shp"
    borders_path = ne_root / "cultural/ne_50m_admin_0_boundary_lines_land.shp"
    if not coast_path.exists() or not borders_path.exists():
        raise ImportError("Natural Earth coastlines/borders are not cached.")

    rgb, extent = _tercile_rgb_and_extent(tercile)
    lon_w, lon_e, lat_s, lat_n = extent
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.imshow(rgb, extent=extent, origin="lower", aspect="auto", zorder=1)
    coast = gpd.read_file(coast_path).cx[lon_w:lon_e, lat_s:lat_n]
    borders = gpd.read_file(borders_path).cx[lon_w:lon_e, lat_s:lat_n]
    coast.plot(ax=ax, color="#222222", linewidth=0.8, zorder=3)
    borders.plot(ax=ax, color="#555555", linewidth=0.6, zorder=4)
    ax.set_xlim(lon_w, lon_e)
    ax.set_ylim(lat_s, lat_n)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(color="#777777", linewidth=0.3, alpha=0.5)
    ax.set_title(title)
    ax.legend(
        handles=_legend_handles(),
        loc="lower right",
        framealpha=0.92,
        fontsize=8,
        title="Dominant tercile",
        title_fontsize=8,
    )
    return fig


def save_netcdf(tercile):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tercile.rename("tercile_probability").to_netcdf(NC)
    print(f"  saved {NC}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true",
                    help="use offline synthetic data instead of fetching real data")
    args = ap.parse_args()

    print("=" * 64)
    print("  WVG / LOGISTIC — " + ("synthetic" if args.synthetic else "real data"))
    print("=" * 64)

    sst, obs = build_synthetic() if args.synthetic else build_real()

    # Custom WVG definition. Region inputs follow Rosetta's bbox convention:
    # [lat_s, lat_n, lon_w, lon_e].
    wvg = ds.Index.custom(
        name="wvg_candidate",
        regions={
            "nino34": [-5, 5, 190, 240],
            "wnp": [20, 35, 160, 210],
            "wep": [-15, 20, 120, 160],
            "wsp": [-30, -15, 155, 210],
        },
        combine=lambda z: z["nino34"] - (z["wnp"] + z["wep"] + z["wsp"]) / 3,
    )
    train_years = [int(y) for y in obs.year.values[:-1]]
    fcst_year = int(obs.year.values[-1])
    sst_train = sst.sel(year=train_years)
    sst_fcst = sst.sel(year=[fcst_year])
    obs_train = obs.sel(year=train_years)
    idx_fcst = float(wvg.reduce(sst_fcst, climatology=sst_train).values.reshape(-1)[0])
    print(f"  custom WVG index: {len(train_years)} training yrs; "
          f"forecasting {fcst_year} (its WVG = {idx_fcst:+.2f})")

    p = ds.calibrate(
        predictor_hindcast=sst_train,
        obs=obs_train,
        predictor_forecast=sst_fcst,
        method=ds.LogitConfig(
            index=wvg,
            model="icpac_independent",
            predictor_level="model_mean",
            detrend=False,
            significance=None,
            regularization=None,
        ),
    )
    assert p.dims[0] == "tercile" and p.sizes["tercile"] == 3
    # skipna=False so masked/uncalibratable cells stay NaN (not summed to 0).
    s = p.sum("tercile", skipna=False).values
    finite = np.isfinite(s)
    assert finite.any() and np.allclose(s[finite], 1.0, atol=1e-9)
    bn, nn, an = (float(p.sel(tercile=t).mean()) for t in (0, 1, 2))
    print(f"  domain-mean P(below/normal/above) = {bn:.0%} / {nn:.0%} / {an:.0%}")

    if args.synthetic:
        # The low-latitude half is index-responsive; with a known driver it
        # should be more decisive there than in the noise half.
        resp = float(p.isel(lat=slice(0, 3)).sel(tercile=0).mean())
        noise = float(p.isel(lat=slice(3, 6)).sel(tercile=0).mean())
        assert resp != noise
        print(f"  responsive P(below)={resp:.2f} vs noise={noise:.2f}")
    print("PASS: valid per-cell logistic tercile forecast")

    save_png(p, f"WVG / logistic — dominant tercile ({fcst_year})"
             if not args.synthetic else "WVG / logistic — dominant tercile (synthetic)")
    save_netcdf(p)
    print("\nWVG/logistic demo complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
