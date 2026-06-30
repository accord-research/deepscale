"""Forecast output plots: tercile, deterministic, exceedance, flex-PDF."""

from pathlib import Path

import numpy as np
from .._optional import require_optional


_HINT = "pip install deepscale[plotting]"

# Dominant-tercile color saturation: probability above 1/3 at which
# the color reaches full intensity. 0.37 = 70% probability cap.
_TERCILE_SAT = 0.37

# Natural Earth coastline/border shapefiles, as cached by cartopy. Used to draw
# a basemap via geopandas when cartopy itself is not installed.
_NE_ROOT = Path.home() / ".local" / "share" / "cartopy" / "shapefiles" / "natural_earth"
_NE_COAST = _NE_ROOT / "physical" / "ne_50m_coastline.shp"
_NE_BORDERS = _NE_ROOT / "cultural" / "ne_50m_admin_0_boundary_lines_land.shp"


def _new_fig(ax, figsize=(8, 5)):
    import importlib
    require_optional("matplotlib", _HINT)
    plt = importlib.import_module("matplotlib.pyplot")
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    return plt, fig, ax


def _tercile_rgb(probs, red_cat, blue_cat):
    """Dominant-tercile RGB image. `probs` is (tercile=3, lat, lon).

    Cells that are not a complete finite tercile triple (significance-masked or
    uncalibratable cells arrive as all-NaN) are left blank (white) rather than
    painted into a category. Without this, ``argmax``/``max`` over an all-NaN
    cell return category 0 / NaN, so masked cells would render as a confident
    below/above colour (or NaN pixels) instead of as "no valid forecast".
    """
    valid = np.isfinite(probs).all(axis=0)
    dom_cat = probs.argmax(axis=0)
    dom_prob = probs.max(axis=0)
    intensity = np.clip((dom_prob - 1 / 3) / _TERCILE_SAT, 0.0, 1.0)

    rgb = np.ones(dom_cat.shape + (3,))
    is_red = (dom_cat == red_cat) & valid
    is_blue = (dom_cat == blue_cat) & valid
    is_normal = (dom_cat == 1) & valid
    rgb[is_red] = np.stack(
        [np.ones(is_red.sum()), 1 - intensity[is_red], 1 - intensity[is_red]], axis=-1
    )
    rgb[is_blue] = np.stack(
        [1 - intensity[is_blue], 1 - intensity[is_blue], np.ones(is_blue.sum())], axis=-1
    )
    rgb[is_normal] = np.stack(
        [1 - 0.4 * intensity[is_normal]] * 3, axis=-1
    )
    return rgb


def _make_tercile_axes(plt, extent, figsize):
    """Best-available single-map axes: a cartopy GeoAxes if cartopy is
    importable, otherwise a plain axes. Returns (fig, ax, is_geo)."""
    try:
        import importlib
        ccrs = importlib.import_module("cartopy.crs")
    except Exception:
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax, False
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    return fig, ax, True


def _draw_cartopy_basemap(ax):
    import importlib
    cfeature = importlib.import_module("cartopy.feature")
    ax.coastlines(resolution="50m", linewidth=0.8, color="#333333")
    ax.add_feature(cfeature.BORDERS, linewidth=0.6, edgecolor="#555555")
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="#777777", alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False


def _to_0_360(gdf):
    """Shift a -180..180 GeoDataFrame into the 0-360 longitude convention.

    Natural Earth ships in -180..180; forecast grids are often 0-360. Any
    geometry sitting in the western hemisphere (min longitude < 0) is translated
    by +360 so coastlines line up with 0-360 data instead of being clipped to
    the eastern hemisphere. Geometries straddling the prime meridian are rare in
    coastline/border line data and shift wholesale, which is acceptable for a
    context basemap.
    """
    from shapely.affinity import translate

    def _shift(geom):
        return translate(geom, xoff=360.0) if geom.bounds[0] < 0 else geom

    return gdf.assign(geometry=gdf.geometry.apply(_shift))


def _draw_geopandas_basemap(ax, extent):
    """Overplot Natural Earth coastlines/borders via geopandas. Returns True if
    drawn, False if geopandas or the cached shapefiles are unavailable."""
    try:
        import importlib
        gpd = importlib.import_module("geopandas")
    except Exception:
        return False
    if not (_NE_COAST.exists() and _NE_BORDERS.exists()):
        return False
    lon_w, lon_e, lat_s, lat_n = extent
    # Natural Earth is -180..180. When the forecast grid uses the 0-360
    # convention (any longitude past 180), shift the shapefile geometries into
    # 0-360 first; otherwise .cx selects only the eastern hemisphere and clips
    # out coastlines west of the prime meridian.
    use_0_360 = lon_e > 180.0
    try:
        for path, color, lw, z in (
            (_NE_COAST, "#333333", 0.8, 3),
            (_NE_BORDERS, "#555555", 0.6, 4),
        ):
            gdf = gpd.read_file(path)
            if use_0_360:
                gdf = _to_0_360(gdf)
            gdf.cx[lon_w:lon_e, lat_s:lat_n].plot(
                ax=ax, color=color, linewidth=lw, zorder=z)
    except Exception:
        return False
    return True


def plot_tercile_forecast(pr_fcst, *, ax=None, title=None, variable_kind="precip"):
    """Dominant-tercile probability map (IRI/PyCPT convention).

    For each grid point, identifies the tercile (below/normal/above) with
    maximum probability and colors it according to the variable convention:

    - `variable_kind="precip"` (default; IRI precipitation convention):
        * below-normal (drier) -> red
        * normal              -> grey
        * above-normal (wetter) -> blue

    - `variable_kind="temp"` (IRI temperature convention; matches the
      everyday "red = hot, blue = cold" intuition):
        * below-normal (cooler) -> blue
        * normal              -> grey
        * above-normal (warmer) -> red

    Color intensity scales with `(max_prob - 1/3)`, saturating at +0.37
    (i.e. 70% probability) so highly confident forecasts don't wash out.

    Coastlines and national borders are drawn over the map when cartopy or
    geopandas (with cached Natural Earth shapefiles) is available, and quietly
    skipped otherwise. A legend below the map shows the three categories.

    Input shape: (tercile=3, lat, lon), values in [0, 1] summing to 1.
    """
    if variable_kind == "precip":
        red_cat = 0
        below_label = "Below normal (drier)"
        above_label = "Above normal (wetter)"
    elif variable_kind == "temp":
        red_cat = 2
        below_label = "Below normal (cooler)"
        above_label = "Above normal (warmer)"
    else:
        raise ValueError(
            f"variable_kind must be 'precip' or 'temp', got {variable_kind!r}"
        )
    blue_cat = 2 if red_cat == 0 else 0

    import importlib
    require_optional("matplotlib", _HINT)
    plt = importlib.import_module("matplotlib.pyplot")
    Patch = importlib.import_module("matplotlib.patches").Patch

    from .._spatial import spatial_dims

    lat_dim, lon_dim = spatial_dims(pr_fcst, context="plot_tercile_forecast")
    rgb = _tercile_rgb(
        pr_fcst.transpose("tercile", lat_dim, lon_dim).values, red_cat, blue_cat)
    lon = pr_fcst[lon_dim].values
    lat = pr_fcst[lat_dim].values
    extent = (float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max()))

    if ax is None:
        fig, ax, is_geo = _make_tercile_axes(plt, extent, figsize=(8, 5.5))
    else:
        fig = ax.figure
        is_geo = hasattr(ax, "coastlines")

    if is_geo:
        ccrs = importlib.import_module("cartopy.crs")
        ax.imshow(rgb, extent=extent, origin="lower", transform=ccrs.PlateCarree())
        _draw_cartopy_basemap(ax)
    else:
        ax.imshow(rgb, extent=extent, origin="lower", aspect="auto", zorder=1)
        drew = _draw_geopandas_basemap(ax, extent)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        if drew:
            ax.grid(color="#777777", linewidth=0.3, alpha=0.5)

    ax.set_title(title or "Dominant tercile probability")

    below_color = (1.0, 0.0, 0.0) if red_cat == 0 else (0.0, 0.0, 1.0)
    above_color = (1.0, 0.0, 0.0) if red_cat == 2 else (0.0, 0.0, 1.0)
    legend_handles = [
        Patch(facecolor=below_color, edgecolor="black", linewidth=0.5, label=below_label),
        Patch(facecolor="#999999",   edgecolor="black", linewidth=0.5, label="Normal"),
        Patch(facecolor=above_color, edgecolor="black", linewidth=0.5, label=above_label),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=3,
        frameon=True,
        framealpha=0.9,
        fontsize=8,
        title="Dominant tercile  (intensity = confidence)",
        title_fontsize=8,
    )
    return fig


def plot_deterministic_forecast(det_fcst, *, ax=None, title=None,
                                cmap="RdBu_r", center=None):
    """Single-panel pcolormesh of a deterministic field. Input: (lat, lon)."""
    plt, fig, ax = _new_fig(ax)
    if center is not None:
        import importlib
        TwoSlopeNorm = importlib.import_module("matplotlib.colors").TwoSlopeNorm
        v = float(np.abs(det_fcst.values - center).max())
        norm = TwoSlopeNorm(vmin=center - v, vcenter=center, vmax=center + v)
        im = ax.pcolormesh(det_fcst.lon, det_fcst.lat, det_fcst.values,
                           cmap=cmap, norm=norm)
    else:
        im = ax.pcolormesh(det_fcst.lon, det_fcst.lat, det_fcst.values, cmap=cmap)
    ax.set_xlabel("Lon")
    ax.set_ylabel("Lat")
    if title:
        ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046)
    return fig


def plot_exceedance_probability(exceedance_prob, threshold, *, ax=None):
    """Map of P(forecast > threshold). Input: (lat, lon), values in [0, 1]."""
    plt, fig, ax = _new_fig(ax)
    im = ax.pcolormesh(
        exceedance_prob.lon, exceedance_prob.lat, exceedance_prob.values,
        cmap="viridis", vmin=0, vmax=1,
    )
    ax.set_xlabel("Lon")
    ax.set_ylabel("Lat")
    ax.set_title(f"P(forecast > {threshold})")
    plt.colorbar(im, ax=ax, fraction=0.046, label="probability")
    return fig


def plot_flex_pdf(fcst_mu, fcst_scale, climo_mu, climo_scale, *,
                  location, ax=None):
    """Forecast vs climatology Gaussian PDFs at a single point.

    `*_mu` / `*_scale` may be scalars or DataArrays. If DataArrays, the
    nearest grid point to `location=(lon, lat)` is selected.
    """
    plt, fig, ax = _new_fig(ax, figsize=(7, 4))

    def _scalar(v):
        if hasattr(v, "sel"):
            lon, lat = location
            return float(v.sel(lon=lon, lat=lat, method="nearest"))
        return float(v)

    f_mu, f_sc = _scalar(fcst_mu), _scalar(fcst_scale)
    c_mu, c_sc = _scalar(climo_mu), _scalar(climo_scale)

    lo = min(f_mu - 4 * f_sc, c_mu - 4 * c_sc)
    hi = max(f_mu + 4 * f_sc, c_mu + 4 * c_sc)
    x = np.linspace(lo, hi, 400)

    def _gauss(x, mu, sc):
        return np.exp(-0.5 * ((x - mu) / sc) ** 2) / (sc * np.sqrt(2 * np.pi))

    ax.plot(x, _gauss(x, c_mu, c_sc), color="grey", linewidth=2, label="Climatology")
    ax.plot(x, _gauss(x, f_mu, f_sc), color="tab:blue", linewidth=2, label="Forecast")
    ax.fill_between(x, _gauss(x, f_mu, f_sc), color="tab:blue", alpha=0.15)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title(f"PDF at lon={location[0]}, lat={location[1]}")
    ax.legend()
    return fig
