"""Spatial maps: gridded fields and per-region choropleths.

Two figures the monitoring decks lean on and deepscale could not yet draw:

:func:`plot_field_map` renders a gridded ``(lat, lon)`` field — a percentile map,
a rank-of-record map, a seasonal total. It carries a *highlight* mode for the
"driest / wettest on record" pixels the Early Estimates maps mark in red.

:func:`plot_choropleth` fills administrative polygons by a per-region value —
the SMPG woreda percentile maps. It takes the output of
:func:`rosetta.zonal` (a value per ``region``) plus the geometries, and needs no
knowledge of what admin level or country it is drawing.

Both are colormap-and-geometry helpers, not analyses: they render a DataArray
someone else computed. Coastlines/borders use cartopy when it is installed and
degrade to a plain lat/lon axis (with optional geometry outlines) when it is
not, so the pixel path has no hard cartopy dependency.
"""
from __future__ import annotations

import importlib

import numpy as np

from .._optional import require_optional

_HINT = "pip install deepscale[plotting]"

# Percentile / probability fields live in [0, 1]. A perceptually flat sequential
# map reads "low is bad" unambiguously for rainfall; the caller can override.
_PERCENTILE_CMAP = "YlGnBu"
# The deck marks the record-driest pixels in a saturated red. One colour, so it
# reads as a category ("on record") not a continuum.
_RECORD_COLOR = "#c0272d"


def _pyplot():
    require_optional("matplotlib", _HINT)
    return importlib.import_module("matplotlib.pyplot")


def _try_cartopy():
    try:
        import cartopy.crs as ccrs  # noqa: F401
        import cartopy.feature as cfeature  # noqa: F401
        return importlib.import_module("cartopy.crs"), importlib.import_module(
            "cartopy.feature")
    except ImportError:
        return None, None


def _geo_axes(ax, figsize):
    """A (plt, fig, ax, ccrs-or-None) tuple; a cartopy GeoAxes if available."""
    plt = _pyplot()
    ccrs, _ = _try_cartopy()
    if ax is not None:
        return plt, ax.figure, ax, ccrs
    if ccrs is not None:
        fig, ax = plt.subplots(figsize=figsize,
                               subplot_kw={"projection": ccrs.PlateCarree()})
    else:
        fig, ax = plt.subplots(figsize=figsize)
    return plt, fig, ax, ccrs


def _add_basemap(ax, ccrs):
    if ccrs is None:
        return
    _, cfeature = _try_cartopy()
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="#666666")


def _transform(ccrs):
    return {"transform": ccrs.PlateCarree()} if ccrs is not None else {}


def plot_field_map(
    da,
    *,
    ax=None,
    cmap=None,
    vmin=None,
    vmax=None,
    highlight=None,
    highlight_label="driest on record",
    boundaries=None,
    title=None,
    cbar_label=None,
    figsize=(8, 7),
):
    """Render a gridded ``(lat, lon)`` field as a map.

    Parameters
    ----------
    da : xr.DataArray
        Must have lat/lon dims (any of the usual aliases). Any extra dims must
        already be reduced to a single 2-D slice.
    highlight : float, optional
        Overpaint cells equal to this value in a single saturated colour — the
        "driest on record" convention (pass ``highlight=1`` over a
        :func:`deepscale.rank_of_record` field). NaN-safe.
    boundaries : GeoDataFrame / GeoSeries, optional
        Admin outlines to overlay (drawn as unfilled edges).
    cmap, vmin, vmax : optional
        Default to a sequential map on ``[0, 1]`` when the data looks like a
        percentile field, else matplotlib's autoscaling.

    Returns
    -------
    matplotlib Figure
    """
    from .._spatial import spatial_dims

    lat, lon = spatial_dims(da, context="plot_field_map")
    if da.ndim != 2:
        raise ValueError(
            f"plot_field_map needs a 2-D (lat, lon) field; got dims {tuple(da.dims)}. "
            "Select or reduce the extra dimensions first."
        )

    plt, fig, ax, ccrs = _geo_axes(ax, figsize)
    da = da.sortby([lat, lon])
    values = da.transpose(lat, lon).values

    looks_like_fraction = np.nanmin(values) >= 0.0 and np.nanmax(values) <= 1.0
    if cmap is None:
        cmap = _PERCENTILE_CMAP if looks_like_fraction else "viridis"
    if vmin is None and looks_like_fraction:
        vmin, vmax = 0.0, 1.0

    mesh = ax.pcolormesh(da[lon].values, da[lat].values, values,
                         cmap=cmap, vmin=vmin, vmax=vmax, **_transform(ccrs))
    fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04,
                 label=cbar_label or (da.name or "value"))

    if highlight is not None:
        mask = np.isclose(values, float(highlight))
        if mask.any():
            painted = np.where(mask, 1.0, np.nan)
            from matplotlib.colors import ListedColormap
            ax.pcolormesh(da[lon].values, da[lat].values, painted,
                          cmap=ListedColormap([_RECORD_COLOR]),
                          vmin=0, vmax=1, **_transform(ccrs))
            ax.plot([], [], "s", color=_RECORD_COLOR, label=highlight_label)
            ax.legend(loc="upper right", frameon=True, fontsize=8)

    _overlay_boundaries(ax, boundaries, ccrs)
    _add_basemap(ax, ccrs)
    if title:
        ax.set_title(title)
    if ccrs is None:
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return fig


def plot_choropleth(
    values,
    geometries,
    *,
    by=None,
    ax=None,
    cmap=None,
    vmin=None,
    vmax=None,
    missing_color="#e8e8e8",
    edgecolor="#ffffff",
    linewidth=0.2,
    title=None,
    cbar_label=None,
    figsize=(8, 8),
):
    """Fill admin polygons by a per-``region`` value.

    Parameters
    ----------
    values : xr.DataArray
        Indexed by a ``region`` dim — the output of :func:`rosetta.zonal`. Its
        ``region`` labels are matched to ``geometries`` on ``by``.
    geometries : GeoDataFrame
        One row per region.
    by : str, optional
        Column of ``geometries`` holding the region key that matches
        ``values``'s ``region`` coordinate. Defaults to the GeoDataFrame index.
    missing_color : colour
        Fill for regions with no value (NaN) — a district the grid never
        covered. Drawn, not dropped, so the map has no holes.

    Returns
    -------
    matplotlib Figure
    """
    require_optional("matplotlib", _HINT)
    gpd = _require_geopandas()
    plt = _pyplot()

    region_dim = "region" if "region" in values.dims else values.dims[0]
    keys = values[region_dim].values
    frame = geometries.copy()
    if by is None:
        lookup = {i: float(v) for i, v in zip(range(len(frame)), np.asarray(values.values, float))}
        frame["_value"] = [lookup.get(i, np.nan) for i in range(len(frame))]
    else:
        if by not in frame.columns:
            raise ValueError(
                f"column {by!r} not found in geometries; have {list(frame.columns)}"
            )
        lookup = {k: float(v) for k, v in zip(keys, np.asarray(values.values, float))}
        frame["_value"] = frame[by].map(lookup)

    if frame.crs is not None and frame.crs.to_epsg() != 4326:
        frame = frame.to_crs("EPSG:4326")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    finite = frame["_value"].to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    looks_like_fraction = finite.size and finite.min() >= 0.0 and finite.max() <= 1.0
    if cmap is None:
        cmap = _PERCENTILE_CMAP if looks_like_fraction else "viridis"
    if vmin is None and looks_like_fraction:
        vmin, vmax = 0.0, 1.0

    frame.plot(
        column="_value", ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
        edgecolor=edgecolor, linewidth=linewidth, legend=True,
        legend_kwds={"label": cbar_label or (values.name or "value"),
                     "fraction": 0.046, "pad": 0.04},
        missing_kwds={"color": missing_color, "edgecolor": edgecolor,
                      "linewidth": linewidth, "label": "no data"},
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def _overlay_boundaries(ax, boundaries, ccrs):
    if boundaries is None:
        return
    gpd = _require_geopandas()
    frame = boundaries
    if hasattr(frame, "geometry"):
        if getattr(frame, "crs", None) is not None and frame.crs.to_epsg() != 4326:
            frame = frame.to_crs("EPSG:4326")
        kw = {"ax": ax, "facecolor": "none", "edgecolor": "#333333", "linewidth": 0.4}
        if ccrs is not None:
            kw["transform"] = ccrs.PlateCarree()
        frame.boundary.plot(**kw)


def _require_geopandas():
    try:
        return importlib.import_module("geopandas")
    except ImportError as e:
        raise ImportError(
            "Choropleths and boundary overlays need geopandas. "
            "Install the plotting + geo extras."
        ) from e
