"""House-styled tercile maps: configurable palette, probability bins, masks.

The default :func:`plot_tercile_forecast` renders the generic IRI red/grey/blue
dominant-tercile map. Operational outlooks (ICPAC GHACOF, ACMAD) instead use a
*binned* dominant-category palette — each category has a ramp of colours keyed to
which probability band the dominant tercile falls in — plus a dry-season mask,
a country clip, and lakes. :class:`TercileStyle` captures that house style as
data (so it is auditable in the notebook, not hidden in the library), and
:func:`render_styled_terciles` / :func:`plot_tercile_comparison` render it.

Styling values live with the caller; this module only knows how to draw them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .._optional import require_optional

_HINT = "pip install deepscale[plotting]"

# Natural Earth admin-0 polygons + lakes, as cached by cartopy (optional; the
# country clip / lakes / boundary overlays degrade gracefully if absent).
_NE_ROOT = Path.home() / ".local" / "share" / "cartopy" / "shapefiles" / "natural_earth"
_NE_COUNTRIES = _NE_ROOT / "cultural" / "ne_50m_admin_0_countries.shp"
_NE_LAKES = _NE_ROOT / "physical" / "ne_50m_lakes.shp"


@dataclass
class TercileStyle:
    """A binned dominant-tercile map style (palette + masks + clip).

    ``below_colors`` / ``normal_colors`` / ``above_colors`` are equal-length
    ramps of colours (hex or RGB triples), from the lowest to the highest
    probability band. ``prob_bins`` are the ascending probability breakpoints in
    *percent*; there must be exactly one more bin than there are colours (each
    colour paints the interval ``[bins[i], bins[i+1])``). The dominant category's
    probability selects the band, and the category selects the ramp.

    Optional overlays:
      * ``dry_mask``  – bool ``(lat, lon)`` DataArray, True where the season is
        climatologically dry; those cells are painted ``dry_color``.
      * ``clip_to``   – list of country names; cells outside them become nodata.
      * ``lakes``     – draw lakes in ``lake_color``.
      * ``nodata_color`` – cells with no valid forecast (or clipped out).
    """
    below_colors: list
    normal_colors: list
    above_colors: list
    prob_bins: list
    dry_mask: object = None
    clip_to: object = None
    lakes: bool = False
    dry_color: str = "#bebebe"
    lake_color: str = "#78b8f8"
    nodata_color: str = "#ffffff"
    boundary_color: str = "#333333"

    def __post_init__(self):
        n = len(self.below_colors)
        if not (len(self.normal_colors) == len(self.above_colors) == n):
            raise ValueError("below/normal/above colour ramps must be equal length")
        if len(self.prob_bins) != n + 1:
            raise ValueError(
                f"prob_bins must have one more entry than colours "
                f"(got {len(self.prob_bins)} bins for {n} colours)")

    def ramps_rgb(self):
        """The three ramps as (n, 3) float arrays in 0-1."""
        import matplotlib.colors as mcolors
        return [np.array([mcolors.to_rgb(c) for c in ramp])
                for ramp in (self.below_colors, self.normal_colors, self.above_colors)]


def _binned_rgb(probs, lat, lon, style):
    """Dominant-category, probability-binned RGB image for imshow (origin upper).

    ``probs`` is ``(3, ny, nx)`` fractional probabilities (below/normal/above).
    """
    import matplotlib.colors as mcolors

    pct = probs * 100.0
    stack = np.stack([np.nan_to_num(pct[k], nan=-1.0) for k in range(3)])
    dom = np.argmax(stack, axis=0)
    domp = np.nanmax(np.where(np.isfinite(pct), pct, np.nan), axis=0)
    valid = np.isfinite(domp)

    bins = np.asarray(style.prob_bins, float)
    ramps = style.ramps_rgb()
    ncol = len(ramps[0])
    idx = np.clip(np.digitize(domp, bins) - 1, 0, ncol - 1)

    rgb = np.empty(dom.shape + (3,))
    rgb[:] = mcolors.to_rgb(style.nodata_color)
    for k in range(3):
        sel = valid & (dom == k)
        rgb[sel] = ramps[k][idx[sel]]

    if style.dry_mask is not None:
        dm = style.dry_mask
        dm = dm.interp(lat=lat, lon=lon, method="nearest") if (
            dm.sizes.get("lat") != len(lat) or dm.sizes.get("lon") != len(lon)) else dm
        dmv = np.asarray(dm.transpose("lat", "lon").values, bool)
        rgb[dmv] = mcolors.to_rgb(style.dry_color)

    # descending-lat orientation for imshow(origin="upper")
    order = np.argsort(lat)
    return rgb[order][::-1], (float(lon.min()), float(lon.max()),
                             float(lat.min()), float(lat.max()))


def _country_gdf(names=None):
    """Natural Earth admin-0 polygons (optionally filtered to `names`), or None."""
    try:
        import importlib
        gpd = importlib.import_module("geopandas")
    except Exception:
        return None
    if not _NE_COUNTRIES.exists():
        return None
    gdf = gpd.read_file(_NE_COUNTRIES)
    if names:
        col = next((c for c in ("NAME", "ADMIN", "NAME_LONG", "SOVEREIGNT")
                    if c in gdf.columns), None)
        if col is not None:
            want = set(names)
            gdf = gdf[gdf[col].isin(want) | gdf.get("ADMIN", gdf[col]).isin(want)]
    return gdf


def render_styled_terciles(ax, probs, style, *, title=None, small=False):
    """Draw a binned dominant-tercile map with `style` onto `ax`.

    ``probs`` is a ``(tercile, lat, lon)`` fractional-probability DataArray.
    Returns ``ax``. Country clip / lakes / boundaries are best-effort (drawn when
    geopandas + cached Natural Earth shapefiles are available, else skipped).
    """
    from .._spatial import spatial_dims
    lat_dim, lon_dim = spatial_dims(probs, context="render_styled_terciles")
    p = probs.transpose("tercile", lat_dim, lon_dim)
    lat = p[lat_dim].values
    lon = p[lon_dim].values
    rgb, extent = _binned_rgb(p.values, lat, lon, style)

    ax.imshow(rgb, extent=extent, origin="upper", interpolation="nearest",
              aspect="equal", zorder=1)

    gdf = _country_gdf(style.clip_to)
    if gdf is not None:
        gdf.boundary.plot(ax=ax, color=style.boundary_color,
                          linewidth=0.3 if small else 0.6, zorder=4)
    if style.lakes and _NE_LAKES.exists():
        try:
            import importlib
            gpd = importlib.import_module("geopandas")
            gpd.read_file(_NE_LAKES).cx[extent[0]:extent[1], extent[2]:extent[3]].plot(
                ax=ax, color=style.lake_color, linewidth=0, zorder=3)
        except Exception:
            pass

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    if small:
        ax.set_xticks([]); ax.set_yticks([])
    else:
        ax.set_xlabel("lon"); ax.set_ylabel("lat")
    if title:
        ax.set_title(title)
    return ax


def _tilt(probs):
    """P(above) - P(below) in percentage points, for the difference panel."""
    return (probs.sel(tercile=2) - probs.sel(tercile=0)) * 100.0


def plot_tercile_comparison(ours, reference, *, style, axes=None,
                            diff_cmap="RdBu", diff_limit=40.0, labels=None):
    """One row: our forecast, the reference, and their signed tercile-tilt diff.

    ``ours`` and ``reference`` are ``(tercile, lat, lon)`` fractional-probability
    DataArrays. ``axes`` is an iterable of 3 axes (created if None). The third
    panel shows ``P(above)-P(below)`` of ours minus reference (percentage points),
    in ``diff_cmap`` clamped to ``±diff_limit``. Returns ``(fig, diff_image)``.
    """
    import importlib
    require_optional("matplotlib", _HINT)
    plt = importlib.import_module("matplotlib.pyplot")

    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    else:
        fig = axes[0].figure
    lab = labels or ("forecast", "reference", "difference")

    render_styled_terciles(axes[0], ours, style, title=lab[0])
    render_styled_terciles(axes[1], reference, style, title=lab[1])

    ref_i = reference.interp(lat=ours.lat.values, lon=ours.lon.values)
    diff = (_tilt(ours) - _tilt(ref_i)).transpose("lat", "lon")
    lat = ours.lat.values; lon = ours.lon.values
    order = np.argsort(lat)
    img = diff.values[order][::-1]
    extent = (float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max()))
    diff_im = axes[2].imshow(img, extent=extent, origin="upper", cmap=diff_cmap,
                             vmin=-diff_limit, vmax=diff_limit, aspect="equal")
    gdf = _country_gdf(style.clip_to)
    if gdf is not None:
        gdf.boundary.plot(ax=axes[2], color=style.boundary_color, linewidth=0.4, zorder=4)
    axes[2].set_xlim(extent[0], extent[1]); axes[2].set_ylim(extent[2], extent[3])
    axes[2].set_title(lab[2])
    return fig, diff_im
