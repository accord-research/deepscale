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


def _tercile_codes(probs, prob_bins):
    """Integer class code per cell for a discrete tercile palette.

    ``probs`` is (tercile=3, lat, lon), tercile order (below, normal, above),
    fractional (0-1). Returns (code, valid). n = len(prob_bins)-1 bins:
    above -> 0..n-1, normal -> n..2n-1, below -> 2n..3n-1; -1 = no valid triple.
    """
    valid = np.isfinite(probs).all(axis=0)
    dom = np.argmax(np.where(np.isfinite(probs), probs, -1.0), axis=0)  # 0=below 1=normal 2=above
    prob_pct = np.where(valid, np.max(probs, axis=0) * 100.0, np.nan)
    n = len(prob_bins) - 1
    pbin = np.clip(np.digitize(prob_pct, prob_bins) - 1, 0, n - 1)
    base = {2: 0, 1: n, 0: 2 * n}
    code = np.full(prob_pct.shape, -1, dtype=int)
    for d, b in base.items():
        sel = valid & (dom == d)
        code[sel] = b + pbin[sel]
    return code, valid


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


def _discrete_cmap(style):
    import importlib
    mcolors = importlib.import_module("matplotlib.colors")
    palette = list(style.above_colors) + list(style.normal_colors) + list(style.below_colors) + [style.dry_color]
    cmap = mcolors.ListedColormap(palette)
    cmap.set_bad(style.nodata_color)
    return cmap, len(palette)


def _tercile_style_legend(ax, style, below_label, above_label):
    import importlib
    Patch = importlib.import_module("matplotlib.patches").Patch

    def _weak(colors):
        return colors[1] if len(colors) > 1 else colors[-1]

    handles = [
        Patch(facecolor="none", edgecolor="none", label=r"$\bf{Above}$"),
        Patch(facecolor=style.above_colors[-1], label="strong"),
        Patch(facecolor=_weak(style.above_colors), label="weak"),
        Patch(facecolor=_weak(style.normal_colors), edgecolor="0.6", label="Near normal"),
        Patch(facecolor="none", edgecolor="none", label=r"$\bf{Below}$"),
        Patch(facecolor=style.below_colors[-1], label="strong"),
        Patch(facecolor=_weak(style.below_colors), label="weak"),
        Patch(facecolor=style.dry_color, label="Dry season"),
    ]
    if style.lakes:
        handles.append(Patch(facecolor=style.lake_color, label="Lake"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncol=4, frameon=True, framealpha=0.9, fontsize=7,
              title="Probability of category", title_fontsize=8)


_COUNTRY_GEOM_CACHE = {}


def _country_geometry(names):
    """Prepared union geometry of Natural Earth admin_0 countries matching `names`."""
    import importlib
    shpreader = importlib.import_module("cartopy.io.shapereader")
    unary_union = importlib.import_module("shapely.ops").unary_union
    prep = importlib.import_module("shapely.prepared").prep
    key = tuple(sorted(names))
    if key not in _COUNTRY_GEOM_CACHE:
        rdr = shpreader.Reader(shpreader.natural_earth(
            resolution="10m", category="cultural", name="admin_0_countries"))
        geoms = [r.geometry for r in rdr.records()
                 if r.attributes.get("NAME", "") in names
                 or r.attributes.get("NAME_LONG", "") in names]
        if not geoms:
            raise ValueError(f"No Natural Earth countries matched names={names!r}")
        # buffer in degrees to include coastal border cells
        _COUNTRY_GEOM_CACHE[key] = prep(unary_union(geoms).buffer(0.3))
    return _COUNTRY_GEOM_CACHE[key]


def _region_masks(lat, lon, style):
    """Boolean (nlat, nlon) region masks derived from a ``TercileStyle``.

    Returns ``(dry, outside)``; each is a bool array, or ``None`` if the
    corresponding style field is unset:
      - ``dry``: cells to grey out, from ``style.dry_mask``.
      - ``outside``: cells outside ``style.clip_to``.

    ``style.dry_mask`` may be:
      - a coordinate-bearing DataArray (has ``lat``/``lon`` coords): aligned to
        the plotted grid by nearest-neighbor interpolation on coordinate VALUE,
        so it lands on the correct geographic cells regardless of the mask's
        resolution, registration offset, or latitude ordering. Cells outside the
        mask's coverage (NaN after interpolation) are treated as not-dry.
      - a bare ndarray (no coords): must match the grid shape exactly, since there
        is no coordinate information to align it by; a mismatch raises a clear
        ``ValueError`` instead of a raw positional-indexing failure.

    Shared by ``_apply_style_masks`` (tercile codes) and ``plot_field``
    (continuous fields) so both mask identically.
    """
    import importlib
    lat = np.asarray(lat); lon = np.asarray(lon)
    shape = (lat.shape[0], lon.shape[0])
    dry = None
    if style is not None and style.dry_mask is not None:
        if hasattr(style.dry_mask, "interp"):
            dm = style.dry_mask.astype(float).interp(
                lat=lat, lon=lon, method="nearest").transpose("lat", "lon")
            dry = np.asarray(dm.values) > 0.5   # NaN (outside coverage) -> not dry
        else:
            dry = np.asarray(style.dry_mask, dtype=bool)
            if dry.shape != shape:
                raise ValueError(
                    f"dry_mask ndarray shape {dry.shape} does not match the field "
                    f"shape {shape}; pass a coordinate-bearing xarray DataArray "
                    f"(lat/lon) to auto-align, or match the grid."
                )
    outside = None
    if style is not None and style.clip_to is not None:
        Point = importlib.import_module("shapely.geometry").Point
        geom = (style.clip_to if not isinstance(style.clip_to, (list, tuple))
                else _country_geometry(list(style.clip_to)))
        # Normalize lon to -180..180 for Natural Earth containment tests.
        lon180 = ((lon + 180) % 360) - 180
        inside = np.zeros(shape, dtype=bool)
        for i, la in enumerate(lat):
            for j, lo in enumerate(lon180):
                if geom.contains(Point(float(lo), float(la))):
                    inside[i, j] = True
        outside = ~inside
    return dry, outside


def _apply_style_masks(code, lat, lon, style):
    """Grey dry cells (code 3n) and mask cells outside the clip geometry (-1)."""
    n = len(style.prob_bins) - 1
    dry, outside = _region_masks(lat, lon, style)
    if dry is not None:
        # Dry cells are greyed unconditionally (matches the GHACOF reference); the
        # clip below then restricts to the domain.
        code[dry] = 3 * n
    if outside is not None:
        code[outside] = -1
    return code


def plot_tercile_forecast(pr_fcst, *, style=None, ax=None, title=None,
                          variable_kind="precip", legend=True):
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

    Lakes (`style.lakes`) are drawn only on the cartopy/geo path (when cartopy
    is available); the geopandas fallback does not draw lakes.

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
    values = pr_fcst.transpose("tercile", lat_dim, lon_dim).values
    lon = pr_fcst[lon_dim].values
    lat = pr_fcst[lat_dim].values
    extent = (float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max()))
    if style is not None and style.extent is not None:
        extent = tuple(style.extent)

    if ax is None:
        fig, ax, is_geo = _make_tercile_axes(plt, extent, figsize=(8, 5.5))
    else:
        fig = ax.figure
        is_geo = hasattr(ax, "coastlines")

    if style is None:
        rgb = _tercile_rgb(values, red_cat, blue_cat)
        if is_geo:
            ccrs = importlib.import_module("cartopy.crs")
            ax.imshow(rgb, extent=extent, origin="lower", transform=ccrs.PlateCarree())
            _draw_cartopy_basemap(ax)
        else:
            ax.imshow(rgb, extent=extent, origin="lower", aspect="auto", zorder=1)
            drew = _draw_geopandas_basemap(ax, extent)
            ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
            ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
            if drew:
                ax.grid(color="#777777", linewidth=0.3, alpha=0.5)
    else:
        code, _ = _tercile_codes(values, style.prob_bins)
        cmap, npal = _discrete_cmap(style)
        code = _apply_style_masks(code, lat, lon, style)
        code_masked = np.ma.masked_less(code, 0)
        if is_geo:
            ccrs = importlib.import_module("cartopy.crs")
            ax.pcolormesh(lon, lat, code_masked, cmap=cmap, vmin=0, vmax=npal - 1,
                          shading="auto", transform=ccrs.PlateCarree())
            _draw_cartopy_basemap(ax)
            if style.lakes and is_geo:
                cfeature = importlib.import_module("cartopy.feature")
                ax.add_feature(cfeature.NaturalEarthFeature("physical", "lakes", "10m"),
                               facecolor=style.lake_color, edgecolor="none", zorder=3)
        else:
            ax.pcolormesh(lon, lat, code_masked, cmap=cmap, vmin=0, vmax=npal - 1, shading="auto")
            ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])

    ax.set_title(title or "Dominant tercile probability")
    if legend and style is None:
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
    elif legend and style is not None:
        _tercile_style_legend(ax, style, below_label, above_label)
    return fig


def plot_field(field, *, style=None, ax=None, cmap="RdBu_r", vmin=None, vmax=None,
               center=None, title=None, grey_dry=True):
    """Continuous (lat, lon) field on the same styled basemap as ``plot_terciles``.

    Draws ``field`` with ``pcolormesh`` using the identical map extent,
    coastlines / borders / gridlines, dry-cell greying, country clip, and lakes
    that ``plot_terciles`` applies for the same ``style`` -- so a difference or
    anomaly panel lines up cell-for-cell with the tercile panels beside it. The
    colors are the caller's (any Matplotlib ``cmap``); the framing and masks come
    entirely from ``style``. Region-agnostic: nothing here encodes a region.

    ``cmap`` / ``vmin`` / ``vmax`` are the usual diverging-map controls; pass
    ``center`` instead to anchor a ``TwoSlopeNorm`` at a value (e.g. 0). Set
    ``grey_dry=False`` to leave dry cells transparent rather than greyed.

    Returns the Matplotlib mappable, for ``fig.colorbar``.
    """
    import importlib
    require_optional("matplotlib", _HINT)
    plt = importlib.import_module("matplotlib.pyplot")
    from .._spatial import spatial_dims

    lat_dim, lon_dim = spatial_dims(field, context="plot_field")
    fld = field.transpose(lat_dim, lon_dim)
    lat = np.asarray(fld[lat_dim].values)
    lon = np.asarray(fld[lon_dim].values)
    values = np.array(fld.values, dtype=float)
    extent = (float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max()))
    if style is not None and style.extent is not None:
        extent = tuple(style.extent)

    if ax is None:
        fig, ax, is_geo = _make_tercile_axes(plt, extent, figsize=(8, 5.5))
    else:
        fig = ax.figure
        is_geo = hasattr(ax, "coastlines")
        if is_geo:
            ccrs = importlib.import_module("cartopy.crs")
            ax.set_extent(extent, crs=ccrs.PlateCarree())

    dry, outside = _region_masks(lat, lon, style) if style is not None else (None, None)
    if outside is not None:
        values[outside] = np.nan            # outside the clip -> nodata (transparent)
    if grey_dry and dry is not None:
        values[dry] = np.nan                # dry -> drawn as grey by the overlay below
    masked = np.ma.masked_invalid(values)

    norm = None
    if center is not None:
        TwoSlopeNorm = importlib.import_module("matplotlib.colors").TwoSlopeNorm
        lo = vmin if vmin is not None else float(np.nanmin(values))
        hi = vmax if vmax is not None else float(np.nanmax(values))
        norm = TwoSlopeNorm(vmin=lo, vcenter=center, vmax=hi)
    Colormap = importlib.import_module("matplotlib.colors").Colormap
    cmap_obj = cmap.copy() if isinstance(cmap, Colormap) else plt.get_cmap(cmap).copy()
    if style is not None:
        cmap_obj.set_bad(style.nodata_color)   # clipped/outside cells render as nodata
    kw = dict(cmap=cmap_obj, shading="auto")
    if norm is not None:
        kw["norm"] = norm
    else:
        kw["vmin"] = vmin; kw["vmax"] = vmax

    def _grey_overlay():
        if not (grey_dry and dry is not None):
            return
        ListedColormap = importlib.import_module("matplotlib.colors").ListedColormap
        cells = (dry & ~outside) if outside is not None else dry
        grey = np.where(cells, 1.0, np.nan)
        gkw = dict(cmap=ListedColormap([style.dry_color]), vmin=0, vmax=1,
                   shading="auto", zorder=2)
        if is_geo:
            ccrs = importlib.import_module("cartopy.crs")
            ax.pcolormesh(lon, lat, np.ma.masked_invalid(grey),
                          transform=ccrs.PlateCarree(), **gkw)
        else:
            ax.pcolormesh(lon, lat, np.ma.masked_invalid(grey), **gkw)

    if is_geo:
        ccrs = importlib.import_module("cartopy.crs")
        im = ax.pcolormesh(lon, lat, masked, transform=ccrs.PlateCarree(), **kw)
        _grey_overlay()
        _draw_cartopy_basemap(ax)
        if style is not None and style.lakes:
            cfeature = importlib.import_module("cartopy.feature")
            ax.add_feature(cfeature.NaturalEarthFeature("physical", "lakes", "10m"),
                           facecolor=style.lake_color, edgecolor="none", zorder=3)
    else:
        im = ax.pcolormesh(lon, lat, masked, **kw)
        _grey_overlay()
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])

    ax.set_title(title or "")
    return im


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
