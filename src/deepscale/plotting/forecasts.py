"""Forecast output plots: tercile, deterministic, exceedance, flex-PDF."""

import numpy as np
from .._optional import require_optional


_HINT = "pip install deepscale[plotting]"

# Dominant-tercile color saturation: probability above 1/3 at which
# the color reaches full intensity. 0.37 = 70% probability cap.
_TERCILE_SAT = 0.37


def _new_fig(ax, figsize=(8, 5)):
    import importlib
    require_optional("matplotlib", _HINT)
    plt = importlib.import_module("matplotlib.pyplot")
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    return plt, fig, ax


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

    A legend in the lower-right corner shows the three categories.

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

    plt, fig, ax = _new_fig(ax)
    import importlib
    Patch = importlib.import_module("matplotlib.patches").Patch

    probs = pr_fcst.values  # (3, lat, lon)
    dom_cat = probs.argmax(axis=0)
    dom_prob = probs.max(axis=0)
    intensity = np.clip((dom_prob - 1 / 3) / _TERCILE_SAT, 0.0, 1.0)

    rgb = np.ones(dom_cat.shape + (3,))
    is_red = dom_cat == red_cat
    is_blue = dom_cat == blue_cat
    is_normal = dom_cat == 1
    rgb[is_red] = np.stack([
        np.ones(is_red.sum()),
        1 - intensity[is_red],
        1 - intensity[is_red],
    ], axis=-1)
    rgb[is_blue] = np.stack([
        1 - intensity[is_blue],
        1 - intensity[is_blue],
        np.ones(is_blue.sum()),
    ], axis=-1)
    rgb[is_normal] = np.stack([
        1 - 0.4 * intensity[is_normal],
        1 - 0.4 * intensity[is_normal],
        1 - 0.4 * intensity[is_normal],
    ], axis=-1)

    lon = pr_fcst.lon.values
    lat = pr_fcst.lat.values
    ax.imshow(
        rgb,
        extent=(lon.min(), lon.max(), lat.min(), lat.max()),
        origin="lower",
        aspect="auto",
    )
    ax.set_xlabel("Lon")
    ax.set_ylabel("Lat")
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
        loc="lower right",
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
