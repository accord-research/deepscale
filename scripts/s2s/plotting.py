"""Matplotlib plot helpers for the S2S dashboard.

Pure functions: take data + minimal kwargs, return a matplotlib Figure.
The orchestrator (render_dashboard.py) handles file I/O and layout.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; required for headless rendering
import matplotlib.pyplot as plt

# Dark theme matching site/theme.css so rendered PNGs blend into the dark
# dashboard page (and look right in GitHub's image viewer too). Applied to
# matplotlib's global rcParams at import — this module is dashboard-only.
_DARK_BG = "#1a1a1a"      # site/theme.css --bg
_DARK_FG = "#e5e5e5"      # --text
_DARK_MUTED = "#9ca3af"   # --muted
_DARK_BORDER = "#2a2a2a"  # --border

_DARK_RC = {
    "figure.facecolor": _DARK_BG,
    "figure.edgecolor": _DARK_BG,
    "savefig.facecolor": _DARK_BG,
    "savefig.edgecolor": _DARK_BG,
    "axes.facecolor": _DARK_BG,
    "axes.edgecolor": _DARK_BORDER,
    "axes.labelcolor": _DARK_FG,
    "axes.titlecolor": _DARK_FG,
    "text.color": _DARK_FG,
    "xtick.color": _DARK_MUTED,
    "ytick.color": _DARK_MUTED,
    "xtick.labelcolor": _DARK_FG,
    "ytick.labelcolor": _DARK_FG,
    "grid.color": _DARK_BORDER,
    "legend.facecolor": _DARK_BG,
    "legend.edgecolor": _DARK_BORDER,
    "legend.labelcolor": _DARK_FG,
}


def apply_dark_theme() -> None:
    """Apply the dashboard dark palette (see site/theme.css) to matplotlib globally."""
    plt.rcParams.update(_DARK_RC)


apply_dark_theme()

import numpy as np
import xarray as xr


_AMOUNT_CMAP = "YlGnBu"  # sequential precip: light = dry, blue-green = wet
_DIFF_CMAP = "BrBG"      # diverging anomaly: brown = drier, teal = wetter (centered on 0)


def _pending_panel(ax, label: str) -> None:
    """Placeholder for an obs/difference cell whose observation hasn't landed yet.

    Keeps the 3-column layout intact (so obs-pending issuances don't collapse to a
    narrow strip) and labels the empty space instead of leaving it blank.
    """
    ax.text(
        0.5, 0.5, label, ha="center", va="center",
        fontsize=9, color=_DARK_MUTED, transform=ax.transAxes,
    )


def comparison_grid(
    obs: xr.Dataset | None,
    methods: Mapping[str, xr.Dataset],
    *,
    dekad_label: str,
) -> plt.Figure:
    """Per-method spatial comparison.

    ``methods``: ordered mapping {method_name: Dataset with a 'mean' var on (lat, lon)}.
    ``obs``: Dataset with 'mean', or None when the observation for this dekad isn't
    archived yet. Each method gets a row of ``observed | forecast | (forecast - obs)``
    — sequential amounts on a shared scale plus a diverging difference panel centered
    on zero. The layout is always three columns: when obs hasn't landed yet, the
    observed and difference cells show a "pending" placeholder rather than collapsing
    to a single narrow column.
    """
    names = list(methods)
    n = len(names)
    has_obs = obs is not None
    # Always observed | forecast | difference. Holding the 3-column shape even for
    # obs-pending issuances keeps the most-recent (default) view from collapsing
    # into a narrow vertical strip that reads as wrongly oriented.
    ncols = 3
    fig, axes = plt.subplots(
        n, ncols, figsize=(3.4 * ncols, 3.0 * n), constrained_layout=True, squeeze=False
    )

    # Shared amounts scale across obs + every forecast.
    amount_fields = [methods[k]["mean"].values for k in names]
    if has_obs:
        amount_fields.append(obs["mean"].values)
    finite = np.concatenate([f.ravel() for f in amount_fields])
    finite = finite[np.isfinite(finite)]
    amax = float(np.percentile(finite, 98)) if finite.size else 1.0

    # Shared symmetric difference scale (centered on zero).
    dmax = 1.0
    if has_obs:
        obs_vals = obs["mean"].values
        diff_all = np.concatenate(
            [(methods[k]["mean"].values - obs_vals).ravel() for k in names]
        )
        diff_all = diff_all[np.isfinite(diff_all)]
        if diff_all.size:
            dmax = float(np.percentile(np.abs(diff_all), 98)) or 1.0

    amesh = dmesh = None
    for i, name in enumerate(names):
        fc = methods[name]["mean"]
        ax_obs, ax_fc, ax_diff = axes[i, 0], axes[i, 1], axes[i, 2]

        # Forecast always occupies the middle column, so the column meaning is
        # identical whether or not obs is present.
        amesh = ax_fc.pcolormesh(
            fc["lon"], fc["lat"], fc.values,
            cmap=_AMOUNT_CMAP, vmin=0, vmax=amax, shading="auto",
        )
        if has_obs:
            ob = obs["mean"]
            ax_obs.pcolormesh(
                ob["lon"], ob["lat"], ob.values,
                cmap=_AMOUNT_CMAP, vmin=0, vmax=amax, shading="auto",
            )
            dmesh = ax_diff.pcolormesh(
                fc["lon"], fc["lat"], fc.values - ob.values,
                cmap=_DIFF_CMAP, vmin=-dmax, vmax=dmax, shading="auto",
            )
        else:
            _pending_panel(ax_obs, "observed\n(pending)")
            _pending_panel(ax_diff, "difference\n(pending)")

        if i == 0:
            ax_obs.set_title("observed (CHIRPS)", fontsize=10)
            ax_fc.set_title("forecast", fontsize=10)
            ax_diff.set_title("difference (fcst − obs)", fontsize=10)

        # Method name as a horizontal row label (rotation=0) so it reads
        # left-to-right rather than sideways up the y-axis.
        ax_obs.set_ylabel(
            name, rotation=0, ha="right", va="center",
            fontsize=11, fontweight="bold", labelpad=12,
        )
        for ax in (ax_obs, ax_fc, ax_diff):
            ax.set_xticks([])
            ax.set_yticks([])

    allax = axes.ravel().tolist()
    if has_obs:
        fig.colorbar(amesh, ax=allax, location="right", shrink=0.38, anchor=(0.0, 0.80),
                     label="precip (mm/day)")
        fig.colorbar(dmesh, ax=allax, location="right", shrink=0.38, anchor=(0.0, 0.18),
                     label="Δ mm/day  brown=drier · teal=wetter")
    else:
        fig.colorbar(amesh, ax=allax, location="right", shrink=0.6,
                     label="precip (mm/day)")

    fig.suptitle(f"Comparison by method — {dekad_label}", fontsize=13)
    return fig


_METRIC_LABELS: dict[str, tuple[str, str, str]] = {
    # metric_key: (display_title, description, units)
    "acc": ("ACC", "Anomaly Correlation (spatial)", "[-1, 1]"),
    "rmse": ("RMSE", "Root Mean Squared Error", "mm/day"),
    "bias": ("Bias", "Mean(forecast) − Mean(obs)", "mm/day"),
    "rpss": ("RPSS", "Ranked Probability Skill Score vs equiprobable", "[-∞, 1]"),
}

# Metrics where 0 is a meaningful baseline (worth drawing a dashed zero line).
_SKILL_ZERO_LINE = {"acc", "bias", "rpss"}


def metrics_panel(
    scores: Sequence[Mapping],
    *,
    country: str,
    metrics: Sequence[str] = ("acc", "rmse", "bias", "rpss"),
) -> plt.Figure:
    """Render one subplot per metric showing each method's series over dekads."""
    if not scores:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, f"No verification scores yet for {country}",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    # Group scores by method, sorted by target_dekad.
    by_method: dict[str, list[Mapping]] = {}
    for r in scores:
        by_method.setdefault(r["method"], []).append(r)
    for series in by_method.values():
        series.sort(key=lambda r: r["target_dekad"])

    available = [m for m in metrics if any(m in r for s in by_method.values() for r in s)]
    n = max(len(available), 1)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.8 * n), constrained_layout=True, squeeze=False)

    for ax, metric in zip(axes[:, 0], available):
        title, desc, units = _METRIC_LABELS.get(metric, (metric.upper(), "", ""))
        for method_name, series in sorted(by_method.items()):
            xs = [r["target_dekad"] for r in series if metric in r]
            ys = [r[metric] for r in series if metric in r]
            if xs:
                ax.plot(xs, ys, marker="o", linewidth=1.5, label=method_name)
        if metric in _SKILL_ZERO_LINE:
            ax.axhline(0, color="#888", linewidth=0.8, linestyle="--", zorder=0)
        ax.set_title(f"{title} — {desc}", fontsize=11, fontweight="bold", loc="left")
        ax.set_ylabel(units, fontsize=10)
        ax.tick_params(axis="x", rotation=30, labelsize=9)
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(True, alpha=0.25)

    # One shared legend for the whole figure, placed OUTSIDE the axes (bottom
    # strip) — every subplot has the same methods, and loc="best" per-axes used
    # to land the legend on top of the data when lines span the full width.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="outside lower center",
                   ncols=min(len(by_method), 5), fontsize=9)

    fig.suptitle(f"S2S testbed metrics — {country}", fontsize=13, fontweight="bold")
    return fig
