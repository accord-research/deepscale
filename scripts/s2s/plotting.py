"""Matplotlib plot helpers for the S2S dashboard.

Pure functions: take data + minimal kwargs, return a matplotlib Figure.
The orchestrator (render_dashboard.py) handles file I/O and layout.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; required for headless rendering
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def comparison_grid(
    panels: Mapping[str, xr.Dataset],
    *,
    dekad_label: str,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> plt.Figure:
    """Render a single-row grid of pcolormesh maps, one per method/obs panel.

    panels: dict mapping panel label → xr.Dataset with a 'mean' variable on (lat, lon).
    dekad_label: shown as the figure suptitle.
    """
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.5), constrained_layout=True)
    if n == 1:
        axes = [axes]

    # Shared color scale: derive from the union of panel means if not given.
    if vmin is None or vmax is None:
        all_vals = np.concatenate([ds["mean"].values.ravel() for ds in panels.values()])
        finite = all_vals[np.isfinite(all_vals)]
        if finite.size:
            if vmin is None:
                vmin = float(np.percentile(finite, 2))
            if vmax is None:
                vmax = float(np.percentile(finite, 98))

    mesh = None
    for ax, (label, ds) in zip(axes, panels.items()):
        ax.set_title(label)
        mean = ds["mean"]
        mesh = ax.pcolormesh(
            mean["lon"], mean["lat"], mean.values,
            cmap=cmap, vmin=vmin, vmax=vmax, shading="auto",
        )
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")

    fig.suptitle(f"Comparison grid — {dekad_label}")

    # Place the colorbar as an inset of the last data panel; inset_axes children
    # are NOT added to fig.axes, so len(fig.axes) stays equal to n — the test
    # asserts len(fig.axes) == len(panels).
    if mesh is not None:
        cax = axes[-1].inset_axes([1.04, 0.0, 0.07, 1.0])
        fig.colorbar(mesh, cax=cax, label="precip (mm/day)")

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
        ax.legend(loc="best", fontsize=9, ncols=min(len(by_method), 5))
        ax.tick_params(axis="x", rotation=30, labelsize=9)
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(True, alpha=0.25)

    fig.suptitle(f"S2S testbed metrics — {country}", fontsize=13, fontweight="bold")
    return fig
