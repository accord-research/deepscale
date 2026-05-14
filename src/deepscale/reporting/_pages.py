"""Private page-rendering primitives. Each primitive takes the open
PdfPages plus the data it needs and draws exactly one page.

Importing this module does NOT load matplotlib. Each function calls
`require_optional` at entry so callers without the [plotting] extra get
a clear ImportError instead of a confusing AttributeError later.
"""

# Per-metric colormap and value range. Single source of truth shared by
# both `map_grid_page` (single-method SVSLRF maps) and `comparison_map_grid_page`
# (multi-method comparison maps). When adding a new metric, add it here.
_METRIC_STYLE = {
    "rpss":       {"cmap": "RdBu",   "vmin": -1, "vmax": 1},
    "pearson_r":  {"cmap": "RdBu",   "vmin": -1, "vmax": 1},
    "spearman_r": {"cmap": "RdBu",   "vmin": -1, "vmax": 1},
    "roc_bn":     {"cmap": "viridis", "vmin": 0, "vmax": 1},
    "roc_nn":     {"cmap": "viridis", "vmin": 0, "vmax": 1},
    "roc_an":     {"cmap": "viridis", "vmin": 0, "vmax": 1},
    "reliability": {"cmap": "viridis_r", "vmin": 0, "vmax": 0.5},
    "rmse":       {"cmap": "viridis", "vmin": 0, "vmax": None},
    "hss":        {"cmap": "RdBu",   "vmin": -1, "vmax": 1},
    "groc":       {"cmap": "viridis", "vmin": 0.5, "vmax": 1},
}


def _style_for(metric):
    """Return (cmap, vmin, vmax) for a metric. Falls back to viridis/auto if unknown."""
    style = _METRIC_STYLE.get(metric, {"cmap": "viridis", "vmin": None, "vmax": None})
    return style["cmap"], style["vmin"], style["vmax"]


from .._optional import require_optional

_HINT = "pip install deepscale[plotting]"


def title_page(pdf, *, title, subtitle, metadata):
    """Title page: large title, optional subtitle, metadata key/value list.

    Only keys present in metadata are rendered; missing keys are silently
    skipped (no '(unknown <key>)' placeholders). Known keys are rendered in
    preferred order (region, target, init, predictand, method), then any
    extras follow in insertion order.
    """
    require_optional("matplotlib", _HINT)
    import matplotlib.pyplot as plt

    # Render only keys present in metadata; preserve preferred order for known
    # keys, then append any extras in insertion order.
    preferred_order = ("region", "target", "init", "predictand", "method")
    keys_in_order = [k for k in preferred_order if k in metadata]
    extras = [k for k in metadata.keys() if k not in preferred_order]
    keys_in_order.extend(extras)
    n_rows = len(keys_in_order)

    # Size the page to its content rather than wasting a full US-letter sheet.
    # title (0.5") + subtitle (0.5") + gap (0.5") + rows (0.45" each).
    content_height = 0.5 + 0.5 + 0.5 + n_rows * 0.45
    content_height = max(3.5, min(content_height, 11))  # clamp to [3.5", 11"]

    fig = plt.figure(figsize=(8.5, content_height))

    # Position everything from the top using inches → figure-fraction.
    def y_for_inch(top_inch):
        return 1.0 - (top_inch / content_height)

    fig.text(0.1, y_for_inch(0.7), title, fontsize=22, fontweight="bold")
    if subtitle:
        fig.text(0.1, y_for_inch(1.1), subtitle, fontsize=14, color="#444")

    top = 1.8
    for key in keys_in_order:
        value = metadata[key]
        fig.text(0.1, y_for_inch(top), f"{key.capitalize()}:", fontsize=11, fontweight="bold")
        fig.text(0.30, y_for_inch(top), str(value), fontsize=11)
        top += 0.45  # 0.45 inches per row

    pdf.savefig(fig)
    plt.close(fig)


def scalar_table_page(pdf, scores, *, title="Metrics summary"):
    """A single page showing metric → scalar values as a two-column table."""
    require_optional("matplotlib", _HINT)
    import matplotlib.pyplot as plt

    n_rows = len(scores) + 1  # +1 for header row
    content_height = 1.5 + 0.35 * n_rows  # title area + per-row height
    content_height = max(3.0, min(content_height, 11))  # clamp to [3.0", 11"]
    fig, ax = plt.subplots(figsize=(8.5, content_height))
    ax.set_axis_off()
    ax.set_title(title, fontsize=16, fontweight="bold", loc="left", pad=20)

    rows = [[k, f"{v:.4g}" if isinstance(v, (int, float)) else str(v)]
            for k, v in scores.items()]
    if not rows:
        ax.text(0.5, 0.5, "(no metrics)", ha="center", va="center", fontsize=12)
    else:
        table = ax.table(
            cellText=rows,
            colLabels=["Metric", "Value"],
            cellLoc="left",
            colLoc="left",
            loc="upper left",
            colWidths=[0.35, 0.25],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 1.2)

    pdf.savefig(fig)
    plt.close(fig)


def map_grid_page(pdf, maps, *, ncols=3):
    """One PDF page: grid of per-metric spatial maps.

    `maps` is {metric_name: xr.DataArray(lat, lon)}. Per-metric cmap/vrange
    pulled from _METRIC_STYLE for cross-panel comparability.
    """
    require_optional("matplotlib", _HINT)
    require_optional("cartopy", _HINT)
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    n = len(maps)
    if n == 0:
        return
    nrows = (n + ncols - 1) // ncols
    fig = plt.figure(figsize=(4 * ncols, 3.2 * nrows))
    for i, (name, da) in enumerate(maps.items()):
        cmap, vmin, vmax = _style_for(name)
        ax = fig.add_subplot(nrows, ncols, i + 1, projection=ccrs.PlateCarree())
        ax.coastlines(linewidth=0.6)
        # PlateCarree assumption matches deepscale's regional convention.
        im = ax.pcolormesh(
            da["lon"], da["lat"], da.values,
            transform=ccrs.PlateCarree(),
            cmap=cmap, vmin=vmin, vmax=vmax, shading="auto",
        )
        ax.set_title(name, fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def roc_page(pdf, roc_diagram):
    """One PDF page: ROC curves for the three terciles (BN/NN/AN).

    `roc_diagram` is {"bn": {"fpr","tpr","area"}, "nn": {...}, "an": {...}}
    as produced by ROCMetric.compute_diagram().
    """
    require_optional("matplotlib", _HINT)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))
    cat_colors = {"bn": "tab:red", "nn": "tab:grey", "an": "tab:blue"}
    cat_names = {"bn": "Below normal", "nn": "Normal", "an": "Above normal"}

    for cat in ("bn", "nn", "an"):
        entry = roc_diagram.get(cat)
        if entry is None:
            continue
        ax.plot(
            entry["fpr"], entry["tpr"],
            color=cat_colors[cat], linewidth=1.5,
            label=f"{cat_names[cat]} (area = {entry['area']:.2f})",
        )
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.6, label="No skill")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (per tercile)")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")

    pdf.savefig(fig)
    plt.close(fig)


def reliability_page(pdf, reliability_diagram):
    """One PDF page: reliability diagram for the three terciles.

    `reliability_diagram` is the list-of-dicts payload from
    ReliabilityMetric.compute_diagram().
    """
    require_optional("matplotlib", _HINT)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))
    cat_colors = {"bn": "tab:red", "nn": "tab:grey", "an": "tab:blue"}
    cat_names = {"bn": "Below normal", "nn": "Normal", "an": "Above normal"}

    for entry in reliability_diagram:
        cat = entry["tercile"]
        bins = entry["bins"]
        if not bins:
            continue
        xs = [b["mean_prob"] for b in bins]
        ys = [b["obs_freq"] for b in bins]
        # marker size scales with sample count, clipped for readability
        sizes = [min(80, 12 + b["n"]) for b in bins]
        ax.plot(xs, ys, "-", color=cat_colors.get(cat, "black"), linewidth=1.2)
        ax.scatter(xs, ys, s=sizes, color=cat_colors.get(cat, "black"),
                   label=cat_names.get(cat, cat))

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.6, label="Perfect calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Forecast probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability diagram")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")

    pdf.savefig(fig)
    plt.close(fig)


def heatmap_page(pdf, df, *, title="Skill comparison"):
    """One PDF page: methods × metrics scalar heatmap.

    `df` is a pandas DataFrame indexed by method, columns are metric names.
    Cells are scalars; NaN is rendered as a blank cell.
    """
    require_optional("matplotlib", _HINT)
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(df.columns) + 2),
                                    max(4, 0.5 * len(df.index) + 2)))
    data = df.values.astype(float)
    im = ax.imshow(data, aspect="auto", cmap="RdBu", vmin=-1, vmax=1)

    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_yticklabels(df.index)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

    # Cell annotations
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if abs(v) > 0.5 else "black")

    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def comparison_map_grid_page(pdf, metric, maps):
    """One PDF page: side-by-side spatial maps for the same metric across methods.

    `maps` is {method_name: xr.DataArray(lat, lon)}. Shared colorbar from
    _METRIC_STYLE so methods are visually comparable.
    """
    require_optional("matplotlib", _HINT)
    require_optional("cartopy", _HINT)
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    n = len(maps)
    if n == 0:
        return
    cmap, vmin, vmax = _style_for(metric)

    fig = plt.figure(figsize=(4 * n, 4))
    fig.suptitle(f"Spatial skill: {metric}", fontsize=14, fontweight="bold")
    im = None
    for i, (method_name, da) in enumerate(maps.items()):
        ax = fig.add_subplot(1, n, i + 1, projection=ccrs.PlateCarree())
        ax.coastlines(linewidth=0.6)
        im = ax.pcolormesh(
            da["lon"], da["lat"], da.values,
            transform=ccrs.PlateCarree(),
            cmap=cmap, vmin=vmin, vmax=vmax, shading="auto",
        )
        ax.set_title(method_name, fontsize=11)

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.18, 0.02, 0.65])
    fig.colorbar(im, cax=cbar_ax)

    pdf.savefig(fig)
    plt.close(fig)
