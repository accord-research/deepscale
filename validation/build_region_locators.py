#!/usr/bin/env python3
"""Build high-resolution locator maps for the validation report."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from shapely.geometry import box


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "validation" / "figures"
NE = Path.home() / ".local/share/cartopy/shapefiles/natural_earth"
LAND = NE / "physical/ne_110m_land.shp"
ADMIN0_LINES = NE / "cultural/ne_10m_admin_0_boundary_lines_land.shp"
ADMIN1 = NE / "cultural/ne_10m_admin_1_states_provinces.shp"


STYLE = {
    "land": "#e6eee8",
    "boundary": "#7f9188",
    "admin": "#aebbb6",
    "focus": "#dce9df",
    "domain_fill": "#c96a5f",
    "domain_edge": "#a8322b",
    "grid": "#d7dfdb",
    "text": "#24352f",
}


def _read_layers():
    land = gpd.read_file(LAND).to_crs("EPSG:4326")
    admin0 = gpd.read_file(ADMIN0_LINES).to_crs("EPSG:4326")
    admin1 = gpd.read_file(ADMIN1).to_crs("EPSG:4326")
    return land, admin0, admin1


def _crop(gdf: gpd.GeoDataFrame, extent: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    min_lon, max_lon, min_lat, max_lat = extent
    return gdf[gdf.intersects(box(min_lon, min_lat, max_lon, max_lat))]


def _draw_base(ax, land, admin0, admin1, extent):
    min_lon, max_lon, min_lat, max_lat = extent
    bounds = box(min_lon, min_lat, max_lon, max_lat)
    _crop(land, extent).plot(ax=ax, color=STYLE["land"], edgecolor="none", zorder=1)
    _crop(admin0, extent).plot(ax=ax, color=STYLE["boundary"], linewidth=0.75, zorder=3)
    _crop(admin1, extent).plot(ax=ax, color=STYLE["admin"], linewidth=0.35, zorder=2)
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("#f8faf9")
    ax.grid(True, color=STYLE["grid"], linewidth=0.45, alpha=0.8)
    ax.tick_params(axis="both", labelsize=8, colors="#66736f", length=0)
    for spine in ax.spines.values():
        spine.set_edgecolor("#ccd6d2")
    return bounds


def _add_domain(ax, bbox: tuple[float, float, float, float], label: str):
    min_lat, max_lat, min_lon, max_lon = bbox
    rect = Rectangle(
        (min_lon, min_lat),
        max_lon - min_lon,
        max_lat - min_lat,
        facecolor=STYLE["domain_fill"],
        edgecolor=STYLE["domain_edge"],
        linewidth=2.0,
        alpha=0.34,
        zorder=6,
    )
    ax.add_patch(rect)
    ax.text(
        min_lon + 0.18,
        max_lat - 0.35,
        label,
        color=STYLE["domain_edge"],
        fontsize=9,
        fontweight="bold",
        va="top",
        zorder=7,
    )


def _add_scalebar(ax, length_deg: float, label: str):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x = x0 + (x1 - x0) * 0.06
    y = y0 + (y1 - y0) * 0.08
    ax.plot([x, x + length_deg], [y, y], color=STYLE["domain_edge"], linewidth=4, solid_capstyle="round", zorder=8)
    ax.text(x, y + (y1 - y0) * 0.035, label, color=STYLE["text"], fontsize=8, va="bottom", zorder=8)


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {path}")


def texas(land, admin0, admin1):
    fig, ax = plt.subplots(figsize=(6.1, 4.0), constrained_layout=True)
    extent = (-106.8, -88.0, 24.5, 38.0)
    _draw_base(ax, land, admin0, admin1, extent)
    texas_shape = admin1[(admin1["admin"] == "United States of America") & (admin1["name"] == "Texas")]
    texas_shape.plot(ax=ax, color=STYLE["focus"], edgecolor=STYLE["boundary"], linewidth=1.2, zorder=4)
    _add_domain(ax, (30.0, 35.0, -100.0, -95.0), "Validation domain")
    _add_scalebar(ax, 2.0, "2 deg")
    ax.set_title("Texas Fixture", loc="left", fontsize=16, fontweight="bold", color=STYLE["text"])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    _save(fig, FIGURES / "region_texas_locator_report.png")


def ethiopia(land, admin0, admin1):
    fig, ax = plt.subplots(figsize=(6.1, 4.0), constrained_layout=True)
    extent = (27.0, 51.5, 0.0, 18.0)
    _draw_base(ax, land, admin0, admin1, extent)
    ethiopia_shape = admin1[admin1["admin"] == "Ethiopia"]
    ethiopia_shape.dissolve().plot(ax=ax, color=STYLE["focus"], edgecolor=STYLE["boundary"], linewidth=1.2, zorder=4)
    ethiopia_shape.boundary.plot(ax=ax, color="#b9a29b", linewidth=0.35, zorder=5)
    _add_domain(ax, (3.0, 15.0, 33.0, 48.0), "Validation domain")
    _add_scalebar(ax, 3.0, "3 deg")
    ax.set_title("Ethiopia FMA Fixture", loc="left", fontsize=16, fontweight="bold", color=STYLE["text"])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    _save(fig, FIGURES / "region_ethiopia_locator_report.png")


def main() -> int:
    land, admin0, admin1 = _read_layers()
    texas(land, admin0, admin1)
    ethiopia(land, admin0, admin1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
