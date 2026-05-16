"""SeasonalMMEResult -> output/<country>/<season>/<YYYY-MM>/ tree."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless runner

import matplotlib.pyplot as plt


def write_output(
    *,
    result: Any,
    country: str,
    season: str,
    init_year: int,
    init_month: int,
    root: str | Path,
) -> Path:
    """Write forecast.nc, tercile_map.png, skill_metrics.json under root."""
    base = Path(root) / country / season / f"{init_year}-{init_month:02d}"
    base.mkdir(parents=True, exist_ok=True)

    # forecast.nc — full tercile forecast plus deterministic mean
    nc_path = base / "forecast.nc"
    ds = result.tercile_forecast.to_dataset(name="tercile_forecast")
    ds["forecast"] = result.forecast
    ds.attrs["country"] = country
    ds.attrs["season"] = season
    ds.attrs["init_year"] = int(init_year)
    ds.attrs["init_month"] = int(init_month)
    ds.attrs["method"] = result.metadata.get("method", "")
    ds.to_netcdf(nc_path)

    # tercile_map.png — headline three-panel below/normal/above
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2), constrained_layout=True)
    for ax, tercile_idx, label in zip(
        axes, (0, 1, 2), ("Below", "Normal", "Above")
    ):
        ax.imshow(
            result.tercile_forecast.sel(tercile=tercile_idx).values,
            origin="lower",
            aspect="auto",
            vmin=0.0,
            vmax=0.6,
            cmap="RdYlBu" if tercile_idx == 0 else "Greens" if tercile_idx == 1 else "PuRd",
        )
        ax.set_title(f"P({label}-normal)")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{country.title()} {season} — init {init_year}-{init_month:02d}")
    fig.savefig(base / "tercile_map.png", dpi=110)
    plt.close(fig)

    # skill_metrics.json — flat float scores only
    scores = {
        k: float(v)
        for k, v in result.skill_report.scores.items()
        if isinstance(v, (int, float))
    }
    (base / "skill_metrics.json").write_text(json.dumps(scores, indent=2))

    return base
