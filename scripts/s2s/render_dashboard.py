"""S2S dashboard renderer.

Walks the issuance store + verification dir and produces:
  - dashboard/<country>/<issuance>/comparison.png — one grid per (country, issuance)
  - dashboard/<country>/metrics.png               — one metrics panel per country
  - dashboard/index.html                          — top-level index with thumbnails / links

Invocation:
  uv run python -m scripts.s2s.render_dashboard \\
      --store-root issuances --verification-root verification \\
      --dashboard-root dashboard --config scripts/s2s/s2s.yml
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from datetime import date
from pathlib import Path

import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from rosetta import fetch as rosetta_fetch  # noqa: E402

from scripts.s2s.config import S2SConfig, load_config  # noqa: E402
from scripts.s2s.plotting import comparison_grid, metrics_panel  # noqa: E402
from scripts.s2s.verify import (  # noqa: E402
    _bbox_to_region,
    _obs_at_dekad_daily,
    _obs_at_dekad_rolling,
)


def _obs_panel_for_target(
    obs_clim: xr.DataArray, obs_live: xr.DataArray | None, target: date, variable: str,
) -> xr.Dataset | None:
    """Return CHIRPS observations for ``target`` dekad as a Dataset matching the
    method-panel shape (variable 'mean' on (lat, lon)). Tries finalized obs
    first, falls back to the daily live feed."""
    field = _obs_at_dekad_rolling(obs_clim, target)
    if field is None and obs_live is not None:
        field = _obs_at_dekad_daily(obs_live, target)
    if field is None:
        return None
    return xr.Dataset({"mean": field})


def _list_issuances(store_root: Path, country: str) -> list[date]:
    p = store_root / country
    if not p.exists():
        return []
    out = []
    for d in sorted(p.iterdir()):
        if not d.is_dir():
            continue
        try:
            out.append(date.fromisoformat(d.name))
        except ValueError:
            continue
    return out


def _list_targets(store_root: Path, country: str, issuance: date) -> list[date]:
    p = store_root / country / issuance.isoformat()
    if not p.exists():
        return []
    targets: set[date] = set()
    for method_dir in p.iterdir():
        if not method_dir.is_dir():
            continue
        for nc in method_dir.glob("dekad_*.nc"):
            try:
                targets.add(date.fromisoformat(nc.stem.removeprefix("dekad_")))
            except ValueError:
                continue
    return sorted(targets)


def _load_method_panel(
    store_root: Path, country: str, issuance: date, method: str, target: date,
) -> xr.Dataset | None:
    p = store_root / country / issuance.isoformat() / method / f"dekad_{target.isoformat()}.nc"
    if not p.exists():
        return None
    return xr.open_dataset(p)


def _load_scores(verif_root: Path, country: str) -> list[dict]:
    p = verif_root / country / "scores.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def render_dashboard(
    *,
    store_root: Path,
    verification_root: Path,
    dashboard_root: Path,
    config_path: Path | str,
) -> None:
    cfg: S2SConfig = load_config(config_path)
    dashboard_root = Path(dashboard_root)
    dashboard_root.mkdir(parents=True, exist_ok=True)

    index_entries: list[tuple[str, list[tuple[date, Path]]]] = []

    for country in sorted(cfg.countries):
        cc = cfg.countries[country]
        issuances = _list_issuances(Path(store_root), country)
        country_entries: list[tuple[date, Path]] = []

        # Fetch CHIRPS once per country, covering the climatology window plus
        # any issuance years we'll render. Used to label each comparison grid
        # with the actual observed precip for the same dekad.
        obs_clim = None
        obs_live = None
        if issuances:
            clim_lo, clim_hi = cfg.climatology_years
            end_year = max(i.year for i in issuances)
            try:
                obs_clim = rosetta_fetch(
                    product=cc.obs,
                    variable=cc.variable,
                    region=_bbox_to_region(cc.bbox),
                    hindcast=(clim_lo, max(clim_hi, end_year)),
                )[cc.variable].load()
            except Exception as e:  # noqa: BLE001 — render is best-effort
                print(f"[render] obs fetch failed for {country}: {e}")
            # The live feed is only useful for target dekads inside the
            # ~3-month window where CHIRPS dekadal hasn't finalized yet.
            # For older issuances we'd otherwise iterate `chirps_raw_live`
            # over thousands of pre-window dates that all return "no data
            # available" — wasted minutes per country and a real risk of
            # hanging the workflow once the issuance store accumulates
            # historical entries. Only fetch live obs when at least one
            # comparison target falls inside the live-feed window.
            if cc.obs_live:
                from datetime import date as _date, timedelta
                today = _date.today()
                live_window_start = today - timedelta(days=90)
                # render only uses the first/most-recent target per issuance
                # (`targets[0]`) for the comparison grid, so check those.
                needs_live = False
                for iss in issuances:
                    targets = _list_targets(Path(store_root), country, iss)
                    if targets and targets[0] >= live_window_start:
                        needs_live = True
                        break
                if needs_live:
                    try:
                        obs_live = rosetta_fetch(
                            product=cc.obs_live,
                            variable=cc.variable,
                            region=_bbox_to_region(cc.bbox),
                            hindcast=(today.year, today.year),
                        )[cc.variable].load()
                    except Exception as e:  # noqa: BLE001 — live obs is optional
                        print(f"[render] live obs unavailable for {country}: {e}")

        for issuance in issuances:
            targets = _list_targets(Path(store_root), country, issuance)
            if not targets:
                continue
            # For the first/most-recent target dekad, build the comparison row.
            target = targets[0]
            panels: dict = {}
            # Obs panel first so the reader sees truth on the left.
            if obs_clim is not None:
                obs_ds = _obs_panel_for_target(obs_clim, obs_live, target, cc.variable)
                if obs_ds is not None:
                    panels["obs (CHIRPS)"] = obs_ds
            for method in cc.methods:
                ds = _load_method_panel(Path(store_root), country, issuance, method, target)
                if ds is not None:
                    panels[method] = ds
            if not panels:
                continue
            fig = comparison_grid(panels, dekad_label=f"{country} {issuance} → {target}")
            out_dir = dashboard_root / country / issuance.isoformat()
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "comparison.png"
            fig.savefig(out_path, dpi=80, bbox_inches="tight")
            country_entries.append((issuance, out_path.relative_to(dashboard_root)))

        # Metrics panel per country (always emit, even if empty).
        scores = _load_scores(Path(verification_root), country)
        metrics_fig = metrics_panel(scores, country=country)
        metrics_path = dashboard_root / country / "metrics.png"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_fig.savefig(metrics_path, dpi=80, bbox_inches="tight")

        index_entries.append((country, country_entries))

    _write_index(dashboard_root, index_entries)


def _write_index(dashboard_root: Path, entries: list[tuple[str, list[tuple[date, Path]]]]) -> None:
    parts = ["<!doctype html>", "<html><head><meta charset='utf-8'>",
             "<title>S2S testbed dashboard</title>",
             "<style>body{font-family:sans-serif;max-width:1100px;margin:1em auto;padding:0 1em;background:#111;color:#eee} "
             "h1{font-size:1.4em} h2{font-size:1.1em;margin-top:1.5em} "
             "img{max-width:100%;height:auto;display:block;margin:0.5em 0;border:1px solid #444;background:#222} "
             "a{color:#8df}</style></head><body>",
             "<h1>S2S testbed dashboard</h1>"]
    for country, issuances in entries:
        parts.append(f"<h2>{html.escape(country)}</h2>")
        parts.append(f"<p><strong>Metrics:</strong></p><img src='{country}/metrics.png' alt='{country} metrics'>")
        if not issuances:
            parts.append("<p><em>No issuances rendered.</em></p>")
            continue
        parts.append("<p><strong>Comparison grids by issuance:</strong></p>")
        for issuance, rel in sorted(issuances):
            parts.append(f"<h3>{issuance.isoformat()}</h3>"
                         f"<img src='{html.escape(str(rel))}' alt='{country} {issuance}'>")
    parts.append("</body></html>")
    (dashboard_root / "index.html").write_text("\n".join(parts))


def _cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store-root", required=True, type=Path)
    ap.add_argument("--verification-root", required=True, type=Path)
    ap.add_argument("--dashboard-root", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()
    render_dashboard(
        store_root=args.store_root,
        verification_root=args.verification_root,
        dashboard_root=args.dashboard_root,
        config_path=args.config,
    )


if __name__ == "__main__":
    _cli()
