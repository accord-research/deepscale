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
from datetime import date, timedelta
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
        if field is not None:
            # Live daily feed is on its own (finer, unmasked) grid; match the
            # climatology/forecast grid so the comparison triptych lines up
            # cell-for-cell with the forecast panels.
            field = field.interp(
                lat=obs_clim["lat"], lon=obs_clim["lon"], method="nearest",
            )
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
    window_days = cfg.comparison_window_days

    explorer: dict[str, list[str]] = {}

    for country in sorted(cfg.countries):
        cc = cfg.countries[country]
        issuances = _list_issuances(Path(store_root), country)

        # Rolling recent window (relative to this country's latest issuance) for the
        # comparison maps — keeps the page, the render cost, and the gh-pages branch
        # bounded as the testbed runs for months/years. The skill-over-time metrics
        # below still use the full history.
        windowed: list[date] = []
        if issuances:
            latest = max(issuances)
            cutoff = latest - timedelta(days=window_days)
            windowed = [i for i in issuances if i >= cutoff]

        # Fetch CHIRPS once per country, but only when there are windowed issuances
        # whose comparison panels we'll actually draw. Used to label each grid with
        # the observed precip for the same dekad.
        obs_clim = None
        obs_live = None
        if windowed:
            clim_lo, clim_hi = cfg.climatology_years
            end_year = max(i.year for i in windowed)
            try:
                obs_clim = rosetta_fetch(
                    product=cc.obs,
                    variable=cc.variable,
                    region=_bbox_to_region(cc.bbox),
                    hindcast=(clim_lo, max(clim_hi, end_year)),
                )[cc.variable].load()
            except Exception as e:  # noqa: BLE001 — render is best-effort
                print(f"[render] obs fetch failed for {country}: {e}")
            # Live feed only matters for target dekads CHIRPS dekadal hasn't
            # finalized yet (~90-day window of today). Only fetch when a windowed
            # issuance's target falls inside it.
            if cc.obs_live:
                live_window_start = date.today() - timedelta(days=90)
                needs_live = False
                for iss in windowed:
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
                            hindcast=(date.today().year, date.today().year),
                            # Bypass rosetta's year-granularity cache so the
                            # recent-obs window is always fresh (see verify.py).
                            cache=False,
                        )[cc.variable].load()
                    except Exception as e:  # noqa: BLE001 — live obs is optional
                        print(f"[render] live obs unavailable for {country}: {e}")

        rendered: list[tuple[date, bool]] = []
        for issuance in windowed:
            targets = _list_targets(Path(store_root), country, issuance)
            if not targets:
                continue
            target = targets[0]
            obs_ds = None
            if obs_clim is not None:
                obs_ds = _obs_panel_for_target(obs_clim, obs_live, target, cc.variable)
            method_panels: dict = {}
            for method in cc.methods:
                ds = _load_method_panel(Path(store_root), country, issuance, method, target)
                if ds is not None:
                    method_panels[method] = ds
            if not method_panels:
                continue
            fig = comparison_grid(obs_ds, method_panels, dekad_label=f"{country} {issuance} → {target}")
            out_dir = dashboard_root / country / issuance.isoformat()
            out_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_dir / "comparison.png", dpi=80, bbox_inches="tight")
            # "Full" = the reforecast was available, so at least one downscaling
            # method (anything beyond the raw + climatology baselines) ran.
            is_full = any(m not in ("raw", "climatology") for m in method_panels)
            rendered.append((issuance, is_full))

        # Metrics panel per country (always emit, full history).
        scores = _load_scores(Path(verification_root), country)
        metrics_fig = metrics_panel(scores, country=country)
        metrics_path = dashboard_root / country / "metrics.png"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_fig.savefig(metrics_path, dpi=80, bbox_inches="tight")

        explorer[country] = [
            {"d": d.isoformat(), "full": full}
            for d, full in sorted(rendered, key=lambda t: t[0], reverse=True)
        ]

    _write_index(dashboard_root, explorer)


def _write_index(dashboard_root: Path, explorer: dict[str, list[dict]]) -> None:
    parts = [
        "<!doctype html>",
        "<html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>S2S testbed dashboard</title>",
        "<link rel='stylesheet' href='../theme.css'>",
        "</head><body>",
        "<header><p><a href='../'>&larr; All forecasts</a></p>",
        "<h1>Sub-seasonal (S2S) testbed</h1></header>",
        "<h2>Forecast vs. observed &mdash; explore by issuance</h2>",
    ]
    if any(explorer.values()):
        parts.append(
            "<p class='controls'>"
            "<label>Country: <select id='sel-country'></select></label> "
            "<label>Issuance: <select id='sel-issuance'></select></label> "
            "<label><input type='checkbox' id='full-only' checked> "
            "Full weeks only (with downscaling)</label>"
            "</p>"
            "<img id='comparison' alt='forecast vs observed comparison grid'>"
        )
        parts.append(
            "<script>\n"
            # Each entry is {d: 'YYYY-MM-DD', full: bool}; full == the reforecast
            # was available so the downscaling methods ran (not just raw+clim).
            f"const COMPARISONS = {json.dumps(explorer)};\n"
            "const csel = document.getElementById('sel-country');\n"
            "const isel = document.getElementById('sel-issuance');\n"
            "const img = document.getElementById('comparison');\n"
            "const fullOnly = document.getElementById('full-only');\n"
            "for (const c of Object.keys(COMPARISONS)) {\n"
            "  if (COMPARISONS[c].length) {\n"
            "    const o = document.createElement('option'); o.value = c; o.textContent = c;\n"
            "    csel.appendChild(o);\n"
            "  }\n"
            "}\n"
            "function fillIssuances() {\n"
            "  isel.innerHTML = '';\n"
            "  const list = COMPARISONS[csel.value].filter(e => !fullOnly.checked || e.full);\n"
            "  for (const e of list) {\n"
            "    const o = document.createElement('option'); o.value = e.d;\n"
            "    o.textContent = e.full ? e.d : e.d + '  (baselines only)';\n"
            "    isel.appendChild(o);\n"
            "  }\n"
            "}\n"
            "function showComparison() {\n"
            "  if (!isel.value) { img.removeAttribute('src'); img.alt = 'no issuance in view'; return; }\n"
            "  img.src = csel.value + '/' + isel.value + '/comparison.png';\n"
            "  img.alt = csel.value + ' ' + isel.value;\n"
            "}\n"
            "csel.addEventListener('change', () => { fillIssuances(); showComparison(); });\n"
            "isel.addEventListener('change', showComparison);\n"
            "fullOnly.addEventListener('change', () => { fillIssuances(); showComparison(); });\n"
            "fillIssuances(); showComparison();\n"
            "</script>"
        )
    else:
        parts.append("<p><em>No comparison maps in the current window.</em></p>")
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
