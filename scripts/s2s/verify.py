"""S2S verification pass.

Scans the issuance store, fetches the relevant CHIRPS dekadal obs,
scores every pending (country, issuance, method, target_dekad) pair,
and appends a record per pair to ``verification/<country>/scores.jsonl``.

Idempotent: a pair already scored is skipped on subsequent runs.

Invocation:
  uv run python -m scripts.s2s.verify \\
      --store-root issuances --verification-root verification \\
      --config scripts/s2s/s2s.yml
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT.parent / "rosetta" / "src"))

from scripts.s2s.config import S2SConfig, load_config  # noqa: E402
from scripts.s2s.issuance_store import list_pending_pairs, read_issuance  # noqa: E402
from scripts.s2s.scoring import append_score_record, load_scored_keys, score_pair  # noqa: E402


def rosetta_fetch(product, **kwargs):
    """Thin wrapper around rosetta.fetch — easier to patch in tests."""
    import rosetta
    return rosetta.fetch(product=product, **kwargs)


def _bbox_to_region(bbox: dict) -> list[float]:
    return [bbox["min_lat"], bbox["max_lat"], bbox["min_lon"], bbox["max_lon"]]


def _obs_at_dekad_rolling(obs_full: xr.DataArray, target: date) -> xr.DataArray | None:
    """Return the (lat, lon) obs for the 10-day window starting at ``target``,
    when ``obs_full`` came from chirps_v2(agg_days=10) (timestamps where the
    value at day T is the rolling mean of [T-9, T]; we want T = target + 9).
    Returns None if the obs for that timestamp isn't archived yet.
    """
    ts = target + timedelta(days=9)
    ts64 = xr_timestamp_for(ts)
    if ts64 not in obs_full["time"].values:
        return None
    return obs_full.sel(time=ts64)


def _obs_at_dekad_daily(obs_full: xr.DataArray, target: date) -> xr.DataArray | None:
    """Return the (lat, lon) mean obs for the 10-day window starting at
    ``target``, when ``obs_full`` came from chirps_raw_live (daily values,
    no rolling). Computes the mean of [target, target+9] inclusive.
    Returns None if any of those 10 days are missing (window isn't complete yet).
    """
    import numpy as np
    days = [target + timedelta(days=i) for i in range(10)]
    have = set(obs_full["time"].values.astype("datetime64[D]").tolist())
    needed = [np.datetime64(d.isoformat(), "D") for d in days]
    if not all(n in have for n in needed):
        return None
    window = obs_full.sel(time=[xr_timestamp_for(d) for d in days])
    return window.mean("time")


def xr_timestamp_for(d: date):
    """Convert a date to the numpy datetime64 used by xarray's time coord."""
    import numpy as np
    return np.datetime64(d.isoformat(), "ns")


def _obs_climatology_for_dekad(
    obs_full: xr.DataArray, target: date, climatology_years: tuple[int, int]
) -> xr.DataArray:
    """Return (year, lat, lon) — one entry per climatology year for the
    10-day window starting at ``target``. Mirrors run_issuance's helper.
    """
    y0, y1 = climatology_years
    obs = obs_full.sel(time=slice(f"{y0}-01-01", f"{y1}-12-31"))
    target_offset_doy = (target + timedelta(days=9)).timetuple().tm_yday
    mask = obs.time.dt.dayofyear == target_offset_doy
    sel = obs.where(mask, drop=True)
    sel = (
        sel.assign_coords(year=("time", sel.time.dt.year.values))
        .swap_dims({"time": "year"})
        .drop_vars("time")
    )
    return sel


def verify(
    *,
    store_root: Path,
    verification_root: Path,
    cfg: S2SConfig,
    countries: list[str] | None = None,
) -> None:
    """Score every pending pair in the issuance store.

    Iterates each configured country, fetching that country's obs once
    (bbox-cropped) and walking the country's pending pairs.
    """
    target_countries = countries or sorted(cfg.countries)

    for country in target_countries:
        cc = cfg.countries[country]
        jsonl = Path(verification_root) / country / "scores.jsonl"
        already = load_scored_keys(jsonl)
        pending = [k for k in list_pending_pairs(Path(store_root), already_scored=already)
                   if k[0] == country]
        if not pending:
            continue

        # Climatology obs (finalized chirps_v2, agg_days=10 rolling).
        clim_lo, clim_hi = cfg.climatology_years
        end_year = max(p[3].year for p in pending)
        obs_clim = rosetta_fetch(
            product=cc.obs,
            variable=cc.variable,
            region=_bbox_to_region(cc.bbox),
            hindcast=(clim_lo, max(clim_hi, end_year)),
        )[cc.variable].load()

        # Optional live obs (preliminary daily feed, ~8-day lag) for recent
        # target dekads that haven't entered the finalized record yet.
        obs_live = None
        if cc.obs_live:
            from datetime import date as _date
            today = _date.today()
            # Live feed has data through ~today - 8 days; clamp the fetch
            # window so sheerwater doesn't enumerate future dates.
            earliest_target = min(p[3] for p in pending)
            live_lo = earliest_target.year
            live_hi = today.year
            try:
                obs_live = rosetta_fetch(
                    product=cc.obs_live,
                    variable=cc.variable,
                    region=_bbox_to_region(cc.bbox),
                    hindcast=(live_lo, live_hi),
                )[cc.variable].load()
            except Exception as e:  # noqa: BLE001 — live feed is best-effort
                print(f"[verify] live obs unavailable for {country}: {e}")

        for (_, issuance, method, target) in pending:
            # Try finalized obs first (correct semantics for past dekads);
            # fall back to the live feed for recent dekads.
            obs_field = _obs_at_dekad_rolling(obs_clim, target)
            if obs_field is None and obs_live is not None:
                obs_field = _obs_at_dekad_daily(obs_live, target)
            if obs_field is None:
                continue
            clim = _obs_climatology_for_dekad(obs_clim, target, cfg.climatology_years)
            pred = read_issuance(Path(store_root), country, issuance, method, target)
            metrics = score_pair(pred, obs_field, clim)

            record = {
                "country": country,
                "issuance": issuance.isoformat(),
                "method": method,
                "target_dekad": target.isoformat(),
                **metrics,
            }
            append_score_record(jsonl, record)


def _cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store-root", required=True, type=Path)
    ap.add_argument("--verification-root", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()
    cfg = load_config(args.config)
    verify(
        store_root=args.store_root,
        verification_root=args.verification_root,
        cfg=cfg,
    )


if __name__ == "__main__":
    _cli()
