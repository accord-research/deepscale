"""Pair-scoring helpers for the S2S verification pass.

Given a single (predicted, observed) pair compute:

- ACC (Anomaly Correlation Coefficient) — Pearson r between pred-anom and obs-anom.
- RMSE
- bias (mean(pred) - mean(obs))
- RPSS — if the predicted Dataset carries ``tercile_probs``.

Results are appended to a JSONL file. The verifier deduplicates by
re-loading the JSONL into a set of (country, issuance, method, dekad)
tuples before scanning the issuance store.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import xarray as xr


def _to_anomaly(field: xr.DataArray, clim: xr.DataArray) -> xr.DataArray:
    """Subtract the per-cell long-term mean."""
    return field - clim.mean("year")


def _pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel()
    b = b.ravel()
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return float("nan")
    a = a[mask]
    b = b[mask]
    a -= a.mean()
    b -= b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom == 0:
        return float("nan")
    return float((a * b).sum() / denom)


def _rpss(probs: xr.DataArray, obs: xr.DataArray, clim: xr.DataArray) -> float:
    """RPSS vs. equiprobable climatology.

    probs: (category, lat, lon), categories = below/normal/above, sums to 1.
    obs:   (lat, lon).
    clim:  (year, lat, lon) — used to compute per-cell terciles.
    """
    t33 = clim.quantile(1 / 3, dim="year").drop_vars("quantile")
    t67 = clim.quantile(2 / 3, dim="year").drop_vars("quantile")
    obs_cat = xr.where(obs < t33, 0, xr.where(obs > t67, 2, 1)).values

    # 3-category one-hot for obs.
    onehot = np.zeros((3, *obs_cat.shape), dtype="float32")
    for c in range(3):
        onehot[c] = (obs_cat == c).astype("float32")

    cum_probs = np.cumsum(probs.values, axis=0)
    cum_obs = np.cumsum(onehot, axis=0)
    rps = np.sum((cum_probs - cum_obs) ** 2, axis=0)

    clim_probs = np.full_like(cum_probs, 1 / 3, dtype="float32")
    clim_probs = np.cumsum(clim_probs, axis=0)
    rps_clim = np.sum((clim_probs - cum_obs) ** 2, axis=0)

    rps_mean = np.nanmean(rps)
    rps_clim_mean = np.nanmean(rps_clim)
    if rps_clim_mean == 0:
        return float("nan")
    return float(1.0 - rps_mean / rps_clim_mean)


def score_pair(pred: xr.Dataset, obs: xr.DataArray, clim: xr.DataArray) -> dict:
    """Compute the metric record for one (pred, obs) pair.

    Always emits acc/rmse/bias. Emits rpss iff pred carries tercile_probs.
    """
    pred_mean = pred["mean"]
    pred_anom = _to_anomaly(pred_mean, clim).values
    obs_anom = _to_anomaly(obs, clim).values
    diff = pred_mean.values - obs.values

    record: dict = {
        "acc": _pearson_r(pred_anom, obs_anom),
        "rmse": float(np.sqrt(np.nanmean(diff ** 2))),
        "bias": float(np.nanmean(diff)),
    }
    if "tercile_probs" in pred.data_vars:
        record["rpss"] = _rpss(pred["tercile_probs"], obs, clim)
    return record


def append_score_record(path: Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def load_scored_keys(path: Path) -> set[tuple[str, date, str, date]]:
    """Return the set of (country, issuance, method, target_dekad) tuples
    already present in the JSONL — used to skip them on re-run."""
    path = Path(path)
    out: set[tuple[str, date, str, date]] = set()
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        out.add((
            rec["country"],
            date.fromisoformat(rec["issuance"]),
            rec["method"],
            date.fromisoformat(rec["target_dekad"]),
        ))
    return out
