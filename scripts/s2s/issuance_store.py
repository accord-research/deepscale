"""Issuance-store filesystem layout and I/O.

Layout (relative to a configurable store_root, e.g. ``issuances/``):

    <country>/<issuance YYYY-MM-DD>/<method>/dekad_YYYY-MM-DD.nc

Each NetCDF holds an xr.Dataset with at minimum a ``mean`` variable
(lat × lon). Ensemble-producing methods also carry ``tercile_probs``
(category × lat × lon), category coord = ["below", "normal", "above"].
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import xarray as xr


def issuance_path(
    store_root: Path,
    country: str,
    issuance: date,
    method: str,
    target_dekad: date,
) -> Path:
    return (
        Path(store_root) / country / issuance.isoformat() / method
        / f"dekad_{target_dekad.isoformat()}.nc"
    )


def write_issuance(
    store_root: Path,
    country: str,
    issuance: date,
    method: str,
    target_dekad: date,
    ds: xr.Dataset,
) -> Path:
    path = issuance_path(store_root, country, issuance, method, target_dekad)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)
    return path


def read_issuance(
    store_root: Path,
    country: str,
    issuance: date,
    method: str,
    target_dekad: date,
) -> xr.Dataset:
    return xr.open_dataset(issuance_path(store_root, country, issuance, method, target_dekad))


def list_pending_pairs(
    store_root: Path,
    already_scored: set[tuple[str, date, str, date]],
) -> list[tuple[str, date, str, date]]:
    """Walk the store and return every (country, issuance, method, dekad)
    tuple that is present on disk but absent from ``already_scored``.

    The Plan C verifier passes its already-scored set (loaded from the
    verification JSONL) so re-runs are a no-op on already-scored pairs.
    """
    out: list[tuple[str, date, str, date]] = []
    root = Path(store_root)
    if not root.exists():
        return out
    for country_dir in sorted(root.iterdir()):
        if not country_dir.is_dir():
            continue
        for issuance_dir in sorted(country_dir.iterdir()):
            if not issuance_dir.is_dir():
                continue
            issuance = date.fromisoformat(issuance_dir.name)
            for method_dir in sorted(issuance_dir.iterdir()):
                if not method_dir.is_dir():
                    continue
                for nc in sorted(method_dir.glob("dekad_*.nc")):
                    dekad = date.fromisoformat(nc.stem.removeprefix("dekad_"))
                    key = (country_dir.name, issuance, method_dir.name, dekad)
                    if key not in already_scored:
                        out.append(key)
    return out
