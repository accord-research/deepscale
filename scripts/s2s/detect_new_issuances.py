"""Compute which ECMWF S2S issuances need processing on this run.

ECMWF S2S publishes Mon and Thu at 00 UTC. This helper:
  - enumerates valid issuance dates in a window
  - filters out issuances already present in the local store
  - prints the remaining dates one per line (for consumption by a
    GitHub Actions matrix via JSON-ified jq)

Invocation:
  uv run python -m scripts.s2s.detect_new_issuances \\
      --store-root issuance-store-checkout \\
      --country kenya --lookback-days 14
"""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path


# Monday = 0, Thursday = 3 in Python's weekday() convention.
_VALID_WEEKDAYS = {0, 3}


def issuances_in_window(start: date, end: date) -> list[date]:
    """Inclusive list of Mon/Thu dates in [start, end]."""
    out: list[date] = []
    cur = start
    while cur <= end:
        if cur.weekday() in _VALID_WEEKDAYS:
            out.append(cur)
        cur += timedelta(days=1)
    return out


def new_since(*, store_root: Path, country: str, since: date, until: date) -> list[date]:
    """Issuances in [since, until] that don't already have a directory on disk."""
    existing: set[date] = set()
    country_root = Path(store_root) / country
    if country_root.exists():
        for d in country_root.iterdir():
            if d.is_dir():
                try:
                    existing.add(date.fromisoformat(d.name))
                except ValueError:
                    continue
    return [d for d in issuances_in_window(since, until) if d not in existing]


def _cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store-root", required=True, type=Path)
    ap.add_argument("--country", required=True)
    ap.add_argument("--lookback-days", type=int, default=14)
    ap.add_argument("--today", default=None, help="YYYY-MM-DD; default today")
    args = ap.parse_args()
    today = date.fromisoformat(args.today) if args.today else date.today()
    out = new_since(
        store_root=args.store_root, country=args.country,
        since=today - timedelta(days=args.lookback_days), until=today,
    )
    # Print as a JSON array — easy to consume from a workflow matrix expression.
    print(json.dumps([d.isoformat() for d in out]))


if __name__ == "__main__":
    _cli()
