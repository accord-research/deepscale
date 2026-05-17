"""End-to-end nightly pipeline against live CDS for Kenya MAM only.

Skipped by default in CI (integration marker excluded). Run locally with:
    uv run pytest tests/test_nightly_integration.py -m integration -v
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def trimmed_config(tmp_path):
    """A 3-year, 1-model variant of nightly.yml for fast CDS roundtrip."""
    import yaml

    src = REPO_ROOT / "scripts" / "nightly" / "nightly.yml"
    data = yaml.safe_load(src.read_text())
    data["countries"] = {"kenya": data["countries"]["kenya"]}
    # seasonal_mme requires >=5 years of overlap; use 6 to leave headroom.
    data["countries"]["kenya"]["hindcast_period"] = [2010, 2015]
    data["countries"]["kenya"]["models"] = ["c3s/ecmwf"]
    data["countries"]["kenya"]["seasons"] = {
        "MAM": data["countries"]["kenya"]["seasons"]["MAM"],
    }
    cfg_path = tmp_path / "nightly.yml"
    cfg_path.write_text(yaml.safe_dump(data))
    return cfg_path


@pytest.mark.integration
def test_run_country_end_to_end(trimmed_config, tmp_path):
    if not Path.home().joinpath(".cdsapirc").is_file():
        pytest.skip("~/.cdsapirc not present")

    out = tmp_path / "out"
    out.mkdir()
    today_str = "2026-02-15"  # mid-Feb so Kenya MAM has Feb 2026 init in range

    proc = subprocess.run(
        [
            sys.executable, "-m", "scripts.nightly.run_country",
            "--country", "kenya",
            "--today", today_str,
            "--output-root", str(out),
            "--config", str(trimmed_config),
        ],
        cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        capture_output=True, text=True, timeout=60 * 60,
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"

    base = out / "kenya" / "MAM" / "2026-02"
    assert (base / "forecast.nc").stat().st_size > 0
    assert (base / "tercile_map.png").stat().st_size > 0
    metrics = json.loads((base / "skill_metrics.json").read_text())
    assert "rpss" in metrics
