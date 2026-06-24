"""Smoke tests for the WVG/logistic and ensemble-regression example demos.

The demos default to real data (CDS via Rosetta). Their ``--synthetic`` mode is
offline, deterministic, and self-asserting, so we exercise that in the fast unit
gate — it doubles as end-to-end regression coverage for Index +
logistic_forecast + the ensemble_regression method (and that each demo always
writes its PNG).
"""
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _run(script):
    return subprocess.run(
        [sys.executable, str(REPO / "examples" / script), "--synthetic"],
        capture_output=True, text=True, cwd=str(REPO),
    )


@pytest.mark.parametrize(
    "script, png",
    [
        ("demo_logistic_wvg.py", "logistic_wvg_tercile.png"),
        ("demo_ensemble_regression.py", "ensemble_regression_tercile.png"),
    ],
)
def test_demo_runs_end_to_end(script, png):
    r = _run(script)
    assert r.returncode == 0, r.stderr
    assert "PASS:" in r.stdout
    assert "complete." in r.stdout
    assert (REPO / "examples" / "output" / png).exists()
