"""Compare CPT intermediate CCA outputs against DeepScale internals.

Run after `validation/cca_synthetic_cpt_parity.py run-cpt`. It uses the CPT
workspace text outputs from the synthetic fixture and compares EOF/CCA scores to
a full-sample DeepScale fit. This is not a hindcast parity test; it diagnoses
whether the fitted decomposition agrees before crossvalidated prediction.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import xarray as xr

from deepscale.methods.cca import CCAMethod


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation" / "results"
FIXTURE = OUT / "cca_synthetic_fixture.nc"
WORKSPACE = OUT / "cca_synthetic_cpt_workspace"
RESULT = OUT / "cca_cpt_intermediate_diagnostics.json"


@dataclass
class SeriesComparison:
    name: str
    mode: int
    best_sign: int
    corr: float
    rmse: float
    deepscale_std: float
    cpt_std: float


def _read_cpt_table(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("xmlns") or stripped.startswith("cpt:"):
            continue
        parts = stripped.split()
        if "-" in parts[0] and parts[0][:4].isdigit():
            rows.append([float(x) for x in parts[1:]])
    if not rows:
        raise RuntimeError(f"No numeric time rows parsed from {path}")
    return np.asarray(rows, dtype=float)


def _read_canonical_correlations(path: Path) -> list[float]:
    rows = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("xmlns") or stripped.startswith("cpt:") or stripped.startswith("correlation"):
            continue
        parts = stripped.split()
        if parts[0].isdigit():
            rows.append(float(parts[1]))
    return rows


def _compare(name: str, deepscale: np.ndarray, cpt: np.ndarray) -> list[SeriesComparison]:
    out = []
    for idx in range(deepscale.shape[1]):
        candidates = []
        for sign in (1, -1):
            ds = sign * deepscale[:, idx]
            corr = np.corrcoef(ds, cpt[:, idx])[0, 1]
            rmse = np.sqrt(np.mean((ds - cpt[:, idx]) ** 2))
            candidates.append((abs(corr), -rmse, sign, corr, rmse))
        _, _, sign, corr, rmse = max(candidates)
        out.append(
            SeriesComparison(
                name=name,
                mode=idx + 1,
                best_sign=sign,
                corr=float(corr),
                rmse=float(rmse),
                deepscale_std=float(deepscale[:, idx].std()),
                cpt_std=float(cpt[:, idx].std()),
            )
        )
    return out


def main() -> int:
    fixture = xr.open_dataset(FIXTURE)
    model = CCAMethod(n_modes=2, x_eof_modes=2, y_eof_modes=2)
    model.fit(fixture["hindcast"], fixture["obs"])

    comparisons = []
    comparisons += _compare(
        "predictor_eof_scores_scaled",
        model.tsx_.T * model.svx_,
        _read_cpt_table(WORKSPACE / "predictor_eof_timeseries.txt"),
    )
    comparisons += _compare(
        "predictand_eof_scores_scaled",
        model.tsy_.T * model.svy_,
        _read_cpt_table(WORKSPACE / "predictand_eof_timeseries.txt"),
    )
    comparisons += _compare(
        "predictor_cca_scores",
        (model.s_ @ model.tsx_).T,
        _read_cpt_table(WORKSPACE / "predictor_cca_timeseries.txt"),
    )
    comparisons += _compare(
        "predictand_cca_scores",
        (model.r_.T @ model.tsy_).T,
        _read_cpt_table(WORKSPACE / "predictand_cca_timeseries.txt"),
    )

    payload = {
        "fixture": str(FIXTURE),
        "workspace": str(WORKSPACE),
        "deepscale_canonical_correlations": [float(x) for x in model.mu_],
        "cpt_canonical_correlations": _read_canonical_correlations(WORKSPACE / "cca_canonical_correlation.txt"),
        "comparisons": [asdict(c) for c in comparisons],
        "notes": [
            "EOF score signs are arbitrary, so each mode is compared after choosing the sign with highest agreement",
            "CPT EOF timeseries are compared to DeepScale unit-norm scores multiplied by singular values",
            "This is a full-sample decomposition diagnostic, not a crossvalidated hindcast comparison",
        ],
    }
    RESULT.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"Wrote {RESULT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
