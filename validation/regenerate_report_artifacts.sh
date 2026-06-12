#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: validation/regenerate_report_artifacts.sh [--skip-pycpt]

Regenerates the validation report's controlled CHIRPS result artifacts and
figures. Outputs are written under:

  validation/results/
  validation/figures/

Environment variables:

  ROSETTA_SRC    Path to the Rosetta source tree. Defaults to ../rosetta/src
                 if it exists, otherwise falls back to the current PYTHONPATH.
  PYCPT_IMAGE    Docker image for PyCPT/CPT. Defaults to pycpt:2.10.4.

PyCPT/CPT CCA artifacts require Docker and the PyCPT image. Use --skip-pycpt
to regenerate the non-PyCPT artifacts only.
EOF
}

SKIP_PYCPT=0
for arg in "$@"; do
  case "$arg" in
    --skip-pycpt)
      SKIP_PYCPT=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      usage >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${ROSETTA_SRC:-}" && -d "$REPO_ROOT/../rosetta/src" ]]; then
  export ROSETTA_SRC="$REPO_ROOT/../rosetta/src"
fi
if [[ -n "${ROSETTA_SRC:-}" ]]; then
  export PYTHONPATH="${ROSETTA_SRC}${PYTHONPATH:+:$PYTHONPATH}"
fi

PYCPT_IMAGE="${PYCPT_IMAGE:-pycpt:2.10.4}"

run_uv_validation() {
  uv run --with xsdba --with 'setuptools<80' "$@"
}

run_uv_bcsd() {
  uv run --with scikit-downscale --with 'scikit-learn<1.6' --with 'setuptools<80' "$@"
}

echo "== Controlled CHIRPS method scores =="
run_uv_validation python validation/downscaling_validation.py --preset long_texas_1991_2020
run_uv_validation python validation/downscaling_validation.py --preset ethiopia_fma_1991_2020

echo "== BCSD reference diagnostics =="
run_uv_bcsd python validation/bcsd_reference_validation.py --preset long_texas_1991_2020
run_uv_bcsd python validation/bcsd_reference_validation.py --preset ethiopia_fma_1991_2020

echo "== QM/DQM reference diagnostics =="
run_uv_validation python validation/qm_dqm_diagnostics.py --preset long_texas_1991_2020 --coarsen-factor 10
run_uv_validation python validation/qm_dqm_diagnostics.py --preset ethiopia_fma_1991_2020 --coarsen-factor 10

if [[ "$SKIP_PYCPT" -eq 0 ]]; then
  echo "== PyCPT/CPT CCA parity diagnostics =="
  run_uv_validation python validation/pycpt_cca_parity.py prepare \
    --product obs/chirps-v3-monthly \
    --start-year 1991 \
    --end-year 2020 \
    --bbox 30 35 -100 -95 \
    --coarsen-factor 10
  docker run --rm -v "$REPO_ROOT/validation:/work/validation" -w /work "$PYCPT_IMAGE" \
    bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate pycpt && python validation/pycpt_cca_parity.py run-cpt'
  run_uv_validation python validation/pycpt_cca_parity.py compare

  run_uv_validation python validation/pycpt_cca_parity.py prepare \
    --tag ethiopia_fma_1991_2020 \
    --product obs/chirps-v3-monthly \
    --start-year 1991 \
    --end-year 2020 \
    --bbox 3 15 33 48 \
    --months 2 3 4 \
    --coarsen-factor 10
  docker run --rm -v "$REPO_ROOT/validation:/work/validation" -w /work "$PYCPT_IMAGE" \
    bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate pycpt && python validation/pycpt_cca_parity.py run-cpt --tag ethiopia_fma_1991_2020'
  run_uv_validation python validation/pycpt_cca_parity.py compare --tag ethiopia_fma_1991_2020
else
  echo "== Skipping PyCPT/CPT CCA parity diagnostics =="
fi

echo "== Summary/reference figures =="
plot_args=(
  python validation/plot_validation_results.py
  --skip-metrics
  --delta-reference-figure delta_reference_diagnostics_1991_2020.png
  --qm-reference-figure qm_reference_metric_bars_1991_2020.png
  --dqm-reference-figure dqm_reference_metric_bars_1991_2020.png
)
if [[ "$SKIP_PYCPT" -eq 1 ]]; then
  plot_args+=(--metrics-only)
fi
run_uv_validation "${plot_args[@]}"

echo "Done. Results are in validation/results/ and figures are in validation/figures/."
