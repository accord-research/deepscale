#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONPATH="/Users/david/rosetta/src"

echo "[$(date)] Starting real ECMWF validation overnight run"

run_preset() {
  local preset="$1"
  echo
  echo "[$(date)] Running preset: ${preset}"
  uv run --with xsdba --with 'setuptools<80' \
    python validation/real_forecast_validation.py --preset "${preset}"
  echo "[$(date)] Finished preset: ${preset}"
}

run_preset ecmwf_east_africa_mam_smoke_2001_2005
run_preset ecmwf_east_africa_mam_2001_2016
run_preset ecmwf_east_africa_ond_2001_2016

echo
echo "[$(date)] Real ECMWF validation overnight run complete"
