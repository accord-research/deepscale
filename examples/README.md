# DeepScale Examples

## `seasonal_forecast_eastafrica_mam.py` (§8 reference)

End-to-end PyCPT-parity reference: a 7-phase multi-model ensemble seasonal
forecast for East Africa MAM, a DeepScale + Rosetta port of
`pycpt-reference/pycpt_seasonal_forecast.py`. Phases are individually runnable.

```bash
# full real run (CDS creds + network):
python examples/seasonal_forecast_eastafrica_mam.py --phase 0 1 2
# plan only (no network):
python examples/seasonal_forecast_eastafrica_mam.py --dry-run
# synthetic smoke (no network, exercises the real pipeline):
python examples/seasonal_forecast_eastafrica_mam.py --tiny --phase 0 1 2 3 5 6
```

**Model coverage:** PyCPT's reference blends 10 GCMs. SPEAR / SPEARb /
CanSIPS-IC4 (2 of them) are currently **excluded** — their data lived in the
sunset IRI Data Library. The Columbia CCSR successor is live and serves them,
but needs a dedicated adapter (rosetta #14); until then the MME runs with the
verified-available models and Phase 0 prints exactly which were used/skipped.

## `demo_forecast.py`

End-to-end example that runs:

1. Rosetta data fetch (ERA5 + C3S/ECMWF)
2. DeepScale method optimization
3. Tercile forecast generation
4. LOYO cross-validated skill report
5. Plot output to `deepscale/examples/output/demo_forecast.png`

### Prerequisites

1. Clone both repos side by side:

```bash
git clone https://github.com/jataware/deepscale.git
git clone https://github.com/jataware/rosetta.git
```

2. Install dependencies for both:

```bash
cd rosetta && uv sync
cd ../deepscale && uv sync
```

3. Configure CDS credentials in `~/.cdsapirc` (see `rosetta/README.md` for details).
4. Accept CDS dataset licenses for ERA5 and C3S seasonal products.

### Run

From repository root:

```bash
python deepscale/examples/demo_forecast.py
```

Backward-compatible entrypoint (same behavior):

```bash
python demo_forecast.py
```

### Local outputs

The script writes local artifacts to `deepscale/examples/output/`:

- `demo_cache/*.nc`
- `demo_forecast.png`

These are intentionally ignored by git.
