# DeepScale Examples

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
