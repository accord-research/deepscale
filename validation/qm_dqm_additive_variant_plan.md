# QM/DQM Additive-Factor Variant Plan

Date: 2026-06-09

## Status

Do not change the current empirical `qm`/`dqm` defaults during this validation
pass. The current empirical implementation is internally consistent with its
direct CDF mapping convention, and validation artifacts already pin that behavior
against independent NumPy sorted-column oracles.

This plan records the implementation shape for a later explicit
additive-factor convention.

## Proposed API

Keep `variant` as the distribution family selector:

- `variant="empirical"`
- `variant="parametric"`

Add a separate empirical convention selector:

- `convention="direct_cdf"`: current DeepScale behavior and default.
- `convention="additive_factor"`: xsdba-style behavior.

Constructor shape:

```python
QuantileMappingMethod(variant="empirical", convention="direct_cdf")
DetrendedQuantileMappingMethod(variant="empirical", convention="direct_cdf")
```

For `variant="parametric"`, reject non-default conventions or ignore only if the
API documentation is explicit. Prefer rejecting with a clear `ValueError` to
avoid implying the knob affects Gaussian mapping.

## QM Behavior

Current direct-CDF convention:

1. Sort historical GCM and observed columns.
2. Estimate `q = F_gcm(x)` using midpoint plotting positions.
3. Return `F_obs^-1(q)`.
4. Clamp outside training extrema through `numpy.interp` endpoint behavior.

Additive-factor convention:

1. Compute historical quantiles `hist_q` and observed quantiles `ref_q`.
2. Compute additive adjustment factors `af_q = ref_q - hist_q`.
3. Interpolate `af_q` over `hist_q` at the forecast value.
4. Return `x + af(x)`.
5. Keep tail behavior explicit. The validation oracle currently uses constant
   endpoint extrapolation when matching `xsdba`.

## DQM Behavior

Keep the existing detrending scaffold:

1. Estimate and remove the centered GCM trend.
2. Estimate and remove the observed trend for fitting.
3. Apply the selected empirical convention to the detrended forecast.
4. Re-add the GCM trend at the forecast time.

The additive-factor convention should operate on detrended values only; it
should not change the trend preservation logic.

## Test Targets

Add unit tests before changing production behavior:

- `QuantileMappingMethod(variant="empirical")` still defaults to
  `convention="direct_cdf"` and existing tests pass unchanged.
- Invalid convention raises `ValueError`.
- Parametric QM rejects `convention="additive_factor"` or documents a no-op
  explicitly.
- Current empirical direct-CDF output matches the existing sorted-column oracle.
- Additive-factor QM matches the NumPy `factor_xsdba_linear` or
  `factor_xsdba_nearest` oracle from `validation/empirical_qm_conventions.py`.
- Additive-factor DQM preserves the GCM trend on the existing trended DQM test
  fixture.
- Public `deepscale.downscale(..., method="qm", convention="additive_factor")`
  forwards the convention into the method if the pipeline currently supports
  arbitrary method kwargs.

## Validation Artifacts To Pin

Use these as regression references:

- `validation/results/empirical_qm_conventions_texas.json`
- `validation/results/empirical_qm_conventions_long_texas_1991_2020.json`
- `validation/results/empirical_qm_conventions_east_africa_2001_2020.json`
- `validation/results/empirical_qm_conventions_east_africa_mam_2001_2020.json`
- `validation/results/empirical_qm_conventions_east_africa_ond_2001_2020.json`

The key expected relationship is:

- Direct-CDF DeepScale empirical QM matches the NumPy `deepscale_sorted` oracle.
- Additive-factor QM should match the NumPy `factor_xsdba_*` oracle and the
  corresponding `xsdba` result within interpolation/extrapolation tolerances.

## Open Product Decisions

- Should precipitation support multiplicative factors in addition to additive
  factors?
- Should the additive-factor convention use linear interpolation or nearest
  interpolation by default?
- Should tail extrapolation remain constant, or should it allow linear tail
  extrapolation?
- Should the public API expose `convention=` directly, or should the variant be
  named more explicitly, such as `variant="empirical_additive"`?
