# Empirical QM/DQM Validation Decision Memo

Date: 2026-06-08

## Question

Should DeepScale's empirical `qm`/`dqm` implementation be changed to match the
additive adjustment-factor convention used by `xsdba`, or should the current
direct empirical CDF value-mapping behavior remain as-is?

## Short Answer

Do not change the existing empirical `qm`/`dqm` behavior during the validation
pause. The validation evidence supports adding a separately named or explicitly
configured additive-factor variant later, with tests and compatibility notes.

The current DeepScale empirical path is internally consistent and reproducible
by an independent NumPy sorted-column oracle. It is not a broken implementation
of its own convention. However, on the CHIRPS high/low fixture, the additive
adjustment-factor convention used by `xsdba` performs materially better.

## Evidence

Harnesses:

- `validation/qm_dqm_diagnostics.py`
- `validation/empirical_qm_conventions.py`

Raw outputs:

- `validation/results/qm_dqm_diagnostics_texas.json`
- `validation/results/qm_dqm_diagnostics_long_texas_1991_2020.json`
- `validation/results/empirical_qm_conventions_texas.json`
- `validation/results/empirical_qm_conventions_long_texas_1991_2020.json`

Short Texas fixture, 2010-2021:

| Method | Reference | Variant | MAE | RMSE | Corr |
|---|---|---|---:|---:|---:|
| `dqm` | DeepScale | parametric | 0.0714 | 0.0951 | 0.9925 |
| `qm` | DeepScale | parametric | 0.0718 | 0.0958 | 0.9924 |
| `dqm` | xsdba | nq8 | 0.0734 | 0.0965 | 0.9923 |
| `qm` | xsdba | nq8 | 0.0739 | 0.0982 | 0.9920 |
| `qm` | DeepScale | empirical current | 0.1375 | 0.2375 | 0.9600 |
| `qm` | NumPy | deepscale_sorted oracle | 0.1376 | 0.2375 | 0.9600 |
| `dqm` | DeepScale | empirical current | 0.1743 | 0.2803 | 0.9388 |

Long Texas fixture, requested 1991-2020, actual 1998-2020:

| Method | Reference | Variant | MAE | RMSE | Corr |
|---|---|---|---:|---:|---:|
| `dqm` | DeepScale | parametric | 0.0708 | 0.0937 | 0.9922 |
| `qm` | DeepScale | parametric | 0.0709 | 0.0939 | 0.9921 |
| `dqm` | xsdba | default | 0.0740 | 0.0965 | 0.9917 |
| `qm` | xsdba | nq8 | 0.0736 | 0.0966 | 0.9916 |
| `qm` | xsdba/NumPy | additive factor linear | 0.0728 | 0.0958 | 0.9918 |
| `qm` | DeepScale | empirical current | 0.0937 | 0.1527 | 0.9806 |
| `qm` | NumPy | deepscale_sorted oracle | 0.0941 | 0.1530 | 0.9805 |
| `dqm` | DeepScale | empirical current | 0.1004 | 0.1639 | 0.9772 |

## Interpretation

The empirical QM discrepancy is primarily a convention difference:

- DeepScale empirical QM maps values directly through sorted historical columns
  using plotting positions.
- `xsdba` empirical QM computes adjustment factors, such as `ref_q - hist_q`,
  interpolates those factors over historical quantiles, then applies the factor
  to the simulated value.

The independent NumPy diagnostics reproduce both conventions:

- The sorted-column oracle reproduces DeepScale empirical QM.
- The additive-factor oracle reproduces `xsdba` EQM exactly on this fixture.

The longer training window improves DeepScale empirical QM/DQM substantially,
which suggests the current empirical path is sample-hungry rather than simply
malfunctioning. But it still trails additive-factor EQM/DQM and DeepScale's own
parametric QM/DQM on these high/low tests.

## Recommendation

Keep the existing empirical behavior stable for now. Add a follow-up task to
introduce an explicit additive-factor variant after the coding pause.

Suggested implementation shape:

- Keep current empirical behavior as a named compatibility mode.
- Add an additive adjustment-factor empirical mode for `qm`.
- Add the same convention for `dqm`, preserving detrending semantics.
- Make the default a deliberate product decision, not an accidental validation
  side effect.
- Add tests that pin both conventions against the current NumPy oracles.

Suggested naming options:

- `method="qm", convention="direct_cdf"` for current behavior.
- `method="qm", convention="additive_factor"` for `xsdba`-style behavior.
- Equivalent `dqm` convention knob if the existing API supports parameters.

## Risks

Changing the existing empirical default could silently alter historical outputs.
That would be risky for validation, reproducibility, and any downstream users
who already interpret current `qm`/`dqm` outputs as direct empirical mappings.

Adding a new explicit variant avoids that risk and gives us a clean way to
compare both conventions in future benchmarks.

## Open Checks

- Repeat the convention diagnostic on an Africa CHIRPS high/low box.
- Test temperature or another continuous variable if available.
- Decide whether precipitation should support multiplicative adjustment factors
  in addition to additive factors.
- Confirm how forecast-time extrapolation should behave at quantile tails.
