"""§9 leakage-discipline regression for the CV tercile path (issue #25).

`to_tercile_cv` must score each year with held-out (leave-one-out) discipline:
a year's tercile probabilities depend on its *forecast* and on the boundaries /
PEV estimated from the *other* years — never on that year's own observation.
These tests pin that guarantee at the orchestration boundary so a future change
that quietly leaks the held-out year (the classic skill-inflation bug called out
in tercile.py's module docstring) fails loudly.
"""
import numpy as np
import pytest
import xarray as xr

from deepscale.tercile import to_tercile_cv


def _obs(seed=3, n=12):
    rng = np.random.default_rng(seed)
    years = list(range(2000, 2000 + n))
    vals = 5.0 + rng.standard_normal((n, 2, 2))
    return xr.DataArray(vals, dims=("year", "lat", "lon"),
                        coords={"year": years, "lat": [0.0, 1.0], "lon": [0.0, 1.0]})


@pytest.mark.parametrize("method", ["gaussian_loo", "bootstrap"])
def test_held_out_year_independent_of_its_own_obs(method):
    obs = _obs()
    cv_pred = obs + 0.1            # fixed predictions, computed before corruption

    base = to_tercile_cv(cv_pred, obs, method=method)

    # Corrupt ONLY 2005's observation to an extreme; keep cv_pred fixed.
    obs_corrupt = obs.copy()
    obs_corrupt.loc[{"year": 2005}] = 999.0
    corrupted = to_tercile_cv(cv_pred, obs_corrupt, method=method)

    # Leakage discipline: 2005's tercile probs exclude 2005's own obs, so they
    # are UNCHANGED. If boundaries/PEV leaked the held-out year, they'd shift.
    np.testing.assert_allclose(
        base.sel(year=2005).values, corrupted.sel(year=2005).values, atol=1e-9,
        err_msg=f"{method}: held-out year leaked its own obs into its terciles",
    )


@pytest.mark.parametrize("method", ["gaussian_loo", "bootstrap"])
def test_corruption_does_reach_other_years(method):
    """Guard against a trivially-passing test: corrupting 2005 MUST change the
    other years' terciles (their LOO sets include 2005)."""
    obs = _obs()
    cv_pred = obs + 0.1
    base = to_tercile_cv(cv_pred, obs, method=method)
    obs_corrupt = obs.copy()
    obs_corrupt.loc[{"year": 2005}] = 999.0
    corrupted = to_tercile_cv(cv_pred, obs_corrupt, method=method)
    assert not np.allclose(
        base.sel(year=2006).values, corrupted.sel(year=2006).values
    ), f"{method}: corrupting one year left another year identical — LOO not engaged"


def test_cv_terciles_are_valid_probabilities():
    obs = _obs()
    out = to_tercile_cv(obs + 0.1, obs, method="gaussian_loo")
    s = out.sum("tercile").values
    np.testing.assert_allclose(s[np.isfinite(s)], 1.0, atol=1e-6)
    assert out.min() >= -1e-9
