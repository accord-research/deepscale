"""Climatological positioning of a value within a historical record.

Three verbs, each reducing over one dimension of a reference record and each
agnostic to whatever dimensions remain. That is what makes them reusable: the
same call positions a gridded seasonal total against 45 years of history
(``(lat, lon)`` survive), an admin-unit total (``region`` survives), or a
single station series (nothing survives).

* :func:`accumulate` -- totals over an arbitrary window of a time axis.
* :func:`percentile_of` -- where a value falls in a reference distribution.
* :func:`rank_of_record` -- the integer rank, so "driest on record" is
  ``rank_of_record(...) == 1``.

Together these cover the "accumulated totals and historical ranks for arbitrary
accumulation windows" presentation layer that monitoring products (CHC's Early
Estimates among them) are built from, without any of them knowing what an
accumulation window is *for*.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import norm

from .time import infer_cadence, season_step, season_times

__all__ = [
    "accumulate",
    "frequency_below",
    "percentile_of",
    "rank_of_record",
    "seasonal_stack",
]

_HOW = {"sum", "mean", "max", "min"}


def seasonal_stack(
    da: xr.DataArray,
    season,
    *,
    time_dim: str = "time",
    cadence: str | None = None,
    years=None,
) -> xr.DataArray:
    """Reshape a continuous time series into one season per year.

    Turns ``(time, ...)`` spanning decades into ``(year, step, ...)``, where
    ``step`` is the ordinal position within the season. That layout is what
    makes years comparable: step 3 of the 1997 season and step 3 of the 2026
    season are the same point in the season, which their calendar timestamps
    never are.

    Parameters
    ----------
    da : xr.DataArray
        A time series with a datetime ``time_dim``. Any other dims survive.
    season : see :func:`deepscale.time.season_bounds`
        The window to extract, e.g. ``"JJAS"``.
    years : sequence of int, optional
        Season start years to extract. Defaults to every year the record could
        cover. Years with no data in the season are dropped; years with partial
        data are kept, padded with NaN.

    Returns
    -------
    xr.DataArray
        ``(year, step, ...)``, with a ``step`` dim long enough for the longest
        season in ``years`` (they can differ by a day across leap years) and a
        ``season_start`` coordinate on ``year``.
    """
    if time_dim not in da.dims:
        raise ValueError(f"{time_dim!r} not found on data with dims {tuple(da.dims)}")
    cadence = cadence or infer_cadence(da[time_dim])

    stamps = pd.DatetimeIndex(np.asarray(da[time_dim].values))
    if years is None:
        # A wraparound season starting in year Y can reach into Y+1, so the
        # earliest possible season year is one before the record's first stamp.
        years = range(int(stamps.year.min()) - 1, int(stamps.year.max()) + 1)
    years = [int(y) for y in years]

    width = max(len(season_times(season, y, cadence)) for y in years)

    slices, kept = [], []
    for year in years:
        steps = season_step(da[time_dim], season, year=year, cadence=cadence)
        inside = steps.values >= 0
        if not inside.any():
            continue
        sub = da.isel({time_dim: inside})
        sub = (
            sub.assign_coords(step=(time_dim, steps.values[inside]))
            .swap_dims({time_dim: "step"})
            .drop_vars(time_dim)
            .reindex(step=np.arange(width))
        )
        slices.append(sub)
        kept.append(year)

    if not slices:
        raise ValueError(f"no data falls inside season {season!r} for years {years}")

    stacked = xr.concat(slices, dim=pd.Index(kept, name="year"))
    return stacked.assign_coords(
        season_start=("year", [season_times(season, y, cadence)[0] for y in kept])
    )


def accumulate(
    da: xr.DataArray,
    *,
    window: int | None = None,
    dim: str = "time",
    how: str = "sum",
    min_count: int | None = None,
) -> xr.DataArray:
    """Accumulate ``da`` along ``dim``.

    Parameters
    ----------
    da : xr.DataArray
        Per-step increments (e.g. dekadal rainfall totals).
    window : int, optional
        Length of a trailing rolling window in steps. ``None`` (the default)
        collapses ``dim`` entirely, returning one total.
    dim : str
        The axis to accumulate along.
    how : {"sum", "mean", "max", "min"}
        The reduction.
    min_count : int, optional
        Minimum number of non-NaN steps required to produce a value. Defaults
        to requiring every step (``window`` for rolling, the full axis length
        otherwise), so a partially-missing accumulation is NaN rather than a
        silent under-count. Pass ``1`` to accumulate whatever is present.

    Returns
    -------
    xr.DataArray
        With ``dim`` dropped (``window is None``) or retained with the same
        length, each stamp holding the accumulation ending there.
    """
    if how not in _HOW:
        raise ValueError(f"how must be one of {sorted(_HOW)}, got {how!r}")
    if dim not in da.dims:
        raise ValueError(f"dim {dim!r} not found on data with dims {tuple(da.dims)}")

    if window is None:
        n = int(da.sizes[dim])
        min_count = n if min_count is None else min_count
        if how == "sum":
            return da.sum(dim, skipna=True, min_count=min_count)
        reduced = getattr(da, how)(dim, skipna=True)
        valid = da.notnull().sum(dim) >= min_count
        return reduced.where(valid)

    if window < 1 or window > da.sizes[dim]:
        raise ValueError(
            f"window must be between 1 and the length of {dim!r} "
            f"({da.sizes[dim]}), got {window}"
        )
    min_periods = window if min_count is None else min_count
    rolling = da.rolling({dim: window}, min_periods=min_periods)
    return getattr(rolling, how)()


def _check_reference(values, climatology, dim):
    """Coerce ``values`` to a DataArray and assert it doesn't carry ``dim``."""
    if not isinstance(values, xr.DataArray):
        values = xr.DataArray(values)
    if dim not in climatology.dims:
        raise ValueError(
            f"dim {dim!r} not found on climatology with dims {tuple(climatology.dims)}"
        )
    if dim in values.dims:
        raise ValueError(
            f"values must not carry the reference dim {dim!r}; it is the axis "
            "being reduced over. Select or accumulate it away first."
        )
    return values


def percentile_of(
    values: xr.DataArray,
    climatology: xr.DataArray,
    *,
    dim: str = "year",
    method: str = "empirical",
) -> xr.DataArray:
    """Position ``values`` in the distribution of ``climatology`` along ``dim``.

    Returns a fraction in ``[0, 1]``: 0.05 means the value sits at the 5th
    percentile of the reference record (drier than 95% of it, for rainfall).

    Parameters
    ----------
    values : xr.DataArray
        Must not have ``dim``; every other dim broadcasts against
        ``climatology``.
    climatology : xr.DataArray
        The reference record, carrying ``dim``.
    method : {"empirical", "weibull", "gaussian"}
        ``"empirical"`` is the mid-rank estimator: the fraction strictly below,
        plus half the fraction tied. It is bounded by ``[0, 1]`` and needs no
        distributional assumption. ``"weibull"`` uses the ``rank / (n + 1)``
        plotting position, which never returns exactly 0 or 1 -- useful when
        the result feeds a transform with infinite tails. ``"gaussian"`` fits a
        normal to the reference and evaluates its CDF, which extrapolates
        beyond the observed range but assumes symmetry (questionable for
        rainfall).

    Notes
    -----
    NaN in ``values`` propagates. NaN in ``climatology`` is excluded from the
    reference, so a cell with a short record is positioned against the years it
    does have; cells with no valid years return NaN.
    """
    values = _check_reference(values, climatology, dim)

    if method == "gaussian":
        mean = climatology.mean(dim, skipna=True)
        std = climatology.std(dim, skipna=True)
        std = xr.where(std < 1e-12, np.nan, std)
        z = (values - mean) / std
        return xr.apply_ufunc(norm.cdf, z, dask="parallelized", keep_attrs=False)

    n_valid = climatology.notnull().sum(dim)
    below = (climatology < values).sum(dim)
    tied = (climatology == values).sum(dim)

    if method == "empirical":
        frac = (below + 0.5 * tied) / n_valid
    elif method == "weibull":
        frac = (below + 0.5 * tied + 0.5) / (n_valid + 1)
    else:
        raise ValueError(
            f"method must be 'empirical', 'weibull' or 'gaussian', got {method!r}"
        )

    frac = frac.where(n_valid > 0)
    # `(clim < values)` is False wherever `values` is NaN, so the comparison
    # silently reports percentile 0 for missing data. Restore the NaN.
    return frac.where(values.notnull())


def frequency_below(
    sample: xr.DataArray,
    climatology: xr.DataArray,
    *,
    q: float = 1.0 / 3.0,
    dim: str = "year",
) -> xr.DataArray:
    """Fraction of ``sample`` below the ``q``-th quantile of ``climatology``.

    At every cell (all dims except ``dim``), the ``q``-th quantile of
    ``climatology`` along ``dim`` sets a threshold, and the return is the share
    of ``sample`` entries along ``dim`` that fall below it -- a value in
    ``[0, 1]``. With ``q=1/3`` and each analog year's seasonal total as
    ``sample``, this is the "below-normal frequency" composite: the fraction of
    analog years that landed in the dry tercile at each location.

    Both ``sample`` and ``climatology`` carry ``dim`` -- they may be the same
    record, or ``sample`` may be a subset of it (the analog years). Every other
    dimension is preserved, so the same call yields a grid, an admin unit, or a
    single station series.

    Parameters
    ----------
    sample : xr.DataArray
        The values whose below-threshold frequency is wanted, carrying ``dim``.
    climatology : xr.DataArray
        The reference record defining the threshold, carrying ``dim``.
    q : float in ``[0, 1]``
        The quantile of ``climatology`` used as the threshold. ``1/3`` is the
        below-normal tercile; ``0.1`` a 1-in-10 dry threshold, and so on.

    Notes
    -----
    NaN in ``sample`` is excluded from the fraction, so a cell is scored over the
    sample years it does have. A cell whose ``climatology`` has no valid year
    (hence no threshold) returns NaN rather than a spurious 0.
    """
    if dim not in sample.dims:
        raise ValueError(f"sample must carry {dim!r}; got {tuple(sample.dims)}")
    if dim not in climatology.dims:
        raise ValueError(
            f"climatology must carry {dim!r}; got {tuple(climatology.dims)}"
        )
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q}")

    threshold = climatology.quantile(q, dim=dim, skipna=True)
    threshold = threshold.drop_vars("quantile", errors="ignore")

    # `.where(sample.notnull())` keeps NaN sample entries out of the mean (a bare
    # `sample < threshold` would count them as False, understating the frequency).
    below = (sample < threshold).where(sample.notnull())
    freq = below.mean(dim, skipna=True)
    return freq.where(threshold.notnull())


def rank_of_record(
    values: xr.DataArray,
    climatology: xr.DataArray,
    *,
    dim: str = "year",
    ascending: bool = True,
) -> xr.DataArray:
    """Rank of ``values`` within ``climatology ∪ {values}`` along ``dim``.

    With ``ascending=True`` (the default) rank 1 is the smallest value, so for
    rainfall ``rank_of_record(...) == 1`` reads "driest on record". With
    ``ascending=False`` rank 1 is the largest.

    The rank is taken over the reference record *including* the value being
    ranked, which is what "driest on record" means when the record contains the
    year in question. When ``values`` is itself one of the reference years this
    is simply its rank within the record; when it is a new value the maximum
    possible rank is ``n + 1``.

    Ties share the better (lower) rank, matching the competition-ranking
    convention: two equal-driest values are both rank 1.
    """
    values = _check_reference(values, climatology, dim)

    better = (climatology < values) if ascending else (climatology > values)
    rank = better.sum(dim) + 1

    n_valid = climatology.notnull().sum(dim)
    rank = rank.where(n_valid > 0)
    return rank.where(values.notnull())
