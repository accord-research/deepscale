"""Combine independent tercile-probability forecasts into one objective forecast.

The generic "combine forecasts" step of a seasonal workflow: given several
already-produced tercile-probability maps (e.g. one MME per predictor experiment,
or several methods), average them — equally or with weights — into a single
outlook. This is the reusable form of the ad-hoc ``xr.concat(...).mean(...)`` that
otherwise gets rewritten in every consumer notebook, and it is grid-aware and
simplex-preserving.

Hierarchy comes for free by composition: combine within a group, then combine the
group results. ACMAD's *component-equal* objective is exactly

    objective = combine_terciles([exp1_mme, exp2_mme, exp3_mme])          # 1/3 each

where each ``exp*_mme`` is itself an equal-weight MME over that experiment's
model x domain members (produced by ``seasonal_mme`` or another
``combine_terciles``). Weighting the three unequally, or nesting further, is just a
matter of the ``weights`` argument.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import xarray as xr

_TERCILE = xr.IndexVariable("tercile", [0, 1, 2])   # 0=below, 1=normal, 2=above


def _as_named_list(components):
    """Normalise a sequence or {name: da} mapping to ([names], [DataArrays])."""
    if isinstance(components, Mapping):
        names = list(components.keys())
        das = list(components.values())
    elif isinstance(components, Sequence):
        das = list(components)
        names = [c.name if getattr(c, "name", None) else f"c{i}" for i, c in enumerate(das)]
    else:
        raise TypeError("components must be a sequence or a {name: DataArray} mapping")
    if len(das) < 1:
        raise ValueError("combine_terciles needs at least one component")
    return names, das


def _target_grid(regrid_to, first):
    """Resolve the lat/lon the components are put on before combining."""
    if regrid_to is None:
        return first.lat.values, first.lon.values
    if isinstance(regrid_to, xr.DataArray):
        return regrid_to.lat.values, regrid_to.lon.values
    lat, lon = regrid_to                     # (lat, lon) arrays
    return np.asarray(lat), np.asarray(lon)


def combine_terciles(components, weights=None, *, regrid_to=None, renormalize=True):
    """Weighted (default equal) combination of tercile-probability forecasts.

    Parameters
    ----------
    components : sequence of DataArray, or {name: DataArray}
        Each a ``(tercile, lat, lon)`` fractional-probability field
        (``tercile=[0,1,2]`` = below/normal/above, summing to ~1 per valid cell;
        cells missing in a component are all-NaN across the three categories).
    weights : sequence or mapping, optional
        Per-component weights (need not sum to 1; they are normalised). Default is
        equal weight — the WMO-style unweighted average.
    regrid_to : DataArray or (lat, lon), optional
        Grid to interpolate every component onto before combining. Default: the
        first component's grid (others are linearly interpolated to it only if
        they differ). Mirrors how the ACMAD/ICPAC objectives downsample a finer
        component onto the common grid.
    renormalize : bool, default True
        Divide the combined probabilities by their tercile sum so every valid
        cell is a proper 3-way simplex.

    Returns
    -------
    DataArray
        The combined ``(tercile, lat, lon)`` forecast on the target grid.

    Notes
    -----
    The average is taken skipping NaN components per cell, so a cell present in
    only some components still combines from those (weights renormalised over the
    present components) — matching ACMAD's per-cell ``nanmean`` behaviour.
    """
    names, das = _as_named_list(components)

    if weights is None:
        w = np.ones(len(das), float)
    elif isinstance(weights, Mapping):
        w = np.array([float(weights[n]) for n in names], float)
    else:
        w = np.asarray([float(x) for x in weights], float)
    if len(w) != len(das):
        raise ValueError("weights length must match number of components")
    if np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be non-negative and not all zero")

    tlat, tlon = _target_grid(regrid_to, das[0])
    aligned = []
    for da in das:
        da = da.transpose("tercile", "lat", "lon")
        same = (da.sizes.get("lat") == len(tlat) and da.sizes.get("lon") == len(tlon)
                and np.array_equal(da.lat.values, tlat)
                and np.array_equal(da.lon.values, tlon))
        aligned.append(da if same else da.interp(lat=tlat, lon=tlon))

    stacked = xr.concat(aligned, dim=xr.IndexVariable("component", names))
    wda = xr.DataArray(w, dims="component", coords={"component": names})

    # Per-cell weighted mean skipping NaN components: sum(w*p)/sum(w present).
    present = stacked.notnull()
    wsum = (stacked * wda).sum("component", skipna=True)
    wtot = (wda * present).sum("component")
    combined = wsum / wtot.where(wtot > 0)

    combined = combined.transpose("tercile", "lat", "lon").assign_coords(tercile=_TERCILE)
    if renormalize:
        total = combined.sum("tercile", skipna=False)
        combined = xr.where(np.isfinite(total) & (total > 0), combined / total, np.nan)
        combined = combined.transpose("tercile", "lat", "lon").assign_coords(tercile=_TERCILE)
    combined.name = "tercile_probability"
    return combined
