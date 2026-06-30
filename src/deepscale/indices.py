"""Teleconnection / SST indices used as scalar predictors for forecasting.

A predictor-based seasonal forecast (e.g. the logistic/WVG stream) does not
consume a gridded model field; it consumes a single index value per year. This
module turns an SST field into such an index.

The headline construction is the Western-V Gradient (WVG, Funk et al.), the
3-box form (the ICPAC primary definition):

    WVG = z(Nino3.4) - (z(WNP) + z(WEP) + z(WSP)) / 3

where each ``z(box)`` is the standardized box-mean SST. ICPAC's own code is
internally inconsistent here (the primary definition is 3-box but a comment and
a subset of model blocks use a 2-box form); deepscale defaults to the 3-box form
and exposes the 2-box variant as ``Index.named("wvg2")``. Standardization uses the
mean/std of a reference (climatology) period so that hindcast and forecast
indices live on the same scale:

    idx_hcst = Index.named("wvg").reduce(sst_hcst)
    idx_fcst = Index.named("wvg").reduce(sst_fcst, climatology=sst_hcst)

The box means are plain (unweighted) spatial averages, matching the operational
ICPAC / WASS2S reference implementations.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import xarray as xr

# Box definitions in degrees: (south, north, west, east). Longitudes are in the
# 0-360 convention (e.g. Nino3.4 spans 190E-240E). reduce() normalizes whatever
# convention the input SST uses, so callers may pass -180..180 or 0..360 data.
_REGIONS = {
    "nino34": dict(south=-5, north=5, west=190, east=240),
    "nino4": dict(south=-5, north=5, west=160, east=210),
    "wep": dict(south=-15, north=20, west=120, east=160),   # West Equatorial Pacific
    "wnp": dict(south=20, north=35, west=160, east=210),     # West North Pacific
    "wsp": dict(south=-30, north=-15, west=155, east=210),   # West South Pacific
}


def _wvg3_combine(z: dict[str, xr.DataArray]) -> xr.DataArray:
    """3-box Western-V Gradient: z(Nino3.4) - mean(z(WNP), z(WEP), z(WSP))."""
    return z["nino34"] - (z["wnp"] + z["wep"] + z["wsp"]) / 3.0


def _wvg2_combine(z: dict[str, xr.DataArray]) -> xr.DataArray:
    """2-box Western-V Gradient variant: z(Nino3.4) - mean(z(WNP), z(WEP))."""
    return z["nino34"] - (z["wnp"] + z["wep"]) / 2.0


# name -> (regions used, combine fn). Single-box indices just return the box z.
_INDICES: dict[str, tuple[tuple[str, ...], Callable[[dict], xr.DataArray]]] = {
    "wvg": (("nino34", "wnp", "wep", "wsp"), _wvg3_combine),    # 3-box (default)
    "wvg2": (("nino34", "wnp", "wep"), _wvg2_combine),          # 2-box variant
    "nino34": (("nino34",), lambda z: z["nino34"]),
    "nino4": (("nino4",), lambda z: z["nino4"]),
}


@dataclass(frozen=True)
class Index:
    """A named teleconnection index that reduces an SST field to a 1-D series.

    Construct via :meth:`named`. Call :meth:`reduce` to turn an SST DataArray
    into a per-time index series.
    """

    name: str
    regions: Mapping[str, object]
    _combine: Callable[[dict], xr.DataArray]

    @classmethod
    def named(cls, name: str) -> "Index":
        """Return a predefined index by name (e.g. ``"wvg"``, ``"nino34"``)."""
        key = name.lower()
        if key not in _INDICES:
            raise KeyError(
                f"Unknown index {name!r}. Known indices: {sorted(_INDICES)}."
            )
        regions, combine = _INDICES[key]
        return cls(
            name=key,
            regions={region: _REGIONS[region] for region in regions},
            _combine=combine,
        )

    @classmethod
    def custom(
        cls,
        *,
        name: str,
        regions: Mapping[str, object],
        combine: Callable[[dict[str, xr.DataArray]], xr.DataArray],
    ) -> "Index":
        """Build a custom index from named regions and a combine function.

        ``regions`` accepts the same simple bbox convention used by Rosetta:
        ``[lat_s, lat_n, lon_w, lon_e]``. If Rosetta is importable, shapefile
        paths and shapely/geopandas geometries are also accepted via
        ``rosetta.region.resolve_region``.

        ``combine`` receives standardized regional series keyed by region name.
        For example, WVG is:

            ``lambda z: z["nino34"] - (z["wnp"] + z["wep"] + z["wsp"]) / 3``
        """
        if not regions:
            raise ValueError("Index.custom requires at least one region.")
        missing = [key for key in regions if not isinstance(key, str) or not key]
        if missing:
            raise ValueError("Index.custom region names must be non-empty strings.")
        return cls(name=name, regions=dict(regions), _combine=combine)

    # -- internals ---------------------------------------------------------
    @staticmethod
    def _spatial_dims(sst: xr.DataArray) -> tuple[str, str]:
        from ._spatial import spatial_dims

        return spatial_dims(sst, context="Index.reduce")

    @staticmethod
    def _resolve_region(region) -> tuple[list[float], object | None]:
        if isinstance(region, Mapping):
            return [
                float(region["south"]),
                float(region["north"]),
                float(region["west"]),
                float(region["east"]),
            ], None
        try:
            seq = list(region)
        except TypeError:
            seq = None
        if seq is not None and len(seq) == 4:
            return [float(v) for v in seq], None

        try:
            from rosetta.region import resolve_region
        except ImportError as e:
            raise TypeError(
                "Index regions must be bbox-like [lat_s, lat_n, lon_w, lon_e] "
                "unless Rosetta is installed for shapefile/geometry support."
            ) from e
        return resolve_region(region)

    @staticmethod
    def _mask_geometry(sub: xr.DataArray, geometry, lat: str, lon: str) -> xr.DataArray:
        try:
            from rosetta.normalize import clip_to_geometry
        except ImportError as e:
            raise TypeError(
                "Index geometry regions require Rosetta's geometry clipping "
                "support. Use a bbox region or install Rosetta."
            ) from e
        ds = sub.to_dataset(name="_index_source")
        clipped = clip_to_geometry(ds, geometry)
        return clipped["_index_source"]

    def _box_series(self, sst: xr.DataArray) -> dict[str, xr.DataArray]:
        """Plain regional-mean SST per region, reduced over space (and member)."""
        lat, lon = self._spatial_dims(sst)
        # Normalize longitude to 0-360 so the box bounds (0-360) always apply.
        lon360 = (sst[lon] % 360)
        sst = sst.assign_coords({lon: lon360}).sortby(lon)
        reduce_dims = [lat, lon]
        if "member" in sst.dims:
            reduce_dims.append("member")
        out = {}
        for name, region in self.regions.items():
            bbox, geometry = self._resolve_region(region)
            lat_s, lat_n, lon_w, lon_e = bbox
            lon_w, lon_e = lon_w % 360, lon_e % 360
            if lon_w <= lon_e:
                lon_mask = (sst[lon] >= lon_w) & (sst[lon] <= lon_e)
            else:
                lon_mask = (sst[lon] >= lon_w) | (sst[lon] <= lon_e)
            sub = sst.where(
                (sst[lat] >= lat_s) & (sst[lat] <= lat_n) & lon_mask
            )
            if geometry is not None:
                sub = self._mask_geometry(sub, geometry, lat, lon)
            out[name] = sub.mean(reduce_dims, skipna=True)
        return out

    def reduce(self, sst: xr.DataArray, climatology: xr.DataArray | None = None) -> xr.DataArray:
        """Reduce an SST field to the index series.

        Parameters
        ----------
        sst : xr.DataArray
            SST with lat/lon dims (named lat/latitude/Y or lon/longitude/X) and
            usually a time/year dim. An optional ``member`` dim is averaged out.
        climatology : xr.DataArray, optional
            Reference SST used for the standardization mean/std. Pass the
            hindcast SST when reducing a forecast year so both indices share a
            scale. Defaults to ``sst`` itself.

        Returns
        -------
        xr.DataArray
            The 1-D index series (over the time/year dim of ``sst``), or a
            scalar if ``sst`` has no time dim.
        """
        boxes = self._box_series(sst)
        clim_boxes = boxes if climatology is None else self._box_series(climatology)

        # Standardize each box against the climatology box stats. The reference
        # dim is whatever non-spatial dim survived (time/year); reduce over all
        # remaining dims so the mean/std are scalars per region.
        z = {}
        for region, series in boxes.items():
            ref = clim_boxes[region]
            mean = ref.mean(skipna=True)
            std = ref.std(skipna=True)
            std = xr.where(std < 1e-12, 1e-12, std)
            z[region] = (series - mean) / std

        index = self._combine(z)
        index.name = self.name
        return index
