from abc import ABC, abstractmethod
import xarray as xr

class StrategyBase(ABC):
    @abstractmethod
    def combine(self, forecasts: list[xr.DataArray], obs: xr.DataArray = None, **kwargs) -> xr.DataArray:
        ...
