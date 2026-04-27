from abc import ABC, abstractmethod
import xarray as xr

class MethodBase(ABC):
    @abstractmethod
    def fit(self, hindcast, obs, **kwargs):
        ...

    @abstractmethod
    def predict(self, forecast, **kwargs):
        ...
