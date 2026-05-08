from .base import StrategyBase
from ..registry import register_strategy


def _as_array(f):
    return f.forecast if hasattr(f, "forecast") else f


@register_strategy("uniform")
class UniformStrategy(StrategyBase):
    def combine(self, forecasts, obs=None, **kwargs):
        arrays = [_as_array(f) for f in forecasts]
        return sum(arrays) / len(arrays)
