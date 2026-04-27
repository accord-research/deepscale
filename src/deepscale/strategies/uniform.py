from .base import StrategyBase
from ..registry import register_strategy

@register_strategy("uniform")
class UniformStrategy(StrategyBase):
    def combine(self, forecasts, obs=None, **kwargs):
        return sum(forecasts) / len(forecasts)
