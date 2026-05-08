"""Skill-weighted ensemble strategy: weights proportional to per-member skill.

Reference: Krishnamurti et al. (1999), Science — "Improved weather and seasonal
climate forecasts from multimodel superensemble predictions" — uses scores
on hindcasts as weights for combining forecasts. The original paper uses
regression weights (closer to the `regression_based` strategy); the simpler
proportional-to-skill scheme implemented here is the common practitioner
shorthand.

Negative skill scores get clipped to zero — members with worse-than-climatology
skill should not contribute. If every member's clipped weight is zero, the
strategy falls back to a uniform combination so the call doesn't divide by zero.
"""

import numpy as np

from .base import StrategyBase
from ..registry import register_strategy
from .uniform import _as_array
from .drop_worst import _resolve_scores


@register_strategy("skill_weighted")
class SkillWeightedStrategy(StrategyBase):
    def combine(self, forecasts, obs=None, scores=None, weights=None, **kwargs):
        arrays = [_as_array(f) for f in forecasts]
        if weights is not None:
            if len(weights) != len(forecasts):
                raise ValueError(
                    f"weights has length {len(weights)}, but {len(forecasts)} forecasts."
                )
            w = np.asarray(weights, dtype=float)
        else:
            w = np.asarray(_resolve_scores(forecasts, scores), dtype=float)
            # Members with non-positive skill don't contribute.
            w = np.clip(w, a_min=0.0, a_max=None)

        total = float(w.sum())
        if total <= 0.0:
            # All-zero (or all-negative-clipped) weights → uniform fallback.
            return sum(arrays) / len(arrays)
        w = w / total

        result = arrays[0] * w[0]
        for arr, wi in zip(arrays[1:], w[1:]):
            result = result + arr * wi
        return result
