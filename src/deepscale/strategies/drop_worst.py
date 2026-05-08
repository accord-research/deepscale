"""Drop-worst ensemble strategy: discard the N lowest-skill members, average the rest.

Reference: Weigel, Liniger & Appenzeller (2008), QJRMS — "Can multi-model
combination really enhance the prediction skill of probabilistic ensemble
forecasts?" — argues that dropping the worst few members of an MME often
beats uniform averaging because uniformly-bad members add noise.
"""

from .base import StrategyBase
from ..registry import register_strategy
from .uniform import _as_array


def _resolve_scores(forecasts, scores_kwarg):
    """Pull a score per forecast, preferring an explicit kwarg over OptimizeResult."""
    if scores_kwarg is not None:
        if len(scores_kwarg) != len(forecasts):
            raise ValueError(
                f"scores has length {len(scores_kwarg)}, but {len(forecasts)} forecasts."
            )
        return list(scores_kwarg)
    scores = []
    for f in forecasts:
        if not hasattr(f, "score"):
            raise ValueError(
                "drop_worst needs a per-forecast score: pass `scores=[...]` "
                "or supply OptimizeResult objects with a `.score` attribute."
            )
        scores.append(f.score)
    return scores


@register_strategy("drop_worst")
class DropWorstStrategy(StrategyBase):
    def combine(self, forecasts, obs=None, n_drop=1, scores=None, **kwargs):
        if n_drop >= len(forecasts):
            raise ValueError(
                f"n_drop={n_drop} would leave nothing to combine "
                f"(have {len(forecasts)} forecasts)."
            )
        scores = _resolve_scores(forecasts, scores)
        # Indexes of the n_drop lowest scores. argsort is ascending → take prefix.
        order = sorted(range(len(scores)), key=lambda i: scores[i])
        keep = sorted(order[n_drop:])
        kept_arrays = [_as_array(forecasts[i]) for i in keep]
        return sum(kept_arrays) / len(kept_arrays)
