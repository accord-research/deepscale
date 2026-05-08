from .registry import get_strategy


def ensemble(forecasts, obs, strategy="uniform", **kwargs):
    """Combine multiple forecasts.

    Each `forecast` is either a DataArray or an `OptimizeResult` (whose
    `.forecast` attribute is the DataArray and whose `.score` attribute
    feeds skill-aware strategies). The strategy is responsible for
    extracting whichever it needs.
    """
    return get_strategy(strategy)().combine(forecasts, obs, **kwargs)
