from .registry import get_strategy

def ensemble(forecasts, obs, strategy="uniform", **kwargs):
    """Combine multiple forecasts. Each forecast is an OptimizeResult or DataArray."""
    arrays = [f.forecast if hasattr(f, "forecast") else f for f in forecasts]
    return get_strategy(strategy)().combine(arrays, obs, **kwargs)
