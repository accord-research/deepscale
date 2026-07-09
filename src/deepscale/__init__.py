from .downscale import downscale
from .optimize import optimize
from .ensemble import ensemble
from .skill import skill, SkillReport
from .compare import skill_compare, ComparisonReport
from .pev import prediction_error_variance
from .flex_forecast import flex_forecast, FlexForecastResult
from .pipelines import seasonal_mme, SeasonalMMEResult
from .indices import Index
from .calibrate import calibrate, LogitConfig
from .climate import accumulate, percentile_of, rank_of_record
from .series import quantile_map, error_bounds, ErrorBounds
from .io import write_terciles, tercile_mae
from .plotting.forecasts import plot_tercile_forecast as plot_terciles
from . import methods     # trigger registration
from . import metrics     # trigger registration
from . import strategies  # trigger registration
from . import time        # calendar / season-step utilities
from . import training     # train / inference separation (§10.2)
from .training import train

__all__ = [
    "downscale",
    "train",
    "optimize",
    "ensemble",
    "skill",
    "SkillReport",
    "skill_compare",
    "ComparisonReport",
    "prediction_error_variance",
    "flex_forecast",
    "FlexForecastResult",
    "seasonal_mme",
    "SeasonalMMEResult",
    "Index",
    "calibrate",
    "LogitConfig",
    "accumulate",
    "percentile_of",
    "rank_of_record",
    "quantile_map",
    "error_bounds",
    "ErrorBounds",
    "write_terciles",
    "tercile_mae",
    "plot_terciles",
]
