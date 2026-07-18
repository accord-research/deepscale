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
from .climate import (
    accumulate,
    frequency_below,
    percentile_of,
    rank_of_record,
    seasonal_stack,
)
from .analog import (
    AnalogSet, analogs_from_years, analogs_from_index, analogs_from_field, analogs_where,
)
from .completion import complete, CompletionResult
from .series import quantile_map, error_bounds, ErrorBounds
from .io import write_terciles, tercile_mae
from .combine import combine_terciles
from .plotting.forecasts import plot_tercile_forecast as plot_terciles
from .plotting.scenarios import plot_accumulation_scenarios, plot_index_scatter
from .plotting.maps import natural_earth_borders, plot_field_map, plot_choropleth
from .plotting.styled import TercileStyle, plot_tercile_comparison
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
    "frequency_below",
    "percentile_of",
    "rank_of_record",
    "seasonal_stack",
    "AnalogSet",
    "analogs_from_years",
    "analogs_from_index",
    "analogs_from_field",
    "analogs_where",
    "complete",
    "CompletionResult",
    "quantile_map",
    "error_bounds",
    "ErrorBounds",
    "write_terciles",
    "tercile_mae",
    "combine_terciles",
    "plot_terciles",
    "plot_accumulation_scenarios",
    "plot_index_scatter",
    "plot_field_map",
    "plot_choropleth",
    "natural_earth_borders",
    "TercileStyle",
    "plot_tercile_comparison",
]
