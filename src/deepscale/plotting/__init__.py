"""Plot helpers for deepscale outputs.

This subpackage's functions require optional dependencies (matplotlib,
cartopy, rioxarray) that aren't installed by default. Install them with:

    pip install deepscale[plotting]

Importing this package itself does NOT load the optional deps; each
function calls `deepscale._optional.require_optional` lazily.
"""

from .domains import plot_domains
from .skill import plot_skill_maps
from .forecasts import (
    plot_tercile_forecast,
    plot_deterministic_forecast,
    plot_exceedance_probability,
    plot_flex_pdf,
)

__all__ = [
    "plot_domains",
    "plot_skill_maps",
    "plot_tercile_forecast",
    "plot_deterministic_forecast",
    "plot_exceedance_probability",
    "plot_flex_pdf",
]
