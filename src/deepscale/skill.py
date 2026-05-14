import warnings
from dataclasses import dataclass, field

import xarray as xr

from .registry import get_metric, _METRICS


@dataclass
class SkillReport:
    scores: dict = field(default_factory=dict)
    spatial: dict = field(default_factory=dict)


PRESETS = {
    # WMO Standardized Verification System for Long-Range Forecasts.
    "svslrf": ["rpss", "roc", "reliability"],
    # "all" is resolved at call time so newly-registered metrics are picked up.
    # Sentinel value None means "expand dynamically from the registry".
    "all": None,
}


def _resolve_metrics(metrics):
    """Expand a `metrics=` argument into a concrete list of metric names."""
    if metrics is None:
        return ["rpss"]
    if isinstance(metrics, str):
        if metrics in PRESETS:
            preset = PRESETS[metrics]
            if preset is None:
                # "all": dedupe aliases so each metric class runs at most once
                seen = set()
                names = []
                for name, cls in _METRICS.items():
                    if cls in seen:
                        continue
                    seen.add(cls)
                    names.append(name)
                return names
            return list(preset)
        # Bare string that isn't a preset: treat as a single-metric name
        return [metrics]
    return list(metrics)


def skill(forecast, obs, metrics=None, spatial=False, **kwargs):
    # Under metrics="all" we sweep in every registered metric, including
    # ones with stricter input contracts than the rest (spread_error_*
    # requires a `member` dim). Any ValueError raised at compute time is
    # treated as "this metric doesn't apply to this input" — skip with a
    # warning so the rest of the report is still produced. For any other
    # `metrics=` shape (explicit list, single-name string, other preset),
    # ValueErrors propagate so usage bugs stay loud.
    skip_incompatible = metrics == "all"
    metrics = _resolve_metrics(metrics)

    report = SkillReport()
    for name in metrics:
        metric = get_metric(name)()
        try:
            result = metric.compute(forecast, obs, spatial=spatial, **kwargs)
        except ValueError as exc:
            if not skip_incompatible:
                raise
            warnings.warn(
                f"skill(metrics='all'): skipping {name!r}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        if isinstance(result, dict):
            report.scores.update(result)
            if spatial:
                report.spatial.update(result)
        else:
            if spatial and isinstance(result, xr.DataArray):
                report.spatial[name] = result
                report.scores[name] = float(result.mean())
            else:
                report.scores[name] = result

    return report
