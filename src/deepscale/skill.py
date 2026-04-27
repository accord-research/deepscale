from dataclasses import dataclass, field
import xarray as xr
from .registry import get_metric

@dataclass
class SkillReport:
    scores: dict = field(default_factory=dict)
    spatial: dict = field(default_factory=dict)

def skill(forecast, obs, metrics=None, spatial=False, **kwargs):
    if metrics is None:
        metrics = ["rpss"]

    report = SkillReport()
    for name in metrics:
        metric = get_metric(name)()
        result = metric.compute(forecast, obs, spatial=spatial, **kwargs)
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
