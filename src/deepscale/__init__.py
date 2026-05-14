from .downscale import downscale
from .optimize import optimize
from .ensemble import ensemble
from .skill import skill, SkillReport
from .compare import skill_compare, ComparisonReport
from . import methods     # trigger registration
from . import metrics     # trigger registration
from . import strategies  # trigger registration

__all__ = [
    "downscale",
    "optimize",
    "ensemble",
    "skill",
    "SkillReport",
    "skill_compare",
    "ComparisonReport",
]
