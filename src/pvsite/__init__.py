"""Multi-objective site selection for utility-scale photovoltaic portfolios."""

__version__ = "0.1.0"

from .candidates import RegionSpec, generate
from .model import Problem, ProblemConfig, site_economics
from .nsga2 import GAConfig, NSGA2

__all__ = [
    "RegionSpec", "generate", "Problem", "ProblemConfig",
    "site_economics", "GAConfig", "NSGA2",
]
