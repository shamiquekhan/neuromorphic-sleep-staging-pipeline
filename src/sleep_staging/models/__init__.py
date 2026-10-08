"""Model architectures for sleep-stage classification.

The repository ships a single production architecture: the 99,477-parameter
`NeuroSleepModel`. There is no teacher model and no distillation dependency.
"""

from .neurosleep_model import NeuroSleepModel, count_parameters
from .components import (
    DepthwiseSeparableConv1d,
    LiteMultiResolutionStem,
    ParametricGaborFEB,
)

__all__ = [
    "NeuroSleepModel",
    "count_parameters",
    "DepthwiseSeparableConv1d",
    "LiteMultiResolutionStem",
    "ParametricGaborFEB",
]
