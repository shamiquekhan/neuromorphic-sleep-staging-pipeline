"""Model architectures for sleep-stage classification.

The repository ships a single production architecture: the 99,477-parameter
`ImprovedStudent`. There is no teacher model and no distillation dependency.
"""

from .improved_student import ImprovedStudent, count_parameters
from .components import (
    DepthwiseSeparableConv1d,
    LiteMultiResolutionStem,
    ParametricGaborFEB,
)

__all__ = [
    "ImprovedStudent",
    "count_parameters",
    "DepthwiseSeparableConv1d",
    "LiteMultiResolutionStem",
    "ParametricGaborFEB",
]
