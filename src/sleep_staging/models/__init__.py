"""Model architectures for sleep-stage classification."""

from .improved_student import ImprovedStudent, count_parameters
from .improved_teacher import ImprovedTeacher
from .components import (
    DepthwiseSeparableConv1d,
    LiteMultiResolutionStem,
    ParametricGaborFEB,
)

__all__ = [
    "ImprovedStudent",
    "ImprovedTeacher",
    "count_parameters",
    "DepthwiseSeparableConv1d",
    "LiteMultiResolutionStem",
    "ParametricGaborFEB",
]
