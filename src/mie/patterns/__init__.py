"""Phase 4: pattern detection and statistical validation."""

from mie.patterns.detectors import DETECTORS, detect_all
from mie.patterns.evaluation import DEFAULT_HORIZONS, PatternEvaluator, ScanResult
from mie.patterns.statistics import (
    ProportionEstimate,
    benjamini_hochberg,
    compare_to_baseline,
    two_proportion_test,
    wilson_interval,
)
from mie.patterns.types import (
    PATTERN_DIRECTIONS,
    Detection,
    Outcome,
    PatternDirection,
    PatternKind,
    PatternStats,
)

__all__ = [
    "DEFAULT_HORIZONS",
    "DETECTORS",
    "PATTERN_DIRECTIONS",
    "Detection",
    "Outcome",
    "PatternDirection",
    "PatternEvaluator",
    "PatternKind",
    "PatternStats",
    "ProportionEstimate",
    "ScanResult",
    "benjamini_hochberg",
    "compare_to_baseline",
    "detect_all",
    "two_proportion_test",
    "wilson_interval",
]
