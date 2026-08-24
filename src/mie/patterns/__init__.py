"""Phase 4: pattern detection and statistical validation."""

from mie.patterns.detectors import DETECTORS, detect_all
from mie.patterns.evaluation import DEFAULT_HORIZONS, PatternEvaluator, ScanResult
from mie.patterns.registry import PatternRegistry
from mie.patterns.sequences import Chain, SequenceMiner, Transition, TransitionMatrix
from mie.patterns.similarity import (
    COMPARISON_FEATURES,
    Analogue,
    SimilarityEngine,
    SimilarityResult,
)
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
    "COMPARISON_FEATURES",
    "DEFAULT_HORIZONS",
    "DETECTORS",
    "PATTERN_DIRECTIONS",
    "Analogue",
    "Chain",
    "Detection",
    "Outcome",
    "PatternDirection",
    "PatternEvaluator",
    "PatternKind",
    "PatternRegistry",
    "PatternStats",
    "ProportionEstimate",
    "ScanResult",
    "SequenceMiner",
    "SimilarityEngine",
    "SimilarityResult",
    "Transition",
    "TransitionMatrix",
    "benjamini_hochberg",
    "compare_to_baseline",
    "detect_all",
    "two_proportion_test",
    "wilson_interval",
]
