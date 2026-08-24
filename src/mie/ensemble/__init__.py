"""Phase 7: calibration, agreement, confidence and the super-prediction gate.

The layer that decides how much of what the models said is worth publishing — and, on
the evidence measured so far, that the answer is none of it.
"""

from mie.ensemble.agreement import (
    AgreementReport,
    independence_weights,
    measure_agreement,
    overlap_matrix,
)
from mie.ensemble.calibration import (
    CalibrationCurve,
    CalibrationLibrary,
    CalibrationRecord,
    ReliabilityBin,
    ReliabilityDiagram,
    classwise_ece,
    reliability_diagram,
)
from mie.ensemble.confidence import ConfidenceFactors, confidence_from
from mie.ensemble.gate import GateCheck, GateDecision, SuperPredictionGate
from mie.ensemble.meta import EnsembleModel, EnsemblePrediction, SkillWeights

__all__ = [
    "AgreementReport",
    "CalibrationCurve",
    "CalibrationLibrary",
    "CalibrationRecord",
    "ConfidenceFactors",
    "EnsembleModel",
    "EnsemblePrediction",
    "GateCheck",
    "GateDecision",
    "ReliabilityBin",
    "ReliabilityDiagram",
    "SkillWeights",
    "SuperPredictionGate",
    "classwise_ece",
    "confidence_from",
    "independence_weights",
    "measure_agreement",
    "overlap_matrix",
    "reliability_diagram",
]
