"""Translating domain objects into the response contract.

One rule governs this module: the *only* path from an ensemble result to a directional
payload runs through :func:`prediction_response`, and it emits a
:class:`~mie.api.schemas.DirectionalCall` only when the ensemble actually published one.
Everything else — suppressed, abstained, below the floor, panel split — becomes
:class:`~mie.api.schemas.InsufficientEvidence` carrying the reason.

Concentrating the decision here is what makes the Phase 10 gate checkable. If routes
each assembled their own payload, "no screen shows a direction without its confidence"
would have to be verified route by route, forever, including routes not written yet.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from mie.api.schemas import (
    CalibrationBin,
    ConfidenceBreakdown,
    DirectionalCall,
    EvidenceItem,
    GateCondition,
    InsufficientEvidence,
    ModelPerformance,
    NewsItem,
    PredictionResponse,
    StateView,
    TimeframeState,
)
from mie.ensemble.confidence import ConfidenceFactors
from mie.ensemble.gate import GateDecision
from mie.ensemble.meta import EnsemblePrediction
from mie.learning.metrics import SliceMetrics
from mie.models.types import Prediction

__all__ = [
    "calibration_bins",
    "gate_conditions",
    "model_performance",
    "news_items",
    "prediction_response",
    "state_view",
]

#: The Brier score of a uniform distribution over three outcomes. Carried into every
#: performance payload so a reader can see immediately whether a model has done better
#: than saying nothing — which, on this data, most have not.
_UNIFORM_BRIER = 2 / 3


def _breakdown(factors: ConfidenceFactors) -> ConfidenceBreakdown:
    return ConfidenceBreakdown(
        value=factors.value,
        skill=round(factors.skill, 4),
        calibration=round(factors.calibration, 4),
        agreement=round(factors.agreement, 4),
        data_quality=round(factors.data_quality, 4),
        sample=round(factors.sample, 4),
        regime_familiarity=round(factors.regime_familiarity, 4),
        limiting_factor=factors.limiting_factor,
        notes=list(factors.notes),
    )


def _evidence(items: Sequence[object]) -> list[EvidenceItem]:
    return [
        EvidenceItem(
            label=getattr(item, "label", ""),
            detail=getattr(item, "detail", ""),
            contribution=float(getattr(item, "contribution", 0.0)),
        )
        for item in items
    ]


def prediction_response(
    result: EnsemblePrediction,
    decision: GateDecision | None = None,
) -> PredictionResponse:
    """Convert an ensemble result into the one shape the API is allowed to return.

    The branch is deliberately conservative: anything other than an unambiguously
    published, actionable call becomes insufficient evidence. There is no third path
    that emits a direction "with caveats" — a caveated direction is still a direction,
    and it would reach a reader as one.
    """
    prediction: Prediction = result.prediction
    horizon = prediction.horizon.label()

    if not result.published or not prediction.is_actionable:
        reasons = list(result.suppressed_because)
        if not reasons:
            reasons = [
                f"confidence {prediction.confidence:.2f} and edge "
                f"{prediction.distribution.directional_edge:+.3f} are below the "
                f"threshold for publishing a direction"
            ]
        if decision is not None:
            reasons.extend(f"gate: {c.name} - {c.detail}" for c in decision.failures[:4])
        return InsufficientEvidence(
            asset=prediction.asset,
            timeframe=str(prediction.timeframe),
            horizon=horizon,
            as_of=prediction.as_of,
            reasons=reasons,
            confidence=_breakdown(result.factors),
            regime=prediction.regime,
            data_quality=prediction.data_quality,
            panel_summary=result.agreement.summary(),
        )

    return DirectionalCall(
        asset=prediction.asset,
        timeframe=str(prediction.timeframe),
        horizon=horizon,
        as_of=prediction.as_of,
        resolves_at=prediction.resolves_at,
        direction=prediction.distribution.most_likely.value,
        probability_up=round(prediction.distribution.up, 6),
        probability_flat=round(prediction.distribution.flat, 6),
        probability_down=round(prediction.distribution.down, 6),
        directional_edge=round(prediction.distribution.directional_edge, 6),
        confidence=prediction.confidence,
        confidence_breakdown=_breakdown(result.factors),
        invalidation=list(prediction.invalidation),
        evidence=_evidence(prediction.evidence),
        counter_evidence=_evidence(prediction.counter_evidence),
        expected_move_pct=prediction.expected_move_pct,
        expected_volatility_pct=prediction.expected_volatility_pct,
        move_threshold_pct=prediction.move_threshold_pct,
        regime=prediction.regime,
        data_quality=prediction.data_quality,
        is_super_prediction=bool(decision and decision.passed),
    )


def gate_conditions(decision: GateDecision) -> list[GateCondition]:
    return [
        GateCondition(name=c.name, passed=c.passed, detail=c.detail) for c in decision.checks
    ]


def state_view(asset: str, state: object) -> StateView:
    """Render a Phase 3 hierarchical state.

    Tolerant of the state object's shape by design: the API should not break because
    the state engine grew a field, and a dashboard that fails to load because one panel
    changed is worse than a panel that renders with a default.
    """
    frames = []
    for entry in getattr(state, "timeframes", []) or []:
        frames.append(
            TimeframeState(
                timeframe=str(getattr(entry, "timeframe", "")),
                direction=str(getattr(entry, "direction", "unknown")),
                strength=float(getattr(entry, "strength", 0.0)),
                confidence=float(getattr(entry, "confidence", 0.0)),
                volatility=str(getattr(entry, "volatility", "unknown")),
                momentum=float(getattr(entry, "momentum", 0.0)),
            )
        )
    return StateView(
        asset=asset.upper(),
        as_of=getattr(state, "as_of", None) or getattr(state, "computed_at", None) or _now(),
        alignment=str(getattr(state, "alignment", "unknown")),
        bias_score=round(float(getattr(state, "confidence", 0.0)) * _bias_sign(state), 4),
        agreement=float(getattr(state, "agreement", 0.0)),
        regime=str(getattr(state, "regime", "unknown")),
        timeframes=frames,
        conflict="; ".join(getattr(state, "conflicts", []) or []),
    )


def model_performance(
    slices: Sequence[SliceMetrics], weights: dict[tuple[str, str], float] | None = None
) -> list[ModelPerformance]:
    lookup = weights or {}
    return [
        ModelPerformance(
            model_id=entry.model_id,
            dimension=entry.dimension,
            value=entry.value,
            outcomes=entry.count,
            brier=entry.brier,
            accuracy=entry.accuracy,
            log_loss=entry.log_loss,
            weight=lookup.get((entry.model_id, entry.value), 0.0),
            has_evidence=entry.has_evidence,
            uniform_brier=_UNIFORM_BRIER,
        )
        for entry in slices
    ]


def calibration_bins(diagram: object) -> list[CalibrationBin]:
    return [
        CalibrationBin(
            lower=b.lower,
            upper=b.upper,
            count=b.count,
            stated=b.mean_predicted,
            observed=b.observed,
            interval_low=b.observed_low,
            interval_high=b.observed_high,
            consistent=b.contains_nominal,
        )
        for b in getattr(diagram, "bins", [])
        if b.count > 0
    ]


def news_items(rows: Sequence[Any]) -> list[NewsItem]:
    items = []
    for row in rows:
        relevance = getattr(row, "relevance", {}) or {}
        items.append(
            NewsItem(
                title=str(getattr(row, "title", "")),
                url=str(getattr(row, "url", "") or ""),
                published_at=row.published_at,
                category=str(getattr(row, "category", "other")),
                sentiment=str(getattr(row, "sentiment", "neutral")),
                importance=float(getattr(row, "importance", 0.0)),
                coverage=int(getattr(row, "coverage", 1)),
                assets=sorted(relevance) if isinstance(relevance, dict) else [],
            )
        )
    return items


def _bias_sign(state: object) -> int:
    """Turn the hierarchy's directional bias into a sign for the bias score."""
    bias = str(getattr(state, "bias", "")).lower()
    if "bull" in bias or bias == "up":
        return 1
    if "bear" in bias or bias == "down":
        return -1
    return 0


def _now():
    from mie.core.timeframes import utcnow

    return utcnow()
