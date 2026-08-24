"""The meta-model: combining models that have earned a weight, and no others.

An ensemble is usually sold as free accuracy — average enough forecasters and the
errors cancel. That is true when the forecasters have skill and their errors are
independent. When they do not, averaging produces something worse than any input: a
number with the *appearance* of consensus, whose confidence grows with the number of
models rather than with the evidence.

So weights here come from one place only — measured out-of-sample skill against
climatology, per regime, significant after correction across the whole family of slices
tested. A model with no demonstrated skill in the current regime receives weight zero
and does not contribute. Not a small weight: zero.

The consequence, on the data this repository has actually measured, is that the
ensemble publishes nothing. Phase 6 found no model beating climatology on any slice, so
every weight is zero, so :class:`EnsembleModel` abstains. That is the system working.
The alternative — a floor weight, an equal-weight fallback, "some signal is better than
none" — would produce a confident-looking output backed by nothing, which is the exact
failure mode the whole design is built to prevent.

The machinery is nonetheless real and tested: given models with genuine skill it
weights them correctly, pools them, calibrates the result and publishes with a
confidence that tracks the evidence. The tests demonstrate both directions, because a
component that only ever returns "insufficient evidence" is indistinguishable from one
that is broken.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from mie.core.logging import get_logger
from mie.ensemble.agreement import (
    AgreementReport,
    independence_weights,
    measure_agreement,
)
from mie.ensemble.calibration import CalibrationLibrary
from mie.ensemble.confidence import ConfidenceFactors, confidence_from
from mie.models.base import PredictionContext, Predictor
from mie.models.types import Distribution, Outcome, Prediction, PredictionEvidence

log = get_logger(__name__)

__all__ = ["EnsembleModel", "EnsemblePrediction", "SkillWeights"]

#: Skill below this is treated as zero even when significant. A statistically real
#: edge of 0.002 Brier is not a usable one, and letting it through would mean the gate
#: is testing detectability rather than usefulness.
_MIN_USABLE_SKILL = 0.01


@dataclass(slots=True)
class SkillWeights:
    """Per-model, per-regime weights derived from measured skill.

    Built from a Phase 6 :class:`~mie.models.evaluation.EvaluationReport` rather than
    configured, so there is no path by which a model can be given weight without having
    earned it in a walk-forward test.
    """

    #: (model_id, regime) -> weight. Absent means zero.
    weights: dict[tuple[str, str], float] = field(default_factory=dict)
    #: (model_id, regime) -> evaluation points behind that weight.
    samples: dict[tuple[str, str], int] = field(default_factory=dict)
    #: regime -> total evaluation points observed in it.
    regime_samples: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_report(cls, report: object) -> SkillWeights:
        """Derive weights from an evaluation report.

        Only slices that pass the full Phase 6 gate — enough evidence, skill above the
        floor, and significance surviving Benjamini-Hochberg across every slice tested
        — produce a non-zero weight.
        """
        instance = cls()
        regime_totals: dict[str, int] = defaultdict(int)
        for score in getattr(report, "scores", []):
            key = (score.model_id, score.regime)
            instance.samples[key] = score.predictions
            regime_totals[score.regime] = max(regime_totals[score.regime], score.predictions)
            if score.beats_baseline and score.skill >= _MIN_USABLE_SKILL:
                instance.weights[key] = round(score.skill, 5)
        instance.regime_samples = dict(regime_totals)
        return instance

    def weight_for(self, model_id: str, regime: str) -> float:
        """Weight in this regime, falling back to the pooled slice.

        The fallback exists because regime slices are small and a model with broad
        skill should not be silenced by an under-observed regime. It cannot manufacture
        a weight: the pooled slice had to pass the same gate.
        """
        specific = self.weights.get((model_id, regime))
        if specific is not None:
            return specific
        return self.weights.get((model_id, "all"), 0.0)

    def samples_for(self, model_id: str, regime: str) -> int:
        return self.samples.get((model_id, regime)) or self.samples.get((model_id, "all"), 0)

    @property
    def any_skill(self) -> bool:
        return any(weight > 0 for weight in self.weights.values())

    def skilled_models(self) -> set[str]:
        return {model_id for (model_id, _), weight in self.weights.items() if weight > 0}

    def summary(self) -> str:
        if not self.any_skill:
            return "no model carries a non-zero weight: none demonstrated skill"
        parts = [
            f"{model_id}/{regime}={weight:.4f}"
            for (model_id, regime), weight in sorted(self.weights.items())
            if weight > 0
        ]
        return "weights: " + ", ".join(parts)


@dataclass(slots=True)
class EnsemblePrediction:
    """The ensemble's output, with everything behind it kept attached."""

    prediction: Prediction
    members: list[Prediction] = field(default_factory=list)
    agreement: AgreementReport = field(default_factory=AgreementReport)
    factors: ConfidenceFactors = field(default_factory=ConfidenceFactors)
    contributions: dict[str, float] = field(default_factory=dict)
    #: Why nothing was published, if nothing was.
    suppressed_because: list[str] = field(default_factory=list)

    @property
    def published(self) -> bool:
        return not self.suppressed_because and self.prediction.is_actionable

    def summary(self) -> str:
        if not self.published:
            reason = self.suppressed_because[0] if self.suppressed_because else "below threshold"
            return f"{self.prediction.asset}: insufficient evidence - {reason}"
        return self.prediction.summary()


class EnsembleModel(Predictor):
    """Weighted linear pool of models that have demonstrated skill."""

    model_id = "ensemble"
    version = "1"

    def __init__(
        self,
        members: Sequence[Predictor],
        weights: SkillWeights | None = None,
        calibration: CalibrationLibrary | None = None,
    ) -> None:
        self.members = list(members)
        self.weights = weights or SkillWeights()
        self.calibration = calibration or CalibrationLibrary()
        self.warmup_bars = max((m.warmup_bars for m in self.members), default=100)

    def inputs_used(self) -> frozenset[str]:
        if not self.members:
            return frozenset()
        return frozenset().union(*(m.inputs_used() for m in self.members))

    def member_inputs(self) -> Mapping[str, frozenset[str]]:
        return {m.model_id: m.inputs_used() for m in self.members}

    def predict(self, context: PredictionContext) -> Prediction:
        return self.predict_detailed(context).prediction

    def predict_detailed(self, context: PredictionContext) -> EnsemblePrediction:
        """Run the panel and combine it, keeping the full derivation."""
        members: list[Prediction] = []
        for model in self.members:
            try:
                members.append(model.predict(context))
            except Exception as exc:
                log.warning("member_failed", model=model.model_id, error=str(exc)[:200])

        result = EnsemblePrediction(
            prediction=self.abstain(context, "no member produced a prediction"),
            members=members,
        )
        if not members:
            result.suppressed_because = ["every member model failed or was unavailable"]
            return result

        calibrated: dict[str, Distribution] = {}
        calibrated_in_regime = False
        best_improvement = 0.0
        has_any_record = False
        for member in members:
            try:
                distribution, record = self.calibration.calibrate(
                    member.model_id, member.regime, member.distribution, as_of=context.as_of
                )
            except ValueError:
                # The only legitimate calibration for this model was fitted on a window
                # that includes this very point. Applying it would be look-ahead, so at
                # this instant the model is simply uncalibrated — which is a weaker
                # position, not an excuse to use the curve anyway.
                log.debug(
                    "calibration_not_yet_applicable",
                    model=member.model_id,
                    as_of=context.as_of.isoformat(),
                )
                calibrated[member.model_id] = member.distribution
                continue
            calibrated[member.model_id] = distribution
            if record is not None:
                has_any_record = True
                if record.is_usable:
                    best_improvement = max(best_improvement, record.improvement)
                    if self.calibration.has_regime_record(member.model_id, context.regime):
                        calibrated_in_regime = True

        # Agreement is measured on the *calibrated* views: calibration can change a
        # model's direction when it was systematically biased, and voting the raw
        # numbers would count a bias the system has already corrected.
        recalibrated = [
            member.model_copy(update={"distribution": calibrated[member.model_id]})
            for member in members
        ]
        agreement = measure_agreement(recalibrated, self.member_inputs())
        result.agreement = agreement

        contributions = self._contributions(members, context.regime)
        result.contributions = contributions
        total_weight = sum(contributions.values())

        best_skill = max((self.weights.weight_for(m.model_id, context.regime) for m in members), default=0.0)
        samples = max((self.weights.samples_for(m.model_id, context.regime) for m in members), default=0)

        factors = confidence_from(
            best_skill=best_skill,
            skill_is_significant=total_weight > 0,
            has_calibration=has_any_record,
            calibration_in_regime=calibrated_in_regime,
            calibration_improvement=best_improvement,
            consensus_share=agreement.consensus_share,
            effective_agreement=agreement.effective_agreement,
            family_count=len(self.members),
            data_quality=context.data_quality,
            evaluation_samples=samples,
            regime_samples=self.weights.regime_samples.get(context.regime, 0),
        )
        result.factors = factors

        if total_weight <= 0:
            result.suppressed_because = [
                "no member has demonstrated out-of-sample skill against climatology "
                "in this regime, so there is nothing to weight"
            ]
            result.prediction = self._abstention(context, agreement, factors, result.suppressed_because)
            return result

        pooled = _linear_pool(
            {model_id: calibrated[model_id] for model_id in contributions}, contributions
        )

        reasons: list[str] = []
        if agreement.is_split:
            # A split panel is not averaged into a confident middle. Publishing the
            # mean of two contradictory views would report agreement that does not
            # exist, which is the specific thing §12's gate forbids.
            reasons.append(
                f"models disagree: only {agreement.consensus_share:.0%} of weighted "
                f"votes behind {agreement.majority.value if agreement.majority else 'any direction'}"
            )
        if not factors.publishable:
            reasons.append(
                f"confidence {factors.value:.2f} below the publication floor "
                f"(limited by {factors.limiting_factor})"
            )

        if reasons:
            result.suppressed_because = reasons
            result.prediction = self._abstention(context, agreement, factors, reasons)
            return result

        result.prediction = self.build(
            context,
            distribution=pooled,
            # Confidence is the measured factor product, not derived from the pooled
            # probabilities. A sharp distribution built from unreliable models must not
            # be able to talk itself into confidence. Data quality is already one of
            # those factors, so it is not applied a second time.
            confidence=factors.value,
            apply_data_quality=False,
            evidence=self._evidence(members, contributions, agreement),
            counter_evidence=self._counter_evidence(members, agreement),
            invalidation=self._invalidation(context, pooled),
        )
        return result

    # ------------------------------------------------------------------ internals

    def _contributions(self, members: Sequence[Prediction], regime: str) -> dict[str, float]:
        """Weight per member: measured skill, discounted by shared inputs.

        Abstaining members are excluded rather than pooled toward uniform: a model with
        nothing to say should not drag the ensemble toward the middle, because that
        would be indistinguishable from it actively predicting "flat".
        """
        independence = independence_weights(self.member_inputs())
        contributions: dict[str, float] = {}
        for member in members:
            if member.confidence <= 0.0:
                continue
            skill = self.weights.weight_for(member.model_id, regime)
            if skill <= 0:
                continue
            contributions[member.model_id] = round(
                skill * independence.get(member.model_id, 1.0), 6
            )
        return contributions

    def _abstention(
        self,
        context: PredictionContext,
        agreement: AgreementReport,
        factors: ConfidenceFactors,
        reasons: Sequence[str],
    ) -> Prediction:
        """A published non-answer, carrying its reasoning.

        Deliberately not a bare uniform distribution: "insufficient evidence" is a
        result, and it is more useful when it says which condition failed.
        """
        prediction = self.abstain(context, reasons[0] if reasons else "insufficient evidence")
        return prediction.model_copy(
            update={
                "evidence": [
                    PredictionEvidence(label="agreement", detail=agreement.summary()),
                    PredictionEvidence(label="confidence", detail=factors.explain()),
                ],
                "counter_evidence": [
                    PredictionEvidence(label="suppressed", detail=reason)
                    for reason in reasons[:5]
                ],
            }
        )

    def _evidence(
        self,
        members: Sequence[Prediction],
        contributions: Mapping[str, float],
        agreement: AgreementReport,
    ) -> list[PredictionEvidence]:
        evidence = [
            PredictionEvidence(
                label="weighted consensus",
                detail=agreement.summary(),
                contribution=agreement.consensus_share,
            )
        ]
        ranked = sorted(contributions.items(), key=lambda pair: -pair[1])
        by_id = {m.model_id: m for m in members}
        for model_id, weight in ranked[:5]:
            member = by_id.get(model_id)
            if member is None:
                continue
            evidence.append(
                PredictionEvidence(
                    label=model_id,
                    detail=f"{member.distribution} (weight {weight:.4f})",
                    contribution=member.distribution.directional_edge,
                )
            )
        return evidence

    def _counter_evidence(
        self, members: Sequence[Prediction], agreement: AgreementReport
    ) -> list[PredictionEvidence]:
        """Dissent and abstention, published rather than dropped.

        §25 asks for disagreement to be shown. A consensus reported without its
        dissenters is a different claim from the one the panel actually made.
        """
        by_id = {m.model_id: m for m in members}
        counter = [
            PredictionEvidence(
                label=f"{model_id} dissents",
                detail=str(by_id[model_id].distribution) if model_id in by_id else "",
                contribution=(
                    by_id[model_id].distribution.directional_edge if model_id in by_id else 0.0
                ),
            )
            for model_id in agreement.dissenting[:4]
        ]
        if agreement.abstained:
            counter.append(
                PredictionEvidence(
                    label="abstained",
                    detail=", ".join(sorted(agreement.abstained)[:6]),
                )
            )
        return counter

    def _invalidation(self, context: PredictionContext, pooled: Distribution) -> list[str]:
        threshold = context.threshold_pct
        direction = pooled.most_likely
        price = context.price
        conditions = [
            f"a close beyond {threshold:.2f}% against {direction.value} within the horizon",
            "a regime change away from " + context.regime,
            "data quality falling below 0.8 for the underlying feed",
        ]
        if price > 0 and direction is not Outcome.FLAT:
            level = price * (1 - threshold / 100) if direction is Outcome.UP else price * (
                1 + threshold / 100
            )
            conditions.insert(0, f"price crossing {level:.2f}")
        return conditions


def _linear_pool(
    distributions: Mapping[str, Distribution], weights: Mapping[str, float]
) -> Distribution:
    """Weighted linear pool.

    Linear rather than logarithmic: the log pool is sharper, and sharpening a set of
    forecasts whose calibration is uncertain is precisely the wrong direction. The
    linear pool is conservative — its entropy is at least the weighted mean of its
    inputs' — which is the right bias when the inputs may be wrong together.
    """
    total = sum(weights.get(name, 0.0) for name in distributions)
    if total <= 0:
        return Distribution.uniform()
    up = flat = down = 0.0
    for name, distribution in distributions.items():
        share = weights.get(name, 0.0) / total
        up += distribution.up * share
        flat += distribution.flat * share
        down += distribution.down * share
    return Distribution(up=up, flat=flat, down=down)
