"""The super-prediction gate.

A "super prediction" is the strongest thing this system is willing to say, and §12
defines it by conjunction: at least six of eight independent model families agreeing,
**and** a calibration record in the current regime. Disagreement suppresses the signal
entirely rather than averaging into a confident-looking middle.

Every condition is a veto. That is the design: a gate whose criteria trade off against
each other can be satisfied by a very strong showing on one axis compensating for a
fatal weakness on another — nine models agreeing enthusiastically about a market
nobody has calibrated against. Conjunction makes the strongest possible claim require
the strongest possible evidence on every axis at once.

The gate reports *why* it refused, condition by condition. A gate that only returns a
boolean is untestable in practice: when it says no, nobody can tell whether it is
working or broken, and the temptation to loosen it becomes irresistible.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from mie.ensemble.calibration import CalibrationLibrary
from mie.ensemble.meta import EnsemblePrediction

__all__ = ["GateCheck", "GateDecision", "SuperPredictionGate"]


@dataclass(frozen=True, slots=True)
class GateCheck:
    """One condition, its verdict and the numbers behind it."""

    name: str
    passed: bool
    detail: str

    def __str__(self) -> str:  # pragma: no cover
        return f"{'PASS' if self.passed else 'FAIL'}  {self.name}: {self.detail}"


@dataclass(slots=True)
class GateDecision:
    """Whether a super prediction is warranted, and the full audit trail."""

    passed: bool = False
    checks: list[GateCheck] = field(default_factory=list)

    @property
    def failures(self) -> list[GateCheck]:
        return [c for c in self.checks if not c.passed]

    @property
    def reasons(self) -> list[str]:
        return [f"{c.name}: {c.detail}" for c in self.failures]

    def report(self) -> str:
        header = (
            "SUPER PREDICTION"
            if self.passed
            else f"no super prediction ({len(self.failures)} of {len(self.checks)} conditions unmet)"
        )
        return "\n".join([header, *(f"  {c}" for c in self.checks)])


class SuperPredictionGate:
    """Decides whether the ensemble's output qualifies as a super prediction."""

    def __init__(
        self,
        min_agreeing_families: int = 6,
        total_families: int = 8,
        min_consensus_share: float = 0.75,
        min_confidence: float = 0.65,
        min_data_quality: float = 0.85,
        require_regime_calibration: bool = True,
    ) -> None:
        self.min_agreeing_families = min_agreeing_families
        self.total_families = total_families
        self.min_consensus_share = min_consensus_share
        self.min_confidence = min_confidence
        self.min_data_quality = min_data_quality
        self.require_regime_calibration = require_regime_calibration

    def evaluate(
        self,
        result: EnsemblePrediction,
        calibration: CalibrationLibrary | None = None,
        member_ids: Sequence[str] | None = None,
    ) -> GateDecision:
        decision = GateDecision()
        agreement = result.agreement
        prediction = result.prediction
        regime = prediction.regime
        panel = list(member_ids or result.contributions or agreement.votes)

        decision.checks.append(
            GateCheck(
                "panel size",
                len(result.members) >= self.total_families,
                f"{len(result.members)} models ran, need {self.total_families}",
            )
        )

        decision.checks.append(
            GateCheck(
                "families agreeing",
                agreement.agreeing >= self.min_agreeing_families,
                f"{agreement.agreeing} of {agreement.participants} voting models agree "
                f"on {agreement.majority.value if agreement.majority else 'no direction'}, "
                f"need {self.min_agreeing_families}",
            )
        )

        # Headcount is not enough. Six models reading the same feature vector are one
        # opinion repeated six times, and the independence discount is what makes the
        # "independent families" in the requirement mean something.
        decision.checks.append(
            GateCheck(
                "independent agreement",
                agreement.effective_agreement >= self.min_agreeing_families,
                f"{agreement.effective_agreement:.2f} effective votes after discounting "
                f"shared inputs, need {self.min_agreeing_families}",
            )
        )

        decision.checks.append(
            GateCheck(
                "no material dissent",
                not agreement.is_split
                and agreement.consensus_share >= self.min_consensus_share,
                f"{agreement.consensus_share:.0%} of weighted votes behind the majority, "
                f"{len(agreement.dissenting)} dissenting, need "
                f"{self.min_consensus_share:.0%}",
            )
        )

        if self.require_regime_calibration:
            library = calibration
            calibrated = (
                [m for m in panel if library.has_regime_record(m, regime)] if library else []
            )
            decision.checks.append(
                GateCheck(
                    "calibrated in this regime",
                    len(calibrated) >= self.min_agreeing_families,
                    f"{len(calibrated)} of {len(panel)} contributing models have a usable "
                    f"calibration record in regime '{regime}', need "
                    f"{self.min_agreeing_families}",
                )
            )

        decision.checks.append(
            GateCheck(
                "demonstrated skill",
                bool(result.contributions),
                f"{len(result.contributions)} models carry a non-zero skill weight"
                if result.contributions
                else "no model has demonstrated skill against climatology",
            )
        )

        decision.checks.append(
            GateCheck(
                "confidence",
                result.factors.value >= self.min_confidence,
                f"{result.factors.value:.2f}, need {self.min_confidence:.2f} "
                f"(limited by {result.factors.limiting_factor})",
            )
        )

        decision.checks.append(
            GateCheck(
                "data quality",
                prediction.data_quality >= self.min_data_quality,
                f"{prediction.data_quality:.2f}, need {self.min_data_quality:.2f}",
            )
        )

        decision.checks.append(
            GateCheck(
                "ensemble published",
                result.published,
                "the ensemble published a directional view"
                if result.published
                else (result.suppressed_because or ["below the publication floor"])[0],
            )
        )

        decision.passed = all(c.passed for c in decision.checks)
        return decision
