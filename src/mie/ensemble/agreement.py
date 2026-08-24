"""Measuring agreement between models that are not actually independent.

Requirement §12 asks for eight independent model families and treats their agreement as
corroboration. That inference is only valid if the independence is real. Two models
reading the same feature vector will agree constantly, and counting that agreement as
two votes is double-counting the same evidence — the single most common way an
ensemble manufactures false confidence.

So agreement here is *discounted by input overlap*. Each model declares what it draws
on via :meth:`~mie.models.base.Predictor.inputs_used`, and a model's vote is weighted
by how little it shares with the rest of the panel:

    weight_i = 1 / (1 + Σ_{j≠i} jaccard(inputs_i, inputs_j))

Eight genuinely disjoint models each carry weight 1.0 and the effective count equals
the headcount. Two models reading identical inputs carry 0.5 each and contribute one
vote between them, which is what they are worth. The measure degrades smoothly rather
than requiring a judgement call about where "independent" stops.

This is a lower bound on dependence, not a proof of independence: models with disjoint
declared inputs can still be driven by the same underlying market factor. It catches
the shared-substrate case, which is the one that is both common and invisible.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from mie.models.types import Outcome, Prediction

__all__ = [
    "AgreementReport",
    "independence_weights",
    "measure_agreement",
    "overlap_matrix",
]


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        # Two models declaring no inputs are unfalsifiably identical; treat them as
        # fully overlapping rather than fully independent.
        return 1.0
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def overlap_matrix(inputs: Mapping[str, frozenset[str]]) -> dict[tuple[str, str], float]:
    """Pairwise Jaccard overlap of declared inputs, for auditing the panel."""
    names = sorted(inputs)
    return {
        (a, b): round(_jaccard(inputs[a], inputs[b]), 4)
        for index, a in enumerate(names)
        for b in names[index + 1 :]
    }


def independence_weights(inputs: Mapping[str, frozenset[str]]) -> dict[str, float]:
    """How much each model's vote is worth, given what it shares with the others."""
    names = sorted(inputs)
    weights: dict[str, float] = {}
    for name in names:
        shared = sum(_jaccard(inputs[name], inputs[other]) for other in names if other != name)
        weights[name] = round(1.0 / (1.0 + shared), 4)
    return weights


@dataclass(slots=True)
class AgreementReport:
    """How much the panel agrees, and how much that agreement is worth."""

    #: Direction each model leaned, excluding abstentions.
    votes: dict[str, Outcome] = field(default_factory=dict)
    #: Models that declined to forecast.
    abstained: list[str] = field(default_factory=list)
    #: Independence weight per voting model.
    weights: dict[str, float] = field(default_factory=dict)
    #: The direction carrying the most weighted support.
    majority: Outcome | None = None
    #: Raw headcount agreeing with the majority.
    agreeing: int = 0
    #: Headcount-equivalent agreement after discounting shared inputs.
    effective_agreement: float = 0.0
    #: Total independence weight across all voting models.
    total_weight: float = 0.0
    #: Models leaning the other way.
    dissenting: list[str] = field(default_factory=list)
    #: Mean absolute directional edge among voting models.
    mean_edge: float = 0.0

    @property
    def participants(self) -> int:
        return len(self.votes)

    @property
    def consensus_share(self) -> float:
        """Weighted share of the panel behind the majority, in [0, 1]."""
        if self.total_weight <= 0:
            return 0.0
        return round(self.effective_agreement / self.total_weight, 4)

    @property
    def is_split(self) -> bool:
        """Whether the panel is meaningfully divided.

        A split panel does not average into a confident middle. It is a signal that
        the models are reading different things and one of them is wrong, which is
        information — and the correct response is to publish less, not to blend.
        """
        return bool(self.dissenting) and self.consensus_share < 0.7

    def summary(self) -> str:
        if self.majority is None:
            return f"no direction: {len(self.abstained)} abstained of {len(self.abstained) + self.participants}"
        return (
            f"{self.majority.value}: {self.agreeing}/{self.participants} models "
            f"({self.effective_agreement:.2f} effective, "
            f"{self.consensus_share:.0%} of weight), "
            f"{len(self.dissenting)} dissenting, {len(self.abstained)} abstained"
        )


def measure_agreement(
    predictions: Sequence[Prediction],
    inputs: Mapping[str, frozenset[str]],
    min_edge: float = 0.04,
) -> AgreementReport:
    """Measure weighted directional agreement across a panel of predictions.

    ``min_edge`` keeps a model that is essentially neutral from being recorded as
    voting: a +0.005 lean is not an opinion, and treating it as one would let a panel
    of shrugs look like a consensus.
    """
    report = AgreementReport()
    all_weights = independence_weights(inputs) if inputs else {}

    for prediction in predictions:
        model_id = prediction.model_id
        edge = prediction.distribution.directional_edge
        if prediction.confidence <= 0.0 or abs(edge) < min_edge:
            report.abstained.append(model_id)
            continue
        direction = Outcome.UP if edge > 0 else Outcome.DOWN
        report.votes[model_id] = direction
        report.weights[model_id] = all_weights.get(model_id, 1.0)

    if not report.votes:
        return report

    report.total_weight = round(sum(report.weights.values()), 4)
    report.mean_edge = round(
        sum(
            abs(p.distribution.directional_edge)
            for p in predictions
            if p.model_id in report.votes
        )
        / len(report.votes),
        4,
    )

    tally: dict[Outcome, float] = {}
    for model_id, direction in report.votes.items():
        tally[direction] = tally.get(direction, 0.0) + report.weights[model_id]

    # Ties break toward no majority rather than toward whichever outcome sorts first:
    # a panel split exactly down the middle has not chosen a direction.
    best = max(tally.values())
    leaders = [direction for direction, weight in tally.items() if weight >= best - 1e-9]
    if len(leaders) != 1:
        report.dissenting = sorted(report.votes)
        return report

    report.majority = leaders[0]
    report.agreeing = sum(1 for d in report.votes.values() if d is report.majority)
    report.effective_agreement = round(best, 4)
    report.dissenting = sorted(
        model_id for model_id, d in report.votes.items() if d is not report.majority
    )
    return report
