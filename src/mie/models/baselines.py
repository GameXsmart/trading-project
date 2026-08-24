"""Baselines: the bar every model has to clear.

Phase 6's gate is that a model beats a baseline out-of-sample or does not ship. That
makes the choice of baseline the most consequential decision in the phase — a weak one
certifies nothing, and picking it *after* seeing model results would be the purest form
of self-deception available here.

Two baselines, for different reasons:

**Climatology** predicts the unconditional frequency of each outcome, measured from
data before the prediction point. It is the honest bar: it encodes "what usually
happens" with no skill at all, and any model that cannot beat it has demonstrated no
knowledge of the present whatsoever. On low-signal financial data climatology is
*hard* to beat, which is precisely why it is the primary standard here.

**Persistence** predicts that the last move repeats. It is the folk baseline, weaker
than climatology on mean-reverting data and stronger on trending data. It is included
because it is the standard others quote, and because a model that beats climatology
but loses to persistence has found trend-following and should say so.

Both are proper forecasters emitting the same envelope, so they are scored by exactly
the same code as the models. A baseline evaluated by a different path is not a
comparison.
"""

from __future__ import annotations

from collections import Counter

from mie.models.base import PredictionContext, Predictor
from mie.models.types import Distribution, Outcome, Prediction, PredictionEvidence

__all__ = ["ClimatologyBaseline", "PersistenceBaseline", "UniformBaseline"]


class ClimatologyBaseline(Predictor):
    """Predicts the historical frequency of each outcome.

    The primary standard. Uses only bars available at ``as_of``, so it is a legitimate
    out-of-sample forecaster rather than an in-sample summary — computing these
    frequencies over the whole history, including the future, would make the baseline
    unbeatable for reasons having nothing to do with skill.
    """

    model_id = "baseline_climatology"
    warmup_bars = 200

    def inputs_used(self) -> frozenset[str]:
        return frozenset({"price"})

    def predict(self, context: PredictionContext) -> Prediction:
        if not context.has_enough_history(self.warmup_bars):
            return self.abstain(context, "insufficient history for a frequency estimate")

        counts = _outcome_frequencies(context)
        total = sum(counts.values())
        if total < 30:
            return self.abstain(context, "too few completed horizons to estimate frequencies")

        # Laplace smoothing: an outcome absent from the sample is rare, not impossible,
        # and a zero probability would produce an infinite log-loss the first time it
        # occurred.
        distribution = Distribution(
            up=(counts[Outcome.UP] + 1) / (total + 3),
            flat=(counts[Outcome.FLAT] + 1) / (total + 3),
            down=(counts[Outcome.DOWN] + 1) / (total + 3),
        )
        return self.build(
            context,
            distribution=distribution,
            # Deliberately modest. Climatology is reliable but says nothing about
            # *now*, and presenting it as confident would be a category error.
            confidence=0.30,
            evidence=[
                PredictionEvidence(
                    label="historical base rates",
                    detail=f"{total} completed horizons in the trailing window",
                )
            ],
            invalidation=["a regime change would alter the base rates themselves"],
        )


class PersistenceBaseline(Predictor):
    """Predicts that the most recent move continues.

    The folk baseline. Its edge is scaled by how large the last move was relative to
    the classification threshold, so a marginal wobble does not produce a confident
    directional call.
    """

    model_id = "baseline_persistence"
    warmup_bars = 30

    def inputs_used(self) -> frozenset[str]:
        return frozenset({"price"})

    def predict(self, context: PredictionContext) -> Prediction:
        if not context.has_enough_history(self.warmup_bars):
            return self.abstain(context, "insufficient history")

        threshold = context.threshold_pct
        recent = context.returns(context.horizon.bars + 1)[-context.horizon.bars :]
        if not recent:
            return self.abstain(context, "no recent returns")

        move = sum(recent)
        if threshold <= 0:
            return self.abstain(context, "degenerate volatility")

        edge = max(-1.0, min(1.0, move / (threshold * 2.0)))
        return self.build(
            context,
            distribution=Distribution.from_edge(edge * 0.6, flat_mass=0.34),
            confidence=0.30,
            evidence=[
                PredictionEvidence(
                    label="last move continues",
                    detail=f"{move:+.2f}% over the previous {len(recent)} bars",
                    contribution=edge,
                )
            ],
            invalidation=[f"a reversal beyond {threshold:.2f}% would falsify continuation"],
        )


class UniformBaseline(Predictor):
    """Predicts one third for each outcome. The floor.

    Useful as a sanity check rather than as a standard: any forecaster that cannot beat
    a coin with three faces is broken, not merely unskilled. It also gives the Brier
    scale a fixed reference point across assets and regimes.
    """

    model_id = "baseline_uniform"
    warmup_bars = 1

    def inputs_used(self) -> frozenset[str]:
        return frozenset()

    def predict(self, context: PredictionContext) -> Prediction:
        return self.build(
            context,
            distribution=Distribution.uniform(),
            confidence=0.0,
            evidence=[PredictionEvidence(label="no information used")],
        )


def _outcome_frequencies(context: PredictionContext) -> Counter[Outcome]:
    """Outcome frequencies over non-overlapping horizons in the available history.

    Non-overlapping because adjacent horizons share most of their window and are not
    independent samples; counting them all would understate the uncertainty in the
    frequency estimate itself.
    """
    bars = context.horizon.bars
    threshold = context.threshold_pct
    closes = [c.close for c in context.candles if c.close > 0]
    counts: Counter[Outcome] = Counter()
    for index in range(0, len(closes) - bars, max(1, bars)):
        entry = closes[index]
        if entry <= 0:
            continue
        change = (closes[index + bars] - entry) / entry * 100.0
        counts[Outcome.classify(change, threshold)] += 1
    return counts
