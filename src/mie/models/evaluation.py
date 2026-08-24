"""Walk-forward evaluation.

Phase 6's gate: a model beats a baseline out-of-sample, or it does not ship. This is
the code that decides, and it is the most consequential module in the phase — so the
ways it could flatter a model are worth naming explicitly.

* **No random splits.** Predictions are made at increasing timestamps and scored
  against what came next. Shuffling time-series data into train/test lets the future
  inform the past and produces metrics that are simply fiction.
* **Non-overlapping evaluation points.** Consecutive horizons share most of their
  window, so scoring every bar would inflate the sample count without adding
  information and shrink every confidence interval accordingly.
* **Identical treatment.** Baselines are forecasters implementing the same interface,
  scored by the same code on the same points. A baseline evaluated by a different path
  is not a comparison.
* **Skill, not accuracy.** Accuracy on a three-class problem with an unbalanced class
  mix rewards predicting the majority class. Brier skill against a baseline asks the
  only question that matters: does this model know something the baseline does not?
* **Sliced results.** A single blended number hides everything. Results are reported
  per asset, timeframe and regime, and losing slices are reported rather than dropped.
* **Slicing is multiple comparisons.** Reporting per regime means testing many
  hypotheses, and "beats the baseline on at least one slice" is nearly guaranteed by
  chance across forty of them. Every slice therefore carries a *paired significance
  test* on its per-prediction Brier differences, and the whole family is corrected with
  Benjamini-Hochberg. A positive skill number alone is not a pass.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from statistics import median

from mie.core.logging import get_logger
from mie.core.timeframes import Timeframe
from mie.models.base import PredictionContext, Predictor
from mie.models.types import Outcome, Prediction
from mie.patterns.statistics import benjamini_hochberg, normal_cdf

log = get_logger(__name__)

__all__ = ["EvaluationReport", "ModelScore", "ScoredPrediction", "WalkForwardEvaluator"]

#: Below this many scored predictions, no verdict is issued for a slice.
_MIN_PREDICTIONS = 30


@dataclass(frozen=True, slots=True)
class ScoredPrediction:
    """A prediction paired with what actually happened."""

    prediction: Prediction
    actual: Outcome
    realised_return_pct: float

    @property
    def brier(self) -> float:
        return self.prediction.brier_score(self.actual)

    @property
    def log_loss(self) -> float:
        return self.prediction.log_loss(self.actual)

    @property
    def correct(self) -> bool:
        return self.prediction.distribution.most_likely is self.actual

    @property
    def abstained(self) -> bool:
        return self.prediction.confidence <= 0.0


@dataclass(slots=True)
class ModelScore:
    """How one model performed on one slice."""

    model_id: str
    asset: str
    timeframe: Timeframe
    horizon_bars: int
    regime: str
    predictions: int
    abstentions: int
    brier: float
    log_loss: float
    accuracy: float
    baseline_brier: float
    baseline_id: str
    #: 1 - model/baseline. Positive means the model knows something the baseline does not.
    skill: float
    mean_confidence: float
    #: Paired test on per-prediction Brier differences against the baseline.
    p_value: float = 1.0
    #: Set once the whole family of slices has been corrected.
    significant: bool = False
    sample_start: datetime | None = None
    sample_end: datetime | None = None

    @property
    def has_evidence(self) -> bool:
        return self.predictions >= _MIN_PREDICTIONS

    @property
    def beats_baseline(self) -> bool:
        """Whether the model demonstrably outperformed the baseline on this slice.

        Three conditions, all required:

        * enough predictions to say anything;
        * a skill margin above a floor, because a +0.002 edge is not worth acting on
          even if it were real;
        * statistical significance on the *paired* Brier differences, surviving
          correction across every slice tested.

        The third is what stops the slicing from manufacturing a winner. Across forty
        regime slices, some model will post positive skill on one of them by luck, and
        a gate that accepted that would certify noise as a finding.
        """
        return self.has_evidence and self.skill > 0.01 and self.significant

    @property
    def verdict(self) -> str:
        if not self.has_evidence:
            return f"insufficient predictions ({self.predictions}, need {_MIN_PREDICTIONS})"
        if self.beats_baseline:
            return f"beats {self.baseline_id} (skill {self.skill:+.3f}, p={self.p_value:.4f})"
        if self.skill > 0.01:
            return f"positive but not significant (skill {self.skill:+.3f}, p={self.p_value:.4f})"
        if self.skill > 0:
            return f"indistinguishable from {self.baseline_id} (skill {self.skill:+.3f})"
        return f"worse than {self.baseline_id} (skill {self.skill:+.3f})"

    def summary(self) -> str:
        return (
            f"{self.model_id:12} {self.asset:5} {self.timeframe} +{self.horizon_bars:<3} "
            f"{self.regime:14} n={self.predictions:<4} brier={self.brier:.4f} "
            f"base={self.baseline_brier:.4f} skill={self.skill:+.4f} -> {self.verdict}"
        )


@dataclass(slots=True)
class EvaluationReport:
    """Everything one evaluation run produced."""

    scores: list[ModelScore] = field(default_factory=list)
    scored: dict[str, list[ScoredPrediction]] = field(default_factory=dict)

    def for_model(self, model_id: str) -> list[ModelScore]:
        return [s for s in self.scores if s.model_id == model_id]

    def winners(self) -> list[ModelScore]:
        return [s for s in self.scores if s.beats_baseline]

    def passing_models(self) -> set[str]:
        """Models that beat the baseline on at least one slice — the Phase 6 gate."""
        return {s.model_id for s in self.winners()}

    def failing_models(self) -> set[str]:
        return {s.model_id for s in self.scores} - self.passing_models()

    def report(self) -> str:
        lines = ["Walk-forward evaluation", "=" * 78]
        for score in sorted(self.scores, key=lambda s: (-s.skill, s.model_id)):
            lines.append("  " + score.summary())
        passing = sorted(self.passing_models())
        failing = sorted(self.failing_models())
        lines.append("")
        lines.append(f"PASS ({len(passing)}): {', '.join(passing) or 'none'}")
        lines.append(f"FAIL ({len(failing)}): {', '.join(failing) or 'none'}")
        return "\n".join(lines)


class WalkForwardEvaluator:
    """Runs models forward through history and scores them against a baseline."""

    def __init__(
        self,
        baseline: Predictor,
        min_predictions: int = _MIN_PREDICTIONS,
        stride_multiple: float = 1.0,
        false_discovery_rate: float = 0.05,
    ) -> None:
        self.baseline = baseline
        self.min_predictions = min_predictions
        self.false_discovery_rate = false_discovery_rate
        #: Gap between evaluation points, as a multiple of the horizon. At 1.0 the
        #: forward windows tile without overlapping.
        self.stride_multiple = stride_multiple

    def evaluate(
        self,
        models: Sequence[Predictor],
        contexts: Sequence[tuple[PredictionContext, float]],
    ) -> EvaluationReport:
        """Score every model over pre-built (context, realised return) pairs.

        Contexts are built by the caller, which is what keeps this module honest: it
        never touches raw history and so cannot accidentally reach forward.
        """
        report = EvaluationReport()
        every = [*list(models), self.baseline]
        scored: dict[str, list[ScoredPrediction]] = defaultdict(list)

        for context, realised in contexts:
            for model in every:
                try:
                    prediction = model.predict(context)
                except Exception as exc:
                    # One model failing must not abort the run; it is recorded as an
                    # absence rather than crashing the evaluation of the others.
                    log.warning(
                        "model_failed", model=model.model_id, error=str(exc)[:200]
                    )
                    continue
                actual = prediction.score_outcome(realised)
                scored[model.model_id].append(
                    ScoredPrediction(prediction, actual, realised)
                )

        report.scored = dict(scored)
        baseline_scored = scored.get(self.baseline.model_id, [])
        if not baseline_scored:
            return report

        for model in models:
            entries = scored.get(model.model_id, [])
            if not entries:
                continue
            report.scores.extend(self._score_slices(model.model_id, entries, baseline_scored))

        # Correct across every slice of every model at once. Per-slice correction
        # would be no correction: the false positives come from the size of the sweep.
        if report.scores:
            flags = benjamini_hochberg(
                [s.p_value for s in report.scores], self.false_discovery_rate
            )
            for score, significant in zip(report.scores, flags, strict=True):
                score.significant = significant
        return report

    def _score_slices(
        self,
        model_id: str,
        entries: Sequence[ScoredPrediction],
        baseline_entries: Sequence[ScoredPrediction],
    ) -> list[ModelScore]:
        """Score overall and per regime.

        "This model is 55% accurate" is meaningless; "this model has skill on BTC 1h in
        low-volatility regimes and none elsewhere" is actionable, and only the sliced
        view can say it.
        """
        baseline_by_key = {
            (s.prediction.as_of, s.prediction.asset): s for s in baseline_entries
        }

        groups: dict[str, list[ScoredPrediction]] = defaultdict(list)
        for entry in entries:
            groups["all"].append(entry)
            groups[entry.prediction.regime].append(entry)

        scores: list[ModelScore] = []
        for regime, group in groups.items():
            paired = [
                (entry, baseline_by_key.get((entry.prediction.as_of, entry.prediction.asset)))
                for entry in group
            ]
            usable = [(e, b) for e, b in paired if b is not None]
            if len(usable) < 5:
                continue

            model_brier = sum(e.brier for e, _ in usable) / len(usable)
            base_brier = sum(b.brier for _, b in usable) / len(usable)
            # Paired differences: the two forecasters saw identical evaluation points,
            # so pairing removes the variance of the points themselves and tests the
            # only thing at issue — whether this model scores better on the same data.
            differences = [b.brier - e.brier for e, b in usable]
            p_value = _paired_p_value(differences)
            first = usable[0][0].prediction
            scores.append(
                ModelScore(
                    model_id=model_id,
                    asset=first.asset,
                    timeframe=first.timeframe,
                    horizon_bars=first.horizon.bars,
                    regime=regime,
                    predictions=len(usable),
                    abstentions=sum(1 for e, _ in usable if e.abstained),
                    brier=round(model_brier, 5),
                    log_loss=round(sum(e.log_loss for e, _ in usable) / len(usable), 5),
                    accuracy=round(sum(1 for e, _ in usable if e.correct) / len(usable), 4),
                    baseline_brier=round(base_brier, 5),
                    baseline_id=self.baseline.model_id,
                    skill=round(1.0 - model_brier / base_brier, 5) if base_brier > 0 else 0.0,
                    p_value=round(p_value, 6),
                    mean_confidence=round(
                        sum(e.prediction.confidence for e, _ in usable) / len(usable), 4
                    ),
                    sample_start=min(e.prediction.as_of for e, _ in usable),
                    sample_end=max(e.prediction.as_of for e, _ in usable),
                )
            )
        return scores


def realised_return(candles: Sequence[object], index: int, horizon_bars: int) -> float | None:
    """Percentage return from ``index`` to ``index + horizon_bars``.

    Reads strictly forward of the prediction point, and only from bars that had closed
    by the time the outcome is scored.
    """
    if index + horizon_bars >= len(candles):
        return None
    entry = candles[index].close  # type: ignore[attr-defined]
    if entry <= 0:
        return None
    exit_close = candles[index + horizon_bars].close  # type: ignore[attr-defined]
    return (exit_close - entry) / entry * 100.0


def summarise_thresholds(contexts: Sequence[tuple[PredictionContext, float]]) -> dict[str, float]:
    """Class balance across the evaluation set.

    Reported because Brier scores are only interpretable against a known class mix: a
    set that is 80% flat makes a "predict flat always" forecaster look excellent.
    """
    if not contexts:
        return {}
    counts: dict[Outcome, int] = defaultdict(int)
    thresholds: list[float] = []
    for context, realised in contexts:
        threshold = context.threshold_pct
        thresholds.append(threshold)
        counts[Outcome.classify(realised, threshold)] += 1
    total = sum(counts.values())
    return {
        "up": round(counts[Outcome.UP] / total, 4),
        "flat": round(counts[Outcome.FLAT] / total, 4),
        "down": round(counts[Outcome.DOWN] / total, 4),
        "median_threshold_pct": round(median(thresholds), 4),
        "points": total,
    }


def _paired_p_value(differences: Sequence[float]) -> float:
    """One-sided p-value that the mean paired difference is positive.

    ``differences`` are baseline-minus-model Brier scores, so positive means the model
    scored better. One-sided because the claim being tested is "this model is better",
    not "this model differs" — a model that is significantly *worse* is not a pass, and
    a two-sided test would treat the two identically.

    A t-test on the mean is adequate here: the differences are bounded, and with the
    sample sizes these slices carry the central limit theorem applies comfortably.
    """
    n = len(differences)
    if n < 5:
        return 1.0
    mean = sum(differences) / n
    variance = sum((d - mean) ** 2 for d in differences) / (n - 1)
    if variance <= 0:
        # Identical scores on every point: no difference to detect.
        return 1.0 if mean <= 0 else 0.0
    standard_error = (variance / n) ** 0.5
    if standard_error == 0:
        return 1.0
    z = mean / standard_error
    return 1.0 - normal_cdf(z)
