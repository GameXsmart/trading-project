"""Isotonic calibration, and the evidence that it helped.

A model that says 70% should be right 70% of the time. Most are not: they are
systematically over- or under-confident, and the distortion is usually monotone —
which is what makes isotonic regression the right tool. It learns an arbitrary
non-decreasing map from stated probability to observed frequency without assuming a
functional form, so it can fix "says 80, means 55" without being told the shape.

Isotonic regression is also, unusually among the tools in this repository, *dangerous*.
It has enough freedom to fit noise exactly. Given 40 points it will reproduce them and
report perfect calibration, and a calibration layer that flatters itself is worse than
none — it converts a known-untrustworthy number into an apparently-trustworthy one.

Three defences, all structural:

* **The curve is fitted on one window and judged on a later one.** Whether calibration
  improved is measured out-of-sample or it is not measured.
* **A curve that does not improve held-out calibration is discarded** and the identity
  map kept. The default is to leave the model's numbers alone.
* **Records carry ``fitted_through``**, and applying a record to a prediction made at
  or before that instant raises. A calibration fitted on data that includes the
  prediction it is calibrating is look-ahead, in a place where it would be invisible.

Per model *and* per regime, because the direction of a model's miscalibration is not
a constant: a trend follower is overconfident in chop and roughly honest in a trend,
and a single pooled curve would average those into a map that is wrong in both.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

from mie.core.logging import get_logger
from mie.models.types import Distribution, Outcome
from mie.patterns.statistics import wilson_interval

log = get_logger(__name__)

__all__ = [
    "CalibrationCurve",
    "CalibrationLibrary",
    "CalibrationRecord",
    "ReliabilityBin",
    "ReliabilityDiagram",
    "classwise_ece",
    "reliability_diagram",
]

#: Below this many held-out observations, no curve is fitted at all. Isotonic
#: regression on a small sample is interpolation wearing a lab coat.
_MIN_FIT_SAMPLES = 120

#: The fitted curve must beat the identity map by at least this much held-out ECE to be
#: kept. A margin, not zero, because half the time noise alone produces a small
#: improvement and adopting those would be selecting on noise.
_MIN_ECE_IMPROVEMENT = 0.005

#: Fraction of the (chronologically ordered) sample used to fit; the rest judges.
_FIT_FRACTION = 0.6


@dataclass(frozen=True, slots=True)
class CalibrationCurve:
    """A monotone map from stated probability to calibrated probability.

    Stored as breakpoints and evaluated by linear interpolation between them, which
    keeps the map continuous — a step function would make two nearly identical inputs
    produce visibly different outputs for no reason the data supports.
    """

    #: Sorted, strictly increasing input probabilities.
    xs: tuple[float, ...] = ()
    #: Non-decreasing calibrated outputs, same length as ``xs``.
    ys: tuple[float, ...] = ()

    @property
    def is_identity(self) -> bool:
        return not self.xs

    def apply(self, probability: float) -> float:
        probability = max(0.0, min(1.0, probability))
        if self.is_identity:
            return probability
        xs, ys = self.xs, self.ys
        if probability <= xs[0]:
            return ys[0]
        if probability >= xs[-1]:
            return ys[-1]
        # Small breakpoint counts make a linear scan cheaper than bisection setup.
        for index in range(1, len(xs)):
            if probability <= xs[index]:
                left_x, right_x = xs[index - 1], xs[index]
                left_y, right_y = ys[index - 1], ys[index]
                if right_x <= left_x:
                    return right_y
                weight = (probability - left_x) / (right_x - left_x)
                return left_y + weight * (right_y - left_y)
        return ys[-1]

    @classmethod
    def identity(cls) -> CalibrationCurve:
        return cls()

    @classmethod
    def fit(cls, pairs: Sequence[tuple[float, float]]) -> CalibrationCurve:
        """Fit an isotonic map from (stated probability, outcome indicator) pairs.

        The indicator is 1.0 when the outcome occurred. Pool-adjacent-violators gives
        the least-squares non-decreasing fit; ties in the stated probability are
        merged first so that the same input cannot map to two outputs.
        """
        if len(pairs) < 10:
            return cls.identity()

        grouped: dict[float, list[float]] = defaultdict(list)
        for probability, indicator in pairs:
            grouped[round(max(0.0, min(1.0, probability)), 4)].append(indicator)

        xs = sorted(grouped)
        if len(xs) < 2:
            # A single distinct stated probability gives no shape to learn: the fit
            # would be a constant mapping every possible input to one output, which is
            # a far stronger claim than the data supports.
            return cls.identity()
        values = [sum(grouped[x]) / len(grouped[x]) for x in xs]
        weights = [float(len(grouped[x])) for x in xs]

        fitted = _pool_adjacent_violators(values, weights)
        return cls(xs=tuple(xs), ys=tuple(max(0.0, min(1.0, y)) for y in fitted))


def _pool_adjacent_violators(values: Sequence[float], weights: Sequence[float]) -> list[float]:
    """Pool-adjacent-violators: the weighted least-squares non-decreasing fit.

    Walks left to right maintaining blocks; whenever a block's mean would fall below
    its predecessor's the two are merged and the check repeats backwards. Linear time,
    and the result is the exact isotonic solution rather than an approximation.
    """
    block_values: list[float] = []
    block_weights: list[float] = []
    block_sizes: list[int] = []

    for value, weight in zip(values, weights, strict=True):
        block_values.append(value)
        block_weights.append(weight)
        block_sizes.append(1)
        while len(block_values) > 1 and block_values[-2] > block_values[-1]:
            weight_sum = block_weights[-2] + block_weights[-1]
            pooled = (
                block_values[-2] * block_weights[-2] + block_values[-1] * block_weights[-1]
            ) / weight_sum
            block_values[-2:] = [pooled]
            block_weights[-2:] = [weight_sum]
            block_sizes[-2:] = [block_sizes[-2] + block_sizes[-1]]

    result: list[float] = []
    for value, size in zip(block_values, block_sizes, strict=True):
        result.extend([value] * size)
    return result


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    """One bucket of a reliability diagram."""

    lower: float
    upper: float
    count: int
    mean_predicted: float
    observed: float
    #: Wilson interval on the observed frequency.
    observed_low: float
    observed_high: float

    @property
    def gap(self) -> float:
        return self.observed - self.mean_predicted

    @property
    def contains_nominal(self) -> bool:
        """Whether the stated probability is inside the observed frequency's interval.

        The honest version of "is this bin calibrated". A point criterion such as
        "within 5 points" mostly measures how many observations landed in the bin: at
        n=40 the sampling error alone is around ±15 points, so a bin can fail a 5-point
        rule while being perfectly calibrated, and pass it while being badly wrong.
        """
        return self.observed_low <= self.mean_predicted <= self.observed_high

    @property
    def within_tolerance(self) -> bool:
        """The literal Phase 7 gate: observed within 5 points of stated.

        Kept because it is the criterion the requirement names, and reported alongside
        :attr:`contains_nominal` rather than instead of it.
        """
        return abs(self.gap) <= 0.05


@dataclass(slots=True)
class ReliabilityDiagram:
    """Observed frequency against stated probability."""

    bins: list[ReliabilityBin] = field(default_factory=list)
    ece: float = 0.0
    samples: int = 0

    def populated(self, minimum: int = 20) -> list[ReliabilityBin]:
        return [b for b in self.bins if b.count >= minimum]

    def within_tolerance(self, minimum: int = 20) -> bool:
        """Whether every sufficiently populated bin meets the stated tolerance."""
        populated = self.populated(minimum)
        return bool(populated) and all(b.within_tolerance for b in populated)

    def statistically_calibrated(self, minimum: int = 20) -> bool:
        """Whether every populated bin's observed frequency is consistent with its
        stated probability, accounting for how few observations each bin holds."""
        populated = self.populated(minimum)
        return bool(populated) and all(b.contains_nominal for b in populated)

    def report(self) -> str:  # pragma: no cover - display affordance
        lines = [f"reliability ({self.samples} observations, ECE {self.ece:.4f})"]
        for entry in self.bins:
            if entry.count == 0:
                continue
            mark = "ok " if entry.contains_nominal else "OFF"
            lines.append(
                f"  [{entry.lower:.1f}-{entry.upper:.1f}) n={entry.count:<5} "
                f"stated={entry.mean_predicted:.3f} observed={entry.observed:.3f} "
                f"[{entry.observed_low:.3f}, {entry.observed_high:.3f}] {mark}"
            )
        return "\n".join(lines)


def reliability_diagram(
    pairs: Sequence[tuple[float, float]], bin_count: int = 10
) -> ReliabilityDiagram:
    """Bin (stated probability, outcome indicator) pairs and measure calibration."""
    diagram = ReliabilityDiagram(samples=len(pairs))
    if not pairs:
        return diagram

    buckets: list[list[tuple[float, float]]] = [[] for _ in range(bin_count)]
    for probability, indicator in pairs:
        probability = max(0.0, min(1.0, probability))
        index = min(bin_count - 1, int(probability * bin_count))
        buckets[index].append((probability, indicator))

    total = len(pairs)
    ece = 0.0
    for index, bucket in enumerate(buckets):
        lower, upper = index / bin_count, (index + 1) / bin_count
        if not bucket:
            diagram.bins.append(ReliabilityBin(lower, upper, 0, 0.0, 0.0, 0.0, 0.0))
            continue
        count = len(bucket)
        mean_predicted = sum(p for p, _ in bucket) / count
        successes = sum(1 for _, indicator in bucket if indicator >= 0.5)
        observed = successes / count
        low, high = wilson_interval(successes, count)
        diagram.bins.append(
            ReliabilityBin(
                lower=lower,
                upper=upper,
                count=count,
                mean_predicted=round(mean_predicted, 5),
                observed=round(observed, 5),
                observed_low=round(low, 5),
                observed_high=round(high, 5),
            )
        )
        ece += (count / total) * abs(observed - mean_predicted)

    diagram.ece = round(ece, 5)
    return diagram


def classwise_ece(
    entries: Sequence[tuple[Distribution, Outcome]], bin_count: int = 10
) -> float:
    """Expected calibration error averaged over the three outcome classes.

    Multiclass calibration is not one number, and taking only the top-class probability
    — the common shortcut — would leave a model free to be wildly wrong about the two
    classes it did not name. All three are measured.
    """
    if not entries:
        return 0.0
    total = 0.0
    for outcome in Outcome:
        pairs = [
            (distribution.probability(outcome), 1.0 if actual is outcome else 0.0)
            for distribution, actual in entries
        ]
        total += reliability_diagram(pairs, bin_count).ece
    return round(total / len(Outcome), 5)


@dataclass(slots=True)
class CalibrationRecord:
    """A fitted calibration for one model in one regime, plus the proof it helped."""

    model_id: str
    regime: str
    curves: dict[Outcome, CalibrationCurve] = field(default_factory=dict)
    samples: int = 0
    holdout_samples: int = 0
    ece_before: float = 0.0
    ece_after: float = 0.0
    #: Only data with ``as_of`` at or before this instant informed the curve.
    fitted_through: datetime | None = None
    #: Whether the fitted curve beat the identity map out-of-sample.
    improved: bool = False

    @property
    def is_usable(self) -> bool:
        """Whether this record should be applied to live predictions.

        Records that failed to improve held-out calibration are *kept* rather than
        deleted: their existence is what tells the confidence layer that this model has
        been measured in this regime and found not to be reliably calibrated, which is
        different from never having been measured at all.
        """
        return self.improved and bool(self.curves)

    @property
    def improvement(self) -> float:
        return round(self.ece_before - self.ece_after, 5)

    def apply(self, distribution: Distribution, as_of: datetime | None = None) -> Distribution:
        """Map a distribution through the curves and renormalise.

        Isotonic maps are fitted per class independently, so the three outputs do not
        sum to one. Renormalising is the standard remedy and preserves their ordering.
        """
        if as_of is not None and self.fitted_through is not None and as_of <= self.fitted_through:
            raise ValueError(
                f"calibration for {self.model_id}/{self.regime} was fitted through "
                f"{self.fitted_through.isoformat()}; applying it to a prediction at "
                f"{as_of.isoformat()} would use the future to calibrate the past"
            )
        if not self.is_usable:
            return distribution

        mapped = {
            outcome: self.curves.get(outcome, CalibrationCurve.identity()).apply(
                distribution.probability(outcome)
            )
            for outcome in Outcome
        }
        total = sum(mapped.values())
        if total <= 0:
            # Every class mapped to zero: the curve is degenerate on this input, and
            # the uncalibrated distribution is the safer answer.
            return distribution
        return Distribution(
            up=mapped[Outcome.UP], flat=mapped[Outcome.FLAT], down=mapped[Outcome.DOWN]
        )

    def summary(self) -> str:
        if not self.curves:
            return f"{self.model_id}/{self.regime}: not enough data ({self.samples})"
        verdict = "kept" if self.improved else "discarded (identity retained)"
        return (
            f"{self.model_id}/{self.regime}: n={self.samples} "
            f"ECE {self.ece_before:.4f} -> {self.ece_after:.4f} "
            f"({self.improvement:+.4f}) {verdict}"
        )


class CalibrationLibrary:
    """Calibration records for every (model, regime) pair that has enough evidence."""

    def __init__(
        self,
        min_samples: int = _MIN_FIT_SAMPLES,
        min_improvement: float = _MIN_ECE_IMPROVEMENT,
        fit_fraction: float = _FIT_FRACTION,
    ) -> None:
        self.min_samples = min_samples
        self.min_improvement = min_improvement
        self.fit_fraction = fit_fraction
        self._records: dict[tuple[str, str], CalibrationRecord] = {}

    # ------------------------------------------------------------------ fitting

    def fit(self, entries: Sequence[object]) -> list[CalibrationRecord]:
        """Fit records from scored predictions.

        Accepts anything exposing ``prediction`` and ``actual`` — in practice Phase 6's
        :class:`~mie.models.evaluation.ScoredPrediction`. Typed loosely on purpose, so
        the calibration layer does not depend on the evaluator's internals.
        """
        grouped: dict[tuple[str, str], list[tuple[datetime, Distribution, Outcome]]] = (
            defaultdict(list)
        )
        for entry in entries:
            prediction = getattr(entry, "prediction", None)
            actual = getattr(entry, "actual", None)
            if prediction is None or actual is None:
                continue
            # Abstentions carry no probabilistic claim; calibrating them would be
            # fitting a curve to a constant.
            if prediction.confidence <= 0.0:
                continue
            key = (prediction.model_id, prediction.regime)
            grouped[key].append((prediction.as_of, prediction.distribution, actual))
            grouped[(prediction.model_id, "all")].append(
                (prediction.as_of, prediction.distribution, actual)
            )

        produced: list[CalibrationRecord] = []
        for (model_id, regime), rows in grouped.items():
            record = self._fit_one(model_id, regime, rows)
            self._records[(model_id, regime)] = record
            produced.append(record)
        return produced

    def _fit_one(
        self,
        model_id: str,
        regime: str,
        rows: Sequence[tuple[datetime, Distribution, Outcome]],
    ) -> CalibrationRecord:
        ordered = sorted(rows, key=lambda row: row[0])
        record = CalibrationRecord(model_id=model_id, regime=regime, samples=len(ordered))
        if len(ordered) < self.min_samples:
            return record

        cut = max(1, int(len(ordered) * self.fit_fraction))
        train, holdout = ordered[:cut], ordered[cut:]
        if len(holdout) < 20:
            return record

        curves = {
            outcome: CalibrationCurve.fit(
                [
                    (distribution.probability(outcome), 1.0 if actual is outcome else 0.0)
                    for _, distribution, actual in train
                ]
            )
            for outcome in Outcome
        }

        raw = [(distribution, actual) for _, distribution, actual in holdout]
        calibrated = [
            (_map_distribution(distribution, curves), actual) for distribution, actual in raw
        ]
        record.curves = curves
        record.holdout_samples = len(holdout)
        record.ece_before = classwise_ece(raw)
        record.ece_after = classwise_ece(calibrated)
        record.improved = record.improvement >= self.min_improvement
        # The curve saw everything up to the last training point, so it may only be
        # applied strictly after it.
        record.fitted_through = train[-1][0]

        # Debug rather than info: one line per (model, regime) pair is dozens of
        # lines restating a table the caller is about to print.
        log.debug(
            "calibration_fitted",
            model=model_id,
            regime=regime,
            samples=len(ordered),
            ece_before=record.ece_before,
            ece_after=record.ece_after,
            kept=record.improved,
        )
        return record

    # ------------------------------------------------------------------ lookup

    def add(self, record: CalibrationRecord) -> None:
        """Register a record directly.

        For records loaded from storage rather than fitted in this process. Fitting
        remains the only way to *create* one, so a record can never claim an
        improvement it did not demonstrate.
        """
        self._records[(record.model_id, record.regime)] = record

    def record_for(self, model_id: str, regime: str) -> CalibrationRecord | None:
        """The record for this regime, falling back to the pooled one.

        The fallback is reported by the record's own ``regime`` field, so a caller can
        tell "calibrated in this regime" from "calibrated on average" — a distinction
        the super-prediction gate depends on.
        """
        return self._records.get((model_id, regime)) or self._records.get((model_id, "all"))

    def has_regime_record(self, model_id: str, regime: str) -> bool:
        """Whether this model has a *usable* record in this specific regime."""
        record = self._records.get((model_id, regime))
        return record is not None and record.is_usable

    def calibrate(
        self, model_id: str, regime: str, distribution: Distribution, as_of: datetime | None = None
    ) -> tuple[Distribution, CalibrationRecord | None]:
        """Apply calibration if a usable record exists; otherwise pass through."""
        record = self.record_for(model_id, regime)
        if record is None or not record.is_usable:
            return distribution, record
        return record.apply(distribution, as_of=as_of), record

    @property
    def records(self) -> list[CalibrationRecord]:
        return list(self._records.values())

    def usable(self) -> list[CalibrationRecord]:
        return [r for r in self._records.values() if r.is_usable]

    def report(self) -> str:  # pragma: no cover - display affordance
        if not self._records:
            return "no calibration records"
        lines = ["Calibration", "=" * 78]
        lines.extend("  " + r.summary() for r in sorted(self._records.values(), key=str))
        kept = len(self.usable())
        lines.append("")
        lines.append(f"usable records: {kept} of {len(self._records)}")
        return "\n".join(lines)


def _map_distribution(
    distribution: Distribution, curves: dict[Outcome, CalibrationCurve]
) -> Distribution:
    mapped = {
        outcome: curves.get(outcome, CalibrationCurve.identity()).apply(
            distribution.probability(outcome)
        )
        for outcome in Outcome
    }
    if sum(mapped.values()) <= 0:
        return distribution
    return Distribution(
        up=mapped[Outcome.UP], flat=mapped[Outcome.FLAT], down=mapped[Outcome.DOWN]
    )
