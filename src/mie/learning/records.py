"""The prediction record and its hash.

Requirement §14 says storing predictions is not learning. True — but storing them
*correctly* is the precondition for finding out whether anything learned, and there are
two ways to get it wrong that would make every later number meaningless.

**Editing after the fact.** A prediction that can be revised once the outcome is known
is not a forecast, it is a description, and accuracy computed from it is circular. So
the record is written once and never updated. Nothing in the write path merges.

**Silent corruption.** A stored distribution that drifts — a migration, a bad backfill,
a partial write — would be indistinguishable from the model having been better than it
was. So each record carries a hash over the fields that constitute the claim, and the
hash is verified on read. A record that fails verification is refused, not repaired:
whatever it now says is not what the model said.

The hash deliberately covers only the *claim* — model, asset, horizon, distribution,
confidence, threshold, prediction instant. Not `created_at`, not the evidence blob.
Including mutable annotations would make the hash fail for reasons that have nothing to
do with the forecast's integrity, and a check that cries wolf is a check that gets
switched off.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from mie.core.timeframes import Timeframe, utcnow
from mie.models.types import Distribution, Horizon, Outcome, Prediction

__all__ = [
    "PredictionRecord",
    "ResolvedOutcome",
    "content_hash",
    "prediction_id",
    "volatility_bucket",
]

#: Buckets for slicing by how violent the market was at prediction time. Boundaries are
#: per-bar median absolute return in percent, chosen from the measured distribution on
#: hourly crypto rather than picked round: the median hourly move across BTC/ETH/SOL
#: sits near 0.25%, so these split the observed range rather than labelling almost
#: everything "normal".
_VOLATILITY_BOUNDS: tuple[tuple[float, str], ...] = (
    (0.15, "very_low"),
    (0.30, "low"),
    (0.55, "normal"),
    (1.00, "high"),
)


def volatility_bucket(median_abs_return_pct: float) -> str:
    """Label a volatility level for slicing.

    Recorded at prediction time, never derived later. Reconstructing "how volatile was
    it then" after the fact requires re-deriving history, and any bug in that
    derivation would silently re-slice every past result.
    """
    for bound, label in _VOLATILITY_BOUNDS:
        if median_abs_return_pct < bound:
            return label
    return "very_high"


def content_hash(
    *,
    model_id: str,
    model_version: str,
    asset: str,
    timeframe: str,
    horizon_bars: int,
    as_of: datetime,
    up: float,
    flat: float,
    down: float,
    confidence: float,
    move_threshold_pct: float,
) -> str:
    """SHA-256 over the fields that constitute the forecast.

    Probabilities are rounded to nine places before hashing. Float formatting is not
    guaranteed identical across platforms or driver round-trips, and a hash that fails
    because SQLite returned a value one ULP away would flag integrity failures that are
    not integrity failures.
    """
    payload = "|".join(
        [
            model_id,
            model_version,
            asset.upper(),
            timeframe,
            str(horizon_bars),
            as_of.isoformat(),
            f"{up:.9f}",
            f"{flat:.9f}",
            f"{down:.9f}",
            f"{confidence:.9f}",
            f"{move_threshold_pct:.9f}",
        ]
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def prediction_id(model_id: str, asset: str, timeframe: str, horizon_bars: int, as_of: datetime) -> str:
    """A stable identity for one model's view of one asset at one instant.

    Derived rather than random, so that re-running the same prediction point produces
    the same id and collides with the existing row instead of inserting a second copy.
    Deduplication by construction — a re-run cannot inflate the sample.
    """
    digest = hashlib.sha256(
        f"{model_id}|{asset.upper()}|{timeframe}|{horizon_bars}|{as_of.isoformat()}".encode()
    ).hexdigest()
    return digest[:40]


@dataclass(slots=True)
class PredictionRecord:
    """A prediction as stored: the claim, its identity, and its integrity check."""

    prediction_id: str
    content_hash: str
    model_id: str
    model_version: str
    asset: str
    timeframe: Timeframe
    horizon_bars: int
    as_of: datetime
    resolves_at: datetime
    distribution: Distribution
    confidence: float
    move_threshold_pct: float
    reference_price: float
    regime: str
    volatility_bucket: str
    data_quality: float
    is_actionable: bool
    evidence: dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    created_at: datetime = field(default_factory=utcnow)

    @classmethod
    def of(
        cls,
        prediction: Prediction,
        volatility: str = "unknown",
        evidence: dict[str, Any] | None = None,
    ) -> PredictionRecord:
        """Build a storable record from a live prediction."""
        timeframe = str(prediction.timeframe)
        identity = prediction_id(
            prediction.model_id,
            prediction.asset,
            timeframe,
            prediction.horizon.bars,
            prediction.as_of,
        )
        digest = content_hash(
            model_id=prediction.model_id,
            model_version=prediction.model_version,
            asset=prediction.asset,
            timeframe=timeframe,
            horizon_bars=prediction.horizon.bars,
            as_of=prediction.as_of,
            up=prediction.distribution.up,
            flat=prediction.distribution.flat,
            down=prediction.distribution.down,
            confidence=prediction.confidence,
            move_threshold_pct=prediction.move_threshold_pct,
        )
        return cls(
            prediction_id=identity,
            content_hash=digest,
            model_id=prediction.model_id,
            model_version=prediction.model_version,
            asset=prediction.asset,
            timeframe=prediction.timeframe,
            horizon_bars=prediction.horizon.bars,
            as_of=prediction.as_of,
            resolves_at=prediction.resolves_at,
            distribution=prediction.distribution,
            confidence=prediction.confidence,
            move_threshold_pct=prediction.move_threshold_pct,
            reference_price=prediction.reference_price,
            regime=prediction.regime,
            volatility_bucket=volatility,
            data_quality=prediction.data_quality,
            is_actionable=prediction.is_actionable,
            evidence=evidence
            or {
                "for": [e.label for e in prediction.evidence],
                "against": [e.label for e in prediction.counter_evidence],
                "invalidation": list(prediction.invalidation),
            },
        )

    @property
    def horizon(self) -> Horizon:
        return Horizon(bars=self.horizon_bars, timeframe=self.timeframe)

    def verify(self) -> bool:
        """Whether the stored content still hashes to the recorded digest."""
        return self.content_hash == content_hash(
            model_id=self.model_id,
            model_version=self.model_version,
            asset=self.asset,
            timeframe=str(self.timeframe),
            horizon_bars=self.horizon_bars,
            as_of=self.as_of,
            up=self.distribution.up,
            flat=self.distribution.flat,
            down=self.distribution.down,
            confidence=self.confidence,
            move_threshold_pct=self.move_threshold_pct,
        )

    def is_due(self, now: datetime | None = None, settle: timedelta | None = None) -> bool:
        """Whether the horizon has elapsed far enough for the outcome to be final.

        ``settle`` is a grace period past the nominal resolution time. The bar covering
        the resolution instant is still forming at that instant; resolving from it would
        read an incomplete candle, which is the same look-ahead error in reverse.
        """
        moment = now or utcnow()
        grace = settle if settle is not None else self.timeframe.delta
        return moment >= self.resolves_at + grace


@dataclass(slots=True)
class ResolvedOutcome:
    """What happened, scored against the prediction that anticipated it."""

    prediction_id: str
    model_id: str
    asset: str
    timeframe: Timeframe
    horizon_bars: int
    regime: str
    volatility_bucket: str
    #: The instant the prediction was made. Carried through so outcomes from different
    #: models can be paired on the point they both forecast, which is what makes a
    #: paired significance test possible at all.
    as_of: datetime
    resolved_at: datetime
    realised_direction: Outcome
    realised_move_pct: float
    exit_price: float
    brier: float
    log_loss: float
    correct: bool
    probability_of_truth: float
    scored_at: datetime = field(default_factory=utcnow)

    @classmethod
    def score(
        cls,
        record: PredictionRecord,
        exit_price: float,
        resolved_at: datetime,
    ) -> ResolvedOutcome | None:
        """Score a record against a realised exit price.

        Uses the threshold *stored with the prediction*, not a freshly computed one.
        Re-deriving the threshold at resolution time would score the forecast against a
        different question from the one it answered.
        """
        if record.reference_price <= 0 or exit_price <= 0:
            return None
        move = (exit_price - record.reference_price) / record.reference_price * 100.0
        actual = Outcome.classify(move, record.move_threshold_pct)
        probability = record.distribution.probability(actual)
        brier = sum(
            (record.distribution.probability(o) - (1.0 if o is actual else 0.0)) ** 2
            for o in Outcome
        )
        import math

        return cls(
            prediction_id=record.prediction_id,
            model_id=record.model_id,
            asset=record.asset,
            timeframe=record.timeframe,
            horizon_bars=record.horizon_bars,
            regime=record.regime,
            volatility_bucket=record.volatility_bucket,
            as_of=record.as_of,
            resolved_at=resolved_at,
            realised_direction=actual,
            realised_move_pct=round(move, 6),
            exit_price=exit_price,
            brier=round(brier, 8),
            log_loss=round(-math.log(max(probability, 1e-12)), 8),
            correct=record.distribution.most_likely is actual,
            probability_of_truth=round(probability, 8),
        )
