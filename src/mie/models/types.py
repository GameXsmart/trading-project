"""The prediction contract.

Every model emits this same envelope, so the ensemble, the store and the UI never
special-case a model. Committed in ARCHITECTURE §9 before any model existed, which is
the point: a contract designed around a model's convenience stops being a contract.

Three properties are load-bearing.

**Probability is not confidence.** Probability is the model's estimate of an outcome;
confidence is how much the system trusts that estimate right now, given regime, recent
calibration, data quality and how much evidence there was. A 70%-up call from a model
with no calibration record in the current regime is published with low confidence, and
collapsing the two into one number destroys the distinction the whole design rests on.

**A distribution, never a point.** "Up" is not a prediction; `{up: .52, flat: .31,
down: .17}` is. The flat class is not padding — most of the time the market does not
move enough to matter, and a two-class model is forced to pretend otherwise.

**Falsifiable or worthless.** Every prediction carries invalidation conditions. A
forecast that cannot be wrong cannot be evaluated, and Phase 9 exists to evaluate
these.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mie.core.timeframes import Timeframe, utcnow

__all__ = [
    "Distribution",
    "Horizon",
    "Outcome",
    "Prediction",
    "PredictionEvidence",
]


class Outcome(StrEnum):
    """The three things that can happen over a horizon."""

    UP = "up"
    FLAT = "flat"
    DOWN = "down"

    @property
    def sign(self) -> int:
        return {"up": 1, "flat": 0, "down": -1}[self.value]

    @classmethod
    def classify(cls, return_pct: float, threshold_pct: float) -> Outcome:
        """Bucket a realised return.

        ``threshold_pct`` should be scaled to the asset's own recent volatility, not
        fixed: a 0.5% move is a large event on a quiet day and noise on a violent one,
        and a fixed band would make the classes mean different things in different
        regimes — which is precisely what breaks regime-sliced evaluation later.
        """
        if return_pct > threshold_pct:
            return cls.UP
        if return_pct < -threshold_pct:
            return cls.DOWN
        return cls.FLAT


class Distribution(BaseModel):
    """A probability distribution over the three outcomes."""

    model_config = ConfigDict(frozen=True)

    up: float = 1 / 3
    flat: float = 1 / 3
    down: float = 1 / 3

    @model_validator(mode="after")
    def _normalise(self) -> Distribution:
        total = self.up + self.flat + self.down
        if total <= 0:
            raise ValueError("distribution must have positive mass")
        # Renormalise rather than reject: models compose scores that rarely sum to
        # exactly one, and forcing every caller to normalise invites the one that
        # forgets.
        if abs(total - 1.0) > 1e-9:
            object.__setattr__(self, "up", self.up / total)
            object.__setattr__(self, "flat", self.flat / total)
            object.__setattr__(self, "down", self.down / total)
        return self

    def probability(self, outcome: Outcome) -> float:
        return {Outcome.UP: self.up, Outcome.FLAT: self.flat, Outcome.DOWN: self.down}[
            outcome
        ]

    @property
    def most_likely(self) -> Outcome:
        return max(Outcome, key=self.probability)

    @property
    def directional_edge(self) -> float:
        """How far from an even up/down split, in [-1, 1]. Ignores the flat mass."""
        return self.up - self.down

    @property
    def entropy(self) -> float:
        """Shannon entropy in bits, 0 (certain) to ~1.585 (maximally uncertain).

        A useful sanity check on any model: a forecaster whose entropy is routinely
        near zero on data this noisy is overconfident, not skilled.
        """
        import math

        return -sum(
            p * math.log2(p)
            for p in (self.up, self.flat, self.down)
            if p > 0
        )

    @classmethod
    def uniform(cls) -> Distribution:
        return cls(up=1 / 3, flat=1 / 3, down=1 / 3)

    @classmethod
    def from_edge(cls, edge: float, flat_mass: float = 0.34) -> Distribution:
        """Build a distribution from a directional lean in [-1, 1].

        The flat mass is held fixed and the remainder split according to the edge, so
        a model expressing a weak view produces a genuinely uncertain distribution
        instead of a confident one with a small margin.
        """
        edge = max(-1.0, min(1.0, edge))
        directional = max(0.0, 1.0 - flat_mass)
        up = directional * (0.5 + edge / 2.0)
        return cls(up=up, flat=flat_mass, down=directional - up)

    def blend(self, other: Distribution, weight: float = 0.5) -> Distribution:
        """Linear pool with another distribution."""
        weight = max(0.0, min(1.0, weight))
        return Distribution(
            up=self.up * (1 - weight) + other.up * weight,
            flat=self.flat * (1 - weight) + other.flat * weight,
            down=self.down * (1 - weight) + other.down * weight,
        )

    def __str__(self) -> str:  # pragma: no cover - display affordance
        return f"up {self.up:.0%} / flat {self.flat:.0%} / down {self.down:.0%}"


class Horizon(BaseModel):
    """How far ahead a prediction reaches."""

    model_config = ConfigDict(frozen=True)

    bars: int
    timeframe: Timeframe

    @property
    def duration(self) -> timedelta:
        return self.timeframe.delta * self.bars

    @property
    def hours(self) -> float:
        return self.duration.total_seconds() / 3600.0

    def label(self) -> str:
        hours = self.hours
        if hours < 1:
            return f"{hours * 60:.0f}m"
        if hours < 48:
            return f"{hours:.0f}h"
        return f"{hours / 24:.1f}d"

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.bars}x{self.timeframe} ({self.label()})"


class PredictionEvidence(BaseModel):
    """One reason for or against a prediction."""

    model_config = ConfigDict(frozen=True)

    label: str
    detail: str = ""
    #: Contribution on the [-1, 1] direction axis.
    contribution: float = 0.0

    def __str__(self) -> str:  # pragma: no cover
        arrow = "+" if self.contribution > 0 else "-" if self.contribution < 0 else "="
        return f"{arrow} {self.label}" + (f" ({self.detail})" if self.detail else "")


class Prediction(BaseModel):
    """One model's view of one asset over one horizon.

    Written *before* the outcome is knowable and never mutated, so Phase 9 can score
    it honestly. Anything that would need to change after the fact belongs in the
    outcome record, not here.
    """

    model_config = ConfigDict(frozen=True)

    model_id: str
    model_version: str = "1"
    asset: str
    timeframe: Timeframe
    horizon: Horizon
    #: The moment the prediction was made. Only data with a close time at or before
    #: this instant may have informed it.
    as_of: datetime
    distribution: Distribution = Field(default_factory=Distribution.uniform)
    #: Trust in this estimate, distinct from the probabilities themselves.
    confidence: float = 0.0
    expected_move_pct: float = 0.0
    expected_volatility_pct: float = 0.0
    evidence: list[PredictionEvidence] = Field(default_factory=list)
    counter_evidence: list[PredictionEvidence] = Field(default_factory=list)
    #: Concrete conditions that would falsify this call.
    invalidation: list[str] = Field(default_factory=list)
    #: The threshold used to define up/flat/down, so the outcome is scored the same
    #: way it was predicted.
    move_threshold_pct: float = 0.0
    regime: str = "unknown"
    data_quality: float = 1.0
    reference_price: float = 0.0
    created_at: datetime = Field(default_factory=utcnow)

    @property
    def is_actionable(self) -> bool:
        """Whether this is worth publishing at all.

        Deliberately strict. A prediction barely distinguishable from the unconditional
        distribution, or one the system does not trust, is noise dressed as insight —
        and "insufficient evidence" is an acceptable output.
        """
        return self.confidence >= 0.35 and abs(self.distribution.directional_edge) >= 0.08

    @property
    def resolves_at(self) -> datetime:
        return self.as_of + self.horizon.duration

    def score_outcome(self, realised_return_pct: float) -> Outcome:
        """Bucket a realised return using this prediction's own threshold."""
        return Outcome.classify(realised_return_pct, self.move_threshold_pct)

    def brier_score(self, actual: Outcome) -> float:
        """Multiclass Brier score for this prediction. Lower is better.

        The proper scoring rule of choice here: unlike accuracy it rewards
        well-calibrated uncertainty, so a model that says 40/35/25 and is wrong is
        penalised less than one that says 90/5/5 and is wrong — which is exactly the
        incentive a low-signal domain requires.
        """
        return sum(
            (self.distribution.probability(outcome) - (1.0 if outcome is actual else 0.0)) ** 2
            for outcome in Outcome
        )

    def log_loss(self, actual: Outcome, floor: float = 1e-12) -> float:
        """Negative log-likelihood of the realised outcome."""
        import math

        return -math.log(max(self.distribution.probability(actual), floor))

    def summary(self) -> str:
        return (
            f"{self.asset} {self.horizon.label()} [{self.model_id}]: "
            f"{self.distribution} | confidence {self.confidence:.0%} | "
            f"{'actionable' if self.is_actionable else 'insufficient evidence'}"
        )
