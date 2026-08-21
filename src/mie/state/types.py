"""Market-state vocabulary.

The types every later phase reasons about. Two ideas shape them:

**Direction, strength and confidence are three different things.** Direction is which
way; strength is how forcefully; confidence is how much the system trusts its own
reading. Collapsing them into one number destroys exactly the information the
prediction layer needs — a weak-but-certain drift and a violent-but-ambiguous lurch
are not the same market, and averaging them into "moderately bullish" describes
neither.

**Conflict between timeframes is a state, not an error.** A bullish daily with a
bearish 15m is a pullback inside an uptrend. Averaging those to "neutral" throws away
the most actionable reading on the board, so :class:`Alignment` names the conflict
patterns explicitly instead.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from mie.core.timeframes import Timeframe, utcnow

__all__ = [
    "Alignment",
    "Direction",
    "Evidence",
    "MarketState",
    "Regime",
    "TimeframeState",
]


class Direction(StrEnum):
    """Directional bias, ordered from most bearish to most bullish."""

    STRONG_DOWN = "strong_down"
    DOWN = "down"
    WEAK_DOWN = "weak_down"
    NEUTRAL = "neutral"
    WEAK_UP = "weak_up"
    UP = "up"
    STRONG_UP = "strong_up"

    @property
    def score(self) -> float:
        """Position on a [-1, 1] axis, for arithmetic that genuinely needs a number."""
        return _DIRECTION_SCORES[self]

    @property
    def sign(self) -> int:
        """-1, 0 or +1. Used for agreement tests, where magnitude is not the question."""
        score = self.score
        return 0 if score == 0 else (1 if score > 0 else -1)

    @classmethod
    def from_score(cls, score: float) -> Direction:
        """Nearest direction to a [-1, 1] score.

        Thresholds are deliberately wide around zero: most of the time markets are
        not going anywhere in particular, and a classifier that refuses to say so is
        just a random number generator with opinions.
        """
        if score >= 0.65:
            return cls.STRONG_UP
        if score >= 0.35:
            return cls.UP
        if score >= 0.12:
            return cls.WEAK_UP
        if score <= -0.65:
            return cls.STRONG_DOWN
        if score <= -0.35:
            return cls.DOWN
        if score <= -0.12:
            return cls.WEAK_DOWN
        return cls.NEUTRAL

    @property
    def is_bullish(self) -> bool:
        return self.score > 0

    @property
    def is_bearish(self) -> bool:
        return self.score < 0


_DIRECTION_SCORES: dict[Direction, float] = {
    Direction.STRONG_DOWN: -1.0,
    Direction.DOWN: -0.6,
    Direction.WEAK_DOWN: -0.25,
    Direction.NEUTRAL: 0.0,
    Direction.WEAK_UP: 0.25,
    Direction.UP: 0.6,
    Direction.STRONG_UP: 1.0,
}


class Regime(StrEnum):
    """Market regime — the context a prediction has to be interpreted inside.

    Regime is not just direction. A market can rise in a grinding low-volatility
    trend or in a violent short-squeeze recovery, and a model calibrated on one is
    not calibrated on the other. Phase 9 slices every performance metric by this
    field precisely because "the model is 60% accurate" means nothing without it.
    """

    STRONG_BULL = "strong_bull"
    BULL = "bull"
    WEAK_BULL = "weak_bull"
    NEUTRAL = "neutral"
    WEAK_BEAR = "weak_bear"
    BEAR = "bear"
    STRONG_BEAR = "strong_bear"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    CAPITULATION = "capitulation"
    RECOVERY = "recovery"


class Alignment(StrEnum):
    """How the timeframe hierarchy relates to itself.

    This is the field that makes multi-timeframe analysis worth doing. Without it,
    every conflicting reading degrades to "neutral" and the system becomes blind at
    exactly the moments that matter most.
    """

    #: Every timeframe agrees. The cleanest and rarest condition.
    ALIGNED_BULLISH = "aligned_bullish"
    ALIGNED_BEARISH = "aligned_bearish"
    #: Higher timeframes bullish, lower timeframes bearish — a dip inside an uptrend.
    PULLBACK_IN_UPTREND = "pullback_in_uptrend"
    #: Higher timeframes bearish, lower timeframes bullish — a bounce inside a downtrend.
    RALLY_IN_DOWNTREND = "rally_in_downtrend"
    #: Lower timeframes have turned against a fading higher-timeframe trend.
    POSSIBLE_REVERSAL = "possible_reversal"
    #: No coherent structure anywhere.
    RANGEBOUND = "rangebound"
    #: Genuinely contradictory without a clean hierarchy — say so rather than guess.
    CONFLICTED = "conflicted"

    @property
    def is_conflicted(self) -> bool:
        return self in (Alignment.CONFLICTED, Alignment.POSSIBLE_REVERSAL)


class Evidence(BaseModel):
    """One reason behind a reading, with its direction and weight.

    Every state carries these. A classification that cannot enumerate what produced
    it cannot be debugged, cannot be explained to a user, and cannot be improved when
    it turns out to be wrong.
    """

    model_config = ConfigDict(frozen=True)

    label: str
    detail: str = ""
    #: Contribution on the [-1, 1] direction axis.
    contribution: float = 0.0
    #: Relative importance within its group.
    weight: float = 1.0

    def __str__(self) -> str:  # pragma: no cover - display affordance
        arrow = "↑" if self.contribution > 0 else "↓" if self.contribution < 0 else "→"
        return f"{arrow} {self.label}" + (f" ({self.detail})" if self.detail else "")


class TimeframeState(BaseModel):
    """The read on one timeframe."""

    model_config = ConfigDict(frozen=True)

    asset: str
    timeframe: Timeframe
    as_of: datetime
    direction: Direction
    #: How forceful the move is, in [0, 1] — distinct from which way it points.
    strength: float = 0.0
    #: How much to trust this reading, in [0, 1].
    confidence: float = 0.0
    score: float = 0.0
    evidence: list[Evidence] = Field(default_factory=list)
    counter_evidence: list[Evidence] = Field(default_factory=list)
    volatility_pct: float | None = None
    close: float | None = None
    data_quality: float = 1.0

    @property
    def is_usable(self) -> bool:
        """Below this, the reading should inform nothing downstream.

        Publishing a direction the system does not believe in is worse than
        publishing nothing, because it looks identical to one it does believe in.
        """
        return self.confidence >= 0.25

    def summary(self) -> str:
        return (
            f"{self.timeframe}: {self.direction} "
            f"(strength {self.strength:.2f}, confidence {self.confidence:.2f})"
        )


class MarketState(BaseModel):
    """The hierarchical read across every timeframe for one asset."""

    model_config = ConfigDict(frozen=True)

    asset: str
    as_of: datetime = Field(default_factory=utcnow)
    #: Per-timeframe states, slowest first. Stored in full, not just summarised —
    #: the explanation panel and regime-conditional evaluation both need the levels.
    timeframes: list[TimeframeState] = Field(default_factory=list)
    bias: Direction = Direction.NEUTRAL
    alignment: Alignment = Alignment.RANGEBOUND
    regime: Regime = Regime.NEUTRAL
    #: Agreement across the hierarchy, in [0, 1].
    agreement: float = 0.0
    confidence: float = 0.0
    interpretation: str = ""
    evidence: list[Evidence] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    data_quality: float = 1.0
    details: dict[str, Any] = Field(default_factory=dict)

    def state_for(self, timeframe: Timeframe) -> TimeframeState | None:
        return next((s for s in self.timeframes if s.timeframe is timeframe), None)

    @property
    def macro(self) -> TimeframeState | None:
        """The slowest available timeframe — the one that sets the prior."""
        return self.timeframes[0] if self.timeframes else None

    @property
    def micro(self) -> TimeframeState | None:
        """The fastest available timeframe."""
        return self.timeframes[-1] if self.timeframes else None

    @property
    def is_actionable(self) -> bool:
        """Whether this state is coherent and trusted enough to build on."""
        return self.confidence >= 0.4 and not self.alignment.is_conflicted

    def summary(self) -> str:
        return (
            f"{self.asset}: {self.bias} | {self.alignment} | {self.regime} "
            f"(agreement {self.agreement:.0%}, confidence {self.confidence:.0%})"
        )
