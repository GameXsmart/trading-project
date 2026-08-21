"""Multi-timeframe hierarchy.

Combines per-timeframe states into one :class:`MarketState`.

The whole reason this module exists is that **timeframes are not peers and must not
be averaged**. Averaging is the obvious implementation and it is wrong in a specific,
expensive way: a bullish daily with a bearish 15m averages to "neutral", which is the
one description that fits neither. The actual state is a pullback inside an uptrend —
arguably the most actionable configuration on the board — and averaging erases it.

So the hierarchy is read structurally instead:

* **Higher timeframes set the prior.** A daily trend is a fact about the market that a
  fifteen-minute wobble does not repeal.
* **Lower timeframes update it.** They say where inside that structure price currently
  sits, and they are the first place a genuine reversal shows up.
* **The relationship between the two is named**, not numerically flattened.

Weighting is by timeframe rank, so the daily counts for more than the 5m — but the
weights only ever produce the *bias*, never the alignment classification.
"""

from __future__ import annotations

from collections.abc import Sequence

from mie.core.logging import get_logger
from mie.core.timeframes import Timeframe, utcnow
from mie.state.types import (
    Alignment,
    Direction,
    Evidence,
    MarketState,
    Regime,
    TimeframeState,
)

log = get_logger(__name__)

__all__ = ["HierarchyAnalyzer", "split_hierarchy"]

#: Volatility bands, as annualised percent. Crypto's "normal" is another market's
#: crisis, so these are calibrated to crypto rather than to equities.
_LOW_VOL = 35.0
_HIGH_VOL = 110.0

def split_hierarchy(
    states: Sequence[TimeframeState],
) -> tuple[list[TimeframeState], list[TimeframeState]]:
    """Split states into (higher, lower) — structural versus tactical.

    The split is **relative to the set being analysed**, not anchored to a fixed
    timeframe. "Structural" and "tactical" are positional roles: analysing 1d/4h/1h
    makes the 1h the tactical read, while analysing 1h/15m/5m makes the 5m tactical.

    An absolute hinge gets this badly wrong. Pinned at 1h, a 1d/4h/1h request puts
    every timeframe on the structural side, leaving the tactical group empty — and
    since pullback and counter-trend detection compare the two groups, those readings
    become silently unreachable. That failure is invisible in unit tests built from
    hand-made states and only shows up against real data.

    With an odd number of timeframes the middle one counts as structural: the prior
    should be the broader of the two readings.
    """
    if not states:
        return [], []
    ordered = sorted(states, key=lambda s: -s.timeframe.rank)
    if len(ordered) == 1:
        return ordered, []
    cut = (len(ordered) + 1) // 2
    return ordered[:cut], ordered[cut:]


class HierarchyAnalyzer:
    """Reduces per-timeframe states to one coherent market state."""

    def analyse(
        self, asset: str, states: Sequence[TimeframeState], data_quality: float = 1.0
    ) -> MarketState:
        # Slowest first: the macro read leads, everything else qualifies it.
        ordered = sorted(states, key=lambda s: -s.timeframe.rank)
        usable = [s for s in ordered if s.is_usable]

        if not usable:
            return MarketState(
                asset=asset.upper(),
                as_of=utcnow(),
                timeframes=list(ordered),
                bias=Direction.NEUTRAL,
                alignment=Alignment.RANGEBOUND,
                regime=Regime.NEUTRAL,
                agreement=0.0,
                confidence=0.0,
                interpretation=(
                    "Insufficient evidence: no timeframe produced a reading confident "
                    "enough to act on."
                ),
                data_quality=data_quality,
            )

        bias, weighted_score = self._bias(usable)
        agreement = self._agreement(usable)
        higher, lower = split_hierarchy(usable)
        alignment = self._alignment(higher, lower, agreement)
        regime = self._regime(usable, bias, alignment)
        confidence = self._confidence(usable, agreement, alignment, data_quality)
        conflicts = self._conflicts(higher, lower)
        interpretation = self._interpret(asset, bias, alignment, regime, higher, lower)

        evidence = [
            Evidence(
                label=f"{s.timeframe} is {s.direction}",
                detail=f"strength {s.strength:.2f}, confidence {s.confidence:.2f}",
                contribution=s.direction.score,
                weight=_rank_weight(s.timeframe),
            )
            for s in usable
        ]

        return MarketState(
            asset=asset.upper(),
            as_of=max(s.as_of for s in usable),
            timeframes=list(ordered),
            bias=bias,
            alignment=alignment,
            regime=regime,
            agreement=round(agreement, 4),
            confidence=round(confidence, 4),
            interpretation=interpretation,
            evidence=evidence,
            conflicts=conflicts,
            data_quality=data_quality,
            details={
                "weighted_score": round(weighted_score, 4),
                "higher_timeframes": [str(s.timeframe) for s in higher],
                "lower_timeframes": [str(s.timeframe) for s in lower],
                "usable_timeframes": len(usable),
                "total_timeframes": len(ordered),
            },
        )

    # ------------------------------------------------------------------ pieces

    def _bias(self, states: Sequence[TimeframeState]) -> tuple[Direction, float]:
        """Overall directional bias, weighted by timeframe rank and confidence.

        This produces the headline direction only. It deliberately does *not* decide
        the alignment: that is what keeps a conflicted market from being reported as a
        clean weak trend just because the slower timeframes outvoted the faster ones.
        """
        total_weight = 0.0
        total = 0.0
        for state in states:
            weight = _rank_weight(state.timeframe) * max(0.1, state.confidence)
            total += state.score * weight
            total_weight += weight
        score = total / total_weight if total_weight else 0.0
        return Direction.from_score(score), score

    def _agreement(self, states: Sequence[TimeframeState]) -> float:
        """Rank-weighted directional agreement across the hierarchy, in [0, 1]."""
        if len(states) < 2:
            return 1.0 if states else 0.0
        voting = [s for s in states if s.direction.sign != 0]
        if not voting:
            return 0.0  # everything neutral: no agreement to speak of, and no conflict
        bullish = sum(_rank_weight(s.timeframe) for s in voting if s.direction.is_bullish)
        bearish = sum(_rank_weight(s.timeframe) for s in voting if s.direction.is_bearish)
        total = bullish + bearish
        return abs(bullish - bearish) / total if total else 0.0

    def _alignment(
        self,
        higher: Sequence[TimeframeState],
        lower: Sequence[TimeframeState],
        agreement: float,
    ) -> Alignment:
        """Name the relationship between structural and tactical timeframes."""
        # Each group is read through its most informative member for its role: the
        # structural group is led by its slowest timeframe, the tactical group by its
        # fastest, since a correction appears at the fast end first.
        higher_sign = _net_sign(higher)
        lower_sign = _net_sign(lower, fast_first=True)

        if higher_sign == 0 and lower_sign == 0:
            return Alignment.RANGEBOUND

        if higher_sign > 0 and lower_sign >= 0:
            return Alignment.ALIGNED_BULLISH if agreement > 0.6 else Alignment.CONFLICTED
        if higher_sign < 0 and lower_sign <= 0:
            return Alignment.ALIGNED_BEARISH if agreement > 0.6 else Alignment.CONFLICTED

        # The interesting cases: the two halves point opposite ways.
        if higher_sign > 0 and lower_sign < 0:
            # A weakening prior with the tactical timeframes already turned is a
            # candidate reversal rather than a routine dip.
            return (
                Alignment.POSSIBLE_REVERSAL
                if _is_fading(higher)
                else Alignment.PULLBACK_IN_UPTREND
            )
        if higher_sign < 0 and lower_sign > 0:
            return (
                Alignment.POSSIBLE_REVERSAL
                if _is_fading(higher)
                else Alignment.RALLY_IN_DOWNTREND
            )

        # One half is neutral and the other is not: structure without confirmation.
        if higher_sign == 0:
            return Alignment.CONFLICTED
        return Alignment.ALIGNED_BULLISH if higher_sign > 0 else Alignment.ALIGNED_BEARISH

    def _regime(
        self,
        states: Sequence[TimeframeState],
        bias: Direction,
        alignment: Alignment,
    ) -> Regime:
        """Classify the regime the market is currently in.

        Volatility is checked before direction, because a violent market is a
        different environment regardless of which way it is pointing, and models
        calibrated in calm conditions do not transfer into it.
        """
        volatilities = [s.volatility_pct for s in states if s.volatility_pct is not None]
        volatility = sum(volatilities) / len(volatilities) if volatilities else None
        strength = sum(s.strength for s in states) / len(states) if states else 0.0

        if volatility is not None and volatility >= _HIGH_VOL:
            # Extreme volatility with a decisive downside bias is capitulation; the
            # mirror case is a recovery snap-back rather than an orderly bull trend.
            if bias.is_bearish and strength > 0.5:
                return Regime.CAPITULATION
            if bias.is_bullish and alignment is Alignment.RALLY_IN_DOWNTREND:
                return Regime.RECOVERY
            return Regime.HIGH_VOLATILITY

        if volatility is not None and volatility <= _LOW_VOL and abs(bias.score) < 0.35:
            # Quiet and directionless: which way the structure leans distinguishes
            # accumulation from distribution.
            if alignment is Alignment.PULLBACK_IN_UPTREND:
                return Regime.ACCUMULATION
            if alignment is Alignment.RALLY_IN_DOWNTREND:
                return Regime.DISTRIBUTION
            return Regime.LOW_VOLATILITY

        if alignment is Alignment.POSSIBLE_REVERSAL:
            return Regime.RECOVERY if bias.is_bullish else Regime.DISTRIBUTION

        return {
            Direction.STRONG_UP: Regime.STRONG_BULL,
            Direction.UP: Regime.BULL,
            Direction.WEAK_UP: Regime.WEAK_BULL,
            Direction.NEUTRAL: Regime.NEUTRAL,
            Direction.WEAK_DOWN: Regime.WEAK_BEAR,
            Direction.DOWN: Regime.BEAR,
            Direction.STRONG_DOWN: Regime.STRONG_BEAR,
        }[bias]

    def _confidence(
        self,
        states: Sequence[TimeframeState],
        agreement: float,
        alignment: Alignment,
        data_quality: float,
    ) -> float:
        """Trust in the combined reading.

        A conflicted hierarchy is penalised rather than averaged away: when the
        timeframes genuinely disagree, low confidence *is* the correct output.
        """
        weights = [_rank_weight(s.timeframe) for s in states]
        mean_confidence = sum(
            s.confidence * w for s, w in zip(states, weights, strict=True)
        ) / sum(weights)

        # Agreement contributes but cannot dominate: unanimous timeframes reading
        # weak signals should not produce high confidence.
        combined = mean_confidence * (0.55 + 0.45 * agreement)
        if alignment.is_conflicted:
            combined *= 0.6
        if len(states) < 2:
            # A single timeframe cannot corroborate itself.
            combined *= 0.75
        return max(0.0, min(1.0, combined * max(0.0, min(1.0, data_quality))))

    def _conflicts(
        self, higher: Sequence[TimeframeState], lower: Sequence[TimeframeState]
    ) -> list[str]:
        """Explicit list of disagreements, surfaced rather than smoothed over."""
        conflicts: list[str] = []
        for high in higher:
            for low in lower:
                if high.direction.sign and low.direction.sign and high.direction.sign != low.direction.sign:
                    conflicts.append(
                        f"{high.timeframe} is {high.direction} while "
                        f"{low.timeframe} is {low.direction}"
                    )
        return conflicts[:6]

    def _interpret(
        self,
        asset: str,
        bias: Direction,
        alignment: Alignment,
        regime: Regime,
        higher: Sequence[TimeframeState],
        lower: Sequence[TimeframeState],
    ) -> str:
        """A sentence a human can check against the chart."""
        # Groups are described by their *net* reading, not as though every member
        # agrees: a structural group can be net-negative while containing one mildly
        # positive timeframe, and "higher timeframes remain negative" would then be
        # an overstatement of what the data actually says.
        high_label = _describe(higher)
        low_label = _describe(lower)

        if alignment is Alignment.PULLBACK_IN_UPTREND:
            return (
                f"{asset} is in a pullback within a larger uptrend: higher timeframes "
                f"[{high_label}] are net constructive while lower timeframes "
                f"[{low_label}] are correcting. This is a short-term retracement inside "
                f"a bullish structure, not a bearish market."
            )
        if alignment is Alignment.RALLY_IN_DOWNTREND:
            return (
                f"{asset} is rallying inside a larger downtrend: higher timeframes "
                f"[{high_label}] are net negative while lower timeframes [{low_label}] "
                f"are bouncing. This is a counter-trend move, not a confirmed recovery."
            )
        if alignment is Alignment.POSSIBLE_REVERSAL:
            return (
                f"{asset} may be turning: the higher-timeframe trend [{high_label}] is "
                f"losing strength while lower timeframes [{low_label}] have already "
                f"reversed. Treat this as an unconfirmed transition, not an established "
                f"trend in either direction."
            )
        if alignment is Alignment.ALIGNED_BULLISH:
            return (
                f"{asset} is bullish across the hierarchy [{high_label}; {low_label}] "
                f"in a {regime} regime. Timeframes corroborate each other."
            )
        if alignment is Alignment.ALIGNED_BEARISH:
            return (
                f"{asset} is bearish across the hierarchy [{high_label}; {low_label}] "
                f"in a {regime} regime. Timeframes corroborate each other."
            )
        if alignment is Alignment.RANGEBOUND:
            return (
                f"{asset} has no clear directional structure on any timeframe "
                f"[{high_label}; {low_label}]. Rangebound conditions."
            )
        return (
            f"{asset} shows conflicting signals without a clean hierarchy "
            f"[{high_label}; {low_label}]. Bias reads {bias}, but the timeframes do not "
            f"corroborate it — treat directional conclusions with caution."
        )


# ---------------------------------------------------------------------- helpers


def _describe(states: Sequence[TimeframeState]) -> str:
    """Describe a group by its members and its net lean.

    Naming the net explicitly keeps the sentence honest when the group is not
    unanimous, which is common and is exactly the case a summary tends to gloss over.
    """
    if not states:
        return "none"
    members = ", ".join(f"{s.timeframe} {s.direction}" for s in states)
    if len(states) < 2:
        return members
    net = {1: "net bullish", -1: "net bearish", 0: "net flat"}[_net_sign(states)]
    return f"{members} — {net}"


def _rank_weight(timeframe: Timeframe) -> float:
    """Weight by position in the hierarchy.

    Roughly geometric in rank: a daily reading carries several times the weight of a
    five-minute one, because it describes a structure that takes far longer to change
    and is correspondingly harder to invalidate.
    """
    return 1.0 + timeframe.rank * 0.6


def _net_sign(states: Sequence[TimeframeState], fast_first: bool = False) -> int:
    """Weighted net direction of a group, as -1, 0 or +1.

    Weighted by timeframe rank, deliberately *not* by confidence. Confidence already
    governs this reading in two other places — unusable states are filtered out before
    they get here, and :func:`_is_fading` decides whether a weak structural trend can
    absorb a counter-move. Folding it in a third time makes a low-confidence trend
    disappear into the dead zone, which reports genuine structure as noise.

    ``fast_first`` inverts the weighting for the tactical group. Within the structural
    group the slowest timeframe leads, because it describes the most durable
    structure. Within the tactical group the *fastest* leads, because a correction
    shows up there first — weighting that group slow-first lets a lukewarm mid
    timeframe cancel out the very signal the group exists to detect.
    """
    if not states:
        return 0
    weights = {
        s.timeframe: (1.0 / _rank_weight(s.timeframe) if fast_first else _rank_weight(s.timeframe))
        for s in states
    }
    total = sum(s.direction.score * weights[s.timeframe] for s in states)
    weight = sum(weights.values())
    normalised = total / weight if weight else 0.0
    # A deliberately wide dead zone: a barely-positive net reading is not a direction,
    # and treating it as one manufactures conflicts that are really just noise.
    if normalised > 0.10:
        return 1
    if normalised < -0.10:
        return -1
    return 0


def _is_fading(states: Sequence[TimeframeState]) -> bool:
    """Whether the structural trend is weak enough to be losing its grip.

    This is what separates a routine pullback from a candidate reversal: a strong
    higher-timeframe trend absorbs a counter-move, a weak one may not.
    """
    if not states:
        return False
    mean_strength = sum(s.strength for s in states) / len(states)
    mean_confidence = sum(s.confidence for s in states) / len(states)
    return mean_strength < 0.4 or mean_confidence < 0.45
