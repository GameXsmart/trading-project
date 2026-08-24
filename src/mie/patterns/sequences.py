"""Sequence mining — the "chains" of requirement §6.

Instead of only asking *what pattern exists right now*, this asks **what sequence of
events tends to surround this one**: high volatility → liquidation spike → reversal →
volume expansion, and so on.

The temptation here is enormous and the discipline has to match it. Any sufficiently
long price history contains a vast number of sequences, and searching them all will
produce spectacular-looking chains that are pure combinatorics. Three safeguards:

* **Sequences are declared by enumeration, then tested** — the same discipline the
  detectors follow. The miner does not search for whatever chain looks best and then
  report it as a discovery; it enumerates every chain that occurred often enough and
  tests all of them.
* **The multiple-comparison correction covers the whole enumeration.** Mining pairs
  and triples across a pattern alphabet generates thousands of hypotheses. Testing
  them at an uncorrected p < 0.05 would manufacture dozens of "chains" from noise.
* **The comparison is against the market's own behaviour**, not a coin flip, and not
  against the base rate of the chain's final element — which would quietly answer a
  different, easier question.

A transition matrix over market states is provided alongside, because "what usually
follows this state" is often better answered by a transition probability with an
interval than by a mined sequence.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from itertools import pairwise

from mie.core.logging import get_logger
from mie.core.timeframes import Timeframe
from mie.patterns.statistics import (
    ProportionEstimate,
    benjamini_hochberg,
    compare_to_baseline,
    wilson_interval,
)
from mie.patterns.types import Detection, PatternKind

log = get_logger(__name__)

__all__ = ["Chain", "SequenceMiner", "Transition", "TransitionMatrix"]

#: Chains shorter than this are single events, which the detectors already cover.
_MIN_CHAIN_LENGTH = 2
_MAX_CHAIN_LENGTH = 3

#: A chain seen fewer times than this cannot support any claim.
_MIN_OCCURRENCES = 25

#: Maximum bars between consecutive links. Beyond this the two events are not part of
#: the same episode, they merely both happened.
_MAX_GAP_BARS = 12


@dataclass(frozen=True, slots=True)
class Chain:
    """An ordered sequence of pattern events with its measured forward behaviour."""

    steps: tuple[PatternKind, ...]
    asset: str
    timeframe: Timeframe
    horizon_bars: int
    occurrences: int
    estimate: ProportionEstimate
    median_return_pct: float

    @property
    def is_informative(self) -> bool:
        return (
            self.occurrences >= _MIN_OCCURRENCES
            and self.estimate.significant
            and self.estimate.interval_excludes_baseline
        )

    @property
    def label(self) -> str:
        return " -> ".join(str(step) for step in self.steps)

    def summary(self) -> str:
        verdict = "informative" if self.is_informative else "indistinguishable from chance"
        return (
            f"{self.label} ({self.asset} {self.timeframe} +{self.horizon_bars}): "
            f"{self.estimate.summary()} -> {verdict}"
        )


@dataclass(frozen=True, slots=True)
class Transition:
    """Probability of moving from one state to another, with its uncertainty."""

    source: str
    target: str
    count: int
    total: int
    probability: float
    low: float
    high: float

    def summary(self) -> str:
        return (
            f"{self.source} -> {self.target}: {self.probability:.0%} "
            f"[{self.low:.0%}-{self.high:.0%}] (n={self.total})"
        )


@dataclass(slots=True)
class TransitionMatrix:
    """Empirical state-transition probabilities with confidence intervals."""

    transitions: list[Transition] = field(default_factory=list)
    states: list[str] = field(default_factory=list)

    def from_state(self, state: str) -> list[Transition]:
        return sorted(
            (t for t in self.transitions if t.source == state),
            key=lambda t: -t.probability,
        )

    def most_likely_after(self, state: str) -> Transition | None:
        candidates = self.from_state(state)
        return candidates[0] if candidates else None


class SequenceMiner:
    """Enumerates pattern chains and state transitions, then tests them."""

    def __init__(
        self,
        horizons: Sequence[int] = (12,),
        min_occurrences: int = _MIN_OCCURRENCES,
        max_gap_bars: int = _MAX_GAP_BARS,
        max_length: int = _MAX_CHAIN_LENGTH,
        false_discovery_rate: float = 0.05,
    ) -> None:
        self.horizons = tuple(horizons)
        self.min_occurrences = min_occurrences
        self.max_gap_bars = max_gap_bars
        self.max_length = max_length
        self.false_discovery_rate = false_discovery_rate

    # ------------------------------------------------------------------ chains

    def mine(
        self,
        detections: Sequence[Detection],
        closes: Sequence[float],
        index_of: dict[datetime, int],
        asset: str,
        timeframe: Timeframe,
    ) -> list[Chain]:
        """Enumerate chains that occurred often enough, and test every one of them."""
        ordered = sorted(detections, key=lambda d: d.at)
        positioned = [
            (index_of[d.at], d.kind) for d in ordered if d.at in index_of
        ]
        if len(positioned) < self.min_occurrences:
            return []

        raw: list[Chain] = []
        for length in range(_MIN_CHAIN_LENGTH, self.max_length + 1):
            occurrences = self._collect(positioned, length)
            for steps, indices in occurrences.items():
                if len(indices) < self.min_occurrences:
                    continue
                for horizon in self.horizons:
                    chain = self._test(steps, indices, closes, horizon, asset, timeframe)
                    if chain is not None:
                        raw.append(chain)

        return self._apply_fdr(raw)

    def _collect(
        self, positioned: Sequence[tuple[int, PatternKind]], length: int
    ) -> dict[tuple[PatternKind, ...], list[int]]:
        """Every chain of ``length`` consecutive detections within the gap limit.

        Keyed by the sequence of kinds; the value is the bar index of each chain's
        final element, which is the moment the chain becomes observable.
        """
        found: dict[tuple[PatternKind, ...], list[int]] = defaultdict(list)
        for start in range(len(positioned) - length + 1):
            window = positioned[start : start + length]
            gaps = [
                window[i + 1][0] - window[i][0] for i in range(len(window) - 1)
            ]
            # Every gap must be positive (a real progression, not simultaneous
            # detections on one bar) and short enough to be the same episode.
            if any(gap <= 0 or gap > self.max_gap_bars for gap in gaps):
                continue
            steps = tuple(kind for _, kind in window)
            found[steps].append(window[-1][0])
        return found

    def _test(
        self,
        steps: tuple[PatternKind, ...],
        indices: Sequence[int],
        closes: Sequence[float],
        horizon: int,
        asset: str,
        timeframe: Timeframe,
    ) -> Chain | None:
        """Measure a chain's forward outcome against the unconditional rate."""
        # Thin overlapping occurrences, as elsewhere: consecutive chains share most of
        # their forward window and are not independent trials.
        thinned: list[int] = []
        for index in sorted(set(indices)):
            if thinned and index - thinned[-1] < horizon:
                continue
            if index + horizon < len(closes) and closes[index] > 0:
                thinned.append(index)
        if len(thinned) < 5:
            return None

        returns = [
            (closes[i + horizon] - closes[i]) / closes[i] * 100.0 for i in thinned
        ]
        rose = sum(1 for r in returns if r > 0)
        baseline_rose, baseline_total = _baseline_up_rate(closes, horizon)
        estimate = compare_to_baseline(rose, len(thinned), baseline_rose, baseline_total)
        return Chain(
            steps=steps,
            asset=asset.upper(),
            timeframe=timeframe,
            horizon_bars=horizon,
            occurrences=len(thinned),
            estimate=estimate,
            median_return_pct=sorted(returns)[len(returns) // 2],
        )

    def _apply_fdr(self, chains: Sequence[Chain]) -> list[Chain]:
        """Correct across every chain tested, not per chain.

        Mining pairs and triples generates thousands of hypotheses. Without this the
        output would be a list of impressive chains assembled entirely from noise.
        """
        if not chains:
            return []
        rejected = benjamini_hochberg(
            [c.estimate.p_value for c in chains], self.false_discovery_rate
        )
        from dataclasses import replace

        return [
            replace(chain, estimate=replace(chain.estimate, significant=is_significant))
            for chain, is_significant in zip(chains, rejected, strict=True)
        ]

    # ------------------------------------------------------------- transitions

    def transition_matrix(
        self, states: Sequence[str], min_observations: int = 20
    ) -> TransitionMatrix:
        """Empirical transition probabilities between consecutive states.

        Self-transitions are excluded: states persist for long stretches, so including
        them produces a matrix whose every row says "most likely, no change" — true,
        and completely uninformative about what happens when something does change.
        """
        counts: Counter[tuple[str, str]] = Counter()
        totals: Counter[str] = Counter()
        for current, following in pairwise(states):
            if current == following:
                continue
            counts[(current, following)] += 1
            totals[current] += 1

        transitions = []
        for (source, target), count in counts.items():
            total = totals[source]
            if total < min_observations:
                continue
            low, high = wilson_interval(count, total)
            transitions.append(
                Transition(
                    source=source,
                    target=target,
                    count=count,
                    total=total,
                    probability=count / total,
                    low=low,
                    high=high,
                )
            )
        return TransitionMatrix(
            transitions=sorted(transitions, key=lambda t: (t.source, -t.probability)),
            states=sorted(set(states)),
        )


def _baseline_up_rate(closes: Sequence[float], horizon: int) -> tuple[int, int]:
    rose = total = 0
    for index in range(0, len(closes) - horizon, horizon):
        if closes[index] <= 0:
            continue
        total += 1
        rose += int(closes[index + horizon] > closes[index])
    return rose, total
