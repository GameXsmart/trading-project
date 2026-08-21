"""Pattern registry — the gate between detection and influence.

Phase 4's rule is that a detector whose base rate is statistically indistinguishable
from chance is *removed, not shipped with a caveat*. This module is where that rule
acquires teeth.

The distinction it enforces is between two very different uses of a pattern:

* **Descriptive.** "A breakout occurred here" is an observation about the chart. It is
  true regardless of what happened next, and it is useful for annotating a display.
* **Predictive.** "A breakout occurred, therefore price is more likely to rise" is a
  claim about the future, and it requires evidence.

Detectors keep running — the observation is cheap and honest. What the registry
withholds is *influence*: unless a pattern has stored statistics showing a significant
edge over the unconditional baseline for that exact asset, timeframe and horizon, it
contributes nothing to any prediction. There is no caveat mode, no reduced weight, no
"directionally suggestive". It either earned its place or it did not.

Evidence is keyed per (pattern, asset, timeframe, horizon) on purpose. Measurement
showed a pattern can clear the bar on one asset and fail on another — which is
precisely the situation where a single global verdict would be wrong in both
directions at once.
"""

from __future__ import annotations

from collections.abc import Sequence

from mie.core.logging import get_logger
from mie.core.timeframes import Timeframe
from mie.patterns.types import Detection, PatternKind, PatternStats

log = get_logger(__name__)

__all__ = ["PatternRegistry"]


class PatternRegistry:
    """Holds measured pattern statistics and decides what may influence predictions."""

    def __init__(self, stats: Sequence[PatternStats] = ()) -> None:
        self._stats: dict[tuple[PatternKind, str, str, int], PatternStats] = {}
        for stat in stats:
            self.add(stat)

    def add(self, stat: PatternStats) -> None:
        self._stats[self._key(stat.kind, stat.asset, stat.timeframe, stat.horizon_bars)] = stat

    def extend(self, stats: Sequence[PatternStats]) -> None:
        for stat in stats:
            self.add(stat)

    @staticmethod
    def _key(
        kind: PatternKind, asset: str, timeframe: Timeframe, horizon: int
    ) -> tuple[PatternKind, str, str, int]:
        return (kind, asset.upper(), str(timeframe), horizon)

    # ----------------------------------------------------------------- lookup

    def stats_for(
        self, kind: PatternKind, asset: str, timeframe: Timeframe, horizon: int
    ) -> PatternStats | None:
        return self._stats.get(self._key(kind, asset, timeframe, horizon))

    def is_informative(
        self, kind: PatternKind, asset: str, timeframe: Timeframe, horizon: int
    ) -> bool:
        """Whether this pattern earned the right to influence predictions here.

        Absence of evidence is treated as absence of permission. An unmeasured pattern
        is not "probably fine" — it is simply unproven, and unproven patterns do not
        get to move a prediction.
        """
        stat = self.stats_for(kind, asset, timeframe, horizon)
        return stat is not None and stat.is_informative

    def admitted(self) -> list[PatternStats]:
        """Every pattern that survived measurement."""
        return sorted(
            (s for s in self._stats.values() if s.is_informative),
            key=lambda s: -abs(s.estimate.edge),
        )

    def rejected(self) -> list[PatternStats]:
        """Every pattern that was measured and failed."""
        return sorted(
            (s for s in self._stats.values() if not s.is_informative),
            key=lambda s: s.estimate.p_value,
        )

    # ------------------------------------------------------------------ gate

    def filter_detections(
        self, detections: Sequence[Detection], horizon: int
    ) -> list[Detection]:
        """Keep only detections whose pattern is evidenced for this horizon.

        This is the call every downstream consumer must go through. Anything that
        reads raw detector output instead is bypassing the gate, and the gate is the
        only thing standing between measured findings and confident folklore.
        """
        kept = [
            d
            for d in detections
            if self.is_informative(d.kind, d.asset, d.timeframe, horizon)
        ]
        if len(kept) != len(detections):
            log.debug(
                "pattern_detections_gated",
                horizon=horizon,
                kept=len(kept),
                withheld=len(detections) - len(kept),
            )
        return kept

    def expected_edge(
        self, detection: Detection, horizon: int
    ) -> float:
        """Measured edge over baseline, or 0.0 if the pattern is not evidenced.

        Returning the *measured* edge rather than a hand-assigned weight is the point:
        downstream code cannot inflate a pattern's importance beyond what the history
        actually supports.
        """
        stat = self.stats_for(detection.kind, detection.asset, detection.timeframe, horizon)
        if stat is None or not stat.is_informative:
            return 0.0
        return stat.estimate.edge

    def report(self) -> str:
        """Human-readable summary of what passed and what did not."""
        admitted = self.admitted()
        rejected = self.rejected()
        lines = [
            f"{len(admitted)} of {len(self._stats)} measured pattern/horizon pairs are "
            f"informative:"
        ]
        lines.extend(f"  ADMITTED  {s.summary()}" for s in admitted)
        if rejected:
            lines.append(f"{len(rejected)} withheld from the predictive path:")
            lines.extend(f"  withheld  {s.summary()}" for s in rejected[:10])
            if len(rejected) > 10:
                lines.append(f"  ... and {len(rejected) - 10} more")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._stats)
