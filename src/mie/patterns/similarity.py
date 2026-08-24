"""Historical similarity search.

Answers "when has the market looked like this before, and what happened next?" — and,
just as importantly, is willing to answer "it has not".

The naive version of this idea is dangerous. Take the current feature vector, find the
k nearest historical vectors, report what followed. That will always return an answer,
because k nearest neighbours exist in any dataset regardless of whether any of them is
actually *similar*. The result is a confident-looking distribution assembled from
situations that have nothing to do with the present one.

Four constraints make the search honest:

* **A distance ceiling.** Neighbours beyond it are not analogues, and if too few
  survive, the correct output is `insufficient evidence` rather than the nearest
  available strangers.
* **Scale-free features only.** Comparing raw prices would match BTC in 2021 to BTC in
  2021 and nothing else. The comparison runs on bounded, dimensionless quantities
  (RSI, %b, ADX, ATR as a percentage of price) so a match means "behaving similarly",
  not "priced similarly".
* **Normalisation from the past only.** Feature means and spreads are computed from
  data available at the query point. Standardising over the whole history would leak
  the future into the scaling.
* **An embargo.** Neighbours must be at least one horizon in the past, so a
  "neighbour" cannot be a bar whose forward window overlaps the query's own.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from statistics import median

from mie.core.logging import get_logger
from mie.core.timeframes import Timeframe
from mie.patterns.statistics import ProportionEstimate, compare_to_baseline

log = get_logger(__name__)

__all__ = ["COMPARISON_FEATURES", "Analogue", "SimilarityEngine", "SimilarityResult"]

#: Features used for comparison. Every one is bounded or dimensionless, so two eras
#: with wildly different price levels can still be recognised as behaving alike.
COMPARISON_FEATURES: tuple[str, ...] = (
    "rsi_14",
    "bb_20.percent_b",
    "bb_20.bandwidth",
    "adx_14.adx",
    "adx_14.plus_di",
    "adx_14.minus_di",
    "atr_14.atr_pct",
    "roc_10",
    "stoch_14.k",
    "realised_vol_20",
    "vwap.vwap_distance_pct",
    "structure_trend",
)

#: A neighbour counts as an analogue only if it is this much closer than a *typical*
#: pair drawn from the same history.
#:
#: Calibrated rather than guessed, and expressed relatively rather than absolutely.
#: For k standardised, roughly independent dimensions the expected distance between
#: two unrelated samples is sqrt(2) — measured at 1.426 on a year of BTC 1h data
#: against a theoretical 1.414, so the geometry behaves as predicted. A fixed ceiling
#: of 1.0 therefore encoded "closer than about the 35th percentile of random pairs",
#: which sounds selective and is really just an arbitrary point on a distribution
#: whose scale nobody checked.
#:
#: Deriving the ceiling from the observed dispersion instead makes it self-calibrating:
#: the same fraction means the same thing on a choppy altcoin as on BTC, without
#: retuning per asset.
_DISTANCE_FRACTION = 0.85

#: Sanity bound: however dispersed the data, a neighbour further than this is not an
#: analogue of anything.
_ABSOLUTE_MAX_DISTANCE = 1.5

#: Fewer analogues than this and the answer is "insufficient evidence".
_MIN_ANALOGUES = 20


@dataclass(frozen=True, slots=True)
class Analogue:
    """One historical moment judged similar to the query."""

    at: datetime
    distance: float
    forward_return_pct: float
    max_favourable_pct: float
    max_adverse_pct: float

    @property
    def rose(self) -> bool:
        return self.forward_return_pct > 0


@dataclass(slots=True)
class SimilarityResult:
    """What history says about situations resembling the query."""

    asset: str
    timeframe: Timeframe
    horizon_bars: int
    as_of: datetime
    analogues: list[Analogue] = field(default_factory=list)
    estimate: ProportionEstimate | None = None
    median_return_pct: float = 0.0
    mean_return_pct: float = 0.0
    best_case_pct: float = 0.0
    worst_case_pct: float = 0.0
    searched: int = 0
    rejected_as_dissimilar: int = 0
    #: The distance ceiling actually applied, for auditability.
    distance_ceiling: float = 0.0

    @property
    def has_evidence(self) -> bool:
        """Whether enough genuine analogues were found to say anything at all."""
        return len(self.analogues) >= _MIN_ANALOGUES and self.estimate is not None

    @property
    def is_informative(self) -> bool:
        """Whether the analogues actually differ from the market's default behaviour.

        This is an **uncorrected single-query test**, which is appropriate at
        inference time: one question is being asked about one moment. A caller
        sweeping many assets, horizons or timeframes at once is running a family of
        tests and must apply :func:`~mie.patterns.statistics.benjamini_hochberg`
        across the results — six queries at an uncorrected p < 0.05 will produce a
        spurious "finding" roughly a quarter of the time.
        """
        return (
            self.has_evidence
            and self.estimate is not None
            and self.estimate.p_value < 0.05
            and self.estimate.interval_excludes_baseline
        )

    def summary(self) -> str:
        if not self.has_evidence:
            return (
                f"{self.asset} {self.timeframe}: insufficient evidence — only "
                f"{len(self.analogues)} comparable historical situations found "
                f"(searched {self.searched}, {self.rejected_as_dissimilar} too dissimilar)"
            )
        assert self.estimate is not None
        verdict = "differs from baseline" if self.is_informative else "matches baseline"
        return (
            f"{self.asset} {self.timeframe} +{self.horizon_bars}: "
            f"{len(self.analogues)} analogues rose {self.estimate.rate:.0%} of the time "
            f"[{self.estimate.low:.0%}-{self.estimate.high:.0%}] vs baseline "
            f"{self.estimate.baseline:.0%}; median move {self.median_return_pct:+.2f}% "
            f"({verdict})"
        )


class SimilarityEngine:
    """Finds historical analogues of a market state and reports what followed."""

    def __init__(
        self,
        features: Sequence[str] = COMPARISON_FEATURES,
        distance_fraction: float = _DISTANCE_FRACTION,
        min_analogues: int = _MIN_ANALOGUES,
        top_k: int = 200,
        max_distance: float | None = None,
    ) -> None:
        self.features = tuple(features)
        self.distance_fraction = distance_fraction
        self.min_analogues = min_analogues
        self.top_k = top_k
        #: An explicit override pins the ceiling instead of calibrating it, which is
        #: what the tests use to exercise the guard deterministically.
        self.max_distance = max_distance

    def search(
        self,
        history: Sequence[tuple[datetime, Mapping[str, float]]],
        closes: Sequence[float],
        query_index: int,
        horizon: int,
        asset: str,
        timeframe: Timeframe,
    ) -> SimilarityResult:
        """Find analogues of ``history[query_index]`` among strictly earlier bars.

        ``history`` and ``closes`` are parallel and ordered oldest-first. Everything at
        or after ``query_index`` is invisible to the search.
        """
        result = SimilarityResult(
            asset=asset.upper(),
            timeframe=timeframe,
            horizon_bars=horizon,
            as_of=history[query_index][0] if query_index < len(history) else datetime.min,
        )
        # A neighbour needs its own forward window to have completed, and that window
        # must not reach into the query's present.
        latest_usable = query_index - horizon
        if latest_usable < self.min_analogues:
            return result

        query = history[query_index][1]
        past = history[:latest_usable]
        stats = self._normalisation(past)
        if not stats:
            return result

        query_vector = self._vector(query, stats)
        if query_vector is None:
            return result

        vectors: list[tuple[int, list[float]]] = []
        for index, (_, features) in enumerate(past):
            vector = self._vector(features, stats)
            if vector is not None:
                vectors.append((index, vector))

        ceiling = self.max_distance or _calibrate_ceiling(
            [v for _, v in vectors], self.distance_fraction
        )
        result.distance_ceiling = ceiling

        scored: list[tuple[float, int]] = []
        for index, vector in vectors:
            distance = _mean_distance(query_vector, vector)
            result.searched += 1
            if distance > ceiling:
                result.rejected_as_dissimilar += 1
                continue
            scored.append((distance, index))

        scored.sort()
        analogues: list[Analogue] = []
        for distance, index in scored[: self.top_k]:
            entry = closes[index]
            exit_close = closes[index + horizon]
            if entry <= 0:
                continue
            window = closes[index + 1 : index + horizon + 1]
            analogues.append(
                Analogue(
                    at=past[index][0],
                    distance=distance,
                    forward_return_pct=(exit_close - entry) / entry * 100.0,
                    max_favourable_pct=(max(window) - entry) / entry * 100.0,
                    max_adverse_pct=(min(window) - entry) / entry * 100.0,
                )
            )

        result.analogues = analogues
        if len(analogues) < self.min_analogues:
            return result

        returns = [a.forward_return_pct for a in analogues]
        result.median_return_pct = median(returns)
        result.mean_return_pct = sum(returns) / len(returns)
        result.best_case_pct = max(a.max_favourable_pct for a in analogues)
        result.worst_case_pct = min(a.max_adverse_pct for a in analogues)

        rose = sum(1 for a in analogues if a.rose)
        baseline_rose, baseline_total = _baseline_up_rate(closes[:latest_usable], horizon)
        result.estimate = compare_to_baseline(rose, len(analogues), baseline_rose, baseline_total)
        return result

    # ------------------------------------------------------------------ vectors

    def _normalisation(
        self, past: Sequence[tuple[datetime, Mapping[str, float]]]
    ) -> dict[str, tuple[float, float]]:
        """Per-feature centre and spread, computed from past data only.

        Median and MAD rather than mean and standard deviation: feature distributions
        here are skewed and occasionally spiky, and a single extreme bar would
        otherwise inflate the scale enough to make everything look similar.
        """
        stats: dict[str, tuple[float, float]] = {}
        for name in self.features:
            values = [f[name] for _, f in past if name in f]
            if len(values) < 30:
                continue
            centre = median(values)
            spread = median([abs(v - centre) for v in values]) * 1.4826
            if spread <= 0:
                continue
            stats[name] = (centre, spread)
        return stats

    def _vector(
        self, features: Mapping[str, float], stats: Mapping[str, tuple[float, float]]
    ) -> list[float] | None:
        """Standardise a feature vector, or None if too much of it is missing."""
        vector: list[float] = []
        for name in self.features:
            if name not in stats:
                continue
            value = features.get(name)
            if value is None:
                return None
            centre, spread = stats[name]
            vector.append((value - centre) / spread)
        return vector or None


def _calibrate_ceiling(vectors: Sequence[Sequence[float]], fraction: float) -> float:
    """Derive the similarity ceiling from the data's own dispersion.

    Samples pairs on a fixed stride rather than randomly, so the ceiling is fully
    reproducible: an analysis that returns different analogues on a re-run is not an
    analysis anyone can check.
    """
    if len(vectors) < 50:
        return _ABSOLUTE_MAX_DISTANCE
    stride = max(1, len(vectors) // 200)
    sample = vectors[::stride]
    distances = [
        _mean_distance(sample[i], sample[j])
        for i in range(0, len(sample) - 1, 2)
        for j in (i + 1,)
    ]
    if not distances:
        return _ABSOLUTE_MAX_DISTANCE
    typical = sum(distances) / len(distances)
    return min(_ABSOLUTE_MAX_DISTANCE, typical * fraction)


def _mean_distance(left: Sequence[float], right: Sequence[float]) -> float:
    """Root-mean-square distance per dimension.

    Averaging over dimensions rather than summing keeps the threshold interpretable
    and stable when a feature is unavailable and the vector is shorter.
    """
    if not left or len(left) != len(right):
        return math.inf
    total = sum((a - b) ** 2 for a, b in zip(left, right, strict=True))
    return math.sqrt(total / len(left))


def _baseline_up_rate(closes: Sequence[float], horizon: int) -> tuple[int, int]:
    """Unconditional rate of a positive forward return over non-overlapping windows."""
    rose = total = 0
    for index in range(0, len(closes) - horizon, horizon):
        entry = closes[index]
        if entry <= 0:
            continue
        total += 1
        rose += int(closes[index + horizon] > entry)
    return rose, total
