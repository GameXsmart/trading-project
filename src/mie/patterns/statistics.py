"""Statistics for pattern validation.

Small, dependency-free, and deliberately conservative. Everything here exists to
answer one question honestly: **is this pattern telling us anything the market was
not already doing?**

Three ideas do most of the work, and each guards against a specific way of fooling
yourself with backtests:

1. **Compare against the unconditional base rate, never against 50%.** Crypto drifts.
   If BTC closed higher on 54% of hours in the sample, a pattern that is "56%
   accurate" has an edge of two points, not six — and quite possibly none at all. A
   detector measured against a coin flip will look predictive purely because the
   market went up.

2. **Report an interval, never a point estimate.** A 70% hit rate on ten samples and
   on ten thousand are completely different claims. The Wilson interval is used rather
   than the textbook normal approximation because the latter misbehaves badly at small
   samples and near 0 or 1 — precisely where over-eager pattern claims live.

3. **Correct for multiple comparisons.** Testing twelve detectors across ten assets
   and four timeframes is 480 hypotheses; at p < 0.05 roughly 24 will look significant
   from noise alone. Benjamini-Hochberg controls the false discovery rate, so
   "significant" keeps meaning something after a wide sweep.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "ProportionEstimate",
    "benjamini_hochberg",
    "compare_to_baseline",
    "normal_cdf",
    "two_proportion_test",
    "wilson_interval",
]


@dataclass(frozen=True, slots=True)
class ProportionEstimate:
    """A measured proportion with its uncertainty and its edge over a baseline."""

    successes: int
    trials: int
    rate: float
    low: float
    high: float
    baseline: float
    edge: float
    p_value: float
    #: Set by :func:`benjamini_hochberg` once the whole family has been tested.
    significant: bool = False

    @property
    def interval_excludes_baseline(self) -> bool:
        """Whether the confidence interval clears the baseline entirely."""
        return self.low > self.baseline or self.high < self.baseline

    def summary(self) -> str:
        return (
            f"{self.rate:.1%} [{self.low:.1%}-{self.high:.1%}] "
            f"vs baseline {self.baseline:.1%} "
            f"(edge {self.edge:+.1%}, n={self.trials}, p={self.p_value:.4f})"
        )


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation ``p ± z·sqrt(p(1-p)/n)`` because that form
    produces impossible bounds outside [0, 1] and collapses to zero width at p = 0 or
    p = 1 — which is exactly where a rarely-firing pattern would otherwise appear to be
    a certainty. Three wins out of three is not a 100% hit rate, and Wilson says so.
    """
    if trials <= 0:
        return 0.0, 1.0
    p = successes / trials
    denominator = 1.0 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denominator
    margin = (
        z
        * math.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2))
        / denominator
    )
    return max(0.0, centre - margin), min(1.0, centre + margin)


def normal_cdf(x: float) -> float:
    """Standard normal CDF via the error function — no SciPy dependency needed."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def two_proportion_test(
    successes_a: int, trials_a: int, successes_b: int, trials_b: int
) -> float:
    """Two-sided p-value for the difference between two proportions.

    Pooled two-proportion z-test. Used to ask whether a pattern's hit rate differs
    from the unconditional rate over the *same* sample — which is the only comparison
    that isolates the pattern from the market's own drift.
    """
    if trials_a <= 0 or trials_b <= 0:
        return 1.0
    p_a = successes_a / trials_a
    p_b = successes_b / trials_b
    pooled = (successes_a + successes_b) / (trials_a + trials_b)
    if pooled in (0.0, 1.0):
        return 1.0  # no variance to speak of; nothing can be distinguished
    standard_error = math.sqrt(pooled * (1 - pooled) * (1 / trials_a + 1 / trials_b))
    if standard_error == 0:
        return 1.0
    z = (p_a - p_b) / standard_error
    return 2.0 * (1.0 - normal_cdf(abs(z)))


def compare_to_baseline(
    successes: int,
    trials: int,
    baseline_successes: int,
    baseline_trials: int,
    z: float = 1.96,
) -> ProportionEstimate:
    """Measure a pattern's hit rate against the unconditional rate.

    ``baseline_*`` describe what happened over the same window *without* conditioning
    on the pattern. This is the comparison that matters: a detector is only
    informative if it beats simply always guessing the market's prevailing direction.
    """
    rate = successes / trials if trials else 0.0
    baseline = baseline_successes / baseline_trials if baseline_trials else 0.5
    low, high = wilson_interval(successes, trials, z)
    p_value = two_proportion_test(successes, trials, baseline_successes, baseline_trials)
    return ProportionEstimate(
        successes=successes,
        trials=trials,
        rate=rate,
        low=low,
        high=high,
        baseline=baseline,
        edge=rate - baseline,
        p_value=p_value,
    )


def benjamini_hochberg(
    p_values: list[float], false_discovery_rate: float = 0.05
) -> list[bool]:
    """Benjamini-Hochberg step-up procedure. Returns a rejection mask.

    Controls the expected proportion of false positives among the results declared
    significant. Bonferroni would also control error but is far too strict for a
    screening sweep of this size — it would reject genuine weak edges along with the
    noise, and weak edges are the only kind this domain realistically offers.

    Testing 480 detector/asset/timeframe combinations at an uncorrected p < 0.05 would
    yield roughly 24 "significant" results from pure noise. That is not a hypothetical
    concern: it is the single most common way pattern research produces confident
    nonsense.
    """
    n = len(p_values)
    if n == 0:
        return []
    ordered = sorted(range(n), key=lambda i: p_values[i])
    rejected = [False] * n
    largest_k = -1
    for rank, index in enumerate(ordered, start=1):
        if p_values[index] <= false_discovery_rate * rank / n:
            largest_k = rank
    # Step-up: everything at or below the largest passing rank is rejected too.
    for rank, index in enumerate(ordered, start=1):
        if rank <= largest_k:
            rejected[index] = True
    return rejected
