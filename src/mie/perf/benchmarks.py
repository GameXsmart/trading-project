"""Latency budgets, and measurement against them.

Phase 12's gate is a performance claim, and performance claims rot faster than any
other kind: they are true when written, and nobody notices when they stop being true
because nothing fails. So the budgets are declared here as data, measured against real
stored data, and reported pass or fail — the same shape as every other gate in this
repository.

Two decisions worth stating.

**Budgets are per operation, not a single global number.** "The system is fast enough"
cannot be acted on. "Building one prediction context takes 4ms against a budget of
25ms, and answering one API prediction takes 380ms against a budget of 1500ms" tells
you which one to look at when it changes.

**Budgets are set from what the operation is for, not from what it currently does.**
Setting a budget to the current measurement guarantees a pass and measures nothing. A
live poller has a bar boundary to hit; an interactive request has a person waiting; a
backtest has an analyst's patience. Those are the numbers below, and some of them have
a lot of headroom on purpose — headroom is what a budget is for.

What is *not* here is as deliberate. There is no Redis, no NATS, and no second
language runtime, because the profiling that justified the two optimisations this phase
did make did not indict anything those would fix. §12's own constraint is that a second
toolchain must be earned; the same standard applies to a second datastore.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from mie.core.logging import get_logger

log = get_logger(__name__)

__all__ = ["LATENCY_BUDGETS", "Benchmark", "BenchmarkReport", "measure"]


@dataclass(frozen=True, slots=True)
class Benchmark:
    """One measured operation against its declared budget."""

    name: str
    #: What the operation is allowed to take, in milliseconds.
    budget_ms: float
    #: Median across samples. Median rather than mean: one scheduler hiccup should not
    #: decide whether a budget is met.
    median_ms: float
    #: The worst sample, reported because a p50 that passes while p100 is ten times
    #: worse is a system with a stutter, and a stutter is what a user notices.
    worst_ms: float
    samples: int
    #: Scale the measurement describes — assets, bars, series — so a number can be
    #: compared against a different one later.
    scale: str = ""
    note: str = ""

    @property
    def within_budget(self) -> bool:
        return self.median_ms <= self.budget_ms

    @property
    def headroom(self) -> float:
        """How many times over the measurement could grow before breaching."""
        if self.median_ms <= 0:
            return float("inf")
        return round(self.budget_ms / self.median_ms, 1)

    def summary(self) -> str:
        verdict = "ok" if self.within_budget else "OVER"
        return (
            f"{self.name:38} {self.median_ms:9.2f}ms  "
            f"budget {self.budget_ms:8.0f}ms  worst {self.worst_ms:9.2f}ms  "
            f"{self.headroom:6.1f}x  {verdict}"
            + (f"  [{self.scale}]" if self.scale else "")
        )


@dataclass(slots=True)
class BenchmarkReport:
    """A set of measurements and whether the whole budget was met."""

    benchmarks: list[Benchmark] = field(default_factory=list)
    #: Data-quality events recorded during the run, for the "no increase versus the
    #: polling baseline" half of the gate.
    quality_events: int = 0
    baseline_quality_events: int = 0

    @property
    def passed(self) -> bool:
        return all(b.within_budget for b in self.benchmarks) and not self.quality_regressed

    @property
    def quality_regressed(self) -> bool:
        return self.quality_events > self.baseline_quality_events

    def over_budget(self) -> list[Benchmark]:
        return [b for b in self.benchmarks if not b.within_budget]

    def report(self) -> str:
        lines = ["Latency budgets", "=" * 96]
        lines.extend("  " + b.summary() for b in self.benchmarks)
        lines.append("")
        breached = self.over_budget()
        lines.append(
            f"{len(self.benchmarks) - len(breached)} of {len(self.benchmarks)} "
            f"within budget"
            + (f"; OVER: {', '.join(b.name for b in breached)}" if breached else "")
        )
        lines.append(
            f"data-quality events: {self.quality_events} "
            f"(baseline {self.baseline_quality_events})"
            + ("  REGRESSED" if self.quality_regressed else "")
        )
        return "\n".join(lines)


def measure(
    name: str,
    operation: Callable[[], object],
    budget_ms: float,
    samples: int = 5,
    warmup: int = 1,
    scale: str = "",
    note: str = "",
) -> Benchmark:
    """Time an operation and score it against its budget.

    A warmup run is discarded. The first call to almost anything here pays for an
    import, a connection or a cold page cache, and including it would measure startup
    rather than the operation — which is exactly the number that would then be
    optimised, uselessly.
    """
    for _ in range(max(0, warmup)):
        operation()

    timings: list[float] = []
    for _ in range(max(1, samples)):
        start = time.perf_counter()
        operation()
        timings.append((time.perf_counter() - start) * 1000.0)

    result = Benchmark(
        name=name,
        budget_ms=budget_ms,
        median_ms=round(statistics.median(timings), 3),
        worst_ms=round(max(timings), 3),
        samples=len(timings),
        scale=scale,
        note=note,
    )
    log.info(
        "benchmark",
        name=name,
        median_ms=result.median_ms,
        budget_ms=budget_ms,
        within=result.within_budget,
    )
    return result


async def measure_async(
    name: str,
    operation: Callable[[], object],
    budget_ms: float,
    samples: int = 5,
    warmup: int = 1,
    scale: str = "",
    note: str = "",
) -> Benchmark:
    """As :func:`measure`, for a coroutine factory."""
    for _ in range(max(0, warmup)):
        await operation()  # type: ignore[misc]

    timings: list[float] = []
    for _ in range(max(1, samples)):
        start = time.perf_counter()
        await operation()  # type: ignore[misc]
        timings.append((time.perf_counter() - start) * 1000.0)

    return Benchmark(
        name=name,
        budget_ms=budget_ms,
        median_ms=round(statistics.median(timings), 3),
        worst_ms=round(max(timings), 3),
        samples=len(timings),
        scale=scale,
        note=note,
    )


#: The declared budgets. Each is justified by what the operation is for.
LATENCY_BUDGETS: dict[str, tuple[float, str]] = {
    # A live poller must finish a full sweep of the universe well inside the shortest
    # bar it collects. One minute is the shortest timeframe, so a sweep has to fit in
    # a fraction of that or it falls behind permanently rather than transiently.
    "poll sweep (all assets, 1 timeframe)": (
        15_000,
        "a 1m bar closes every 60s; a sweep that takes longer never catches up",
    ),
    # A person is waiting. Above about a second an interface stops feeling live.
    "api: latest prediction": (
        1_500,
        "an interactive request with someone watching",
    ),
    "api: asset grid": (2_000, "the dashboard's first paint"),
    "api: correlation matrix": (3_000, "a whole-universe cross-product"),
    # Nothing is waiting, but an analyst's patience is finite and a backtest that takes
    # minutes per configuration is a backtest that gets run once.
    "build one prediction context": (25, "called hundreds of times per evaluation"),
    "walk-forward contexts (1 asset)": (5_000, "the unit of a backtest"),
    "features: one bar": (5, "computed per bar per series, incrementally"),
    "query: recent bars": (250, "the read behind almost everything"),
}
