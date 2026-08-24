"""Fold construction: which bars a fit may see, and which it is judged on.

Three ideas, in order of how often they are got wrong.

**No random splits.** Shuffling a price series into train and test lets the future
inform the past, and the resulting metrics are fiction. Folds here are strictly
chronological: every test window lies entirely after the training window that produced
the model being tested.

**Purging.** Chronological ordering alone is not enough. A training point at bar *t*
is labelled by what happened at *t + horizon*, so a training point close to the
boundary carries information from *inside* the test window. The fix is to drop the
last `horizon` bars of every training window — they are the ones whose labels reach
across. Omitting this is the most common way a walk-forward backtest quietly leaks,
precisely because the split *looks* clean.

**Embargo.** Serial correlation runs the other way too: the first test points sit close
enough to the training data that they are not really independent of it. A small gap
after the purge boundary buys that independence back. The embargo is smaller than the
purge because it addresses correlation rather than outright label overlap.

Every window records its exact bar range and timestamps, so what a fit was allowed to
see is auditable after the fact rather than inferred from the code that produced it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from mie.core.types import Candle

__all__ = ["DataWindow", "Fold", "FoldScheme", "generate_folds"]

_EPOCH = datetime.fromtimestamp(0, tz=UTC)


class FoldScheme(StrEnum):
    """How the training window moves between folds."""

    #: Training window grows; every fold sees all history before it. Matches how a live
    #: system accumulates data.
    EXPANDING = "expanding"
    #: Training window is a fixed length that slides forward. Tests whether skill
    #: depends on a specific era rather than on recent history in general.
    ROLLING = "rolling"


@dataclass(frozen=True, slots=True)
class DataWindow:
    """An exact, auditable range of bars."""

    start_index: int
    end_index: int
    start_time: datetime
    end_time: datetime

    @property
    def bars(self) -> int:
        return max(0, self.end_index - self.start_index)

    @property
    def is_empty(self) -> bool:
        return self.bars <= 0

    def contains_index(self, index: int) -> bool:
        return self.start_index <= index < self.end_index

    def overlaps(self, other: DataWindow) -> bool:
        return self.start_index < other.end_index and other.start_index < self.end_index

    def label(self) -> str:
        return (
            f"[{self.start_index}:{self.end_index}) "
            f"{self.start_time:%Y-%m-%d}..{self.end_time:%Y-%m-%d} ({self.bars} bars)"
        )

    def __str__(self) -> str:  # pragma: no cover
        return self.label()

    @classmethod
    def of(cls, candles: Sequence[Candle], start: int, end: int) -> DataWindow:
        start = max(0, min(start, len(candles)))
        end = max(start, min(end, len(candles)))
        if start >= len(candles) or end <= start:
            moment = candles[min(start, len(candles) - 1)].open_time if candles else _EPOCH
            return cls(start, start, moment, moment)
        return cls(
            start_index=start,
            end_index=end,
            start_time=candles[start].open_time,
            end_time=candles[end - 1].close_time,
        )


@dataclass(frozen=True, slots=True)
class Fold:
    """One train/test split, with the gap between them recorded explicitly."""

    index: int
    train: DataWindow
    test: DataWindow
    #: Bars dropped from the end of the training window because their labels reach
    #: into the test window.
    purge_bars: int
    #: Bars skipped after the purge, to break serial correlation across the boundary.
    embargo_bars: int
    scheme: FoldScheme

    @property
    def is_usable(self) -> bool:
        return not self.train.is_empty and not self.test.is_empty

    @property
    def gap_bars(self) -> int:
        return self.test.start_index - self.train.end_index

    def leaks(self) -> bool:
        """Whether this fold's own construction is unsound.

        Checked rather than assumed: the gap must be at least the purge, and the two
        windows must not overlap. A fold that fails this invalidates every number
        computed from it, so the harness refuses to run it.
        """
        return self.train.overlaps(self.test) or self.gap_bars < self.purge_bars

    def summary(self) -> str:
        return (
            f"fold {self.index}: train {self.train.label()} "
            f"-> gap {self.gap_bars} -> test {self.test.label()}"
        )


def generate_folds(
    candles: Sequence[Candle],
    horizon_bars: int,
    folds: int = 5,
    warmup_bars: int = 400,
    scheme: FoldScheme = FoldScheme.EXPANDING,
    embargo_bars: int | None = None,
    min_test_bars: int = 50,
    min_train_bars: int | None = None,
) -> list[Fold]:
    """Split history into chronological folds with purge and embargo gaps.

    ``horizon_bars`` sets the purge directly: a label reaching *h* bars forward
    contaminates exactly the last *h* training bars, so that is how many are dropped.
    Deriving it rather than configuring it removes the chance of the two drifting
    apart when someone changes the horizon.

    ``warmup_bars`` and ``min_train_bars`` are separate quantities that are easy to
    conflate. The warmup is history a model needs before it can say anything at all;
    the minimum training span is how much *predictable* range the first fold must have
    on top of that. Setting the first test window to begin at the end of the warmup
    would give fold zero a training window in which no model can produce a prediction,
    and a calibration fitted on nothing but abstentions.
    """
    total = len(candles)
    purge = max(0, horizon_bars)
    embargo = max(0, horizon_bars // 4 if embargo_bars is None else embargo_bars)
    gap = purge + embargo
    minimum_train = warmup_bars if min_train_bars is None else max(0, min_train_bars)
    offset = warmup_bars + minimum_train

    usable = total - offset - gap
    if folds < 1 or usable < folds * min_test_bars:
        return []

    test_bars = usable // folds
    if test_bars < min_test_bars:
        return []

    produced: list[Fold] = []
    for index in range(folds):
        test_start = offset + gap + index * test_bars
        test_end = test_start + test_bars if index < folds - 1 else total
        train_end = test_start - gap
        train_start = 0 if scheme is FoldScheme.EXPANDING else max(0, train_end - test_bars * 2)

        fold = Fold(
            index=index,
            train=DataWindow.of(candles, train_start, train_end),
            test=DataWindow.of(candles, test_start, test_end),
            purge_bars=purge,
            embargo_bars=embargo,
            scheme=scheme,
        )
        if fold.is_usable and not fold.leaks():
            produced.append(fold)
    return produced
