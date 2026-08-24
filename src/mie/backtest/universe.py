"""Survivorship: backtesting the universe as it was, not as it survived.

The failure is easy to describe and easy to commit. You backtest across "the top ten
coins", take today's list, and pull their history. Every asset in that list is one that
*still exists* — which means the sample has been filtered by an outcome that was not
knowable at the time, and the filter is correlated with exactly what is being measured.
Assets that collapsed are absent, so measured returns are too high, measured volatility
too low, and any strategy tested this way inherits a tailwind that did not exist.

Crypto makes this worse than equities do. Delisting is common, fast, and frequently
total: a token can go from a top-fifty listing to no liquid venue inside a quarter, and
the survivors are not a random sample of what was there.

The defence is to select the universe *as of* each backtest date. This module makes
that the only convenient way to do it: :meth:`HistoricalUniverse.active_at` answers
what was tradeable at a moment, and :meth:`survivorship_gap` reports how much a
present-day list would have differed.

**Current state, stated plainly: no delistings are recorded.** All ten configured
assets have been continuously listed for the whole stored history, so on this data the
as-of universe and the survivor universe are identical and the correction changes
nothing. That is a fact about a small, young, deliberately liquid universe — not
evidence that survivorship bias is unimportant. The mechanism exists so that the first
delisting is handled correctly rather than discovered afterwards, and
:meth:`survivorship_gap` will report a non-zero number the moment one is recorded.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime

from mie.core.timeframes import utcnow

__all__ = ["AssetListing", "HistoricalUniverse", "SurvivorshipGap"]


@dataclass(frozen=True, slots=True)
class AssetListing:
    """When an asset was tradeable, and when it stopped being so."""

    symbol: str
    listed_at: datetime | None = None
    delisted_at: datetime | None = None
    #: Why it left, if it did. Recorded because "delisted from one venue" and "the
    #: project failed" are different facts with different implications for a backtest.
    reason: str = ""

    def active_at(self, moment: datetime) -> bool:
        if self.listed_at is not None and moment < self.listed_at:
            return False
        return not (self.delisted_at is not None and moment >= self.delisted_at)

    @property
    def survives(self) -> bool:
        return self.delisted_at is None

    def __str__(self) -> str:  # pragma: no cover
        window = f"{self.listed_at:%Y-%m-%d}" if self.listed_at else "?"
        window += f" .. {self.delisted_at:%Y-%m-%d}" if self.delisted_at else " .. present"
        return f"{self.symbol} ({window})" + (f" - {self.reason}" if self.reason else "")


@dataclass(frozen=True, slots=True)
class SurvivorshipGap:
    """How much a present-day universe differs from the historical one."""

    moment: datetime
    as_of_universe: tuple[str, ...]
    survivor_universe: tuple[str, ...]

    @property
    def missing(self) -> tuple[str, ...]:
        """Assets that existed then but would be absent from a survivor-based test."""
        return tuple(sorted(set(self.as_of_universe) - set(self.survivor_universe)))

    @property
    def spurious(self) -> tuple[str, ...]:
        """Assets a survivor-based test would include that did not yet exist."""
        return tuple(sorted(set(self.survivor_universe) - set(self.as_of_universe)))

    @property
    def is_clean(self) -> bool:
        return not self.missing and not self.spurious

    @property
    def bias_fraction(self) -> float:
        """Share of the as-of universe that a survivor-based test would silently drop."""
        if not self.as_of_universe:
            return 0.0
        return round(len(self.missing) / len(self.as_of_universe), 4)

    def summary(self) -> str:
        if self.is_clean:
            return (
                f"{self.moment:%Y-%m-%d}: no survivorship gap "
                f"({len(self.as_of_universe)} assets, none delisted)"
            )
        return (
            f"{self.moment:%Y-%m-%d}: {len(self.missing)} of "
            f"{len(self.as_of_universe)} assets ({self.bias_fraction:.0%}) would be "
            f"dropped by a survivor-based universe: {', '.join(self.missing)}"
        )


@dataclass(slots=True)
class HistoricalUniverse:
    """The observation universe, with its listing history."""

    listings: list[AssetListing] = field(default_factory=list)

    @classmethod
    def from_symbols(cls, symbols: Iterable[str]) -> HistoricalUniverse:
        """Build from a plain symbol list, assuming continuous listing.

        The assumption is explicit here rather than implied by the absence of the
        question. Anything known to have been delisted must be added as an
        :class:`AssetListing` with a `delisted_at`, or a backtest over this universe
        will carry survivorship bias regardless of what the rest of this module does.
        """
        return cls([AssetListing(symbol=s.upper()) for s in symbols])

    def add(self, listing: AssetListing) -> None:
        self.listings = [entry for entry in self.listings if entry.symbol != listing.symbol]
        self.listings.append(listing)

    def active_at(self, moment: datetime) -> tuple[str, ...]:
        """Symbols tradeable at ``moment`` — the correct universe for a backtest then."""
        return tuple(sorted(e.symbol for e in self.listings if e.active_at(moment)))

    def survivors(self) -> tuple[str, ...]:
        """Symbols still listed today — the universe a naive backtest would use."""
        return tuple(sorted(e.symbol for e in self.listings if e.survives))

    def delisted(self) -> tuple[AssetListing, ...]:
        return tuple(e for e in self.listings if not e.survives)

    def survivorship_gap(self, moment: datetime | None = None) -> SurvivorshipGap:
        """Quantify what a survivor-based universe would get wrong at ``moment``."""
        when = moment or utcnow()
        return SurvivorshipGap(
            moment=when,
            as_of_universe=self.active_at(when),
            survivor_universe=self.survivors(),
        )

    def audit(self, moments: Sequence[datetime]) -> list[SurvivorshipGap]:
        """Gaps across a series of moments — one per fold, in practice."""
        return [self.survivorship_gap(moment) for moment in moments]

    def report(self) -> str:  # pragma: no cover - display affordance
        lines = [f"Universe: {len(self.listings)} assets"]
        delisted = self.delisted()
        if not delisted:
            lines.append("  no delistings recorded - as-of and survivor universes coincide")
        else:
            lines.append(f"  {len(delisted)} delisted:")
            lines.extend(f"    {entry}" for entry in delisted)
        return "\n".join(lines)
