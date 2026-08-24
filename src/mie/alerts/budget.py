"""The rate budget: the part that decides whether anyone still reads this.

Phase 11's gate is not about detection. It is: *alert volume under a simulated volatile
week stays within a rate budget — an alerting system nobody reads is worse than none.*
That is the right criterion, because the failure mode of alerting is not missing an
event, it is producing so many that the recipient stops looking, at which point the
system has negative value: it consumed attention and then trained the reader to ignore
the one alert that mattered.

Four mechanisms, applied in order, each answering a different question.

1. **Deduplication** — "have I already said exactly this?" A rule re-evaluated on the
   next tick usually produces the identical alert, and a system that treats each
   re-derivation as a new event is a clock, not a monitor.
2. **Cooldown** — "have I said something of this kind about this asset recently?" The
   second volume anomaly on BTC within the hour is not twice the information.
3. **Budget** — "have I already spent this hour's attention?" A hard ceiling on volume
   per hour and per day, regardless of how much is happening.
4. **Reserve** — critical alerts draw on a separate allowance, so a noisy hour of
   market chatter cannot crowd out the message that the data feed has collapsed.

And one property that matters more than any of them: **suppression is never silent.**
A reader who is told fourteen things while forty were suppressed, and does not know
about the forty, believes they are seeing everything. So suppressed alerts are counted
by reason and surfaced as a digest, and the digest is exempt from the budget — a
suppression notice that can itself be suppressed is worse than useless, because it
fails exactly when it is most needed.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from mie.alerts.types import Alert, AlertKind, Severity
from mie.core.logging import get_logger
from mie.core.timeframes import utcnow

log = get_logger(__name__)

__all__ = ["Decision", "RateBudget", "Suppression"]


class Suppression(str):
    """Why an alert was not delivered. A string subclass so it renders directly."""

    DUPLICATE = "duplicate of a recent alert"
    COOLDOWN = "same kind and asset alerted recently"
    HOURLY = "hourly budget exhausted"
    DAILY = "daily budget exhausted"


@dataclass(frozen=True, slots=True)
class Decision:
    """Whether an alert goes out, and why not if it does not."""

    alert: Alert
    delivered: bool
    reason: str = ""

    def __str__(self) -> str:  # pragma: no cover
        return f"{'sent' if self.delivered else 'held'}: {self.alert.title}" + (
            f" ({self.reason})" if self.reason else ""
        )


@dataclass(slots=True)
class RateBudget:
    """A sliding-window budget with cooldowns, dedup and a critical reserve.

    Defaults are chosen for a human reading on a phone, not for a dashboard: roughly
    six an hour and thirty a day is already at the edge of what someone will keep
    paying attention to across a volatile week.
    """

    per_hour: int = 6
    per_day: int = 30
    #: Minimum gap between alerts of the same kind about the same asset.
    cooldown: timedelta = timedelta(hours=4)
    #: Window in which an identical alert is treated as already said.
    dedup_window: timedelta = timedelta(hours=12)
    #: Critical alerts beyond the ordinary budget, per hour and per day. Small on
    #: purpose: a reserve large enough to be comfortable is a second budget.
    critical_reserve_per_hour: int = 2
    critical_reserve_per_day: int = 8
    #: How often a suppression digest is emitted, at most.
    digest_every: timedelta = timedelta(hours=6)

    _delivered: deque[tuple[datetime, Severity]] = field(default_factory=deque, init=False)
    _last_by_scope: dict[tuple[str, str], datetime] = field(default_factory=dict, init=False)
    _last_by_key: dict[str, datetime] = field(default_factory=dict, init=False)
    _suppressed: Counter[str] = field(default_factory=Counter, init=False)
    _suppressed_kinds: Counter[str] = field(default_factory=Counter, init=False)
    _last_digest: datetime | None = field(default=None, init=False)

    # ------------------------------------------------------------------ admission

    def admit(self, alert: Alert, now: datetime | None = None) -> Decision:
        """Decide whether one alert is delivered."""
        moment = now or alert.at or utcnow()

        # A digest reports on suppression, so it must never be subject to it.
        if alert.is_digest:
            self._record(moment, alert.level)
            return Decision(alert, True)

        last_identical = self._last_by_key.get(alert.dedup_key)
        if last_identical and moment - last_identical < self.dedup_window:
            return self._hold(alert, Suppression.DUPLICATE)

        last_scope = self._last_by_scope.get(alert.scope)
        if last_scope and moment - last_scope < self.cooldown:
            return self._hold(alert, Suppression.COOLDOWN)

        hour_count = self._count_since(moment - timedelta(hours=1))
        if hour_count >= self.per_hour and not self._has_reserve(
            alert.level, moment, timedelta(hours=1), self.critical_reserve_per_hour
        ):
            return self._hold(alert, Suppression.HOURLY)

        day_count = self._count_since(moment - timedelta(days=1))
        if day_count >= self.per_day and not self._has_reserve(
            alert.level, moment, timedelta(days=1), self.critical_reserve_per_day
        ):
            return self._hold(alert, Suppression.DAILY)

        self._record(moment, alert.level)
        self._last_by_scope[alert.scope] = moment
        self._last_by_key[alert.dedup_key] = moment
        return Decision(alert, True)

    def admit_all(
        self, alerts: Iterable[Alert], now: datetime | None = None
    ) -> list[Decision]:
        """Admit a batch, most severe first.

        Ordering matters when the budget is nearly exhausted: processing in arrival
        order would let three routine notices consume the last of the hour and leave a
        critical one held behind them.
        """
        ordered = sorted(
            alerts, key=lambda a: (-a.level, a.at)
        )
        return [self.admit(alert, now) for alert in ordered]

    # -------------------------------------------------------------------- digest

    def pending_digest(self, now: datetime | None = None) -> Alert | None:
        """A summary of what was suppressed, if anything, and if it is time.

        Emitted as an ordinary alert so it travels through the same channels. Without
        this, the budget would be indistinguishable from a quiet market — and a reader
        who cannot tell those apart will draw the wrong conclusion from silence.
        """
        moment = now or utcnow()
        if not self._suppressed:
            return None
        if self._last_digest and moment - self._last_digest < self.digest_every:
            return None

        total = sum(self._suppressed.values())
        by_reason = ", ".join(
            f"{count} {reason}" for reason, count in self._suppressed.most_common()
        )
        by_kind = ", ".join(
            f"{kind} x{count}" for kind, count in self._suppressed_kinds.most_common(5)
        )
        self._last_digest = moment
        self._suppressed = Counter()
        self._suppressed_kinds = Counter()
        return Alert(
            kind=AlertKind.SUPPRESSION_DIGEST,
            asset="SYSTEM",
            title=f"{total} alerts suppressed",
            detail=(
                f"Held back to stay within the rate budget: {by_reason}."
                + (f" Most held: {by_kind}." if by_kind else "")
                + " Silence here means budgeted, not quiet."
            ),
            severity=Severity.INFO,
            at=moment,
            is_digest=True,
        )

    # ------------------------------------------------------------------ inspection

    @property
    def suppressed_total(self) -> int:
        return sum(self._suppressed.values())

    def delivered_since(self, moment: datetime) -> int:
        return self._count_since(moment)

    def capacity_remaining(self, now: datetime | None = None) -> dict[str, int]:
        moment = now or utcnow()
        return {
            "hour": max(0, self.per_hour - self._count_since(moment - timedelta(hours=1))),
            "day": max(0, self.per_day - self._count_since(moment - timedelta(days=1))),
        }

    # ------------------------------------------------------------------ internals

    def _hold(self, alert: Alert, reason: str) -> Decision:
        self._suppressed[reason] += 1
        self._suppressed_kinds[alert.kind.value] += 1
        return Decision(alert, False, reason)

    def _record(self, moment: datetime, severity: Severity) -> None:
        self._delivered.append((moment, severity))
        cutoff = moment - timedelta(days=1)
        while self._delivered and self._delivered[0][0] < cutoff:
            self._delivered.popleft()

    def _count_since(self, moment: datetime) -> int:
        return sum(1 for when, _ in self._delivered if when >= moment)

    def _has_reserve(
        self, severity: Severity, now: datetime, window: timedelta, allowance: int
    ) -> bool:
        """Whether a critical alert may draw on the reserve.

        Only ``CRITICAL`` qualifies. Letting ``IMPORTANT`` in would make the reserve a
        second budget, and the whole point is that it stays empty until something
        genuinely breaks.
        """
        if severity < Severity.CRITICAL:
            return False
        used = sum(
            1
            for when, level in self._delivered
            if when >= now - window and level >= Severity.CRITICAL
        )
        return used < allowance
