"""The alert engine: evaluate, budget, deliver, and account for what was held.

The order is deliberate and the accounting is the interesting part.

Rules produce candidates. The budget decides which of them a human actually sees. The
channels deliver those. And then — the step that is usually missing — the engine
reports how many were held back and why, so that silence can be read correctly.

That last point is the whole design. An alerting system's failure mode is not missing
an event; it is producing enough that the reader stops looking. Every mechanism here
exists to spend a limited attention budget well, and the digest exists so the reader
knows the budget is being spent at all.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

from mie.alerts.budget import Decision, RateBudget
from mie.alerts.channels import Channel, ConsoleChannel, DeliveryResult
from mie.alerts.rules import DEFAULT_RULES, AlertContext, Rule
from mie.alerts.types import Alert
from mie.core.logging import get_logger
from mie.core.timeframes import utcnow

log = get_logger(__name__)

__all__ = ["AlertEngine", "AlertRun"]


@dataclass(slots=True)
class AlertRun:
    """What one evaluation produced, including what it chose not to say."""

    at: datetime = field(default_factory=utcnow)
    raised: list[Alert] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    deliveries: dict[str, list[DeliveryResult]] = field(default_factory=dict)
    #: Alerts a rule produced but that failed their own validity requirements.
    rejected: list[str] = field(default_factory=list)

    @property
    def delivered(self) -> list[Alert]:
        return [d.alert for d in self.decisions if d.delivered]

    @property
    def held(self) -> list[Decision]:
        return [d for d in self.decisions if not d.delivered]

    @property
    def delivered_count(self) -> int:
        return len(self.delivered)

    def failures(self) -> list[DeliveryResult]:
        return [
            result
            for results in self.deliveries.values()
            for result in results
            if not result.delivered and result.error not in ("below channel minimum",)
        ]

    def summary(self) -> str:
        return (
            f"{len(self.raised)} raised, {self.delivered_count} delivered, "
            f"{len(self.held)} held"
            + (f", {len(self.rejected)} rejected as invalid" if self.rejected else "")
        )


class AlertEngine:
    """Runs the rules, spends the budget, and keeps the books."""

    def __init__(
        self,
        rules: Sequence[Rule] | None = None,
        budget: RateBudget | None = None,
        channels: Sequence[Channel] | None = None,
    ) -> None:
        self.rules = list(rules if rules is not None else DEFAULT_RULES)
        self.budget = budget or RateBudget()
        self.channels = list(channels if channels is not None else [ConsoleChannel()])

    # ------------------------------------------------------------------ evaluate

    def raise_alerts(self, context: AlertContext) -> tuple[list[Alert], list[str]]:
        """Run every rule over one context.

        A rule that raises is logged and skipped rather than aborting the run: one
        broken rule must not silence the other ten, and a monitoring system that fails
        closed is a monitoring system that fails silently.
        """
        raised: list[Alert] = []
        rejected: list[str] = []
        for rule in self.rules:
            try:
                raised.extend(rule.evaluate(context))
            except ValueError as exc:
                # A directional alert missing its confidence or invalidation. The type
                # refused to construct it, which is the guard working.
                rejected.append(f"{rule.name}: {exc}")
                log.warning("alert_rejected", rule=rule.name, error=str(exc)[:200])
            except Exception as exc:  # pragma: no cover - defensive
                rejected.append(f"{rule.name}: {exc}")
                log.error("alert_rule_failed", rule=rule.name, error=str(exc)[:200])
        return raised, rejected

    async def run(
        self, contexts: Sequence[AlertContext], now: datetime | None = None
    ) -> AlertRun:
        """Evaluate, budget and deliver for a batch of contexts."""
        moment = now or utcnow()
        result = AlertRun(at=moment)

        for context in contexts:
            raised, rejected = self.raise_alerts(context)
            result.raised.extend(raised)
            result.rejected.extend(rejected)

        result.decisions = self.budget.admit_all(result.raised, moment)

        digest = self.budget.pending_digest(moment)
        if digest is not None:
            # Admitted through the same path so it is recorded, but exempt from the
            # budget: a suppression notice that can be suppressed fails exactly when
            # it is needed.
            result.decisions.append(self.budget.admit(digest, moment))

        for decision in result.decisions:
            if decision.delivered:
                result.deliveries[decision.alert.dedup_key] = await self._deliver(
                    decision.alert
                )

        log.info(
            "alert_run",
            raised=len(result.raised),
            delivered=result.delivered_count,
            held=len(result.held),
            capacity=self.budget.capacity_remaining(moment),
        )
        return result

    # ------------------------------------------------------------------ delivery

    async def _deliver(self, alert: Alert) -> list[DeliveryResult]:
        """Send to every enabled channel, concurrently, tolerating failures."""
        enabled = [channel for channel in self.channels if channel.enabled]
        if not enabled:
            return []
        results = await asyncio.gather(
            *(self._send_one(channel, alert) for channel in enabled),
            return_exceptions=False,
        )
        for result in results:
            if not result.delivered and result.error not in ("below channel minimum",):
                log.warning(
                    "alert_delivery_failed", channel=result.channel, error=result.error
                )
        return list(results)

    @staticmethod
    async def _send_one(channel: Channel, alert: Alert) -> DeliveryResult:
        try:
            return await channel.send(alert)
        except Exception as exc:  # pragma: no cover - defensive
            return DeliveryResult(channel.name, False, str(exc)[:200])

    # ---------------------------------------------------------------- inspection

    def capacity(self, now: datetime | None = None) -> dict[str, int]:
        return self.budget.capacity_remaining(now)

    def report(self, run: AlertRun) -> str:
        lines = [f"Alert run @ {run.at:%Y-%m-%d %H:%M}", "=" * 78]
        for alert in run.delivered:
            lines.append(f"  [{alert.level.label:9}] {alert.asset:6} {alert.title}")
        if run.held:
            lines.append("")
            counts: dict[str, int] = {}
            for decision in run.held:
                counts[decision.reason] = counts.get(decision.reason, 0) + 1
            for reason, count in sorted(counts.items(), key=lambda kv: -kv[1]):
                lines.append(f"  held {count}: {reason}")
        if run.rejected:
            lines.append("")
            lines.extend(f"  rejected: {reason}" for reason in run.rejected[:5])
        lines.append("")
        lines.append(run.summary())
        return "\n".join(lines)
