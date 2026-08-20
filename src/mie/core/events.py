"""Event bus.

The seam between ingestion and every analytical phase that follows. Phase 1
publishes `candle.closed`; Phase 2's feature engine will subscribe to it without
ingestion knowing anything about features.

The in-process implementation is deliberate: a single deployable with zero
infrastructure is the right shape until throughput forces a split. `EventBus` is an
interface precisely so that swapping in NATS or Redpanda later is a config change
rather than a rewrite.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from mie.core.logging import get_logger
from mie.core.timeframes import utcnow
from mie.core.types import Candle, QualityEvent

log = get_logger(__name__)

__all__ = ["Event", "EventBus", "InProcessEventBus", "Topics", "candle_closed", "quality_event"]

Handler = Callable[["Event"], Awaitable[None]]


class Topics:
    CANDLE_CLOSED = "candle.closed"
    CANDLE_UPDATED = "candle.updated"
    QUALITY_EVENT = "data.quality"
    INGEST_COMPLETED = "ingest.completed"
    PROVIDER_STATE = "provider.state"


@dataclass(slots=True)
class Event:
    topic: str
    payload: Any
    ts: Any = field(default_factory=utcnow)
    meta: dict[str, Any] = field(default_factory=dict)


class EventBus(ABC):
    @abstractmethod
    async def publish(self, event: Event) -> None: ...

    @abstractmethod
    def subscribe(self, topic: str, handler: Handler) -> None: ...


class InProcessEventBus(EventBus):
    """Asyncio fan-out with isolated handler failures.

    A subscriber that raises must never break ingestion or starve its siblings, so
    handlers are gathered with exceptions returned and logged rather than raised.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[Handler]] = defaultdict(list)

    def subscribe(self, topic: str, handler: Handler) -> None:
        self._handlers[topic].append(handler)
        log.debug("subscribed", topic=topic, handler=getattr(handler, "__name__", repr(handler)))

    async def publish(self, event: Event) -> None:
        handlers = self._handlers.get(event.topic, ())
        if not handlers:
            return
        results = await asyncio.gather(*(h(event) for h in handlers), return_exceptions=True)
        for handler, result in zip(handlers, results, strict=True):
            if isinstance(result, BaseException):
                log.error(
                    "event_handler_failed",
                    topic=event.topic,
                    handler=getattr(handler, "__name__", repr(handler)),
                    error=str(result),
                )

    def subscriber_count(self, topic: str) -> int:
        return len(self._handlers.get(topic, ()))


def candle_closed(candle: Candle) -> Event:
    return Event(
        topic=Topics.CANDLE_CLOSED,
        payload=candle,
        meta={"asset": candle.asset, "timeframe": str(candle.timeframe), "source": candle.source},
    )


def quality_event(event: QualityEvent) -> Event:
    return Event(
        topic=Topics.QUALITY_EVENT,
        payload=event,
        meta={"severity": str(event.severity), "type": str(event.event_type)},
    )
