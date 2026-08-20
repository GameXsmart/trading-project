"""Provider manager: one façade over N data sources.

Requirement §3 asks that a single API failure must not destroy the system. That is
this class's entire job. Callers ask for candles; the manager decides who can serve
them, in what order, skips sources whose circuit is open, records why a failover
happened, and reports honestly when nobody could answer.

Failover is never silent. Every fallback emits a ``PROVIDER_FAILOVER`` quality event,
because "the data arrived, but from the third-choice venue" is information the
confidence layer needs.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import datetime

from mie.config.settings import Settings
from mie.core.errors import NotSupported, ProviderError
from mie.core.logging import get_logger
from mie.core.timeframes import Timeframe
from mie.core.types import (
    Candle,
    FundingRate,
    GlobalMetricsPoint,
    MarketType,
    OpenInterestPoint,
    ProviderHealth,
    QualityEvent,
    QualityEventType,
    QualitySeverity,
)
from mie.providers.base import MarketDataProvider
from mie.providers.binance import BinanceProvider
from mie.providers.coinbase import CoinbaseProvider
from mie.providers.coingecko import CoinGeckoProvider
from mie.providers.fake import FakeProvider
from mie.providers.kraken import KrakenProvider

log = get_logger(__name__)

__all__ = ["PROVIDER_REGISTRY", "FetchOutcome", "ProviderManager", "build_providers"]

#: Name → implementation. Adding a venue is a registry entry plus a config block;
#: nothing else in the system needs to change.
PROVIDER_REGISTRY: dict[str, type[MarketDataProvider]] = {
    "binance": BinanceProvider,
    "coinbase": CoinbaseProvider,
    "kraken": KrakenProvider,
    "coingecko": CoinGeckoProvider,
    "fake": FakeProvider,
}


class FetchOutcome:
    """Result of a managed fetch: the data, who served it, and what went wrong."""

    __slots__ = ("attempts", "candles", "error", "events", "provider")

    def __init__(
        self,
        candles: list[Candle],
        provider: str | None,
        attempts: list[tuple[str, str]],
        events: list[QualityEvent],
        error: str | None = None,
    ) -> None:
        self.candles = candles
        self.provider = provider
        self.attempts = attempts  # (provider, outcome) in the order tried
        self.events = events
        self.error = error

    @property
    def ok(self) -> bool:
        return self.provider is not None and self.error is None

    def __repr__(self) -> str:  # pragma: no cover
        return f"<FetchOutcome provider={self.provider} candles={len(self.candles)} err={self.error}>"


def build_providers(settings: Settings) -> list[MarketDataProvider]:
    """Instantiate every enabled provider, wiring in per-asset symbol overrides."""
    providers: list[MarketDataProvider] = []
    for config in settings.enabled_providers():
        implementation = PROVIDER_REGISTRY.get(config.name)
        if implementation is None:
            log.warning("unknown_provider_skipped", provider=config.name)
            continue
        overrides = {
            asset.symbol: asset.overrides[config.name]
            for asset in settings.universe.assets
            if config.name in asset.overrides
        }
        providers.append(implementation(config, overrides))
    return providers


class ProviderManager:
    """Priority-ordered failover across providers."""

    def __init__(self, providers: Sequence[MarketDataProvider]) -> None:
        self._providers = sorted(providers, key=lambda p: (p.config.priority, p.name))
        self._health: dict[str, ProviderHealth] = {}

    # ------------------------------------------------------------------ accessors

    @property
    def providers(self) -> list[MarketDataProvider]:
        return list(self._providers)

    def get(self, name: str) -> MarketDataProvider | None:
        return next((p for p in self._providers if p.name == name), None)

    def candidates(
        self,
        asset: str,
        timeframe: Timeframe,
        market_type: MarketType = MarketType.SPOT,
        preferred: str | None = None,
    ) -> list[MarketDataProvider]:
        """Providers that can serve this request, best first.

        A preferred provider is honoured only if it is genuinely capable; asking for
        12h data from Coinbase should fail over, not fail.
        """
        capable = [p for p in self._providers if p.supports(asset, timeframe, market_type)]
        if preferred:
            chosen = next((p for p in capable if p.name == preferred), None)
            return [chosen] if chosen else []
        return capable

    # -------------------------------------------------------------------- fetching

    async def fetch_ohlcv(
        self,
        asset: str,
        timeframe: Timeframe,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
        quote: str = "USDT",
        preferred: str | None = None,
        market_type: MarketType = MarketType.SPOT,
    ) -> FetchOutcome:
        """Try capable providers in order until one returns data."""
        attempts: list[tuple[str, str]] = []
        events: list[QualityEvent] = []
        candidates = self.candidates(asset, timeframe, market_type, preferred)

        if not candidates:
            message = f"no provider serves {asset} {timeframe}"
            events.append(
                QualityEvent(
                    event_type=QualityEventType.PROVIDER_ERROR,
                    severity=QualitySeverity.ERROR,
                    source="manager",
                    asset=asset,
                    timeframe=timeframe,
                    message=message,
                )
            )
            return FetchOutcome([], None, attempts, events, message)

        for index, provider in enumerate(candidates):
            if not provider.breaker.allows():
                attempts.append((provider.name, "circuit_open"))
                log.debug("provider_skipped_circuit_open", provider=provider.name, asset=asset)
                continue
            try:
                candles = await provider.fetch_ohlcv(
                    asset, timeframe, start=start, end=end, limit=limit, quote=quote
                )
            except NotSupported as exc:
                # Capability gap, not a fault: do not penalise the breaker for it.
                attempts.append((provider.name, "not_supported"))
                log.debug("provider_not_supported", provider=provider.name, error=str(exc))
                continue
            except ProviderError as exc:
                provider.breaker.record_failure(str(exc))
                attempts.append((provider.name, "error"))
                events.append(
                    QualityEvent(
                        event_type=QualityEventType.PROVIDER_ERROR,
                        severity=QualitySeverity.WARNING,
                        source=provider.name,
                        asset=asset,
                        timeframe=timeframe,
                        message=str(exc)[:500],
                        details={"circuit": provider.breaker.snapshot()},
                    )
                )
                log.warning(
                    "provider_fetch_failed",
                    provider=provider.name,
                    asset=asset,
                    timeframe=str(timeframe),
                    error=str(exc)[:200],
                )
                continue

            provider.breaker.record_success()

            if not candles:
                # An empty window is not necessarily an error — it may simply predate
                # the listing — so record it and let the next provider try.
                attempts.append((provider.name, "empty"))
                events.append(
                    QualityEvent(
                        event_type=QualityEventType.EMPTY_RESPONSE,
                        severity=QualitySeverity.INFO,
                        source=provider.name,
                        asset=asset,
                        timeframe=timeframe,
                        window_start=start,
                        window_end=end,
                        message="provider returned no candles for the requested window",
                    )
                )
                continue

            attempts.append((provider.name, "ok"))
            if index > 0:
                events.append(
                    QualityEvent(
                        event_type=QualityEventType.PROVIDER_FAILOVER,
                        severity=QualitySeverity.WARNING,
                        source=provider.name,
                        asset=asset,
                        timeframe=timeframe,
                        message=(
                            f"served by fallback provider {provider.name} after "
                            f"{index} higher-priority provider(s) failed"
                        ),
                        details={"attempts": attempts},
                    )
                )
            return FetchOutcome(candles, provider.name, attempts, events)

        message = f"all providers failed for {asset} {timeframe}: {attempts}"
        log.error("all_providers_failed", asset=asset, timeframe=str(timeframe), attempts=attempts)
        return FetchOutcome([], None, attempts, events, message)

    async def fetch_funding(self, asset: str, limit: int = 100) -> list[FundingRate]:
        for provider in self._providers:
            if not provider.capabilities.supports_funding or not provider.breaker.allows():
                continue
            try:
                return await provider.fetch_funding(asset, limit=limit)
            except (NotSupported, ProviderError) as exc:
                log.debug("funding_unavailable", provider=provider.name, error=str(exc)[:200])
        return []

    async def fetch_open_interest(self, asset: str, limit: int = 100) -> list[OpenInterestPoint]:
        for provider in self._providers:
            if not provider.capabilities.supports_open_interest or not provider.breaker.allows():
                continue
            try:
                return await provider.fetch_open_interest(asset, limit=limit)
            except (NotSupported, ProviderError) as exc:
                log.debug("oi_unavailable", provider=provider.name, error=str(exc)[:200])
        return []

    async def fetch_global_metrics(self) -> GlobalMetricsPoint | None:
        for provider in self._providers:
            if not provider.capabilities.supports_global_metrics or not provider.breaker.allows():
                continue
            try:
                return await provider.fetch_global_metrics()
            except (NotSupported, ProviderError) as exc:
                log.debug("global_metrics_unavailable", provider=provider.name, error=str(exc)[:200])
        return None

    # ---------------------------------------------------------------- cross-source

    async def compare_sources(
        self,
        asset: str,
        timeframe: Timeframe,
        limit: int = 10,
        tolerance_pct: float = 0.5,
    ) -> list[QualityEvent]:
        """Fetch the same recent window from every capable provider and compare.

        Exchange prices genuinely differ — different books, different quote currency,
        different fee structures — so small spreads are expected. What this catches is
        the pathological case: a stale feed still serving yesterday's price, or a
        venue whose data has broken in a way that looks fine in isolation. That is
        only visible by comparison, which is why it deserves a dedicated check.
        """
        capable = [p for p in self.candidates(asset, timeframe) if p.breaker.allows()]
        if len(capable) < 2:
            return []

        async def _safe(provider: MarketDataProvider) -> tuple[str, list[Candle]]:
            try:
                return provider.name, await provider.fetch_ohlcv(asset, timeframe, limit=limit)
            except (ProviderError, NotSupported):
                return provider.name, []

        results = dict(await asyncio.gather(*(_safe(p) for p in capable)))
        series = {
            name: {c.open_time: c for c in candles if c.is_final}
            for name, candles in results.items()
            if candles
        }
        if len(series) < 2:
            return []

        reference_name = next(iter(series))
        events: list[QualityEvent] = []
        for name, candles in series.items():
            if name == reference_name:
                continue
            shared = set(series[reference_name]) & set(candles)
            for open_time in sorted(shared):
                left = series[reference_name][open_time].close
                right = candles[open_time].close
                if left <= 0:
                    continue
                deviation = abs(left - right) / left * 100.0
                if deviation > tolerance_pct:
                    events.append(
                        QualityEvent(
                            event_type=QualityEventType.SOURCE_DISCREPANCY,
                            severity=(
                                QualitySeverity.ERROR
                                if deviation > tolerance_pct * 4
                                else QualitySeverity.WARNING
                            ),
                            source=name,
                            asset=asset,
                            timeframe=timeframe,
                            window_start=open_time,
                            window_end=timeframe.close_time(open_time),
                            message=(
                                f"close differs from {reference_name} by {deviation:.2f}% "
                                f"({right:.6g} vs {left:.6g})"
                            ),
                            details={
                                "reference": reference_name,
                                "reference_close": left,
                                "close": right,
                                "deviation_pct": round(deviation, 4),
                            },
                        )
                    )
        return events

    # --------------------------------------------------------------------- health

    async def health(self) -> dict[str, ProviderHealth]:
        """Probe every provider concurrently; a probe failure is itself a result."""
        results = await asyncio.gather(
            *(p.health() for p in self._providers), return_exceptions=True
        )
        health: dict[str, ProviderHealth] = {}
        for provider, result in zip(self._providers, results, strict=True):
            if isinstance(result, BaseException):
                health[provider.name] = ProviderHealth(
                    provider=provider.name, ok=False, error=str(result)[:300]
                )
            else:
                health[provider.name] = result
        self._health = health
        return health

    def last_health(self) -> dict[str, ProviderHealth]:
        return dict(self._health)

    async def close(self) -> None:
        await asyncio.gather(*(p.close() for p in self._providers), return_exceptions=True)

    async def __aenter__(self) -> ProviderManager:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()
