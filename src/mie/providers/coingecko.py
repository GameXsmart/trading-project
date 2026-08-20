"""CoinGecko provider — market-wide aggregates.

Not a candle source. It supplies the cross-sectional context the per-asset feeds
cannot: BTC dominance, total market capitalisation, aggregate volume, and stablecoin
share. Those are inputs to regime detection (§7) and to the rotation logic in the
cross-asset model, so they are collected on their own slow cadence.

The free tier is aggressively rate-limited, which is why this provider sits at
priority 90 and polls every ten minutes rather than every cycle.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from mie.core.errors import NotSupported, ProviderError
from mie.core.logging import get_logger
from mie.core.timeframes import Timeframe, utcnow
from mie.core.types import Candle, GlobalMetricsPoint, MarketType
from mie.providers.base import HttpProvider, ProviderCapabilities

log = get_logger(__name__)

__all__ = ["CoinGeckoProvider"]

# Ticker symbols that identify a stablecoin for the purpose of the share metric.
_STABLECOINS = ("usdt", "usdc", "dai", "fdusd", "usde", "tusd", "usds", "pyusd")


class CoinGeckoProvider(HttpProvider):
    name = "coingecko"
    kind = "aggregator"
    base_url = "https://api.coingecko.com/api/v3"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            timeframes=frozenset(),  # no OHLCV role at all
            market_types=frozenset({MarketType.SPOT}),
            max_candles_per_request=0,
            supports_global_metrics=True,
        )

    async def _health_probe(self) -> None:
        await self._get("/ping")

    async def fetch_ohlcv(
        self,
        asset: str,
        timeframe: Timeframe,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
        quote: str = "USDT",
    ) -> list[Candle]:
        # CoinGecko does publish OHLC, but on fixed windows with coarse, non-configurable
        # granularity and aggregated cross-venue pricing. Mixing that into a series built
        # from exchange candles would produce a silently inconsistent history, so this
        # provider declines the role outright.
        raise NotSupported(self.name, "OHLCV is served by exchange providers, not the aggregator")

    async def fetch_global_metrics(self) -> GlobalMetricsPoint:
        payload = await self._get("/global")
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, dict):
            raise ProviderError(self.name, "unexpected /global payload")

        dominance: dict[str, Any] = data.get("market_cap_percentage") or {}
        total_cap = (data.get("total_market_cap") or {}).get("usd")
        total_volume = (data.get("total_volume") or {}).get("usd")
        stable_share = sum(float(dominance.get(coin, 0.0)) for coin in _STABLECOINS)

        ts = data.get("updated_at")
        moment = datetime.fromtimestamp(int(ts), tz=utcnow().tzinfo) if ts else utcnow()

        return GlobalMetricsPoint(
            source=self.name,
            ts=moment,
            btc_dominance=_maybe_float(dominance.get("btc")),
            eth_dominance=_maybe_float(dominance.get("eth")),
            total_market_cap_usd=_maybe_float(total_cap),
            total_volume_24h_usd=_maybe_float(total_volume),
            stablecoin_share=stable_share or None,
        )


def _maybe_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
