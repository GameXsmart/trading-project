"""Binance provider — public, read-only endpoints only.

Primary source: it covers every timeframe the engine analyses, offers the deepest
free history, and is the only one of the configured venues exposing derivatives
context (funding and open interest), which the regime model in Phase 7 needs.

No API key is used. Nothing here can place, cancel, or query an order.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from mie.config.settings import ProviderConfig
from mie.core.errors import NotSupported, ProviderError
from mie.core.logging import get_logger
from mie.core.timeframes import Timeframe, ensure_utc, utcnow
from mie.core.types import Candle, FundingRate, MarketType, OpenInterestPoint
from mie.providers.base import HttpProvider, ProviderCapabilities

log = get_logger(__name__)

__all__ = ["BinanceProvider"]

# Binance's interval strings happen to match ours, but relying on that coincidence
# would break silently the day either side changes. Map explicitly.
_INTERVALS: dict[Timeframe, str] = {
    Timeframe.M1: "1m",
    Timeframe.M5: "5m",
    Timeframe.M15: "15m",
    Timeframe.M30: "30m",
    Timeframe.H1: "1h",
    Timeframe.H4: "4h",
    Timeframe.H12: "12h",
    Timeframe.D1: "1d",
    Timeframe.W1: "1w",
}

# Open-interest history is only published on this coarser grid.
_OI_PERIODS: dict[Timeframe, str] = {
    Timeframe.M5: "5m",
    Timeframe.M15: "15m",
    Timeframe.M30: "30m",
    Timeframe.H1: "1h",
    Timeframe.H4: "4h",
    Timeframe.H12: "12h",
    Timeframe.D1: "1d",
}


class BinanceProvider(HttpProvider):
    name = "binance"
    kind = "exchange"
    base_url = "https://api.binance.com"

    #: Derivatives live on a different host than spot.
    futures_url = "https://fapi.binance.com"

    def __init__(self, config: ProviderConfig, symbol_overrides: dict[str, str] | None = None) -> None:
        super().__init__(config, symbol_overrides)
        # api.binance.com is geo-restricted in some jurisdictions; the .us endpoint
        # serves the same kline shape on a smaller universe.
        if config.options.get("use_us_endpoint"):
            self._url = config.base_url or "https://api.binance.us"
            self._futures_enabled = False
        else:
            self._futures_enabled = True

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            timeframes=frozenset(_INTERVALS),
            market_types=frozenset({MarketType.SPOT, MarketType.PERP}),
            max_candles_per_request=1000,
            max_history_days=None,  # effectively back to listing
            supports_funding=self._futures_enabled,
            supports_open_interest=self._futures_enabled,
        )

    def _default_symbol(self, asset: str, quote: str) -> str:
        # Binance quotes the majors against USDT, not USD.
        return f"{asset}{'USDT' if quote.upper() == 'USD' else quote.upper()}"

    async def _health_probe(self) -> None:
        await self._get("/api/v3/ping")

    async def fetch_ohlcv(
        self,
        asset: str,
        timeframe: Timeframe,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
        quote: str = "USDT",
    ) -> list[Candle]:
        interval = _INTERVALS.get(timeframe)
        if interval is None:
            raise NotSupported(self.name, f"timeframe {timeframe}")

        symbol = self.symbol_for(asset, quote)
        params: dict[str, Any] = {
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit or 1000, 1000),
        }
        if start is not None:
            params["startTime"] = _ms(start)
        if end is not None:
            # endTime is inclusive on Binance, so step back a millisecond to keep our
            # half-open [start, end) convention and avoid duplicating a boundary bar.
            params["endTime"] = _ms(end) - 1

        payload = await self._get("/api/v3/klines", params)
        if not isinstance(payload, list):
            raise ProviderError(self.name, f"unexpected klines payload: {type(payload).__name__}")

        now = utcnow()
        candles: list[Candle] = []
        for row in payload:
            try:
                open_time = _from_ms(row[0])
                candles.append(
                    Candle(
                        asset=asset.upper(),
                        quote=_quote_of(symbol, asset),
                        source=self.name,
                        market_type=MarketType.SPOT,
                        timeframe=timeframe,
                        open_time=open_time,
                        open=float(row[1]),
                        high=float(row[2]),
                        low=float(row[3]),
                        close=float(row[4]),
                        volume=float(row[5]),
                        quote_volume=float(row[7]) if row[7] is not None else None,
                        trades=int(row[8]) if row[8] is not None else None,
                        # The bar covering "now" is still forming. Marking it final
                        # here is how look-ahead gets into a live pipeline.
                        is_final=timeframe.close_time(open_time) <= now,
                    )
                )
            except (IndexError, TypeError, ValueError) as exc:
                raise ProviderError(self.name, f"malformed kline row {row!r}: {exc}") from exc
        return candles

    async def fetch_funding(self, asset: str, limit: int = 100) -> list[FundingRate]:
        if not self._futures_enabled:
            raise NotSupported(self.name, "funding unavailable on the US endpoint")
        symbol = f"{asset.upper()}USDT"
        payload = await self._get(
            f"{self.futures_url}/fapi/v1/fundingRate",
            {"symbol": symbol, "limit": min(limit, 1000)},
        )
        if not isinstance(payload, list):
            raise ProviderError(self.name, "unexpected fundingRate payload")
        return [
            FundingRate(
                asset=asset.upper(),
                source=self.name,
                ts=_from_ms(row["fundingTime"]),
                rate=float(row["fundingRate"]),
                interval_hours=8.0,
                mark_price=float(row["markPrice"]) if row.get("markPrice") else None,
            )
            for row in payload
        ]

    async def fetch_open_interest(
        self, asset: str, limit: int = 100, period: Timeframe = Timeframe.H1
    ) -> list[OpenInterestPoint]:
        if not self._futures_enabled:
            raise NotSupported(self.name, "open interest unavailable on the US endpoint")
        binance_period = _OI_PERIODS.get(period)
        if binance_period is None:
            raise NotSupported(self.name, f"open interest period {period}")
        symbol = f"{asset.upper()}USDT"
        payload = await self._get(
            f"{self.futures_url}/futures/data/openInterestHist",
            {"symbol": symbol, "period": binance_period, "limit": min(limit, 500)},
        )
        if not isinstance(payload, list):
            raise ProviderError(self.name, "unexpected openInterestHist payload")
        return [
            OpenInterestPoint(
                asset=asset.upper(),
                source=self.name,
                ts=_from_ms(row["timestamp"]),
                open_interest=float(row["sumOpenInterest"]),
                open_interest_value=float(row["sumOpenInterestValue"]),
            )
            for row in payload
        ]


def _ms(moment: datetime) -> int:
    return int(ensure_utc(moment).timestamp() * 1000)


def _from_ms(value: Any) -> datetime:
    return datetime.fromtimestamp(int(value) / 1000, tz=utcnow().tzinfo)


def _quote_of(symbol: str, asset: str) -> str:
    """Recover the quote currency from a concatenated venue symbol."""
    stripped = symbol.upper().removeprefix(asset.upper())
    return stripped or "USDT"
