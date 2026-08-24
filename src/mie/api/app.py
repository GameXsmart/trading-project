"""The HTTP and WebSocket surface.

Read-only by construction. There is no endpoint that places an order, moves funds, or
mutates a stored prediction, and there is no code path from a prediction to an
execution venue — §21's safety boundary is a property of what this module does not
contain, so the absence is stated here where someone auditing the API will look.

Two practical decisions worth explaining.

**Weights come from storage, not from a fresh evaluation.** Re-running a walk-forward
evaluation to answer one HTTP request would take minutes. Phase 9 already persists what
was learned, so the API reads it. That also means the API cannot accidentally *become*
the learning loop and start fitting on request traffic.

**Calibration and the ensemble are cached with a short TTL.** Refitting per request
would be wasteful, but caching forever would let the dashboard show a weight the loop
has since revised. The TTL makes the staleness bounded and visible rather than
unbounded and invisible.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median
from typing import Any

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from mie.api.schemas import (
    AssetSummary,
    CalibrationBin,
    GateCondition,
    ModelPerformance,
    NewsItem,
    PredictionResponse,
    QualitySummary,
    StateView,
    SystemStatus,
)
from mie.api.views import (
    calibration_bins,
    gate_conditions,
    model_performance,
    news_items,
    prediction_response,
    state_view,
)
from mie.config.settings import Settings, load_settings
from mie.core.errors import StorageError
from mie.core.logging import configure_logging, get_logger
from mie.core.timeframes import Timeframe, utcnow
from mie.ensemble.calibration import CalibrationLibrary, reliability_diagram
from mie.ensemble.gate import SuperPredictionGate
from mie.ensemble.meta import EnsembleModel, SkillWeights
from mie.features.engine import _row_to_candle
from mie.learning.metrics import slice_outcomes
from mie.models.predictors import ALL_MODELS
from mie.models.runner import ContextSource
from mie.models.types import Horizon, Outcome
from mie.storage.db import Database
from mie.storage.repositories import (
    DerivativesRepository,
    FeatureRepository,
    NewsEventRepository,
    OHLCVRepository,
    PredictionRepository,
    QualityRepository,
)

log = get_logger(__name__)

__all__ = ["create_app"]

_STATIC = Path(__file__).parent / "static"

#: How long a fitted calibration library and weight table stay cached. Short enough
#: that a dashboard never shows a weight the loop revised an hour ago.
_CACHE_TTL = timedelta(minutes=5)

#: Bars loaded per asset for a live prediction. Enough for every model's warmup with
#: room to spare; unbounded loading would make one request able to exhaust memory.
_CONTEXT_BARS = 1200


def _primary_source(settings: Settings) -> str:
    """The provider whose series the API reads.

    A single venue, deliberately. Mixing sources within one series is how an indicator
    becomes an average of two different order books — the same reason Phase 1 keys
    candles by instrument rather than by asset.
    """
    providers = settings.enabled_providers()
    return providers[0].name if providers else "binance"


@dataclass
class _Cached:
    value: Any
    at: datetime

    def fresh(self, ttl: timedelta = _CACHE_TTL) -> bool:
        return utcnow() - self.at < ttl


class _Engine:
    """Assembles the prediction stack from stored state, with caching."""

    def __init__(self, database: Database, settings: Settings) -> None:
        self.database = database
        self.settings = settings
        self.source = _primary_source(settings)
        self._calibration: _Cached | None = None
        self._weights: _Cached | None = None
        self._lock = asyncio.Lock()

    async def weights(self) -> SkillWeights:
        """Skill weights as persisted by the Phase 9 loop.

        Read, never computed here. The API is a reader of what was learned; letting it
        fit would make request traffic an input to the model.
        """
        if self._weights and self._weights.fresh():
            return self._weights.value
        async with self.database.session() as session:
            rows = await PredictionRepository(session).weights()
        table = SkillWeights()
        for row in rows:
            if row.weight > 0:
                table.weights[(row.model_id, row.regime)] = row.weight
            table.samples[(row.model_id, row.regime)] = row.samples
            table.regime_samples[row.regime] = max(
                table.regime_samples.get(row.regime, 0), row.samples
            )
        self._weights = _Cached(table, utcnow())
        return table

    async def calibration(self) -> CalibrationLibrary:
        if self._calibration and self._calibration.fresh():
            return self._calibration.value
        async with self.database.session() as session:
            repo = PredictionRepository(session)
            records = await repo.records()
            outcomes = await repo.outcomes()
        library = CalibrationLibrary()
        library.fit(_scored_pairs(records, outcomes))
        self._calibration = _Cached(library, utcnow())
        return library

    async def predict(
        self, asset: str, timeframe: Timeframe, horizon_bars: int
    ) -> tuple[Any, Any]:
        """Build the latest context and run the ensemble over it."""
        async with self.database.session() as session:
            # fetch_recent, not fetch: `fetch` with a limit takes the *earliest* rows,
            # which would build a context at the start of stored history and label it
            # current. Its own docstring warns about this and I walked into it anyway;
            # the symptom was a dashboard confidently dated ten months ago.
            rows = await OHLCVRepository(session).fetch_recent(
                asset, timeframe, source=self.source, limit=_CONTEXT_BARS
            )
            features = await FeatureRepository(session).fetch(
                asset,
                timeframe,
                source=self.source,
                start=rows[0].open_time if rows else None,
            )
            derivatives = DerivativesRepository(session)
            since = rows[0].open_time if rows else None
            funding = await derivatives.funding_history(asset, start=since)
            open_interest = await derivatives.open_interest_history(asset, start=since)
        if len(rows) < 250:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"not enough stored history for {asset.upper()} {timeframe} "
                    f"({len(rows)} bars). Run `mie backfill {asset.upper()} {timeframe}`."
                ),
            )

        source_name = self.source
        candles = [_row_to_candle(r, asset, timeframe, source_name) for r in rows]
        history = [(f.open_time, f.payload) for f in features]
        source = ContextSource(
            asset,
            timeframe,
            candles,
            history,
            funding=funding,
            open_interest=open_interest,
        )
        context = source.context_at(len(source.candles) - 1, Horizon(bars=horizon_bars, timeframe=timeframe))
        if context is None:
            raise HTTPException(status_code=503, detail="could not build a prediction context")

        ensemble = EnsembleModel(
            [m() for m in ALL_MODELS], await self.weights(), await self.calibration()
        )
        result = ensemble.predict_detailed(context)
        decision = SuperPredictionGate().evaluate(
            result, ensemble.calibration, [m.model_id for m in ensemble.members]
        )
        return result, decision


@dataclass(frozen=True)
class _Pair:
    """What the calibrator needs: something that said a distribution, and what happened.

    :class:`~mie.learning.records.PredictionRecord` already exposes every attribute the
    calibrator reads — model id, regime, instant, confidence, distribution — so it is
    passed straight through rather than copied into a parallel shape that could drift
    away from it.
    """

    prediction: Any
    actual: Outcome


def _scored_pairs(records: list[Any], outcomes: list[Any]) -> list[_Pair]:
    """Pair stored prediction rows with their resolved outcomes."""
    from mie.cli import _row_to_outcome, _row_to_record

    by_id = {r.prediction_id: _row_to_record(r) for r in records}
    pairs: list[_Pair] = []
    for row in outcomes:
        record = by_id.get(row.prediction_id)
        if record is None or record.confidence <= 0:
            continue
        pairs.append(_Pair(prediction=record, actual=_row_to_outcome(row).realised_direction))
    return pairs


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application. Read-only: nothing here mutates market state."""
    resolved = settings or load_settings()
    configure_logging(resolved.app.log_level, json_output=resolved.app.log_json)
    source_name = _primary_source(resolved)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        database = Database(resolved)
        await database.ensure_ready()
        await _require_schema(database)
        app.state.database = database
        app.state.settings = resolved
        app.state.engine = _Engine(database, resolved)
        try:
            yield
        finally:
            await database.dispose()

    app = FastAPI(
        title="Crypto Market Intelligence Engine",
        description=(
            "Analytical, read-only. Produces probabilistic assessments with explicit "
            "uncertainty. Does not execute trades and has no order-execution path."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    # ------------------------------------------------------------------ status

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "served_at": utcnow().isoformat(), "executes_trades": False}

    @app.get("/api/status", response_model=SystemStatus)
    async def status() -> SystemStatus:
        engine: _Engine = app.state.engine
        async with app.state.database.session() as session:
            counts = await PredictionRepository(session).counts()
            bars = 0
            for symbol in resolved.universe.symbols():
                bars += len(
                    await OHLCVRepository(session).fetch(
                        symbol, Timeframe.H1, source=source_name, limit=100000
                    )
                )
        weights = await engine.weights()
        skilled = len(weights.skilled_models())
        return SystemStatus(
            assets_tracked=len(resolved.universe.symbols()),
            bars_stored=bars,
            predictions_recorded=counts["predictions"],
            outcomes_resolved=counts["outcomes"],
            models_with_weight=skilled,
            publishes_predictions=skilled > 0,
            headline=(
                "No model has demonstrated skill against a climatology baseline, so no "
                "directional call is published. This is a measured result, not a fault."
                if skilled == 0
                else f"{skilled} models carry a non-zero weight."
            ),
        )

    @app.get("/api/assets", response_model=list[AssetSummary])
    async def assets() -> list[AssetSummary]:
        out: list[AssetSummary] = []
        async with app.state.database.session() as session:
            ohlcv = OHLCVRepository(session)
            quality = QualityRepository(session)
            for symbol in resolved.universe.symbols():
                rows = await ohlcv.fetch_recent(
                    symbol, Timeframe.H1, source=source_name, limit=200
                )
                if not rows:
                    out.append(AssetSummary(asset=symbol, bars_stored=0))
                    continue
                closes = [r.close for r in rows if r.close > 0]
                returns = [
                    abs(closes[i] - closes[i - 1]) / closes[i - 1] * 100.0
                    for i in range(1, len(closes))
                ]
                day = closes[-25] if len(closes) >= 25 else closes[0]
                score = await quality.get_score(
                    source_name, symbol, Timeframe.H1
                )
                out.append(
                    AssetSummary(
                        asset=symbol,
                        price=round(closes[-1], 6),
                        change_24h_pct=round((closes[-1] - day) / day * 100.0, 4) if day else 0.0,
                        volatility_pct=round(median(returns), 4) if returns else 0.0,
                        data_quality=round(float(score), 4),
                        bars_stored=len(rows),
                        last_bar=rows[-1].open_time,
                    )
                )
        return out

    # -------------------------------------------------------------- prediction

    @app.get("/api/prediction/{asset}", response_model=PredictionResponse)
    async def prediction(
        asset: str,
        timeframe: str = Query("1h"),
        horizon: int = Query(12, ge=1, le=200),
    ) -> Any:
        """The system's current view — or, far more often, why it has none.

        Returns one of exactly two shapes. There is no third form that carries a
        direction with a caveat attached, because a caveated direction still reaches a
        reader as a direction.
        """
        frame = _timeframe(timeframe)
        result, decision = await app.state.engine.predict(asset, frame, horizon)
        return prediction_response(result, decision)

    @app.get("/api/gate/{asset}", response_model=list[GateCondition])
    async def gate(
        asset: str, timeframe: str = Query("1h"), horizon: int = Query(12, ge=1, le=200)
    ) -> list[GateCondition]:
        """Every super-prediction condition, passing or failing, with its numbers."""
        frame = _timeframe(timeframe)
        _, decision = await app.state.engine.predict(asset, frame, horizon)
        return gate_conditions(decision)

    @app.get("/api/state/{asset}", response_model=StateView)
    async def state(asset: str) -> StateView:
        from mie.state.engine import StateEngine

        engine = StateEngine(app.state.database, resolved, source=source_name)
        try:
            computed = await engine.build(asset.upper())
        except Exception as exc:
            raise HTTPException(
                status_code=404, detail=f"no state for {asset.upper()}: {exc}"
            ) from exc
        return state_view(asset, computed)

    # ------------------------------------------------------------ performance

    @app.get("/api/models", response_model=list[ModelPerformance])
    async def models(dimension: str = Query("overall")) -> list[ModelPerformance]:
        async with app.state.database.session() as session:
            repo = PredictionRepository(session)
            outcome_rows = await repo.outcomes()
            weight_rows = await repo.weights()
        if not outcome_rows:
            return []
        from mie.cli import _row_to_outcome

        table = slice_outcomes([_row_to_outcome(r) for r in outcome_rows])
        weights = {(w.model_id, w.regime): w.weight for w in weight_rows}
        return model_performance(table.for_dimension(dimension), weights)

    @app.get("/api/calibration", response_model=list[CalibrationBin])
    async def calibration(model: str = Query("")) -> list[CalibrationBin]:
        async with app.state.database.session() as session:
            repo = PredictionRepository(session)
            records = await repo.records()
            outcomes = await repo.outcomes()
        if not outcomes:
            return []
        from mie.cli import _row_to_outcome, _row_to_record

        by_id = {r.prediction_id: _row_to_record(r) for r in records}
        pairs: list[tuple[float, float]] = []
        for row in outcomes:
            record = by_id.get(row.prediction_id)
            if record is None or record.confidence <= 0:
                continue
            if model and record.model_id != model:
                continue
            actual = _row_to_outcome(row).realised_direction
            for outcome in Outcome:
                pairs.append(
                    (
                        record.distribution.probability(outcome),
                        1.0 if outcome is actual else 0.0,
                    )
                )
        return calibration_bins(reliability_diagram(pairs))

    @app.get("/api/quality", response_model=list[QualitySummary])
    async def quality() -> list[QualitySummary]:
        async with app.state.database.session() as session:
            scores = await QualityRepository(session).all_scores()
        return [
            QualitySummary(
                source=s.source,
                asset=s.asset,
                timeframe=s.timeframe,
                score=round(float(s.score), 4),
                events_24h=int(getattr(s, "events_24h", 0) or 0),
            )
            for s in scores
        ]

    @app.get("/api/news", response_model=list[NewsItem])
    async def news(hours: int = Query(168, ge=1, le=720), limit: int = Query(50, le=200)) -> list[NewsItem]:
        async with app.state.database.session() as session:
            rows = await NewsEventRepository(session).recent(hours=hours, limit=limit)
        return news_items(rows)

    @app.get("/api/correlation")
    async def correlation(timeframe: str = Query("1h"), window: int = Query(200, ge=30, le=2000)) -> dict[str, Any]:
        """Pairwise return correlation across the stored universe."""
        frame = _timeframe(timeframe)
        series: dict[str, list[float]] = {}
        async with app.state.database.session() as session:
            for symbol in resolved.universe.symbols():
                rows = await OHLCVRepository(session).fetch_recent(
                    symbol, frame, source=source_name, limit=window + 1
                )
                closes = [r.close for r in rows if r.close > 0]
                if len(closes) > 30:
                    series[symbol] = [
                        (closes[i] - closes[i - 1]) / closes[i - 1]
                        for i in range(1, len(closes))
                    ]
        symbols = sorted(series)
        matrix = [
            [round(_correlation(series[a], series[b]), 4) for b in symbols] for a in symbols
        ]
        return {"assets": symbols, "matrix": matrix, "window": window}

    # -------------------------------------------------------------- websocket

    @app.websocket("/ws")
    async def stream(socket: WebSocket, asset: str = "BTC", interval: int = 15) -> None:
        """Push status and the current prediction shape on an interval.

        Push-only. A socket that accepted commands would be a mutation surface, and
        this service does not have one.
        """
        await socket.accept()
        frame = Timeframe.H1
        try:
            while True:
                payload: dict[str, Any] = {"served_at": utcnow().isoformat()}
                try:
                    result, decision = await app.state.engine.predict(asset, frame, 12)
                    payload["prediction"] = prediction_response(result, decision).model_dump(
                        mode="json"
                    )
                except HTTPException as exc:
                    payload["error"] = exc.detail
                await socket.send_json(payload)
                await asyncio.sleep(max(5, interval))
        except WebSocketDisconnect:  # pragma: no cover - network path
            return
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("websocket_closed", error=str(exc)[:200])
            with contextlib.suppress(Exception):
                await socket.close()

    # -------------------------------------------------------------- dashboard

    @app.get("/", response_class=HTMLResponse)
    async def dashboard() -> HTMLResponse:
        page = _STATIC / "dashboard.html"
        if not page.exists():  # pragma: no cover - packaging guard
            return HTMLResponse("<h1>dashboard asset missing</h1>", status_code=500)
        return HTMLResponse(page.read_text(encoding="utf-8"))

    @app.exception_handler(HTTPException)
    async def _http_error(_: Any, exc: HTTPException) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    return app


async def _require_schema(database: Database) -> None:
    """Fail at startup, clearly, rather than 500 on every route.

    Serving a dashboard against an empty database produces a page of identical
    unexplained errors, and the actual problem — the schema was never created — is
    nowhere on it. One legible failure at boot is worth more than nine illegible ones
    at request time.
    """
    from sqlalchemy import inspect

    async with database.engine.connect() as conn:
        tables = await conn.run_sync(lambda sync: inspect(sync).get_table_names())
    missing = {"ohlcv", "assets", "instruments"} - set(tables)
    if missing:
        raise StorageError(
            f"database schema is missing {', '.join(sorted(missing))}. "
            f"Run `mie db init` before `mie serve`."
        )


def _timeframe(value: str) -> Timeframe:
    try:
        return Timeframe(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"unknown timeframe {value!r}") from exc


def _correlation(left: list[float], right: list[float]) -> float:
    """Pearson correlation over the overlapping tail of two return series."""
    size = min(len(left), len(right))
    if size < 30:
        return 0.0
    a, b = left[-size:], right[-size:]
    mean_a = sum(a) / size
    mean_b = sum(b) / size
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b, strict=True))
    var_a = sum((x - mean_a) ** 2 for x in a)
    var_b = sum((y - mean_b) ** 2 for y in b)
    if var_a <= 0 or var_b <= 0:
        return 0.0
    return cov / (var_a**0.5 * var_b**0.5)
