"""The API contract, and the Phase 10 display gate.

The gate reads: *no screen can display a directional call without its confidence and
invalidation conditions visible in the same view.* Enforced in a template that would be
a convention; enforced in the type system it is a property. So the tests here mostly
try to *violate* it and require a validation error — a directional payload with no
invalidation, with a confidence below the publication floor, with a confidence that
disagrees with its own breakdown.

The second thing tested here is what the service does not contain. §21 says the
platform is analytical and must not connect predictions to order execution. That is a
claim about absence, which is exactly the sort of claim that quietly stops being true,
so it is asserted: every route is read-only, and no route name or path suggests
otherwise.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError
from tests.conftest import FIXED_NOW
from tests.test_ensemble import _full_calibration, _full_weights, _panel
from tests.test_models import HOUR, candles, context, drifting

from mie.api.app import _correlation, create_app
from mie.api.schemas import (
    ConfidenceBreakdown,
    DirectionalCall,
    InsufficientEvidence,
)
from mie.api.views import prediction_response
from mie.ensemble.gate import SuperPredictionGate
from mie.ensemble.meta import EnsembleModel

# --------------------------------------------------------------------- helpers


def _source_name(settings) -> str:
    from mie.api.app import _primary_source

    return _primary_source(settings)


def _breakdown(value: float = 0.7) -> ConfidenceBreakdown:
    return ConfidenceBreakdown(
        value=value,
        skill=0.9,
        calibration=0.9,
        agreement=0.9,
        data_quality=1.0,
        sample=0.9,
        regime_familiarity=0.9,
        limiting_factor="sample",
    )


def _call(**overrides):
    payload = {
        "asset": "BTC",
        "timeframe": "1h",
        "horizon": "12h",
        "as_of": FIXED_NOW,
        "resolves_at": FIXED_NOW + timedelta(hours=12),
        "direction": "up",
        "probability_up": 0.5,
        "probability_flat": 0.3,
        "probability_down": 0.2,
        "directional_edge": 0.3,
        "confidence": 0.7,
        "confidence_breakdown": _breakdown(0.7),
        "invalidation": ["a close below 100"],
    }
    payload.update(overrides)
    return DirectionalCall(**payload)


def _ctx(**kwargs):
    return context(drifting(300), **kwargs)


def _published():
    """A panel that genuinely publishes, so the positive path is exercised too."""
    models = _panel([0.5] * 8)
    regimes = ["range_low_vol", "unknown", "range_high_vol", "uptrend_low_vol"]
    library = _full_calibration(models, regimes)
    ensemble = EnsembleModel(models, _full_weights(models), library)
    result = ensemble.predict_detailed(_ctx())
    decision = SuperPredictionGate().evaluate(
        result, library, [m.model_id for m in models]
    )
    return result, decision


def _suppressed():
    """The measured state of this system: nothing has earned a weight."""
    models = _panel([0.4] * 8)
    ensemble = EnsembleModel(models)
    result = ensemble.predict_detailed(_ctx())
    decision = SuperPredictionGate().evaluate(result, None, [m.model_id for m in models])
    return result, decision


# ------------------------------------------------------------- the display gate


class TestDisplayGate:
    """Phase 10's gate, enforced where a UI cannot bypass it."""

    def test_a_valid_call_carries_direction_confidence_and_invalidation(self) -> None:
        call = _call()
        assert call.direction == "up"
        assert call.confidence > 0
        assert call.invalidation
        assert call.is_guaranteed is False

    def test_a_directional_call_without_invalidation_is_rejected(self) -> None:
        """A forecast that cannot be wrong cannot be evaluated."""
        with pytest.raises(ValidationError, match="invalidation condition"):
            _call(invalidation=[])

    def test_whitespace_invalidation_does_not_satisfy_the_gate(self) -> None:
        """Otherwise the requirement is met by a blank string in a list."""
        with pytest.raises(ValidationError, match="invalidation condition"):
            _call(invalidation=["", "   "])

    def test_a_call_below_the_publication_floor_is_rejected(self) -> None:
        """Below the floor the correct answer is insufficient evidence, not a quiet call."""
        with pytest.raises(ValidationError, match="publication floor"):
            _call(confidence=0.2, confidence_breakdown=_breakdown(0.2))

    def test_confidence_must_match_its_own_breakdown(self) -> None:
        """Or the explanation shown to a reader explains a different number."""
        with pytest.raises(ValidationError, match="decomposition"):
            _call(confidence=0.7, confidence_breakdown=_breakdown(0.5))

    def test_probabilities_must_sum_to_one(self) -> None:
        with pytest.raises(ValidationError, match="sum to 1"):
            _call(probability_up=0.9, probability_flat=0.9, probability_down=0.9)

    def test_insufficient_evidence_must_say_why(self) -> None:
        """A blank panel tells a reader nothing and reads as a loading failure."""
        with pytest.raises(ValidationError, match="must say why"):
            InsufficientEvidence(
                asset="BTC", timeframe="1h", horizon="12h", as_of=FIXED_NOW, reasons=[]
            )

    def test_insufficient_evidence_carries_no_probabilities_at_all(self) -> None:
        """It cannot be misread as a weak directional call, because it has no direction."""
        payload = InsufficientEvidence(
            asset="BTC", timeframe="1h", horizon="12h", as_of=FIXED_NOW,
            reasons=["no model has demonstrated skill"],
        ).model_dump()
        assert "direction" not in payload
        assert not any(key.startswith("probability") for key in payload)

    def test_the_headline_carries_all_four_things_at_once(self) -> None:
        """Direction, probability, confidence and falsifiability in one string.

        Written as one string on purpose: four separate template fragments can end up
        on four separate screens, and then the gate is satisfied by the code and
        violated by the interface.
        """
        headline = _call().headline()
        assert "UP" in headline
        assert "50%" in headline
        assert "confidence" in headline
        assert "invalidated by" in headline

    def test_every_directional_payload_declares_it_is_not_a_guarantee(self) -> None:
        """§21: prediction and guarantee must be unmistakable from each other."""
        payload = _call().model_dump()
        assert payload["is_guaranteed"] is False
        assert "is_guaranteed" in payload


class TestPredictionResponse:
    def test_a_suppressed_ensemble_becomes_insufficient_evidence(self) -> None:
        result, decision = _suppressed()
        response = prediction_response(result, decision)
        assert isinstance(response, InsufficientEvidence)
        assert response.reasons
        assert any("skill" in r for r in response.reasons)

    def test_a_published_ensemble_becomes_a_directional_call(self) -> None:
        """The positive path. Without it, the tests above would pass on a converter
        that returned insufficient evidence unconditionally."""
        result, decision = _published()
        response = prediction_response(result, decision)
        assert isinstance(response, DirectionalCall)
        assert response.invalidation
        assert response.confidence >= 0.35

    def test_the_converter_never_emits_a_direction_without_the_rest(self) -> None:
        """The gate, checked over both branches rather than asserted about one."""
        for result, decision in (_suppressed(), _published()):
            response = prediction_response(result, decision)
            if isinstance(response, DirectionalCall):
                assert response.confidence >= 0.35
                assert response.invalidation
                assert response.confidence_breakdown.value == response.confidence

    def test_gate_failures_are_folded_into_the_reasons(self) -> None:
        result, decision = _suppressed()
        response = prediction_response(result, decision)
        assert isinstance(response, InsufficientEvidence)
        assert any(r.startswith("gate:") for r in response.reasons)

    def test_an_empty_suppression_list_still_produces_a_reason(self) -> None:
        """The schema forbids an empty reason list, so the converter must supply one."""
        result, _ = _suppressed()
        result.suppressed_because = []
        response = prediction_response(result, None)
        assert isinstance(response, InsufficientEvidence)
        assert response.reasons


# ------------------------------------------------------------------- the service


@pytest.fixture
async def client(settings, database):
    """The API over a real, empty-but-migrated database.

    Depends on ``database`` because the service refuses to start without a schema —
    which is deliberate, and tested below.
    """
    app = create_app(settings)
    transport = ASGITransport(app=app)
    async with (
        AsyncClient(transport=transport, base_url="http://test") as http,
        app.router.lifespan_context(app),
    ):
        yield http


class TestRoutes:
    async def test_health_reports_that_it_does_not_trade(self, client) -> None:
        response = await client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["executes_trades"] is False

    async def test_status_explains_an_empty_predictions_panel(self, client) -> None:
        """Ambiguity between 'loading', 'broken' and 'nothing to say' is the failure."""
        response = await client.get("/api/status")
        assert response.status_code == 200
        body = response.json()
        assert body["executes_trades"] is False
        assert body["publishes_predictions"] is False
        assert "measured result" in body["headline"]

    async def test_assets_lists_the_configured_universe(self, client) -> None:
        response = await client.get("/api/assets")
        assert response.status_code == 200
        assert {row["asset"] for row in response.json()}

    async def test_thin_history_is_a_clear_404_not_a_crash(self, client) -> None:
        response = await client.get("/api/prediction/BTC")
        assert response.status_code == 404
        assert "backfill" in response.json()["detail"]

    async def test_an_unknown_timeframe_is_rejected(self, client) -> None:
        response = await client.get("/api/prediction/BTC?timeframe=3y")
        assert response.status_code == 400

    async def test_models_and_calibration_are_empty_rather_than_failing(self, client) -> None:
        """No stored outcomes is a state, not an error."""
        assert (await client.get("/api/models")).json() == []
        assert (await client.get("/api/calibration")).json() == []

    async def test_news_and_quality_return_lists(self, client) -> None:
        assert (await client.get("/api/news")).status_code == 200
        assert (await client.get("/api/quality")).status_code == 200

    async def test_correlation_returns_a_square_matrix(self, client) -> None:
        body = (await client.get("/api/correlation")).json()
        assert len(body["matrix"]) == len(body["assets"])
        assert all(len(row) == len(body["assets"]) for row in body["matrix"])

    async def test_the_dashboard_is_served(self, client) -> None:
        response = await client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    async def test_the_dashboard_marks_predictions_as_not_guaranteed(self, client) -> None:
        page = (await client.get("/")).text
        assert "not a guaranteed outcome" in page
        assert "never executes trades" in page

    async def test_the_dashboard_treats_insufficient_evidence_as_a_state(self, client) -> None:
        """Not a spinner, not a blank panel, not an error."""
        page = (await client.get("/")).text
        assert "Insufficient evidence" in page
        assert "measured result, not a failure to load" in page

    async def test_openapi_documents_both_prediction_shapes(self, client) -> None:
        schema = (await client.get("/openapi.json")).json()
        names = set(schema["components"]["schemas"])
        assert "DirectionalCall" in names
        assert "InsufficientEvidence" in names


class TestFreshness:
    """A dashboard dated ten months ago is worse than one that says it has no data."""

    async def test_the_prediction_context_is_built_from_the_newest_bars(
        self, client, database, settings
    ) -> None:
        from mie.storage.repositories import OHLCVRepository, ReferenceRepository

        series = candles(drifting(400))
        async with database.session() as session:
            ReferenceRepository.clear_cache()
            await OHLCVRepository(session).upsert_candles(
                [c.model_copy(update={"source": _source_name(settings)}) for c in series]
            )
            await session.commit()

        response = await client.get("/api/prediction/BTC?horizon=6")
        assert response.status_code == 200
        as_of = response.json()["as_of"][:19]
        # The newest stored bar, not the oldest: `fetch` with a limit takes the wrong end.
        assert as_of == HOUR.close_time(series[-1].open_time).isoformat()[:19]

    async def test_asset_prices_come_from_the_newest_bars(
        self, client, database, settings
    ) -> None:
        from mie.storage.repositories import OHLCVRepository, ReferenceRepository

        series = candles(drifting(400))
        async with database.session() as session:
            ReferenceRepository.clear_cache()
            await OHLCVRepository(session).upsert_candles(
                [c.model_copy(update={"source": _source_name(settings)}) for c in series]
            )
            await session.commit()

        rows = {r["asset"]: r for r in (await client.get("/api/assets")).json()}
        assert rows["BTC"]["price"] == pytest.approx(series[-1].close, abs=1e-6)


class TestDerivativesReachTheModels:
    """A wire that was never connected, and went unnoticed for eleven phases.

    Funding and open interest were collected but never passed into a prediction
    context, so the orderflow model abstained on every point in every evaluation.
    That looked like a finding about markets and was a missing argument.
    """

    async def test_funding_history_round_trips(self, database, settings) -> None:
        from mie.core.types import FundingRate
        from mie.storage.repositories import DerivativesRepository, ReferenceRepository

        source = _source_name(settings)
        points = [
            FundingRate(
                asset="BTC",
                source=source,
                ts=FIXED_NOW - timedelta(hours=8 * i),
                rate=0.0001 * (i + 1),
            )
            for i in range(6)
        ]
        async with database.session() as session:
            ReferenceRepository.clear_cache()
            await DerivativesRepository(session).upsert_funding(points)
            await session.commit()
        async with database.session() as session:
            history = await DerivativesRepository(session).funding_history("BTC")

        assert len(history) == len(points)
        assert [t for t, _ in history] == sorted(t for t, _ in history)
        assert all(isinstance(rate, float) for _, rate in history)

    async def test_the_live_context_carries_funding(
        self, client, database, settings
    ) -> None:
        """The regression that matters: the API's prediction path must pass it on."""
        from mie.core.types import FundingRate
        from mie.storage.repositories import (
            DerivativesRepository,
            OHLCVRepository,
            ReferenceRepository,
        )

        source = _source_name(settings)
        series = candles(drifting(400))
        async with database.session() as session:
            ReferenceRepository.clear_cache()
            await OHLCVRepository(session).upsert_candles(
                [c.model_copy(update={"source": source}) for c in series]
            )
            await DerivativesRepository(session).upsert_funding(
                [
                    FundingRate(
                        asset="BTC",
                        source=source,
                        ts=HOUR.close_time(bar.open_time),
                        rate=0.0001 * (index % 7),
                    )
                    for index, bar in enumerate(series[::8])
                ]
            )
            await session.commit()

        result, _ = await client._transport.app.state.engine.predict("BTC", HOUR, 6)
        orderflow = next(p for p in result.members if p.model_id == "orderflow")
        assert "insufficient funding history" not in str(orderflow.evidence)


class TestStartup:
    async def test_serving_without_a_schema_fails_clearly(self, settings) -> None:
        """One legible failure at boot beats nine illegible ones at request time."""
        from mie.core.errors import StorageError

        app = create_app(settings)
        with pytest.raises(StorageError, match="mie db init"):
            async with app.router.lifespan_context(app):
                pass


class TestSafetyBoundary:
    """§21 is a claim about absence, so it is asserted rather than assumed."""

    def test_no_route_mutates_anything(self, settings) -> None:
        app = create_app(settings)
        mutating = {"POST", "PUT", "PATCH", "DELETE"}
        offenders = [
            (route.path, sorted(methods & mutating))
            for route in app.routes
            if (methods := getattr(route, "methods", set()) or set()) & mutating
        ]
        assert offenders == []

    def test_no_route_hints_at_execution(self, settings) -> None:
        app = create_app(settings)
        forbidden = ("order", "trade", "execute", "buy", "sell", "withdraw", "position")
        paths = [getattr(route, "path", "") for route in app.routes]
        assert not [p for p in paths if any(word in p.lower() for word in forbidden)]


class TestCorrelationMath:
    def test_a_series_correlates_perfectly_with_itself(self) -> None:
        series = [c.close for c in candles(drifting(200))]
        assert _correlation(series, series) == pytest.approx(1.0)

    def test_an_inverted_series_correlates_negatively(self) -> None:
        series = [c.close for c in candles(drifting(200))]
        assert _correlation(series, [-x for x in series]) == pytest.approx(-1.0)

    def test_too_short_a_series_returns_zero_rather_than_a_fake_number(self) -> None:
        assert _correlation([1.0, 2.0], [1.0, 2.0]) == 0.0

    def test_a_constant_series_has_no_correlation(self) -> None:
        assert _correlation([1.0] * 50, list(range(50))) == 0.0
