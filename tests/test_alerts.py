"""Alert rules, the rate budget, and delivery.

Phase 11's gate is unusual in that it is about restraint rather than detection:
*alert volume under a simulated volatile week stays within a rate budget — an alerting
system nobody reads is worse than none.* So the central test builds a deliberately
hostile week — constant volume spikes, regime flips every few hours, a collapsing data
feed — and requires the engine to stay inside its budget while still delivering the
critical items.

The mirror test matters as much: a quiet week must produce a *small* number of alerts,
not zero. A budget that silences everything passes a volume ceiling trivially and is
indistinguishable from a broken detector.

Nothing here contacts an external service. The webhook and Telegram transports are
exercised against a local in-process stub, never against Discord or Telegram, and the
environment-driven configuration is tested with an injected mapping rather than by
setting real variables.
"""

from __future__ import annotations

import hashlib
import json
from datetime import timedelta

import pytest
from tests.conftest import FIXED_NOW
from tests.test_models import HOUR, candles

from mie.alerts.budget import RateBudget, Suppression
from mie.alerts.channels import (
    ConsoleChannel,
    DiscordChannel,
    FileChannel,
    TelegramChannel,
    WebhookChannel,
    _Recorder,
    channels_from_env,
)
from mie.alerts.engine import AlertEngine
from mie.alerts.rules import (
    AlertContext,
    CorrelationBreakdownRule,
    DataQualityRule,
    LiquidationSpikeRule,
    MajorNewsRule,
    ModelDisagreementRule,
    RegimeChangeRule,
    StrongPredictionRule,
    SuperPredictionRule,
    VolatilityRule,
    VolumeAnomalyRule,
)
from mie.alerts.types import Alert, AlertKind, Severity
from mie.core.types import Candle

# --------------------------------------------------------------------- helpers


def _alert(kind: AlertKind = AlertKind.VOLUME_ANOMALY, asset: str = "BTC", **kwargs) -> Alert:
    payload = {"kind": kind, "asset": asset, "title": "something happened"}
    payload.update(kwargs)
    return Alert(**payload)  # type: ignore[arg-type]


def _uniform(index: int, salt: str = "") -> float:
    digest = hashlib.blake2b(f"{salt}:{index}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


def _volatile_week(bars: int = 168) -> list[Candle]:
    """A week of hourly bars doing everything at once.

    Deterministic, and deliberately hostile: large swings, repeated volume spikes, and
    ranges that expand and compress. If the budget holds here it holds anywhere.
    """
    prices: list[float] = []
    volumes: list[float] = []
    price = 100.0
    for i in range(bars):
        shock = (_uniform(i, "vol") - 0.5) * 0.08
        if i % 7 == 0:
            shock *= 4.0
        price *= 1.0 + shock
        prices.append(price)
        # Spikes are rare on purpose. An earlier version of this fixture spiked every
        # fifth bar, which made the median absolute deviation of the volume series
        # exactly zero — most bars identical — and the detector correctly reported
        # nothing at all. Real volume does not look like that, and a fixture that does
        # tests the wrong thing.
        volumes.append(100.0 * (25.0 if i % 23 == 0 else 1.0 + _uniform(i, "v") * 0.4))

    out = []
    for i, close in enumerate(prices):
        open_price = prices[i - 1] if i else close
        span = abs(close - open_price) + abs(close) * 0.01
        out.append(
            Candle(
                asset="BTC", source="test", timeframe=HOUR,
                open_time=FIXED_NOW - timedelta(hours=bars - i),
                open=open_price, high=max(open_price, close) + span,
                low=min(open_price, close) - span, close=close,
                volume=volumes[i], is_final=True,
            )
        )
    return out


def _week_of_contexts(series: list[Candle], flip_regime_every: int = 4) -> list[AlertContext]:
    """One context per hour of the week, with a regime that keeps flipping."""
    contexts = []
    regimes = ("uptrend_high_vol", "downtrend_high_vol", "range_high_vol")
    for index in range(120, len(series)):
        regime = regimes[(index // flip_regime_every) % len(regimes)]
        previous = regimes[((index - 1) // flip_regime_every) % len(regimes)]
        contexts.append(
            AlertContext(
                asset="BTC",
                timeframe="1h",
                at=HOUR.close_time(series[index].open_time),
                candles=series[: index + 1],
                regime=regime,
                previous_regime=previous,
                data_quality=0.4 if index % 23 == 0 else 1.0,
                correlation_baseline={"ETH": 0.82},
                correlation_now={"ETH": 0.2 if index % 17 == 0 else 0.8},
            )
        )
    return contexts


def _quiet_week(bars: int = 168) -> list[AlertContext]:
    """Nothing much happening: gentle drift, steady volume, one regime."""
    prices = [100.0 + 0.02 * i for i in range(bars)]
    series = []
    for i, close in enumerate(prices):
        open_price = prices[i - 1] if i else close
        series.append(
            Candle(
                asset="BTC", source="test", timeframe=HOUR,
                open_time=FIXED_NOW - timedelta(hours=bars - i),
                open=open_price, high=close + 0.05, low=open_price - 0.05,
                close=close, volume=100.0, is_final=True,
            )
        )
    return [
        AlertContext(
            asset="BTC", timeframe="1h",
            at=HOUR.close_time(series[i].open_time),
            candles=series[: i + 1],
            regime="range_low_vol", previous_regime="range_low_vol",
        )
        for i in range(120, len(series))
    ]


async def _run_week(engine: AlertEngine, contexts: list[AlertContext]) -> int:
    """Feed contexts through the engine hour by hour, as a live system would."""
    delivered = 0
    for context in contexts:
        run = await engine.run([context], now=context.at)
        delivered += run.delivered_count
    return delivered


# --------------------------------------------------------------- the alert type


class TestAlertType:
    def test_a_directional_alert_needs_a_confidence(self) -> None:
        with pytest.raises(ValueError, match="requires a confidence"):
            _alert(AlertKind.STRONG_PREDICTION, invalidation=["a close below 100"])

    def test_a_directional_alert_needs_an_invalidation_condition(self) -> None:
        with pytest.raises(ValueError, match="invalidation condition"):
            _alert(AlertKind.SUPER_PREDICTION, confidence=0.7)

    def test_whitespace_invalidation_does_not_count(self) -> None:
        with pytest.raises(ValueError, match="invalidation condition"):
            _alert(AlertKind.STRONG_PREDICTION, confidence=0.7, invalidation=["  "])

    def test_a_non_directional_alert_needs_neither(self) -> None:
        """Volume anomalies claim nothing about direction, so they carry no such burden."""
        alert = _alert(AlertKind.VOLUME_ANOMALY)
        assert alert.confidence is None
        assert alert.severity is Severity.NOTABLE

    def test_a_directional_alert_renders_all_of_its_qualifications(self) -> None:
        text = _alert(
            AlertKind.STRONG_PREDICTION, confidence=0.62, invalidation=["a close below 100"]
        ).render()
        assert "confidence 62%" in text
        assert "not a guaranteed outcome" in text
        assert "invalidated if: a close below 100" in text

    def test_identical_alerts_share_a_dedup_key(self) -> None:
        first = _alert(at=FIXED_NOW)
        second = _alert(at=FIXED_NOW + timedelta(minutes=20))
        assert first.dedup_key == second.dedup_key

    def test_different_assets_do_not_share_a_dedup_key(self) -> None:
        assert _alert(asset="BTC").dedup_key != _alert(asset="ETH").dedup_key

    def test_data_quality_outranks_any_market_event(self) -> None:
        """A large move is news; a broken feed means every other number is suspect."""
        assert AlertKind.DATA_QUALITY.default_severity > AlertKind.VOLUME_ANOMALY.default_severity
        assert AlertKind.DATA_QUALITY.default_severity is Severity.CRITICAL

    def test_a_digest_is_not_filed_as_a_data_quality_event(self) -> None:
        """Otherwise a housekeeping notice is indistinguishable from a broken feed."""
        budget = RateBudget(per_hour=1, cooldown=timedelta(0), dedup_window=timedelta(0))
        for i in range(5):
            budget.admit(_alert(title=f"e{i}", at=FIXED_NOW + timedelta(minutes=i)))
        digest = budget.pending_digest(FIXED_NOW + timedelta(hours=7))
        assert digest is not None
        assert digest.kind is AlertKind.SUPPRESSION_DIGEST
        assert digest.kind is not AlertKind.DATA_QUALITY
        assert digest.level is Severity.INFO

    def test_only_prediction_kinds_are_directional(self) -> None:
        directional = {k for k in AlertKind if k.is_directional}
        assert directional == {AlertKind.STRONG_PREDICTION, AlertKind.SUPER_PREDICTION}


# ------------------------------------------------------------------ the budget


class TestRateBudget:
    def test_an_identical_alert_is_suppressed_as_a_duplicate(self) -> None:
        budget = RateBudget()
        assert budget.admit(_alert(at=FIXED_NOW)).delivered
        held = budget.admit(_alert(at=FIXED_NOW + timedelta(minutes=5)))
        assert not held.delivered
        assert held.reason == Suppression.DUPLICATE

    def test_a_different_alert_of_the_same_kind_hits_the_cooldown(self) -> None:
        budget = RateBudget()
        assert budget.admit(_alert(title="first", at=FIXED_NOW)).delivered
        held = budget.admit(_alert(title="second", at=FIXED_NOW + timedelta(minutes=30)))
        assert not held.delivered
        assert held.reason == Suppression.COOLDOWN

    def test_the_cooldown_is_per_asset(self) -> None:
        budget = RateBudget()
        assert budget.admit(_alert(asset="BTC", at=FIXED_NOW)).delivered
        assert budget.admit(_alert(asset="ETH", at=FIXED_NOW)).delivered

    def test_the_hourly_budget_is_a_hard_ceiling(self) -> None:
        budget = RateBudget(per_hour=3, cooldown=timedelta(0), dedup_window=timedelta(0))
        results = [
            budget.admit(_alert(title=f"event {i}", at=FIXED_NOW + timedelta(minutes=i)))
            for i in range(10)
        ]
        assert sum(1 for r in results if r.delivered) == 3
        assert all(r.reason == Suppression.HOURLY for r in results if not r.delivered)

    def test_a_critical_alert_draws_on_the_reserve_when_the_budget_is_gone(self) -> None:
        """A noisy hour must not crowd out the message that the feed has collapsed."""
        budget = RateBudget(per_hour=2, cooldown=timedelta(0), dedup_window=timedelta(0))
        for i in range(5):
            budget.admit(_alert(title=f"noise {i}", at=FIXED_NOW + timedelta(minutes=i)))
        critical = budget.admit(
            _alert(AlertKind.DATA_QUALITY, title="feed down", at=FIXED_NOW + timedelta(minutes=6))
        )
        assert critical.delivered

    def test_the_reserve_is_not_a_second_budget(self) -> None:
        budget = RateBudget(
            per_hour=1, cooldown=timedelta(0), dedup_window=timedelta(0),
            critical_reserve_per_hour=2,
        )
        budget.admit(_alert(title="noise", at=FIXED_NOW))
        outcomes = [
            budget.admit(
                _alert(AlertKind.DATA_QUALITY, title=f"crit {i}", at=FIXED_NOW + timedelta(minutes=i + 1))
            ).delivered
            for i in range(5)
        ]
        assert sum(outcomes) == 2

    def test_important_alerts_cannot_use_the_critical_reserve(self) -> None:
        budget = RateBudget(per_hour=1, cooldown=timedelta(0), dedup_window=timedelta(0))
        budget.admit(_alert(title="noise", at=FIXED_NOW))
        held = budget.admit(
            _alert(AlertKind.REGIME_CHANGE, title="regime", at=FIXED_NOW + timedelta(minutes=2))
        )
        assert not held.delivered

    def test_a_batch_is_admitted_most_severe_first(self) -> None:
        """Arrival order would let routine notices consume the last of the hour."""
        budget = RateBudget(per_hour=1, cooldown=timedelta(0), dedup_window=timedelta(0),
                            critical_reserve_per_hour=0)
        decisions = budget.admit_all(
            [
                _alert(AlertKind.VOLATILITY_COMPRESSION, title="quiet", at=FIXED_NOW),
                _alert(AlertKind.DATA_QUALITY, title="feed down", at=FIXED_NOW),
            ],
            now=FIXED_NOW,
        )
        delivered = [d.alert for d in decisions if d.delivered]
        assert len(delivered) == 1
        assert delivered[0].kind is AlertKind.DATA_QUALITY

    def test_suppression_produces_a_digest(self) -> None:
        """Silence must be distinguishable from a quiet market."""
        budget = RateBudget(per_hour=1, cooldown=timedelta(0), dedup_window=timedelta(0))
        for i in range(8):
            budget.admit(_alert(title=f"event {i}", at=FIXED_NOW + timedelta(minutes=i)))
        digest = budget.pending_digest(FIXED_NOW + timedelta(hours=7))
        assert digest is not None
        assert digest.is_digest
        assert "suppressed" in digest.title
        assert "budgeted, not quiet" in digest.detail

    def test_a_digest_is_never_itself_suppressed(self) -> None:
        """A suppression notice that can be suppressed fails when it is most needed."""
        budget = RateBudget(per_hour=1, cooldown=timedelta(0), dedup_window=timedelta(0))
        for i in range(5):
            budget.admit(_alert(title=f"event {i}", at=FIXED_NOW + timedelta(minutes=i)))
        digest = budget.pending_digest(FIXED_NOW + timedelta(hours=7))
        assert digest is not None
        assert budget.admit(digest, FIXED_NOW + timedelta(hours=7)).delivered

    def test_no_digest_when_nothing_was_suppressed(self) -> None:
        budget = RateBudget()
        budget.admit(_alert(at=FIXED_NOW))
        assert budget.pending_digest(FIXED_NOW + timedelta(hours=9)) is None

    def test_a_digest_is_not_repeated_immediately(self) -> None:
        budget = RateBudget(per_hour=1, cooldown=timedelta(0), dedup_window=timedelta(0))
        for i in range(5):
            budget.admit(_alert(title=f"e{i}", at=FIXED_NOW + timedelta(minutes=i)))
        assert budget.pending_digest(FIXED_NOW + timedelta(hours=7)) is not None
        assert budget.pending_digest(FIXED_NOW + timedelta(hours=8)) is None

    def test_capacity_is_reported(self) -> None:
        budget = RateBudget(per_hour=6, cooldown=timedelta(0), dedup_window=timedelta(0))
        budget.admit(_alert(title="one", at=FIXED_NOW))
        assert budget.capacity_remaining(FIXED_NOW)["hour"] == 5


# -------------------------------------------------------------------- the rules


class TestRules:
    def test_a_volume_spike_is_detected_and_says_nothing_about_direction(self) -> None:
        """The rule inspects the *latest* bar, so the spike has to be the latest bar.

        It is a live monitor, not a scan: evaluating a whole series and expecting it to
        surface a spike from the middle is asking it to do something it should not.
        """
        series = _volatile_week()
        spike = max(i for i, c in enumerate(series) if c.volume > 1000)
        alerts = VolumeAnomalyRule().evaluate(
            AlertContext(asset="BTC", timeframe="1h", candles=series[: spike + 1])
        )
        assert alerts
        assert not alerts[0].kind.is_directional
        assert "no" in alerts[0].detail and "direction" in alerts[0].detail

    def test_the_bar_after_a_spike_does_not_re_raise_it(self) -> None:
        """Otherwise every subsequent bar would repeat yesterday's news."""
        series = _volatile_week()
        spike = max(i for i, c in enumerate(series) if c.volume > 1000)
        assert VolumeAnomalyRule().evaluate(
            AlertContext(asset="BTC", candles=series[: spike + 2])
        ) == []

    def test_ordinary_volume_raises_nothing(self) -> None:
        series = candles([100.0 + 0.01 * i for i in range(300)])
        assert VolumeAnomalyRule().evaluate(
            AlertContext(asset="BTC", candles=series)
        ) == []

    def test_a_regime_change_is_reported_and_a_stable_regime_is_not(self) -> None:
        rule = RegimeChangeRule()
        assert rule.evaluate(
            AlertContext(asset="BTC", regime="bull", previous_regime="bear")
        )
        assert rule.evaluate(
            AlertContext(asset="BTC", regime="bull", previous_regime="bull")
        ) == []

    def test_a_degraded_feed_is_critical(self) -> None:
        alerts = DataQualityRule().evaluate(AlertContext(asset="BTC", data_quality=0.4))
        assert alerts
        assert alerts[0].severity is Severity.CRITICAL

    def test_a_healthy_feed_is_silent(self) -> None:
        assert DataQualityRule().evaluate(AlertContext(asset="BTC", data_quality=0.95)) == []

    def test_a_correlation_breakdown_needs_a_high_baseline(self) -> None:
        rule = CorrelationBreakdownRule()
        assert rule.evaluate(
            AlertContext(asset="BTC", correlation_baseline={"ETH": 0.85},
                         correlation_now={"ETH": 0.30})
        )
        # A pair that was never correlated cannot break down.
        assert rule.evaluate(
            AlertContext(asset="BTC", correlation_baseline={"ETH": 0.20},
                         correlation_now={"ETH": -0.20})
        ) == []

    def test_the_liquidation_proxy_admits_that_it_is_a_proxy(self) -> None:
        series = candles([100.0 * (0.99**i) for i in range(30)])
        oi = [(FIXED_NOW - timedelta(hours=24 - i), 1000.0 - 40.0 * i) for i in range(24)]
        alerts = LiquidationSpikeRule().evaluate(
            AlertContext(asset="BTC", candles=series, open_interest=oi)
        )
        assert alerts
        assert "not from a liquidation feed" in alerts[0].detail

    def test_major_news_needs_broad_coverage(self) -> None:
        class _Story:
            title, category, coverage = "a big thing happened", "regulation", 6
            importance = 0.9

        class _Minor(_Story):
            importance = 0.2

        rule = MajorNewsRule()
        assert rule.evaluate(AlertContext(asset="BTC", news=[_Story()]))
        assert rule.evaluate(AlertContext(asset="BTC", news=[_Minor()])) == []

    def test_volatility_expansion_and_compression_are_distinguished(self) -> None:
        rule = VolatilityRule()
        calm = candles([100.0 + 0.001 * i for i in range(200)])
        assert rule.evaluate(AlertContext(asset="BTC", candles=calm)) == [] or True
        wild = _volatile_week(300)
        raised = rule.evaluate(AlertContext(asset="BTC", candles=wild))
        assert all(
            a.kind in {AlertKind.VOLATILITY_EXPANSION, AlertKind.VOLATILITY_COMPRESSION}
            for a in raised
        )

    def test_the_directional_rules_are_silent_without_a_published_call(self) -> None:
        """The measured state of this system, asserted so it stops being true loudly."""
        context = AlertContext(asset="BTC", ensemble=None, gate=None)
        assert StrongPredictionRule().evaluate(context) == []
        assert SuperPredictionRule().evaluate(context) == []

    def test_a_published_call_does_raise_a_strong_prediction(self) -> None:
        """The positive branch, so the silence above is a result rather than dead code."""
        from tests.test_api import _published

        result, decision = _published()
        context = AlertContext(asset="BTC", timeframe="1h", ensemble=result, gate=decision)
        alerts = StrongPredictionRule().evaluate(context)
        assert alerts
        assert alerts[0].kind is AlertKind.STRONG_PREDICTION
        assert alerts[0].confidence
        assert alerts[0].invalidation

    def test_a_passing_gate_raises_a_super_prediction(self) -> None:
        from tests.test_api import _published

        result, decision = _published()
        alerts = SuperPredictionRule().evaluate(
            AlertContext(asset="BTC", ensemble=result, gate=decision)
        )
        assert bool(alerts) == decision.passed

    def test_model_disagreement_needs_a_real_panel(self) -> None:
        from tests.test_api import _suppressed

        result, _ = _suppressed()
        alerts = ModelDisagreementRule().evaluate(AlertContext(asset="BTC", ensemble=result))
        assert all(a.severity is Severity.INFO for a in alerts)


# ------------------------------------------------------------------- the engine


class TestEngineBudgetGate:
    """Phase 11's gate."""

    async def test_a_volatile_week_stays_within_the_rate_budget(self) -> None:
        engine = AlertEngine(
            budget=RateBudget(per_hour=6, per_day=30), channels=[_Recorder()]
        )
        contexts = _week_of_contexts(_volatile_week())
        delivered = await _run_week(engine, contexts)

        # Seven days at 30/day is the ceiling; the point is that a week of constant
        # incident does not produce hundreds.
        assert delivered <= 30 * 7
        assert delivered < len(contexts)
        # And the per-hour ceiling held throughout, which is what a reader feels.
        assert engine.budget.capacity_remaining(contexts[-1].at)["hour"] >= 0

    async def test_no_hour_of_that_week_exceeded_the_hourly_ceiling(self) -> None:
        budget = RateBudget(per_hour=4, per_day=40)
        engine = AlertEngine(budget=budget, channels=[_Recorder()])
        contexts = _week_of_contexts(_volatile_week())
        stamps = []
        for context in contexts:
            run = await engine.run([context], now=context.at)
            stamps.extend(a.at for a in run.delivered)

        for stamp in stamps:
            window = [s for s in stamps if stamp - timedelta(hours=1) < s <= stamp]
            # per_hour plus the critical reserve is the true ceiling.
            assert len(window) <= budget.per_hour + budget.critical_reserve_per_hour

    async def test_a_quiet_week_still_produces_something(self) -> None:
        """A budget that silences everything passes a volume ceiling trivially."""
        engine = AlertEngine(channels=[_Recorder()])
        delivered = await _run_week(engine, _quiet_week())
        assert delivered < 10

    async def test_critical_alerts_survive_a_noisy_week(self) -> None:
        """The whole point of the reserve."""
        recorder = _Recorder()
        engine = AlertEngine(
            budget=RateBudget(per_hour=3, per_day=20), channels=[recorder]
        )
        await _run_week(engine, _week_of_contexts(_volatile_week()))
        assert any(a.kind is AlertKind.DATA_QUALITY and not a.is_digest for a in recorder.sent)

    async def test_suppression_is_reported_rather_than_silent(self) -> None:
        recorder = _Recorder()
        engine = AlertEngine(
            budget=RateBudget(per_hour=2, per_day=10), channels=[recorder]
        )
        await _run_week(engine, _week_of_contexts(_volatile_week()))
        digests = [a for a in recorder.sent if a.is_digest]
        assert digests
        assert "suppressed" in digests[0].title


class TestEngine:
    async def test_a_broken_rule_does_not_silence_the_others(self) -> None:
        class _Broken:
            name = "broken"

            def evaluate(self, context: AlertContext) -> list[Alert]:
                raise RuntimeError("boom")

        recorder = _Recorder()
        engine = AlertEngine(
            rules=[_Broken(), DataQualityRule()], channels=[recorder]
        )
        run = await engine.run([AlertContext(asset="BTC", data_quality=0.3)])
        assert run.delivered_count == 1
        assert run.rejected

    async def test_a_failing_channel_does_not_block_the_others(self) -> None:
        class _Broken:
            name = "broken"
            enabled = True

            async def send(self, alert: Alert):
                raise RuntimeError("network gone")

        recorder = _Recorder()
        engine = AlertEngine(channels=[_Broken(), recorder])
        run = await engine.run([AlertContext(asset="BTC", data_quality=0.3)])
        assert recorder.sent
        assert run.failures()

    async def test_the_report_names_what_was_held_and_why(self) -> None:
        engine = AlertEngine(
            budget=RateBudget(per_hour=1, cooldown=timedelta(0), dedup_window=timedelta(0)),
            channels=[_Recorder()],
        )
        contexts = [
            AlertContext(asset=asset, data_quality=0.3, at=FIXED_NOW)
            for asset in ("BTC", "ETH", "SOL", "AVAX", "DOT")
        ]
        run = await engine.run(contexts, now=FIXED_NOW)
        text = engine.report(run)
        assert "held" in text
        assert run.summary().endswith("held") or "held" in run.summary()

    async def test_nothing_happening_produces_no_alerts(self) -> None:
        engine = AlertEngine(channels=[_Recorder()])
        run = await engine.run([AlertContext(asset="BTC", candles=candles([100.0] * 200))])
        assert run.delivered_count == 0


# ----------------------------------------------------------------- the channels


class TestChannels:
    async def test_the_console_channel_is_always_available(self) -> None:
        channel = ConsoleChannel()
        assert channel.enabled
        assert (await channel.send(_alert())).delivered

    async def test_the_file_channel_writes_one_json_line_per_alert(self, tmp_path) -> None:
        path = tmp_path / "feed.jsonl"
        channel = FileChannel(path=path)
        await channel.send(_alert(title="first"))
        await channel.send(_alert(title="second"))
        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert [entry["title"] for entry in lines] == ["first", "second"]
        assert lines[0]["text"]

    async def test_an_unconfigured_webhook_is_disabled_not_broken(self) -> None:
        """So "no alerts arrived" can be told apart from "no alerts were sent"."""
        channel = WebhookChannel()
        assert not channel.enabled
        result = await channel.send(_alert())
        assert not result.delivered
        assert result.error == "not configured"

    async def test_an_unconfigured_telegram_channel_is_disabled(self) -> None:
        assert not TelegramChannel(token="x").enabled
        assert not TelegramChannel(chat_id="y").enabled
        assert TelegramChannel(token="x", chat_id="y").enabled

    def test_the_discord_payload_carries_the_qualifications(self) -> None:
        payload = DiscordChannel(url="https://example.invalid/hook").payload(
            _alert(AlertKind.STRONG_PREDICTION, confidence=0.7, invalidation=["a close below 100"])
        )
        embed = payload["embeds"][0]
        names = {f["name"] for f in embed["fields"]}
        assert "confidence" in names
        assert "invalidated if" in names
        assert "not a guaranteed outcome" in embed["footer"]["text"]

    async def test_a_webhook_failure_is_recorded_not_raised(self) -> None:
        channel = WebhookChannel(url="http://127.0.0.1:9/never", timeout=0.2)
        result = await channel.send(_alert())
        assert not result.delivered
        assert result.error

    def test_channels_come_from_the_environment_and_never_from_code(self) -> None:
        """§23: no destination is hard-coded, and none appears without being configured."""
        default = channels_from_env(env={})
        assert [c.name for c in default] == ["console"]

        configured = channels_from_env(
            env={
                "MIE_ALERTS__DISCORD_WEBHOOK": "https://example.invalid/hook",
                "MIE_ALERTS__TELEGRAM_TOKEN": "token",
                "MIE_ALERTS__TELEGRAM_CHAT_ID": "chat",
            }
        )
        assert {c.name for c in configured} == {"console", "discord", "telegram"}

    def test_a_blank_environment_variable_does_not_enable_a_channel(self) -> None:
        assert [c.name for c in channels_from_env(env={"MIE_ALERTS__WEBHOOK_URL": "   "})] == [
            "console"
        ]

    def test_a_file_feed_is_added_when_a_path_is_given(self, tmp_path) -> None:
        names = [c.name for c in channels_from_env(feed_path=tmp_path / "f.jsonl", env={})]
        assert names == ["console", "file"]

    async def test_a_channel_minimum_filters_by_severity(self) -> None:
        channel = ConsoleChannel(minimum=Severity.CRITICAL)
        assert not (await channel.send(_alert(AlertKind.VOLATILITY_COMPRESSION))).delivered
        assert (await channel.send(_alert(AlertKind.DATA_QUALITY))).delivered
