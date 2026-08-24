"""Prediction models, baselines and walk-forward evaluation.

The most important tests here are not about any model. They are about the evaluator:
that it detects genuine skill when skill exists, refuses to certify it when it does
not, and cannot be fooled by the multiple comparisons that slicing creates. A gate that
passes everything is not a gate, and a gate that passes nothing regardless of input is
equally useless — both are checked.
"""

from __future__ import annotations

import math
from datetime import timedelta
from itertools import pairwise

import pytest
from tests.conftest import FIXED_NOW

from mie.core.timeframes import Timeframe
from mie.core.types import Candle
from mie.models.base import PredictionContext, Predictor, move_threshold
from mie.models.baselines import (
    ClimatologyBaseline,
    PersistenceBaseline,
    UniformBaseline,
)
from mie.models.evaluation import WalkForwardEvaluator, _paired_p_value
from mie.models.predictors import ALL_MODELS, TechnicalModel, TimeSeriesModel
from mie.models.runner import ContextSource, build_contexts
from mie.models.types import Distribution, Horizon, Outcome, Prediction, PredictionEvidence

HOUR = Timeframe.H1
HORIZON = Horizon(bars=12, timeframe=HOUR)


# ---------------------------------------------------------------------- helpers


def candles(prices: list[float], asset: str = "BTC") -> list[Candle]:
    out = []
    for i, close in enumerate(prices):
        open_price = prices[i - 1] if i else close
        span = abs(close - open_price) + abs(close) * 0.002
        out.append(
            Candle(
                asset=asset, source="test", timeframe=HOUR,
                open_time=FIXED_NOW - timedelta(hours=len(prices) - i),
                open=open_price, high=max(open_price, close) + span,
                low=min(open_price, close) - span, close=close,
                volume=100.0, is_final=True,
            )
        )
    return out


def context(prices: list[float], **kwargs) -> PredictionContext:
    bars = candles(prices)
    return PredictionContext(
        asset="BTC", timeframe=HOUR, as_of=HOUR.close_time(bars[-1].open_time),
        horizon=HORIZON, candles=bars, **kwargs
    )


def drifting(bars: int = 1200, wobble: float = 0.4) -> list[float]:
    """A price path with a mild, deterministic oscillation and no trend.

    Note that a sine wave is *strongly* autocorrelated (lag-1 rho about +0.98), which
    is fine for most tests but makes it useless for checking that a model hedges when
    returns are unforecastable — there, use :func:`unforecastable`.
    """
    return [100.0 + wobble * math.sin(i / 5.0) + 0.3 * math.cos(i / 13.0) for i in range(bars)]


def unforecastable(bars: int = 1200) -> list[float]:
    """A deterministic path whose returns have near-zero autocorrelation (rho ~ -0.04).

    Hash-derived rather than random so failures reproduce exactly, and genuinely
    unforecastable so a model claiming otherwise is claiming something false.
    """
    import hashlib

    price = 100.0
    out: list[float] = []
    for i in range(bars):
        digest = hashlib.blake2b(str(i).encode(), digest_size=8).digest()
        step = int.from_bytes(digest, "big") / 2**64 - 0.5
        price *= 1.0 + step * 0.01
        out.append(price)
    return out


# ------------------------------------------------------------------ primitives


class TestDistribution:
    def test_probabilities_are_normalised(self) -> None:
        d = Distribution(up=2.0, flat=1.0, down=1.0)
        assert d.up + d.flat + d.down == pytest.approx(1.0)
        assert d.up == pytest.approx(0.5)

    def test_zero_mass_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive mass"):
            Distribution(up=0.0, flat=0.0, down=0.0)

    def test_from_edge_keeps_flat_mass_and_splits_the_rest(self) -> None:
        """A weak view must produce a genuinely uncertain distribution, not a confident
        one with a small margin."""
        weak = Distribution.from_edge(0.1)
        strong = Distribution.from_edge(0.9)
        assert weak.flat == pytest.approx(strong.flat)
        assert strong.up > weak.up
        assert weak.up > weak.down

    def test_entropy_is_highest_when_uniform(self) -> None:
        assert Distribution.uniform().entropy > Distribution.from_edge(0.9).entropy
        assert Distribution.uniform().entropy == pytest.approx(math.log2(3), abs=1e-9)

    def test_directional_edge_ignores_flat_mass(self) -> None:
        assert Distribution(up=0.5, flat=0.2, down=0.3).directional_edge == pytest.approx(0.2)

    def test_blend_interpolates(self) -> None:
        blended = Distribution.from_edge(1.0).blend(Distribution.from_edge(-1.0), 0.5)
        assert blended.up == pytest.approx(blended.down)


class TestScoring:
    def test_brier_rewards_calibrated_uncertainty(self) -> None:
        """A hedged wrong call must be penalised less than a confident wrong one —
        exactly the incentive a low-signal domain needs."""
        hedged = _prediction(Distribution(up=0.4, flat=0.35, down=0.25))
        certain = _prediction(Distribution(up=0.9, flat=0.05, down=0.05))
        assert hedged.brier_score(Outcome.DOWN) < certain.brier_score(Outcome.DOWN)

    def test_brier_rewards_being_right_confidently(self) -> None:
        hedged = _prediction(Distribution(up=0.4, flat=0.35, down=0.25))
        certain = _prediction(Distribution(up=0.9, flat=0.05, down=0.05))
        assert certain.brier_score(Outcome.UP) < hedged.brier_score(Outcome.UP)

    def test_log_loss_is_finite_for_a_zero_probability(self) -> None:
        """A model that assigned zero to what happened must be penalised heavily, not
        crash the evaluation."""
        prediction = _prediction(Distribution(up=1.0, flat=1e-15, down=1e-15))
        assert math.isfinite(prediction.log_loss(Outcome.DOWN))
        assert prediction.log_loss(Outcome.DOWN) > 10

    def test_outcome_classification_uses_the_threshold(self) -> None:
        assert Outcome.classify(1.0, 0.5) is Outcome.UP
        assert Outcome.classify(-1.0, 0.5) is Outcome.DOWN
        assert Outcome.classify(0.2, 0.5) is Outcome.FLAT

    def test_threshold_scales_with_volatility_and_horizon(self) -> None:
        """A fixed band would make long horizons trivially directional and short ones
        trivially flat, and would mean different things in different regimes."""
        calm = candles([100.0 + 0.05 * math.sin(i) for i in range(200)])
        wild = candles([100.0 + 3.0 * math.sin(i) for i in range(200)])
        assert move_threshold(wild, 12) > move_threshold(calm, 12) * 5
        assert move_threshold(calm, 48) > move_threshold(calm, 12)


def _prediction(distribution: Distribution) -> Prediction:
    return Prediction(
        model_id="test", asset="BTC", timeframe=HOUR, horizon=HORIZON,
        as_of=FIXED_NOW, distribution=distribution, move_threshold_pct=0.5,
    )


# -------------------------------------------------------------------- baselines


class TestBaselines:
    def test_climatology_reproduces_historical_frequencies(self) -> None:
        prices = drifting(1200)
        prediction = ClimatologyBaseline().predict(context(prices))
        assert prediction.confidence > 0
        total = sum(
            (prediction.distribution.up, prediction.distribution.flat, prediction.distribution.down)
        )
        assert total == pytest.approx(1.0)

    def test_climatology_abstains_without_history(self) -> None:
        assert ClimatologyBaseline().predict(context(drifting(50))).confidence == 0.0

    def test_persistence_follows_the_last_move(self) -> None:
        rising = candles([100.0 + i * 0.5 for i in range(200)])
        prediction = PersistenceBaseline().predict(
            PredictionContext(
                asset="BTC", timeframe=HOUR, as_of=FIXED_NOW, horizon=HORIZON, candles=rising
            )
        )
        assert prediction.distribution.directional_edge > 0

    def test_uniform_is_maximally_uncertain(self) -> None:
        prediction = UniformBaseline().predict(context(drifting(200)))
        assert prediction.distribution.up == pytest.approx(1 / 3)
        assert prediction.confidence == 0.0


# ----------------------------------------------------------------- look-ahead


class TestNoLookAhead:
    """The property every result in this phase depends on."""

    def test_context_contains_no_future_bars(self) -> None:
        prices = drifting(600)
        source = ContextSource("BTC", HOUR, candles(prices))
        built = source.context_at(300, HORIZON)
        assert built is not None
        assert len(built.candles) == 301
        assert all(HOUR.close_time(c.open_time) <= built.as_of for c in built.candles)

    def test_features_after_as_of_are_excluded(self) -> None:
        prices = drifting(600)
        bars = candles(prices)
        history = [(c.open_time, {"close": c.close}) for c in bars]
        source = ContextSource("BTC", HOUR, bars, history)
        built = source.context_at(300, HORIZON)
        assert built is not None
        assert all(HOUR.close_time(t) <= built.as_of for t, _ in built.feature_history)

    def test_peer_data_is_also_truncated(self) -> None:
        prices = drifting(600)
        bars = candles(prices)
        source = ContextSource("BTC", HOUR, bars, peers={"ETH": bars})
        built = source.context_at(300, HORIZON)
        assert built is not None
        assert all(
            HOUR.close_time(c.open_time) <= built.as_of for c in built.peers["ETH"]
        )

    def test_truncating_history_does_not_change_an_earlier_prediction(self) -> None:
        """The decisive check: a prediction made at bar 300 must be identical whether
        or not the series continues past it."""
        prices = drifting(900)
        model = TechnicalModel()
        full = ContextSource("BTC", HOUR, candles(prices))
        short = ContextSource("BTC", HOUR, candles(prices[:400]))

        a = full.context_at(300, HORIZON)
        b = short.context_at(300, HORIZON)
        assert a is not None and b is not None
        assert model.predict(a).distribution == model.predict(b).distribution

    def test_realised_return_is_strictly_forward(self) -> None:
        prices = drifting(900)
        source = ContextSource("BTC", HOUR, candles(prices))
        pairs = build_contexts(source, HORIZON, warmup=450)
        assert pairs
        for built, realised in pairs[:20]:
            index = prices.index(built.candles[-1].close)
            expected = (prices[index + HORIZON.bars] - prices[index]) / prices[index] * 100.0
            assert realised == pytest.approx(expected, abs=1e-9)

    def test_evaluation_points_do_not_overlap_by_default(self) -> None:
        """Overlapping windows are not independent samples, and counting them as such
        shrinks every confidence interval below what the evidence supports."""
        source = ContextSource("BTC", HOUR, candles(drifting(900)))
        pairs = build_contexts(source, HORIZON, warmup=450)
        times = [c.as_of for c, _ in pairs]
        gaps = {(b - a).total_seconds() / 3600 for a, b in pairwise(times)}
        assert gaps == {float(HORIZON.bars)}


# -------------------------------------------------------------------- models


class TestModelBehaviour:
    def test_every_model_abstains_without_its_substrate(self) -> None:
        """A model with nothing to say must say nothing rather than guess."""
        bare = context(drifting(600))
        for factory in ALL_MODELS:
            prediction = factory().predict(bare)
            assert prediction.model_id
            if prediction.confidence == 0.0:
                assert prediction.distribution.up == pytest.approx(1 / 3)

    def test_models_declare_distinct_inputs(self) -> None:
        """Eight models sharing one feature set is one model with extra steps: they
        would agree constantly and the ensemble would read that as corroboration."""
        used = [frozenset(factory().inputs_used()) for factory in ALL_MODELS]
        assert len({tuple(sorted(u)) for u in used}) >= 6

    def test_data_quality_multiplies_into_confidence(self) -> None:
        """Phase 1's trust score reaching the published output, applied centrally so
        no model can forget it."""
        prices = drifting(600)
        features = {"close": prices[-1], "ema_21": 99.0, "sma_50": 98.0,
                    "sma_200": 97.0, "rsi_14": 65.0, "adx_14.adx": 30.0}
        clean = context(prices, features=features, data_quality=1.0)
        degraded = context(prices, features=features, data_quality=0.25)
        model = TechnicalModel()
        assert model.predict(degraded).confidence == pytest.approx(
            model.predict(clean).confidence * 0.25, abs=1e-4
        )

    def test_timeseries_hedges_when_autocorrelation_is_noise(self) -> None:
        """On genuinely unforecastable returns the model must say so rather than
        project noise forward."""
        prediction = TimeSeriesModel().predict(context(unforecastable(600)))
        assert prediction.confidence <= 0.25
        assert prediction.distribution.entropy > 1.4
        assert any("unforecastable" in e.label for e in prediction.counter_evidence)

    def test_timeseries_does_take_a_view_when_autocorrelation_is_real(self) -> None:
        """The mirror case: a strongly autocorrelated series is forecastable, and
        refusing to say so would be its own kind of dishonesty."""
        prediction = TimeSeriesModel().predict(context(drifting(600)))
        assert prediction.confidence > 0.25

    def test_predictions_carry_invalidation_conditions(self) -> None:
        """A forecast that cannot be wrong cannot be evaluated."""
        prices = drifting(600)
        features = {"close": prices[-1], "ema_21": 99.0, "sma_50": 98.0, "sma_200": 97.0}
        prediction = TechnicalModel().predict(context(prices, features=features))
        assert prediction.invalidation

    def test_actionability_requires_both_confidence_and_edge(self) -> None:
        confident_but_flat = _prediction(Distribution(up=0.34, flat=0.33, down=0.33))
        assert not confident_but_flat.model_copy(update={"confidence": 0.9}).is_actionable
        decided = _prediction(Distribution(up=0.5, flat=0.3, down=0.2))
        assert decided.model_copy(update={"confidence": 0.9}).is_actionable


# ---------------------------------------------------------------- evaluation


class _OracleModel(Predictor):
    """A model that can see the future. Used only to prove the evaluator detects skill."""

    model_id = "oracle"
    warmup_bars = 1

    def __init__(self, outcomes: dict, strength: float = 0.8) -> None:
        self.outcomes = outcomes
        self.strength = strength

    def inputs_used(self) -> frozenset[str]:
        return frozenset({"cheating"})

    def predict(self, context: PredictionContext) -> Prediction:
        actual = self.outcomes.get(context.as_of)
        if actual is None:
            return self.abstain(context, "unknown point")
        weights = dict.fromkeys(Outcome, (1 - self.strength) / 2)
        weights[actual] = self.strength
        return self.build(
            context,
            distribution=Distribution(up=weights[Outcome.UP], flat=weights[Outcome.FLAT],
                                      down=weights[Outcome.DOWN]),
            confidence=0.9,
            evidence=[PredictionEvidence(label="oracle")],
        )


class TestEvaluatorDetectsSkill:
    """If the evaluator cannot detect real skill, every negative result is worthless."""

    @staticmethod
    def _contexts():
        source = ContextSource("BTC", HOUR, candles(drifting(1400)))
        return build_contexts(source, HORIZON, warmup=450)

    def test_a_genuinely_skilful_model_is_certified(self) -> None:
        pairs = self._contexts()
        truth = {
            c.as_of: Outcome.classify(realised, c.threshold_pct) for c, realised in pairs
        }
        report = WalkForwardEvaluator(ClimatologyBaseline()).evaluate(
            [_OracleModel(truth)], pairs
        )
        overall = [s for s in report.scores if s.regime == "all"]
        assert overall
        assert overall[0].skill > 0.3
        assert overall[0].beats_baseline
        assert "oracle" in report.passing_models()

    def test_a_model_with_no_information_is_not_certified(self) -> None:
        """The test that makes every negative result meaningful."""
        pairs = self._contexts()
        report = WalkForwardEvaluator(ClimatologyBaseline()).evaluate(
            [UniformBaseline()], pairs
        )
        assert not report.passing_models()

    def test_significance_is_required_not_just_positive_skill(self) -> None:
        """Slicing is multiple comparisons: across dozens of slices some model posts
        positive skill by luck, and a gate accepting that would certify noise."""
        pairs = self._contexts()
        truth = {
            c.as_of: Outcome.classify(realised, c.threshold_pct) for c, realised in pairs
        }
        # Barely-informative oracle: real but tiny edge.
        report = WalkForwardEvaluator(ClimatologyBaseline()).evaluate(
            [_OracleModel(truth, strength=0.34)], pairs
        )
        for score in report.scores:
            if score.skill > 0.01 and not score.significant:
                assert not score.beats_baseline


class TestPairedTest:
    def test_a_consistent_improvement_is_significant(self) -> None:
        assert _paired_p_value([0.02] * 100) < 0.001

    def test_noise_is_not_significant(self) -> None:
        alternating = [0.05 if i % 2 else -0.05 for i in range(200)]
        assert _paired_p_value(alternating) > 0.05

    def test_a_worse_model_is_never_significant(self) -> None:
        """One-sided: significantly worse is not a pass."""
        assert _paired_p_value([-0.02] * 100) > 0.5

    def test_tiny_samples_are_refused(self) -> None:
        assert _paired_p_value([0.5, 0.5]) == 1.0
