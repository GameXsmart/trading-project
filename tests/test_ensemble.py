"""Calibration, agreement, confidence and the super-prediction gate.

The Phase 7 gate is three claims, and each is tested here directly:

* reliability is within tolerance for what the system publishes;
* deliberately induced disagreement produces *no* super prediction;
* degrading the input feed measurably lowers published confidence.

As in Phase 6, every suppression test is paired with a test that the same machinery
*does* fire when the evidence is there. A gate that refuses everything is not a strict
gate — it is a broken one, and the two are indistinguishable without both directions.
"""

from __future__ import annotations

import hashlib
from datetime import timedelta
from itertools import pairwise

import pytest
from tests.conftest import FIXED_NOW
from tests.test_models import HORIZON, HOUR, candles, context, drifting

from mie.ensemble.agreement import (
    independence_weights,
    measure_agreement,
    overlap_matrix,
)
from mie.ensemble.calibration import (
    CalibrationCurve,
    CalibrationLibrary,
    CalibrationRecord,
    _pool_adjacent_violators,
    classwise_ece,
    reliability_diagram,
)
from mie.ensemble.confidence import confidence_from
from mie.ensemble.gate import SuperPredictionGate
from mie.ensemble.meta import EnsembleModel, SkillWeights, _linear_pool
from mie.models.base import PredictionContext, Predictor
from mie.models.baselines import ClimatologyBaseline
from mie.models.evaluation import ScoredPrediction, WalkForwardEvaluator
from mie.models.predictors import ALL_MODELS
from mie.models.runner import ContextSource, build_contexts
from mie.models.types import (
    Distribution,
    Outcome,
    Prediction,
    PredictionEvidence,
)

# --------------------------------------------------------------------- helpers


def _uniform(index: int, salt: str = "") -> float:
    """Deterministic pseudo-uniform in [0, 1).

    Hash-derived rather than ``random`` so a failure reproduces exactly. Tests about
    statistical machinery that cannot be replayed are tests nobody can debug.
    """
    digest = hashlib.blake2b(f"{salt}:{index}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


def _draw(index: int, probabilities: dict[Outcome, float], salt: str) -> Outcome:
    """Sample an outcome deterministically from a target distribution."""
    draw = _uniform(index, salt)
    cumulative = 0.0
    for outcome in Outcome:
        cumulative += probabilities[outcome]
        if draw < cumulative:
            return outcome
    return Outcome.DOWN


def _prediction(
    distribution: Distribution,
    *,
    model_id: str = "m",
    regime: str = "range_low_vol",
    index: int = 0,
    confidence: float = 0.5,
    data_quality: float = 1.0,
) -> Prediction:
    return Prediction(
        model_id=model_id,
        asset="BTC",
        timeframe=HOUR,
        horizon=HORIZON,
        as_of=FIXED_NOW + timedelta(hours=index),
        distribution=distribution,
        confidence=confidence,
        regime=regime,
        data_quality=data_quality,
        move_threshold_pct=0.4,
    )


def _scored(
    stated: Distribution,
    truth: dict[Outcome, float],
    count: int,
    salt: str,
    *,
    model_id: str = "m",
    regime: str = "range_low_vol",
    start: int = 0,
) -> list[ScoredPrediction]:
    """A run of predictions all stating ``stated`` while the world does ``truth``."""
    entries = []
    for i in range(count):
        actual = _draw(start + i, truth, salt)
        entries.append(
            ScoredPrediction(
                _prediction(stated, model_id=model_id, regime=regime, index=start + i),
                actual,
                1.0 if actual is Outcome.UP else -1.0 if actual is Outcome.DOWN else 0.0,
            )
        )
    return entries


class _FamilyModel(Predictor):
    """A stand-in for one independent model family, with a fixed view."""

    warmup_bars = 1

    def __init__(self, name: str, edge: float, substrate: str | None = None) -> None:
        self.model_id = name
        self.edge = edge
        self.substrate = substrate or name

    def inputs_used(self) -> frozenset[str]:
        return frozenset({self.substrate})

    def predict(self, ctx: PredictionContext) -> Prediction:
        if self.edge == 0.0:
            return self.abstain(ctx, "no view")
        return self.build(
            ctx,
            distribution=Distribution.from_edge(self.edge),
            confidence=0.6,
            evidence=[PredictionEvidence(label=self.model_id, contribution=self.edge)],
        )


def _panel(edges: list[float]) -> list[Predictor]:
    return [_FamilyModel(f"family{i}", edge) for i, edge in enumerate(edges)]


def _full_weights(models: list[Predictor], skill: float = 0.05) -> SkillWeights:
    """Skill weights as if every model had passed the Phase 6 gate."""
    return SkillWeights(
        weights={(m.model_id, "all"): skill for m in models},
        samples={(m.model_id, "all"): 500 for m in models},
        regime_samples={"range_low_vol": 500, "unknown": 500, "range_high_vol": 500},
    )


def _full_calibration(models: list[Predictor], regimes: list[str]) -> CalibrationLibrary:
    """A library asserting every model is calibrated in every listed regime."""
    library = CalibrationLibrary()
    for model in models:
        for regime in regimes:
            library.add(
                CalibrationRecord(
                    model_id=model.model_id,
                    regime=regime,
                    curves={o: CalibrationCurve.identity() for o in Outcome},
                    samples=500,
                    holdout_samples=200,
                    ece_before=0.05,
                    ece_after=0.02,
                    improved=True,
                )
            )
    return library


def _ctx(**kwargs) -> PredictionContext:
    return context(drifting(300), **kwargs)


# ------------------------------------------------------------------ isotonic


class TestPoolAdjacentViolators:
    def test_already_monotone_data_is_unchanged(self) -> None:
        values = [0.1, 0.3, 0.5, 0.9]
        assert _pool_adjacent_violators(values, [1.0] * 4) == pytest.approx(values)

    def test_a_violation_is_pooled_to_the_weighted_mean(self) -> None:
        fitted = _pool_adjacent_violators([0.8, 0.2], [1.0, 1.0])
        assert fitted == pytest.approx([0.5, 0.5])

    def test_a_decreasing_series_collapses_to_one_level(self) -> None:
        fitted = _pool_adjacent_violators([0.9, 0.6, 0.3, 0.0], [1.0] * 4)
        assert fitted == pytest.approx([0.45] * 4)

    def test_output_is_never_decreasing(self) -> None:
        values = [_uniform(i, "pav") for i in range(50)]
        fitted = _pool_adjacent_violators(values, [1.0] * 50)
        assert all(b >= a - 1e-12 for a, b in pairwise(fitted))

    def test_weights_pull_the_pooled_value(self) -> None:
        """A heavily-weighted point dominates the block it is pooled into."""
        fitted = _pool_adjacent_violators([1.0, 0.0], [1.0, 9.0])
        assert fitted[0] == pytest.approx(0.1)


class TestCalibrationCurve:
    def test_identity_passes_probabilities_through(self) -> None:
        curve = CalibrationCurve.identity()
        assert curve.is_identity
        assert curve.apply(0.42) == pytest.approx(0.42)

    def test_interpolates_between_breakpoints(self) -> None:
        curve = CalibrationCurve(xs=(0.0, 1.0), ys=(0.0, 0.5))
        assert curve.apply(0.5) == pytest.approx(0.25)

    def test_clamps_outside_the_fitted_range(self) -> None:
        curve = CalibrationCurve(xs=(0.2, 0.8), ys=(0.3, 0.6))
        assert curve.apply(0.0) == pytest.approx(0.3)
        assert curve.apply(1.0) == pytest.approx(0.6)

    def test_learns_a_shrinking_map_from_an_overconfident_forecaster(self) -> None:
        """Says 0.9, is right 0.5 of the time: the curve should learn to shrink it."""
        pairs = [(0.9, 1.0 if _uniform(i, "shrink") < 0.5 else 0.0) for i in range(400)]
        pairs += [(0.1, 1.0 if _uniform(i, "shrink2") < 0.1 else 0.0) for i in range(400)]
        pairs += [(0.5, 1.0 if _uniform(i, "shrink3") < 0.3 else 0.0) for i in range(400)]
        curve = CalibrationCurve.fit(pairs)
        assert curve.apply(0.9) < 0.65
        assert curve.apply(0.9) > curve.apply(0.1)

    def test_refuses_to_fit_a_tiny_sample(self) -> None:
        assert CalibrationCurve.fit([(0.5, 1.0)] * 5).is_identity


# --------------------------------------------------------------- reliability


class TestReliability:
    def test_a_well_calibrated_forecaster_is_within_tolerance(self) -> None:
        """The literal Phase 7 gate: of everything published at 70%, 70% occurs."""
        pairs = [(0.7, 1.0 if _uniform(i, "rel") < 0.7 else 0.0) for i in range(2000)]
        diagram = reliability_diagram(pairs)
        assert diagram.within_tolerance()
        assert diagram.statistically_calibrated()
        assert diagram.ece < 0.05

    def test_an_overconfident_forecaster_fails_the_tolerance(self) -> None:
        pairs = [(0.7, 1.0 if _uniform(i, "over") < 0.4 else 0.0) for i in range(2000)]
        diagram = reliability_diagram(pairs)
        assert not diagram.within_tolerance()
        assert not diagram.statistically_calibrated()
        assert diagram.ece > 0.2

    def test_an_empty_diagram_does_not_claim_calibration(self) -> None:
        """Vacuous truth is the classic way an 'all bins pass' check passes."""
        assert not reliability_diagram([]).within_tolerance()
        assert not reliability_diagram([(0.5, 1.0)]).within_tolerance()

    def test_a_small_bin_is_not_declared_miscalibrated_by_sampling_noise(self) -> None:
        """15 observations cannot distinguish 0.7 from 0.5; the interval must say so."""
        pairs = [(0.7, 1.0 if i < 9 else 0.0) for i in range(15)]
        entry = next(b for b in reliability_diagram(pairs).bins if b.count == 15)
        assert entry.contains_nominal
        assert not entry.within_tolerance

    def test_classwise_ece_sees_error_in_classes_the_model_did_not_name(self) -> None:
        """Scoring only the top class would let the other two be arbitrarily wrong."""
        stated = Distribution(up=0.5, flat=0.4, down=0.1)
        entries = [
            (stated, _draw(i, {Outcome.UP: 0.5, Outcome.FLAT: 0.1, Outcome.DOWN: 0.4}, "cw"))
            for i in range(1500)
        ]
        assert classwise_ece(entries) > 0.15


# --------------------------------------------------------------- calibration


#: Stated up-probabilities the biased fixture emits.
_LEVELS = (0.55, 0.65, 0.75, 0.85)


def _overstated(level: float) -> Distribution:
    return Distribution(up=level, flat=(1 - level) / 2, down=(1 - level) / 2)


def _truth_for(level: float, shrink: float) -> dict[Outcome, float]:
    """What actually happens when the model states ``level``.

    ``shrink`` of 1.0 is a perfectly calibrated model; below that the model overstates
    its edge by a constant factor, which is the common real miscalibration and the one
    a monotone map can fix.
    """
    up = 0.5 + (level - 0.5) * shrink
    return {Outcome.UP: up, Outcome.FLAT: (1 - up) / 2, Outcome.DOWN: (1 - up) / 2}


def _miscalibrated(
    count: int, shrink: float, salt: str, model_id: str, regime: str = "range_low_vol"
) -> list[ScoredPrediction]:
    """Scored predictions from a model whose stated edge is ``shrink`` times real."""
    entries = []
    for i in range(count):
        level = _LEVELS[i % len(_LEVELS)]
        actual = _draw(i, _truth_for(level, shrink), salt)
        entries.append(
            ScoredPrediction(
                _prediction(_overstated(level), model_id=model_id, regime=regime, index=i),
                actual,
                1.0 if actual is Outcome.UP else -1.0 if actual is Outcome.DOWN else 0.0,
            )
        )
    return entries


class TestCalibrationLibrary:
    @staticmethod
    def _biased(count: int = 800, model_id: str = "biased") -> list[ScoredPrediction]:
        """A model that overstates its edge by a factor of three."""
        return _miscalibrated(count, 0.35, "biased", model_id)

    def test_a_systematically_biased_model_gets_a_kept_curve(self) -> None:
        library = CalibrationLibrary()
        library.fit(self._biased())
        record = library.record_for("biased", "range_low_vol")
        assert record is not None
        assert record.improved
        assert record.is_usable
        assert record.improvement > 0.005

    def test_calibration_moves_the_distribution_toward_the_truth(self) -> None:
        library = CalibrationLibrary()
        library.fit(self._biased())
        stated = _overstated(0.85)
        calibrated, record = library.calibrate("biased", "range_low_vol", stated)
        assert record is not None and record.is_usable
        assert calibrated.up < stated.up
        # The model states 0.85 when the truth is 0.5 + 0.35*0.35 = 0.6225.
        assert calibrated.up == pytest.approx(0.6225, abs=0.12)

    def test_calibration_preserves_the_ordering_of_the_stated_levels(self) -> None:
        """Isotonic corrects the scale, never the ranking."""
        library = CalibrationLibrary()
        library.fit(self._biased())
        mapped = [
            library.calibrate("biased", "range_low_vol", _overstated(level))[0].up
            for level in _LEVELS
        ]
        assert all(b >= a - 1e-9 for a, b in pairwise(mapped))

    def test_an_already_calibrated_model_keeps_the_identity_map(self) -> None:
        """The default is to leave a model's numbers alone."""
        library = CalibrationLibrary()
        library.fit(_miscalibrated(800, 1.0, "honest", "honest"))
        record = library.record_for("honest", "range_low_vol")
        assert record is not None
        assert not record.improved
        assert not record.is_usable
        stated = _overstated(0.75)
        calibrated, _ = library.calibrate("honest", "range_low_vol", stated)
        assert calibrated.up == pytest.approx(stated.up)

    def test_too_little_data_produces_no_curve_at_all(self) -> None:
        library = CalibrationLibrary()
        library.fit(self._biased(count=40))
        record = library.record_for("biased", "range_low_vol")
        assert record is not None
        assert record.curves == {}
        assert not record.is_usable

    def test_abstentions_are_not_calibrated(self) -> None:
        """A model with nothing to say offers no probabilistic claim to correct."""
        entries = [
            ScoredPrediction(
                _prediction(Distribution.uniform(), model_id="quiet", confidence=0.0, index=i),
                Outcome.UP,
                1.0,
            )
            for i in range(400)
        ]
        library = CalibrationLibrary()
        library.fit(entries)
        assert library.record_for("quiet", "range_low_vol") is None

    def test_applying_a_record_to_its_own_fitting_window_raises(self) -> None:
        """The look-ahead defence: calibrating the past with the future must fail."""
        library = CalibrationLibrary()
        library.fit(self._biased())
        record = library.record_for("biased", "range_low_vol")
        assert record is not None and record.fitted_through is not None
        with pytest.raises(ValueError, match="calibrate the past"):
            record.apply(_overstated(0.85), as_of=record.fitted_through)
        later = record.fitted_through + timedelta(hours=1)
        record.apply(_overstated(0.85), as_of=later)

    def test_regime_records_are_distinguished_from_pooled_ones(self) -> None:
        library = CalibrationLibrary()
        library.fit(self._biased())
        assert library.has_regime_record("biased", "range_low_vol")
        assert not library.has_regime_record("biased", "downtrend_high_vol")
        # A pooled record still answers the lookup, but says which regime it came from.
        fallback = library.record_for("biased", "downtrend_high_vol")
        assert fallback is not None
        assert fallback.regime == "all"

    def test_a_calibrated_distribution_still_sums_to_one(self) -> None:
        library = CalibrationLibrary()
        library.fit(self._biased())
        calibrated, _ = library.calibrate("biased", "range_low_vol", _overstated(0.85))
        assert calibrated.up + calibrated.flat + calibrated.down == pytest.approx(1.0)


# ---------------------------------------------------------------- agreement


class TestIndependenceWeighting:
    def test_disjoint_models_each_carry_a_full_vote(self) -> None:
        inputs = {f"m{i}": frozenset({f"s{i}"}) for i in range(8)}
        assert set(independence_weights(inputs).values()) == {1.0}

    def test_two_identical_models_share_one_vote(self) -> None:
        inputs = {"a": frozenset({"price"}), "b": frozenset({"price"})}
        weights = independence_weights(inputs)
        assert weights == {"a": 0.5, "b": 0.5}
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_partial_overlap_is_discounted_partially(self) -> None:
        inputs = {
            "a": frozenset({"price", "features"}),
            "b": frozenset({"features", "news"}),
        }
        weights = independence_weights(inputs)
        assert 0.5 < weights["a"] < 1.0

    def test_models_declaring_nothing_are_treated_as_dependent(self) -> None:
        """Two models with no declared inputs are unfalsifiably identical."""
        inputs = {"a": frozenset(), "b": frozenset()}
        assert independence_weights(inputs) == {"a": 0.5, "b": 0.5}

    def test_the_shipped_panel_is_genuinely_independent(self) -> None:
        """§12 asks for independent families; this is the audit, not an assumption."""
        inputs = {m().model_id: m().inputs_used() for m in ALL_MODELS}
        weights = independence_weights(inputs)
        assert len(inputs) == 8
        assert min(weights.values()) > 0.6
        assert max(overlap_matrix(inputs).values()) <= 0.5


class TestAgreement:
    def test_a_unanimous_panel_reaches_full_consensus(self) -> None:
        inputs = {f"m{i}": frozenset({f"s{i}"}) for i in range(8)}
        predictions = [
            _prediction(Distribution.from_edge(0.3), model_id=f"m{i}") for i in range(8)
        ]
        report = measure_agreement(predictions, inputs)
        assert report.majority is Outcome.UP
        assert report.agreeing == 8
        assert report.effective_agreement == pytest.approx(8.0)
        assert report.consensus_share == pytest.approx(1.0)
        assert not report.is_split

    def test_an_evenly_split_panel_chooses_no_direction(self) -> None:
        inputs = {f"m{i}": frozenset({f"s{i}"}) for i in range(8)}
        predictions = [
            _prediction(Distribution.from_edge(0.3 if i < 4 else -0.3), model_id=f"m{i}")
            for i in range(8)
        ]
        report = measure_agreement(predictions, inputs)
        assert report.majority is None
        assert report.is_split

    def test_a_narrow_majority_still_counts_as_split(self) -> None:
        inputs = {f"m{i}": frozenset({f"s{i}"}) for i in range(8)}
        predictions = [
            _prediction(Distribution.from_edge(0.3 if i < 5 else -0.3), model_id=f"m{i}")
            for i in range(8)
        ]
        report = measure_agreement(predictions, inputs)
        assert report.majority is Outcome.UP
        assert report.agreeing == 5
        assert report.consensus_share == pytest.approx(0.625)
        assert report.is_split

    def test_six_agreeing_clones_are_not_six_independent_votes(self) -> None:
        """The double-counting this module exists to prevent."""
        inputs = {f"m{i}": frozenset({"price"}) for i in range(6)}
        predictions = [
            _prediction(Distribution.from_edge(0.3), model_id=f"m{i}") for i in range(6)
        ]
        report = measure_agreement(predictions, inputs)
        assert report.agreeing == 6
        assert report.effective_agreement == pytest.approx(1.0, abs=0.01)

    def test_abstentions_and_shrugs_are_not_votes(self) -> None:
        inputs = {f"m{i}": frozenset({f"s{i}"}) for i in range(3)}
        predictions = [
            _prediction(Distribution.from_edge(0.3), model_id="m0"),
            _prediction(Distribution.uniform(), model_id="m1", confidence=0.0),
            # Confident but essentially neutral: an opinion of no consequence.
            _prediction(Distribution.from_edge(0.005), model_id="m2"),
        ]
        report = measure_agreement(predictions, inputs)
        assert report.participants == 1
        assert sorted(report.abstained) == ["m1", "m2"]


# --------------------------------------------------------------- confidence


def _confident(**overrides) -> object:
    base = {
        "best_skill": 0.05,
        "skill_is_significant": True,
        "has_calibration": True,
        "calibration_in_regime": True,
        "calibration_improvement": 0.02,
        "consensus_share": 1.0,
        "effective_agreement": 8.0,
        "family_count": 8,
        "data_quality": 1.0,
        "evaluation_samples": 500,
        "regime_samples": 500,
    }
    base.update(overrides)
    return confidence_from(**base)  # type: ignore[arg-type]


class TestConfidence:
    def test_everything_measured_and_agreeing_is_publishable(self) -> None:
        factors = _confident()
        assert factors.publishable
        assert factors.value == pytest.approx(0.85)

    def test_confidence_is_capped_below_certainty(self) -> None:
        """Never claim certainty about future market movements — §21."""
        assert _confident(best_skill=10.0).value <= 0.85

    def test_no_demonstrated_skill_means_no_confidence(self) -> None:
        factors = _confident(skill_is_significant=False, best_skill=0.0)
        assert factors.value == 0.0
        assert not factors.publishable
        assert factors.limiting_factor == "skill"
        assert any("skill" in note for note in factors.notes)

    def test_no_calibration_record_means_no_confidence(self) -> None:
        factors = _confident(has_calibration=False)
        assert factors.value == 0.0
        assert factors.limiting_factor == "calibration"

    def test_pooled_calibration_is_a_heavy_discount_not_a_pass(self) -> None:
        factors = _confident(calibration_in_regime=False)
        assert 0 < factors.value < _confident().value
        assert any("regime" in note for note in factors.notes)

    def test_degrading_the_feed_lowers_published_confidence(self) -> None:
        """The Phase 7 gate, and §20 made visible."""
        values = [_confident(data_quality=q).value for q in (1.0, 0.8, 0.5, 0.2)]
        assert all(b < a for a, b in pairwise(values))
        assert values[-1] < 0.35

    def test_disagreement_lowers_confidence(self) -> None:
        assert _confident(consensus_share=0.55, effective_agreement=4.0).value < _confident().value

    def test_a_thin_sample_lowers_confidence(self) -> None:
        thin = _confident(evaluation_samples=40)
        assert thin.value < _confident().value
        assert thin.limiting_factor == "sample"

    def test_an_unfamiliar_regime_lowers_confidence(self) -> None:
        unfamiliar = _confident(regime_samples=10)
        assert unfamiliar.value < _confident().value
        assert unfamiliar.limiting_factor == "regime_familiarity"

    def test_the_explanation_names_the_limiting_factor(self) -> None:
        text = _confident(data_quality=0.3).explain()
        assert "data_quality" in text
        assert "insufficient evidence" in text


# ----------------------------------------------------------------- ensemble


class TestSkillWeights:
    def test_only_slices_passing_the_phase_6_gate_earn_a_weight(self) -> None:
        pairs = build_contexts(
            ContextSource("BTC", HOUR, candles(drifting(1400))), HORIZON, warmup=450
        )
        report = WalkForwardEvaluator(ClimatologyBaseline()).evaluate(
            [m() for m in ALL_MODELS], pairs
        )
        weights = SkillWeights.from_report(report)
        for (model_id, regime), weight in weights.weights.items():
            score = next(
                s for s in report.scores if s.model_id == model_id and s.regime == regime
            )
            assert score.beats_baseline
            assert weight > 0

    def test_an_empty_report_yields_no_weights(self) -> None:
        weights = SkillWeights.from_report(object())
        assert not weights.any_skill
        assert weights.weight_for("anything", "any_regime") == 0.0
        assert "none demonstrated skill" in weights.summary()

    def test_a_regime_weight_beats_the_pooled_fallback(self) -> None:
        weights = SkillWeights(weights={("m", "all"): 0.05, ("m", "chop"): 0.02})
        assert weights.weight_for("m", "chop") == 0.02
        assert weights.weight_for("m", "trend") == 0.05


class TestLinearPool:
    def test_weights_shift_the_pool(self) -> None:
        pooled = _linear_pool(
            {"a": Distribution(up=1.0, flat=0.0, down=0.0), "b": Distribution.uniform()},
            {"a": 3.0, "b": 1.0},
        )
        assert pooled.up == pytest.approx(0.75 + 0.25 / 3)

    def test_the_pool_is_never_sharper_than_its_inputs(self) -> None:
        """Linear pooling is conservative by construction; the log pool is not."""
        sharp_up = Distribution(up=0.9, flat=0.05, down=0.05)
        sharp_down = Distribution(up=0.05, flat=0.05, down=0.9)
        pooled = _linear_pool({"a": sharp_up, "b": sharp_down}, {"a": 1.0, "b": 1.0})
        assert pooled.entropy > sharp_up.entropy

    def test_zero_total_weight_falls_back_to_uniform(self) -> None:
        pooled = _linear_pool({"a": Distribution.from_edge(0.9)}, {"a": 0.0})
        assert pooled.up == pytest.approx(1 / 3)


class TestEnsembleSuppression:
    def test_unskilled_models_produce_no_prediction(self) -> None:
        """The measured state of this repository: nothing has earned a weight."""
        models = _panel([0.4] * 8)
        result = EnsembleModel(models).predict_detailed(_ctx())
        assert not result.published
        assert not result.contributions
        assert "demonstrated out-of-sample skill" in result.suppressed_because[0]
        assert result.prediction.confidence == 0.0
        assert result.prediction.distribution.up == pytest.approx(1 / 3)

    def test_the_shipped_panel_currently_publishes_nothing(self) -> None:
        """End to end with the real Phase 6 models and their real (zero) weights."""
        pairs = build_contexts(
            ContextSource("BTC", HOUR, candles(drifting(1400))), HORIZON, warmup=450
        )
        report = WalkForwardEvaluator(ClimatologyBaseline()).evaluate(
            [m() for m in ALL_MODELS], pairs
        )
        ensemble = EnsembleModel([m() for m in ALL_MODELS], SkillWeights.from_report(report))
        published = [
            ensemble.predict_detailed(ctx) for ctx, _ in pairs[:20]
        ]
        assert not any(p.published for p in published)

    def test_disagreement_suppresses_rather_than_averaging(self) -> None:
        """Half up, half down must not become a confident-looking middle."""
        models = _panel([0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5])
        ensemble = EnsembleModel(
            models,
            _full_weights(models),
            _full_calibration(models, ["range_low_vol", "unknown", "range_high_vol"]),
        )
        result = ensemble.predict_detailed(_ctx())
        assert not result.published
        assert any("disagree" in reason for reason in result.suppressed_because)

    def test_the_dissenters_are_published_with_the_result(self) -> None:
        """§25: if models disagree, show the disagreement."""
        models = _panel([0.5] * 6 + [-0.5, -0.5])
        ensemble = EnsembleModel(
            models,
            _full_weights(models),
            _full_calibration(models, ["range_low_vol", "unknown", "range_high_vol"]),
        )
        result = ensemble.predict_detailed(_ctx())
        assert len(result.agreement.dissenting) == 2
        labels = " ".join(e.label for e in result.prediction.counter_evidence)
        assert "family6" in labels or "family7" in labels

    def test_abstaining_members_do_not_drag_the_pool_toward_flat(self) -> None:
        models = _panel([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0])
        ensemble = EnsembleModel(
            models,
            _full_weights(models),
            _full_calibration(models, ["range_low_vol", "unknown", "range_high_vol"]),
        )
        result = ensemble.predict_detailed(_ctx())
        assert set(result.contributions) == {f"family{i}" for i in range(6)}
        assert result.prediction.distribution.up == pytest.approx(
            Distribution.from_edge(0.5).up
        )

    def test_calibration_fitted_over_this_point_is_not_applied(self) -> None:
        """Rather than raising or cheating, the model is simply uncalibrated here.

        Found by the guard firing on a sweep script that fitted calibration over all
        history and then replayed it from the beginning. The ensemble must degrade to
        "uncalibrated at this instant" instead of using a curve that saw the answer.
        """
        models = _panel([0.5] * 8)
        library = CalibrationLibrary()
        ctx = _ctx()
        for model in models:
            library.add(
                CalibrationRecord(
                    model_id=model.model_id,
                    regime=ctx.regime,
                    curves={o: CalibrationCurve.identity() for o in Outcome},
                    samples=500,
                    holdout_samples=200,
                    ece_before=0.05,
                    ece_after=0.02,
                    improved=True,
                    # Fitted through a moment *after* the prediction being made.
                    fitted_through=ctx.as_of + timedelta(days=30),
                )
            )
        result = EnsembleModel(models, _full_weights(models), library).predict_detailed(ctx)
        assert not result.published
        assert result.factors.calibration == 0.0

    def test_a_failing_member_does_not_abort_the_ensemble(self) -> None:
        class _Broken(_FamilyModel):
            def predict(self, ctx: PredictionContext) -> Prediction:
                raise RuntimeError("boom")

        models = [*_panel([0.5] * 7), _Broken("family7", 0.5)]
        ensemble = EnsembleModel(
            models,
            _full_weights(models),
            _full_calibration(models, ["range_low_vol", "unknown", "range_high_vol"]),
        )
        result = ensemble.predict_detailed(_ctx())
        assert len(result.members) == 7


class TestEnsemblePublishes:
    """The other direction: with real evidence the machinery does fire.

    Without these, every suppression test above would also pass on an ensemble that
    simply returned "insufficient evidence" unconditionally.
    """

    @staticmethod
    def _agreeing(quality: float = 1.0):
        models = _panel([0.5] * 8)
        ensemble = EnsembleModel(
            models,
            _full_weights(models),
            _full_calibration(models, ["range_low_vol", "unknown", "range_high_vol"]),
        )
        return ensemble, ensemble.predict_detailed(_ctx(data_quality=quality))

    def test_a_skilled_agreeing_calibrated_panel_publishes(self) -> None:
        _, result = self._agreeing()
        assert result.published
        assert result.prediction.is_actionable
        assert result.prediction.distribution.most_likely is Outcome.UP
        assert result.prediction.confidence > 0.35
        assert result.prediction.invalidation

    def test_published_confidence_equals_its_own_decomposition(self) -> None:
        """A number the UI cannot explain from its parts is an assertion."""
        _, result = self._agreeing()
        assert result.prediction.confidence == pytest.approx(result.factors.value)

    def test_degrading_the_feed_measurably_lowers_published_confidence(self) -> None:
        """The Phase 7 gate, end to end through the ensemble."""
        values = [self._agreeing(q)[1].prediction.confidence for q in (1.0, 0.9, 0.7)]
        assert all(b < a for a, b in pairwise(values))

    def test_a_badly_degraded_feed_suppresses_the_prediction_entirely(self) -> None:
        _, result = self._agreeing(quality=0.25)
        assert not result.published
        assert any("confidence" in reason for reason in result.suppressed_because)

    def test_calibration_is_applied_before_the_panel_votes(self) -> None:
        """A bias the system has already corrected must not also be voted on."""
        models = _panel([0.5] * 8)
        library = _full_calibration(models, ["range_low_vol", "unknown", "range_high_vol"])
        # A curve that inverts every model's view: up becomes unlikely.
        for model in models:
            library.add(
                CalibrationRecord(
                    model_id=model.model_id,
                    regime="unknown",
                    curves={
                        Outcome.UP: CalibrationCurve(xs=(0.0, 1.0), ys=(0.0, 0.05)),
                        Outcome.FLAT: CalibrationCurve.identity(),
                        Outcome.DOWN: CalibrationCurve(xs=(0.0, 1.0), ys=(0.0, 1.0)),
                    },
                    samples=500,
                    holdout_samples=200,
                    ece_before=0.05,
                    ece_after=0.02,
                    improved=True,
                )
            )
        ctx = _ctx()
        result = EnsembleModel(models, _full_weights(models), library).predict_detailed(ctx)
        if ctx.regime == "unknown":
            assert result.agreement.majority is Outcome.DOWN


# ---------------------------------------------------------------------- gate


class TestSuperPredictionGate:
    @staticmethod
    def _setup(edges: list[float], quality: float = 1.0):
        models = _panel(edges)
        library = _full_calibration(
            models, ["range_low_vol", "unknown", "range_high_vol", "uptrend_low_vol"]
        )
        ensemble = EnsembleModel(models, _full_weights(models), library)
        result = ensemble.predict_detailed(_ctx(data_quality=quality))
        decision = SuperPredictionGate().evaluate(
            result, library, [m.model_id for m in models]
        )
        return result, decision

    def test_a_unanimous_calibrated_skilled_panel_earns_a_super_prediction(self) -> None:
        """Proves the gate is not simply always false."""
        _, decision = self._setup([0.5] * 8)
        assert decision.passed, decision.report()

    def test_induced_disagreement_produces_no_super_prediction(self) -> None:
        """The Phase 7 gate, stated verbatim in the requirements."""
        _, decision = self._setup([0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5])
        assert not decision.passed
        names = {c.name for c in decision.failures}
        assert "families agreeing" in names
        assert "no material dissent" in names

    def test_five_of_eight_is_not_enough(self) -> None:
        _, decision = self._setup([0.5] * 5 + [-0.5] * 3)
        assert not decision.passed
        assert "families agreeing" in {c.name for c in decision.failures}

    def test_six_of_eight_with_two_quiet_models_passes(self) -> None:
        """Abstention is not dissent: six agreeing and two silent still clears the bar."""
        _, decision = self._setup([0.5] * 6 + [0.0, 0.0])
        assert decision.passed, decision.report()

    def test_six_agreeing_clones_do_not_clear_the_independence_bar(self) -> None:
        models = [_FamilyModel(f"clone{i}", 0.5, substrate="price") for i in range(8)]
        library = _full_calibration(
            models, ["range_low_vol", "unknown", "range_high_vol", "uptrend_low_vol"]
        )
        result = EnsembleModel(models, _full_weights(models), library).predict_detailed(_ctx())
        decision = SuperPredictionGate().evaluate(
            result, library, [m.model_id for m in models]
        )
        assert not decision.passed
        assert "independent agreement" in {c.name for c in decision.failures}

    def test_no_calibration_in_this_regime_blocks_the_gate(self) -> None:
        models = _panel([0.5] * 8)
        result = EnsembleModel(models, _full_weights(models)).predict_detailed(_ctx())
        decision = SuperPredictionGate().evaluate(
            result, CalibrationLibrary(), [m.model_id for m in models]
        )
        assert not decision.passed
        assert "calibrated in this regime" in {c.name for c in decision.failures}

    def test_no_demonstrated_skill_blocks_the_gate(self) -> None:
        models = _panel([0.5] * 8)
        library = _full_calibration(
            models, ["range_low_vol", "unknown", "range_high_vol", "uptrend_low_vol"]
        )
        result = EnsembleModel(models, SkillWeights(), library).predict_detailed(_ctx())
        decision = SuperPredictionGate().evaluate(
            result, library, [m.model_id for m in models]
        )
        assert not decision.passed
        assert "demonstrated skill" in {c.name for c in decision.failures}

    def test_degraded_data_blocks_the_gate(self) -> None:
        _, decision = self._setup([0.5] * 8, quality=0.5)
        assert not decision.passed
        assert "data quality" in {c.name for c in decision.failures}

    def test_every_condition_is_reported_whether_it_passed_or_not(self) -> None:
        """A gate that only says no is untestable in practice."""
        _, decision = self._setup([0.5, -0.5] * 4)
        assert len(decision.checks) >= 8
        assert decision.reasons
        assert "no super prediction" in decision.report()
