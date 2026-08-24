"""Confidence: how much the system trusts its own probability estimate.

The distinction this module exists to preserve, restated because everything downstream
depends on it: **probability is not confidence**. A model saying 70% up is a claim about
the market. Confidence is a claim about that claim — and the two can be pulled apart in
both directions. A well-calibrated model in a familiar regime saying 55% deserves high
confidence in a modest probability. An uncalibrated model in an unprecedented regime
saying 85% deserves the opposite.

Confidence here is a product of independent multiplicative factors, each in [0, 1] and
each named. Multiplicative rather than a weighted average because these are *veto*
conditions, not votes: if the data is untrustworthy it does not matter how much the
models agree, and an average would let a strong factor mask a fatal one. The product
form means any single factor near zero collapses the result, which is the behaviour
requirement §20 asks for.

The factors, and why each exists:

* **skill** — has any contributing model demonstrated out-of-sample skill against
  climatology, in this regime? This is the factor that connects Phase 6's measurement
  to Phase 7's output. With no skilled models it is zero, and confidence is zero. That
  is not a degenerate case to be worked around; it is the honest answer to the data.
* **calibration** — does a usable calibration record exist here? An uncalibrated
  probability is a number, not an estimate.
* **agreement** — weighted consensus across independent families, from
  :mod:`~mie.ensemble.agreement`.
* **data quality** — Phase 1's trust score. §20: degraded data lowers confidence
  rather than being ignored.
* **sample** — how much evidence the skill and calibration estimates rest on. A model
  measured on 40 points is not as trustworthy as one measured on 4,000, even if both
  look equally good.
* **regime familiarity** — whether this regime has been seen enough to say anything
  about behaviour in it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["ConfidenceFactors", "confidence_from"]

#: Nothing published above this, ever. Certainty about future market movements is not
#: available at any level of evidence, and a system that can print 95% will eventually
#: print it about something it does not understand.
_MAX_CONFIDENCE = 0.85

#: Below this, the output is labelled insufficient evidence rather than published.
_PUBLISH_FLOOR = 0.35

#: Predictions needed behind a skill or calibration estimate before the sample factor
#: reaches 1.0. Chosen so that the factor is still climbing at the sample sizes these
#: evaluations actually produce, rather than saturating immediately.
_SAMPLE_SATURATION = 400


@dataclass(slots=True)
class ConfidenceFactors:
    """The decomposition behind one confidence number.

    Kept as data rather than collapsed into a float so the UI can answer "why is this
    only 40%?" with the specific factor responsible. A confidence score whose
    derivation cannot be inspected is an assertion.
    """

    skill: float = 0.0
    calibration: float = 0.0
    agreement: float = 0.0
    data_quality: float = 1.0
    sample: float = 0.0
    regime_familiarity: float = 0.0
    #: Free-text notes on anything that suppressed the result.
    notes: list[str] = field(default_factory=list)

    @property
    def value(self) -> float:
        """The published confidence: the product, capped, then scaled by data quality.

        Data quality is applied *after* the cap rather than inside it. Capping the
        whole product would let the ceiling hide feed degradation — a system with
        everything else at 1.0 would publish 0.85 on pristine data and 0.85 again on
        data it half trusts, because both products clip. §20 requires the opposite:
        degraded data must visibly lower confidence, at every level.
        """
        core = (
            _clamp(self.skill)
            * _clamp(self.calibration)
            * _clamp(self.agreement)
            * _clamp(self.sample)
            * _clamp(self.regime_familiarity)
        )
        return round(min(_MAX_CONFIDENCE, core) * _clamp(self.data_quality), 4)

    @property
    def publishable(self) -> bool:
        return self.value >= _PUBLISH_FLOOR

    @property
    def limiting_factor(self) -> str:
        """Which factor is holding the result down — the answer to 'why so low?'."""
        named = {
            "skill": self.skill,
            "calibration": self.calibration,
            "agreement": self.agreement,
            "data_quality": self.data_quality,
            "sample": self.sample,
            "regime_familiarity": self.regime_familiarity,
        }
        return min(named, key=lambda key: named[key])

    def explain(self) -> str:
        parts = ", ".join(
            f"{name}={value:.2f}"
            for name, value in (
                ("skill", self.skill),
                ("calibration", self.calibration),
                ("agreement", self.agreement),
                ("quality", self.data_quality),
                ("sample", self.sample),
                ("regime", self.regime_familiarity),
            )
        )
        verdict = "publishable" if self.publishable else "insufficient evidence"
        return f"confidence {self.value:.2f} ({verdict}) <- {parts}; limited by {self.limiting_factor}"


def confidence_from(
    *,
    best_skill: float,
    skill_is_significant: bool,
    has_calibration: bool,
    calibration_in_regime: bool,
    calibration_improvement: float,
    consensus_share: float,
    effective_agreement: float,
    family_count: int,
    data_quality: float,
    evaluation_samples: int,
    regime_samples: int,
    min_regime_samples: int = 100,
) -> ConfidenceFactors:
    """Assemble the confidence factors from measured quantities.

    Every argument is something the system has *measured*, not something a model
    asserted about itself. A model's own confidence is deliberately not an input here:
    self-reported confidence is exactly the quantity that needs auditing.
    """
    factors = ConfidenceFactors(data_quality=_clamp(data_quality))

    # --- skill -------------------------------------------------------------
    if not skill_is_significant or best_skill <= 0:
        factors.skill = 0.0
        factors.notes.append(
            "no contributing model has demonstrated significant out-of-sample skill "
            "against climatology"
        )
    else:
        # A Brier skill of 0.05 against climatology on hourly crypto would be a strong
        # result; the scale reflects that rather than treating 1.0 as reachable.
        factors.skill = _clamp(0.4 + min(0.6, best_skill / 0.05 * 0.6))

    # --- calibration -------------------------------------------------------
    if not has_calibration:
        factors.calibration = 0.0
        factors.notes.append("no calibration record for these models")
    elif not calibration_in_regime:
        # A pooled record is evidence of *something*, but the requirement is a record
        # in the current regime, so this is a heavy discount rather than a pass.
        factors.calibration = 0.45
        factors.notes.append("calibrated only on pooled data, not in the current regime")
    else:
        factors.calibration = _clamp(0.7 + min(0.3, max(0.0, calibration_improvement) / 0.02 * 0.3))

    # --- agreement ---------------------------------------------------------
    if family_count <= 0 or effective_agreement <= 0:
        factors.agreement = 0.0
        factors.notes.append("no model expressed a directional view")
    else:
        # Both the share of the panel and how much of the panel voted at all: eight
        # models where six agree is stronger than two models where both agree.
        coverage = min(1.0, effective_agreement / max(1.0, family_count * 0.5))
        factors.agreement = _clamp(consensus_share * coverage)
        if consensus_share < 0.7:
            factors.notes.append(
                f"panel is split ({consensus_share:.0%} of weighted votes behind the majority)"
            )

    # --- sample ------------------------------------------------------------
    factors.sample = _clamp(min(1.0, evaluation_samples / _SAMPLE_SATURATION))
    if evaluation_samples < 100:
        factors.notes.append(
            f"only {evaluation_samples} evaluation points behind these estimates"
        )

    # --- regime familiarity -------------------------------------------------
    factors.regime_familiarity = _clamp(min(1.0, regime_samples / max(1, min_regime_samples)))
    if regime_samples < min_regime_samples:
        factors.notes.append(
            f"current regime observed only {regime_samples} times "
            f"(want {min_regime_samples})"
        )

    if factors.data_quality < 0.8:
        factors.notes.append(f"data quality degraded to {factors.data_quality:.2f}")

    return factors


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
