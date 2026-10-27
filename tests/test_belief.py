"""Hypothesis belief, candidate evidence and the calibrators of the stop rule (kznav.policy.belief, .calibration)."""
import numpy as np
import pytest

from kznav.policy.belief import BeliefParams, CandidateEvidence, HypothesisBelief, logit, sigmoid
from kznav.policy.calibration import (
    CalibrationArtifact,
    IsotonicCalibrator,
    PlattCalibrator,
    calibrator_from_dict,
    choose_threshold,
)

NAMES = ["cauldron", "cooking pot", "wok"]


def params(**kw):
    base = dict(rho_right=0.7, rho_wrong=0.1, base_rate=0.3, evidence_power=1.0, weight_floor=0.0)
    base.update(kw)
    return BeliefParams(**base)


class TestHypothesisBelief:
    def test_prior_is_normalised(self):
        b = HypothesisBelief(NAMES, [6, 3, 1], params())
        assert b.weights == pytest.approx([0.6, 0.3, 0.1])

    def test_zero_prior_becomes_uniform(self):
        b = HypothesisBelief(NAMES, [0, 0, 0], params())
        assert b.weights == pytest.approx([1 / 3] * 3)

    def test_uninformative_verifier_changes_nothing(self):
        b = HypothesisBelief(NAMES, [0.6, 0.3, 0.1], params())
        b.update([1.0, 0.0, 0.0], p_target=0.3)  # q == base rate -> likelihood ratio 1
        assert b.weights == pytest.approx([0.6, 0.3, 0.1])

    def test_equal_match_changes_nothing(self):
        b = HypothesisBelief(NAMES, [0.6, 0.3, 0.1], params())
        b.update([0.5, 0.5, 0.5], p_target=0.95)
        assert b.weights == pytest.approx([0.6, 0.3, 0.1])

    def test_positive_verification_boosts_matched_hypothesis(self):
        b = HypothesisBelief(NAMES, [0.4, 0.4, 0.2], params())
        w = b.update([0.0, 1.0, 0.0], p_target=0.95)
        assert w[1] > 0.4 and w[0] < 0.4 and w[2] < 0.2
        assert w.sum() == pytest.approx(1.0)

    def test_negative_verification_suppresses_matched_hypothesis(self):
        b = HypothesisBelief(NAMES, [0.4, 0.4, 0.2], params())
        w = b.update([0.0, 1.0, 0.0], p_target=0.02)
        assert w[1] < 0.4 and w[0] > 0.4

    def test_hand_computed_update(self):
        p = params()
        b = HypothesisBelief(["a", "b"], [0.5, 0.5], p)
        q = 0.9
        lam = (q / (1 - q)) / (0.3 / 0.7)
        like_a = 0.7 * lam + 0.3          # matched (m = 1): rho = rho_right
        like_b = 0.1 * lam + 0.9          # unmatched (m = 0): rho = rho_wrong
        b.update([1.0, 0.0], q)
        assert b.weights[0] == pytest.approx(like_a / (like_a + like_b))

    def test_updates_commute(self):
        e1 = ([1.0, 0.2, 0.0], 0.9)
        e2 = ([0.0, 0.8, 0.3], 0.2)
        b1 = HypothesisBelief(NAMES, [0.5, 0.3, 0.2], params())
        b2 = HypothesisBelief(NAMES, [0.5, 0.3, 0.2], params())
        b1.update(*e1); b1.update(*e2)
        b2.update(*e2); b2.update(*e1)
        assert b1.weights == pytest.approx(b2.weights)

    def test_floor_prevents_elimination(self):
        b = HypothesisBelief(NAMES, [0.34, 0.33, 0.33], params(weight_floor=0.02))
        for _ in range(20):
            b.update([1.0, 0.0, 0.0], p_target=0.001)
        w = b.weights
        assert w.min() >= 0.02 - 1e-12
        assert w.sum() == pytest.approx(1.0)
        assert w[0] == pytest.approx(0.02)

    def test_tempering_reduces_step(self):
        full = HypothesisBelief(["a", "b"], [0.5, 0.5], params())
        half = HypothesisBelief(["a", "b"], [0.5, 0.5], params(evidence_power=0.5))
        full.update([1, 0], 0.95)
        half.update([1, 0], 0.95)
        assert 0.5 < half.weights[0] < full.weights[0]

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            params(rho_right=0.1, rho_wrong=0.2)
        with pytest.raises(ValueError):
            params(base_rate=0.0)
        with pytest.raises(ValueError):
            HypothesisBelief(NAMES, [1, 1], params())
        with pytest.raises(ValueError):
            HypothesisBelief(NAMES, [1, 1, 1], params(weight_floor=0.4))


class TestCandidateEvidence:
    def test_no_observation_is_base_rate(self):
        assert CandidateEvidence(0.3).probability() == pytest.approx(0.3)

    def test_single_observation_returns_it(self):
        assert CandidateEvidence(0.3).add(0.8) == pytest.approx(0.8)

    def test_two_observations_add_in_logit_space(self):
        ev = CandidateEvidence(0.3)
        ev.add(0.8)
        p = ev.add(0.7)
        expected = sigmoid(logit(0.8) + logit(0.7) - logit(0.3))
        assert p == pytest.approx(expected)

    def test_view_discount(self):
        ev = CandidateEvidence(0.3, view_discount=0.5)
        ev.add(0.9)
        expected = sigmoid(logit(0.3) + 0.5 * (logit(0.9) - logit(0.3)))
        assert ev.probability() == pytest.approx(expected)

    def test_first_view_full_applies_the_discount_to_repeated_views_only(self):
        """D-069: one view gives exactly the calibrated q the stop threshold was fitted on; later views are discounted."""
        ev = CandidateEvidence(0.2, view_discount=0.5, first_view_full=True)
        assert ev.add(0.35) == pytest.approx(0.35)
        expected = sigmoid(logit(0.35) + 0.5 * (logit(0.6) - logit(0.2)))
        assert ev.add(0.6) == pytest.approx(expected)
        shrunk = CandidateEvidence(0.2, view_discount=0.5)
        assert shrunk.add(0.35) < 0.35          # the pre-D-069 rule pulls a single view towards the base rate


class TestCalibrators:
    def test_platt_recovers_parameters(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 2, size=20_000)
        p_raw = 1 / (1 + np.exp(-x))
        y = rng.uniform(size=x.size) < 1 / (1 + np.exp(-(0.5 * x + 0.3)))
        cal = PlattCalibrator.fit(p_raw, y)
        assert cal.a == pytest.approx(0.5, abs=0.05)
        assert cal.b == pytest.approx(0.3, abs=0.05)
        assert calibrator_from_dict(cal.to_dict())(p_raw[:5]) == pytest.approx(cal(p_raw[:5]))

    def test_platt_needs_both_classes(self):
        with pytest.raises(ValueError):
            PlattCalibrator.fit([0.2, 0.9], [1, 1])

    def test_isotonic_pav(self):
        cal = IsotonicCalibrator.fit([0.1, 0.2, 0.3, 0.4], [1, 0, 1, 1])
        assert cal([0.1, 0.2, 0.3, 0.4]) == pytest.approx([0.5, 0.5, 1.0, 1.0])
        grid = np.linspace(0, 1, 101)
        assert np.all(np.diff(cal(grid)) >= -1e-12)

    def test_threshold_selection(self):
        p, y = [0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]
        best = choose_threshold(p, y, objective="max_f1")
        assert best.threshold == pytest.approx(0.8) and best.f1 == pytest.approx(1.0)
        prec = choose_threshold([0.1, 0.4, 0.6, 0.8, 0.9], [0, 1, 0, 1, 1], objective="min_precision",
                                min_precision=1.0)
        assert prec.threshold == pytest.approx(0.8) and prec.precision == 1.0

    def test_artifact_guards_against_leakage(self, tmp_path):
        art = CalibrationArtifact(
            calibrator={"kind": "identity"}, threshold={"threshold": 0.7}, backend="qolda", model_id="qolda-4b",
            prompt_sha256="abc", dev_split="dev", dev_episode_ids=["e1", "e2"], n_samples=2, base_rate=0.3,
            created_utc="2026-09-24T00:00:00+00:00",
        )
        art.save(tmp_path / "cal.json")
        loaded = CalibrationArtifact.load(tmp_path / "cal.json")
        loaded.assert_disjoint_from(["e3", "e4"])
        with pytest.raises(ValueError, match="disjoint"):
            loaded.assert_disjoint_from(["e2", "e9"])
