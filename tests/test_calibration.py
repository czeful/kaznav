"""Calibration metrics (ECE, reliability diagram) and binary classification metrics, including the pilot's numbers."""
import json
from pathlib import Path

import numpy as np
import pytest

from kznav.eval.calibration import (
    brier_score,
    ece_binary,
    ece_top_label,
    mce_binary,
    reliability_diagram,
)
from kznav.eval.classification import binary_metrics, threshold_metrics

PILOT = Path(__file__).parent / "data" / "pilot_verification.jsonl"


def pilot_rows():
    return [json.loads(l) for l in PILOT.read_text(encoding="utf-8").splitlines() if l.strip()]


class TestECE:
    def test_perfectly_calibrated_bins(self):
        p = [0.2] * 5 + [0.8] * 5
        y = [1, 0, 0, 0, 0] + [1, 1, 1, 1, 0]
        assert ece_binary(p, y) == pytest.approx(0.0, abs=1e-12)

    def test_maximally_wrong(self):
        p, y = [0.9] * 10, [0] * 10
        assert ece_binary(p, y) == pytest.approx(0.9)
        assert brier_score(p, y) == pytest.approx(0.81)
        assert mce_binary(p, y) == pytest.approx(0.9)

    def test_large_calibrated_sample_small_ece(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(size=200_000)
        y = rng.uniform(size=p.size) < p
        assert ece_binary(p, y) < 0.01

    def test_overconfident_model_large_ece(self):
        rng = np.random.default_rng(1)
        p_true = rng.uniform(0.05, 0.95, size=50_000)
        y = rng.uniform(size=p_true.size) < p_true
        logit = np.log(p_true / (1 - p_true))
        p_over = 1 / (1 + np.exp(-3 * logit))
        assert ece_binary(p_over, y) > 0.1
        assert ece_binary(p_true, y) < 0.02

    def test_reliability_diagram_counts(self):
        rd = reliability_diagram([0.0, 0.05, 0.15, 1.0], [0, 0, 1, 1], n_bins=10)
        assert rd.counts.sum() == 4
        assert rd.counts[0] == 2 and rd.counts[1] == 1 and rd.counts[9] == 1
        assert len(rd.rows()) == 10

    def test_input_validation(self):
        with pytest.raises(ValueError):
            ece_binary([1.2], [1])
        with pytest.raises(ValueError):
            ece_binary([0.5], [2])
        with pytest.raises(ValueError):
            ece_binary([0.5, 0.2], [1])


class TestReproducesPilot:
    """Numbers from qolda check/results/03_candidate_verification_summary.csv (pilot report)."""

    @pytest.mark.parametrize(
        "category,acc,brier,ece",
        [
            (None, 0.84375, 0.11539584887010695, 0.15735284771418506),
            ("cultural", 0.75, 0.1636999460173219, 0.2630255073769003),
            ("universal", 0.9375, 0.067091751722892, 0.08951890168003579),
        ],
    )
    def test_pilot_numbers(self, category, acc, brier, ece):
        rows = [r for r in pilot_rows() if category is None or r["category"] == category]
        p = [r["probability_yes"] for r in rows]
        y = [r["gold_label"] for r in rows]
        assert threshold_metrics(p, y, 0.5).accuracy == pytest.approx(acc, abs=1e-12)
        assert brier_score(p, y) == pytest.approx(brier, abs=1e-12)
        assert ece_top_label(p, y) == pytest.approx(ece, abs=1e-12)


class TestBinaryMetrics:
    def test_known_confusion_matrix(self):
        m = binary_metrics([1, 1, 0, 0, 1], [1, 0, 0, 1, 1])
        assert (m.tp, m.fp, m.tn, m.fn) == (2, 1, 1, 1)
        assert m.accuracy == pytest.approx(0.6)
        assert m.precision == pytest.approx(2 / 3)
        assert m.recall == pytest.approx(2 / 3)
        assert m.f1 == pytest.approx(2 / 3)
        assert m.false_positive_rate == pytest.approx(0.5)

    def test_undefined_precision_is_nan(self):
        import math

        m = binary_metrics([0, 0, 0], [1, 0, 1])
        assert math.isnan(m.precision) and math.isnan(m.f1) and m.recall == 0.0
