"""Statistics: bootstrap and Wilson intervals, effect sizes, exact McNemar and the aggregate tables."""
import math

import numpy as np
import pandas as pd
import pytest

from kznav.eval.aggregate import paired_contrast, summarize
from kznav.eval.stats import (
    bootstrap_ci,
    cohens_d_independent,
    cohens_d_paired,
    mcnemar_exact,
    mean_ci,
    paired_bootstrap_diff,
    wilson_ci,
)


class TestBootstrap:
    def test_constant_sample_has_degenerate_ci(self):
        ci = bootstrap_ci([0.3] * 20)
        assert ci.low == ci.estimate == ci.high
        assert ci.estimate == pytest.approx(0.3)

    def test_deterministic_for_seed(self):
        x = np.random.default_rng(1).normal(size=50)
        assert bootstrap_ci(x, seed=7) == bootstrap_ci(x, seed=7)

    def test_width_matches_normal_theory_for_large_n(self):
        x = np.random.default_rng(2).normal(size=2000)
        ci = bootstrap_ci(x, n_boot=4000)
        expected = 2 * 1.96 * x.std(ddof=1) / math.sqrt(x.size)
        assert (ci.high - ci.low) == pytest.approx(expected, rel=0.1)
        assert ci.low < x.mean() < ci.high

    def test_coverage_near_nominal(self):
        rng = np.random.default_rng(3)
        hits = 0
        reps = 200
        for r in range(reps):
            x = rng.normal(loc=1.0, size=40)
            ci = bootstrap_ci(x, n_boot=2000, seed=r)
            hits += ci.low <= 1.0 <= ci.high
        assert 0.88 <= hits / reps <= 0.99

    def test_rejects_non_finite(self):
        with pytest.raises(ValueError):
            bootstrap_ci([1.0, math.inf])
        with pytest.raises(ValueError):
            bootstrap_ci([])

    def test_paired_diff_uses_pairs(self):
        rng = np.random.default_rng(4)
        base = rng.normal(size=100) * 10      # large between-episode variance
        a, b = base + 0.5, base               # constant within-pair difference
        ci = paired_bootstrap_diff(a, b)
        assert ci.estimate == pytest.approx(0.5)
        assert ci.low == pytest.approx(0.5) and ci.high == pytest.approx(0.5)

    def test_paired_length_mismatch(self):
        with pytest.raises(ValueError):
            paired_bootstrap_diff([1, 2, 3], [1, 2])


class TestWilson:
    def test_zero_of_ten(self):
        ci = wilson_ci(0, 10)
        assert ci.low == pytest.approx(0.0, abs=1e-12)
        assert ci.high == pytest.approx(0.27753, abs=1e-4)

    def test_half(self):
        ci = wilson_ci(5, 10)
        assert (ci.low, ci.high) == pytest.approx((0.23659, 0.76341), abs=1e-4)

    def test_mean_ci_picks_wilson_for_binary(self):
        assert mean_ci([1, 0, 1, 1]).method == "wilson"
        assert mean_ci([0.2, 0.5, 0.9]).method == "percentile-bootstrap"


class TestEffectSizes:
    def test_cohens_d_independent(self):
        a, b = [1, 2, 3, 4, 5], [3, 4, 5, 6, 7]
        assert cohens_d_independent(a, b) == pytest.approx(-1.264911, abs=1e-6)
        assert cohens_d_independent(a, b, hedges=True) == pytest.approx(-1.264911 * (1 - 3 / 31), abs=1e-6)

    def test_cohens_d_paired(self):
        assert cohens_d_paired([3, 5, 7], [2, 3, 4]) == pytest.approx(2.0)

    def test_zero_sd_gives_nan(self):
        assert math.isnan(cohens_d_paired([2, 3, 4], [1, 2, 3]))
        assert math.isnan(cohens_d_independent([1, 1, 1], [1, 1, 1]))


class TestMcNemar:
    def test_all_discordant_one_way(self):
        r = mcnemar_exact([True] * 10 + [False] * 5, [False] * 10 + [False] * 5)
        assert (r.only_a, r.only_b) == (10, 0)
        assert r.p_value == pytest.approx(2 / 1024)

    def test_balanced(self):
        r = mcnemar_exact([1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1])
        assert r.p_value == 1.0

    def test_no_discordant(self):
        assert mcnemar_exact([1, 0], [1, 0]).p_value == 1.0


class TestAggregate:
    def frame(self):
        rows = []
        for model, bonus in (("qolda", 0.2), ("qwen", 0.0)):
            for i in range(20):
                rows.append({"backend": model, "dataset": "d", "split": "val", "episode_id": str(i),
                             "command_id": "c", "success": (i + int(bonus * 10)) % 2 == 0,
                             "spl": min(1.0, 0.3 + bonus + (i % 5) * 0.1),
                             "dts": math.inf if i == 3 else 0.5})
        return pd.DataFrame(rows)

    def test_summarize(self):
        s = summarize(self.frame(), ["success", "spl", "dts"], by=["backend"], n_boot=500)
        dts = s[(s.backend == "qolda") & (s.metric == "dts")].iloc[0]
        assert dts.n == 19 and dts.n_excluded == 1
        succ = s[(s.backend == "qolda") & (s.metric == "success")].iloc[0]
        assert succ.ci_method == "wilson"

    def test_paired_contrast(self):
        out = paired_contrast(self.frame(), arm_col="backend", arm_a="qolda", arm_b="qwen", metric="spl", n_boot=500)
        assert out["n_pairs"] == 20
        assert out["diff"] > 0
        assert out["diff_ci_low"] <= out["diff"] <= out["diff_ci_high"]

    def test_unpaired_arms_raise(self):
        df = self.frame()
        df = df[~((df.backend == "qwen") & (df.episode_id == "0"))]
        with pytest.raises(ValueError, match="not paired"):
            paired_contrast(df, arm_col="backend", arm_a="qolda", arm_b="qwen", metric="spl")
