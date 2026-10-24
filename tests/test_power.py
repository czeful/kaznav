"""Power analysis (D-049): analytic McNemar n, Monte-Carlo power of the scene sign-flip test, pilot estimates."""
import numpy as np
import pytest

from kznav.eval.power import PowerParams, estimate_from_pairs, mcnemar_n, required_n, simulate_pairs, simulate_power


def test_mcnemar_n_matches_the_connor_formula_and_grows_with_clustering_and_contrasts():
    # z_.975 = 1.95996, z_.8 = 0.84162: n = (1.95996 sqrt(.3) + 0.84162 sqrt(.29))^2 / .01 = 233.1 -> 234
    assert mcnemar_n(0.3, 0.1, 0.3) == 234
    assert mcnemar_n(0.3, 0.1, 0.3, m=5) > 234 and mcnemar_n(0.3, 0.1, 0.3, design_effect=1.5) == 350
    with pytest.raises(ValueError):
        mcnemar_n(0.3, 0.2, 0.1)


def test_simulated_tables_have_the_requested_marginals():
    pp = PowerParams(p_a=0.3, delta=0.15, psi=0.35, n=2000, k_scenes=20, sigma_scene=0.0, tau=0.0)
    d = simulate_pairs(pp, np.random.default_rng(0), 50)
    assert d.mean() == pytest.approx(0.15, abs=0.01)
    assert (d != 0).mean() == pytest.approx(0.35, abs=0.01)


def test_type_one_error_and_power_behave():
    null = simulate_power(PowerParams(p_a=0.3, delta=0.0, psi=0.3, n=300, alpha=0.05), n_sims=400, n_perm=1000)
    assert 0.02 <= null["power"] <= 0.09
    weak = simulate_power(PowerParams(p_a=0.3, delta=0.1, psi=0.3, n=150, m=5), n_sims=300, n_perm=2000)
    strong = simulate_power(PowerParams(p_a=0.3, delta=0.1, psi=0.3, n=1000, m=5), n_sims=300, n_perm=2000)
    assert weak["power"] < 0.5 < strong["power"]
    noisy = simulate_power(PowerParams(p_a=0.3, delta=0.1, psi=0.3, n=400, tau=0.15), n_sims=300, n_perm=2000)
    clean = simulate_power(PowerParams(p_a=0.3, delta=0.1, psi=0.3, n=400, tau=0.0), n_sims=300, n_perm=2000)
    assert noisy["power"] < clean["power"]                        # effect heterogeneity across scenes costs power


def test_required_n_and_parameter_checks():
    n = required_n(PowerParams(p_a=0.3, delta=0.2, psi=0.3, n=100, m=1), n_sims=200, n_perm=1500)
    assert n is not None and n <= 200
    with pytest.raises(ValueError):
        PowerParams(p_a=0.3, delta=0.2, psi=0.1, n=100)
    with pytest.raises(ValueError):
        PowerParams(p_a=0.95, delta=0.1, psi=0.2, n=100)


def test_pilot_estimates():
    a = np.array([1, 1, 0, 0, 1, 0, 0, 0, 1, 0] * 6, float)
    b = np.array([1, 0, 1, 0, 1, 1, 0, 0, 1, 1] * 6, float)
    est = estimate_from_pairs(a, b, np.repeat(np.arange(6), 10))
    assert est["p_a"] == pytest.approx(0.4) and est["delta"] == pytest.approx(0.2) and est["psi"] == pytest.approx(0.4)
    assert est["k_scenes"] == 6 and est["sigma_scene"] == pytest.approx(0.0) and "tau" in est
