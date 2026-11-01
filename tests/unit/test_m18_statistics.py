"""Unit tests for from-scratch statistics against scipy.stats test oracle (SYS-17)."""

import numpy as np
import pytest
from scipy import stats

from autogen.analytics.statistics import (
    bootstrap_ci,
    cliffs_delta,
    cohens_d,
    confidence_interval,
    interquartile_range,
    mean,
    median,
    normal_quantile,
    sample_stats,
    sample_variance,
    standard_deviation,
    standard_error,
    student_t_quantile,
)


def test_descriptive_stats_match_numpy_and_scipy():
    rng = np.random.default_rng(101)
    for _ in range(20):
        data = rng.normal(loc=15.0, scale=3.5, size=50).tolist()

        m = mean(data)
        v = sample_variance(data, ddof=1)
        s = standard_deviation(data, ddof=1)
        se = standard_error(data)
        med = median(data)
        iqr_val = interquartile_range(data)

        # Oracle comparisons
        assert pytest.approx(m, abs=1e-12) == float(np.mean(data))
        assert pytest.approx(v, abs=1e-12) == float(np.var(data, ddof=1))
        assert pytest.approx(s, abs=1e-12) == float(np.std(data, ddof=1))
        assert pytest.approx(se, abs=1e-12) == float(stats.sem(data))
        assert pytest.approx(med, abs=1e-12) == float(np.median(data))
        assert pytest.approx(iqr_val, abs=1e-12) == float(stats.iqr(data, scale=1.0))


def test_normal_quantile_accuracy():
    probabilities = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99, 0.999]
    for p in probabilities:
        approx = normal_quantile(p)
        oracle = float(stats.norm.ppf(p))
        assert pytest.approx(approx, abs=1e-7) == oracle


def test_student_t_quantile_accuracy():
    probabilities = [0.90, 0.95, 0.975, 0.99]
    for df in [1, 2, 5, 10, 20, 50, 100]:
        for p in probabilities:
            approx = student_t_quantile(df=df, p=p)
            oracle = float(stats.t.ppf(p, df=df))
            # Critical values match scipy oracle closely
            assert pytest.approx(approx, rel=1e-2) == oracle


def test_confidence_interval_matches_scipy():
    rng = np.random.default_rng(202)
    data = rng.normal(loc=50.0, scale=10.0, size=30).tolist()

    ci_l, ci_u = confidence_interval(data, confidence=0.95, method="t")
    oracle_l, oracle_u = stats.t.interval(0.95, df=len(data) - 1, loc=np.mean(data), scale=stats.sem(data))

    assert pytest.approx(ci_l, abs=1e-3) == float(oracle_l)
    assert pytest.approx(ci_u, abs=1e-3) == float(oracle_u)


def test_cohens_d_effect_size():
    # Two identical groups should yield 0.0 effect size
    g1 = [1.0, 2.0, 3.0, 4.0, 5.0]
    g2 = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert cohens_d(g1, g2) == 0.0

    # Shifted group with variance in differences
    g3 = [x + float(i + 1) for i, x in enumerate(g1)]
    d_val = cohens_d(g3, g1)
    assert d_val > 0.0
    # Paired Cohen's d
    d_paired = cohens_d(g3, g1, paired=True)
    assert d_paired > 0.0


def test_cliffs_delta_effect_size():
    # Identical groups: delta = 0
    g1 = [10.0, 20.0, 30.0]
    g2 = [10.0, 20.0, 30.0]
    assert cliffs_delta(g1, g2) == 0.0

    # Strictly dominating group: delta = 1.0
    g_high = [100.0, 200.0, 300.0]
    g_low = [1.0, 2.0, 3.0]
    assert cliffs_delta(g_high, g_low) == 1.0
    assert cliffs_delta(g_low, g_high) == -1.0


def test_bootstrap_ci_coverage():
    # Known normal distribution: 95% bootstrap CI should cover true mean (mu=10.0)
    rng = np.random.default_rng(303)
    covered = 0
    trials = 25
    true_mu = 10.0

    for seed in range(trials):
        data = rng.normal(loc=true_mu, scale=2.0, size=30).tolist()
        l_bound, u_bound = bootstrap_ci(data, confidence=0.95, n_resamples=1000, seed=seed)
        if l_bound <= true_mu <= u_bound:
            covered += 1

    coverage_rate = covered / trials
    # Empirical coverage should be at least 85%
    assert coverage_rate >= 0.85


def test_sample_stats_dto():
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    s = sample_stats(data, confidence=0.95)
    assert s.n == 7
    assert s.mean == 4.0
    assert s.median == 4.0
    assert s.ci_lower < s.mean < s.ci_upper
