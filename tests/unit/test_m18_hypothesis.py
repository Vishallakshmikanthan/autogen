"""Unit tests for hypothesis tests and corrections against scipy oracles (SYS-17)."""

import numpy as np
import pytest
from scipy import stats

from autogen.analytics.hypothesis import (
    compare_samples,
    diagnostic_normality_check,
    holm_bonferroni,
    paired_permutation_test,
    paired_t_test,
    two_sample_permutation_test,
    wilcoxon_signed_rank,
)


def test_paired_t_test_matches_scipy_ttest_rel():
    rng = np.random.default_rng(404)
    for _ in range(15):
        s1 = rng.normal(loc=20.0, scale=4.0, size=25).tolist()
        s2 = [x + float(rng.normal(0.5, 1.0)) for x in s1]

        t_stat, p_val = paired_t_test(s1, s2)
        oracle = stats.ttest_rel(s1, s2)

        # t-statistic and p-value should match scipy oracle
        assert pytest.approx(t_stat, abs=1e-8) == float(oracle.statistic)
        assert pytest.approx(p_val, abs=1e-5) == float(oracle.pvalue)


def test_paired_permutation_test_significance():
    # Definitively separated paired samples
    s1 = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]
    s2 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    stat, p_val = paired_permutation_test(s1, s2, n_permutations=2000, seed=505)
    assert stat == 9.0  # mean difference
    assert p_val <= 0.01  # highly significant


def test_two_sample_permutation_test():
    s1 = [20.0, 21.0, 22.0, 23.0]
    s2 = [1.0, 2.0, 3.0, 4.0]
    stat, p_val = two_sample_permutation_test(s1, s2, n_permutations=2000, seed=606)
    assert stat > 0.0
    assert p_val <= 0.05


def test_wilcoxon_signed_rank_matches_direction():
    s1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    s2 = [x + 5.0 for x in s1]
    stat, p_val = wilcoxon_signed_rank(s2, s1)
    assert p_val <= 0.01


def test_diagnostic_normality_check():
    rng = np.random.default_rng(707)
    normal_data = rng.normal(loc=0.0, scale=1.0, size=100).tolist()
    is_norm, diag = diagnostic_normality_check(normal_data)
    assert is_norm is True
    assert "skewness" in diag
    assert "kurtosis" in diag

    # Exponentially distributed data (strongly skewed)
    skewed_data = rng.exponential(scale=2.0, size=100).tolist()
    is_norm_exp, diag_exp = diagnostic_normality_check(skewed_data)
    # Skewness should be elevated
    assert diag_exp["skewness"] > 0.8


def test_holm_bonferroni_correction_and_monotonicity():
    raw_p_values = [0.005, 0.01, 0.03, 0.04, 0.20]
    results = holm_bonferroni(raw_p_values, alpha=0.05)

    adj_p_vals = [r[0] for r in results]
    # Check monotonicity of adjusted p-values relative to raw sorted order
    sorted_pairs = sorted(zip(raw_p_values, adj_p_vals, strict=False), key=lambda x: x[0])
    for i in range(len(sorted_pairs) - 1):
        assert sorted_pairs[i][1] <= sorted_pairs[i + 1][1]

    # Verify first test is significant
    assert results[0][1] is True
    # Verify last test is not significant
    assert results[4][1] is False


def test_compare_samples_wrapper():
    baseline = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]
    candidate = [b + 5.0 + float(i) for i, b in enumerate(baseline)]

    res = compare_samples(sample_baseline=baseline, sample_candidate=candidate, paired=True)
    assert res.is_significant is True
    assert res.effect_size > 0.0
    assert "Significant" in res.verdict
