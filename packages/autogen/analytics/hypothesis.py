"""Hypothesis testing, assumption diagnostics, and multiple testing corrections from scratch (SYS-17)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from autogen.analytics.statistics import cliffs_delta, cohens_d, mean, standard_deviation


class HypothesisTestResult(BaseModel):
    """Structured report of a statistical hypothesis test."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    test_name: str
    statistic: float
    p_value: float = Field(ge=0.0, le=1.0)
    alpha: float = Field(ge=0.001, le=0.5, default=0.05)
    is_significant: bool
    effect_size_name: str
    effect_size: float
    assumptions_satisfied: bool
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    verdict: str


def betacf(a: float, b: float, x: float, max_iter: int = 150, eps: float = 1e-14) -> float:
    """Evaluates continued fraction for incomplete beta function using modified Lentz's method."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d

    for m in range(1, max_iter + 1):
        m2 = 2 * m
        # Even step
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c

        # Odd step
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        del_val = d * c
        h *= del_val
        if abs(del_val - 1.0) < eps:
            break

    return float(h)


def incomplete_beta(a: float, b: float, x: float) -> float:
    """Computes regularized incomplete beta function I_x(a, b)."""
    if x < 0.0 or x > 1.0:
        raise ValueError(f"x must be in [0, 1], got {x}")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0

    # Symmetry transformation for convergence
    log_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    factor = math.exp(a * math.log(x) + b * math.log(1.0 - x) - log_beta)

    if x < (a + 1.0) / (a + b + 2.0):
        return factor * betacf(a, b, x) / a
    else:
        return 1.0 - factor * betacf(b, a, 1.0 - x) / b


def student_t_pvalue(t_stat: float, df: int) -> float:
    """Calculates exact two-sided p-value for Student's t distribution: p = I_{df/(df+t^2)}(df/2, 1/2)."""
    if df < 1:
        raise ValueError(f"Degrees of freedom must be >= 1, got {df}")
    if math.isnan(t_stat):
        return 1.0
    if t_stat == 0.0:
        return 1.0

    t2 = t_stat * t_stat
    x = df / (df + t2)
    p = incomplete_beta(df / 2.0, 0.5, x)
    return float(min(1.0, max(0.0, p)))


def diagnostic_normality_check(data: Sequence[float]) -> tuple[bool, dict[str, float]]:
    """Evaluates sample skewness and kurtosis to assess plausibility of normality.

    Normality is flagged as plausible if |skewness| <= 1.0 and |excess kurtosis| <= 1.5.
    """
    n = len(data)
    if n < 4:
        return True, {"skewness": 0.0, "kurtosis": 0.0, "n": float(n)}

    m = mean(data)
    s2 = sum((x - m) ** 2 for x in data) / n
    if s2 < 1e-14:
        return True, {"skewness": 0.0, "kurtosis": 0.0, "n": float(n)}

    s = math.sqrt(s2)
    m3 = sum((x - m) ** 3 for x in data) / n
    m4 = sum((x - m) ** 4 for x in data) / n

    skewness = m3 / (s**3)
    kurtosis_excess = (m4 / (s**4)) - 3.0

    is_plausible = abs(skewness) <= 1.0 and abs(kurtosis_excess) <= 1.5
    return is_plausible, {
        "skewness": float(skewness),
        "kurtosis": float(kurtosis_excess),
        "n": float(n),
    }


def paired_t_test(sample1: Sequence[float], sample2: Sequence[float]) -> tuple[float, float]:
    """Calculates paired t-test: t = mean(diff) / (std(diff)/sqrt(N)) and exact two-sided p-value."""
    n1, n2 = len(sample1), len(sample2)
    if n1 != n2:
        raise ValueError(f"Paired t-test requires equal sample sizes: {n1} != {n2}")
    if n1 < 2:
        return 0.0, 1.0

    diffs = [float(x - y) for x, y in zip(sample1, sample2, strict=False)]
    d_mean = mean(diffs)
    d_std = standard_deviation(diffs, ddof=1)

    if d_std < 1e-14:
        # All differences identical
        if abs(d_mean) < 1e-14:
            return 0.0, 1.0
        # Complete constant shift
        return float(math.copysign(1e9, d_mean)), 0.0

    se = d_std / math.sqrt(n1)
    t_stat = d_mean / se
    p_val = student_t_pvalue(t_stat, df=n1 - 1)
    return float(t_stat), float(p_val)


def paired_permutation_test(
    sample1: Sequence[float],
    sample2: Sequence[float],
    n_permutations: int = 10000,
    seed: int = 42,
) -> tuple[float, float]:
    """Monte Carlo paired sign-flip permutation test for mean difference."""
    n1, n2 = len(sample1), len(sample2)
    if n1 != n2:
        raise ValueError(f"Paired permutation test requires equal sample sizes: {n1} != {n2}")
    if n1 == 0:
        return 0.0, 1.0

    diffs = np.asarray([float(x - y) for x, y in zip(sample1, sample2, strict=False)], dtype=np.float64)
    obs_diff = float(np.mean(diffs))

    if np.all(diffs == 0.0):
        return 0.0, 1.0

    rng = np.random.default_rng(seed)
    # Signs are randomly +/- 1
    signs = rng.choice([-1.0, 1.0], size=(n_permutations, n1))
    perm_means = np.mean(signs * diffs, axis=1)

    count_extreme = np.count_nonzero(np.abs(perm_means) >= abs(obs_diff) - 1e-14)
    p_val = (count_extreme + 1.0) / (n_permutations + 1.0)

    return float(obs_diff), float(min(1.0, max(0.0, p_val)))


def two_sample_permutation_test(
    sample1: Sequence[float],
    sample2: Sequence[float],
    n_permutations: int = 10000,
    seed: int = 42,
) -> tuple[float, float]:
    """Two-sample Monte Carlo permutation test for difference in means."""
    n1, n2 = len(sample1), len(sample2)
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0

    arr1 = np.asarray(sample1, dtype=np.float64)
    arr2 = np.asarray(sample2, dtype=np.float64)
    obs_diff = float(np.mean(arr1) - np.mean(arr2))

    pooled = np.concatenate([arr1, arr2])
    rng = np.random.default_rng(seed)
    count_extreme = 0

    for _ in range(n_permutations):
        shuffled = rng.permutation(pooled)
        perm_diff = np.mean(shuffled[:n1]) - np.mean(shuffled[n1:])
        if abs(perm_diff) >= abs(obs_diff) - 1e-14:
            count_extreme += 1

    p_val = (count_extreme + 1.0) / (n_permutations + 1.0)
    return float(obs_diff), float(min(1.0, max(0.0, p_val)))


def wilcoxon_signed_rank(
    sample1: Sequence[float],
    sample2: Sequence[float],
) -> tuple[float, float]:
    """Calculates Wilcoxon signed-rank test for paired samples with normal approximation."""
    n1, n2 = len(sample1), len(sample2)
    if n1 != n2:
        raise ValueError(f"Wilcoxon signed-rank test requires equal sample sizes: {n1} != {n2}")

    diffs = [float(x - y) for x, y in zip(sample1, sample2, strict=False)]
    nonzero_diffs = [d for d in diffs if abs(d) > 1e-12]
    n = len(nonzero_diffs)
    if n == 0:
        return 0.0, 1.0

    # Sort by absolute difference
    abs_diffs = [(abs(d), math.copysign(1.0, d)) for d in nonzero_diffs]
    abs_diffs.sort(key=lambda item: item[0])

    # Assign ranks with fractional tie resolution
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and abs(abs_diffs[j + 1][0] - abs_diffs[i][0]) < 1e-12:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1

    w_pos = sum(ranks[idx] for idx in range(n) if abs_diffs[idx][1] > 0)
    w_neg = sum(ranks[idx] for idx in range(n) if abs_diffs[idx][1] < 0)
    w_stat = min(w_pos, w_neg)

    # Normal approximation for n >= 10 (or asymptotic fallback)
    mean_w = n * (n + 1) / 4.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0
    se_w = math.sqrt(var_w)

    z = (w_stat - mean_w + 0.5) / se_w if w_stat < mean_w else (w_stat - mean_w - 0.5) / se_w
    p_val = math.erfc(abs(z) / math.sqrt(2.0))
    return float(w_stat), float(min(1.0, max(0.0, p_val)))


def holm_bonferroni(
    p_values: Sequence[float],
    alpha: float = 0.05,
) -> list[tuple[float, bool]]:
    """Applies Holm-Bonferroni step-down correction for multiple comparisons.

    Returns a list of (adjusted_p_value, is_significant) matching original input order.
    """
    m = len(p_values)
    if m == 0:
        return []

    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted: list[float] = [0.0] * m

    cum_max = 0.0
    for rank, (orig_idx, p_val) in enumerate(indexed):
        mult = m - rank
        adj = min(1.0, mult * p_val)
        cum_max = max(cum_max, adj)
        adjusted[orig_idx] = cum_max

    results = [(adj_p, adj_p <= alpha) for adj_p in adjusted]
    return results


def compare_samples(
    sample_baseline: Sequence[float],
    sample_candidate: Sequence[float],
    paired: bool = True,
    alpha: float = 0.05,
    seed: int = 42,
) -> HypothesisTestResult:
    """Rigorous comparison between baseline and candidate evaluation runs."""
    d_val = cohens_d(sample_candidate, sample_baseline, paired=paired)
    c_delta = cliffs_delta(sample_candidate, sample_baseline)

    diag: dict[str, Any]
    if paired:
        diffs = [float(c - b) for c, b in zip(sample_candidate, sample_baseline, strict=False)]
        is_normal, diag_diff = diagnostic_normality_check(diffs)
        diag = diag_diff
        if is_normal and len(diffs) >= 8:
            test_name = "Paired Student's t-test"
            stat, p_val = paired_t_test(sample_candidate, sample_baseline)
        else:
            test_name = "Paired Permutation Test"
            stat, p_val = paired_permutation_test(sample_candidate, sample_baseline, seed=seed)
    else:
        norm_b, diag_b = diagnostic_normality_check(sample_baseline)
        norm_c, diag_c = diagnostic_normality_check(sample_candidate)
        is_normal = norm_b and norm_c
        diag = {"baseline": diag_b, "candidate": diag_c}
        test_name = "Two-Sample Permutation Test"
        stat, p_val = two_sample_permutation_test(sample_candidate, sample_baseline, seed=seed)

    is_sig = bool(p_val <= alpha)
    verdict = (
        f"Significant difference detected (p={p_val:.4e} <= {alpha}, Cohen's d={d_val:.2f}, Cliff's delta={c_delta:.2f})."
        if is_sig
        else f"No statistically significant difference (p={p_val:.4e} > {alpha})."
    )

    return HypothesisTestResult(
        test_name=test_name,
        statistic=float(stat),
        p_value=float(p_val),
        alpha=alpha,
        is_significant=is_sig,
        effect_size_name="Cohen's d / Cliff's delta",
        effect_size=float(d_val),
        assumptions_satisfied=is_normal,
        diagnostics=diag,
        verdict=verdict,
    )
