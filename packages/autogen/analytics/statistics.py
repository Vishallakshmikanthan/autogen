"""Descriptive statistics, effect sizes, and bootstrap estimation from scratch (SYS-17)."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class SampleStats(BaseModel):
    """Container for rigorous summary statistics of an empirical sample."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n: int = Field(ge=0)
    mean: float
    variance: float = Field(ge=0.0)
    std_dev: float = Field(ge=0.0)
    standard_error: float = Field(ge=0.0)
    median: float
    iqr: float = Field(ge=0.0)
    ci_lower: float
    ci_upper: float
    confidence_level: float = Field(ge=0.5, le=1.0)


def normal_quantile(p: float) -> float:
    """Computes the standard normal inverse CDF Phi^-1(p) using Acklam's rational approximation.

    Accurate to within 1.15e-9 across the domain (0, 1).
    """
    if p <= 0.0 or p >= 1.0:
        raise ValueError(f"Probability p must be in (0, 1), got {p}")

    # Coefficients in rational approximations
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        return float(num / den)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        num = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        den = (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        return float(num / den)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        num = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        return float(num / den)


def student_t_quantile(df: int, p: float) -> float:
    """Calculates the critical value t_(p, df) for Student's t distribution.

    Uses exact closed forms for df=1 (Cauchy) and df=2, and Hill-Davis
    Cornish-Fisher asymptotic series expansion for df >= 3.
    """
    if df < 1:
        raise ValueError(f"Degrees of freedom df must be >= 1, got {df}")
    if p <= 0.0 or p >= 1.0:
        raise ValueError(f"Probability p must be in (0, 1), got {p}")

    # Exact closed forms
    if df == 1:
        return float(math.tan(math.pi * (p - 0.5)))
    if df == 2:
        return float((2.0 * p - 1.0) / math.sqrt(2.0 * p * (1.0 - p)))

    # Standard two-tailed 95% (p=0.975) exact values for small degrees of freedom
    exact_t_0975: dict[int, float] = {
        3: 3.182446305284263,
        4: 2.7764451051977987,
        5: 2.5705818366147395,
        6: 2.4469118511449685,
        7: 2.3646242510103,
        8: 2.3060041350333704,
        9: 2.2621571627409915,
        10: 2.2281388519649385,
    }
    if abs(p - 0.975) < 1e-6 and df in exact_t_0975:
        return exact_t_0975[df]

    # Cornish-Fisher asymptotic expansion from standard normal quantile z
    z = normal_quantile(p)
    if df > 500:
        return z

    z2 = z * z
    z3 = z2 * z
    z5 = z3 * z2
    z7 = z5 * z2

    term1 = z
    term2 = (z3 + z) / (4.0 * df)
    term3 = (5.0 * z5 + 16.0 * z3 + 3.0 * z) / (96.0 * df * df)
    term4 = (3.0 * z7 + 19.0 * z5 + 17.0 * z3 - 15.0 * z) / (384.0 * df * df * df)

    return float(term1 + term2 + term3 + term4)


def mean(data: Sequence[float]) -> float:
    """Calculates sample mean: bar(x) = (1/N) * sum(x_i)."""
    if len(data) == 0:
        return 0.0
    return float(sum(data) / len(data))


def sample_variance(data: Sequence[float], ddof: int = 1) -> float:
    """Calculates unbiased sample variance with Bessel's correction: s^2 = sum(x - bar(x))^2 / (N - ddof)."""
    n = len(data)
    if n <= ddof:
        return 0.0
    m = mean(data)
    ss = sum((x - m) ** 2 for x in data)
    return float(max(0.0, ss / (n - ddof)))


def standard_deviation(data: Sequence[float], ddof: int = 1) -> float:
    """Calculates sample standard deviation: s = sqrt(s^2)."""
    return float(math.sqrt(sample_variance(data, ddof=ddof)))


def standard_error(data: Sequence[float]) -> float:
    """Calculates sample standard error: SE = s / sqrt(N)."""
    n = len(data)
    if n <= 1:
        return 0.0
    s = standard_deviation(data, ddof=1)
    return float(s / math.sqrt(n))


def median(data: Sequence[float]) -> float:
    """Calculates sample median from scratch."""
    if len(data) == 0:
        return 0.0
    sorted_d = sorted(data)
    n = len(sorted_d)
    mid = n // 2
    if n % 2 == 1:
        return float(sorted_d[mid])
    return float((sorted_d[mid - 1] + sorted_d[mid]) / 2.0)


def interquartile_range(data: Sequence[float]) -> float:
    """Calculates Interquartile Range (IQR = Q3 - Q1) using linear rank interpolation."""
    if len(data) < 2:
        return 0.0
    arr = np.asarray(data, dtype=np.float64)
    q25 = float(np.percentile(arr, 25))
    q75 = float(np.percentile(arr, 75))
    return float(max(0.0, q75 - q25))


def confidence_interval(
    data: Sequence[float],
    confidence: float = 0.95,
    method: Literal["t", "normal"] = "t",
) -> tuple[float, float]:
    """Calculates two-sided confidence interval for the sample mean."""
    n = len(data)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        val = float(data[0])
        return val, val

    m = mean(data)
    se = standard_error(data)
    if se < 1e-12:
        return m, m

    alpha = 1.0 - confidence
    p = 1.0 - alpha / 2.0

    if method == "normal" or n >= 120:
        crit = normal_quantile(p)
    else:
        crit = student_t_quantile(df=n - 1, p=p)

    margin = crit * se
    return float(m - margin), float(m + margin)


def bootstrap_ci(
    data: Sequence[float],
    stat_fn: Callable[[Sequence[float]], float] | None = None,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:
    """Computes empirical percentile bootstrap confidence interval with explicit RNG."""
    n = len(data)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        val = float(data[0])
        return val, val

    eval_fn = stat_fn if stat_fn is not None else mean
    arr = np.asarray(data, dtype=np.float64)
    rng = np.random.default_rng(seed)

    boot_estimates = np.empty(n_resamples, dtype=np.float64)
    indices = rng.integers(0, n, size=(n_resamples, n))

    for b in range(n_resamples):
        sample = arr[indices[b]]
        boot_estimates[b] = eval_fn(sample)

    alpha = 1.0 - confidence
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    ci_l = float(np.percentile(boot_estimates, lower_pct))
    ci_u = float(np.percentile(boot_estimates, upper_pct))
    return ci_l, ci_u


def cohens_d(group1: Sequence[float], group2: Sequence[float], paired: bool = False) -> float:
    """Calculates Cohen's d effect size between two groups.

    For independent samples:
        s_pooled = sqrt(((n1-1)*s1^2 + (n2-1)*s2^2) / (n1 + n2 - 2))
        d = (mean1 - mean2) / s_pooled
    For paired samples:
        d_z = mean(diff) / std(diff)
    """
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0

    if paired:
        if n1 != n2:
            raise ValueError(f"Paired Cohen's d requires equal sample sizes: {n1} != {n2}")
        diffs = [float(x - y) for x, y in zip(group1, group2, strict=False)]
        s_diff = standard_deviation(diffs, ddof=1)
        if s_diff < 1e-12:
            return 0.0
        return float(mean(diffs) / s_diff)

    m1, m2 = mean(group1), mean(group2)
    v1 = sample_variance(group1, ddof=1)
    v2 = sample_variance(group2, ddof=1)

    pooled_var = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
    if pooled_var < 1e-12:
        return 0.0

    pooled_std = math.sqrt(pooled_var)
    return float((m1 - m2) / pooled_std)


def cliffs_delta(group1: Sequence[float], group2: Sequence[float]) -> float:
    """Calculates Cliff's delta non-parametric effect size: delta in [-1, 1].

    delta = #(x_i > y_j) - #(x_i < y_j) / (n1 * n2)
    """
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return 0.0

    arr1 = np.asarray(group1, dtype=np.float64)
    arr2 = np.asarray(group2, dtype=np.float64)

    # Vectorized outer comparison
    diff_matrix = np.subtract.outer(arr1, arr2)
    greater = np.count_nonzero(diff_matrix > 0)
    less = np.count_nonzero(diff_matrix < 0)

    delta = (greater - less) / (n1 * n2)
    return float(min(1.0, max(-1.0, delta)))


def sample_stats(data: Sequence[float], confidence: float = 0.95) -> SampleStats:
    """Builds a complete, validated SampleStats DTO for an observed sample."""
    n = len(data)
    m = mean(data)
    v = sample_variance(data, ddof=1) if n > 1 else 0.0
    s = standard_deviation(data, ddof=1) if n > 1 else 0.0
    se = standard_error(data) if n > 1 else 0.0
    med = median(data)
    iqr_val = interquartile_range(data)
    ci_l, ci_u = confidence_interval(data, confidence=confidence)

    return SampleStats(
        n=n,
        mean=m,
        variance=v,
        std_dev=s,
        standard_error=se,
        median=med,
        iqr=iqr_val,
        ci_lower=ci_l,
        ci_upper=ci_u,
        confidence_level=confidence,
    )
