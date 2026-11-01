"""Property-based invariant tests for statistical calculations (SYS-17)."""

import math

from hypothesis import given, settings
from hypothesis import strategies as st

from autogen.analytics.hypothesis import holm_bonferroni
from autogen.analytics.statistics import (
    cliffs_delta,
    sample_variance,
    standard_deviation,
    standard_error,
)


@settings(max_examples=100)
@given(st.lists(st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False), min_size=2, max_size=50))
def test_sample_variance_non_negative(data: list[float]):
    v = sample_variance(data, ddof=1)
    s = standard_deviation(data, ddof=1)
    se = standard_error(data)

    assert v >= 0.0
    assert s >= 0.0
    assert se >= 0.0
    assert math.isclose(s * s, v, rel_tol=1e-5, abs_tol=1e-8)


@settings(max_examples=100)
@given(
    st.lists(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False), min_size=1, max_size=25),
    st.lists(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False), min_size=1, max_size=25),
)
def test_cliffs_delta_bounded_interval(g1: list[float], g2: list[float]):
    delta = cliffs_delta(g1, g2)
    assert -1.0 <= delta <= 1.0

    # Invariance: delta(A, B) == -delta(B, A)
    reverse_delta = cliffs_delta(g2, g1)
    assert math.isclose(delta, -reverse_delta, abs_tol=1e-9)


@settings(max_examples=50)
@given(
    st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False), min_size=1, max_size=20),
    st.floats(min_value=0.01, max_value=0.10),
)
def test_holm_bonferroni_invariants(p_vals: list[float], alpha: float):
    results = holm_bonferroni(p_vals, alpha=alpha)
    assert len(results) == len(p_vals)

    for adj_p, is_sig in results:
        assert 0.0 <= adj_p <= 1.0
        assert is_sig == (adj_p <= alpha)
