"""Unit tests for ML data pipeline, feature scaling, and leakage prevention (SYS-11, Module M12)."""

from __future__ import annotations

import numpy as np
import pytest

from autogen.intelligence.ml.contracts import DataLeakageError, Dataset
from autogen.intelligence.ml.data_pipeline import (
    GroupRunSplitter,
    MinMaxScaler,
    PolynomialFeatures,
    StandardScaler,
    check_data_leakage,
    group_k_fold,
)


def test_standard_scaler_and_zero_variance_protection() -> None:
    """Verifies that StandardScaler centers mean to 0, scale to 1, and handles constant columns."""
    X = np.array([
        [1.0, 5.0, 10.0],
        [2.0, 5.0, 20.0],
        [3.0, 5.0, 30.0],
        [4.0, 5.0, 40.0],
    ])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Column 0: standard normal
    assert abs(np.mean(X_scaled[:, 0])) < 1e-10
    assert abs(np.std(X_scaled[:, 0]) - 1.0) < 1e-10

    # Column 1: constant 5.0 -> scale set to 1.0, mean is 5.0 -> scaled values are 0.0
    assert np.all(X_scaled[:, 1] == 0.0)

    # Transform on unseen data uses train mean/scale
    X_unseen = np.array([[2.5, 5.0, 25.0]])
    X_trans = scaler.transform(X_unseen)
    assert abs(X_trans[0, 0]) < 1e-10  # 2.5 is the train mean for col 0


def test_min_max_scaler() -> None:
    """Verifies that MinMaxScaler scales features exactly to [min, max]."""
    X = np.array([
        [10.0, -5.0],
        [20.0, 0.0],
        [30.0, 5.0],
    ])
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    X_scaled = scaler.fit_transform(X)

    assert np.all(X_scaled >= 0.0)
    assert np.all(X_scaled <= 1.0)
    assert abs(X_scaled[0, 0] - 0.0) < 1e-10
    assert abs(X_scaled[2, 0] - 1.0) < 1e-10


def test_polynomial_features() -> None:
    """Verifies generation of polynomial interaction features."""
    X = np.array([
        [2.0, 3.0],
        [1.0, 4.0],
    ])
    poly = PolynomialFeatures(degree=2, include_bias=True)
    X_poly = poly.fit_transform(X)
    # Expected columns: bias (1), x1, x2, x1^2, x1*x2, x2^2 -> 6 columns
    assert X_poly.shape == (2, 6)
    assert np.all(X_poly[:, 0] == 1.0)
    assert X_poly[0, 3] == 4.0   # 2^2
    assert X_poly[0, 4] == 6.0   # 2*3
    assert X_poly[0, 5] == 9.0   # 3^2


def test_leakage_guard_raises_on_overlap() -> None:
    """Verifies that deliberate data leakage raises DataLeakageError."""
    train_groups = frozenset([1, 2, 3])
    leaky_test_groups = frozenset([3, 4, 5])

    with pytest.raises(DataLeakageError, match="Data leakage detected"):
        check_data_leakage(train_groups, leaky_test_groups)


def test_group_run_splitter_strictly_disjoint() -> None:
    """Verifies that GroupRunSplitter produces completely disjoint groups between train/val/test."""
    n_runs = 20
    ticks = 10
    total = n_runs * ticks
    group_ids = np.repeat(np.arange(n_runs), ticks)
    X = np.random.randn(total, 4)
    y = np.random.randn(total)

    dataset = Dataset(features=X, targets=y, group_ids=group_ids)
    splitter = GroupRunSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=123)
    split = splitter.split(dataset)

    assert split.train_groups.isdisjoint(split.val_groups)
    assert split.train_groups.isdisjoint(split.test_groups)
    assert split.val_groups.isdisjoint(split.test_groups)

    # Check total group coverage
    all_groups = split.train_groups.union(split.val_groups).union(split.test_groups)
    assert len(all_groups) == n_runs


def test_group_k_fold_splits() -> None:
    """Verifies that group_k_fold creates mutually exclusive folds with zero cross-contamination."""
    n_runs = 15
    ticks = 5
    group_ids = np.repeat(np.arange(n_runs), ticks)
    dataset = Dataset(features=np.zeros((n_runs * ticks, 2)), targets=np.zeros(n_runs * ticks), group_ids=group_ids)

    folds = group_k_fold(dataset, n_splits=5, seed=42)
    assert len(folds) == 5

    for train_idx, test_idx in folds:
        train_grps = set(group_ids[train_idx])
        test_grps = set(group_ids[test_idx])
        assert train_grps.isdisjoint(test_grps)

