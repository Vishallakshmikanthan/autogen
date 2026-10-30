"""Unit tests for Decision Trees and Random Forests against scikit-learn test oracles (SYS-11, Module M12)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier as SklearnRF
from sklearn.tree import DecisionTreeClassifier as SklearnDT

from autogen.intelligence.ml.metrics import accuracy_score, r2_score
from autogen.intelligence.ml.trees import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)


def test_decision_tree_classifier_learns_non_linear_boundary() -> None:
    """Verifies that DecisionTree solves a non-linear XOR pattern."""
    X = np.array([
        [-1.0, -1.0],
        [-1.0, 1.0],
        [1.0, -1.0],
        [1.0, 1.0],
    ])
    y = np.array([0, 1, 1, 0])

    dt = DecisionTreeClassifier(criterion="gini", max_depth=3, seed=42)
    dt.fit(X, y)
    preds = dt.predict(X)

    assert np.all(preds == y)
    probs = dt.predict_proba(X)
    assert probs.shape == (4, 2)


def test_random_forest_classifier_beats_baseline_and_compares_with_sklearn() -> None:
    """Verifies RandomForest on multi-cluster synthetic benchmark."""
    rng = np.random.default_rng(42)
    N = 250
    # Cluster 0
    X0 = rng.normal(loc=[-2.0, -2.0], scale=0.8, size=(N // 2, 2))
    # Cluster 1
    X1 = rng.normal(loc=[2.0, 2.0], scale=0.8, size=(N // 2, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * (N // 2) + [1] * (N // 2))

    # Shuffle
    perm = rng.permutation(N)
    X, y = X[perm], y[perm]

    X_train, X_test = X[:180], X[180:]
    y_train, y_test = y[:180], y[180:]

    rf = RandomForestClassifier(n_estimators=25, max_depth=5, seed=42)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)

    sk_rf = SklearnRF(n_estimators=25, max_depth=5, random_state=42)
    sk_rf.fit(X_train, y_train)
    sk_preds = sk_rf.predict(X_test)

    acc = accuracy_score(y_test, preds)
    sk_acc = accuracy_score(y_test, sk_preds)
    base_acc = accuracy_score(y_test, np.full_like(y_test, int(np.argmax(np.bincount(y_train)))))

    assert acc > 0.90
    assert acc > base_acc
    assert abs(acc - sk_acc) <= 0.05
    assert rf.oob_score_ is not None and rf.oob_score_ > 0.85


def test_decision_tree_and_random_forest_regressors() -> None:
    """Verifies DecisionTreeRegressor and RandomForestRegressor on non-linear sinusoidal function."""
    rng = np.random.default_rng(77)
    N = 150
    X = rng.uniform(-np.pi, np.pi, size=(N, 1))
    y = np.sin(X[:, 0]) + rng.normal(0, 0.1, size=N)

    X_train, X_test = X[:100], X[100:]
    y_train, y_test = y[:100], y[100:]

    dt_reg = DecisionTreeRegressor(max_depth=5, seed=42)
    dt_reg.fit(X_train, y_train)
    dt_preds = dt_reg.predict(X_test)

    rf_reg = RandomForestRegressor(n_estimators=20, max_depth=5, seed=42)
    rf_reg.fit(X_train, y_train)
    rf_preds = rf_reg.predict(X_test)

    base_r2 = r2_score(y_test, np.full_like(y_test, np.mean(y_train)))
    dt_r2 = r2_score(y_test, dt_preds)
    rf_r2 = r2_score(y_test, rf_preds)

    assert dt_r2 > 0.80
    assert rf_r2 > dt_r2  # Forest reduces variance of single tree
    assert rf_r2 > base_r2
