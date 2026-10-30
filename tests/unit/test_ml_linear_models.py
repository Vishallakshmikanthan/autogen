"""Unit tests for Linear and Logistic Regression against scikit-learn test oracles (SYS-11, Module M12)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression as SklearnLogistic, Ridge as SklearnRidge

from autogen.intelligence.ml.linear import LinearRegression, LogisticRegression
from autogen.intelligence.ml.metrics import accuracy_score, r2_score


def test_linear_regression_matches_sklearn_ridge() -> None:
    """Verifies that our closed-form Ridge Normal Equation matches scikit-learn's Ridge solver."""
    rng = np.random.default_rng(42)
    N, D = 100, 5
    X = rng.normal(size=(N, D))
    true_w = np.array([1.5, -2.0, 0.5, 3.2, -1.1])
    true_b = 4.2
    y = X @ true_w + true_b + rng.normal(0, 0.1, size=N)

    alpha = 0.5

    # Our implementation
    model = LinearRegression(alpha=alpha, fit_intercept=True, solver="normal_equation")
    model.fit(X, y)

    # Scikit-learn oracle
    sk_model = SklearnRidge(alpha=alpha, fit_intercept=True)
    sk_model.fit(X, y)

    # Assert coefficients match within numerical tolerance
    assert np.max(np.abs(model.coef_ - sk_model.coef_)) < 1e-4
    assert abs(model.intercept_ - sk_model.intercept_) < 1e-4

    # Predictions match
    preds = model.predict(X)
    sk_preds = sk_model.predict(X)
    assert np.max(np.abs(preds - sk_preds)) < 1e-4


def test_linear_regression_beats_trivial_baseline() -> None:
    """Verifies that Linear Regression strictly outperforms a mean predictor baseline."""
    rng = np.random.default_rng(123)
    N, D = 200, 3
    X = rng.uniform(-5, 5, size=(N, D))
    y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + rng.normal(0, 0.5, size=N)

    X_train, X_test = X[:150], X[150:]
    y_train, y_test = y[:150], y[150:]

    model = LinearRegression(alpha=0.01)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    baseline_pred = np.full_like(y_test, np.mean(y_train))

    r2_model = r2_score(y_test, y_pred)
    r2_base = r2_score(y_test, baseline_pred)

    assert r2_model > 0.95
    assert r2_model > r2_base


def test_logistic_regression_matches_sklearn_and_beats_baseline() -> None:
    """Verifies Logistic Regression against scikit-learn LogisticRegression oracle."""
    rng = np.random.default_rng(999)
    N, D = 300, 4
    X = rng.normal(size=(N, D))
    # Separable hyperplane
    logits = 2.0 * X[:, 0] - 1.5 * X[:, 1] + X[:, 2]
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (probs >= 0.5).astype(int)

    X_train, X_test = X[:200], X[200:]
    y_train, y_test = y[:200], y[200:]

    C = 1.0
    model = LogisticRegression(C=C, learning_rate=0.1, max_iter=2000, seed=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    pred_probs = model.predict_proba(X_test)

    # Scikit-learn oracle
    sk_model = SklearnLogistic(C=C, solver="lbfgs")
    sk_model.fit(X_train, y_train)
    sk_preds = sk_model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    sk_acc = accuracy_score(y_test, sk_preds)

    # Trivial baseline: majority class
    majority_class = int(np.argmax(np.bincount(y_train)))
    base_acc = accuracy_score(y_test, np.full_like(y_test, majority_class))

    assert acc > 0.85
    assert acc > base_acc
    # Within small tolerance of sklearn L-BFGS
    assert abs(acc - sk_acc) <= 0.05
    assert pred_probs.shape == (len(X_test), 2)
