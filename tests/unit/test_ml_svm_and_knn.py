"""Unit tests for KNN and SVM against scikit-learn test oracles (SYS-11, Module M12)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.neighbors import (
    KNeighborsClassifier as SklearnKNNClf,
    KNeighborsRegressor as SklearnKNNReg,
)
from sklearn.svm import SVC as SklearnSVC

from autogen.intelligence.ml.metrics import accuracy_score, r2_score
from autogen.intelligence.ml.neighbors import KNNClassifier, KNNRegressor
from autogen.intelligence.ml.svm import PegasosSVM


def test_knn_classifier_matches_sklearn() -> None:
    """Verifies that our KNNClassifier predictions match scikit-learn's KNeighborsClassifier."""
    rng = np.random.default_rng(42)
    N = 100
    X = rng.normal(size=(N, 3))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    knn = KNNClassifier(n_neighbors=5, weights="uniform")
    knn.fit(X, y)
    preds = knn.predict(X)

    sk_knn = SklearnKNNClf(n_neighbors=5, weights="uniform")
    sk_knn.fit(X, y)
    sk_preds = sk_knn.predict(X)

    # Must match 100% on identical nearest neighbors
    assert np.all(preds == sk_preds)


def test_knn_regressor_matches_sklearn() -> None:
    """Verifies that our KNNRegressor predictions match scikit-learn's KNeighborsRegressor."""
    rng = np.random.default_rng(123)
    N = 80
    X = rng.normal(size=(N, 2))
    y = X[:, 0] * 2.0 - X[:, 1]

    knn_reg = KNNRegressor(n_neighbors=5, weights="uniform")
    knn_reg.fit(X, y)
    preds = knn_reg.predict(X)

    sk_knn = SklearnKNNReg(n_neighbors=5, weights="uniform")
    sk_knn.fit(X, y)
    sk_preds = sk_knn.predict(X)

    assert np.max(np.abs(preds - sk_preds)) < 1e-10


def test_pegasos_svm_learns_linear_and_rbf_margins() -> None:
    """Verifies Pegasos SVM on linear and non-linear RBF separable data."""
    rng = np.random.default_rng(42)
    # Linearly separable clusters
    c1 = rng.normal(loc=[-2.0, -2.0], scale=0.5, size=(40, 2))
    c2 = rng.normal(loc=[2.0, 2.0], scale=0.5, size=(40, 2))
    X_lin = np.vstack([c1, c2])
    y_lin = np.array([0] * 40 + [1] * 40)

    svm_lin = PegasosSVM(lambda_reg=0.01, kernel="linear", max_iter=1500, seed=42)
    svm_lin.fit(X_lin, y_lin)
    preds_lin = svm_lin.predict(X_lin)
    assert accuracy_score(y_lin, preds_lin) >= 0.95

    # RBF kernel on circular boundary (x^2 + y^2 > 1)
    pts = rng.uniform(-2.0, 2.0, size=(100, 2))
    y_rbf = (pts[:, 0]**2 + pts[:, 1]**2 > 1.2).astype(int)

    svm_rbf = PegasosSVM(lambda_reg=0.05, kernel="rbf", gamma=1.0, max_iter=2000, seed=42)
    svm_rbf.fit(pts, y_rbf)
    preds_rbf = svm_rbf.predict(pts)
    assert accuracy_score(y_rbf, preds_rbf) >= 0.85

