"""Unit tests for Principal Component Analysis (PCA) against scikit-learn test oracle (SYS-11, Module M12)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.decomposition import PCA as SklearnPCA

from autogen.intelligence.ml.decomposition import PCA


def test_pca_matches_sklearn_and_orthonormality() -> None:
    """Verifies that our PCA singular values and explained variance match scikit-learn exactly."""
    rng = np.random.default_rng(42)
    N, D = 120, 6
    X = rng.normal(size=(N, D))
    # Inject correlated linear structure
    X[:, 2] = 2.0 * X[:, 0] + 0.5 * X[:, 1]
    X[:, 5] = -1.5 * X[:, 3]

    pca = PCA(n_components=4)
    pca.fit(X)

    sk_pca = SklearnPCA(n_components=4)
    sk_pca.fit(X)

    # 1. Orthonormality check: V_k @ V_k.T == I_k
    assert pca.components_ is not None
    identity_k = pca.components_ @ pca.components_.T
    assert np.max(np.abs(identity_k - np.eye(4))) < 1e-10

    # 2. Singular values match sklearn
    assert pca.singular_values_ is not None
    assert np.max(np.abs(pca.singular_values_ - sk_pca.singular_values_)) < 1e-6

    # 3. Explained variance ratio match
    assert pca.explained_variance_ratio_ is not None
    assert np.max(np.abs(pca.explained_variance_ratio_ - sk_pca.explained_variance_ratio_)) < 1e-6


def test_pca_reconstruction_error_non_increasing_invariant() -> None:
    """Verifies invariant: reconstruction error strictly decreases or stays equal as components increase."""
    rng = np.random.default_rng(99)
    N, D = 100, 5
    X = rng.normal(size=(N, D))

    errors = []
    for k in range(1, D + 1):
        pca_k = PCA(n_components=k)
        pca_k.fit(X)
        err = pca_k.reconstruction_error(X)
        errors.append(err)

    for i in range(len(errors) - 1):
        assert errors[i + 1] <= errors[i] + 1e-9, (
            f"Reconstruction error increased from k={i+1} to k={i+2}: {errors[i]} -> {errors[i+1]}"
        )

    # At full components k=D, reconstruction error should be approximately zero
    assert errors[-1] < 1e-10
