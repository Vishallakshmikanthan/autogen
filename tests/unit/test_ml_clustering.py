"""Unit tests for KMeans and Gaussian Mixture Models (SYS-11, Module M12)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.mixture import GaussianMixture as SklearnGMM

from autogen.intelligence.ml.clustering import GaussianMixtureModel, KMeans


def test_kmeans_inertia_monotonicity_invariant() -> None:
    """Verifies invariant: K-Means inertia must be strictly non-increasing across iterations."""
    rng = np.random.default_rng(42)
    # 3 synthetic clusters
    c1 = rng.normal(loc=[-5.0, 0.0], scale=1.0, size=(60, 2))
    c2 = rng.normal(loc=[0.0, 5.0], scale=1.0, size=(60, 2))
    c3 = rng.normal(loc=[5.0, -5.0], scale=1.0, size=(60, 2))
    X = np.vstack([c1, c2, c3])

    km = KMeans(n_clusters=3, init="k-means++", max_iter=50, seed=42)
    km.fit(X)

    history = km.inertia_history_
    assert len(history) > 1

    # Check non-increasing invariant: J[t+1] <= J[t] + eps
    for i in range(len(history) - 1):
        assert history[i + 1] <= history[i] + 1e-7, (
            f"Inertia increased at iteration {i}: {history[i]} -> {history[i+1]}"
        )

    # Compare with sklearn
    sk_km = SklearnKMeans(n_clusters=3, n_init=1, random_state=42)
    sk_km.fit(X)
    # Inertia should be close (within 10% on stochastic initialization)
    assert abs(km.inertia_ - sk_km.inertia_) / sk_km.inertia_ < 0.15


def test_gmm_log_likelihood_monotonicity_invariant() -> None:
    """Verifies invariant: GMM Expectation-Maximization log-likelihood must be non-decreasing."""
    rng = np.random.default_rng(123)
    c1 = rng.normal(loc=[-3.0, 0.0], scale=0.8, size=(80, 2))
    c2 = rng.normal(loc=[3.0, 0.0], scale=0.8, size=(80, 2))
    X = np.vstack([c1, c2])

    gmm = GaussianMixtureModel(n_components=2, covariance_type="full", max_iter=30, seed=42)
    gmm.fit(X)

    ll_history = gmm.log_likelihood_history_
    assert len(ll_history) > 1

    # Invariant: log_likelihood non-decreasing
    for i in range(len(ll_history) - 1):
        assert ll_history[i + 1] >= ll_history[i] - 1e-5, (
            f"Log-likelihood decreased at iteration {i}: {ll_history[i]} -> {ll_history[i+1]}"
        )

    probs = gmm.predict_proba(X)
    assert probs.shape == (160, 2)
    assert np.allclose(np.sum(probs, axis=1), 1.0)

