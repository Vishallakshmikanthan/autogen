"""Unsupervised Clustering: K-Means (k-means++) and Gaussian Mixture Models (EM) (SYS-11, Module M12)."""

from __future__ import annotations

from typing import Literal, cast
import numpy as np

from autogen.intelligence.ml.contracts import Clusterer
from autogen.intelligence.ml.neighbors import pairwise_euclidean_distances


class KMeans(Clusterer):
    """K-Means clustering with k-means++ initialization and monotonic inertia tracking."""

    def __init__(
        self,
        n_clusters: int = 8,
        init: Literal["k-means++", "random"] = "k-means++",
        max_iter: int = 300,
        tol: float = 1e-5,
        seed: int = 42,
    ) -> None:
        if n_clusters < 1:
            raise ValueError(f"n_clusters must be >= 1, got {n_clusters}")
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed

        self.cluster_centers_: np.ndarray | None = None
        self.inertia_: float = 0.0
        self.inertia_history_: list[float] = []
        self.n_iter_: int = 0

    def _init_centroids(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        N, D = X.shape
        if self.init == "random":
            indices = rng.choice(N, size=self.n_clusters, replace=False)
            return X[indices].copy()

        # k-means++ initialization
        centroids = np.zeros((self.n_clusters, D), dtype=np.float64)
        first_idx = int(rng.integers(0, N))
        centroids[0] = X[first_idx]

        for k in range(1, self.n_clusters):
            # Compute distance of each point to nearest existing centroid
            dists = pairwise_euclidean_distances(X, centroids[:k])
            min_dist_sq = np.min(dists, axis=1) ** 2

            total_dist_sq = np.sum(min_dist_sq)
            if total_dist_sq > 0:
                probs = min_dist_sq / total_dist_sq
                next_idx = int(rng.choice(N, p=probs))
            else:
                next_idx = int(rng.integers(0, N))
            centroids[k] = X[next_idx]

        return centroids

    def fit(self, X: np.ndarray) -> KMeans:
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {X.shape}")
        N, D = X.shape
        if N < self.n_clusters:
            raise ValueError(f"Number of samples N={N} < n_clusters={self.n_clusters}")

        rng = np.random.default_rng(self.seed)
        centroids = self._init_centroids(X, rng)
        self.inertia_history_ = []

        for it in range(1, self.max_iter + 1):
            # 1. Assignment step
            dists = pairwise_euclidean_distances(X, centroids)
            labels = np.argmin(dists, axis=1)
            min_dists = np.min(dists, axis=1)
            inertia = float(np.sum(min_dists ** 2))
            self.inertia_history_.append(inertia)

            # 2. Update step
            new_centroids = np.zeros_like(centroids)
            for k in range(self.n_clusters):
                cluster_pts = X[labels == k]
                if len(cluster_pts) > 0:
                    new_centroids[k] = np.mean(cluster_pts, axis=0)
                else:
                    # Empty cluster remedy: sample with maximum distance
                    furthest_idx = int(np.argmax(min_dists))
                    new_centroids[k] = X[furthest_idx]

            center_shift = float(np.max(np.linalg.norm(new_centroids - centroids, axis=1)))
            centroids = new_centroids

            if center_shift < self.tol:
                self.n_iter_ = it
                break
        else:
            self.n_iter_ = self.max_iter

        self.cluster_centers_ = centroids
        self.inertia_ = self.inertia_history_[-1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.cluster_centers_ is None:
            raise RuntimeError("KMeans must be fitted before predict.")
        dists = pairwise_euclidean_distances(X, self.cluster_centers_)
        return cast(np.ndarray, np.argmin(dists, axis=1))

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transforms X to cluster-distance space."""
        if self.cluster_centers_ is None:
            raise RuntimeError("KMeans must be fitted before transform.")
        return pairwise_euclidean_distances(X, self.cluster_centers_)


class GaussianMixtureModel(Clusterer):
    """Gaussian Mixture Model fitted via Expectation-Maximization (EM) with log-sum-exp stabilization."""

    def __init__(
        self,
        n_components: int = 3,
        covariance_type: Literal["diag", "full"] = "full",
        max_iter: int = 100,
        tol: float = 1e-4,
        reg_covar: float = 1e-6,
        seed: int = 42,
    ) -> None:
        if n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}")
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.seed = seed

        self.weights_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.covariances_: np.ndarray | None = None
        self.log_likelihood_history_: list[float] = []
        self.converged_: bool = False
        self.n_iter_: int = 0

    def _log_gaussian_pdf(self, X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        N, D = X.shape
        diff = X - mean
        reg_cov = cov + self.reg_covar * np.eye(D)

        try:
            L = np.linalg.cholesky(reg_cov)
            # solve L y = diff.T
            y = np.linalg.solve(L, diff.T)
            maha = np.sum(y ** 2, axis=0)
            log_det = 2.0 * np.sum(np.log(np.diag(L)))
        except np.linalg.LinAlgError:
            # Fallback if Cholesky fails on ill-conditioned matrix
            sign, log_det = np.linalg.slogdet(reg_cov)
            inv_cov = np.linalg.pinv(reg_cov)
            maha = np.sum((diff @ inv_cov) * diff, axis=1)

        log_prob = -0.5 * (D * np.log(2.0 * np.pi) + log_det + maha)
        return cast(np.ndarray, log_prob)

    def fit(self, X: np.ndarray) -> GaussianMixtureModel:
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {X.shape}")
        N, D = X.shape
        rng = np.random.default_rng(self.seed)

        # Initialize means via KMeans
        km = KMeans(n_clusters=self.n_components, seed=self.seed)
        km.fit(X)
        if km.cluster_centers_ is None:
            raise RuntimeError("KMeans initialization failed to produce cluster centers.")
        self.means_ = km.cluster_centers_.copy()
        self.weights_ = np.ones(self.n_components, dtype=np.float64) / self.n_components

        if self.covariance_type == "full":
            self.covariances_ = np.zeros((self.n_components, D, D), dtype=np.float64)
            for k in range(self.n_components):
                self.covariances_[k] = np.eye(D, dtype=np.float64)
        else:
            self.covariances_ = np.ones((self.n_components, D), dtype=np.float64)

        prev_ll = -np.inf
        self.log_likelihood_history_ = []

        for it in range(1, self.max_iter + 1):
            assert self.covariances_ is not None and self.means_ is not None and self.weights_ is not None
            # 1. E-step: calculate log-responsibilities via log-sum-exp
            log_resp = np.zeros((N, self.n_components), dtype=np.float64)
            for k in range(self.n_components):
                if self.covariance_type == "full":
                    cov_k = self.covariances_[k]
                else:
                    cov_k = np.diag(self.covariances_[k])
                log_resp[:, k] = np.log(self.weights_[k] + 1e-15) + self._log_gaussian_pdf(
                    X, self.means_[k], cov_k
                )

            # Log-sum-exp over components
            max_log = np.max(log_resp, axis=1, keepdims=True)
            log_prob_x = max_log + np.log(np.sum(np.exp(log_resp - max_log), axis=1, keepdims=True))
            current_ll = float(np.sum(log_prob_x))
            self.log_likelihood_history_.append(current_ll)

            # Responsibilities gamma_ik
            gamma = np.exp(log_resp - log_prob_x)

            # 2. M-step: update parameters
            N_k = np.sum(gamma, axis=0) + 1e-15
            self.weights_ = N_k / N

            for k in range(self.n_components):
                self.means_[k] = np.sum(gamma[:, k:k+1] * X, axis=0) / N_k[k]
                diff = X - self.means_[k]

                if self.covariance_type == "full":
                    self.covariances_[k] = (gamma[:, k:k+1] * diff).T @ diff / N_k[k]
                    self.covariances_[k] += self.reg_covar * np.eye(D)
                else:
                    self.covariances_[k] = np.sum(gamma[:, k:k+1] * (diff ** 2), axis=0) / N_k[k]
                    self.covariances_[k] += self.reg_covar

            # Check convergence
            if abs(current_ll - prev_ll) < self.tol:
                self.converged_ = True
                self.n_iter_ = it
                break
            prev_ll = current_ll
        else:
            self.converged_ = False
            self.n_iter_ = self.max_iter

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.means_ is None or self.covariances_ is None or self.weights_ is None:
            raise RuntimeError("GMM must be fitted before predict.")
        N, D = X.shape
        log_resp = np.zeros((N, self.n_components), dtype=np.float64)
        for k in range(self.n_components):
            cov_k = self.covariances_[k] if self.covariance_type == "full" else np.diag(self.covariances_[k])
            log_resp[:, k] = np.log(self.weights_[k] + 1e-15) + self._log_gaussian_pdf(
                X, self.means_[k], cov_k
            )
        max_log = np.max(log_resp, axis=1, keepdims=True)
        log_prob_x = max_log + np.log(np.sum(np.exp(log_resp - max_log), axis=1, keepdims=True))
        return cast(np.ndarray, np.exp(log_resp - log_prob_x))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return cast(np.ndarray, np.argmax(self.predict_proba(X), axis=1))