"""K-Nearest Neighbors (KNN) classifier and regressor from scratch (SYS-11, Module M12)."""

from __future__ import annotations

from typing import Literal, cast
import numpy as np

from autogen.intelligence.ml.contracts import Classifier, Regressor


def pairwise_euclidean_distances(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Computes pairwise Euclidean distances between rows of X (N, D) and Y (M, D)."""
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x y^T
    x_norm_sq = np.sum(X ** 2, axis=1, keepdims=True)
    y_norm_sq = np.sum(Y ** 2, axis=1, keepdims=True)
    dist_sq = x_norm_sq + y_norm_sq.T - 2.0 * (X @ Y.T)
    # Numerical safeguard against tiny negative values
    dist_sq = np.maximum(dist_sq, 0.0)
    return cast(np.ndarray, np.sqrt(dist_sq))


class KNNClassifier(Classifier):
    """K-Nearest Neighbors Classifier supporting uniform and inverse distance weighting."""

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: Literal["uniform", "distance"] = "uniform",
        epsilon: float = 1e-10,
    ) -> None:
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be >= 1")
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.epsilon = epsilon

        self.X_train_: np.ndarray | None = None
        self.y_train_: np.ndarray | None = None
        self.classes_: np.ndarray = np.array([])
        self.n_classes_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> KNNClassifier:
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {X.shape}")
        self.X_train_ = X.copy()
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        self.y_train_ = np.array([class_to_idx[c] for c in y], dtype=np.int32)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.X_train_ is None or self.y_train_ is None:
            raise RuntimeError("KNNClassifier must be fitted before predict.")

        k = min(self.n_neighbors, len(self.X_train_))
        dists = pairwise_euclidean_distances(X, self.X_train_)
        neighbor_idx = np.argsort(dists, axis=1)[:, :k]

        N = X.shape[0]
        probs = np.zeros((N, self.n_classes_), dtype=np.float64)

        for i in range(N):
            n_indices = neighbor_idx[i]
            n_labels = self.y_train_[n_indices]

            if self.weights == "uniform":
                counts = np.bincount(n_labels, minlength=self.n_classes_)
                probs[i] = counts / k
            else:  # distance
                n_dists = dists[i, n_indices]
                w = 1.0 / (n_dists + self.epsilon)
                total_w = np.sum(w)
                for label, weight in zip(n_labels, w):
                    probs[i, label] += weight / total_w

        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        pred_idx = np.argmax(probs, axis=1)
        return cast(np.ndarray, self.classes_[pred_idx])


class KNNRegressor(Regressor):
    """K-Nearest Neighbors Regressor supporting uniform and inverse distance weighting."""

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: Literal["uniform", "distance"] = "uniform",
        epsilon: float = 1e-10,
    ) -> None:
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be >= 1")
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.epsilon = epsilon

        self.X_train_: np.ndarray | None = None
        self.y_train_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> KNNRegressor:
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {X.shape}")
        self.X_train_ = X.copy()
        self.y_train_ = y.astype(np.float64).copy()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.X_train_ is None or self.y_train_ is None:
            raise RuntimeError("KNNRegressor must be fitted before predict.")

        k = min(self.n_neighbors, len(self.X_train_))
        dists = pairwise_euclidean_distances(X, self.X_train_)
        neighbor_idx = np.argsort(dists, axis=1)[:, :k]

        N = X.shape[0]
        preds = np.zeros(N, dtype=np.float64)

        for i in range(N):
            n_indices = neighbor_idx[i]
            n_targets = self.y_train_[n_indices]

            if self.weights == "uniform":
                preds[i] = np.mean(n_targets)
            else:
                n_dists = dists[i, n_indices]
                w = 1.0 / (n_dists + self.epsilon)
                preds[i] = np.sum(w * n_targets) / np.sum(w)

        return preds