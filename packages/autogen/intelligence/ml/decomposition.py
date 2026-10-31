"""Principal Component Analysis (PCA) via SVD/eigendecomposition from scratch (SYS-11, Module M12)."""

from __future__ import annotations

from typing import Any, cast
import numpy as np

from autogen.intelligence.ml.contracts import Transformer


class PCA(Transformer):
    """Principal Component Analysis (PCA) with explained variance and exact reconstruction."""

    def __init__(self, n_components: int | float | None = None) -> None:
        self.n_components = n_components

        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None  # shape (k, D)
        self.explained_variance_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None
        self.singular_values_: np.ndarray | None = None
        self.n_components_: int = 0
        self.n_samples_: int = 0
        self.n_features_: int = 0

    def fit(self, X: np.ndarray) -> PCA:
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {X.shape}")
        self.n_samples_, self.n_features_ = X.shape
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # SVD: X_c = U @ diag(S) @ Vt
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # Explained variance: S^2 / (N - 1)
        ddof = max(1, self.n_samples_ - 1)
        total_variance = np.sum(S ** 2) / ddof
        explained_variance = (S ** 2) / ddof
        explained_variance_ratio = (
            explained_variance / total_variance if total_variance > 0 else np.zeros_like(S)
        )

        # Determine number of components
        max_possible = min(self.n_samples_, self.n_features_)
        if self.n_components is None:
            k = max_possible
        elif isinstance(self.n_components, float) and 0.0 < self.n_components < 1.0:
            # Ratio of variance to keep
            cum_var = np.cumsum(explained_variance_ratio)
            k = int(np.searchsorted(cum_var, self.n_components) + 1)
        else:
            k = min(int(self.n_components), max_possible)

        self.n_components_ = k
        self.components_ = Vt[:k]  # shape (k, D)
        self.singular_values_ = S[:k]
        self.explained_variance_ = explained_variance[:k]
        self.explained_variance_ratio_ = explained_variance_ratio[:k]

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("PCA must be fitted before transform.")
        X_centered = X - self.mean_
        return cast(np.ndarray, X_centered @ self.components_.T)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X_projected: np.ndarray) -> np.ndarray:
        """Projects reduced representation back into original feature space."""
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("PCA must be fitted before inverse_transform.")
        return cast(np.ndarray, X_projected @ self.components_ + self.mean_)

    def reconstruction_error(self, X: np.ndarray) -> float:
        """Computes Frobenius norm of reconstruction residual ||X - X_reconstructed||^2 / N."""
        X_reconstructed = self.inverse_transform(self.transform(X))
        return float(np.mean(np.sum((X - X_reconstructed) ** 2, axis=1)))