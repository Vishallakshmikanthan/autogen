"""Support Vector Machines (SVM) via Pegasos subgradient descent and kernels (SYS-11, Module M12)."""

from __future__ import annotations

from typing import Literal, cast
import numpy as np

from autogen.intelligence.ml.contracts import Classifier
from autogen.intelligence.ml.neighbors import pairwise_euclidean_distances


class PegasosSVM(Classifier):
    """Soft-margin Support Vector Machine using Pegasos (Primal Estimated sub-GrAdient SOlver for SVM).

    Supports linear margins and kernel formulations (RBF, Polynomial).
    """

    def __init__(
        self,
        lambda_reg: float = 0.01,
        kernel: Literal["linear", "rbf", "poly"] = "linear",
        gamma: float | Literal["scale"] = "scale",
        degree: int = 3,
        max_iter: int = 2000,
        batch_size: int = 1,
        seed: int = 42,
    ) -> None:
        if lambda_reg <= 0.0:
            raise ValueError(f"lambda_reg must be positive, got {lambda_reg}")
        self.lambda_reg = lambda_reg
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.seed = seed

        self.gamma_val_: float = 1.0
        self.classes_: np.ndarray = np.array([-1, 1], dtype=np.int32)
        self.X_train_: np.ndarray | None = None
        self.alpha_: np.ndarray | None = None  # Dual coefficients for kernel SVM
        self.w_: np.ndarray | None = None      # Primal weight vector for linear SVM
        self.b_: float = 0.0

    def _compute_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        if self.kernel == "linear":
            return cast(np.ndarray, X1 @ X2.T)
        elif self.kernel == "rbf":
            dist_sq = pairwise_euclidean_distances(X1, X2) ** 2
            return cast(np.ndarray, np.exp(-self.gamma_val_ * dist_sq))
        elif self.kernel == "poly":
            return cast(np.ndarray, (self.gamma_val_ * (X1 @ X2.T) + 1.0) ** self.degree)
        raise ValueError(f"Unsupported kernel {self.kernel}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> PegasosSVM:
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got {X.shape}")
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(f"Binary SVM requires 2 classes, got {len(self.classes_)}")

        N, D = X.shape
        # Map labels to {-1, +1}
        y_pm = np.where(y == self.classes_[1], 1.0, -1.0)

        # Determine gamma
        if self.gamma == "scale":
            var_x = float(np.var(X))
            self.gamma_val_ = 1.0 / (D * var_x) if var_x > 0 else 1.0
        else:
            self.gamma_val_ = float(self.gamma)

        rng = np.random.default_rng(self.seed)

        if self.kernel == "linear":
            # Primal Pegasos in feature space
            w = np.zeros(D, dtype=np.float64)
            b = 0.0

            for t in range(1, self.max_iter + 1):
                eta = 1.0 / (self.lambda_reg * t)
                # Sample batch
                b_idx = rng.choice(N, size=self.batch_size, replace=False)
                X_b = X[b_idx]
                y_b = y_pm[b_idx]

                # Check margin constraint y_i * (w^T x_i + b) < 1
                margins = y_b * (X_b @ w + b)
                violating = margins < 1.0

                grad_w = self.lambda_reg * w
                grad_b = 0.0
                if np.any(violating):
                    grad_w -= np.mean(y_b[violating, np.newaxis] * X_b[violating], axis=0)
                    grad_b -= float(np.mean(y_b[violating]))

                w -= eta * grad_w
                b -= eta * grad_b

                # Projection step
                norm_w = np.linalg.norm(w)
                radius = 1.0 / np.sqrt(self.lambda_reg)
                if norm_w > radius:
                    w = (radius / norm_w) * w

            self.w_ = w
            self.b_ = b
        else:
            # Kernelized Pegasos using Gram matrix
            K = self._compute_kernel(X, X)
            alpha = np.zeros(N, dtype=np.float64)
            b = 0.0

            for t in range(1, self.max_iter + 1):
                eta = 1.0 / (self.lambda_reg * t)
                i_t = int(rng.integers(0, N))
                decision_val = (alpha * y_pm) @ K[:, i_t] + b

                if y_pm[i_t] * decision_val < 1.0:
                    alpha[i_t] += eta
                    b += eta * y_pm[i_t]

            self.X_train_ = X.copy()
            self.alpha_ = alpha * y_pm
            self.b_ = b

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.kernel == "linear":
            if self.w_ is None:
                raise RuntimeError("PegasosSVM must be fitted before decision_function.")
            return cast(np.ndarray, X @ self.w_ + self.b_)
        else:
            if self.X_train_ is None or self.alpha_ is None:
                raise RuntimeError("PegasosSVM must be fitted before decision_function.")
            K_test = self._compute_kernel(X, self.X_train_)
            return cast(np.ndarray, K_test @ self.alpha_ + self.b_)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Platt-style sigmoid approximation mapping decision values to calibrated probabilities."""
        df = self.decision_function(X)
        p1 = 1.0 / (1.0 + np.exp(-np.clip(df, -250.0, 250.0)))
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        df = self.decision_function(X)
        preds = np.where(df >= 0.0, self.classes_[1], self.classes_[0])
        return cast(np.ndarray, preds)
