"""Linear and Logistic Regression models implemented from scratch (SYS-11, Module M12)."""

from __future__ import annotations

from typing import Literal, cast
import numpy as np

from autogen.intelligence.ml.contracts import Classifier, Regressor


class LinearRegression(Regressor):
    """Linear Regression with Ridge L2 regularization, supporting closed-form and iterative SGD solvers."""

    def __init__(
        self,
        alpha: float = 0.0,
        fit_intercept: bool = True,
        solver: Literal["normal_equation", "batch_gd", "sgd", "mini_batch_gd"] = "normal_equation",
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        batch_size: int = 32,
        tol: float = 1e-6,
        seed: int = 42,
    ) -> None:
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.tol = tol
        self.seed = seed

        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.n_iter_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> LinearRegression:
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {X.shape}")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError(f"y must be 1D array of shape ({X.shape[0]},)")

        N, D = X.shape

        if self.solver == "normal_equation":
            self._fit_normal_equation(X, y, N, D)
        else:
            self._fit_gradient_descent(X, y, N, D)

        return self

    def _fit_normal_equation(self, X: np.ndarray, y: np.ndarray, N: int, D: int) -> None:
        if self.fit_intercept:
            X_design = np.hstack([np.ones((N, 1), dtype=np.float64), X])
            # Do not penalize intercept (bias)
            reg_matrix = self.alpha * np.eye(D + 1, dtype=np.float64)
            reg_matrix[0, 0] = 0.0
            theta = np.linalg.solve(X_design.T @ X_design + reg_matrix, X_design.T @ y)
            self.intercept_ = float(theta[0])
            self.coef_ = theta[1:]
        else:
            reg_matrix = self.alpha * np.eye(D, dtype=np.float64)
            self.coef_ = np.linalg.solve(X.T @ X + reg_matrix, X.T @ y)
            self.intercept_ = 0.0
        self.n_iter_ = 1

    def _fit_gradient_descent(self, X: np.ndarray, y: np.ndarray, N: int, D: int) -> None:
        rng = np.random.default_rng(self.seed)
        weights = np.zeros(D, dtype=np.float64)
        bias = 0.0

        batch_sz = 1 if self.solver == "sgd" else (self.batch_size if self.solver == "mini_batch_gd" else N)

        for epoch in range(1, self.max_iter + 1):
            indices = np.arange(N)
            rng.shuffle(indices)
            w_old = weights.copy()

            for start in range(0, N, batch_sz):
                end = min(start + batch_sz, N)
                b_idx = indices[start:end]
                X_b = X[b_idx]
                y_b = y[b_idx]
                m = len(b_idx)

                y_pred = X_b @ weights + (bias if self.fit_intercept else 0.0)
                error = y_pred - y_b

                grad_w = (1.0 / m) * (X_b.T @ error) + self.alpha * weights
                weights -= self.learning_rate * grad_w

                if self.fit_intercept:
                    grad_b = (1.0 / m) * np.sum(error)
                    bias -= self.learning_rate * grad_b

            if np.max(np.abs(weights - w_old)) < self.tol:
                self.n_iter_ = epoch
                break
        else:
            self.n_iter_ = self.max_iter

        self.coef_ = weights
        self.intercept_ = bias

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("LinearRegression must be fitted before predict.")
        return cast(np.ndarray, X @ self.coef_ + self.intercept_)


class LogisticRegression(Classifier):
    """Logistic Regression for binary classification with L2 regularization and gradient descent."""

    def __init__(
        self,
        C: float = 1.0,
        fit_intercept: bool = True,
        learning_rate: float = 0.05,
        max_iter: int = 1000,
        batch_size: int = 32,
        tol: float = 1e-5,
        seed: int = 42,
    ) -> None:
        if C <= 0.0:
            raise ValueError(f"C must be positive, got {C}")
        self.C = C  # Inverse of regularization strength: alpha = 1 / C
        self.fit_intercept = fit_intercept
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.tol = tol
        self.seed = seed

        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.classes_: np.ndarray = np.array([0, 1], dtype=np.int32)
        self.n_iter_: int = 0

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        # Numerically stable sigmoid
        z_safe = np.clip(z, -250.0, 250.0)
        return cast(np.ndarray, 1.0 / (1.0 + np.exp(-z_safe)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> LogisticRegression:
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {X.shape}")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError(f"y must be 1D array of shape ({X.shape[0]},)")

        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(f"Binary logistic regression requires exactly 2 classes, got {len(self.classes_)}")

        # Normalize classes to 0 and 1
        y_binary = (y == self.classes_[1]).astype(np.float64)
        N, D = X.shape
        alpha = 1.0 / self.C

        rng = np.random.default_rng(self.seed)
        weights = np.zeros(D, dtype=np.float64)
        bias = 0.0

        for epoch in range(1, self.max_iter + 1):
            indices = np.arange(N)
            rng.shuffle(indices)
            w_old = weights.copy()

            for start in range(0, N, self.batch_size):
                end = min(start + self.batch_size, N)
                b_idx = indices[start:end]
                X_b = X[b_idx]
                y_b = y_binary[b_idx]
                m = len(b_idx)

                z = X_b @ weights + (bias if self.fit_intercept else 0.0)
                probs = self._sigmoid(z)
                errors = probs - y_b

                grad_w = (1.0 / m) * (X_b.T @ errors) + (alpha / N) * weights
                weights -= self.learning_rate * grad_w

                if self.fit_intercept:
                    grad_b = (1.0 / m) * np.sum(errors)
                    bias -= self.learning_rate * grad_b

            if np.max(np.abs(weights - w_old)) < self.tol:
                self.n_iter_ = epoch
                break
        else:
            self.n_iter_ = self.max_iter

        self.coef_ = weights
        self.intercept_ = bias
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("LogisticRegression must be fitted before predict_proba.")
        z = X @ self.coef_ + self.intercept_
        p1 = self._sigmoid(z)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        preds = (proba >= threshold).astype(int)
        return cast(np.ndarray, self.classes_[preds])
