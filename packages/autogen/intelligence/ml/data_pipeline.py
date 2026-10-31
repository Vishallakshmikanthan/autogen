"""Data pipeline, feature transformers, and leakage-safe dataset splitting (SYS-11, Module M12)."""

from __future__ import annotations

from typing import Any, Sequence, cast
import numpy as np

from autogen.intelligence.ml.contracts import (
    DataLeakageError,
    Dataset,
    SplitResult,
    Transformer,
)


class StandardScaler(Transformer):
    """Standardizes features by removing the mean and scaling to unit variance."""

    def __init__(self, with_mean: bool = True, with_std: bool = True, epsilon: float = 1e-12) -> None:
        self.with_mean = with_mean
        self.with_std = with_std
        self.epsilon = epsilon
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> StandardScaler:
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X.shape}")
        if self.with_mean:
            self.mean_ = np.mean(X, axis=0)
        else:
            self.mean_ = np.zeros(X.shape[1], dtype=np.float64)

        if self.with_std:
            variance = np.var(X, axis=0)
            scale = np.sqrt(variance + self.epsilon)
            # Guard against zero variance columns
            scale[scale < self.epsilon] = 1.0
            self.scale_ = scale
        else:
            self.scale_ = np.ones(X.shape[1], dtype=np.float64)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler must be fitted before transforming data.")
        if X.ndim != 2 or X.shape[1] != self.mean_.shape[0]:
            raise ValueError(f"Feature dimension mismatch: expected {self.mean_.shape[0]}, got {X.shape[1]}")
        return cast(np.ndarray, (X - self.mean_) / self.scale_)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class MinMaxScaler(Transformer):
    """Transforms features by scaling each feature to a given range [min, max]."""

    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0), epsilon: float = 1e-12) -> None:
        self.feature_range = feature_range
        self.epsilon = epsilon
        self.data_min_: np.ndarray | None = None
        self.data_max_: np.ndarray | None = None
        self.data_range_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> MinMaxScaler:
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X.shape}")
        d_min = np.min(X, axis=0)
        d_max = np.max(X, axis=0)
        self.data_min_ = d_min
        self.data_max_ = d_max
        self.data_range_ = np.maximum(d_max - d_min, self.epsilon)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.data_min_ is None or self.data_max_ is None or self.data_range_ is None:
            raise RuntimeError("MinMaxScaler must be fitted before transforming data.")
        r_min, r_max = self.feature_range
        scaled = (X - self.data_min_) / self.data_range_
        return cast(np.ndarray, scaled * (r_max - r_min) + r_min)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class PolynomialFeatures(Transformer):
    """Generates polynomial and interaction features up to degree d."""

    def __init__(self, degree: int = 2, include_bias: bool = True) -> None:
        if degree < 1:
            raise ValueError("Degree must be at least 1")
        self.degree = degree
        self.include_bias = include_bias
        self.num_input_features_: int | None = None

    def fit(self, X: np.ndarray) -> PolynomialFeatures:
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X.shape}")
        self.num_input_features_ = X.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.num_input_features_ is None:
            raise RuntimeError("PolynomialFeatures must be fitted before transform.")
        N, D = X.shape
        cols: list[np.ndarray] = []
        if self.include_bias:
            cols.append(np.ones((N, 1), dtype=np.float64))

        # Degree 1 (original features)
        cols.append(X)

        if self.degree >= 2:
            # Degree 2 combinations
            for i in range(D):
                for j in range(i, D):
                    cols.append((X[:, i] * X[:, j])[:, np.newaxis])

        if self.degree >= 3:
            # Degree 3 combinations
            for i in range(D):
                for j in range(i, D):
                    for k in range(j, D):
                        cols.append((X[:, i] * X[:, j] * X[:, k])[:, np.newaxis])

        return np.hstack(cols)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def check_data_leakage(train_groups: frozenset[Any], test_groups: frozenset[Any]) -> None:
    """Verifies that train and test group sets are strictly disjoint."""
    overlap = train_groups.intersection(test_groups)
    if overlap:
        raise DataLeakageError(
            f"Data leakage detected! Overlapping simulation groups found in both train and test partitions: {overlap}"
        )


class GroupRunSplitter:
    """Splits simulation datasets by group/run/seed ID, ensuring temporal auto-correlation cannot leak."""

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> None:
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {total}")
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def split(self, dataset: Dataset) -> SplitResult:
        if dataset.group_ids.size == 0:
            raise ValueError("Dataset has no group_ids; cannot perform leakage-safe group split.")

        unique_groups = np.unique(dataset.group_ids)
        if len(unique_groups) < 3:
            raise ValueError(
                f"At least 3 unique group runs required for train/val/test split, got {len(unique_groups)}"
            )

        rng = np.random.default_rng(self.seed)
        shuffled_groups = unique_groups.copy()
        rng.shuffle(shuffled_groups)

        n_groups = len(shuffled_groups)
        n_train = max(1, int(np.floor(n_groups * self.train_ratio)))
        n_val = max(1, int(np.floor(n_groups * self.val_ratio)))

        # Ensure at least 1 test group
        if n_train + n_val >= n_groups:
            if n_train > 1:
                n_train -= 1
            elif n_val > 1:
                n_val -= 1

        train_grp_set = frozenset(shuffled_groups[:n_train])
        val_grp_set = frozenset(shuffled_groups[n_train : n_train + n_val])
        test_grp_set = frozenset(shuffled_groups[n_train + n_val :])

        # Active leakage guard check
        check_data_leakage(train_grp_set, test_grp_set)
        check_data_leakage(train_grp_set, val_grp_set)
        check_data_leakage(val_grp_set, test_grp_set)

        train_idx = np.where(np.isin(dataset.group_ids, list(train_grp_set)))[0]
        val_idx = np.where(np.isin(dataset.group_ids, list(val_grp_set)))[0]
        test_idx = np.where(np.isin(dataset.group_ids, list(test_grp_set)))[0]

        return SplitResult(
            train_indices=train_idx,
            val_indices=val_idx,
            test_indices=test_idx,
            train_groups=train_grp_set,
            val_groups=val_grp_set,
            test_groups=test_grp_set,
        )


def group_k_fold(dataset: Dataset, n_splits: int = 5, seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generates K-Fold train/test index splits grouped by simulation run/seed."""
    unique_groups = np.unique(dataset.group_ids)
    if len(unique_groups) < n_splits:
        raise ValueError(f"Cannot perform {n_splits}-fold CV with only {len(unique_groups)} groups.")

    rng = np.random.default_rng(seed)
    shuffled_groups = unique_groups.copy()
    rng.shuffle(shuffled_groups)

    group_folds = np.array_split(shuffled_groups, n_splits)
    splits: list[tuple[np.ndarray, np.ndarray]] = []

    for fold_idx in range(n_splits):
        test_grps = frozenset(group_folds[fold_idx])
        train_grps = frozenset(g for i, gf in enumerate(group_folds) if i != fold_idx for g in gf)

        # Leakage guard
        check_data_leakage(train_grps, test_grps)

        train_indices = np.where(np.isin(dataset.group_ids, list(train_grps)))[0]
        test_indices = np.where(np.isin(dataset.group_ids, list(test_grps)))[0]
        splits.append((train_indices, test_indices))

    return splits