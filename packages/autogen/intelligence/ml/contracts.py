"""Data contracts and interfaces for Classical Machine Learning Engine (SYS-11, Module M12)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, TypeVar, runtime_checkable
import numpy as np


class DataLeakageError(ValueError):
    """Raised when data from evaluation or test groups leaks into the training partition."""


@dataclass(frozen=True, slots=True)
class Dataset:
    """Tabular dataset with features X, targets y, and group episode identifiers."""

    features: np.ndarray          # shape (N, D), dtype float64
    targets: np.ndarray           # shape (N,) or (N, K)
    feature_names: tuple[str, ...] = ()
    group_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))

    def __post_init__(self) -> None:
        if self.features.ndim != 2:
            raise ValueError(f"Features must be 2D array (N, D), got shape {self.features.shape}")
        if self.targets.shape[0] != self.features.shape[0]:
            raise ValueError(f"Features N={self.features.shape[0]} != Targets N={self.targets.shape[0]}")
        if self.group_ids.size > 0 and self.group_ids.shape[0] != self.features.shape[0]:
            raise ValueError(f"Group IDs N={self.group_ids.shape[0]} != Features N={self.features.shape[0]}")

    @property
    def num_samples(self) -> int:
        return int(self.features.shape[0])

    @property
    def num_features(self) -> int:
        return int(self.features.shape[1])


@dataclass(frozen=True, slots=True)
class SplitResult:
    """Leakage-safe partition of indices and group sets."""

    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    train_groups: frozenset[Any]
    val_groups: frozenset[Any]
    test_groups: frozenset[Any]


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    """Metadata recorded for every trained model artifact."""

    model_name: str
    version: str
    hyperparameters: Mapping[str, Any]
    data_hash: str
    created_at_iso: str
    feature_names: tuple[str, ...] = ()


@runtime_checkable
class Estimator(Protocol):
    """Protocol for all ML estimators."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> Any:
        ...


@runtime_checkable
class Classifier(Estimator, Protocol):
    """Protocol for classification estimators."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        ...


@runtime_checkable
class Regressor(Estimator, Protocol):
    """Protocol for regression estimators."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        ...


@runtime_checkable
class Transformer(Protocol):
    """Protocol for feature transformers."""

    def fit(self, X: np.ndarray) -> Any:
        ...

    def transform(self, X: np.ndarray) -> np.ndarray:
        ...

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        ...


@runtime_checkable
class Clusterer(Protocol):
    """Protocol for unsupervised clustering algorithms."""

    def fit(self, X: np.ndarray) -> Any:
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

