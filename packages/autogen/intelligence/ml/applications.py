"""Simulation applications for Classical ML: resource prediction, danger classification, trajectory clustering (SYS-11, Module M12)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

from autogen.intelligence.ml.clustering import KMeans
from autogen.intelligence.ml.contracts import Classifier, Regressor
from autogen.intelligence.ml.decomposition import PCA
from autogen.intelligence.ml.linear import LinearRegression, LogisticRegression
from autogen.intelligence.ml.trees import RandomForestClassifier, RandomForestRegressor


class ResourceYieldPredictor:
    """Predicts continuous food/resource yields in environmental cells from terrain features."""

    def __init__(self, model_type: str = "rf", seed: int = 42) -> None:
        self.model_type = model_type
        self.seed = seed
        self.model: Regressor
        if model_type == "linear":
            self.model = LinearRegression(alpha=0.1, solver="normal_equation")
        else:
            self.model = RandomForestRegressor(n_estimators=30, max_depth=6, seed=seed)

    def fit(self, X: np.ndarray, y: np.ndarray) -> ResourceYieldPredictor:
        self.model.fit(X, y)
        return self

    def predict_yield(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict(features)


class CellDangerClassifier:
    """Classifies environmental danger probability from sensor cues."""

    def __init__(self, model_type: str = "logistic", seed: int = 42) -> None:
        self.model_type = model_type
        self.seed = seed
        self.model: Classifier
        if model_type == "logistic":
            self.model = LogisticRegression(C=1.0, seed=seed)
        else:
            self.model = RandomForestClassifier(n_estimators=30, max_depth=6, seed=seed)

    def fit(self, X: np.ndarray, y: np.ndarray) -> CellDangerClassifier:
        self.model.fit(X, y)
        return self

    def predict_danger_probability(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(features)[:, 1]

    def is_dangerous(self, features: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return self.predict_danger_probability(features) >= threshold


class AgentTrajectoryClusterer:
    """Clusters agent behavioral regimes (Foraging, Patrolling, Idling) using PCA + KMeans."""

    def __init__(self, n_clusters: int = 3, n_pca_components: int = 2, seed: int = 42) -> None:
        self.n_clusters = n_clusters
        self.pca = PCA(n_components=n_pca_components)
        self.kmeans = KMeans(n_clusters=n_clusters, seed=seed)

    def fit(self, trajectory_features: np.ndarray) -> AgentTrajectoryClusterer:
        # Reduced features via PCA
        reduced = self.pca.fit_transform(trajectory_features)
        self.kmeans.fit(reduced)
        return self

    def cluster_trajectories(self, trajectory_features: np.ndarray) -> np.ndarray:
        reduced = self.pca.transform(trajectory_features)
        return self.kmeans.predict(reduced)


@dataclass
class LearnedPerceptionComponent:
    """Pluggable learned perception component integrating trained classifiers into agents."""

    danger_classifier: CellDangerClassifier | None = None
    resource_predictor: ResourceYieldPredictor | None = None

    def assess_danger(self, cell_features: np.ndarray) -> float:
        if self.danger_classifier is None:
            return 0.0
        if cell_features.ndim == 1:
            cell_features = cell_features[np.newaxis, :]
        return float(self.danger_classifier.predict_danger_probability(cell_features)[0])

    def estimate_yield(self, cell_features: np.ndarray) -> float:
        if self.resource_predictor is None:
            return 0.0
        if cell_features.ndim == 1:
            cell_features = cell_features[np.newaxis, :]
        return float(self.resource_predictor.predict_yield(cell_features)[0])
