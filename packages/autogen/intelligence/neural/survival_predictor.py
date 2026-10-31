"""Neural survival and mortality risk predictor for autonomous agents (SYS-13, Module M14)."""

from __future__ import annotations

import numpy as np

from autogen.intelligence.neural.activations import Linear, ReLU, Sigmoid
from autogen.intelligence.neural.layers import DenseLayer
from autogen.intelligence.neural.losses import BinaryCrossEntropy, MeanSquaredError
from autogen.intelligence.neural.mlp import NeuralNetwork
from autogen.intelligence.neural.optimizers import Adam


class NeuralSurvivalPredictor:
    """Predicts expected remaining agent lifespan and short-term mortality hazard risk."""

    def __init__(self, input_dim: int = 6, hidden_dim: int = 32, seed: int = 42) -> None:
        self.input_dim = input_dim

        # Lifespan regression network: (input_dim -> hidden_dim -> 1) with Linear output
        lifespan_layers = [
            DenseLayer(input_dim, hidden_dim, ReLU(), init_method="he", seed=seed),
            DenseLayer(hidden_dim, hidden_dim // 2, ReLU(), init_method="he", seed=seed + 1),
            DenseLayer(hidden_dim // 2, 1, Linear(), init_method="xavier", seed=seed + 2),
        ]
        self.lifespan_net = NeuralNetwork(lifespan_layers)

        # Mortality hazard network: (input_dim -> hidden_dim -> 1) with Sigmoid output
        hazard_layers = [
            DenseLayer(input_dim, hidden_dim, ReLU(), init_method="he", seed=seed + 10),
            DenseLayer(hidden_dim, 1, Sigmoid(), init_method="xavier", seed=seed + 11),
        ]
        self.hazard_net = NeuralNetwork(hazard_layers)

    def predict_lifespan(self, features: np.ndarray) -> np.ndarray:
        """Estimates expected remaining steps of survival."""
        if features.ndim == 1:
            features = features.reshape(1, -1)
        return self.lifespan_net.predict(features).squeeze(-1)

    def predict_mortality_risk(self, features: np.ndarray) -> np.ndarray:
        """Estimates probability P(death in imminent horizon) in [0, 1]."""
        if features.ndim == 1:
            features = features.reshape(1, -1)
        return self.hazard_net.predict(features).squeeze(-1)

    def fit_lifespan(
        self,
        features: np.ndarray,
        lifespans: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 0.005,
    ) -> list[dict[str, float]]:
        """Fits expected remaining lifespan regression model."""
        if lifespans.ndim == 1:
            lifespans = lifespans.reshape(-1, 1)
        loss_fn = MeanSquaredError()
        opt = Adam(lr=lr)
        return self.lifespan_net.fit(features, lifespans, loss_fn=loss_fn, optimizer=opt, epochs=epochs, batch_size=batch_size)

    def fit_hazard(
        self,
        features: np.ndarray,
        mortality_events: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 0.005,
    ) -> list[dict[str, float]]:
        """Fits binary mortality risk model using Binary Cross-Entropy."""
        if mortality_events.ndim == 1:
            mortality_events = mortality_events.reshape(-1, 1)
        loss_fn = BinaryCrossEntropy()
        opt = Adam(lr=lr)
        return self.hazard_net.fit(features, mortality_events, loss_fn=loss_fn, optimizer=opt, epochs=epochs, batch_size=batch_size)
