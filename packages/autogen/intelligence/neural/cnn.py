"""Convolutional Neural Network for local terrain classification (SYS-13, Module M14)."""

from __future__ import annotations

import numpy as np

from autogen.intelligence.neural.activations import ReLU, Softmax
from autogen.intelligence.neural.layers import Conv2DLayer, DenseLayer, FlattenLayer
from autogen.intelligence.neural.losses import CategoricalCrossEntropy
from autogen.intelligence.neural.mlp import NeuralNetwork
from autogen.intelligence.neural.optimizers import Adam


class CNNTerrainClassifier:
    """Convolutional classifier recognizing terrain cell types from local spatial observation patches."""

    TERRAIN_CLASSES = ("EMPTY", "OBSTACLE", "FOOD", "WATER", "HAZARD")

    def __init__(
        self,
        in_channels: int = 1,
        patch_size: int = 5,
        num_classes: int = 5,
        seed: int = 42,
    ) -> None:
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.num_classes = num_classes

        # Architecture:
        # Input: (N, 1, 5, 5)
        # Conv2D(1 -> 4, kernel=3, padding=0): -> (N, 4, 3, 3)
        # Flatten: -> (N, 36)
        # Dense(36 -> 16, ReLU): -> (N, 16)
        # Dense(16 -> num_classes, Softmax): -> (N, num_classes)
        conv = Conv2DLayer(in_channels=in_channels, out_channels=4, kernel_size=3, padding=0, activation=ReLU(), seed=seed)
        conv_out_dim = (patch_size - 3 + 1) * (patch_size - 3 + 1) * 4  # 3 * 3 * 4 = 36
        flatten = FlattenLayer()
        hidden = DenseLayer(in_features=conv_out_dim, out_features=16, activation=ReLU(), init_method="he", seed=seed + 1)
        output = DenseLayer(in_features=16, out_features=num_classes, activation=Softmax(), init_method="xavier", seed=seed + 2)

        self.net = NeuralNetwork([conv, flatten, hidden, output])

    def predict_proba(self, patch: np.ndarray) -> np.ndarray:
        """Returns class probabilities for a single patch or batch of patches."""
        if patch.ndim == 2:
            # (H, W) -> (1, 1, H, W)
            x = patch[np.newaxis, np.newaxis, ...].astype(np.float64)
        elif patch.ndim == 3:
            # (C, H, W) -> (1, C, H, W)
            x = patch[np.newaxis, ...].astype(np.float64)
        else:
            x = patch.astype(np.float64)
        return self.net.predict(x)

    def classify(self, patch: np.ndarray) -> int:
        """Returns integer class index of predicted terrain category."""
        proba = self.predict_proba(patch)
        return int(np.argmax(proba[0]))

    def classify_name(self, patch: np.ndarray) -> str:
        """Returns human-readable terrain name."""
        idx = self.classify(patch)
        if idx < len(self.TERRAIN_CLASSES):
            return self.TERRAIN_CLASSES[idx]
        return f"UNKNOWN_{idx}"

    def fit(
        self,
        X_patches: np.ndarray,
        y_labels: np.ndarray,
        epochs: int = 50,
        batch_size: int = 16,
        lr: float = 0.005,
    ) -> list[dict[str, float]]:
        """Fits CNN weights on patch dataset."""
        if X_patches.ndim == 3:
            X_patches = X_patches[:, np.newaxis, :, :]

        # Convert scalar class indices to one-hot vectors if necessary
        if y_labels.ndim == 1:
            n_samples = y_labels.shape[0]
            one_hot = np.zeros((n_samples, self.num_classes), dtype=np.float64)
            one_hot[np.arange(n_samples), y_labels.astype(int)] = 1.0
            y_data = one_hot
        else:
            y_data = y_labels.astype(np.float64)

        loss_fn = CategoricalCrossEntropy()
        optimizer = Adam(lr=lr)
        return self.net.fit(X_patches, y_data, loss_fn=loss_fn, optimizer=optimizer, epochs=epochs, batch_size=batch_size)
