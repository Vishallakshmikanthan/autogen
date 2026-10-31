"""Multilayer Perceptron (MLP) and sequential neural network engine (SYS-13, Module M14)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from autogen.intelligence.neural.contracts import Layer, LossFunction, NetworkDiagnostics, Optimizer
from autogen.intelligence.neural.losses import MeanSquaredError
from autogen.intelligence.neural.optimizers import Adam, clip_grad_norm


class NeuralNetwork:
    """Sequential feed-forward neural network implemented from scratch in pure NumPy."""

    def __init__(self, layers: Sequence[Layer] | None = None) -> None:
        self.layers: list[Layer] = list(layers) if layers is not None else []
        self.training = True
        self.diagnostics_history: list[NetworkDiagnostics] = []

    def add(self, layer: Layer) -> None:
        """Appends a layer to the sequential architecture."""
        self.layers.append(layer)

    def forward(self, x: np.ndarray, training: bool | None = None) -> np.ndarray:
        """Executes forward pass sequentially through all layers."""
        is_train = self.training if training is None else training
        out = x
        for layer in self.layers:
            out = layer.forward(out, training=is_train)
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Executes backward backpropagation pass in reverse order."""
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def zero_grad(self) -> None:
        """Zeros all accumulated gradients across all layers."""
        for layer in self.layers:
            for g in layer.grads.values():
                g.fill(0.0)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        loss_fn: LossFunction | None = None,
        optimizer: Optimizer | None = None,
        epochs: int = 100,
        batch_size: int = 32,
        val_data: tuple[np.ndarray, np.ndarray] | None = None,
        patience: int | None = None,
        max_grad_norm: float | None = 5.0,
        seed: int = 42,
    ) -> list[dict[str, float]]:
        """Trains network using mini-batch gradient descent with optional early stopping."""
        loss_fn = loss_fn or MeanSquaredError()
        optimizer = optimizer or Adam(lr=0.001)
        rng = np.random.default_rng(seed)

        n_samples = X.shape[0]
        history: list[dict[str, float]] = []

        best_val_loss = float("inf")
        best_state = None
        no_improvement_epochs = 0

        step_counter = 0

        for epoch in range(epochs):
            self.training = True
            indices = np.arange(n_samples)
            rng.shuffle(indices)

            epoch_losses: list[float] = []

            for start_idx in range(0, n_samples, batch_size):
                batch_idx = indices[start_idx : start_idx + batch_size]
                X_batch, y_batch = X[batch_idx], y[batch_idx]

                # 1. Forward pass
                y_pred = self.forward(X_batch, training=True)
                loss_val = loss_fn.loss(y_batch, y_pred)
                epoch_losses.append(loss_val)

                # 2. Backward pass
                grad_out = loss_fn.gradient(y_batch, y_pred)
                self.zero_grad()
                self.backward(grad_out)

                # 3. Optional gradient clipping
                gnorm = 0.0
                if max_grad_norm is not None:
                    gnorm = clip_grad_norm(self.layers, max_grad_norm)

                # 4. Parameter update
                optimizer.step(self.layers)
                step_counter += 1

                diag = NetworkDiagnostics(
                    loss=float(loss_val),
                    gradient_norm=float(gnorm),
                    learning_rate=float(getattr(optimizer, "lr", 0.001)),
                    step=step_counter,
                )
                self.diagnostics_history.append(diag)

            mean_train_loss = float(np.mean(epoch_losses))
            epoch_record = {"epoch": float(epoch + 1), "train_loss": mean_train_loss}

            # Validation evaluation
            if val_data is not None:
                self.training = False
                X_val, y_val = val_data
                val_pred = self.forward(X_val, training=False)
                val_loss = loss_fn.loss(y_val, val_pred)
                epoch_record["val_loss"] = float(val_loss)

                # Early stopping check
                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    best_state = self.state_dict()
                    no_improvement_epochs = 0
                else:
                    no_improvement_epochs += 1
                    if patience is not None and no_improvement_epochs >= patience:
                        # Restore best parameters
                        if best_state is not None:
                            self.load_state_dict(best_state)
                        history.append(epoch_record)
                        break

            history.append(epoch_record)

        return history

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Performs inference with dropout and batch statistics in evaluation mode."""
        return self.forward(x, training=False)

    def state_dict(self) -> dict[str, Any]:
        """Returns deep-copied state dictionary of all layer parameter arrays."""
        state: dict[str, Any] = {}
        for idx, layer in enumerate(self.layers):
            state[f"layer_{idx}"] = {k: np.copy(v) for k, v in layer.params.items()}
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Loads layer parameters from state dictionary."""
        for idx, layer in enumerate(self.layers):
            key = f"layer_{idx}"
            if key in state_dict:
                for p_name, val in state_dict[key].items():
                    if p_name in layer.params:
                        layer.params[p_name][:] = val

    def save_checkpoint(self, path: str | Path) -> None:
        """Saves weights and architecture state to disk via NumPy npz."""

        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        flat_arrays: dict[str, np.ndarray] = {}
        for idx, layer in enumerate(self.layers):
            for p_name, val in layer.params.items():
                flat_arrays[f"layer_{idx}_{p_name}"] = val
        np.savez_compressed(filepath, **flat_arrays)  # type: ignore[arg-type]

    def load_checkpoint(self, path: str | Path) -> None:
        """Loads weights from disk npz checkpoint."""
        filepath = Path(path)
        data = np.load(filepath)
        for idx, layer in enumerate(self.layers):
            for p_name, val in layer.params.items():
                k = f"layer_{idx}_{p_name}"
                if k in data:
                    val[:] = data[k]
