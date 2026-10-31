"""PyTorch verification twin comparing pure NumPy backpropagation to PyTorch autograd (SYS-13, Module M14)."""

from __future__ import annotations

from typing import Any

import numpy as np

from autogen.intelligence.neural.activations import Linear, ReLU, Sigmoid, Tanh
from autogen.intelligence.neural.layers import DenseLayer
from autogen.intelligence.neural.mlp import NeuralNetwork

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    TORCH_AVAILABLE = False


def is_torch_available() -> bool:
    """Returns True if PyTorch is installed and importable."""
    return TORCH_AVAILABLE


class PyTorchTwin:
    """Twin network constructing a torch.nn.Module mirroring a NumPy NeuralNetwork."""

    def __init__(self, numpy_net: NeuralNetwork) -> None:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed in the active environment.")
        self.numpy_net = numpy_net
        self.torch_module = self._build_twin_module()

    def _build_twin_module(self) -> Any:
        modules: list[Any] = []
        for layer in self.numpy_net.layers:
            if isinstance(layer, DenseLayer):
                linear = nn.Linear(layer.in_features, layer.out_features, bias=True)
                # PyTorch stores weights as (out_features, in_features)
                with torch.no_grad():
                    linear.weight.copy_(torch.from_numpy(layer.W.T))
                    linear.bias.copy_(torch.from_numpy(layer.b.squeeze(0)))
                modules.append(linear)

                if isinstance(layer.activation, ReLU):
                    modules.append(nn.ReLU())
                elif isinstance(layer.activation, Sigmoid):
                    modules.append(nn.Sigmoid())
                elif isinstance(layer.activation, Tanh):
                    modules.append(nn.Tanh())
                elif isinstance(layer.activation, Linear):
                    pass
        return nn.Sequential(*modules).double()

    def compare_forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Compares forward pass output between NumPy net and PyTorch module."""
        np_out = self.numpy_net.forward(x, training=False)
        with torch.no_grad():
            torch_x = torch.from_numpy(x).double()
            torch_out = self.torch_module(torch_x).numpy()
        max_diff = float(np.max(np.abs(np_out - torch_out)))
        return np_out, torch_out, max_diff

    def compare_gradients(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> dict[str, float]:
        """Compares analytical gradients between NumPy backprop and PyTorch autograd."""
        from autogen.intelligence.neural.losses import MeanSquaredError

        # 1. NumPy backpropagation
        mse_np = MeanSquaredError()
        y_pred = self.numpy_net.forward(x, training=True)
        grad_out = mse_np.gradient(y, y_pred)
        self.numpy_net.zero_grad()
        self.numpy_net.backward(grad_out)

        # 2. PyTorch autograd
        torch_x = torch.from_numpy(x).double().requires_grad_(False)
        torch_y = torch.from_numpy(y).double()
        self.torch_module.zero_grad()
        torch_pred = self.torch_module(torch_x)
        # Note: PyTorch MSELoss computes mean((y_pred - y)^2), matching MeanSquaredError
        loss_torch = nn.MSELoss()(torch_pred, torch_y)
        loss_torch.backward()

        diffs: dict[str, float] = {}
        dense_idx = 0
        for mod in self.torch_module:
            if isinstance(mod, nn.Linear):
                np_layer = self.numpy_net.layers[dense_idx]
                torch_w_grad = mod.weight.grad.numpy().T
                torch_b_grad = mod.bias.grad.numpy()

                w_diff = float(np.max(np.abs(np_layer.grads["W"] - torch_w_grad)))
                b_diff = float(np.max(np.abs(np_layer.grads["b"].squeeze(0) - torch_b_grad)))

                diffs[f"layer_{dense_idx}_W_max_diff"] = w_diff
                diffs[f"layer_{dense_idx}_b_max_diff"] = b_diff
                dense_idx += 1

        return diffs
