"""Twin oracle tests comparing NumPy MLP against PyTorch when torch is present (SYS-13, Module M14)."""

from __future__ import annotations

import numpy as np
import pytest

from autogen.intelligence.neural.activations import ReLU, Sigmoid
from autogen.intelligence.neural.layers import DenseLayer
from autogen.intelligence.neural.mlp import NeuralNetwork
from autogen.intelligence.neural.torch_twin import PyTorchTwin, is_torch_available


@pytest.mark.mathematical
def test_torch_twin_availability():
    """Verifies that is_torch_available accurately detects PyTorch import capability."""
    avail = is_torch_available()
    assert isinstance(avail, bool)


@pytest.mark.mathematical
def test_torch_twin_forward_and_gradient_equivalence():
    """When PyTorch is installed, asserts NumPy MLP produces identical forward outputs and autograd gradients."""
    if not is_torch_available():
        pytest.skip("PyTorch is not installed in the active environment; skipping twin comparison test.")

    # 4 -> 6 (ReLU) -> 2 (Sigmoid)
    net = NeuralNetwork([
        DenseLayer(4, 6, activation=ReLU(), seed=42),
        DenseLayer(6, 2, activation=Sigmoid(), seed=43),
    ])

    twin = PyTorchTwin(net)

    x = np.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
    ], dtype=np.float64)

    y = np.array([
        [0.2, 0.8],
        [0.7, 0.3],
    ], dtype=np.float64)

    # Forward comparison
    np_out, torch_out, max_diff = twin.compare_forward(x)
    assert max_diff < 1e-6, f"Forward output max difference {max_diff} >= 1e-6"

    # Backward gradient comparison
    diffs = twin.compare_gradients(x, y)
    for name, diff in diffs.items():
        assert diff < 1e-6, f"Gradient {name} max difference {diff} >= 1e-6"
