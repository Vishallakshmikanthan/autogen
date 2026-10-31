"""Numerical finite-difference gradient verification tests for Module M14 (SYS-13)."""

from __future__ import annotations

import numpy as np
import pytest

from autogen.intelligence.neural.activations import LeakyReLU, Linear, ReLU, Sigmoid, Tanh
from autogen.intelligence.neural.layers import Conv2DLayer, DenseLayer
from autogen.intelligence.neural.losses import (
    BinaryCrossEntropy,
    HuberLoss,
    MeanSquaredError,
)
from autogen.intelligence.neural.mlp import NeuralNetwork


def compute_numerical_gradient(
    net: NeuralNetwork,
    x: np.ndarray,
    y: np.ndarray,
    loss_fn,
    eps: float = 1e-7,
) -> dict[str, dict[str, np.ndarray]]:
    """Computes two-sided finite difference gradient dL/d(theta) in float64."""
    num_grads: dict[str, dict[str, np.ndarray]] = {}

    for idx, layer in enumerate(net.layers):
        l_key = f"layer_{idx}"
        num_grads[l_key] = {}
        for p_name, param in layer.params.items():
            grad_arr = np.zeros_like(param, dtype=np.float64)
            it = np.nditer(param, flags=["multi_index"])
            while not it.finished:
                idx_tuple = it.multi_index
                orig_val = param[idx_tuple]

                # theta + eps
                param[idx_tuple] = orig_val + eps
                pred_plus = net.forward(x, training=False)
                loss_plus = loss_fn.loss(y, pred_plus)

                # theta - eps
                param[idx_tuple] = orig_val - eps
                pred_minus = net.forward(x, training=False)
                loss_minus = loss_fn.loss(y, pred_minus)

                # restore
                param[idx_tuple] = orig_val

                # Symmetric difference quotient
                grad_arr[idx_tuple] = (loss_plus - loss_minus) / (2.0 * eps)
                it.iternext()

            num_grads[l_key][p_name] = grad_arr

    return num_grads


def relative_error(g_analytical: np.ndarray, g_numerical: np.ndarray) -> float:
    """Computes norm-relative error: ||g_a - g_n|| / (||g_a|| + ||g_n||)."""
    diff_norm = np.linalg.norm(g_analytical - g_numerical)
    sum_norm = np.linalg.norm(g_analytical) + np.linalg.norm(g_numerical)
    if sum_norm < 1e-12:
        return 0.0
    return float(diff_norm / sum_norm)


@pytest.mark.mathematical
@pytest.mark.parametrize("activation", [Sigmoid(), Tanh(), ReLU(), LeakyReLU(alpha=0.01), Linear()])
def test_dense_layer_numerical_gradient_check(activation):
    """Verifies analytical backpropagation against float64 numerical gradients across activations."""
    rng = np.random.default_rng(1234)
    x = rng.normal(0.0, 1.0, size=(4, 5)).astype(np.float64)
    y = rng.normal(0.0, 1.0, size=(4, 3)).astype(np.float64)

    layer1 = DenseLayer(5, 4, activation=activation, seed=10)
    layer2 = DenseLayer(4, 3, activation=Linear(), seed=20)
    net = NeuralNetwork([layer1, layer2])
    loss_fn = MeanSquaredError()

    # 1. Analytical backprop
    y_pred = net.forward(x, training=True)
    grad_out = loss_fn.gradient(y, y_pred)
    net.zero_grad()
    net.backward(grad_out)

    # 2. Numerical gradient check
    num_grads = compute_numerical_gradient(net, x, y, loss_fn, eps=1e-7)

    for idx, layer in enumerate(net.layers):
        l_key = f"layer_{idx}"
        for p_name in layer.params:
            g_ana = layer.grads[p_name]
            g_num = num_grads[l_key][p_name]
            err = relative_error(g_ana, g_num)
            assert err < 1e-6, f"Layer {idx} {p_name} gradient relative error {err:.2e} >= 1e-6"


@pytest.mark.mathematical
def test_losses_numerical_gradient_check():
    """Verifies analytical gradients of BCE, Huber, and MSE directly against finite differences."""
    rng = np.random.default_rng(5678)
    eps = 1e-7

    # 1. MeanSquaredError
    mse = MeanSquaredError()
    y_t = rng.uniform(0.1, 0.9, size=(5, 3))
    y_p = rng.uniform(0.1, 0.9, size=(5, 3))
    ana_grad = mse.gradient(y_t, y_p)
    num_grad = np.zeros_like(y_p)
    for idx in np.ndindex(y_p.shape):
        y_p_plus = np.copy(y_p)
        y_p_plus[idx] += eps
        y_p_minus = np.copy(y_p)
        y_p_minus[idx] -= eps
        num_grad[idx] = (mse.loss(y_t, y_p_plus) - mse.loss(y_t, y_p_minus)) / (2.0 * eps)
    assert relative_error(ana_grad, num_grad) < 1e-6

    # 2. BinaryCrossEntropy
    bce = BinaryCrossEntropy()
    y_bin_t = rng.integers(0, 2, size=(5, 2)).astype(np.float64)
    y_bin_p = rng.uniform(0.2, 0.8, size=(5, 2))
    ana_grad_bce = bce.gradient(y_bin_t, y_bin_p)
    num_grad_bce = np.zeros_like(y_bin_p)
    for idx in np.ndindex(y_bin_p.shape):
        yp_p = np.copy(y_bin_p)
        yp_p[idx] += eps
        yp_m = np.copy(y_bin_p)
        yp_m[idx] -= eps
        num_grad_bce[idx] = (bce.loss(y_bin_t, yp_p) - bce.loss(y_bin_t, yp_m)) / (2.0 * eps)
    assert relative_error(ana_grad_bce, num_grad_bce) < 1e-6

    # 3. HuberLoss
    huber = HuberLoss(delta=1.0)
    ana_grad_h = huber.gradient(y_t, y_p)
    num_grad_h = np.zeros_like(y_p)
    for idx in np.ndindex(y_p.shape):
        yp_p = np.copy(y_p)
        yp_p[idx] += eps
        yp_m = np.copy(y_p)
        yp_m[idx] -= eps
        num_grad_h[idx] = (huber.loss(y_t, yp_p) - huber.loss(y_t, yp_m)) / (2.0 * eps)
    assert relative_error(ana_grad_h, num_grad_h) < 1e-6


@pytest.mark.mathematical
def test_conv2d_numerical_gradient_check():
    """Verifies Conv2DLayer analytical gradients for weights and biases via finite differences."""
    rng = np.random.default_rng(999)
    x = rng.normal(0.0, 1.0, size=(2, 2, 4, 4)).astype(np.float64)

    conv = Conv2DLayer(in_channels=2, out_channels=2, kernel_size=2, padding=0, activation=Linear(), seed=42)
    # Conv out: (2, 2, 3, 3) -> 18 elements per sample
    y = rng.normal(0.0, 1.0, size=(2, 2, 3, 3)).astype(np.float64)
    loss_fn = MeanSquaredError()

    net = NeuralNetwork([conv])

    # 1. Analytical backward
    y_pred = net.forward(x, training=True)
    grad_out = loss_fn.gradient(y, y_pred)
    net.zero_grad()
    net.backward(grad_out)

    # 2. Numerical gradients
    num_grads = compute_numerical_gradient(net, x, y, loss_fn, eps=1e-7)

    w_err = relative_error(conv.grads["W"], num_grads["layer_0"]["W"])
    b_err = relative_error(conv.grads["b"], num_grads["layer_0"]["b"])

    assert w_err < 1e-6, f"Conv2D W relative error {w_err:.2e} >= 1e-6"
    assert b_err < 1e-6, f"Conv2D b relative error {b_err:.2e} >= 1e-6"
