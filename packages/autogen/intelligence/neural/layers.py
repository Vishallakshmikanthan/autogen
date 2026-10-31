"""Neural network layers with forward and backward propagation (SYS-13, Module M14)."""

from __future__ import annotations

from typing import Literal, cast

import numpy as np

from autogen.intelligence.neural.activations import Linear, ReLU
from autogen.intelligence.neural.contracts import Activation, Layer


class DenseLayer(Layer):
    """Fully-connected (dense) linear layer with activation: a = sigma(X W + b)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: Activation | None = None,
        init_method: Literal["he", "xavier", "lecun"] = "he",
        seed: int | None = None,
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.activation: Activation = activation or Linear()
        rng = np.random.default_rng(seed)

        if init_method == "he":
            # He (Kaiming) normal initialization for ReLU/LeakyReLU
            std = np.sqrt(2.0 / in_features)
            self.W = rng.normal(0.0, std, size=(in_features, out_features)).astype(np.float64)
        elif init_method == "xavier":
            # Xavier (Glorot) uniform initialization for Sigmoid/Tanh
            limit = np.sqrt(6.0 / (in_features + out_features))
            self.W = rng.uniform(-limit, limit, size=(in_features, out_features)).astype(np.float64)
        else:
            # LeCun normal initialization
            std = np.sqrt(1.0 / in_features)
            self.W = rng.normal(0.0, std, size=(in_features, out_features)).astype(np.float64)

        self.b = np.zeros((1, out_features), dtype=np.float64)

        # Gradient storage
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        # Cache for backprop
        self._input_cache: np.ndarray | None = None
        self._z_cache: np.ndarray | None = None
        self._a_cache: np.ndarray | None = None

    @property
    def params(self) -> dict[str, np.ndarray]:
        return {"W": self.W, "b": self.b}

    @property
    def grads(self) -> dict[str, np.ndarray]:
        return {"W": self.dW, "b": self.db}

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        self._input_cache = x
        z = x @ self.W + self.b
        self._z_cache = z
        a = self.activation.forward(z)
        self._a_cache = a
        return a

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._input_cache is None or self._z_cache is None or self._a_cache is None:
            raise RuntimeError("Layer forward pass must precede backward pass.")
        if grad_output.ndim == 1:
            grad_output = grad_output.reshape(1, -1)

        # Element-wise or full Jacobian activation gradient
        if hasattr(self.activation, "backward"):
            delta = self.activation.backward(grad_output, self._z_cache, self._a_cache)
        else:
            act_deriv = self.activation.derivative(self._z_cache, self._a_cache)
            delta = grad_output * act_deriv

        # Compute parameter gradients: dL/dW = X^T * delta, dL/db = sum(delta)
        self.dW = self._input_cache.T @ delta
        self.db = np.sum(delta, axis=0, keepdims=True)

        # Compute gradient with respect to layer inputs: grad_input = delta * W^T
        grad_input = delta @ self.W.T
        return grad_input


class DropoutLayer(Layer):
    """Inverted dropout layer for regularization during training."""

    def __init__(self, p_drop: float = 0.5, seed: int | None = None) -> None:
        if not (0.0 <= p_drop < 1.0):
            raise ValueError(f"Dropout probability must be in [0, 1), got {p_drop}")
        self.p_drop = float(p_drop)
        self.rng = np.random.default_rng(seed)
        self._mask: np.ndarray | None = None

    @property
    def params(self) -> dict[str, np.ndarray]:
        return {}

    @property
    def grads(self) -> dict[str, np.ndarray]:
        return {}

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if not training or self.p_drop == 0.0:
            self._mask = None
            return x
        keep_prob = 1.0 - self.p_drop
        self._mask = (self.rng.random(x.shape) < keep_prob).astype(np.float64) / keep_prob
        return cast(np.ndarray, x * self._mask)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._mask is None:
            return grad_output
        return cast(np.ndarray, grad_output * self._mask)


class FlattenLayer(Layer):
    """Flattens multi-dimensional tensor (e.g. N, C, H, W) into (N, C*H*W)."""

    def __init__(self) -> None:
        self._orig_shape: tuple[int, ...] | None = None

    @property
    def params(self) -> dict[str, np.ndarray]:
        return {}

    @property
    def grads(self) -> dict[str, np.ndarray]:
        return {}

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        self._orig_shape = x.shape
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._orig_shape is None:
            raise RuntimeError("FlattenLayer forward must precede backward.")
        return grad_output.reshape(self._orig_shape)


class Conv2DLayer(Layer):
    """2D spatial cross-correlation layer: a = sigma(Conv2D(X, W) + b)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 0,
        activation: Activation | None = None,
        seed: int | None = None,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation or ReLU()

        rng = np.random.default_rng(seed)
        std = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.W = rng.normal(0.0, std, size=(out_channels, in_channels, kernel_size, kernel_size)).astype(np.float64)
        self.b = np.zeros(out_channels, dtype=np.float64)

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        self._input_cache: np.ndarray | None = None
        self._padded_cache: np.ndarray | None = None
        self._z_cache: np.ndarray | None = None
        self._a_cache: np.ndarray | None = None

    @property
    def params(self) -> dict[str, np.ndarray]:
        return {"W": self.W, "b": self.b}

    @property
    def grads(self) -> dict[str, np.ndarray]:
        return {"W": self.dW, "b": self.db}

    def _pad(self, x: np.ndarray) -> np.ndarray:
        if self.padding == 0:
            return x
        return np.pad(
            x,
            ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
            mode="constant",
        )

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        # Expected shape: (N, C, H, W)
        if x.ndim == 3:
            x = x[np.newaxis, ...]  # (1, C, H, W)
        self._input_cache = x
        x_pad = self._pad(x)
        self._padded_cache = x_pad

        N, _, H_in, W_in = x_pad.shape
        K = self.kernel_size
        H_out = H_in - K + 1
        W_out = W_in - K + 1

        z = np.zeros((N, self.out_channels, H_out, W_out), dtype=np.float64)
        for h in range(H_out):
            for w in range(W_out):
                window = x_pad[:, :, h : h + K, w : w + K]  # (N, C_in, K, K)
                # Einsum sum over C_in, K, K: (N, C_in, K, K) x (C_out, C_in, K, K) -> (N, C_out)
                z[:, :, h, w] = np.einsum("nchw,ochw->no", window, self.W) + self.b

        self._z_cache = z
        a = self.activation.forward(z)
        self._a_cache = a
        return a

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._padded_cache is None or self._input_cache is None or self._z_cache is None or self._a_cache is None:
            raise RuntimeError("Conv2DLayer forward must precede backward.")

        if hasattr(self.activation, "backward"):
            delta = self.activation.backward(grad_output, self._z_cache, self._a_cache)
        else:
            act_deriv = self.activation.derivative(self._z_cache, self._a_cache)
            delta = grad_output * act_deriv  # (N, C_out, H_out, W_out)

        self.dW.fill(0.0)
        self.db = np.sum(delta, axis=(0, 2, 3))

        N, C_in, H_pad, W_pad = self._padded_cache.shape
        K = self.kernel_size
        H_out, W_out = delta.shape[2], delta.shape[3]

        grad_padded = np.zeros_like(self._padded_cache)

        for h in range(H_out):
            for w in range(W_out):
                d_hw = delta[:, :, h, w]  # (N, C_out)
                window = self._padded_cache[:, :, h : h + K, w : w + K]  # (N, C_in, K, K)
                # dW contribution: (N, C_out) x (N, C_in, K, K) -> (C_out, C_in, K, K)
                self.dW += np.einsum("no,nchw->ochw", d_hw, window)
                # grad_padded contribution: (N, C_out) x (C_out, C_in, K, K) -> (N, C_in, K, K)
                grad_padded[:, :, h : h + K, w : w + K] += np.einsum("no,ochw->nchw", d_hw, self.W)

        if self.padding == 0:
            return grad_padded
        return grad_padded[:, :, self.padding : -self.padding, self.padding : -self.padding]
