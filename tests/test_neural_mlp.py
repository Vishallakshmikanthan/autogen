"""Unit, mathematical, and benchmark tests for NeuralNetwork and MLP components (SYS-13, Module M14)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from autogen.intelligence.neural.activations import ReLU, Sigmoid, Softmax
from autogen.intelligence.neural.layers import DenseLayer, DropoutLayer
from autogen.intelligence.neural.losses import (
    BinaryCrossEntropy,
    CategoricalCrossEntropy,
)
from autogen.intelligence.neural.mlp import NeuralNetwork
from autogen.intelligence.neural.optimizers import (
    SGD,
    Adam,
    CosineAnnealingLR,
    ExponentialLR,
    Momentum,
    RMSProp,
    StepLR,
)


@pytest.mark.mathematical
def test_mlp_learns_xor():
    """Verifies that MLP solves the non-linearly separable XOR problem to 100% accuracy."""
    # 2-bit XOR dataset
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    y = np.array([[0.0], [1.0], [1.0], [0.0]], dtype=np.float64)

    # Architecture: 2 -> 8 (Tanh/ReLU) -> 1 (Sigmoid)
    net = NeuralNetwork([
        DenseLayer(2, 8, activation=Sigmoid(), init_method="xavier", seed=42),
        DenseLayer(8, 1, activation=Sigmoid(), init_method="xavier", seed=43),
    ])

    loss_fn = BinaryCrossEntropy()
    optimizer = Adam(lr=0.05)

    history = net.fit(X, y, loss_fn=loss_fn, optimizer=optimizer, epochs=400, batch_size=4, seed=42)

    preds = net.predict(X)
    binary_preds = (preds > 0.5).astype(np.float64)

    np.testing.assert_array_equal(binary_preds, y)
    assert history[-1]["train_loss"] < 0.1


@pytest.mark.mathematical
def test_mlp_fits_sklearn_digits():
    """Verifies that MLP fits scikit-learn digits dataset above 90% test accuracy."""
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    digits = load_digits()
    X, y = digits.data, digits.target

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # One-hot encode targets
    num_classes = 10
    y_train_oh = np.zeros((len(y_train), num_classes), dtype=np.float64)
    y_train_oh[np.arange(len(y_train)), y_train] = 1.0

    y_test_oh = np.zeros((len(y_test), num_classes), dtype=np.float64)
    y_test_oh[np.arange(len(y_test)), y_test] = 1.0

    # 64 -> 32 -> 10
    net = NeuralNetwork([
        DenseLayer(64, 32, activation=ReLU(), init_method="he", seed=10),
        DenseLayer(32, 10, activation=Softmax(), init_method="xavier", seed=11),
    ])

    loss_fn = CategoricalCrossEntropy()
    optimizer = Adam(lr=0.01)

    net.fit(
        X_train,
        y_train_oh,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=30,
        batch_size=32,
        val_data=(X_test, y_test_oh),
        patience=5,
        seed=42,
    )

    test_preds = net.predict(X_test)
    test_acc = float(np.mean(np.argmax(test_preds, axis=1) == y_test))

    assert test_acc > 0.90, f"Digits test accuracy {test_acc:.3f} below required 90%"


@pytest.mark.unit
def test_dropout_layer():
    """Verifies inverted dropout behaviour during training vs evaluation."""
    x = np.ones((100, 100), dtype=np.float64)
    dropout = DropoutLayer(p_drop=0.5, seed=42)

    # During training: ~50% elements dropped, kept elements scaled by 2.0
    out_train = dropout.forward(x, training=True)
    assert np.any(out_train == 0.0)
    assert np.any(out_train == 2.0)
    assert np.isclose(np.mean(out_train), 1.0, atol=0.05)

    # During evaluation: identity pass-through
    out_eval = dropout.forward(x, training=False)
    np.testing.assert_array_equal(out_eval, x)


@pytest.mark.unit
def test_optimizers_step_and_zero_grad():
    """Verifies that SGD, Momentum, RMSProp, and Adam update parameters and zero gradients."""
    for opt_cls in [
        lambda: SGD(lr=0.01, weight_decay=0.01),
        lambda: Momentum(lr=0.01, beta=0.9),
        lambda: RMSProp(lr=0.01, beta=0.9),
        lambda: Adam(lr=0.01, weight_decay=0.01),
    ]:
        layer = DenseLayer(4, 2, seed=42)
        initial_w = np.copy(layer.W)
        layer.dW.fill(1.0)
        layer.db.fill(0.5)

        opt = opt_cls()
        opt.step([layer])

        # Weights must have changed
        assert not np.array_equal(initial_w, layer.W)

        # zero_grad resets gradients
        opt.zero_grad([layer])
        assert np.all(layer.dW == 0.0)
        assert np.all(layer.db == 0.0)


@pytest.mark.unit
def test_lr_schedulers():
    """Verifies step, exponential, and cosine annealing learning rate schedulers."""
    opt = SGD(lr=0.1)

    step_sched = StepLR(opt, step_size=5, gamma=0.5)
    for _ in range(5):
        step_sched.step()
    assert np.isclose(step_sched.current_lr, 0.05)

    opt.lr = 0.1
    exp_sched = ExponentialLR(opt, gamma=0.9)
    exp_sched.step()
    assert np.isclose(exp_sched.current_lr, 0.09)

    opt.lr = 0.1
    cos_sched = CosineAnnealingLR(opt, T_max=10, eta_min=0.0)
    for _ in range(10):
        cos_sched.step()
    assert np.isclose(cos_sched.current_lr, 0.0, atol=1e-6)


@pytest.mark.unit
def test_checkpointing_save_and_load():
    """Verifies bit-accurate parameter restoration from npz checkpoints."""
    net1 = NeuralNetwork([
        DenseLayer(4, 8, activation=ReLU(), seed=1),
        DenseLayer(8, 2, activation=Sigmoid(), seed=2),
    ])

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "model.npz"
        net1.save_checkpoint(ckpt_path)

        net2 = NeuralNetwork([
            DenseLayer(4, 8, activation=ReLU(), seed=99),
            DenseLayer(8, 2, activation=Sigmoid(), seed=100),
        ])
        assert not np.array_equal(net1.layers[0].params["W"], net2.layers[0].params["W"])

        net2.load_checkpoint(ckpt_path)
        np.testing.assert_array_equal(net1.layers[0].params["W"], net2.layers[0].params["W"])
        np.testing.assert_array_equal(net1.layers[1].params["b"], net2.layers[1].params["b"])
