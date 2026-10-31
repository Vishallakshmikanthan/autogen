"""Unit and perception tests for CNN and CNNTerrainClassifier (SYS-13, Module M14)."""

from __future__ import annotations

import numpy as np
import pytest

from autogen.agents.neural.perception import CNNPerceptionModule
from autogen.intelligence.neural.cnn import CNNTerrainClassifier
from autogen.intelligence.neural.layers import Conv2DLayer, FlattenLayer
from autogen.simulation.core.contracts import Observation
from autogen.simulation.state.types import Position


@pytest.mark.unit
def test_conv2d_and_flatten_shapes():
    """Verifies tensor shape transformations through Conv2D and Flatten layers."""
    # Input: (batch=2, channels=1, height=7, width=7)
    x = np.random.normal(0, 1, size=(2, 1, 7, 7))

    # Kernel=3, padding=0 -> output: (2, 4, 5, 5)
    conv = Conv2DLayer(in_channels=1, out_channels=4, kernel_size=3, padding=0)
    out_conv = conv.forward(x, training=False)
    assert out_conv.shape == (2, 4, 5, 5)

    # Flatten -> (2, 4*5*5) = (2, 100)
    flatten = FlattenLayer()
    out_flat = flatten.forward(out_conv, training=False)
    assert out_flat.shape == (2, 100)

    # Backward shape preservation
    grad_flat = np.ones_like(out_flat)
    grad_conv = flatten.backward(grad_flat)
    assert grad_conv.shape == (2, 4, 5, 5)

    grad_input = conv.backward(grad_conv)
    assert grad_input.shape == (2, 1, 7, 7)


@pytest.mark.mathematical
def test_cnn_terrain_classifier_training():
    """Verifies that CNNTerrainClassifier learns to distinguish distinct terrain patch patterns."""
    classifier = CNNTerrainClassifier(in_channels=1, patch_size=5, num_classes=5, seed=42)

    # Synthetic terrain patches:
    # 0: Empty (all 0s)
    # 1: Obstacle (center cell has value 1.0)
    # 2: Food (center cell has value 2.0)
    # 3: Water (center cell has value 3.0)
    # 4: Hazard (center cell has value 4.0)
    X_list = []
    y_list = []

    for class_id in range(5):
        for _ in range(12):
            patch = np.zeros((1, 5, 5), dtype=np.float64)
            patch[0, 2, 2] = float(class_id)
            # Add slight noise
            patch += np.random.normal(0, 0.05, size=(1, 5, 5))
            X_list.append(patch)
            y_list.append(class_id)

    X = np.array(X_list)
    y = np.array(y_list)

    history = classifier.fit(X, y, epochs=40, batch_size=10, lr=0.01)
    assert history[-1]["train_loss"] < history[0]["train_loss"]

    # Test inference on pure patches
    test_patch_obstacle = np.zeros((5, 5))
    test_patch_obstacle[2, 2] = 1.0
    assert classifier.classify_name(test_patch_obstacle) == "OBSTACLE"

    test_patch_food = np.zeros((5, 5))
    test_patch_food[2, 2] = 2.0
    assert classifier.classify_name(test_patch_food) == "FOOD"


@pytest.mark.unit
def test_cnn_perception_module():
    """Verifies CNNPerceptionModule spatial window extraction from Observation."""
    module = CNNPerceptionModule(patch_size=5)

    obs = Observation(
        agent_id="test_percept",
        step=1,
        position=Position(x=10, y=10),
        energy=100.0,
        health=100.0,
        carried_food=0.0,
        visible_cells=(),
        visible_agents=(),
    )

    patch = module.extract_patch(obs)
    assert patch.shape == (5, 5)

    result = module.classify_local_surroundings(obs)
    assert "predicted_terrain" in result
    assert "probabilities" in result
    assert len(result["probabilities"]) == 5
