"""Convolutional perception components for spatial grid observations (SYS-13, Module M14)."""

from __future__ import annotations

from typing import Any

import numpy as np

from autogen.intelligence.neural.cnn import CNNTerrainClassifier
from autogen.simulation.core.contracts import Observation


class CNNPerceptionModule:
    """Extracts local grid observation patches and classifies cell types using CNN."""

    def __init__(self, patch_size: int = 5, classifier: CNNTerrainClassifier | None = None) -> None:
        self.patch_size = patch_size
        self.classifier = classifier or CNNTerrainClassifier(in_channels=1, patch_size=patch_size)

    def extract_patch(self, observation: Observation) -> np.ndarray:
        """Extracts a local (patch_size, patch_size) window around the agent position."""
        half = self.patch_size // 2
        px, py = observation.position.x, observation.position.y

        # Initialize blank patch (0.0 = EMPTY)
        patch = np.zeros((self.patch_size, self.patch_size), dtype=np.float64)

        for cell in observation.visible_cells:
            dx = cell.position.x - px + half
            dy = cell.position.y - py + half
            if 0 <= dx < self.patch_size and 0 <= dy < self.patch_size:
                terrain_name = getattr(cell.terrain, "name", str(cell.terrain)).upper()
                if "OBSTACLE" in terrain_name or "WALL" in terrain_name:
                    patch[dy, dx] = 1.0
                elif getattr(cell, "food", 0.0) > 0.0 or "FOOD" in terrain_name:
                    patch[dy, dx] = 2.0
                elif "WATER" in terrain_name:
                    patch[dy, dx] = 3.0
                elif getattr(cell, "danger", 0.0) > 0.0 or "HAZARD" in terrain_name:
                    patch[dy, dx] = 4.0

        return patch

    def classify_local_surroundings(self, observation: Observation) -> dict[str, Any]:
        """Classifies local surrounding terrain and returns class probabilities."""
        patch = self.extract_patch(observation)
        probs = self.classifier.predict_proba(patch)[0]
        predicted_class = self.classifier.classify_name(patch)
        return {
            "predicted_terrain": predicted_class,
            "probabilities": {
                name: float(probs[i]) for i, name in enumerate(self.classifier.TERRAIN_CLASSES)
            },
        }
