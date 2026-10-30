"""Model serialization and artifact persistence with data hash and provenance (SYS-11, Module M12)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np

from autogen.intelligence.ml.contracts import ModelMetadata
from autogen.intelligence.ml.linear import LinearRegression, LogisticRegression


def save_model_artifact(
    model: Any,
    metadata: ModelMetadata,
    filepath: Path | str,
) -> None:
    """Serializes a trained model with full metadata, provenance hash, and parameter arrays."""
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data: dict[str, Any] = {
        "metadata": {
            "model_name": metadata.model_name,
            "version": metadata.version,
            "hyperparameters": dict(metadata.hyperparameters),
            "data_hash": metadata.data_hash,
            "created_at_iso": metadata.created_at_iso,
            "feature_names": list(metadata.feature_names),
        },
        "model_type": model.__class__.__name__,
    }

    if hasattr(model, "coef_") and model.coef_ is not None:
        data["coef"] = model.coef_.tolist()
    if hasattr(model, "intercept_"):
        data["intercept"] = float(model.intercept_)
    if hasattr(model, "classes_") and len(model.classes_) > 0:
        data["classes"] = model.classes_.tolist()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_linear_model(filepath: Path | str) -> tuple[Any, ModelMetadata]:
    """Loads a serialized linear/logistic model artifact."""
    in_path = Path(filepath)
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta_dict = data["metadata"]
    metadata = ModelMetadata(
        model_name=meta_dict["model_name"],
        version=meta_dict["version"],
        hyperparameters=meta_dict["hyperparameters"],
        data_hash=meta_dict["data_hash"],
        created_at_iso=meta_dict["created_at_iso"],
        feature_names=tuple(meta_dict["feature_names"]),
    )

    model_type = data["model_type"]
    if model_type == "LinearRegression":
        lin_model = LinearRegression(**metadata.hyperparameters)
        lin_model.coef_ = np.array(data["coef"], dtype=np.float64)
        lin_model.intercept_ = float(data["intercept"])
        return lin_model, metadata
    elif model_type == "LogisticRegression":
        log_model = LogisticRegression(**metadata.hyperparameters)
        log_model.coef_ = np.array(data["coef"], dtype=np.float64)
        log_model.intercept_ = float(data["intercept"])
        log_model.classes_ = np.array(data["classes"])
        return log_model, metadata
    else:
        raise ValueError(f"Unsupported model type for load_linear_model: {model_type}")
