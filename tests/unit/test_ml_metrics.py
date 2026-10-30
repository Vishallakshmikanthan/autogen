"""Unit tests for ML metrics against scikit-learn test oracles (SYS-11, Module M12)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn import metrics as skm

from autogen.intelligence.ml import metrics as mym


def test_regression_metrics_match_sklearn() -> None:
    """Verifies that our MAE, MSE, RMSE, and R^2 match scikit-learn."""
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.1, 7.8])

    assert abs(mym.mean_absolute_error(y_true, y_pred) - skm.mean_absolute_error(y_true, y_pred)) < 1e-10
    assert abs(mym.mean_squared_error(y_true, y_pred) - skm.mean_squared_error(y_true, y_pred)) < 1e-10
    assert abs(mym.root_mean_squared_error(y_true, y_pred) - skm.root_mean_squared_error(y_true, y_pred)) < 1e-10
    assert abs(mym.r2_score(y_true, y_pred) - skm.r2_score(y_true, y_pred)) < 1e-10


def test_classification_metrics_match_sklearn() -> None:
    """Verifies that Accuracy, Precision, Recall, F1, and Confusion Matrix match scikit-learn."""
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 1, 1, 0])

    assert abs(mym.accuracy_score(y_true, y_pred) - skm.accuracy_score(y_true, y_pred)) < 1e-10
    assert abs(mym.precision_score(y_true, y_pred) - skm.precision_score(y_true, y_pred)) < 1e-10
    assert abs(mym.recall_score(y_true, y_pred) - skm.recall_score(y_true, y_pred)) < 1e-10
    assert abs(mym.f1_score(y_true, y_pred) - skm.f1_score(y_true, y_pred)) < 1e-10
    assert np.all(mym.confusion_matrix(y_true, y_pred) == skm.confusion_matrix(y_true, y_pred))


def test_roc_auc_matches_sklearn() -> None:
    """Verifies that our pure-NumPy trapezoidal ROC-AUC matches scikit-learn."""
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0])
    y_scores = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9, 0.65, 0.3, 0.75, 0.05])

    my_auc = mym.roc_auc_score(y_true, y_scores)
    sk_auc = skm.roc_auc_score(y_true, y_scores)

    assert abs(my_auc - sk_auc) < 1e-10
