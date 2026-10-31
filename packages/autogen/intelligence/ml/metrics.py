"""Evaluation metrics for classification and regression models implemented from scratch (SYS-11, Module M12)."""

from __future__ import annotations

from typing import Literal
import numpy as np


# =========================================================================
# Regression Metrics
# =========================================================================

def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes MAE: (1/N) * sum(|y_true - y_pred|)."""
    return float(np.mean(np.abs(y_true - y_pred)))


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes MSE: (1/N) * sum((y_true - y_pred)^2)."""
    return float(np.mean((y_true - y_pred) ** 2))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes RMSE: sqrt(MSE)."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes coefficient of determination R^2 = 1 - SS_res / SS_tot."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    y_mean = np.mean(y_true)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return float(1.0 - (ss_res / ss_tot))


# =========================================================================
# Classification Metrics
# =========================================================================

def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes fraction of correctly predicted samples."""
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int | None = None) -> np.ndarray:
    """Computes confusion matrix where C[i, j] is count of true class i predicted as class j."""
    if num_classes is None:
        max_class = max(int(np.max(y_true)), int(np.max(y_pred)))
        num_classes = max_class + 1

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def precision_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: Literal["binary", "macro"] = "binary",
    pos_label: int = 1,
) -> float:
    """Computes precision TP / (TP + FP)."""
    if average == "binary":
        tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
        fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
        denom = tp + fp
        return float(tp / denom) if denom > 0 else 0.0

    classes = np.unique(np.concatenate([y_true, y_pred]))
    precisions = []
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        denom = tp + fp
        precisions.append(tp / denom if denom > 0 else 0.0)
    return float(np.mean(precisions))


def recall_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: Literal["binary", "macro"] = "binary",
    pos_label: int = 1,
) -> float:
    """Computes recall TP / (TP + FN)."""
    if average == "binary":
        tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
        fn = np.sum((y_true == pos_label) & (y_pred != pos_label))
        denom = tp + fn
        return float(tp / denom) if denom > 0 else 0.0

    classes = np.unique(np.concatenate([y_true, y_pred]))
    recalls = []
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        denom = tp + fn
        recalls.append(tp / denom if denom > 0 else 0.0)
    return float(np.mean(recalls))


def f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: Literal["binary", "macro"] = "binary",
    pos_label: int = 1,
) -> float:
    """Computes F1-score: 2 * (Precision * Recall) / (Precision + Recall)."""
    prec = precision_score(y_true, y_pred, average=average, pos_label=pos_label)
    rec = recall_score(y_true, y_pred, average=average, pos_label=pos_label)
    denom = prec + rec
    return float(2.0 * (prec * rec) / denom) if denom > 0 else 0.0


def log_loss(y_true: np.ndarray, y_proba: np.ndarray, eps: float = 1e-15) -> float:
    """Computes binary cross entropy loss."""
    p = np.clip(y_proba, eps, 1.0 - eps)
    loss = -np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))
    return float(loss)


def roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    pos_label: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes Receiver Operating Characteristic (ROC) curve: (fpr, tpr, thresholds)."""
    y_binary = (y_true == pos_label).astype(int)
    n_pos = np.sum(y_binary)
    n_neg = len(y_binary) - n_pos

    if n_pos == 0 or n_neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])

    # Sort descending by scores
    desc_idx = np.argsort(y_scores)[::-1]
    y_sorted = y_binary[desc_idx]
    scores_sorted = y_scores[desc_idx]

    # Find distinct thresholds
    distinct_idx = np.where(np.diff(scores_sorted))[0]
    threshold_idx = np.r_[distinct_idx, len(scores_sorted) - 1]

    tps = np.cumsum(y_sorted)[threshold_idx]
    fps = (threshold_idx + 1) - tps

    tpr = np.r_[0.0, tps / n_pos]
    fpr = np.r_[0.0, fps / n_neg]
    thresholds = np.r_[scores_sorted[0] + 1.0, scores_sorted[threshold_idx]]

    return fpr, tpr, thresholds


def roc_auc_score(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    pos_label: int = 1,
) -> float:
    """Computes Area Under the ROC Curve using the composite trapezoidal rule."""
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=pos_label)
    # Numerical integration: int TPR d(FPR)
    auc = np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0)
    return float(auc)

