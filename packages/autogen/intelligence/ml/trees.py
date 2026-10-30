"""Decision Trees (CART/ID3) and Random Forests implemented from scratch (SYS-11, Module M12)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast
import numpy as np

from autogen.intelligence.ml.contracts import Classifier, Regressor


@dataclass
class _TreeNode:
    feature: int | None = None
    threshold: float | None = None
    left: _TreeNode | None = None
    right: _TreeNode | None = None
    value: Any = None
    proba: np.ndarray | None = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class DecisionTreeClassifier(Classifier):
    """Binary/Multiclass Decision Tree Classifier supporting Gini impurity and Shannon entropy."""

    def __init__(
        self,
        criterion: Literal["gini", "entropy"] = "gini",
        max_depth: int | None = 10,
        min_samples_split: int = 2,
        min_impurity_decrease: float = 0.0,
        max_features: int | float | Literal["sqrt", "log2"] | None = None,
        seed: int = 42,
    ) -> None:
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.seed = seed

        self.root_: _TreeNode | None = None
        self.classes_: np.ndarray = np.array([])
        self.n_classes_: int = 0
        self.n_features_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> DecisionTreeClassifier:
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got {X.shape}")
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]
        self._rng = np.random.default_rng(self.seed)

        # Map classes to contiguous integers 0..K-1
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_int = np.array([class_to_idx[c] for c in y], dtype=np.int32)

        self.root_ = self._grow_tree(X, y_int, depth=0)
        return self

    def _impurity(self, y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        counts = np.bincount(y, minlength=self.n_classes_)
        probs = counts / len(y)
        if self.criterion == "gini":
            return float(1.0 - np.sum(probs ** 2))
        else:  # entropy
            nonzero_p = probs[probs > 0]
            return float(-np.sum(nonzero_p * np.log2(nonzero_p)))

    def _get_max_features(self) -> int:
        if self.max_features is None:
            return self.n_features_
        elif self.max_features == "sqrt":
            return max(1, int(np.sqrt(self.n_features_)))
        elif self.max_features == "log2":
            return max(1, int(np.log2(self.n_features_)))
        elif isinstance(self.max_features, float):
            return max(1, int(self.max_features * self.n_features_))
        return max(1, min(int(self.max_features), self.n_features_))

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> _TreeNode:
        n_samples = len(y)
        counts = np.bincount(y, minlength=self.n_classes_)
        proba = counts / n_samples
        majority_idx = int(np.argmax(counts))

        # Check stopping criteria
        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or n_samples < self.min_samples_split
            or np.max(counts) == n_samples
        ):
            return _TreeNode(value=self.classes_[majority_idx], proba=proba)

        parent_impurity = self._impurity(y)
        m_features = self._get_max_features()
        feat_indices = self._rng.choice(self.n_features_, size=m_features, replace=False)

        best_gain = -1.0
        best_feat: int | None = None
        best_thresh: float | None = None
        best_splits: tuple[np.ndarray, np.ndarray] | None = None

        for feat in feat_indices:
            x_col = X[:, feat]
            thresholds = np.unique(x_col)
            if len(thresholds) <= 1:
                continue

            # Candidate midpoints
            midpoints = (thresholds[:-1] + thresholds[1:]) / 2.0
            for thresh in midpoints:
                left_mask = x_col <= thresh
                n_left = np.sum(left_mask)
                n_right = n_samples - n_left

                if n_left == 0 or n_right == 0:
                    continue

                left_imp = self._impurity(y[left_mask])
                right_imp = self._impurity(y[~left_mask])
                gain = parent_impurity - (n_left / n_samples) * left_imp - (n_right / n_samples) * right_imp

                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = thresh
                    best_splits = (left_mask, ~left_mask)

        if best_gain < self.min_impurity_decrease or best_feat is None or best_splits is None:
            return _TreeNode(value=self.classes_[majority_idx], proba=proba)

        left_mask, right_mask = best_splits
        left_child = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._grow_tree(X[right_mask], y[right_mask], depth + 1)

        return _TreeNode(
            feature=best_feat,
            threshold=best_thresh,
            left=left_child,
            right=right_child,
            value=self.classes_[majority_idx],
            proba=proba,
        )

    def _predict_sample(self, x: np.ndarray, node: _TreeNode) -> tuple[Any, np.ndarray]:
        if node.is_leaf or node.feature is None or node.threshold is None:
            return node.value, node.proba if node.proba is not None else np.zeros(self.n_classes_)

        if x[node.feature] <= node.threshold:
            return self._predict_sample(x, node.left)  # type: ignore[arg-type]
        else:
            return self._predict_sample(x, node.right)  # type: ignore[arg-type]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.root_ is None:
            raise RuntimeError("DecisionTreeClassifier must be fitted before predict.")
        probs = [self._predict_sample(x, self.root_)[1] for x in X]
        return np.array(probs, dtype=np.float64)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.root_ is None:
            raise RuntimeError("DecisionTreeClassifier must be fitted before predict.")
        preds = [self._predict_sample(x, self.root_)[0] for x in X]
        return np.array(preds)


class DecisionTreeRegressor(Regressor):
    """Decision Tree Regressor minimizing squared error / variance."""

    def __init__(
        self,
        max_depth: int | None = 10,
        min_samples_split: int = 2,
        min_impurity_decrease: float = 0.0,
        max_features: int | float | Literal["sqrt", "third"] | None = None,
        seed: int = 42,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.seed = seed

        self.root_: _TreeNode | None = None
        self.n_features_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> DecisionTreeRegressor:
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got {X.shape}")
        self.n_features_ = X.shape[1]
        self._rng = np.random.default_rng(self.seed)
        self.root_ = self._grow_tree(X, y.astype(np.float64), depth=0)
        return self

    def _variance(self, y: np.ndarray) -> float:
        if len(y) <= 1:
            return 0.0
        return float(np.var(y))

    def _get_max_features(self) -> int:
        if self.max_features is None:
            return self.n_features_
        elif self.max_features == "sqrt":
            return max(1, int(np.sqrt(self.n_features_)))
        elif self.max_features == "third":
            return max(1, int(self.n_features_ / 3.0))
        elif isinstance(self.max_features, float):
            return max(1, int(self.max_features * self.n_features_))
        return max(1, min(int(self.max_features), self.n_features_))

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> _TreeNode:
        n_samples = len(y)
        mean_val = float(np.mean(y))

        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or n_samples < self.min_samples_split
            or np.all(y == y[0])
        ):
            return _TreeNode(value=mean_val)

        parent_var = self._variance(y)
        m_features = self._get_max_features()
        feat_indices = self._rng.choice(self.n_features_, size=m_features, replace=False)

        best_reduction = -1.0
        best_feat: int | None = None
        best_thresh: float | None = None
        best_splits: tuple[np.ndarray, np.ndarray] | None = None

        for feat in feat_indices:
            x_col = X[:, feat]
            thresholds = np.unique(x_col)
            if len(thresholds) <= 1:
                continue

            midpoints = (thresholds[:-1] + thresholds[1:]) / 2.0
            for thresh in midpoints:
                left_mask = x_col <= thresh
                n_left = np.sum(left_mask)
                n_right = n_samples - n_left

                if n_left == 0 or n_right == 0:
                    continue

                left_var = self._variance(y[left_mask])
                right_var = self._variance(y[~left_mask])
                reduction = parent_var - (n_left / n_samples) * left_var - (n_right / n_samples) * right_var

                if reduction > best_reduction:
                    best_reduction = reduction
                    best_feat = feat
                    best_thresh = thresh
                    best_splits = (left_mask, ~left_mask)

        if best_reduction < self.min_impurity_decrease or best_feat is None or best_splits is None:
            return _TreeNode(value=mean_val)

        left_mask, right_mask = best_splits
        left_child = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._grow_tree(X[right_mask], y[right_mask], depth + 1)

        return _TreeNode(
            feature=best_feat,
            threshold=best_thresh,
            left=left_child,
            right=right_child,
            value=mean_val,
        )

    def _predict_sample(self, x: np.ndarray, node: _TreeNode) -> float:
        if node.is_leaf or node.feature is None or node.threshold is None:
            return float(node.value)
        if x[node.feature] <= node.threshold:
            return self._predict_sample(x, node.left)  # type: ignore[arg-type]
        else:
            return self._predict_sample(x, node.right)  # type: ignore[arg-type]

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.root_ is None:
            raise RuntimeError("DecisionTreeRegressor must be fitted before predict.")
        return np.array([self._predict_sample(x, self.root_) for x in X], dtype=np.float64)


class RandomForestClassifier(Classifier):
    """Ensemble of bootstrap decision trees with random feature subspaces."""

    def __init__(
        self,
        n_estimators: int = 50,
        criterion: Literal["gini", "entropy"] = "gini",
        max_depth: int | None = 10,
        min_samples_split: int = 2,
        max_features: Literal["sqrt", "log2"] | int | float = "sqrt",
        bootstrap: bool = True,
        seed: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.seed = seed

        self.trees_: list[DecisionTreeClassifier] = []
        self.classes_: np.ndarray = np.array([])
        self.oob_score_: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
        N = X.shape[0]
        self.classes_ = np.unique(y)
        rng = np.random.default_rng(self.seed)

        self.trees_ = []
        oob_counts = np.zeros(N, dtype=np.int32)
        oob_prob_sum = np.zeros((N, len(self.classes_)), dtype=np.float64)

        for i in range(self.n_estimators):
            tree_seed = int(rng.integers(0, 1_000_000))
            tree = DecisionTreeClassifier(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                seed=tree_seed,
            )

            if self.bootstrap:
                boot_idx = rng.choice(N, size=N, replace=True)
                oob_mask = np.ones(N, dtype=bool)
                oob_mask[boot_idx] = False

                tree.fit(X[boot_idx], y[boot_idx])
                self.trees_.append(tree)

                if np.any(oob_mask):
                    probs = tree.predict_proba(X[oob_mask])
                    oob_prob_sum[oob_mask] += probs
                    oob_counts[oob_mask] += 1
            else:
                tree.fit(X, y)
                self.trees_.append(tree)

        # Compute out-of-bag score if bootstrap enabled
        if self.bootstrap:
            evaluated = oob_counts > 0
            if np.any(evaluated):
                oob_preds_idx = np.argmax(oob_prob_sum[evaluated], axis=1)
                oob_preds = self.classes_[oob_preds_idx]
                self.oob_score_ = float(np.mean(oob_preds == y[evaluated]))

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.trees_:
            raise RuntimeError("RandomForestClassifier must be fitted before predict.")
        tree_probs = [tree.predict_proba(X) for tree in self.trees_]
        return cast(np.ndarray, np.mean(tree_probs, axis=0))

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        pred_idx = np.argmax(probs, axis=1)
        return cast(np.ndarray, self.classes_[pred_idx])


class RandomForestRegressor(Regressor):
    """Ensemble of bootstrap regression trees with random feature subspaces."""

    def __init__(
        self,
        n_estimators: int = 50,
        max_depth: int | None = 10,
        min_samples_split: int = 2,
        max_features: Literal["third", "sqrt"] | int | float = "third",
        bootstrap: bool = True,
        seed: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.seed = seed

        self.trees_: list[DecisionTreeRegressor] = []
        self.oob_score_: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> RandomForestRegressor:
        N = X.shape[0]
        rng = np.random.default_rng(self.seed)
        self.trees_ = []
        oob_preds = np.zeros(N, dtype=np.float64)
        oob_counts = np.zeros(N, dtype=np.int32)

        for _ in range(self.n_estimators):
            tree_seed = int(rng.integers(0, 1_000_000))
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                seed=tree_seed,
            )

            if self.bootstrap:
                boot_idx = rng.choice(N, size=N, replace=True)
                oob_mask = np.ones(N, dtype=bool)
                oob_mask[boot_idx] = False

                tree.fit(X[boot_idx], y[boot_idx])
                self.trees_.append(tree)

                if np.any(oob_mask):
                    preds = tree.predict(X[oob_mask])
                    oob_preds[oob_mask] += preds
                    oob_counts[oob_mask] += 1
            else:
                tree.fit(X, y)
                self.trees_.append(tree)

        if self.bootstrap:
            evaluated = oob_counts > 0
            if np.any(evaluated):
                oob_final = oob_preds[evaluated] / oob_counts[evaluated]
                ss_res = np.sum((y[evaluated] - oob_final) ** 2)
                ss_tot = np.sum((y[evaluated] - np.mean(y[evaluated])) ** 2)
                self.oob_score_ = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.trees_:
            raise RuntimeError("RandomForestRegressor must be fitted before predict.")
        preds = [tree.predict(X) for tree in self.trees_]
        return cast(np.ndarray, np.mean(preds, axis=0))
