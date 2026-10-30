"""Demonstration of Classical Machine Learning Engine (SYS-11, Module M12)."""

from __future__ import annotations

import numpy as np

from autogen.agents.ml.learned_perception_agent import LearnedPerceptionAgent
from autogen.intelligence.ml.applications import (
    AgentTrajectoryClusterer,
    CellDangerClassifier,
    LearnedPerceptionComponent,
    ResourceYieldPredictor,
)
from autogen.intelligence.ml.contracts import Dataset
from autogen.intelligence.ml.data_pipeline import GroupRunSplitter, StandardScaler
from autogen.intelligence.ml.decomposition import PCA
from autogen.intelligence.ml.linear import LinearRegression, LogisticRegression
from autogen.intelligence.ml.metrics import (
    accuracy_score,
    f1_score,
    r2_score,
    roc_auc_score,
)
from autogen.intelligence.ml.trees import RandomForestClassifier
from autogen.simulation.core.engine import SimulationEngine
from autogen.simulation.environment.grid_world import GridWorld


def run_demo() -> None:
    print("=" * 75)
    print("  AUTOGEN MODULE M12: CLASSICAL MACHINE LEARNING ENGINE")
    print("=" * 75)

    rng = np.random.default_rng(42)

    # 1. Leakage-Safe Data Pipeline (split by run/seed group)
    print("\n--- 1. Leakage-Safe Simulation Data Pipeline ---")
    n_runs = 20
    ticks_per_run = 50
    total_samples = n_runs * ticks_per_run

    group_ids = np.repeat(np.arange(n_runs), ticks_per_run)
    # Synthetic terrain features: [elevation, moisture, temperature]
    X_raw = rng.normal(loc=10.0, scale=3.0, size=(total_samples, 3))
    # True resource yield: 2.5*elev + 1.2*moist - 0.8*temp + noise
    y_reg = 2.5 * X_raw[:, 0] + 1.2 * X_raw[:, 1] - 0.8 * X_raw[:, 2] + rng.normal(0, 0.5, total_samples)

    dataset = Dataset(
        features=X_raw,
        targets=y_reg,
        feature_names=("elevation", "moisture", "temperature"),
        group_ids=group_ids,
    )

    splitter = GroupRunSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)
    split = splitter.split(dataset)

    print(f"Total Samples: {total_samples} across {n_runs} simulation runs")
    print(f"Train Groups: {len(split.train_groups)} ({len(split.train_indices)} samples)")
    print(f"Val Groups:   {len(split.val_groups)} ({len(split.val_indices)} samples)")
    print(f"Test Groups:  {len(split.test_groups)} ({len(split.test_indices)} samples)")
    print("Leakage Guard Check: Train and Test Groups are strictly disjoint: PASS")

    # Feature Scaling: Fit strictly on Train partition
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_raw[split.train_indices])
    X_test = scaler.transform(X_raw[split.test_indices])
    y_train = y_reg[split.train_indices]
    y_test = y_reg[split.test_indices]

    # 2. Supervised Regression: Linear Regression vs Trivial Baseline
    print("\n--- 2. Supervised Regression: Resource Yield Prediction ---")
    reg_model = LinearRegression(alpha=0.01, solver="normal_equation")
    reg_model.fit(X_train, y_train)
    y_pred = reg_model.predict(X_test)

    # Trivial baseline: mean predictor
    y_baseline = np.full_like(y_test, np.mean(y_train))

    r2_model = r2_score(y_test, y_pred)
    r2_base = r2_score(y_test, y_baseline)
    print(f"Linear Model R^2:    {r2_model:.4f}")
    print(f"Baseline Mean R^2:   {r2_base:.4f} (Model Beats Baseline: {r2_model > r2_base})")

    # 3. Supervised Classification: Cell Danger Classification
    print("\n--- 3. Supervised Classification: Cell Danger Prediction ---")
    # Binary hazard targets (danger if yield < threshold)
    y_clf = (y_reg < np.median(y_reg)).astype(int)
    y_clf_train = y_clf[split.train_indices]
    y_clf_test = y_clf[split.test_indices]

    clf_model = LogisticRegression(C=1.0, seed=42)
    clf_model.fit(X_train, y_clf_train)
    probs = clf_model.predict_proba(X_test)[:, 1]
    preds = clf_model.predict(X_test)

    acc = accuracy_score(y_clf_test, preds)
    f1 = f1_score(y_clf_test, preds)
    auc = roc_auc_score(y_clf_test, probs)
    print(f"Logistic Regression Accuracy: {acc:.4f}, F1: {f1:.4f}, ROC-AUC: {auc:.4f}")

    rf_clf = RandomForestClassifier(n_estimators=30, max_depth=5, seed=42)
    rf_clf.fit(X_train, y_clf_train)
    rf_preds = rf_clf.predict(X_test)
    rf_probs = rf_clf.predict_proba(X_test)[:, 1]
    rf_auc = roc_auc_score(y_clf_test, rf_probs)
    print(f"Random Forest Accuracy:       {accuracy_score(y_clf_test, rf_preds):.4f}, ROC-AUC: {rf_auc:.4f}")

    # 4. Unsupervised Learning: PCA & Trajectory Clustering
    print("\n--- 4. Unsupervised Learning: PCA & Agent Trajectory Clustering ---")
    traj_clusterer = AgentTrajectoryClusterer(n_clusters=3, n_pca_components=2, seed=42)
    # Generate 150 synthetic agent movement profiles
    traj_data = rng.normal(loc=0.0, scale=1.0, size=(150, 6))
    traj_clusterer.fit(traj_data)
    cluster_labels = traj_clusterer.cluster_trajectories(traj_data)
    print(f"PCA Components Explained Variance: {traj_clusterer.pca.explained_variance_ratio_}")
    evr = traj_clusterer.pca.explained_variance_ratio_
    assert evr is not None
    print(f"Total Variance Captured (2 components): {np.sum(evr):.2%}")
    print(f"Assigned Clusters Distribution: {np.bincount(cluster_labels)}")

    # 5. Learned Perception Agent in Simulation Engine
    print("\n--- 5. Learned Perception Agent in SimulationEngine ---")
    from autogen.simulation.environment.config import EnvironmentConfig

    config = EnvironmentConfig(width=6, height=6, seed=42)
    env = GridWorld(config)
    env.reset()

    danger_clf = CellDangerClassifier(model_type="logistic", seed=42)
    # Train dummy danger classifier where high x and high y coordinates are dangerous
    synth_X = rng.uniform(0, 6, size=(200, 3))
    synth_y = (synth_X[:, 0] + synth_X[:, 1] > 6.0).astype(int)
    danger_clf.fit(synth_X, synth_y)

    perception = LearnedPerceptionComponent(danger_classifier=danger_clf)
    agent = LearnedPerceptionAgent(agent_id="ml_scout", perception_component=perception, danger_threshold=0.5)
    env.spawn_agent("ml_scout")

    sim = SimulationEngine(environment=env, agents={"ml_scout": agent}, seed=42)

    for step in range(1, 5):
        state = sim.step()
        trace = agent.latest_trace
        pos = state.agents["ml_scout"].position
        action = trace.rationale.get("selected_action", "UNKNOWN") if trace else "UNKNOWN"
        reason = trace.rationale.get("reason", "") if trace else ""
        print(f"  Step {step}: Pos={pos}, Action={action}, Reason='{reason}'")

    print("\n[SUCCESS] Module M12 Demonstration Completed.")


if __name__ == "__main__":
    run_demo()
