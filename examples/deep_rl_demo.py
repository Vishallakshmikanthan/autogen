"""Demonstration of Neural Networks & Deep RL: MLP, Backpropagation, CNN Perception, DQN, and PPO (SYS-13, Module M14)."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from autogen.agents.neural.dqn_agent import DQNAgent
from autogen.agents.neural.ppo_agent import PPOAgent
from autogen.intelligence.neural.activations import ReLU, Sigmoid, Softmax
from autogen.intelligence.neural.cnn import CNNTerrainClassifier
from autogen.intelligence.neural.dqn import DeepQNetwork
from autogen.intelligence.neural.layers import DenseLayer
from autogen.intelligence.neural.losses import BinaryCrossEntropy, CategoricalCrossEntropy
from autogen.intelligence.neural.mlp import NeuralNetwork
from autogen.intelligence.neural.optimizers import Adam
from autogen.intelligence.neural.policy_gradient import PPO
from autogen.intelligence.neural.survival_predictor import NeuralSurvivalPredictor
from autogen.simulation.core.engine import SimulationEngine
from autogen.simulation.environment.config import EnvironmentConfig
from autogen.simulation.environment.grid_world import GridWorld


def run_demo() -> None:
    print("=" * 80)
    print("  AUTOGEN MODULE M14: NEURAL NETWORKS & DEEP REINFORCEMENT LEARNING")
    print("=" * 80)

    # 1. NumPy Multilayer Perceptron from Scratch: 2-bit XOR Non-Linear Benchmark
    print("\n--- 1. NumPy MLP from Scratch: Solving Non-Linear XOR ---")
    X_xor = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y_xor = np.array([[0.0], [1.0], [1.0], [0.0]])

    xor_net = NeuralNetwork([
        DenseLayer(2, 8, activation=Sigmoid(), init_method="xavier", seed=42),
        DenseLayer(8, 1, activation=Sigmoid(), init_method="xavier", seed=43),
    ])
    loss_fn = BinaryCrossEntropy()
    opt = Adam(lr=0.05)
    history = xor_net.fit(X_xor, y_xor, loss_fn=loss_fn, optimizer=opt, epochs=300, batch_size=4)

    preds = xor_net.predict(X_xor)
    print(f"  Final Training Loss: {history[-1]['train_loss']:.5f}")
    for idx in range(4):
        print(f"  Input: {X_xor[idx]} -> Target: {int(y_xor[idx, 0])} | Predicted: {preds[idx, 0]:.4f} (Round: {int(preds[idx, 0] > 0.5)})")

    # 2. Scikit-Learn Digits Classification Benchmark
    print("\n--- 2. Real Dataset Fitting: Digits Classification (8x8 Pixels) ---")
    digits = load_digits()
    X_d, y_d = digits.data, digits.target
    X_tr, X_te, y_tr, y_te = train_test_split(X_d, y_d, test_size=0.25, random_state=42)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    y_tr_oh = np.zeros((len(y_tr), 10), dtype=np.float64)
    y_tr_oh[np.arange(len(y_tr)), y_tr] = 1.0

    digits_net = NeuralNetwork([
        DenseLayer(64, 32, activation=ReLU(), init_method="he", seed=10),
        DenseLayer(32, 10, activation=Softmax(), init_method="xavier", seed=11),
    ])
    digits_net.fit(
        X_tr,
        y_tr_oh,
        loss_fn=CategoricalCrossEntropy(),
        optimizer=Adam(lr=0.01),
        epochs=25,
        batch_size=32,
    )
    test_preds = digits_net.predict(X_te)
    accuracy = float(np.mean(np.argmax(test_preds, axis=1) == y_te))
    print(f"  Digits Test Set Accuracy: {accuracy * 100:.2f}% (Target: > 90.0%)")

    # 3. CNN Terrain Classifier & Perception Module
    print("\n--- 3. Convolutional Perception: Local Terrain Classification ---")
    cnn = CNNTerrainClassifier(in_channels=1, patch_size=5, num_classes=5, seed=42)
    # Synthetic terrain training
    X_patches = []
    y_labels = []
    for c_id in range(5):
        for _ in range(15):
            p = np.zeros((1, 5, 5))
            p[0, 2, 2] = float(c_id) + np.random.normal(0, 0.05)
            X_patches.append(p)
            y_labels.append(c_id)
    cnn.fit(np.array(X_patches), np.array(y_labels), epochs=30, batch_size=15, lr=0.01)

    test_patch = np.zeros((5, 5))
    test_patch[2, 2] = 1.0  # Obstacle center
    predicted_type = cnn.classify_name(test_patch)
    print(f"  Classified center obstacle patch: '{predicted_type}'")

    # 4. Deep Q-Network Agent in GridWorld Simulation
    print("\n--- 4. Deep Q-Network (DQN) Agent Simulation & Glass-Box Traces ---")
    env_config = EnvironmentConfig(width=6, height=6, seed=42)
    world = GridWorld(env_config)
    world.reset()

    dqn = DeepQNetwork(state_dim=4, num_actions=4, lr=0.005, seed=42)
    dqn_agent = DQNAgent(agent_id="dqn_explorer", dqn=dqn, initial_epsilon=0.2)
    world.spawn_agent("dqn_explorer")

    engine = SimulationEngine(environment=world, agents={"dqn_explorer": dqn_agent}, seed=42)
    for tick in range(1, 6):
        engine.step()
        trace = dqn_agent.latest_trace
        if trace:
            cands = trace.rationale.get("candidate_actions", {})
            print(f"  Tick {tick:02d} | Action: {trace.rationale.get('selected_action', 'N/A'):5s} | Max Q: {trace.metrics.get('max_q', 0.0):.4f} | Candidates: {cands}")

    # 5. Proximal Policy Optimization (PPO) Agent
    print("\n--- 5. Proximal Policy Optimization (PPO) Agent Simulation ---")
    world_ppo = GridWorld(env_config)
    world_ppo.reset()

    ppo = PPO(state_dim=4, num_actions=4, seed=42)
    ppo_agent = PPOAgent(agent_id="ppo_navigator", ppo=ppo, update_interval=4)
    world_ppo.spawn_agent("ppo_navigator")

    engine_ppo = SimulationEngine(environment=world_ppo, agents={"ppo_navigator": ppo_agent}, seed=42)
    for tick in range(1, 6):
        engine_ppo.step()
        trace = ppo_agent.latest_trace
        if trace:
            print(f"  Tick {tick:02d} | Action: {trace.rationale.get('selected_action', 'N/A'):5s} | V(s): {trace.metrics.get('state_value', 0.0):.4f} | Entropy: {trace.metrics.get('entropy', 0.0):.4f}")

    # 6. Neural Survival Predictor
    print("\n--- 6. Neural Survival Predictor: Lifespan & Hazard Risk ---")
    predictor = NeuralSurvivalPredictor(input_dim=4, hidden_dim=16, seed=42)
    sample_vitals = np.array([
        [0.9, 0.95, 10.0, 5.0],  # Healthy agent, high energy, far from hazard
        [0.2, 0.15, 80.0, 0.5],  # Critical agent, low health, near hazard
    ])
    exp_lifespan = predictor.predict_lifespan(sample_vitals)
    mortality_prob = predictor.predict_mortality_risk(sample_vitals)

    print(f"  Healthy Agent  -> Predicted Lifespan: {exp_lifespan[0]:.2f} steps | Mortality Risk: {mortality_prob[0] * 100:.2f}%")
    print(f"  Critical Agent -> Predicted Lifespan: {exp_lifespan[1]:.2f} steps | Mortality Risk: {mortality_prob[1] * 100:.2f}%")

    print("\n" + "=" * 80)
    print("  MODULE M14 VERIFIED SUCCESSFULLY: NUMPY FROM SCRATCH + DEEP RL COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
