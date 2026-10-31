"""Demonstration of Reinforcement Learning Engine: Bandits, Tabular Control, CliffWalking, and Dyna-Q (SYS-12, Module M13)."""

from __future__ import annotations

from typing import Any
import numpy as np

from autogen.agents.rl.q_learning_agent import QLearningAgent
from autogen.intelligence.rl.adapter import CliffWalkingEnv, GymEnvAdapter
from autogen.intelligence.rl.bandits import (
    EpsilonGreedyBandit,
    ThompsonSamplingBandit,
    UCBBandit,
)
from autogen.intelligence.rl.dyna import DynaQ
from autogen.intelligence.rl.tabular import QLearning, SARSA
from autogen.simulation.core.engine import SimulationEngine
from autogen.simulation.environment.config import EnvironmentConfig
from autogen.simulation.environment.grid_world import GridWorld


def run_demo() -> None:
    print("=" * 75)
    print("  AUTOGEN MODULE M13: REINFORCEMENT LEARNING ENGINE")
    print("=" * 75)

    # 1. Multi-Armed Bandits: Regret Comparison
    print("\n--- 1. Multi-Armed Bandits (10 Arms) ---")
    rng = np.random.default_rng(42)
    true_arm_means = np.array([0.1, 0.2, 0.35, 0.5, 0.85, 0.6, 0.4, 0.25, 0.7, 0.15])
    best_mean = float(np.max(true_arm_means))
    best_arm = int(np.argmax(true_arm_means))
    n_pulls = 1500

    eps_bandit = EpsilonGreedyBandit(n_arms=10, eps_0=1.0, decay_lambda=0.996, seed=42)
    ucb_bandit = UCBBandit(n_arms=10, c=1.414, seed=42)
    ts_bandit = ThompsonSamplingBandit(n_arms=10, seed=42)

    bandits: dict[str, tuple[Any, float]] = {
        "Epsilon-Greedy": (eps_bandit, 0.0),
        "UCB1": (ucb_bandit, 0.0),
        "Thompson Sampling": (ts_bandit, 0.0),
    }

    for t in range(1, n_pulls + 1):
        for name, (b, cum_regret) in list(bandits.items()):
            arm = b.select_arm()
            reward = 1.0 if rng.random() < true_arm_means[arm] else 0.0
            b.update(arm, reward)
            regret = best_mean - true_arm_means[arm]
            bandits[name] = (b, cum_regret + regret)

    print(f"True Best Arm: {best_arm} (payoff mean = {best_mean:.2f})")
    for name, (b, cum_regret) in bandits.items():
        chosen_best = int(np.argmax(getattr(b, "q_estimates", getattr(b, "alphas", [0]))))
        print(f"  {name:18s} | Identified Best: Arm {chosen_best} | Total Regret: {cum_regret:.2f}")

    # 2. CliffWalking: Q-Learning (Risky Optimal) vs SARSA (Safe Path)
    print("\n--- 2. CliffWalking Benchmark: SARSA vs Q-Learning ---")
    cliff_env = CliffWalkingEnv()

    ql = QLearning(num_actions=4, alpha=0.5, gamma=1.0, epsilon=0.1, seed=42)
    sarsa = SARSA(num_actions=4, alpha=0.5, gamma=1.0, epsilon=0.1, seed=42)

    # Train Q-Learning
    for _ in range(300):
        s = cliff_env.reset()
        done = False
        while not done:
            a = ql.select_action(s, explore=True)
            res = cliff_env.step(a)
            ql.update(s, a, res.reward, res.observation, res.terminated)
            s = res.observation
            done = res.terminated or res.truncated

    # Train SARSA
    for _ in range(300):
        s = cliff_env.reset()
        done = False
        a = sarsa.select_action(s, explore=True)
        while not done:
            res = cliff_env.step(a)
            next_a = sarsa.select_action(res.observation, explore=True) if not res.terminated else None
            sarsa.update(s, a, res.reward, res.observation, res.terminated, next_a)
            s = res.observation
            a = next_a if next_a is not None else 0
            done = res.terminated or res.truncated

    # Evaluate greedy trajectories
    def eval_path(alg: Any) -> tuple[float, list[tuple[int, int]]]:
        s = cliff_env.reset()
        path = [s]
        total_r = 0.0
        done = False
        while not done and len(path) < 50:
            a = alg.select_action(s, explore=False)
            res = cliff_env.step(a)
            total_r += res.reward
            s = res.observation
            path.append(s)
            done = res.terminated
        return total_r, path

    ql_ret, ql_path = eval_path(ql)
    sarsa_ret, sarsa_path = eval_path(sarsa)

    print(f"Q-Learning Path Length: {len(ql_path)} | Greedy Return: {ql_ret:.1f} (Traverses right along cliff edge row 1)")
    print(f"SARSA Path Length:      {len(sarsa_path)} | Greedy Return: {sarsa_ret:.1f} (Safely detours through upper rows)")

    # 3. Model-Based RL: Dyna-Q Sample Efficiency
    print("\n--- 3. Dyna-Q Sample Efficiency Gain ---")
    dyna_10 = DynaQ(num_actions=4, alpha=0.2, gamma=0.95, epsilon=0.1, n_planning_steps=10, seed=42)
    pure_q = QLearning(num_actions=4, alpha=0.2, gamma=0.95, epsilon=0.1, seed=42)

    # Train on small 6x6 grid
    def steps_to_goal(agent: Any, n_episodes: int) -> list[int]:
        env = CliffWalkingEnv(max_steps=100)
        episode_lengths = []
        for _ in range(n_episodes):
            s = env.reset()
            steps = 0
            done = False
            while not done:
                a = agent.select_action(s, explore=True)
                res = env.step(a)
                agent.update(s, a, res.reward, res.observation, res.terminated)
                s = res.observation
                steps += 1
                done = res.terminated or res.truncated
            episode_lengths.append(steps)
        return episode_lengths

    lengths_dyna = steps_to_goal(dyna_10, 40)
    lengths_q = steps_to_goal(pure_q, 40)
    print(f"Dyna-Q (n_plan=10) Average steps to goal in first 10 episodes: {np.mean(lengths_dyna[:10]):.1f}")
    print(f"Pure Q-Learning    Average steps to goal in first 10 episodes: {np.mean(lengths_q[:10]):.1f}")
    print(f"Sample Efficiency Gain (Speedup): {np.mean(lengths_q[:10]) / np.mean(lengths_dyna[:10]):.2f}x")

    # 4. QLearningAgent Running in SimulationEngine
    print("\n--- 4. QLearningAgent Live Simulation Execution ---")
    config = EnvironmentConfig(width=6, height=6, seed=42)
    world = GridWorld(config)
    world.reset()

    rl_agent = QLearningAgent(agent_id="q_learner", q_learning=ql)
    world.spawn_agent("q_learner")

    engine = SimulationEngine(environment=world, agents={"q_learner": rl_agent}, seed=42)

    for step in range(1, 5):
        state = engine.step()
        trace = rl_agent.latest_trace
        pos = state.agents["q_learner"].position
        action = trace.rationale.get("selected_action", "UNKNOWN") if trace else "UNKNOWN"
        reason = trace.rationale.get("reason", "") if trace else ""
        print(f"  Step {step}: Pos={pos}, Action={action}, Reason='{reason}'")

    print("\n[SUCCESS] Module M13 Demonstration Completed.")


if __name__ == "__main__":
    run_demo()
