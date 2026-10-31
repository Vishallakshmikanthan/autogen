"""RL Training Harness with strict seed separation, evaluation, and telemetry (SYS-12, Module M13)."""

from __future__ import annotations

from typing import Any, Callable
import numpy as np

from autogen.intelligence.rl.contracts import EvaluationMetric, RLAlgorithm, StateEncoder
from autogen.intelligence.rl.encoders import TabularGridEncoder


class RLTrainingHarness:
    """Manages RL training and evaluation with strictly separated seed streams."""

    def __init__(
        self,
        algorithm: RLAlgorithm,
        env_factory: Callable[[int], Any],
        encoder: StateEncoder | None = None,
        train_seed: int = 1000,
        eval_seed: int = 2000,
    ) -> None:
        self.algorithm = algorithm
        self.env_factory = env_factory
        self.encoder = encoder or TabularGridEncoder()

        # Strictly separate SeedSequences
        self.train_seq = np.random.SeedSequence(train_seed)
        self.eval_seq = np.random.SeedSequence(eval_seed)

        self.history: list[EvaluationMetric] = []

    def train(
        self,
        n_episodes: int = 200,
        eval_interval: int = 25,
        n_eval_episodes: int = 10,
    ) -> list[EvaluationMetric]:
        train_seeds = [int(s.generate_state(1)[0]) for s in self.train_seq.spawn(n_episodes)]
        self.history = []

        for ep in range(1, n_episodes + 1):
            seed = train_seeds[ep - 1]
            env = self.env_factory(seed)
            obs = env.reset(seed=seed)
            state = self.encoder.encode(obs)

            terminated = False
            truncated = False
            ep_td_errors: list[float] = []

            while not (terminated or truncated):
                action = self.algorithm.select_action(state, explore=True)
                step_res = env.step(action)
                next_state = self.encoder.encode(step_res.observation)

                td_err = self.algorithm.update(
                    state=state,
                    action=action,
                    reward=step_res.reward,
                    next_state=next_state,
                    terminated=step_res.terminated,
                )
                ep_td_errors.append(abs(td_err))

                state = next_state
                terminated = step_res.terminated
                truncated = step_res.truncated

            if ep % eval_interval == 0:
                metric = self.evaluate(n_eval_episodes=n_eval_episodes)
                self.history.append(metric)

        return self.history

    def evaluate(self, n_eval_episodes: int = 10) -> EvaluationMetric:
        """Evaluates policy greedily (explore=False) on fresh held-out evaluation seeds."""
        eval_seeds = [int(s.generate_state(1)[0]) for s in self.eval_seq.spawn(n_eval_episodes)]
        returns: list[float] = []
        lengths: list[int] = []

        for seed in eval_seeds:
            env = self.env_factory(seed)
            obs = env.reset(seed=seed)
            state = self.encoder.encode(obs)

            ep_return = 0.0
            steps = 0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action = self.algorithm.select_action(state, explore=False)
                step_res = env.step(action)
                ep_return += step_res.reward
                steps += 1
                state = self.encoder.encode(step_res.observation)
                terminated = step_res.terminated
                truncated = step_res.truncated

            returns.append(ep_return)
            lengths.append(steps)

        eps = getattr(self.algorithm, "epsilon", 0.0)
        return EvaluationMetric(
            mean_return=float(np.mean(returns)),
            std_return=float(np.std(returns)),
            mean_length=float(np.mean(lengths)),
            td_error_mean=0.0,
            epsilon=float(eps),
        )

