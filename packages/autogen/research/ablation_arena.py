"""Ablation Arena: systematic statistical evaluation of hybrid cognitive subsystems (SYS-15, Module M17)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from autogen.agents.hybrid.config import HybridAgentConfig
from autogen.agents.hybrid.hybrid_agent import HybridCognitiveAgent
from autogen.analytics.statistics import cohens_d as compute_cohens_d
from autogen.simulation.core.contracts import Observation
from autogen.simulation.environment.grid_world import EnvironmentConfig, GridWorld
from autogen.simulation.state.types import ActionType, Position, TerrainType
from autogen.simulation.state.world_state import CellState


@dataclass
class AblationResult:
    """Statistical evaluation metrics for a specific cognitive architecture configuration."""

    configuration_name: str
    mean_return: float
    std_return: float
    survival_rate: float
    mean_shield_vetoes: float
    cohens_d_vs_full: float = 0.0
    episodes: int = 0
    raw_returns: list[float] = field(default_factory=list)


class AblationArena:
    """Orchestrates systematic ablation experiments across held-out evaluation worlds."""

    def __init__(
        self,
        eval_seeds: list[int] | None = None,
        max_steps_per_episode: int = 30,
        grid_size: int = 16,
    ) -> None:
        self.eval_seeds = eval_seeds or [101, 202, 303, 404, 505, 606, 707, 808, 909, 1001]
        self.max_steps = max_steps_per_episode
        self.grid_size = grid_size

    def evaluate_configuration(
        self,
        name: str,
        config: HybridAgentConfig,
    ) -> AblationResult:
        """Runs the given agent configuration across all evaluation seeds."""
        returns: list[float] = []
        survivals: list[float] = []
        vetoes_list: list[float] = []

        for seed in self.eval_seeds:
            env_config = EnvironmentConfig(
                width=self.grid_size,
                height=self.grid_size,
                seed=seed,
            )
            _world = GridWorld(config=env_config)
            agent = HybridCognitiveAgent(agent_id=f"ablation_{name}", config=config)

            episode_return = 0.0
            survived = True
            shield_vetoes = 0

            # Generate synthetic evaluation episode with hazards and food
            rng = np.random.default_rng(seed)
            hazard_positions = {
                Position(
                    x=int(rng.integers(1, self.grid_size)), y=int(rng.integers(1, self.grid_size))
                )
                for _ in range(self.grid_size // 2)
            }
            food_positions = {
                Position(
                    x=int(rng.integers(1, self.grid_size)), y=int(rng.integers(1, self.grid_size))
                )
                for _ in range(self.grid_size // 3)
            }

            current_pos = Position(x=0, y=0)
            energy = 100.0

            for step in range(self.max_steps):
                # Build local observation
                visible_cells = []
                for h_pos in hazard_positions:
                    if abs(h_pos.x - current_pos.x) <= 2 and abs(h_pos.y - current_pos.y) <= 2:
                        c = CellState(
                            position=h_pos,
                            terrain=TerrainType.OBSTACLE,
                            food=0.0,
                            danger=1.0,
                            movement_cost=1.0,
                        )
                        visible_cells.append(c)
                for f_pos in food_positions:
                    if abs(f_pos.x - current_pos.x) <= 2 and abs(f_pos.y - current_pos.y) <= 2:
                        c = CellState(
                            position=f_pos,
                            terrain=TerrainType.GRASSLAND,
                            food=10.0,
                            danger=0.0,
                            movement_cost=1.0,
                        )
                        visible_cells.append(c)

                obs = Observation(
                    agent_id=agent.agent_id,
                    step=step,
                    position=current_pos,
                    energy=energy,
                    health=100.0,
                    carried_food=0,
                    visible_cells=tuple(visible_cells),
                    visible_agents=(),
                )

                agent.observe(obs)
                action, trace = agent.decide()
                if trace.metrics.get("shield_vetoed", 0) == 1:
                    shield_vetoes += 1

                # Execute transition
                dx, dy = 0, 0
                if action.action_type == ActionType.UP:
                    dy = -1
                elif action.action_type == ActionType.DOWN:
                    dy = 1
                elif action.action_type == ActionType.LEFT:
                    dx = -1
                elif action.action_type == ActionType.RIGHT:
                    dx = 1

                next_pos = Position(
                    x=int(np.clip(current_pos.x + dx, 0, self.grid_size - 1)),
                    y=int(np.clip(current_pos.y + dy, 0, self.grid_size - 1)),
                )
                current_pos = next_pos
                energy -= 1.0

                # Check hazard collision
                step_reward = -0.1
                if current_pos in hazard_positions:
                    step_reward = -50.0
                    energy -= 40.0
                    if energy <= 0:
                        survived = False
                        episode_return += step_reward
                        agent.learn(step_reward, terminal=True)
                        break

                # Check food collection
                if current_pos in food_positions:
                    step_reward = 10.0
                    energy = min(100.0, energy + 20.0)
                    food_positions.remove(current_pos)

                episode_return += step_reward
                agent.learn(step_reward, terminal=False)

            returns.append(episode_return)
            survivals.append(1.0 if survived else 0.0)
            vetoes_list.append(float(shield_vetoes))

        return AblationResult(
            configuration_name=name,
            mean_return=float(np.mean(returns)),
            std_return=float(np.std(returns)),
            survival_rate=float(np.mean(survivals)),
            mean_shield_vetoes=float(np.mean(vetoes_list)),
            episodes=len(returns),
            raw_returns=returns,
        )

    def run_ablation_study(self) -> dict[str, AblationResult]:
        """Runs Full and all ablated variants, computing effect sizes vs Full."""
        configs = {
            "Full": HybridAgentConfig(grid_width=self.grid_size, grid_height=self.grid_size),
            "No_Shield": HybridAgentConfig(
                enable_safety_shield=False, grid_width=self.grid_size, grid_height=self.grid_size
            ),
            "No_Bayes": HybridAgentConfig(
                enable_bayes=False, grid_width=self.grid_size, grid_height=self.grid_size
            ),
            "No_Options": HybridAgentConfig(
                enable_options=False, grid_width=self.grid_size, grid_height=self.grid_size
            ),
            "No_AStar": HybridAgentConfig(
                enable_astar=False, grid_width=self.grid_size, grid_height=self.grid_size
            ),
            "No_Memory": HybridAgentConfig(
                enable_memory=False, grid_width=self.grid_size, grid_height=self.grid_size
            ),
        }

        results: dict[str, AblationResult] = {}
        for name, cfg in configs.items():
            res = self.evaluate_configuration(name, cfg)
            results[name] = res

        # Compute Cohen's d effect size relative to Full
        full_returns = results["Full"].raw_returns
        for name, res in results.items():
            if name != "Full":
                res.cohens_d_vs_full = compute_cohens_d(full_returns, res.raw_returns)

        return results
