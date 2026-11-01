"""Monte Carlo Tree Search (MCTS) with canonical UCT selection and glass-box diagnostics (SYS-15, Module M17)."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import numpy as np

from autogen.intelligence.hybrid.contracts import MCTSNode
from autogen.simulation.state.types import ActionType


@runtime_checkable
class MCTSEnvironmentModel(Protocol):
    """Generative transition and reward model required for MCTS tree search rollouts."""

    def state_key(self, state: Any) -> str:
        """Returns unique string identifier for state representation."""
        ...

    def get_valid_actions(self, state: Any) -> list[ActionType]:
        """Returns valid legal actions applicable in the current state."""
        ...

    def step(self, state: Any, action: ActionType) -> tuple[Any, float, bool]:
        """Simulates environment transition, returning (next_state, reward, is_terminal)."""
        ...


class MonteCarloTreeSearch:
    """Monte Carlo Tree Search with UCT (Upper Confidence bounds applied to Trees)."""

    def __init__(
        self,
        model: MCTSEnvironmentModel,
        exploration_constant: float = 1.414,  # c = sqrt(2)
        rollout_depth: int = 10,
        discount_factor: float = 0.95,
        seed: int = 42,
    ) -> None:
        self.model = model
        self.c = exploration_constant
        self.rollout_depth = rollout_depth
        self.gamma = discount_factor
        self.rng = np.random.default_rng(seed)

    def select_uct_child(self, node: MCTSNode) -> tuple[ActionType, MCTSNode]:
        """Selects child node maximizing Upper Confidence Bound: Q + c * sqrt(ln(N) / n)."""
        best_uct = float("-inf")
        best_action: ActionType | None = None
        best_child: MCTSNode | None = None

        log_n = np.log(max(1, node.visits))
        for action, child in node.children.items():
            if child.visits == 0:
                uct = float("inf")
            else:
                exploitation = child.total_value / child.visits
                exploration = self.c * np.sqrt(log_n / child.visits)
                uct = exploitation + exploration

            if uct > best_uct:
                best_uct = uct
                best_action = action
                best_child = child

        assert best_action is not None and best_child is not None
        return best_action, best_child

    def search(
        self,
        root_state: Any,
        iterations: int = 50,
        step_callback: Callable[[int, ActionType, float], None] | None = None,
    ) -> tuple[ActionType, dict[str, Any]]:
        """Executes MCTS tree search over the generative model, returning robust action and diagnostics."""
        root_key = self.model.state_key(root_state)
        valid_actions = self.model.get_valid_actions(root_state)
        if not valid_actions:
            return ActionType.STAY, {"visits": 0, "q_value": 0.0}

        root = MCTSNode(
            state_key=root_key,
            untried_actions=list(valid_actions),
        )

        for iteration in range(iterations):
            node = root
            sim_state = copy.deepcopy(root_state)
            visited_nodes: list[tuple[MCTSNode, float]] = [(root, 0.0)]
            terminal = False

            # 1. Selection: descend tree using UCT until non-fully expanded node or terminal
            while node.is_fully_expanded() and not node.is_leaf() and not terminal:
                action, node = self.select_uct_child(node)
                sim_state, step_r, terminal = self.model.step(sim_state, action)
                visited_nodes.append((node, step_r))

            # 2. Expansion: expand one untried action if not terminal
            if not terminal and node.untried_actions:
                idx = int(self.rng.integers(0, len(node.untried_actions)))
                action = node.untried_actions.pop(idx)
                sim_state, step_r, terminal = self.model.step(sim_state, action)

                child_actions = [] if terminal else self.model.get_valid_actions(sim_state)
                child_node = MCTSNode(
                    state_key=self.model.state_key(sim_state),
                    parent=node,
                    action_from_parent=action,
                    untried_actions=list(child_actions),
                )
                node.children[action] = child_node
                node = child_node
                visited_nodes.append((node, step_r))

            # 3. Simulation (Rollout): execute random policy until terminal or depth limit
            rollout_rewards: list[float] = []
            depth = 0

            while not terminal and depth < self.rollout_depth:
                rollout_actions = self.model.get_valid_actions(sim_state)
                if not rollout_actions:
                    break
                r_idx = int(self.rng.integers(0, len(rollout_actions)))
                r_action = rollout_actions[r_idx]
                sim_state, r_reward, terminal = self.model.step(sim_state, r_action)
                rollout_rewards.append(r_reward)
                depth += 1

            # 4. Backpropagation: compute discounted return suffix for each visited tree node
            # Compute rollout return starting from the leaf
            rollout_return = 0.0
            for r in reversed(rollout_rewards):
                rollout_return = r + self.gamma * rollout_return

            # Backpropagate backwards through visited tree nodes
            g = rollout_return
            for v_node, step_r in reversed(visited_nodes[1:]):  # nodes below root
                g = step_r + self.gamma * g
                v_node.visits += 1
                v_node.total_value += g

            # Update root visits
            root.visits += 1
            root.total_value += g

            if step_callback is not None and root.children:
                best_temp = max(root.children.items(), key=lambda item: item[1].visits)[0]
                step_callback(iteration, best_temp, root.children[best_temp].q_value)

        # Select action using robust child criterion (most visited child)
        if not root.children:
            chosen_action = self.rng.choice(valid_actions)
            return chosen_action, {"visits": 0, "q_value": 0.0}

        best_action, best_child = max(
            root.children.items(),
            key=lambda item: item[1].visits,
        )

        diagnostics = {
            "root_visits": root.visits,
            "best_action_visits": best_child.visits,
            "best_action_q": best_child.q_value,
            "child_visits": {act.value: child.visits for act, child in root.children.items()},
            "child_q_values": {act.value: child.q_value for act, child in root.children.items()},
        }
        return best_action, diagnostics
