"""Conflict resolution policies for simultaneous movement and contested resource collection."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from autogen.simulation.core.contracts import (
    Action,
    CollectPolicy,
    ConflictPolicy,
)
from autogen.simulation.state.types import (
    DIRECTION_DISPLACEMENTS,
    ActionType,
    Position,
)
from autogen.simulation.state.world_state import WorldState


class ConflictResolver:
    """Arbitrates contested actions occurring simultaneously at phase 5 of simulation tick."""

    def __init__(
        self,
        conflict_policy: ConflictPolicy = ConflictPolicy.CANCEL_ALL,
        collect_policy: CollectPolicy = CollectPolicy.PROPORTIONAL,
    ) -> None:
        self.conflict_policy = conflict_policy
        self.collect_policy = collect_policy

    def resolve_conflicts(
        self,
        current_state: WorldState,
        actions: dict[str, Action],
        rng: np.random.Generator,
    ) -> dict[str, Action]:
        """Resolves collisions and simultaneous claims, returning arbitrated actions."""
        # 1. Map target positions for movement actions
        target_positions: dict[str, Position] = {}
        claims_per_cell: dict[Position, list[str]] = defaultdict(list)

        for agent_id, action in actions.items():
            agent = current_state.agents.get(agent_id)
            if agent is None or not agent.is_alive:
                continue

            curr_pos = agent.position
            if action.action_type in (
                ActionType.UP,
                ActionType.DOWN,
                ActionType.LEFT,
                ActionType.RIGHT,
            ):
                disp = DIRECTION_DISPLACEMENTS[action.action_type]
                target_pos = curr_pos.offset(disp[0], disp[1])
            else:
                target_pos = curr_pos

            target_positions[agent_id] = target_pos
            claims_per_cell[target_pos].append(agent_id)

        # 2. Arbitrate contested target cells
        arbitrated_actions: dict[str, Action] = dict(actions)

        for cell_pos, claimants in claims_per_cell.items():
            if len(claimants) <= 1:
                continue

            # Identify agents actually attempting to move into this cell (not already stationary there)
            moving_claimants = [
                aid for aid in claimants if current_state.agents[aid].position != cell_pos
            ]

            if len(moving_claimants) <= 1:
                continue

            if self.conflict_policy == ConflictPolicy.CANCEL_ALL:
                # All colliding moving agents bounce back to STAY
                for aid in moving_claimants:
                    arbitrated_actions[aid] = Action(
                        agent_id=aid,
                        action_type=ActionType.STAY,
                        parameters={
                            "collision_cancelled": True,
                            "target": (cell_pos.x, cell_pos.y),
                        },
                    )
            elif self.conflict_policy == ConflictPolicy.SEEDED_PRIORITY:
                # Seeded random permutation: one winner selected, others stay
                permuted = list(moving_claimants)
                rng.shuffle(permuted)
                winner = permuted[0]
                for loser in permuted[1:]:
                    arbitrated_actions[loser] = Action(
                        agent_id=loser,
                        action_type=ActionType.STAY,
                        parameters={"collision_priority_lost": True, "winner": winner},
                    )

        return arbitrated_actions
