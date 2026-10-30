"""Discrete 2D GridWorld environment engine implementing exact mathematical dynamics."""

from __future__ import annotations

import numpy as np

from autogen.core.rng import SeedManager
from autogen.simulation.core.contracts import Observation
from autogen.simulation.environment.config import EnvironmentConfig
from autogen.simulation.environment.generator import ProceduralWorldGenerator
from autogen.simulation.environment.weather import WeatherEngine
from autogen.simulation.state.agent_state import AgentState
from autogen.simulation.state.types import (
    DIRECTION_DISPLACEMENTS,
    TERRAIN_PROPERTIES,
    ActionType,
    Position,
)
from autogen.simulation.state.world_state import CellState, WorldState


class GridWorld:
    """Discrete 2D GridWorld environment engine.

    Owns ground-truth simulation state and executes deterministic transitions
    according to docs/mathematical_foundations/world_model.md.
    """

    def __init__(self, config: EnvironmentConfig | None = None) -> None:
        self.config = config or EnvironmentConfig()
        self.seed_manager = SeedManager(self.config.seed)
        self.weather_engine = WeatherEngine(
            self.seed_manager.spawn_stream("weather"),
            enabled=self.config.weather_enabled,
        )
        self._current_state: WorldState | None = None

    @property
    def state(self) -> WorldState:
        """Returns the current immutable WorldState snapshot."""
        if self._current_state is None:
            raise RuntimeError("GridWorld has not been reset yet. Call reset() first.")
        return self._current_state

    def reset(self, seed: int | None = None) -> WorldState:
        """Resets the world to step 0 using a seeded procedural generator."""
        if seed is not None:
            self.config = self.config.model_copy(update={"seed": seed})
            self.seed_manager = SeedManager(seed)
            self.weather_engine = WeatherEngine(
                self.seed_manager.spawn_stream("weather"),
                enabled=self.config.weather_enabled,
            )

        gen_rng = self.seed_manager.spawn_stream("generator")
        generator = ProceduralWorldGenerator(self.config, gen_rng)
        initial_grid = generator.generate()

        initial_weather = self.weather_engine.step(0)

        self._current_state = WorldState(
            step=0,
            width=self.config.width,
            height=self.config.height,
            grid=initial_grid,
            agents={},
            weather=initial_weather,
        )
        return self._current_state

    def restore_state(self, state: WorldState) -> None:
        """Restores the environment state from an external snapshot."""
        self._current_state = state

    def spawn_agent(
        self,
        agent_id: str,
        position: Position | None = None,
        energy: float | None = None,
        health: float | None = None,
    ) -> AgentState:
        """Spawns an agent onto a walkable grid cell."""
        if self._current_state is None:
            raise RuntimeError("Cannot spawn agent before environment is reset.")

        if agent_id in self._current_state.agents:
            raise ValueError(f"Agent with ID '{agent_id}' already exists in world.")

        if position is None:
            # Find a walkable position deterministically
            walkable_positions: list[Position] = []
            for y in range(self.config.height):
                for x in range(self.config.width):
                    if TERRAIN_PROPERTIES[self._current_state.grid[y][x].terrain].walkable:
                        walkable_positions.append(Position(x, y))

            if not walkable_positions:
                raise RuntimeError("No walkable cells available to spawn agent.")

            spawn_rng = self.seed_manager.spawn_stream("agent_spawn")
            chosen_idx = int(spawn_rng.integers(0, len(walkable_positions)))
            position = walkable_positions[chosen_idx]
        else:
            if not self.is_in_bounds(position):
                raise ValueError(f"Spawn position {position} is out of bounds.")
            cell = self._current_state.get_cell(position)
            if not TERRAIN_PROPERTIES[cell.terrain].walkable:
                raise ValueError(f"Spawn position {position} is not walkable ({cell.terrain}).")

        agent_state = AgentState(
            agent_id=agent_id,
            position=position,
            energy=energy if energy is not None else self.config.energy_max,
            health=health if health is not None else self.config.health_max,
            carried_food=0.0,
            starvation_ticks=0,
            is_alive=True,
        )

        updated_agents = dict(self._current_state.agents)
        updated_agents[agent_id] = agent_state

        self._current_state = self._current_state.model_copy(update={"agents": updated_agents})
        return agent_state

    def get_observation(self, agent_id: str, vision_radius: int = 3) -> Observation:
        """Constructs a sensory observation for the specified agent."""
        if self._current_state is None:
            raise RuntimeError("Cannot get observation before environment is reset.")

        agent = self._current_state.agents.get(agent_id)
        if agent is None:
            raise ValueError(f"Agent '{agent_id}' does not exist in environment.")

        pos = agent.position
        w, h = self.config.width, self.config.height

        # Extract visible local cells within Chebyshev radius
        visible_cells: list[CellState] = []
        for dy in range(-vision_radius, vision_radius + 1):
            for dx in range(-vision_radius, vision_radius + 1):
                nx, ny = pos.x + dx, pos.y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    visible_cells.append(self._current_state.grid[ny][nx])

        # Extract visible other agents
        visible_agents: list[tuple[str, Position]] = []
        for other_id, other_agent in self._current_state.agents.items():
            if other_id != agent_id and other_agent.is_alive:
                if (
                    max(abs(other_agent.position.x - pos.x), abs(other_agent.position.y - pos.y))
                    <= vision_radius
                ):
                    visible_agents.append((other_id, other_agent.position))

        return Observation(
            agent_id=agent_id,
            step=self._current_state.step,
            position=pos,
            energy=agent.energy,
            health=agent.health,
            carried_food=agent.carried_food,
            is_alive=agent.is_alive,
            visible_cells=tuple(visible_cells),
            visible_agents=tuple(visible_agents),
        )

    def is_in_bounds(self, pos: Position) -> bool:
        """Checks if coordinate pos is strictly within the grid domain."""
        return 0 <= pos.x < self.config.width and 0 <= pos.y < self.config.height

    def is_walkable(self, pos: Position) -> bool:
        """Checks if coordinate pos is in-bounds and not an obstacle/water/mountain."""
        if not self.is_in_bounds(pos) or self._current_state is None:
            return False
        cell = self._current_state.get_cell(pos)
        return TERRAIN_PROPERTIES[cell.terrain].walkable

    def validate_action(self, agent_id: str, action: ActionType) -> bool:
        """Validates if action can be legally executed by agent_id.

        STAY and COLLECT are always legal actions.
        Movement is legal iff destination is in-bounds and walkable.
        """
        if self._current_state is None:
            return False
        agent = self._current_state.agents.get(agent_id)
        if agent is None or not agent.is_alive:
            return False

        if action in (ActionType.STAY, ActionType.COLLECT):
            return True

        displacement = DIRECTION_DISPLACEMENTS[action]
        target_pos = agent.position.offset(displacement[0], displacement[1])
        return self.is_walkable(target_pos)

    def apply_transitions(self, actions: dict[str, ActionType]) -> WorldState:
        """Applies validated joint agent actions and environmental dynamics.

        Transition order:
        1. Resolve agent movement and energy consumption
        2. Resolve resource collection
        3. Apply environmental resource regeneration
        4. Apply starvation and health updates
        5. Cull deceased agents (N_{t+1} = N_t - D_t)
        6. Advance step clock and return new immutable WorldState
        """
        if self._current_state is None:
            raise RuntimeError("Environment must be reset before taking transitions.")

        current = self._current_state
        next_step = current.step + 1

        # 1. Update weather
        next_weather = self.weather_engine.step(next_step)
        regen_weather_mod = next_weather.get("regen_modifier", 0.0)

        # Copy mutable working structures
        next_agents: dict[str, AgentState] = {}
        # 2D list of mutable food values
        working_food: list[list[float]] = [
            [current.grid[y][x].food for x in range(current.width)] for y in range(current.height)
        ]

        # Map to track food collected per cell in this step
        food_consumed_per_cell: dict[tuple[int, int], float] = {}

        # First pass: movement and collection decisions
        for agent_id, agent in current.agents.items():
            if not agent.is_alive:
                continue

            action = actions.get(agent_id, ActionType.STAY)
            curr_pos = agent.position

            # Initialize delta values
            next_pos = curr_pos
            energy_cost = self.config.cost_stay
            food_collected = 0.0

            if action in (ActionType.UP, ActionType.DOWN, ActionType.LEFT, ActionType.RIGHT):
                disp = DIRECTION_DISPLACEMENTS[action]
                target_pos = curr_pos.offset(disp[0], disp[1])
                if self.is_walkable(target_pos):
                    next_pos = target_pos
                    target_cell = current.get_cell(target_pos)
                    target_props = TERRAIN_PROPERTIES[target_cell.terrain]
                    energy_cost = target_props.move_cost * self.config.cost_base_move
                else:
                    # Illegal move: stays in place, penalizes idle cost
                    next_pos = curr_pos
                    energy_cost = self.config.cost_stay

            elif action == ActionType.COLLECT:
                energy_cost = self.config.cost_collect
                available_food = working_food[curr_pos.y][curr_pos.x]
                food_to_take = min(available_food, self.config.collect_rate)
                if food_to_take > 0.0:
                    food_collected = food_to_take
                    working_food[curr_pos.y][curr_pos.x] -= food_to_take
                    cell_key = (curr_pos.x, curr_pos.y)
                    food_consumed_per_cell[cell_key] = (
                        food_consumed_per_cell.get(cell_key, 0.0) + food_to_take
                    )

            elif action == ActionType.STAY:
                energy_cost = self.config.cost_stay

            # Energy calculation: E_{t+1} = min(E_max, max(0, E_t - C_a + R_f))
            energy_gain = food_collected * self.config.food_energy_ratio
            net_energy = agent.energy - energy_cost + energy_gain
            next_energy = float(np.clip(net_energy, 0.0, self.config.energy_max))

            # Starvation vs Health Separation:
            # Starvation triggers ONLY when energy drops to 0.0
            next_starvation_ticks = agent.starvation_ticks
            next_health = agent.health

            if next_energy == 0.0:
                next_starvation_ticks += 1
                next_health = max(0.0, agent.health - self.config.starvation_damage)
            else:
                next_starvation_ticks = 0
                if next_energy >= 0.5 * self.config.energy_max:
                    next_health = min(
                        self.config.health_max, agent.health + self.config.satiated_healing
                    )

            is_alive = next_health > 0.0

            next_agent_state = AgentState(
                agent_id=agent_id,
                position=next_pos,
                energy=next_energy,
                health=next_health,
                carried_food=agent.carried_food + food_collected,
                starvation_ticks=next_starvation_ticks,
                is_alive=is_alive,
                internal_memory_ref=agent.internal_memory_ref,
            )
            next_agents[agent_id] = next_agent_state

        # Resource dynamics: F_{t+1} = min(F_max, max(0, F_t + G_t - C_t))
        next_cell_rows: list[tuple[CellState, ...]] = []
        for y in range(current.height):
            row_cells: list[CellState] = []
            for x in range(current.width):
                cell = current.grid[y][x]
                props = TERRAIN_PROPERTIES[cell.terrain]

                current_f = working_food[y][x]
                if props.walkable:
                    effective_regen = props.regen_rate * (1.0 + regen_weather_mod)
                    new_f = float(np.clip(current_f + effective_regen, 0.0, props.food_capacity))
                else:
                    new_f = 0.0

                updated_cell = CellState(
                    position=cell.position,
                    terrain=cell.terrain,
                    food=new_f,
                    danger=cell.danger,
                    movement_cost=cell.movement_cost,
                )
                row_cells.append(updated_cell)
            next_cell_rows.append(tuple(row_cells))

        self._current_state = WorldState(
            step=next_step,
            width=current.width,
            height=current.height,
            grid=tuple(next_cell_rows),
            agents=next_agents,
            weather=next_weather,
        )
        return self._current_state
