"""Security property test: Assert sentinel values outside permitted sensory window never leak."""

from autogen.simulation.environment.config import EnvironmentConfig
from autogen.simulation.environment.grid_world import GridWorld
from autogen.simulation.perception.contracts import PerceptionConfig, PerceptionMode
from autogen.simulation.state.types import Position


def test_sentinel_values_outside_window_never_leak():
    """Property test injecting sentinel values outside the permitted window to assert zero leakage."""
    width, height = 30, 30
    cfg = EnvironmentConfig(width=width, height=height, seed=42)
    world = GridWorld(config=cfg)
    world.reset(seed=42)

    # Inject walkable cells and sentinel food
    grid_copy = [list(row) for row in world.state.grid]
    from autogen.simulation.state.types import TerrainType

    grid_copy[10][10] = grid_copy[10][10].model_copy(update={"terrain": TerrainType.GRASSLAND})
    grid_copy[24][24] = grid_copy[24][24].model_copy(update={"terrain": TerrainType.GRASSLAND})

    # Inject high-value sentinel food outside sensory radius (e.g. at (25, 25))
    sentinel_x, sentinel_y = 25, 25
    sentinel_food_amount = 99999.0

    sentinel_cell = grid_copy[sentinel_y][sentinel_x].model_copy(
        update={"food": sentinel_food_amount}
    )
    grid_copy[sentinel_y][sentinel_x] = sentinel_cell

    world._current_state = world.state.model_copy(
        update={"grid": tuple(tuple(r) for r in grid_copy)}
    )

    agent_id = "test_agent"
    agent_pos = Position(10, 10)
    world.spawn_agent(agent_id, agent_pos)

    # Spawn a sentinel outside agent at (24, 24)
    world.spawn_agent("sentinel_agent", Position(24, 24))

    from autogen.simulation.perception.engine import PerceptionEngine

    engine = PerceptionEngine()

    for r in [1, 2, 3, 5, 8]:
        p_cfg = PerceptionConfig(mode=PerceptionMode.LOCAL_WINDOW, radius=r)
        obs = engine.generate_observation(world.state, agent_id, p_cfg)

        # Assert no cell in observation has the sentinel coordinates or sentinel food
        for cell in obs.visible_cells:
            assert cell.food != sentinel_food_amount, (
                f"Sentinel food leaked into observation at radius {r}!"
            )
            assert (cell.position.x, cell.position.y) != (sentinel_x, sentinel_y)
            assert max(abs(cell.position.x - agent_pos.x), abs(cell.position.y - agent_pos.y)) <= r

        # Assert sentinel agent is never visible
        visible_agent_ids = [aid for aid, _ in obs.visible_agents]
        assert "sentinel_agent" not in visible_agent_ids, (
            f"Sentinel agent leaked into visible_agents at radius {r}!"
        )
