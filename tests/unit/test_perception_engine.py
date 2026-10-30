"""Unit tests for PerceptionEngine across all modes, noise, and sensing options."""

from autogen.core.rng import SeedManager
from autogen.simulation.environment.config import EnvironmentConfig
from autogen.simulation.environment.grid_world import GridWorld
from autogen.simulation.perception.contracts import (
    AgentSensingConfig,
    NoiseConfig,
    PerceptionConfig,
    PerceptionMode,
)
from autogen.simulation.perception.engine import PerceptionEngine, bresenham_line
from autogen.simulation.state.types import Position, TerrainType


def test_bresenham_line():
    line = bresenham_line(0, 0, 3, 3)
    assert line == [(0, 0), (1, 1), (2, 2), (3, 3)]

    horizontal = bresenham_line(2, 5, 5, 5)
    assert horizontal == [(2, 5), (3, 5), (4, 5), (5, 5)]

    vertical = bresenham_line(4, 2, 4, 0)
    assert vertical == [(4, 2), (4, 1), (4, 0)]


def test_perception_mode_full():
    cfg = EnvironmentConfig(width=10, height=10, seed=123)
    world = GridWorld(config=cfg)
    world.reset(seed=123)
    world.spawn_agent("agent_1", Position(5, 5))

    engine = PerceptionEngine()
    obs = engine.generate_observation(
        world.state,
        "agent_1",
        PerceptionConfig(mode=PerceptionMode.FULL),
    )

    assert len(obs.visible_cells) == 100
    assert obs.agent_id == "agent_1"
    assert obs.position == Position(5, 5)


def test_perception_mode_local_window():
    cfg = EnvironmentConfig(width=20, height=20, seed=123)
    world = GridWorld(config=cfg)
    world.reset(seed=123)
    grid = [list(r) for r in world.state.grid]
    grid[10][10] = grid[10][10].model_copy(update={"terrain": TerrainType.GRASSLAND})
    world._current_state = world.state.model_copy(update={"grid": tuple(tuple(r) for r in grid)})
    world.spawn_agent("agent_1", Position(10, 10))

    engine = PerceptionEngine()
    # Radius = 2 -> (2*2+1)^2 = 25 cells
    obs = engine.generate_observation(
        world.state,
        "agent_1",
        PerceptionConfig(mode=PerceptionMode.LOCAL_WINDOW, radius=2),
    )

    assert len(obs.visible_cells) == 25
    for cell in obs.visible_cells:
        assert abs(cell.position.x - 10) <= 2
        assert abs(cell.position.y - 10) <= 2


def test_perception_mode_line_of_sight_occlusion():
    # Construct a custom grid with a wall blocking sight
    cfg = EnvironmentConfig(width=10, height=10, seed=42)
    world = GridWorld(config=cfg)
    world.reset(seed=42)

    # Spawn agent at (1, 1)
    world.spawn_agent("agent_1", Position(1, 1))

    # Manually place a mountain/wall obstacle at (2, 1)
    grid_copy = [list(row) for row in world.state.grid]
    blocked_cell = grid_copy[1][2].model_copy(update={"terrain": TerrainType.MOUNTAIN})
    grid_copy[1][2] = blocked_cell
    # Behind the wall at (3, 1) and (4, 1)
    world._current_state = world.state.model_copy(
        update={"grid": tuple(tuple(r) for r in grid_copy)}
    )

    engine = PerceptionEngine()
    obs = engine.generate_observation(
        world.state,
        "agent_1",
        PerceptionConfig(mode=PerceptionMode.LINE_OF_SIGHT, radius=4),
    )

    visible_positions = {(c.position.x, c.position.y) for c in obs.visible_cells}

    # The obstacle at (2, 1) is visible
    assert (2, 1) in visible_positions
    # Cells strictly behind the obstacle along the ray (3, 1) and (4, 1) must be occluded
    assert (3, 1) not in visible_positions
    assert (4, 1) not in visible_positions


def test_perception_sensor_noise_and_agent_sensing():
    cfg = EnvironmentConfig(width=15, height=15, seed=42)
    world = GridWorld(config=cfg)
    world.reset(seed=42)
    grid = [list(r) for r in world.state.grid]
    grid[5][5] = grid[5][5].model_copy(update={"terrain": TerrainType.GRASSLAND, "food": 10.0})
    grid[6][6] = grid[6][6].model_copy(update={"terrain": TerrainType.GRASSLAND, "food": 5.0})
    world._current_state = world.state.model_copy(update={"grid": tuple(tuple(r) for r in grid)})
    world.spawn_agent("agent_1", Position(5, 5))
    world.spawn_agent("agent_2", Position(6, 6))

    seed_mgr = SeedManager(master_seed=42)
    engine = PerceptionEngine(seed_manager=seed_mgr)

    # Test agent anonymization
    p_cfg_anon = PerceptionConfig(
        mode=PerceptionMode.LOCAL_WINDOW,
        radius=3,
        agent_sensing=AgentSensingConfig(anonymous=True, sensing_radius=3),
    )
    obs = engine.generate_observation(world.state, "agent_1", p_cfg_anon)
    assert len(obs.visible_agents) == 1
    assert obs.visible_agents[0][0] == "anonymous"
    assert obs.visible_agents[0][1] == Position(6, 6)

    # Test sensor noise on resources
    p_cfg_noise = PerceptionConfig(
        mode=PerceptionMode.LOCAL_WINDOW,
        radius=5,
        noise=NoiseConfig(enabled=True, resource_noise_std=2.0),
    )
    obs_noise_1 = engine.generate_observation(world.state, "agent_1", p_cfg_noise)
    # Ensure food amounts are non-negative
    for c in obs_noise_1.visible_cells:
        assert c.food >= 0.0
