"""Property-based invariant test: API-driven simulation determinism matches headless run (SYS-18)."""

from fastapi.testclient import TestClient

from apps.api.dependencies import set_db_manager
from apps.api.main import app
from apps.api.persistence.database import DatabaseManager
from autogen.agents.baseline.simple_reflex_agent import SimpleReflexAgent
from autogen.simulation.core.engine import SimulationEngine
from autogen.simulation.environment.config import EnvironmentConfig
from autogen.simulation.environment.grid_world import GridWorld
from autogen.simulation.state.types import TerrainType
from autogen.simulation.state.world_state import canonical_world_hash


def test_api_run_equals_headless_run_determinism():
    seed = 333
    width = 10
    height = 10
    steps = 15

    # 1. Run headless simulation directly
    terrain_weights = {
        TerrainType.GRASSLAND: 0.65,
        TerrainType.FOREST: 0.20,
        TerrainType.DESERT: 0.0,
        TerrainType.WATER: 0.0,
        TerrainType.MOUNTAIN: 0.0,
        TerrainType.OBSTACLE: 0.15,
    }
    env_cfg = EnvironmentConfig(
        width=width,
        height=height,
        seed=seed,
        energy_max=100.0,
        terrain_weights=terrain_weights,
    )
    headless_world = GridWorld(config=env_cfg)
    headless_world.reset(seed=seed)
    headless_world.spawn_agent(agent_id="test_ag", energy=100.0)

    headless_engine = SimulationEngine(environment=headless_world, seed=seed)
    headless_agent = SimpleReflexAgent(agent_id="test_ag")
    headless_engine.register_agent(headless_agent)

    for _ in range(steps):
        headless_engine.step()

    headless_hash = canonical_world_hash(headless_world.state)

    # 2. Run identical simulation via API and worker
    test_db = DatabaseManager(db_path=":memory:")
    set_db_manager(test_db)
    client = TestClient(app)

    sim_res = client.post(
        "/api/v1/simulations",
        json={
            "name": "determinism_sim",
            "seed": seed,
            "width": width,
            "height": height,
            "obstacle_density": 0.15,
            "max_steps": steps,
            "agents": [{"agent_id": "test_ag", "agent_type": "simple_reflex"}],
        },
    )
    sim_id = sim_res.json()["data"]["id"]

    for _ in range(steps):
        client.post(f"/api/v1/simulations/{sim_id}/step")

    state_res = client.get(f"/api/v1/simulations/{sim_id}/state")
    api_hash = state_res.json()["data"]["state_hash"]

    # Invariant: Hash from API-driven execution equals headless execution
    assert api_hash == headless_hash
