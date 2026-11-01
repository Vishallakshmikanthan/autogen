"""Integration tests for FastAPI REST endpoints using TestClient (M19, SYS-18)."""

import pytest
from fastapi.testclient import TestClient

from apps.api.dependencies import set_db_manager
from apps.api.main import app
from apps.api.persistence.database import DatabaseManager


@pytest.fixture(autouse=True)
def setup_test_db():
    # Use isolated in-memory DB for each test
    test_db = DatabaseManager(db_path=":memory:")
    set_db_manager(test_db)
    yield


def test_health_check_endpoint():
    client = TestClient(app)
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "HEALTHY"


def test_environment_endpoints_crud_and_validation():
    client = TestClient(app)

    # 1. Create environment
    create_payload = {
        "name": "arena_alpha",
        "width": 16,
        "height": 16,
        "obstacle_density": 0.12,
        "resource_density": 0.08,
        "energy_max": 100.0,
    }
    res = client.post("/api/v1/environments", json=create_payload)
    assert res.status_code == 201
    env_data = res.json()["data"]
    env_id = env_data["id"]
    assert env_data["name"] == "arena_alpha"

    # 2. Get environment
    get_res = client.get(f"/api/v1/environments/{env_id}")
    assert get_res.status_code == 200
    assert get_res.json()["data"]["id"] == env_id

    # 3. Validation error: invalid width (< 4)
    bad_payload = dict(create_payload)
    bad_payload["width"] = 2
    bad_res = client.post("/api/v1/environments", json=bad_payload)
    assert bad_res.status_code == 422

    # 4. 404 on non-existent environment
    not_found_res = client.get("/api/v1/environments/non_existent_id")
    assert not_found_res.status_code == 404


def test_agent_endpoints_crud():
    client = TestClient(app)
    payload = {
        "name": "forager_beta",
        "agent_type": "simple_reflex",
        "policy_params": {"threshold": 20.0},
    }
    res = client.post("/api/v1/agents", json=payload)
    assert res.status_code == 201
    ag_id = res.json()["data"]["id"]

    get_res = client.get(f"/api/v1/agents/{ag_id}")
    assert get_res.status_code == 200
    assert get_res.json()["data"]["name"] == "forager_beta"


def test_simulation_lifecycle_endpoints():
    client = TestClient(app)

    # 1. Create simulation
    sim_payload = {
        "name": "integration_sim",
        "seed": 101,
        "width": 12,
        "height": 12,
        "max_steps": 25,
        "agents": [{"agent_id": "sim_ag_1", "agent_type": "simple_reflex"}],
    }
    res = client.post("/api/v1/simulations", json=sim_payload)
    assert res.status_code == 201
    sim_id = res.json()["data"]["id"]

    # 2. Step simulation
    step_res = client.post(f"/api/v1/simulations/{sim_id}/step")
    assert step_res.status_code == 200
    assert step_res.json()["data"]["status"] == "STEPPED"

    # 3. Get state snapshot
    state_res = client.get(f"/api/v1/simulations/{sim_id}/state")
    assert state_res.status_code == 200
    assert state_res.json()["data"]["step"] >= 0

    # 4. Reset simulation
    reset_res = client.post(f"/api/v1/simulations/{sim_id}/reset")
    assert reset_res.status_code == 200
    assert reset_res.json()["data"]["status"] == "RESET"

