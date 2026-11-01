"""CLI demonstration of Module M19 Persistence, FastAPI, and Simulation Worker (SYS-18)."""

import sys
from pathlib import Path

# Ensure root workspace is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from apps.api.main import app, export_openapi_json


def main() -> None:
    print("=" * 70)
    print("AUTOGEN AI — M19 Persistence, FastAPI & Worker Demo (SYS-18)")
    print("=" * 70)

    # 1. Export OpenAPI spec for TypeScript generation
    openapi_path = export_openapi_json()
    print(f"\n[1] OpenAPI JSON schema generated at: {openapi_path.resolve()}")

    # 2. Test API client
    client = TestClient(app)

    # Health check
    health = client.get("/health").json()
    print(f"\n[2] System Health Check: {health}")

    # Register Environment
    print("\n[3] Creating Environment definition via REST...")
    env_payload = {
        "name": "demo_arena",
        "width": 16,
        "height": 16,
        "obstacle_density": 0.10,
        "resource_density": 0.15,
        "energy_max": 100.0,
    }
    env_res = client.post("/api/v1/environments", json=env_payload).json()
    env_id = env_res["data"]["id"]
    print(f"    - Environment created: ID = {env_id}, Version = {env_res['data']['version']}")

    # Register Agent
    print("\n[4] Registering Agent architecture via REST...")
    agent_payload = {
        "name": "explorer_alpha",
        "agent_type": "simple_reflex",
        "policy_params": {"curiosity_weight": 0.75},
    }
    agent_res = client.post("/api/v1/agents", json=agent_payload).json()
    ag_id = agent_res["data"]["id"]
    print(f"    - Agent registered: ID = {ag_id}, Type = {agent_res['data']['agent_type']}")

    # Spawn Simulation session
    print("\n[5] Initializing Simulation session and Worker runtime...")
    sim_payload = {
        "name": "demo_simulation_session",
        "seed": 42,
        "env_id": env_id,
        "max_steps": 50,
        "agents": [{"agent_id": "demo_agent_1", "agent_type": "simple_reflex"}],
    }
    sim_res = client.post("/api/v1/simulations", json=sim_payload).json()
    sim_id = sim_res["data"]["id"]
    print(f"    - Simulation spawned: ID = {sim_id}, Status = {sim_res['data']['status']}")

    # Step simulation via API
    print("\n[6] Stepping simulation via worker control endpoints...")
    for _ in range(5):
        client.post(f"/api/v1/simulations/{sim_id}/step")

    # Query latest state snapshot
    state_res = client.get(f"/api/v1/simulations/{sim_id}/state").json()
    print(f"    - Current Step: {state_res['data']['step']}")
    print(f"    - State Hash:   {state_res['data']['state_hash'][:16]}...")

    # Connect WebSocket stream
    print("\n[7] Connecting to WebSocket live stream (/api/v1/simulations/{id}/stream)...")
    with client.websocket_connect(f"/api/v1/simulations/{sim_id}/stream") as ws:
        frame = ws.receive_json()
        print(f"    - Received stream event: type = {frame['event_type']}, step = {frame['step']}")
        print(f"    - Keyframe flag: {frame['delta'].get('is_keyframe')}")

    print("\n" + "=" * 70)
    print("M19 Persistence, FastAPI & Worker demonstration completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
