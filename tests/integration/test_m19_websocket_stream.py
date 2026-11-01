"""Integration tests for WebSocket stream and client playback controls (M19, SYS-18)."""

from fastapi.testclient import TestClient

from apps.api.dependencies import get_worker, set_db_manager
from apps.api.main import app
from apps.api.persistence.database import DatabaseManager


def test_websocket_stream_keyframe_and_controls():
    test_db = DatabaseManager(db_path=":memory:")
    set_db_manager(test_db)
    client = TestClient(app)

    # 1. Create a simulation session first
    sim_res = client.post(
        "/api/v1/simulations",
        json={"name": "ws_test_sim", "seed": 42, "width": 8, "height": 8, "max_steps": 10},
    )
    sim_id = sim_res.json()["data"]["id"]

    # 2. Connect via WebSocket
    with client.websocket_connect(f"/api/v1/simulations/{sim_id}/stream") as websocket:
        # Receive initial keyframe
        frame_msg = websocket.receive_json()
        assert frame_msg["event_type"] == "FRAME"
        assert frame_msg["step"] == 0
        assert frame_msg["delta"]["is_keyframe"] is True

        # Send control command to advance a step
        websocket.send_json({"command": "STEP"})

        # Manually trigger step on worker to push next frame
        worker = get_worker()
        worker.step_once()

        next_frame = websocket.receive_json()
        assert next_frame["event_type"] == "FRAME"
        assert next_frame["step"] == 1

