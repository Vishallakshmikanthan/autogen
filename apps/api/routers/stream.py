"""WebSocket streaming router with backpressure and client control (M19, SYS-18)."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from apps.api.dependencies import get_cmd_queue, get_event_queue, get_worker
from apps.api.worker.contracts import SimulationCommand, SimulationCommandType
from apps.api.worker.delta_encoder import DeltaEncoder
from apps.api.worker.queue import InMemoryJobQueue
from apps.api.worker.runner import SimulationWorker

router = APIRouter(tags=["stream"])


@router.websocket("/api/v1/simulations/{sim_id}/stream")
async def simulation_stream(
    websocket: WebSocket,
    sim_id: str,
    cmd_queue: InMemoryJobQueue = Depends(get_cmd_queue),
    event_queue: InMemoryJobQueue = Depends(get_event_queue),
    worker: SimulationWorker = Depends(get_worker),
) -> None:
    """Delivers throttled, delta-compressed ViewerState frames over WebSocket."""
    await websocket.accept()

    # Send initial full keyframe if queue is empty
    if event_queue.empty() and worker.sim_id == sim_id and worker.prev_viewer_state:
        init_frame = DeltaEncoder.encode(None, worker.prev_viewer_state, is_keyframe=True)
        await websocket.send_json({
            "event_type": "FRAME",
            "sim_id": sim_id,
            "step": worker.current_step,
            "delta": init_frame,
            "state_hash": worker.prev_viewer_state.state_hash,
        })

    async def receive_controls() -> None:
        """Listens for bidirectional playback control messages from the browser client."""
        try:
            while True:
                msg = await websocket.receive_json()
                cmd_str = str(msg.get("command", "")).upper()
                if cmd_str == "PLAY" or cmd_str == "START":
                    cmd_queue.put(SimulationCommand(sim_id=sim_id, command=SimulationCommandType.START))
                elif cmd_str == "PAUSE":
                    cmd_queue.put(SimulationCommand(sim_id=sim_id, command=SimulationCommandType.PAUSE))
                elif cmd_str == "STEP":
                    cmd_queue.put(SimulationCommand(sim_id=sim_id, command=SimulationCommandType.STEP))
                elif cmd_str == "RESET":
                    cmd_queue.put(SimulationCommand(sim_id=sim_id, command=SimulationCommandType.RESET))
                elif cmd_str == "SPEED":
                    speed_val = float(msg.get("speed", 1.0))
                    cmd_queue.put(
                        SimulationCommand(
                            sim_id=sim_id,
                            command=SimulationCommandType.SET_SPEED,
                            params={"speed": speed_val},
                        )
                    )
        except (WebSocketDisconnect, asyncio.CancelledError):
            pass

    control_task = asyncio.create_task(receive_controls())

    try:
        while True:
            # Backpressure handling: if queue backed up, drain stale frames and send keyframe
            if event_queue.qsize() > 10:
                # Drain stale delta frames
                while not event_queue.empty():
                    try:
                        event_queue.get(block=False)
                    except Exception:
                        break
                # Emit resync keyframe
                if worker.sim_id == sim_id and worker.prev_viewer_state:
                    resync_frame = DeltaEncoder.encode(None, worker.prev_viewer_state, is_keyframe=True)
                    await websocket.send_json({
                        "event_type": "FRAME",
                        "sim_id": sim_id,
                        "step": worker.current_step,
                        "delta": resync_frame,
                        "state_hash": worker.prev_viewer_state.state_hash,
                    })

            if not event_queue.empty():
                try:
                    event = event_queue.get(block=False)
                    if isinstance(event, dict) and event.get("sim_id") == sim_id:
                        await websocket.send_json(event)
                except Exception:
                    pass

            await asyncio.sleep(0.01)

    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    finally:
        control_task.cancel()
