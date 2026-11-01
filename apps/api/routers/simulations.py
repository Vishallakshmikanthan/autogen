"""Simulations API router (M19, SYS-18)."""

from __future__ import annotations

import sqlite3
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from apps.api.dependencies import get_cmd_queue, get_db, get_worker
from apps.api.persistence.repositories.environment_repo import EnvironmentRepository
from apps.api.persistence.repositories.simulation_repo import SimulationRepository
from apps.api.schemas.common import APIResponseEnvelope
from apps.api.schemas.simulation_dtos import (
    CreateSimulationRequest,
    SimulationStateResponse,
    SimulationStatusResponse,
)
from apps.api.worker.contracts import SimulationCommand, SimulationCommandType
from apps.api.worker.queue import IJobQueue
from apps.api.worker.runner import SimulationWorker

router = APIRouter(prefix="/api/v1/simulations", tags=["simulations"])


@router.post("", response_model=APIResponseEnvelope[SimulationStatusResponse], status_code=status.HTTP_201_CREATED)
def create_simulation(
    req: CreateSimulationRequest,
    conn: sqlite3.Connection = Depends(get_db),
    worker: SimulationWorker = Depends(get_worker),
) -> Any:
    sim_repo = SimulationRepository(conn)
    env_repo = EnvironmentRepository(conn)

    width = req.width
    height = req.height
    obstacle_density = req.obstacle_density
    env_ver_id = None

    if req.env_id:
        env = env_repo.get_environment(req.env_id)
        if env:
            width = env["width"]
            height = env["height"]
            obstacle_density = env.get("config", {}).get("obstacle_density", obstacle_density)
            env_ver_id = env.get("version_id")

    created = sim_repo.create_simulation(
        name=req.name,
        seed=req.seed,
        env_version_id=env_ver_id,
        max_steps=req.max_steps,
    )
    conn.commit()

    agents_list = [a.model_dump() for a in req.agents] if req.agents else None
    # Initialize worker with simulation
    worker.initialize_simulation(
        sim_id=created["id"],
        width=width,
        height=height,
        obstacle_density=obstacle_density,
        seed=req.seed,
        max_steps=req.max_steps,
        agents_spec=agents_list,
    )

    sim_status = sim_repo.get_simulation(created["id"])
    return APIResponseEnvelope(
        success=True,
        data=SimulationStatusResponse.model_validate(sim_status),
        message="Simulation created",
    )


@router.get("/{sim_id}", response_model=APIResponseEnvelope[SimulationStatusResponse])
def get_simulation(
    sim_id: str,
    conn: sqlite3.Connection = Depends(get_db),
) -> Any:
    repo = SimulationRepository(conn)
    sim = repo.get_simulation(sim_id)
    if sim is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Simulation '{sim_id}' not found")
    return APIResponseEnvelope(
        success=True,
        data=SimulationStatusResponse.model_validate(sim),
        message="Simulation found",
    )


@router.post("/{sim_id}/start", response_model=APIResponseEnvelope[dict[str, str]])
def start_simulation(
    sim_id: str,
    cmd_queue: IJobQueue = Depends(get_cmd_queue),
) -> Any:
    cmd_queue.put(SimulationCommand(sim_id=sim_id, command=SimulationCommandType.START))
    return APIResponseEnvelope(success=True, data={"status": "STARTING"}, message="Start command queued")


@router.post("/{sim_id}/pause", response_model=APIResponseEnvelope[dict[str, str]])
def pause_simulation(
    sim_id: str,
    cmd_queue: IJobQueue = Depends(get_cmd_queue),
) -> Any:
    cmd_queue.put(SimulationCommand(sim_id=sim_id, command=SimulationCommandType.PAUSE))
    return APIResponseEnvelope(success=True, data={"status": "PAUSING"}, message="Pause command queued")


@router.post("/{sim_id}/step", response_model=APIResponseEnvelope[dict[str, str]])
def step_simulation(
    sim_id: str,
    cmd_queue: IJobQueue = Depends(get_cmd_queue),
    worker: SimulationWorker = Depends(get_worker),
) -> Any:
    # Synchronously execute single step for instant HTTP response
    worker.step_once()
    return APIResponseEnvelope(success=True, data={"status": "STEPPED"}, message="Step command executed")


@router.post("/{sim_id}/reset", response_model=APIResponseEnvelope[dict[str, str]])
def reset_simulation(
    sim_id: str,
    cmd_queue: IJobQueue = Depends(get_cmd_queue),
    worker: SimulationWorker = Depends(get_worker),
) -> Any:
    worker.handle_command(SimulationCommand(sim_id=sim_id, command=SimulationCommandType.RESET))
    return APIResponseEnvelope(success=True, data={"status": "RESET"}, message="Reset command executed")


@router.get("/{sim_id}/state", response_model=APIResponseEnvelope[SimulationStateResponse])
def get_simulation_state(
    sim_id: str,
    conn: sqlite3.Connection = Depends(get_db),
    worker: SimulationWorker = Depends(get_worker),
) -> Any:
    repo = SimulationRepository(conn)
    snapshot_row = repo.get_latest_snapshot(sim_id)
    if snapshot_row:
        return APIResponseEnvelope(
            success=True,
            data=SimulationStateResponse(
                sim_id=sim_id,
                step=snapshot_row["step"],
                state_hash=snapshot_row["state_hash"],
                snapshot=snapshot_row["snapshot"],
            ),
            message="State retrieved from snapshot",
        )

    if worker.sim_id == sim_id and worker.prev_viewer_state:
        return APIResponseEnvelope(
            success=True,
            data=SimulationStateResponse(
                sim_id=sim_id,
                step=worker.current_step,
                state_hash=worker.prev_viewer_state.state_hash,
                snapshot=worker.prev_viewer_state.model_dump(),
            ),
            message="State retrieved from live worker",
        )

    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No state snapshot available")


@router.get("/{sim_id}/metrics", response_model=APIResponseEnvelope[list[dict[str, Any]]])
def get_simulation_metrics(
    sim_id: str,
    conn: sqlite3.Connection = Depends(get_db),
) -> Any:
    repo = SimulationRepository(conn)
    metrics = repo.get_metrics(sim_id)
    return APIResponseEnvelope(success=True, data=metrics, message="Metrics retrieved")
