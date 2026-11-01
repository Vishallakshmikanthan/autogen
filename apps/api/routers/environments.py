"""Environments API router (M19, SYS-18)."""

from __future__ import annotations

import sqlite3
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from apps.api.dependencies import get_db
from apps.api.persistence.repositories.environment_repo import EnvironmentRepository
from apps.api.schemas.common import APIResponseEnvelope
from apps.api.schemas.environment_dtos import (
    CreateEnvironmentRequest,
    EnvironmentResponse,
)

router = APIRouter(prefix="/api/v1/environments", tags=["environments"])


@router.post("", response_model=APIResponseEnvelope[EnvironmentResponse], status_code=status.HTTP_201_CREATED)
def create_environment(
    req: CreateEnvironmentRequest,
    conn: sqlite3.Connection = Depends(get_db),
) -> Any:
    repo = EnvironmentRepository(conn)
    cfg = {
        "width": req.width,
        "height": req.height,
        "obstacle_density": req.obstacle_density,
        "resource_density": req.resource_density,
        "energy_max": req.energy_max,
    }
    created = repo.create_environment(name=req.name, width=req.width, height=req.height, config=cfg)
    return APIResponseEnvelope(
        success=True,
        data=EnvironmentResponse.model_validate(created),
        message="Environment created successfully",
    )


@router.get("", response_model=APIResponseEnvelope[list[dict[str, Any]]])
def list_environments(
    limit: int = 50,
    offset: int = 0,
    conn: sqlite3.Connection = Depends(get_db),
) -> Any:
    repo = EnvironmentRepository(conn)
    envs = repo.list_environments(limit=limit, offset=offset)
    return APIResponseEnvelope(success=True, data=envs, message="Environments retrieved")


@router.get("/{env_id}", response_model=APIResponseEnvelope[EnvironmentResponse])
def get_environment(
    env_id: str,
    conn: sqlite3.Connection = Depends(get_db),
) -> Any:
    repo = EnvironmentRepository(conn)
    env = repo.get_environment(env_id)
    if env is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Environment '{env_id}' not found")
    return APIResponseEnvelope(
        success=True,
        data=EnvironmentResponse.model_validate(env),
        message="Environment found",
    )

