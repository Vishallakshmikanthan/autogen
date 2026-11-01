"""Agents API router (M19, SYS-18)."""

from __future__ import annotations

import sqlite3
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from apps.api.dependencies import get_db
from apps.api.persistence.repositories.agent_repo import AgentRepository
from apps.api.schemas.agent_dtos import AgentResponse, CreateAgentRequest
from apps.api.schemas.common import APIResponseEnvelope

router = APIRouter(prefix="/api/v1/agents", tags=["agents"])


@router.post("", response_model=APIResponseEnvelope[AgentResponse], status_code=status.HTTP_201_CREATED)
def create_agent(
    req: CreateAgentRequest,
    conn: sqlite3.Connection = Depends(get_db),
) -> Any:
    repo = AgentRepository(conn)
    created = repo.create_agent(
        name=req.name,
        agent_type=req.agent_type,
        policy_params=req.policy_params,
    )
    return APIResponseEnvelope(
        success=True,
        data=AgentResponse.model_validate(created),
        message="Agent registered successfully",
    )


@router.get("", response_model=APIResponseEnvelope[list[dict[str, Any]]])
def list_agents(
    limit: int = 50,
    offset: int = 0,
    conn: sqlite3.Connection = Depends(get_db),
) -> Any:
    repo = AgentRepository(conn)
    agents = repo.list_agents(limit=limit, offset=offset)
    return APIResponseEnvelope(success=True, data=agents, message="Agents retrieved")


@router.get("/{agent_id}", response_model=APIResponseEnvelope[AgentResponse])
def get_agent(
    agent_id: str,
    conn: sqlite3.Connection = Depends(get_db),
) -> Any:
    repo = AgentRepository(conn)
    agent = repo.get_agent(agent_id)
    if agent is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Agent '{agent_id}' not found")
    return APIResponseEnvelope(
        success=True,
        data=AgentResponse.model_validate(agent),
        message="Agent found",
    )

