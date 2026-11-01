"""Worker command, event, and control protocol definitions (M19, SYS-18)."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SimulationCommandType(str, Enum):
    """Permitted worker control commands."""

    START = "START"
    PAUSE = "PAUSE"
    STEP = "STEP"
    RESET = "RESET"
    STOP = "STOP"
    SET_SPEED = "SET_SPEED"


class SimulationCommand(BaseModel):
    """Inter-process command dispatched to the simulation worker."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sim_id: str
    command: SimulationCommandType
    params: dict[str, Any] = Field(default_factory=dict)


class WorkerEvent(BaseModel):
    """Event emitted by the worker back to the API server and WebSocket broadcast."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sim_id: str
    event_type: str  # "FRAME", "STATE_CHANGE", "FINISHED", "ERROR"
    payload: dict[str, Any]
    timestamp: float

