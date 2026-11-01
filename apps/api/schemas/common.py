"""Common API schemas, response envelopes, and error DTOs (M19, SYS-18)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class APIResponseEnvelope[T](BaseModel):
    """Standardized API response envelope for all REST endpoints."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    success: bool
    data: T
    message: str = "OK"


class ErrorDetail(BaseModel):
    """Structured error payload for failed requests."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    error_code: str
    message: str
    details: dict[str, str] | None = None
