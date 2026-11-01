"""FastAPI application entrypoint and OpenAPI export generator (M19, SYS-18)."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.api.dependencies import get_worker, shutdown_worker
from apps.api.routers import (
    agents_router,
    environments_router,
    experiments_router,
    simulations_router,
    stream_router,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    """Application lifespan manager starting and stopping the simulation worker."""
    # Start worker on startup
    _ = get_worker()
    yield
    # Clean up worker on shutdown
    shutdown_worker()


def create_app() -> FastAPI:
    """Factory creating and configuring the AUTOGEN Research Platform FastAPI application."""
    app = FastAPI(
        title="AUTOGEN AI — Simulation & Research Studio API",
        description="Backend API and streaming engine for autonomous simulation, experiment suites, and algorithm analytics (SYS-18).",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Enable CORS for frontend Research Studio (M20)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register Routers
    app.include_router(environments_router)
    app.include_router(agents_router)
    app.include_router(simulations_router)
    app.include_router(stream_router)
    app.include_router(experiments_router)

    @app.get("/health", tags=["system"])
    def health_check() -> dict[str, str]:
        return {"status": "HEALTHY", "service": "autogen-api", "version": "0.1.0"}

    return app


app = create_app()


def export_openapi_json(output_path: str | Path = "docs/architecture/openapi.json") -> Path:
    """Exports OpenAPI JSON schema for automated TypeScript client generation (M20 integration hook)."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = app.openapi()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)
    return path

