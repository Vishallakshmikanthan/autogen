"""Routers package exports (M19, SYS-18)."""

from apps.api.routers.agents import router as agents_router
from apps.api.routers.environments import router as environments_router
from apps.api.routers.experiments import router as experiments_router
from apps.api.routers.simulations import router as simulations_router
from apps.api.routers.stream import router as stream_router

__all__ = [
    "agents_router",
    "environments_router",
    "experiments_router",
    "simulations_router",
    "stream_router",
]

