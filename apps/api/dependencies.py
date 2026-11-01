"""FastAPI dependency injection providers and state singletons (M19, SYS-18)."""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Generator

from apps.api.persistence.database import DatabaseManager
from apps.api.worker.queue import InMemoryJobQueue
from apps.api.worker.runner import SimulationWorker

_DEFAULT_DB_PATH = os.environ.get("AUTOGEN_DB_PATH", "data/autogen.db")
_db_manager: DatabaseManager | None = None
_cmd_queue = InMemoryJobQueue(maxsize=1000)
_event_queue = InMemoryJobQueue(maxsize=1000)
_worker: SimulationWorker | None = None


def get_db_manager() -> DatabaseManager:
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager(db_path=_DEFAULT_DB_PATH)
    return _db_manager


def set_db_manager(manager: DatabaseManager) -> None:
    global _db_manager, _worker
    _db_manager = manager
    if _worker:
        _worker.db_manager = manager


def get_db() -> Generator[sqlite3.Connection, None, None]:
    mgr = get_db_manager()
    with mgr.connection() as conn:
        yield conn


def get_cmd_queue() -> InMemoryJobQueue:
    return _cmd_queue


def get_event_queue() -> InMemoryJobQueue:
    return _event_queue


def get_worker() -> SimulationWorker:
    global _worker
    if _worker is None:
        _worker = SimulationWorker(
            cmd_queue=_cmd_queue,
            event_queue=_event_queue,
            db_path=_DEFAULT_DB_PATH,
        )
        _worker.db_manager = get_db_manager()
        _worker.start_background()
    return _worker


def shutdown_worker() -> None:
    global _worker
    if _worker is not None:
        _worker.stop_background()
        _worker = None
