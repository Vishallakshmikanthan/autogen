"""Abstract repository protocols and base interfaces (M19, SYS-18)."""

from __future__ import annotations

import sqlite3
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class IRepository(Protocol):
    """Base repository marker protocol."""


class BaseSqlRepository:
    """Base class for SQLite repositories providing helper row-mapping methods."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    @staticmethod
    def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return dict(row)

