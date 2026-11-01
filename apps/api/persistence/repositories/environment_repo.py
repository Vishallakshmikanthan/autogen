"""Environment and environment-version persistence repository (M19, SYS-18)."""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any

from apps.api.persistence.repositories.base import BaseSqlRepository


class EnvironmentRepository(BaseSqlRepository):
    """Repository handling CRUD operations for environments and versioned configurations."""

    def create_environment(
        self,
        name: str,
        width: int,
        height: int,
        config: dict[str, Any],
        env_id: str | None = None,
    ) -> dict[str, Any]:
        eid = env_id or f"env_{uuid.uuid4().hex[:12]}"
        self.conn.execute(
            "INSERT INTO environments (id, name, width, height) VALUES (?, ?, ?, ?);",
            (eid, name, width, height),
        )

        # Create version 1
        cfg_json = json.dumps(config, sort_keys=True)
        cfg_hash = hashlib.sha256(cfg_json.encode("utf-8")).hexdigest()
        ver_id = f"{eid}_v1"
        self.conn.execute(
            "INSERT INTO environment_versions (id, env_id, version, config_json, config_hash) "
            "VALUES (?, ?, 1, ?, ?);",
            (ver_id, eid, cfg_json, cfg_hash),
        )

        return {
            "id": eid,
            "name": name,
            "width": width,
            "height": height,
            "version": 1,
            "version_id": ver_id,
            "config": config,
            "config_hash": cfg_hash,
        }

    def get_environment(self, env_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT e.id, e.name, e.width, e.height, ev.id as version_id, ev.version, ev.config_json, ev.config_hash "
            "FROM environments e "
            "LEFT JOIN environment_versions ev ON e.id = ev.env_id "
            "WHERE e.id = ? ORDER BY ev.version DESC LIMIT 1;",
            (env_id,),
        ).fetchone()

        if row is None:
            return None

        data = dict(row)
        if data.get("config_json"):
            data["config"] = json.loads(data["config_json"])
            del data["config_json"]
        return data

    def list_environments(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT e.id, e.name, e.width, e.height, ev.id as version_id, ev.version, ev.config_hash "
            "FROM environments e "
            "LEFT JOIN environment_versions ev ON e.id = ev.env_id "
            "ORDER BY e.created_at DESC LIMIT ? OFFSET ?;",
            (limit, offset),
        )
        return [dict(r) for r in cursor.fetchall()]

