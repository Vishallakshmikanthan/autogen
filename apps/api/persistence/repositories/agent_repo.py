"""Agent and versioned policy repository (M19, SYS-18)."""

from __future__ import annotations

import json
import uuid
from typing import Any

from apps.api.persistence.repositories.base import BaseSqlRepository


class AgentRepository(BaseSqlRepository):
    """Repository handling CRUD operations for agents and agent policy versions."""

    def create_agent(
        self,
        name: str,
        agent_type: str,
        policy_params: dict[str, Any] | None = None,
        agent_id: str | None = None,
    ) -> dict[str, Any]:
        aid = agent_id or f"ag_{uuid.uuid4().hex[:12]}"
        params = policy_params or {}
        params_json = json.dumps(params, sort_keys=True)

        self.conn.execute(
            "INSERT INTO agents (id, name, agent_type) VALUES (?, ?, ?);",
            (aid, name, agent_type),
        )

        ver_id = f"{aid}_v1"
        self.conn.execute(
            "INSERT INTO agent_versions (id, agent_id, version, policy_params_json) "
            "VALUES (?, ?, 1, ?);",
            (ver_id, aid, params_json),
        )

        return {
            "id": aid,
            "name": name,
            "agent_type": agent_type,
            "version": 1,
            "version_id": ver_id,
            "policy_params": params,
        }

    def get_agent(self, agent_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT a.id, a.name, a.agent_type, av.id as version_id, av.version, av.policy_params_json "
            "FROM agents a "
            "LEFT JOIN agent_versions av ON a.id = av.agent_id "
            "WHERE a.id = ? ORDER BY av.version DESC LIMIT 1;",
            (agent_id,),
        ).fetchone()

        if row is None:
            return None

        data = dict(row)
        if data.get("policy_params_json"):
            data["policy_params"] = json.loads(data["policy_params_json"])
            del data["policy_params_json"]
        return data

    def list_agents(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT a.id, a.name, a.agent_type, av.id as version_id, av.version "
            "FROM agents a "
            "LEFT JOIN agent_versions av ON a.id = av.agent_id "
            "ORDER BY a.created_at DESC LIMIT ? OFFSET ?;",
            (limit, offset),
        )
        return [dict(r) for r in cursor.fetchall()]

