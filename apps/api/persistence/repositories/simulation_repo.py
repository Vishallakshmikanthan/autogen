"""Simulation session, snapshot, and event persistence repository (M19, SYS-18)."""

from __future__ import annotations

import json
import uuid
from typing import Any

from apps.api.persistence.repositories.base import BaseSqlRepository


class SimulationRepository(BaseSqlRepository):
    """Repository handling CRUD operations for simulations, periodic snapshots, and events."""

    def create_simulation(
        self,
        name: str,
        seed: int,
        env_version_id: str | None = None,
        max_steps: int = 200,
        sim_id: str | None = None,
    ) -> dict[str, Any]:
        sid = sim_id or f"sim_{uuid.uuid4().hex[:12]}"
        self.conn.execute(
            "INSERT INTO simulations (id, name, env_version_id, seed, status, current_step, max_steps, state_hash) "
            "VALUES (?, ?, ?, ?, 'CREATED', 0, ?, '');",
            (sid, name, env_version_id, seed, max_steps),
        )
        return self.get_simulation(sid) or {}

    def get_simulation(self, sim_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT id, name, env_version_id, seed, status, current_step, max_steps, state_hash, created_at, updated_at "
            "FROM simulations WHERE id = ?;",
            (sim_id,),
        ).fetchone()
        return self.row_to_dict(row)

    def update_simulation_status(
        self,
        sim_id: str,
        status: str,
        current_step: int | None = None,
        state_hash: str | None = None,
    ) -> None:
        query = "UPDATE simulations SET status = ?, updated_at = CURRENT_TIMESTAMP"
        params: list[Any] = [status]
        if current_step is not None:
            query += ", current_step = ?"
            params.append(current_step)
        if state_hash is not None:
            query += ", state_hash = ?"
            params.append(state_hash)
        query += " WHERE id = ?;"
        params.append(sim_id)
        self.conn.execute(query, params)

    def save_snapshot(
        self,
        sim_id: str,
        step: int,
        state_hash: str,
        snapshot_dict_or_json: dict[str, Any] | str,
    ) -> None:
        raw_json = (
            snapshot_dict_or_json
            if isinstance(snapshot_dict_or_json, str)
            else json.dumps(snapshot_dict_or_json)
        )
        self.conn.execute(
            "INSERT OR REPLACE INTO agent_states (sim_id, step, state_hash, snapshot_json) "
            "VALUES (?, ?, ?, ?);",
            (sim_id, step, state_hash, raw_json),
        )

    def get_latest_snapshot(self, sim_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT step, state_hash, snapshot_json, created_at "
            "FROM agent_states WHERE sim_id = ? ORDER BY step DESC LIMIT 1;",
            (sim_id,),
        ).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["snapshot"] = json.loads(data["snapshot_json"])
        del data["snapshot_json"]
        return data

    def record_event(
        self,
        sim_id: str,
        step: int,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        self.conn.execute(
            "INSERT INTO simulation_events (sim_id, step, event_type, payload_json) "
            "VALUES (?, ?, ?, ?);",
            (sim_id, step, event_type, json.dumps(payload)),
        )

    def get_events(self, sim_id: str, limit: int = 100) -> list[dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT id, sim_id, step, event_type, payload_json, created_at "
            "FROM simulation_events WHERE sim_id = ? ORDER BY step ASC, id ASC LIMIT ?;",
            (sim_id, limit),
        )
        results = []
        for r in cursor.fetchall():
            item = dict(r)
            item["payload"] = json.loads(item["payload_json"])
            del item["payload_json"]
            results.append(item)
        return results

    def save_metric(
        self,
        target_id: str,
        metric_name: str,
        metric_value: float,
        step: int = 0,
        target_type: str = "SIMULATION",
    ) -> None:
        self.conn.execute(
            "INSERT INTO metrics (target_id, target_type, metric_name, metric_value, step) "
            "VALUES (?, ?, ?, ?, ?);",
            (target_id, target_type, metric_name, metric_value, step),
        )

    def get_metrics(self, target_id: str) -> list[dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT metric_name, metric_value, step FROM metrics WHERE target_id = ? ORDER BY step ASC;",
            (target_id,),
        )
        return [dict(r) for r in cursor.fetchall()]

