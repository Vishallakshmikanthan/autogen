"""Experiment, run, and research notes persistence repository (M19, SYS-18)."""

from __future__ import annotations

import json
import uuid
from typing import Any

from apps.api.persistence.repositories.base import BaseSqlRepository


class ExperimentRepository(BaseSqlRepository):
    """Repository handling CRUD operations for M18/M19 experiments and run records."""

    def create_experiment(
        self,
        name: str,
        config_hash: str,
        config: dict[str, Any],
        exp_id: str | None = None,
    ) -> dict[str, Any]:
        eid = exp_id or f"exp_{uuid.uuid4().hex[:12]}"
        cfg_json = json.dumps(config, sort_keys=True)
        self.conn.execute(
            "INSERT INTO experiments (id, name, config_hash, config_json) "
            "VALUES (?, ?, ?, ?);",
            (eid, name, config_hash, cfg_json),
        )
        return {
            "id": eid,
            "name": name,
            "config_hash": config_hash,
            "config": config,
        }

    def get_experiment(self, exp_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT id, name, config_hash, config_json, created_at FROM experiments WHERE id = ?;",
            (exp_id,),
        ).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["config"] = json.loads(data["config_json"])
        del data["config_json"]
        return data

    def record_run(
        self,
        exp_id: str,
        seed: int,
        status: str,
        timing_seconds: float,
        metrics: dict[str, float],
        run_id: str | None = None,
    ) -> dict[str, Any]:
        rid = run_id or f"run_{uuid.uuid4().hex[:12]}"
        metrics_json = json.dumps(metrics)
        self.conn.execute(
            "INSERT INTO experiment_runs (id, exp_id, seed, status, timing_seconds, metrics_json) "
            "VALUES (?, ?, ?, ?, ?, ?);",
            (rid, exp_id, seed, status, timing_seconds, metrics_json),
        )
        return {
            "id": rid,
            "exp_id": exp_id,
            "seed": seed,
            "status": status,
            "timing_seconds": timing_seconds,
            "metrics": metrics,
        }

    def list_runs(self, exp_id: str) -> list[dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT id, exp_id, seed, status, timing_seconds, metrics_json, created_at "
            "FROM experiment_runs WHERE exp_id = ? ORDER BY seed ASC;",
            (exp_id,),
        )
        results = []
        for r in cursor.fetchall():
            item = dict(r)
            item["metrics"] = json.loads(item["metrics_json"])
            del item["metrics_json"]
            results.append(item)
        return results

