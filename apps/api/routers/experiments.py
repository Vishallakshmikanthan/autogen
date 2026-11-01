"""Experiments API router (M19, SYS-18)."""

from __future__ import annotations

import sqlite3
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from apps.api.dependencies import get_db
from apps.api.persistence.repositories.experiment_repo import ExperimentRepository
from apps.api.schemas.common import APIResponseEnvelope
from apps.api.schemas.experiment_dtos import (
    CreateExperimentRequest,
    ExperimentResponse,
    ExperimentRunResultResponse,
)
from autogen.experiments.record import compute_config_hash
from autogen.experiments.runner import ExperimentRunner

router = APIRouter(prefix="/api/v1/experiments", tags=["experiments"])


@router.post("", response_model=APIResponseEnvelope[ExperimentResponse], status_code=status.HTTP_201_CREATED)
def create_experiment(
    req: CreateExperimentRequest,
    conn: sqlite3.Connection = Depends(get_db),
) -> Any:
    repo = ExperimentRepository(conn)
    cfg_hash = compute_config_hash(req.config)
    created = repo.create_experiment(
        name=req.name,
        config_hash=cfg_hash,
        config=req.config.model_dump(),
    )
    return APIResponseEnvelope(
        success=True,
        data=ExperimentResponse.model_validate(created),
        message="Experiment registered successfully",
    )


@router.get("/{exp_id}", response_model=APIResponseEnvelope[ExperimentResponse])
def get_experiment(
    exp_id: str,
    conn: sqlite3.Connection = Depends(get_db),
) -> Any:
    repo = ExperimentRepository(conn)
    exp = repo.get_experiment(exp_id)
    if exp is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Experiment '{exp_id}' not found")
    return APIResponseEnvelope(
        success=True,
        data=ExperimentResponse.model_validate(exp),
        message="Experiment found",
    )


@router.post("/{exp_id}/run", response_model=APIResponseEnvelope[dict[str, Any]])
def run_experiment(
    exp_id: str,
    conn: sqlite3.Connection = Depends(get_db),
) -> Any:
    repo = ExperimentRepository(conn)
    exp = repo.get_experiment(exp_id)
    if exp is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Experiment '{exp_id}' not found")

    from autogen.experiments.config import ExperimentConfig

    cfg = ExperimentConfig.model_validate(exp["config"])
    runner = ExperimentRunner(use_parallel=False)
    suite_record = runner.run(cfg)

    # Persist runs
    for run in suite_record.records:
        repo.record_run(
            exp_id=exp_id,
            seed=run.seed,
            status="COMPLETED",
            timing_seconds=run.timing_seconds,
            metrics=run.metrics,
            run_id=run.run_id,
        )

    return APIResponseEnvelope(
        success=True,
        data={
            "suite_id": suite_record.suite_id,
            "total_runs": suite_record.total_runs,
            "config_hash": suite_record.config_hash,
            "aggregated_metrics": {k: v.model_dump() for k, v in suite_record.aggregated_metrics.items()},
        },
        message="Experiment suite executed and stored",
    )


@router.get("/{exp_id}/results", response_model=APIResponseEnvelope[ExperimentRunResultResponse])
def get_experiment_results(
    exp_id: str,
    conn: sqlite3.Connection = Depends(get_db),
) -> Any:
    repo = ExperimentRepository(conn)
    runs = repo.list_runs(exp_id)
    return APIResponseEnvelope(
        success=True,
        data=ExperimentRunResultResponse(exp_id=exp_id, total_runs=len(runs), runs=runs),
        message="Experiment results retrieved",
    )

