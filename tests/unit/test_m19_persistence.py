"""Unit tests for SQLite persistence, WAL configuration, and repositories (M19, SYS-18)."""

from pathlib import Path

from apps.api.persistence.database import DatabaseManager
from apps.api.persistence.repositories.agent_repo import AgentRepository
from apps.api.persistence.repositories.environment_repo import EnvironmentRepository
from apps.api.persistence.repositories.experiment_repo import ExperimentRepository
from apps.api.persistence.repositories.simulation_repo import SimulationRepository


def test_database_manager_initialization_and_wal(tmp_path: Path):
    db_file = tmp_path / "test.db"
    mgr = DatabaseManager(db_path=db_file)

    with mgr.connection() as conn:
        row = conn.execute("PRAGMA journal_mode;").fetchone()
        assert row[0].lower() == "wal"


def test_environment_repository_crud():
    mgr = DatabaseManager(db_path=":memory:")
    with mgr.connection() as conn:
        repo = EnvironmentRepository(conn)
        env = repo.create_environment(
            name="test_grid",
            width=25,
            height=25,
            config={"obstacle_density": 0.10, "energy_max": 100.0},
        )
        assert env["id"].startswith("env_")
        assert env["version"] == 1

        fetched = repo.get_environment(env["id"])
        assert fetched is not None
        assert fetched["name"] == "test_grid"
        assert fetched["config"]["obstacle_density"] == 0.10

        all_envs = repo.list_environments()
        assert len(all_envs) == 1


def test_agent_repository_crud():
    mgr = DatabaseManager(db_path=":memory:")
    with mgr.connection() as conn:
        repo = AgentRepository(conn)
        ag = repo.create_agent(
            name="test_reflex_agent",
            agent_type="simple_reflex",
            policy_params={"sensory_range": 3},
        )
        assert ag["id"].startswith("ag_")
        assert ag["policy_params"]["sensory_range"] == 3

        fetched = repo.get_agent(ag["id"])
        assert fetched is not None
        assert fetched["agent_type"] == "simple_reflex"


def test_simulation_repository_snapshots_and_events():
    mgr = DatabaseManager(db_path=":memory:")
    with mgr.connection() as conn:
        repo = SimulationRepository(conn)
        sim = repo.create_simulation(name="sim_alpha", seed=42, max_steps=100)
        assert sim["status"] == "CREATED"

        # Update status
        repo.update_simulation_status(sim["id"], status="RUNNING", current_step=10, state_hash="hash_10")
        updated = repo.get_simulation(sim["id"])
        assert updated is not None
        assert updated["status"] == "RUNNING"
        assert updated["current_step"] == 10

        # Snapshot
        repo.save_snapshot(
            sim_id=sim["id"],
            step=10,
            state_hash="hash_10",
            snapshot_dict_or_json={"step": 10, "agents": []},
        )
        snap = repo.get_latest_snapshot(sim["id"])
        assert snap is not None
        assert snap["step"] == 10

        # Event
        repo.record_event(sim["id"], step=10, event_type="COLLECT", payload={"food": 5.0})
        events = repo.get_events(sim["id"])
        assert len(events) == 1
        assert events[0]["event_type"] == "COLLECT"


def test_experiment_repository_runs():
    mgr = DatabaseManager(db_path=":memory:")
    with mgr.connection() as conn:
        repo = ExperimentRepository(conn)
        exp = repo.create_experiment(
            name="benchmark_exp",
            config_hash="abc123hash",
            config={"episodes": 5, "seed": 42},
        )

        repo.record_run(
            exp_id=exp["id"],
            seed=42,
            status="COMPLETED",
            timing_seconds=0.45,
            metrics={"survival_rate": 1.0, "return": 85.0},
        )

        runs = repo.list_runs(exp["id"])
        assert len(runs) == 1
        assert runs[0]["metrics"]["survival_rate"] == 1.0

