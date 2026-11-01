"""Integration tests for SimulationWorker lifecycle and database sync (M19, SYS-18)."""


from apps.api.persistence.database import DatabaseManager
from apps.api.persistence.repositories.simulation_repo import SimulationRepository
from apps.api.worker.contracts import SimulationCommand, SimulationCommandType
from apps.api.worker.queue import InMemoryJobQueue
from apps.api.worker.runner import SimulationWorker


def test_simulation_worker_stepping_and_snapshots():
    cmd_q = InMemoryJobQueue()
    event_q = InMemoryJobQueue()
    db_mgr = DatabaseManager(db_path=":memory:")

    worker = SimulationWorker(
        cmd_queue=cmd_q,
        event_queue=event_q,
        db_path=":memory:",
        snapshot_cadence=2,
    )
    worker.db_manager = db_mgr

    with db_mgr.connection() as conn:
        repo = SimulationRepository(conn)
        repo.create_simulation(name="worker_test_sim", seed=77, sim_id="sim_worker_1", max_steps=5)

    worker.initialize_simulation(
        sim_id="sim_worker_1",
        width=8,
        height=8,
        seed=77,
        max_steps=5,
    )

    # Initial snapshot at step 0 was saved
    with db_mgr.connection() as conn:
        repo = SimulationRepository(conn)
        snap0 = repo.get_latest_snapshot("sim_worker_1")
        assert snap0 is not None
        assert snap0["step"] == 0

    # Step twice
    worker.step_once()
    worker.step_once()

    assert worker.current_step == 2

    # Snapshot at cadence (step 2) was saved
    with db_mgr.connection() as conn:
        repo = SimulationRepository(conn)
        snap2 = repo.get_latest_snapshot("sim_worker_1")
        assert snap2 is not None
        assert snap2["step"] == 2

    # Reset command
    worker.handle_command(SimulationCommand(sim_id="sim_worker_1", command=SimulationCommandType.RESET))
    assert worker.current_step == 0
    assert worker.status == "PAUSED"
