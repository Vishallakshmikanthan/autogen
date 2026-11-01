"""Unit tests for command queue implementations (M19, SYS-18)."""


from apps.api.worker.contracts import SimulationCommand, SimulationCommandType
from apps.api.worker.queue import InMemoryJobQueue


def test_in_memory_job_queue_fifo():
    q = InMemoryJobQueue(maxsize=10)
    assert q.empty() is True
    assert q.qsize() == 0

    cmd1 = SimulationCommand(sim_id="sim_1", command=SimulationCommandType.START)
    cmd2 = SimulationCommand(sim_id="sim_1", command=SimulationCommandType.PAUSE)

    q.put(cmd1)
    q.put(cmd2)

    assert q.qsize() == 2
    assert q.empty() is False

    retrieved1 = q.get(block=False)
    assert retrieved1.command == SimulationCommandType.START

    retrieved2 = q.get(block=False)
    assert retrieved2.command == SimulationCommandType.PAUSE

    assert q.empty() is True
