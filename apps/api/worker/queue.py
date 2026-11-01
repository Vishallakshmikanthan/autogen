"""Job queue implementations for decoupling HTTP handlers from simulation workers (M19, SYS-18)."""

from __future__ import annotations

import multiprocessing as mp
import queue
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class IJobQueue(Protocol):
    """Protocol for command and event queues."""

    def put(self, item: Any, block: bool = True, timeout: float | None = None) -> None: ...
    def get(self, block: bool = True, timeout: float | None = None) -> Any: ...
    def empty(self) -> bool: ...
    def qsize(self) -> int: ...


class InMemoryJobQueue:
    """Thread-safe in-memory FIFO queue for testing and local single-process worker mode."""

    def __init__(self, maxsize: int = 1000) -> None:
        self._q: queue.Queue[Any] = queue.Queue(maxsize=maxsize)

    def put(self, item: Any, block: bool = True, timeout: float | None = None) -> None:
        self._q.put(item, block=block, timeout=timeout)

    def get(self, block: bool = True, timeout: float | None = None) -> Any:
        return self._q.get(block=block, timeout=timeout)

    def empty(self) -> bool:
        return self._q.empty()

    def qsize(self) -> int:
        return self._q.qsize()


class MultiprocessingJobQueue:
    """Inter-process FIFO queue wrapping multiprocessing.Queue for decoupled worker execution."""

    def __init__(self, maxsize: int = 1000) -> None:
        self._q: mp.Queue[Any] = mp.Queue(maxsize=maxsize)

    def put(self, item: Any, block: bool = True, timeout: float | None = None) -> None:
        self._q.put(item, block=block, timeout=timeout)

    def get(self, block: bool = True, timeout: float | None = None) -> Any:
        return self._q.get(block=block, timeout=timeout)

    def empty(self) -> bool:
        return self._q.empty()

    def qsize(self) -> int:
        return self._q.qsize()

