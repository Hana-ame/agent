"""V4 Distributed Worker Queue Adapter Interface.

Defines an abstract execution adapter protocol for offloading edge runs to
distributed workers (Redis Queue / Celery / HTTP RPC).
"""

from __future__ import annotations

import asyncio
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class EdgeTaskPayload:
    """Payload describing an edge execution task for distributed workers."""
    task_id: str = field(default_factory=lambda: f"task_{uuid.uuid4().hex[:12]}")
    session_id: str = ""
    edge_id: str = ""
    edge_type: str = "code"
    input_vertex: str = ""
    output_vertex: str = ""
    input_content: str = ""
    settings: Dict[str, Any] = field(default_factory=dict)
    script: Optional[str] = None
    model: Optional[str] = None
    timeout: float = 120.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeTaskResult:
    """Result returned from a distributed worker edge execution."""
    task_id: str = ""
    edge_id: str = ""
    success: bool = False
    output: str = ""
    error: Optional[str] = None
    duration_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseWorkerQueueV4(ABC):
    """Abstract base for distributed edge execution backends."""

    @abstractmethod
    async def submit_edge(self, task: EdgeTaskPayload) -> str:
        """Submit an edge execution task and return a task ID."""
        ...

    @abstractmethod
    async def poll_result(self, task_id: str) -> Optional[EdgeTaskResult]:
        """Poll for a task result. Returns None if not yet complete."""
        ...

    @abstractmethod
    async def wait_result(self, task_id: str, timeout: float = 120.0) -> EdgeTaskResult:
        """Wait for a task result with timeout."""
        ...

    @abstractmethod
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task. Returns True if cancellation was successful."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the worker queue backend is healthy/reachable."""
        ...


class InMemoryWorkerQueueV4(BaseWorkerQueueV4):
    """In-memory reference implementation of BaseWorkerQueueV4 for testing."""

    def __init__(self):
        self._tasks: Dict[str, EdgeTaskPayload] = {}
        self._results: Dict[str, EdgeTaskResult] = {}
        self._events: Dict[str, asyncio.Event] = {}
        self._cancelled: set = set()

    async def submit_edge(self, task: EdgeTaskPayload) -> str:
        self._tasks[task.task_id] = task
        self._events[task.task_id] = asyncio.Event()
        return task.task_id

    async def poll_result(self, task_id: str) -> Optional[EdgeTaskResult]:
        return self._results.get(task_id)

    async def wait_result(self, task_id: str, timeout: float = 120.0) -> EdgeTaskResult:
        event = self._events.get(task_id)
        if event is None:
            raise ValueError(f"Unknown task: {task_id}")
        await asyncio.wait_for(event.wait(), timeout=timeout)
        result = self._results.get(task_id)
        if result is None:
            raise ValueError(f"No result for completed task: {task_id}")
        return result

    async def cancel_task(self, task_id: str) -> bool:
        if task_id in self._tasks and task_id not in self._results:
            self._cancelled.add(task_id)
            self._results[task_id] = EdgeTaskResult(
                task_id=task_id,
                edge_id=self._tasks[task_id].edge_id,
                success=False,
                error="Task cancelled",
            )
            if task_id in self._events:
                self._events[task_id].set()
            return True
        return False

    async def health_check(self) -> bool:
        return True

    def complete_task(self, task_id: str, result: EdgeTaskResult) -> None:
        """Test helper: manually complete a task with a result."""
        self._results[task_id] = result
        if task_id in self._events:
            self._events[task_id].set()
