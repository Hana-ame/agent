"""Unified Exception Taxonomy for Vertex-Edge Agent Framework."""

from typing import Optional


class FrameworkError(Exception):
    """Base exception for all framework errors."""
    pass


class ExecutionError(FrameworkError):
    """Base exception for runtime workflow execution errors."""
    pass


class GuardAbortError(ExecutionError):
    """Raised when an edge guard condition evaluates to False."""
    def __init__(self, reason: str = "Guard condition not satisfied", edge_id: Optional[str] = None):
        self.reason = reason
        self.edge_id = edge_id
        super().__init__(reason)


class AbortPipeline(GuardAbortError):
    """Backward-compatible alias for GuardAbortError."""
    pass


class HookError(ExecutionError):
    """Raised when a vertex or edge hook fails during execution."""
    pass


class ComputeError(ExecutionError):
    """Raised when an LLM agent computation fails."""
    pass


class SubgraphError(ExecutionError):
    """Raised when an inner nested subgraph execution fails."""
    pass


class DataRejectedError(FrameworkError):
    """Raised when a vertex rejects incoming data via its on_receive script."""
    pass
