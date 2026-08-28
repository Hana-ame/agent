"""Pipeline module — DEPRECATED. Pipeline orchestration logic has moved into Edge.

Background:
Orchestration logic (guard, pre-process, compute, retry/timeout, post-process, schema validation, memory, telemetry)
has now been unified directly within Edge.

This file is preserved solely for backwards compatibility: `from framework.pipeline import Pipeline` continues to work,
where Pipeline is an alias for Edge. New code should use Edge directly.
"""

from .edge import Edge

# Backwards-compatibility alias:
Pipeline = Edge

# Re-export AbortPipeline and other errors for backwards compatibility
from .utils.errors import AbortPipeline, GuardAbortError, HookError, ComputeError

__all__ = ["Pipeline", "AbortPipeline", "GuardAbortError", "HookError", "ComputeError"]
