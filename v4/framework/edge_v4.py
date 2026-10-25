"""V4 Standalone Edge Execution Engine (Façade).

This module re-exports all edge classes from the modular `framework.edges`
package to preserve 100% backward compatibility for all existing imports.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Bootstrap repository root into sys.path
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from framework.edges import (
    AgentProtocol,
    CallableLLMEdgeV4,
    ChatLLMEdgeV4,
    CodeEdgeV4,
    EdgeResultV4,
    EdgeV4,
    GenerateLLMEdgeV4,
    LLMEdgeV4,
    LLMToolEdgeV4,
    MockAgentV4,
    ProcessLLMEdgeV4,
    ReflexiveEdgeV4,
    ToolEdgeV4,
)
from framework.edges.base import _resolve_script_callable

__all__ = [
    "EdgeResultV4",
    "EdgeV4",
    "CodeEdgeV4",
    "ToolEdgeV4",
    "LLMToolEdgeV4",
    "LLMEdgeV4",
    "MockAgentV4",
    "AgentProtocol",
    "ChatLLMEdgeV4",
    "GenerateLLMEdgeV4",
    "ProcessLLMEdgeV4",
    "CallableLLMEdgeV4",
    "ReflexiveEdgeV4",
    "main",
    "parse_args",
]

if __name__ == "__main__":
    from framework.edges.cli import main as _cli_main

    _cli_main()


# ``main`` / ``parse_args`` are re-exported from ``framework.edges.cli`` for
# backwards compatibility. Importing that module eagerly here would drag the CLI
# into every consumer of edge_v4, and then ``python -m framework.edges.cli``
# prints runpy's "already in sys.modules" warning (runpy finds the module in
# sys.modules before it executes it). Resolving on demand keeps both working.
def __getattr__(name):
    if name in ("main", "parse_args"):
        from framework.edges import cli

        return getattr(cli, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
