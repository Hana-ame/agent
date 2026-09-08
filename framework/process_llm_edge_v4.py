"""V4 Process Protocol LLM Edge — thin subclass wrapper.

All logic lives in ``LLMEdgeV4`` (``framework/edges/llm.py``).
This module is kept for backward compatibility.
"""

import sys
from pathlib import Path

# Bootstrap repository root into sys.path to enable execution from any working directory
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from framework.edges.llm import LLMEdgeV4
from framework.edges.registry import register_edge_type


@register_edge_type("llm_process")
class ProcessLLMEdgeV4(LLMEdgeV4):
    """LLMEdgeV4 with agent_mode='process'."""

    def __init__(self, *args, **kwargs):
        kwargs["agent_mode"] = "process"
        kwargs.setdefault("edge_type", "llm_process")
        super().__init__(*args, **kwargs)


if __name__ == "__main__":
    from framework.edges.cli import main
    main()