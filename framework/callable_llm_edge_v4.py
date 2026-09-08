"""V4 Callable Protocol LLM Edge — thin subclass wrapper.

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


class CallableLLMEdgeV4(LLMEdgeV4):
    """LLMEdgeV4 with agent_mode='callable'."""

    def __init__(self, *args, **kwargs):
        kwargs["agent_mode"] = "callable"
        super().__init__(*args, **kwargs)


if __name__ == "__main__":
    from framework.edges.cli import main
    main()