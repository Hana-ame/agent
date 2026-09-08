"""V4 Process Protocol LLM Edge.

Subclass of LLMEdgeV4 dedicated to agent process(raw_input, prompt, ...) invocations.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Bootstrap repository root into sys.path to enable execution from any working directory
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from framework.edge_v4 import LLMEdgeV4
from framework.vertex_v4 import VertexRecordV4


class ProcessLLMEdgeV4(LLMEdgeV4):
    """Executes LLM inference via agent.process(input_content, rendered_prompt, model=..., settings=...)."""

    def __init__(
        self,
        edge_id: str,
        input_vertex: str,
        output_vertex: str,
        model: str = "sensenova-6.8-flash-lite",
        prompt_template: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
        concurrency_limit: Optional[int] = None,
        concurrency_group: Optional[str] = "llm",
        priority: int = 0,
        timeout: Optional[float] = None,
    ):
        super().__init__(
            edge_id=edge_id,
            input_vertex=input_vertex,
            output_vertex=output_vertex,
            model=model,
            prompt_template=prompt_template,
            settings=settings,
            concurrency_limit=concurrency_limit,
            concurrency_group=concurrency_group,
            priority=priority,
            timeout=timeout,
        )
        self.type = "llm_process"

    async def _invoke_agent(
        self,
        agent: Any,
        rendered_prompt: str,
        in_v: VertexRecordV4,
        out_v: VertexRecordV4,
    ) -> Any:
        if not (hasattr(agent, "process") and callable(agent.process)):
            raise TypeError(f"Agent {type(agent).__name__} does not implement callable 'process'")
        res = agent.process(
            in_v.content,
            rendered_prompt,
            model=self.model,
            settings=self.settings,
        )
        if asyncio.iscoroutine(res):
            return await res
        return res


if __name__ == "__main__":
    from framework.edge_v4 import main
    main()

