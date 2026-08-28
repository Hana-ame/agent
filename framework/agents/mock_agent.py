import asyncio
import json
import logging
from typing import Any, Dict, Optional, Callable

logger = logging.getLogger("vertex_edge_agent.agents")
from .base_agent import BaseAgent

class MockAgent(BaseAgent):
    def __init__(self, response_fn: Optional[Callable] = None):
        self._response_fn = response_fn

    async def process(
        self,
        data: Any,
        prompt: str,
        model: str,
        settings: Optional[Dict] = None,
    ) -> Any:
        logger.info("[MockAgent] model=%s", model)
        logger.debug("[MockAgent] data=%s", repr(data)[:200])
        logger.debug("[MockAgent] prompt=%s", prompt[:200] if prompt else "")

        if self._response_fn:
            result = self._response_fn(data, prompt, model, settings)
            if asyncio.iscoroutine(result):
                result = await result
        else:
            if isinstance(data, str):
                result = f"[{model}] {data}"
            elif isinstance(data, dict):
                result = {
                    "_processed": True,
                    "_model": model,
                    "_prompt": prompt,
                    "input": data,
                    "output": f"Processed: {json.dumps(data, default=str)}",
                }
            else:
                result = f"[{model}] {repr(data)}"

        logger.debug("[MockAgent] result=%s", repr(result)[:200])
        return result

