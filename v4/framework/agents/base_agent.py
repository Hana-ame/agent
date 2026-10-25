import logging
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, Optional, Callable
import asyncio
import json

logger = logging.getLogger("vertex_edge_agent.agents")


class BaseAgent(ABC):
    """Abstract base class for all agent engines in the framework."""
    
    def __init__(self, mock: bool = False, mock_handler: Optional[Callable] = None, **kwargs):
        self.mock = mock
        self.mock_handler = mock_handler
        # Any other kwargs can be absorbed or passed up

    async def _mock_process(self, data: Any, prompt: str, model: str, settings: Optional[Dict] = None) -> Any:
        logger.debug("[MockMode:%s] model=%s", self.__class__.__name__, model)
        if self.mock_handler:
            result = self.mock_handler(data, prompt, model, settings)
            if asyncio.iscoroutine(result):
                return await result
            return result
            
        if isinstance(data, str):
            return f"[{model}] {data}"
        elif isinstance(data, dict):
            return {
                "_processed": True,
                "_model": model,
                "_prompt": prompt,
                "input": data,
                "output": f"Processed: {json.dumps(data, default=str)}",
            }
        return f"[{model}] {repr(data)}"

    @abstractmethod
    async def process(
        self,
        data: Any,
        prompt: str,
        model: str,
        settings: Optional[Dict] = None,
    ) -> Any:
        """Process input data with the specified prompt and model."""
        pass

    async def stream_process(
        self,
        data: Any,
        prompt: str,
        model: str,
        settings: Optional[Dict] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream generated response chunks. Default implementation yields the full process response."""
        result = await self.process(data, prompt, model, settings)
        yield str(result)

    async def close(self) -> None:
        """Release underlying client resources."""
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        return False
