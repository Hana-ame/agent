import logging
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, Optional

logger = logging.getLogger("vertex_edge_agent.agents")


class BaseAgent(ABC):
    """Abstract base class for all agent engines in the framework."""

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

