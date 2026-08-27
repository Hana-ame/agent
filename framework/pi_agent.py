"""PI Agent module - Interface for AI / LLM processing.

Provides an abstract base class ``PIAgent`` and two concrete implementations:

* ``MockPIAgent``      – deterministic, for testing
* ``ExternalPIAgent``  – delegates to an installed ``pi_agent`` package
"""

import abc
import json
import logging
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger("vertex_edge_agent.pi_agent")


class PIAgent(abc.ABC):
    """Abstract base class for PI Agent integration."""

    @abc.abstractmethod
    async def process(
        self,
        data: Any,
        prompt: str,
        model: str,
        settings: Optional[Dict] = None,
    ) -> Any:
        """Process *data* through the AI agent.

        Args:
            data:     Input data (string, dict, or any JSON-serialisable value).
            prompt:   The instruction / prompt.
            model:    Model identifier (e.g. ``"gemini-pro"``).
            settings: Extra settings forwarded to the agent.

        Returns:
            Processed result (string or JSON-serialisable value).
        """


class MockPIAgent(PIAgent):
    """Deterministic mock agent for testing.

    By default it echoes data back with model metadata.  Supply a custom
    *response_fn(data, prompt, model, settings) -> result* to override.
    """

    def __init__(self, response_fn: Optional[Callable] = None):
        self._response_fn = response_fn

    async def process(
        self,
        data: Any,
        prompt: str,
        model: str,
        settings: Optional[Dict] = None,
    ) -> Any:
        logger.info("[MockPIAgent] model=%s", model)
        logger.debug("[MockPIAgent] data=%s", repr(data)[:200])
        logger.debug("[MockPIAgent] prompt=%s", prompt[:200] if prompt else "")

        if self._response_fn:
            result = self._response_fn(data, prompt, model, settings)
        else:
            # Default echo with metadata
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

        logger.debug("[MockPIAgent] result=%s", repr(result)[:200])
        return result


class ExternalPIAgent(PIAgent):
    """Delegates to an installed ``pi_agent`` Python package.

    Install via ``pip install pi-agent`` (or equivalent).
    """

    async def process(
        self,
        data: Any,
        prompt: str,
        model: str,
        settings: Optional[Dict] = None,
    ) -> Any:
        logger.info("[ExternalPIAgent] model=%s", model)
        try:
            import pi_agent as pa  # type: ignore[import-untyped]

            result = await pa.run(
                data=data, prompt=prompt, model=model, **(settings or {})
            )
            return result
        except ImportError:
            logger.error(
                "[ExternalPIAgent] 'pi_agent' package not installed. "
                "Use MockPIAgent for testing or install the package."
            )
            raise
