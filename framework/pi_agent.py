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


class HttpPIAgent(PIAgent):
    """Production-grade HTTP Agent for OpenAI-compatible endpoints.
    
    Includes built-in connection pooling, timeouts, and exponential backoff 
    retries for rate limits (429) and server errors (5xx).
    """

    def __init__(self, api_key: str = "public", base_url: str = "https://opencode.ai/zen/v1", max_retries: int = 3):
        import httpx
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.max_retries = max_retries
        # Connection pooling and timeouts
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=50, max_connections=100)
        )

    async def process(
        self,
        data: Any,
        prompt: str,
        model: str,
        settings: Optional[Dict] = None,
    ) -> Any:
        import httpx
        from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log

        settings = settings or {}
        target_model = model if model and model != "default" else "hy3-free"
        
        payload = {
            "model": target_model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": str(data)}
            ],
            **settings.get("llm_kwargs", {})
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # Define robust retry logic internally
        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True
        )
        async def _make_request():
            logger.info(f"[HttpPIAgent] Requesting {target_model}...")
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers
            )
            # Raise exception for 4xx/5xx (except 400/401/404 which shouldn't be retried blindly, but for simplicity we rely on tenacity config or specific handling)
            if response.status_code in (429, 500, 502, 503, 504):
                response.raise_for_status()
            elif response.status_code >= 400:
                # Fatal client errors (400, 401, 403, 404) should fail fast without retries
                logger.error(f"[HttpPIAgent] Fatal HTTP Error: {response.status_code} - {response.text}")
                response.raise_for_status()
                
            return response.json()

        try:
            response_data = await _make_request()
            return response_data['choices'][0]['message']['content']
        except Exception as exc:
            logger.error(f"[HttpPIAgent] Exhausted retries or fatal error: {exc}", exc_info=True)
            raise

    async def close(self):
        """Cleanup connection pool."""
        await self.client.aclose()
