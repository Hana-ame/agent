import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("vertex_edge_agent.agents")
from .base_agent import BaseAgent


class NonRetryableHTTPError(Exception):
    """Raised for HTTP errors that should NOT be retried (400, 401, 403, 404, etc.)."""
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}: {message}")


class HttpLLMAgent(BaseAgent):
    def __init__(self, api_key: str = "public", base_url: str = "https://opencode.ai/zen/v1", max_retries: int = 3):
        import httpx
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.max_retries = max_retries
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=50, max_connections=100)
        )
        self._closed = False

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

        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True
        )
        async def _make_request():
            logger.info(f"[HttpLLMAgent] Requesting {target_model}...")
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers
            )
            if response.status_code in (429, 500, 502, 503, 504):
                # Transient errors — let tenacity retry these
                response.raise_for_status()
            elif response.status_code >= 400:
                # Fatal client errors (400, 401, 403, 404, etc.) — fail immediately
                logger.error(f"[HttpLLMAgent] Fatal HTTP Error: {response.status_code} - {response.text}")
                raise NonRetryableHTTPError(response.status_code, response.text)
                
            return response.json()

        try:
            response_data = await _make_request()
            return response_data['choices'][0]['message']['content']
        except Exception as exc:
            logger.error(f"[HttpLLMAgent] Exhausted retries or fatal error: {exc}", exc_info=True)
            raise

    async def close(self):
        """Close the underlying HTTP client and release connections."""
        if not self._closed:
            await self.client.aclose()
            self._closed = True

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        return False

