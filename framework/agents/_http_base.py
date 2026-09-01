"""Shared plumbing for OpenAI-compatible (``/chat/completions``) LLM agents.

Every HTTP agent in this package differs from every other in a few ways:

* **where the request goes** — base URL, auth, extra headers, proxying
* **what model id it sends** — default model, alias mapping, catalog
* **how fast it may talk** — concurrency ceiling, per-minute budget

Everything else — client lifecycle, retry/backoff, payload assembly, SSE
streaming, error taxonomy — is byte-for-byte identical, so it is implemented
once here and subclassed by ``HttpLLMAgent``, ``OpenCodeAgent`` and

"""

import asyncio
import json
import logging
import time
from contextlib import nullcontext
from typing import Any, AsyncContextManager, AsyncGenerator, Dict, List, Optional

from .base_agent import BaseAgent
from ..utils.errors import ComputeError

logger = logging.getLogger("vertex_edge_agent.agents")

#: HTTP statuses worth retrying. Anything else >= 400 is a client error that
#: will not heal by trying again — fail fast instead of burning credits.
RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})

_CHAT_COMPLETIONS = "/chat/completions"


class NonRetryableHTTPError(Exception):
    """Raised for HTTP errors that should NOT be retried (400, 401, 403, 404, ...)."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}: {message}")


class MalformedResponseError(Exception):
    """Raised when an HTTP 200 body lacks the expected ``choices`` shape.

    Free-tier upstreams occasionally return an error envelope (``{"error": ...}``)
    or an empty body with status 200. Treating that as fatal (KeyError crash)
    killed whole report pipelines; it should be retried like a transient 5xx.
    """


class ThrottleTimeoutError(ComputeError):
    """Raised when an agent cannot get its own concurrency slot / rate budget.

    An agent that self-limits must also be able to *refuse* to wait forever —
    otherwise a saturated endpoint silently hangs the whole graph. A subclass
    of :class:`ComputeError` so edges treat it like any other failed
    computation: FAILED signal, telemetry, and the edge ``retry_policy``.
    """

    def __init__(self, waited: float, kind: str):
        self.waited = waited
        self.kind = kind
        super().__init__(f"{kind} not available after {waited:.2f}s")


class _TokenBucket:
    """Async token bucket: ``rate`` units per ``period`` seconds.

    Refills continuously, burstable up to ``capacity``. Deliberately tiny and
    dependency-free — no third-party rate limiter needed for one shared budget.
    """

    def __init__(self, rate: float, period: float = 60.0, capacity: Optional[float] = None) -> None:
        if rate <= 0 or period <= 0:
            raise ValueError(f"rate and period must be positive, got rate={rate} period={period}")
        self._rate_per_period = float(rate)
        self._period = float(period)
        self._capacity = float(capacity) if capacity is not None else max(1.0, self._rate_per_period)
        self._tokens = self._capacity
        self._last_refill = 0.0
        self._lock = asyncio.Lock()

    def _refill(self, now: float) -> None:
        if now > self._last_refill:
            earned = (now - self._last_refill) * self._rate_per_period / self._period
            self._tokens = min(self._capacity, self._tokens + earned)
            self._last_refill = now

    async def acquire(self, timeout: Optional[float] = None) -> float:
        """Consume one token, waiting for a refill if needed. Returns seconds waited.

        The deadline is checked *after* each sleep rather than by cancelling a
        pending wait, so an exhausted bucket degrades to a clean
        :class:`ThrottleTimeoutError` instead of losing a token mid-grant.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        started = time.monotonic()

        while True:
            await self._lock.acquire()
            try:
                self._refill(time.monotonic())
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return time.monotonic() - started
                deficit = (1.0 - self._tokens) * self._period / self._rate_per_period
            finally:
                self._lock.release()

            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ThrottleTimeoutError(time.monotonic() - started, "rate budget")
                deficit = min(deficit, remaining)
            await asyncio.sleep(deficit)


class _SemaphoreGate:
    """``asyncio.Semaphore`` with a bounded wait.

    Uses ``asyncio.wait_for`` so a saturated gate *refuses* instead of hanging
    the graph. There is a narrow asyncio race where ``wait_for`` can time out
    in the same tick as a ``release()`` and leak one slot — the cost is bounded
    throughput degradation, never a deadlock, which is the right trade for a
    fail-fast concurrency guard.
    """

    def __init__(self, semaphore: asyncio.Semaphore, timeout: Optional[float], kind: str) -> None:
        self._semaphore = semaphore
        self._timeout = timeout
        self._kind = kind

    async def __aenter__(self) -> None:
        started = time.monotonic()
        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=self._timeout)
        except asyncio.TimeoutError:
            raise ThrottleTimeoutError(time.monotonic() - started, self._kind) from None
        waited = time.monotonic() - started
        if waited > 0.5:
            logger.debug("waited %.2fs for %s", waited, self._kind)
        return None

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        self._semaphore.release()
        return False


class _Throttling:
    """Mixin giving an agent its own concurrency ceiling and per-minute budget.

    Two independent gates, because they answer different questions:

    * **concurrency** — *how many* calls may be in flight at once (per call)
    * **budget**      — *how many* attempts may be sent per minute (per attempt)

    Applied to the caller, not the endpoint, so a graph with 32 concurrent
    edges degrades gracefully instead of turning into 32 simultaneous 429s.
    Call :meth:`_setup_throttling` from the subclass constructor.
    """

    def _setup_throttling(
        self,
        max_concurrency: int,
        requests_per_minute: Optional[float],
        queue_timeout: Optional[float],
    ) -> None:
        if max_concurrency < 1:
            raise ValueError(f"max_concurrency must be >= 1, got {max_concurrency}")
        if requests_per_minute is not None and requests_per_minute <= 0:
            raise ValueError(f"requests_per_minute must be > 0 or None, got {requests_per_minute}")
        if queue_timeout is not None and queue_timeout <= 0:
            raise ValueError(f"queue_timeout must be > 0 or None, got {queue_timeout}")

        self.max_concurrency = max_concurrency
        self.requests_per_minute = requests_per_minute
        self.queue_timeout = queue_timeout
        self._in_flight_gate = asyncio.Semaphore(max_concurrency)
        self._rate_budget = (
            _TokenBucket(requests_per_minute, period=60.0) if requests_per_minute else None
        )

    def _concurrency_gate(self) -> AsyncContextManager[None]:
        """Gate around one call. Default: unlimited."""
        if not getattr(self, "max_concurrency", None):
            return nullcontext()
        return _SemaphoreGate(self._in_flight_gate, self.queue_timeout, "concurrency slot")

    async def _acquire_budget(self) -> None:
        """Wait for a rate-budget token before each attempt. Default: instant."""
        if getattr(self, "_rate_budget", None) is not None:
            await self._rate_budget.acquire(timeout=self.queue_timeout)


class _HTTPAgentBase(_Throttling, BaseAgent):
    """Base class for agents speaking the OpenAI chat-completions wire protocol."""

    #: Tag used in log lines, e.g. ``"OpenCodeAgent"``.
    NAME = "HTTP"
    #: Model id sent when the caller passes an empty / ``"default"`` model.
    DEFAULT_MODEL = "default"

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        timeout: float = 300.0,
        extra_headers: Optional[Dict[str, str]] = None,
        default_model: Optional[str] = None,
        proxy: Optional[str] = None,
        trust_env: bool = True,
        mock: bool = False,
        mock_handler = None,
    ) -> None:
        super().__init__(mock=mock, mock_handler=mock_handler)
        import httpx

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.max_retries = max_retries
        self.timeout = timeout
        self.trust_env = trust_env
        self.proxy = proxy
        self.default_model = default_model or self.DEFAULT_MODEL

        self.headers = {
            "Content-Type": "application/json",
            **(extra_headers or {}),
        }
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"


        # Transport proxy: ``proxy`` is the explicit HTTP(S) proxy the request
        # *tunnels through* (corporate egress / SOCKS / authenticated proxy).
        # ``trust_env`` is the fallback that lets httpx read HTTP_PROXY /
        # HTTPS_PROXY from the environment when ``proxy`` is unset. They are
        # independent layers and stack with the gateway URL above.
        self.client = httpx.AsyncClient(
            proxy=proxy,
            timeout=httpx.Timeout(timeout, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=50, max_connections=100),
            trust_env=trust_env,
        )
        self._client_kwargs = dict(
            timeout=httpx.Timeout(timeout, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=50, max_connections=100),
            trust_env=trust_env,
        )
        # Client cache keyed by ``settings["proxy"]`` so each edge/step can
        # pin its own explicit proxy without touching the default client.
        self._proxied_clients: dict = {}
        self._closed = False
        # Per-call transport proxy override: edge ``settings`` may carry
        # ``"proxy": "http://..."`` to pin a call to a specific egress proxy.
        # ``_default_proxy`` is the constructor one we fall back to when
        # settings don't specify any; the lock guards lazy client rebuilds.
        self._default_proxy = proxy
        self._client_lock = asyncio.Lock()
        # Real per-request token usage from upstream responses (``usage`` field).
        self.usage_log: list = []
        logger.debug(
            "[%s] ready base_url=%s default_model=%s proxy=%s trust_env=%s",
            self.NAME, self.base_url, self.default_model, proxy, trust_env,
        )

    # ------------------------------------------------------------------
    # Model / payload resolution — override these to specialise
    # ------------------------------------------------------------------
    def resolve_model(self, model: str, settings: Optional[Dict] = None) -> str:
        """Map a graph-level model name to the upstream model id.

        An empty / ``"default"`` name falls back to :attr:`default_model`.
        Subclasses may remap aliases or validate against a catalog.
        """
        return model if model and model != "default" else self.default_model

    def build_messages(self, data: Any, prompt: str) -> List[Dict[str, str]]:
        """Build the ``[system, user]`` message pair.

        Structured ``data`` (dict / list) is JSON-encoded so the LLM sees a
        stable shape instead of Python ``repr`` noise.
        """
        user = json.dumps(data, ensure_ascii=False) if isinstance(data, (dict, list)) else str(data)
        messages: List[Dict[str, str]] = []
        if prompt:
            messages.append({"role": "system", "content": str(prompt)})
        messages.append({"role": "user", "content": user})
        return messages

    def build_payload(
        self,
        data: Any,
        prompt: str,
        model: str,
        settings: Optional[Dict] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """Assemble the chat-completions request body."""
        settings = settings or {}
        return {
            "model": self.resolve_model(model, settings),
            "messages": self.build_messages(data, prompt),
            **settings.get("llm_kwargs", {}),
            **({"stream": True} if stream else {}),
        }

    def _raise_for_status(self, response: Any) -> None:
        """Split transient failures (retry me) from fatal client errors (stop)."""
        if response.status_code < 400:
            return
        if response.status_code in RETRYABLE_STATUS:
            # Transient — let tenacity retry these.
            response.raise_for_status()
        else:
            # Fatal client error (400, 401, 403, 404, ...) — fail immediately.
            raise NonRetryableHTTPError(response.status_code, response.text)

    # ------------------------------------------------------------------
    # Request execution
    # ------------------------------------------------------------------
    def _client_for(self, settings: Optional[Dict] = None) -> "httpx.AsyncClient":
        """Return the HTTP client matching ``settings["proxy"]``.

        When settings explicitly declare a proxy (e.g. ``http://127.0.0.1:7890``)
        it is honored verbatim via a cached per-proxy client; otherwise the
        default client (constructor ``proxy``) is used. No fallback guessing.
        """
        proxy = (settings or {}).get("proxy") or self.proxy
        if not proxy or proxy == self.proxy:
            return self.client
        client = self._proxied_clients.get(proxy)
        if client is None:
            import httpx

            client = httpx.AsyncClient(proxy=proxy, **self._client_kwargs)
            self._proxied_clients[proxy] = client
        return client

    def _endpoint_url(self, settings: Optional[Dict] = None) -> str:
        """Resolve the complete chat-completions URL.

        ``settings["base_url"]`` (or ``settings["endpoint"]``) is honored
        verbatim when present. It should be the full endpoint URL.
        """
        base = None
        if settings:
            base = settings.get("base_url") or settings.get("endpoint")
        return (base or self.base_url).rstrip("/")

    def _build_client(self, proxy: Optional[str]) -> None:
        """(Re)create the underlying httpx client, routing through ``proxy``."""
        import httpx

        self.proxy = proxy
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=50, max_connections=100),
            proxy=proxy,
            trust_env=self.trust_env,
        )
        self._closed = False

    async def _apply_settings_proxy(self, settings: Optional[Dict]) -> None:
        """Honour a per-call transport proxy from ``settings``.

        When ``settings`` carries ``"proxy": "http://..."``, this call
        tunnels through that proxy instead of the agent's configured one;
        without it the agent falls back to its constructor proxy. The httpx
        client is rebuilt lazily, only when the effective proxy changes.
        """
        target = (settings or {}).get("proxy") or self._default_proxy
        if target != self.proxy:
            async with self._client_lock:
                if target != self.proxy:
                    self._build_client(target)

    async def _post(self, payload: Dict[str, Any], url: Optional[str] = None, settings: Optional[Dict] = None) -> Dict[str, Any]:
        """One POST + error split, gated by the per-attempt rate budget."""
        await self._acquire_budget()
        client = self._client_for(settings)
        response = await client.post(
            url or self._endpoint_url(settings),
            json=payload,
            headers=self.headers,
        )
        self._raise_for_status(response)
        return response.json()

    async def _request_with_retry(self, payload: Dict[str, Any], settings: Optional[Dict] = None) -> Any:
        """Retry a POST on transient failures, inside the concurrency gate."""
        import httpx
        from tenacity import (
            before_sleep_log,
            retry,
            retry_if_exception_type,
            stop_after_attempt,
            wait_exponential,
        )

        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type(
                (httpx.RequestError, httpx.HTTPStatusError, MalformedResponseError)
            ),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        async def _make_request():
            return await self._post(payload, url=self._endpoint_url(settings), settings=settings)

        response = await _make_request()
        usage = response.get("usage") or {}
        if usage:
            details = usage.get("completion_tokens_details") or {}
            self.usage_log.append({
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "reasoning_tokens": details.get("reasoning_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            })
        try:
            choices = response["choices"]
            content = choices[0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            # 免费档偶发 200 但 body 是错误包/空 choices:当作瞬时错误重试,
            # 而不是让整条报告管线因 KeyError 崩溃(2026-08-30 hn 实测踩到)。
            provider = (response or {}).get("model") or (response or {}).get("error")
            raise MalformedResponseError(
                f"upstream returned 200 without choices (model/err={provider!r}); raw={str(response)[:200]}"
            ) from None
        return content

    def get_usage_summary(self) -> dict:
        """Aggregate real token usage recorded from upstream responses.

        ``completion_tokens`` is the model's total generated output; most of it
        is often ``reasoning_tokens`` (internal chain-of-thought) which is NOT
        part of the visible answer. ``visible_tokens`` is the text actually
        returned to the graph.
        """
        prompt = sum(u["prompt_tokens"] for u in self.usage_log)
        completion = sum(u["completion_tokens"] for u in self.usage_log)
        reasoning = sum(u.get("reasoning_tokens", 0) for u in self.usage_log)
        return {
            "calls": len(self.usage_log),
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "reasoning_tokens": reasoning,
            "visible_tokens": max(0, completion - reasoning),
            "total_tokens": prompt + completion,
        }

    # ------------------------------------------------------------------
    # BaseAgent API
    # ------------------------------------------------------------------
    async def process(
        self,
        data: Any,
        prompt: str,
        model: str,
        settings: Optional[Dict] = None,
    ) -> Any:
        if self.mock:
            return await self._mock_process(data, prompt, model, settings)
        await self._apply_settings_proxy(settings)
        payload = self.build_payload(data, prompt, model, settings)
        logger.debug("[%s] -> %s model=%s", self.NAME, self._endpoint_url(settings), payload["model"])

        try:
            async with self._concurrency_gate():
                return await self._request_with_retry(payload, settings=settings)
        except ThrottleTimeoutError as exc:
            # Pre-flight refusal — nothing was sent, so "exhausted retries"
            # would be the wrong story here.
            logger.error("[%s] Throttled: %s", self.NAME, exc)
            raise
        except Exception as exc:
            logger.error("[%s] Exhausted retries or fatal error: %s", self.NAME, exc, exc_info=True)
            raise

    async def stream_process(
        self,
        data: Any,
        prompt: str,
        model: str,
        settings: Optional[Dict] = None,
    ) -> AsyncGenerator[str, None]:
        if self.mock:
            res = await self._mock_process(data, prompt, model, settings)
            yield str(res)
            return
        """Yield content deltas from the OpenAI SSE stream.

        Streams are not retried (a partial stream is not replayable); a
        non-2xx handshake raises :class:`NonRetryableHTTPError` instead.
        """
        await self._apply_settings_proxy(settings)
        payload = self.build_payload(data, prompt, model, settings, stream=True)
        try:
            async with self._concurrency_gate():
                client = self._client_for(settings)
                async with client.stream(
                    "POST",
                    self._endpoint_url(settings),
                    json=payload,
                    headers=self.headers,
                ) as response:
                    if response.status_code >= 400:
                        body = (await response.aread()).decode("utf-8", "replace")
                        raise NonRetryableHTTPError(response.status_code, body)

                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue  # keep-alives, comments, other SSE events
                        body = line[5:].strip()
                        if body == "[DONE]":
                            break
                        try:
                            chunk = json.loads(body)
                        except json.JSONDecodeError:
                            continue
                        choices = chunk.get("choices") or []
                        if not choices:
                            continue  # usage-only / ping frames
                        delta = (choices[0].get("delta") or {}).get("content")
                        if delta:
                            yield delta
        except Exception as exc:
            logger.error("[%s] Stream failed: %s", self.NAME, exc, exc_info=True)
            raise

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def __aenter__(self) -> "_HTTPAgentBase":
        """Async context-manager entry: return self as the managed resource."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Async context-manager exit: release all network handles.

        Idempotent — safe to use with ``async with`` even if the client was
        already closed explicitly via :meth:`close`.
        """
        await self.close()
        return False

    async def close(self) -> None:
        """Close the underlying HTTP client and release connections.

        Idempotent: repeated calls are a safe no-op (``_closed`` guards the
        second teardown). Closes the default client plus every cached
        per-proxy client held in ``_proxied_clients``.
        """
        if not self._closed:
            for c in {id(self.client): self.client, **{id(v): v for v in self._proxied_clients.values()}}.values():
                await c.aclose()
            self._closed = True
        self._proxied_clients.clear()
