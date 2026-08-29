"""Agent for the OpenCode Zen gateway.

OpenCode Zen (``https://opencode.ai/zen/v1``) exposes an OpenAI-compatible
``/chat/completions`` endpoint. The free-tier model ``hy3-free`` needs no
API key — the default token is the literal string ``"public"``. Other,
key-gated models are deliberately NOT catalogued: only ``*-free`` (key-less)
model ids are listed below.

Self-throttling
---------------

Zen rate-limits aggressively, and the default tenacity retry loop only reacts
*after* a 429 — which burns wall-clock time in exponential backoff. This agent
instead limits itself **up front**, which is what makes free-tier graphs
actually work:

* ``max_concurrency`` (default 3) — an ``asyncio.Semaphore`` bounding how many
  Zen calls are in flight at once. A graph with 32 concurrent edges queues
  locally instead of opening 32 simultaneous connections.
* ``requests_per_minute`` (default 20) — a token bucket budgeting attempts per
  minute. Charged per *attempt*, so retries count against the budget rather
  than re-entering an already exhausted endpoint.
* ``queue_timeout`` (default 60 s) — fail fast with :class:`ThrottleTimeoutError`
  instead of hanging the graph forever when the budget cannot be satisfied.

Both gates are agent-local, so each edge's ``settings`` still decides its own
prompt/model — only *when* it gets to speak is coordinated.
"""

import logging
from typing import Dict, Optional

from ._http_base import _HTTPAgentBase

logger = logging.getLogger("vertex_edge_agent.agents")

DEFAULT_ZEN_BASE_URL = "https://opencode.ai/zen/v1"
DEFAULT_ZEN_MODEL = "hy3-free"

#: Free (key-less) models served by OpenCode Zen — only ``*-free`` ids.
#: Informational — used for validation warnings, never enforced.
#: Key-gated models are intentionally absent; the free tier is ``hy3-free``.
KNOWN_ZEN_MODELS: Dict[str, str] = {
    "hy3-free": "DeepSeek",
}


class OpenCodeAgent(_HTTPAgentBase):
    """OpenCode Zen agent — free-tier LLM calls with no API key, self-limited.

    Parameters
    ----------
    base_url:
        Zen endpoint root. Override to hit a self-hosted or regional Zen.
    api_key:
        Defaults to ``"public"``. Pass a real token for authed Zen access.
    default_model:
        Model id used when the graph says ``"default"``.
    max_concurrency:
        Max simultaneous in-flight Zen calls.
    requests_per_minute:
        Max attempts per rolling minute; ``None`` disables the budget.
    queue_timeout:
        Max seconds to wait for a concurrency slot / budget token before
        raising :class:`ThrottleTimeoutError`. ``None`` = wait forever.
    max_retries:
        Total attempts for 429 / 5xx / network failures.
    timeout:
        Overall request timeout in seconds.
    extra_headers:
        Additional request headers.
    proxy:
        Transport-level HTTP(S) proxy the request tunnels through (corporate
        egress / SOCKS / authenticated proxy). Independent of ``trust_env``.
    trust_env:
        Honour ``HTTP_PROXY`` / ``HTTPS_PROXY`` from the environment.
    """

    NAME = "OpenCodeAgent"
    DEFAULT_MODEL = DEFAULT_ZEN_MODEL

    def __init__(
        self,
        base_url: str = DEFAULT_ZEN_BASE_URL,
        api_key: str = "public",
        default_model: str = DEFAULT_ZEN_MODEL,
        max_concurrency: int = 3,
        requests_per_minute: Optional[float] = 20.0,
        queue_timeout: Optional[float] = 60.0,
        max_retries: int = 3,
        timeout: float = 300.0,
        extra_headers: Optional[Dict[str, str]] = None,
        proxy: Optional[str] = None,
        trust_env: bool = True,
    ) -> None:
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            max_retries=max_retries,
            timeout=timeout,
            extra_headers=extra_headers,
            default_model=default_model,
            proxy=proxy,
            trust_env=trust_env,
        )
        self._setup_throttling(max_concurrency, requests_per_minute, queue_timeout)

    # ------------------------------------------------------------------
    # Catalog
    # ------------------------------------------------------------------
    @classmethod
    def available_models(cls) -> Dict[str, str]:
        """Known free Zen model ids mapped to their upstream vendor."""
        return dict(KNOWN_ZEN_MODELS)

    @classmethod
    def is_known_model(cls, model: str) -> bool:
        """Whether ``model`` is in the known free-Zen catalog."""
        return model in KNOWN_ZEN_MODELS

    # ------------------------------------------------------------------
    # Specialisation
    # ------------------------------------------------------------------
    def resolve_model(self, model: str, settings: Optional[Dict] = None) -> str:
        """Resolve the model id, warning when it is not in the free catalog.

        The warning exists because an unknown id surfaces only as a fatal
        HTTP 404 from Zen — logging it at request time points at the cause
        a second earlier than the eventual failure.
        """
        target = super().resolve_model(model, settings)
        if not self.is_known_model(target):
            logger.warning(
                "[%s] model %r is not in the known OpenCode Zen catalog %s — "
                "the gateway may reject it with HTTP 404.",
                self.NAME, target, sorted(KNOWN_ZEN_MODELS),
            )
        return target
