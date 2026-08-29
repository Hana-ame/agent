"""LLM agent that routes every call through a self-hosted proxy/gateway.

Talks OpenAI chat-completions to a proxy — LiteLLM, one-api, an internal
gateway, an auth-forwarding sidecar, a quota meter — which in turn calls
the real upstream provider. The graph stays provider-agnostic while you get
one surface for billing, audit logging, auth, quotas and model routing.

Two independent kinds of "proxied":

* **Application proxy** — ``proxy_url`` points at a chat-completions
  gateway (``http://litellm.internal:4000/v1``). This is the main mode.
* **Transport proxy** — ``trust_env=True`` (the default) makes httpx also
  honour ``HTTP_PROXY`` / ``HTTPS_PROXY`` for egress-restricted networks.

Both stack: a gateway sitting behind a corporate egress proxy works as-is.

The agent also self-limits with the same two gates as
:class:`OpenCodeAgent` — a concurrency ceiling (default 5) so a wide graph
cannot open hundreds of connections to the gateway, plus an optional
``requests_per_minute`` budget (disabled by default: the gateway normally
governs its own rate, so the agent only has to stop itself from over-joining).

Configuration is explicit-first and environment-driven as fallback:

* base URL: ``proxy_url`` > ``base_url`` > ``LLM_PROXY_BASE_URL`` >
  ``OPENAI_BASE_URL`` > ``http://localhost:8000/v1``
* api key:  ``api_key`` > ``LLM_PROXY_API_KEY`` > ``OPENAI_API_KEY`` >
  ``"public"``
"""

import logging
import os
from typing import Dict, Optional, Sequence

from ._http_base import _HTTPAgentBase

logger = logging.getLogger("vertex_edge_agent.agents")

ENV_BASE_URL: Sequence[str] = ("LLM_PROXY_BASE_URL", "OPENAI_BASE_URL")
ENV_API_KEY: Sequence[str] = ("LLM_PROXY_API_KEY", "OPENAI_API_KEY")

DEFAULT_PROXY_BASE_URL = "http://localhost:8000/v1"
DEFAULT_PROXY_MODEL = "gpt-4o-mini"


def _first_env(names: Sequence[str]) -> Optional[str]:
    """First non-empty value among ``names``, else ``None``."""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


class ProxiedLLMAgent(_HTTPAgentBase):
    """Route chat-completions through a self-hosted LLM proxy/gateway.

    Parameters
    ----------
    proxy_url / base_url:
        Gateway root; ``"/chat/completions"`` is appended automatically.
        Either keyword works, ``proxy_url`` wins if both are given.
    api_key:
        Gateway token. Fall back to ``LLM_PROXY_API_KEY`` / ``OPENAI_API_KEY``.
    model_map:
        Graph-level alias -> upstream model id (e.g. ``{"alias": "gpt-5.5"}``).
        Applied *after* the ``"default"`` fallback, so the default model
        can itself be aliased.
    default_model:
        Model id used when the graph says ``"default"``.
    max_concurrency:
        Max simultaneous in-flight calls to the gateway (default 5).
    requests_per_minute:
        Max attempts per rolling minute; ``None`` (the default) disables it,
        letting the gateway govern its own rate.
    queue_timeout:
        Max seconds to wait for a concurrency slot before raising
        :class:`ThrottleTimeoutError`. ``None`` = wait forever.
    max_retries:
        Total attempts for 429 / 5xx / network failures.
    timeout:
        Overall request timeout in seconds.
    extra_headers:
        Additional headers — handy for ``Authorization`` overrides or
        gateway-specific routing hints.
    proxy:
        Transport-level HTTP(S) proxy the request *tunnels through* on its way
        to the gateway (corporate egress / SOCKS / authenticated proxy).
        Independent of ``proxy_url``: ``proxy_url`` is WHO you talk to (the
        gateway), ``proxy`` is HOW your TCP gets there. They stack — a
        gateway sitting behind a corporate egress proxy works as-is.
    trust_env:
        Honour ``HTTP_PROXY`` / ``HTTPS_PROXY`` from the environment.
    """

    NAME = "ProxiedLLMAgent"
    DEFAULT_MODEL = DEFAULT_PROXY_MODEL

    def __init__(
        self,
        proxy_url: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_map: Optional[Dict[str, str]] = None,
        default_model: Optional[str] = None,
        max_concurrency: int = 5,
        requests_per_minute: Optional[float] = None,
        queue_timeout: Optional[float] = 60.0,
        max_retries: int = 3,
        timeout: float = 300.0,
        extra_headers: Optional[Dict[str, str]] = None,
        trust_env: bool = True,
        proxy: Optional[str] = None,
    ) -> None:
        self.model_map = dict(model_map or {})
        resolved_url = proxy_url or base_url or _first_env(ENV_BASE_URL) or DEFAULT_PROXY_BASE_URL
        resolved_key = api_key or _first_env(ENV_API_KEY) or "public"

        if resolved_url == DEFAULT_PROXY_BASE_URL and not os.environ.get("OPENAI_API_KEY"):
            logger.info(
                "[%s] no proxy configured — defaulting to %s (set LLM_PROXY_BASE_URL to override)",
                self.NAME, DEFAULT_PROXY_BASE_URL,
            )

        super().__init__(
            base_url=resolved_url,
            api_key=resolved_key,
            max_retries=max_retries,
            timeout=timeout,
            extra_headers=extra_headers,
            default_model=default_model,
            proxy=proxy,
            trust_env=trust_env,
        )
        self._setup_throttling(max_concurrency, requests_per_minute, queue_timeout)

    def resolve_model(self, model: str, settings: Optional[Dict] = None) -> str:
        """Resolve the model through the alias map.

        The ``"default"`` fallback happens first so ``default_model`` can be
        aliased too (e.g. ``default_model="alias"`` + ``model_map``).
        Unmapped names pass through unchanged.
        """
        target = super().resolve_model(model, settings)
        mapped = self.model_map.get(target)
        if mapped:
            logger.debug("[%s] model %r -> %r via proxy", self.NAME, target, mapped)
            return mapped
        return target
