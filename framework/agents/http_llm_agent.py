"""Generic OpenAI-compatible LLM agent.

Talks ``/chat/completions`` to any OpenAI-compatible endpoint — OpenAI,
OpenRouter, vLLM, llama.cpp, LM Studio, ... Defaulting to the OpenCode Zen
gateway keeps it usable out of the box with no API key.

For an opinionated OpenCode variant use :class:`OpenCodeAgent`; for traffic
that must go through a self-hosted gateway use :class:`ProxiedLLMAgent`.
"""

from typing import Dict, Optional

from ._http_base import _HTTPAgentBase
from .base_agent import BaseAgent  # noqa: F401  (backwards-compatible re-export)
from ._http_base import NonRetryableHTTPError  # noqa: F401  (backwards-compatible re-export)

__all__ = ["HttpLLMAgent", "NonRetryableHTTPError"]


class HttpLLMAgent(_HTTPAgentBase):
    """Generic OpenAI-compatible chat-completions agent.

    The canonical "HTTP node with a proxy": set ``proxy`` to make every
    request *tunnel through* a transport-level HTTP(S) proxy (corporate
    egress / SOCKS / authenticated proxy) on its way to ``base_url``.

    Parameters
    ----------
    api_key:
        Bearer token. OpenCode Zen works with the default ``"public"``.
    base_url:
        Endpoint root, ``"/chat/completions"`` is appended automatically.
    max_retries:
        Total attempts for transient failures (408 / 429 / 5xx / network).
    timeout:
        Overall request timeout in seconds (connect timeout is 10 s).
    extra_headers:
        Additional headers, e.g. ``{"X-Title": "my-agent"}``.
    proxy:
        Transport-level HTTP(S) proxy the request tunnels through, e.g.
        ``"http://user:pass@corp-proxy:3128"`` or ``"socks5://host:1080"``.
        When unset, ``trust_env`` decides whether httpx reads
        ``HTTP_PROXY`` / ``HTTPS_PROXY`` from the environment.
    trust_env:
        Honour ``HTTP_PROXY`` / ``HTTPS_PROXY`` from the environment when
        ``proxy`` is unset. Independent of ``proxy``.
    """

    NAME = "HttpLLMAgent"
    DEFAULT_MODEL = "hy3-free"

    def __init__(
        self,
        api_key: str = "public",
        base_url: str = "https://opencode.ai/zen/v1",
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
            proxy=proxy,
            trust_env=trust_env,
        )
