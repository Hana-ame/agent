import logging
from typing import Dict, Optional, Union

from .base_agent import BaseAgent
from .http_llm_agent import HttpLLMAgent
from .mock_agent import MockAgent
from .opencode_agent import DEFAULT_ZEN_BASE_URL, DEFAULT_ZEN_MODEL, OpenCodeAgent
from .pi_agent_runner import PiAgentRunner
from .proxied_llm_agent import (
    DEFAULT_PROXY_BASE_URL,
    DEFAULT_PROXY_MODEL,
    ProxiedLLMAgent,
)
from ..utils.script_loader import load_class_from_script

logger = logging.getLogger("vertex_edge_agent.agents")

#: Default concurrency ceiling per agent type. ``None`` = unbounded.
_DEFAULT_CONCURRENCY = {"opencode": 3, "proxy": 5, "proxied": 5}

#: String shorthand -> class. One place to keep the spec table in sync.
_STRING_AGENTS = {
    "mock": MockAgent,
    "http": HttpLLMAgent,
    "opencode": OpenCodeAgent,
    "proxy": ProxiedLLMAgent,
    "proxied": ProxiedLLMAgent,
    "pi": PiAgentRunner,
}


def _build_from_dict(agent_spec: Dict) -> BaseAgent:
    """Build an agent from a JSON config block: ``{"type": "http|opencode|proxy", ...}``."""
    agent_type = agent_spec.get("type")

    # Knobs shared by every HTTP agent.
    common = {
        "max_retries": agent_spec.get("max_retries", 3),
        "timeout": agent_spec.get("timeout", 300.0),
        "extra_headers": agent_spec.get("extra_headers"),
        "proxy": agent_spec.get("proxy"),
        "trust_env": agent_spec.get("trust_env", True),
    }

    if agent_type == "http":
        # Plain http stays unbounded — it is the escape hatch for callers who
        # want to drive their own concurrency (e.g. an external job queue).
        return HttpLLMAgent(
            api_key=agent_spec.get("api_key", "public"),
            base_url=agent_spec.get("base_url", DEFAULT_ZEN_BASE_URL),
            **common,
        )

    if agent_type == "opencode":
        return OpenCodeAgent(
            api_key=agent_spec.get("api_key", "public"),
            base_url=agent_spec.get("base_url", DEFAULT_ZEN_BASE_URL),
            default_model=agent_spec.get("model", DEFAULT_ZEN_MODEL),
            max_concurrency=agent_spec.get("max_concurrency", _DEFAULT_CONCURRENCY[agent_type]),
            requests_per_minute=agent_spec.get("requests_per_minute", 20.0),
            queue_timeout=agent_spec.get("queue_timeout", 60.0),
            **common,
        )

    if agent_type in ("proxy", "proxied"):
        return ProxiedLLMAgent(
            proxy_url=agent_spec.get("proxy_url") or agent_spec.get("base_url"),
            api_key=agent_spec.get("api_key"),
            model_map=agent_spec.get("model_map"),
            default_model=agent_spec.get("model", DEFAULT_PROXY_MODEL),
            max_concurrency=agent_spec.get("max_concurrency", _DEFAULT_CONCURRENCY[agent_type]),
            requests_per_minute=agent_spec.get("requests_per_minute"),
            queue_timeout=agent_spec.get("queue_timeout", 60.0),
            **common,
        )

    raise ValueError(f"Unsupported agent config type: {agent_type}")


def get_agent(agent_spec: Union[str, BaseAgent, Dict, None]) -> Optional[BaseAgent]:
    """Resolve an agent spec into a live :class:`BaseAgent` instance.

    Accepted specs:

    * ``None`` / unknown -> ``None`` (edges fall back to :class:`MockAgent`)
    * ``BaseAgent`` instance -> returned unchanged
    * shorthand string -> ``"mock"``, ``"http"``, ``"opencode"``,
      ``"proxy"`` / ``"proxied"``, ``"pi"``
    * ``"path:ClassName"`` -> class loaded from an external script
    * ``dict`` -> ``{"type": "http"|"opencode"|"proxy", ...}`` config block
    """
    if agent_spec is None:
        return None

    if isinstance(agent_spec, BaseAgent):
        return agent_spec

    if isinstance(agent_spec, str):
        agent_cls = _STRING_AGENTS.get(agent_spec)
        if agent_cls is not None:
            return agent_cls()
        if ":" in agent_spec:
            try:
                agent_cls = load_class_from_script(agent_spec, BaseAgent)
                return agent_cls()
            except Exception as e:
                logger.error(f"[Agents] Failed to load agent from {agent_spec}: {e}")
                raise ValueError(f"Failed to load agent {agent_spec}: {e}")
        raise ValueError(f"Unknown agent type: {agent_spec}")

    if isinstance(agent_spec, dict):
        return _build_from_dict(agent_spec)

    return None
