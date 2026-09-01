import logging
from typing import Dict, Optional, Union

from .base_agent import BaseAgent
from .http_llm_agent import HttpLLMAgent
from .pi_agent_runner import PiAgentRunner
from ..utils.script_loader import load_class_from_script

logger = logging.getLogger("vertex_edge_agent.agents")

#: Default concurrency ceiling per agent type. ``None`` = unbounded.
_DEFAULT_CONCURRENCY = {}

#: String shorthand -> class. One place to keep the spec table in sync.
_STRING_AGENTS = {
    "http": HttpLLMAgent,
    "pi": PiAgentRunner,
}

def _build_from_dict(agent_spec: Dict) -> BaseAgent:
    """Build an agent from a JSON config block: ``{"type": "http", ...}``."""
    agent_type = agent_spec.get("type")

    proxy = (
        agent_spec.get("proxy")
        or agent_spec.get("https_proxy")
        or agent_spec.get("HTTPS_PROXY")
    )

    common = {
        "max_retries": agent_spec.get("max_retries", 3),
        "timeout": agent_spec.get("timeout", 300.0),
        "extra_headers": agent_spec.get("extra_headers"),
        "proxy": proxy,
        "trust_env": agent_spec.get("trust_env", True),
    }

    if agent_type == "http":
        return HttpLLMAgent(
            api_key=agent_spec.get("api_key", "public"),
            base_url=agent_spec.get("base_url", "https://opencode.ai/zen/v1/chat/completions"),
            **common,
        )

    raise ValueError(f"Unsupported agent config type: {agent_type}")


def get_agent(agent_spec: Union[str, BaseAgent, Dict, None]) -> Optional[BaseAgent]:
    if agent_spec is None:
        return None

    if isinstance(agent_spec, BaseAgent):
        return agent_spec

    if isinstance(agent_spec, str):
        agent_cls = _STRING_AGENTS.get(agent_spec)
        if agent_spec == "mock":
            return HttpLLMAgent(mock=True)
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
        if agent_spec.get("type") == "opencode":
            agent_spec["type"] = "http"
        return _build_from_dict(agent_spec)

    return None
