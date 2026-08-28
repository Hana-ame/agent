from typing import Union, Dict, Optional
import logging
from .base_agent import BaseAgent
from .mock_agent import MockAgent
from .http_llm_agent import HttpLLMAgent
from .pi_agent_runner import PiAgentRunner
from ..utils.script_loader import load_class_from_script

logger = logging.getLogger("vertex_edge_agent.agents")

def get_agent(agent_spec: Union[str, BaseAgent, Dict, None]) -> Optional[BaseAgent]:
    if agent_spec is None:
        return None
    if isinstance(agent_spec, BaseAgent):
        return agent_spec
    if isinstance(agent_spec, str):
        if agent_spec == "mock":
            return MockAgent()
        elif agent_spec == "http":
            return HttpLLMAgent()
        elif agent_spec == "pi":
            return PiAgentRunner()
        elif ":" in agent_spec:
            try:
                agent_cls = load_class_from_script(agent_spec, BaseAgent)
                return agent_cls()
            except Exception as e:
                logger.error(f"[Agents] Failed to load agent from {agent_spec}: {e}")
                raise ValueError(f"Failed to load agent {agent_spec}: {e}")
        else:
            raise ValueError(f"Unknown agent type: {agent_spec}")
    if isinstance(agent_spec, dict):
        agent_type = agent_spec.get("type")
        if agent_type == "http":
            return HttpLLMAgent(
                api_key=agent_spec.get("api_key", "public"),
                base_url=agent_spec.get("base_url", "https://opencode.ai/zen/v1"),
                max_retries=agent_spec.get("max_retries", 3)
            )
        else:
            raise ValueError(f"Unsupported agent config type: {agent_type}")
    return None
