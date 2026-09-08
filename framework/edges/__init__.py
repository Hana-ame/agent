"""VEA V4 Modular Edge Package.

Provides specialized edge implementations:
- base: EdgeV4, EdgeResultV4, AgentProtocol, MockAgentV4
- code: CodeEdgeV4
- tool: ToolEdgeV4, LLMToolEdgeV4
- llm: LLMEdgeV4, ChatLLMEdgeV4, GenerateLLMEdgeV4, ProcessLLMEdgeV4, CallableLLMEdgeV4
- reflexive: ReflexiveEdgeV4
- cli: standalone CLI execution runner
"""

from framework.edges.base import (
    AgentProtocol,
    EdgeResultV4,
    EdgeV4,
    MockAgentV4,
)
from framework.edges.code import CodeEdgeV4
from framework.edges.tool import LLMToolEdgeV4, ToolEdgeV4
from framework.edges.llm import LLMEdgeV4
from framework.edges.reflexive import ReflexiveEdgeV4

# Re-export subclasses located in separate files for backwards-compatibility
from framework.chat_llm_edge_v4 import ChatLLMEdgeV4
from framework.generate_llm_edge_v4 import GenerateLLMEdgeV4
from framework.process_llm_edge_v4 import ProcessLLMEdgeV4
from framework.callable_llm_edge_v4 import CallableLLMEdgeV4

__all__ = [
    "EdgeResultV4",
    "EdgeV4",
    "CodeEdgeV4",
    "ToolEdgeV4",
    "LLMToolEdgeV4",
    "LLMEdgeV4",
    "MockAgentV4",
    "AgentProtocol",
    "ChatLLMEdgeV4",
    "GenerateLLMEdgeV4",
    "ProcessLLMEdgeV4",
    "CallableLLMEdgeV4",
    "ReflexiveEdgeV4",
]
