"""GraphTool Handlers — Tools callable by LLM that trigger VEA Graph execution internally.

Each handler receives arguments from the LLM, builds and executes a VEA Graph, and returns the result.
Graph configurations can be loaded via GraphRegistry or defined within the handler.

Architecture:
  Client -> Serve layer -> ToolCallEdge (main graph)
                         | LLM selects tool call
                         -> graph_search_tool -> executes search subgraph -> returns result
                         -> graph_summarize_tool -> executes summarize subgraph -> returns result
                         -> LLM consolidates final response

Usage (edge settings):
{
  "tools": [{
    "type": "function",
    "function": {
      "name": "search_knowledge",
      "description": "Search the knowledge base",
      "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"]
      }
    }
  }],
  "tool_handlers": {
    "search_knowledge": "graph_tool_handlers.py:search_knowledge"
  }
}
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import os
from typing import Any, Dict

from framework.agents.http_llm_agent import HttpLLMAgent
from framework.graph import Graph
from framework.executor import Executor

logger = logging.getLogger("graph_tool")

# -- Agent cache (avoid repeated instantiations) --
_agent_cache: Dict[str, HttpLLMAgent] = {}


def _get_agent(agent_key: str = "default") -> HttpLLMAgent:
    """Get or create cached LLM Agent."""
    if agent_key not in _agent_cache:
        _agent_cache[agent_key] = HttpLLMAgent(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            base_url=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
        )
    return _agent_cache[agent_key]


# =====================================================================
# Subgraph 1: Knowledge Search Graph (single edge LLM query)
# =====================================================================

_SEARCH_GRAPH_CONFIG = {
    "metadata": {"name": "search_subgraph"},
    "vertices": [
        {"id": "src", "settings": {"type": "source"},
         "initial_data": [{"channel": "text", "value": ""}]},
        {"id": "sink", "settings": {"type": "sink"}},
    ],
    "edges": [
        {"id": "e_search", "source": "src", "destination": "sink",
         "channel": "text", "concurrency_type": "llm",
         "settings": {
             "prompt": "You are a knowledge base search assistant. Provide a structured summary of search results based on the query.",
             "model": "gpt-4o-mini",
         }},
    ],
}


async def search_knowledge(query: str) -> str:
    """Search knowledge base tool — executes search subgraph and returns structured results.

    Args:
        query: User search keyword or question.

    Returns:
        Structured summary of search results.
    """
    logger.debug("[GraphTool] search_knowledge: %r", query)

    graph = Graph.from_dict(_SEARCH_GRAPH_CONFIG)
    agent = _get_agent()

    await graph.vertices["src"].set_data("text", query)

    executor = Executor(graph=graph, agents=agent, timeout=1200.0)
    result = await executor.run()

    if not result.success:
        return f"Search failed: {result.errors}"

    for vid, info in result.vertex_results.items():
        v = graph.vertices.get(vid)
        if v and v.settings.get("type") == "sink":
            for key, val in info.get("data", {}).items():
                if isinstance(val, str) and val:
                    return val

    return f"No results found for {query}"


# =====================================================================
# Subgraph 2: Multi-step Analysis Graph (two edges: extraction -> analysis)
# =====================================================================

_ANALYZE_GRAPH_CONFIG = {
    "metadata": {"name": "analyze_subgraph"},
    "vertices": [
        {"id": "src", "settings": {"type": "source"},
         "initial_data": [{"channel": "text", "value": ""}]},
        {"id": "mid", "settings": {}},
        {"id": "sink", "settings": {"type": "sink"}},
    ],
    "edges": [
        {"id": "e_extract", "source": "src", "destination": "mid",
         "channel": "text", "concurrency_type": "llm",
         "settings": {
             "prompt": "Extract key points from the following text and output as JSON: {"points": [...]}.",
             "model": "gpt-4o-mini",
         }},
        {"id": "e_analyze", "source": "mid", "destination": "sink",
         "channel": "text", "concurrency_type": "llm",
         "settings": {
             "prompt": "Based on the extracted key points, generate an in-depth analysis and actionable recommendations.",
             "model": "gpt-4o-mini",
         }},
    ],
}


async def analyze_text(text: str) -> str:
    """Analysis tool — executes multi-step analysis subgraph (extraction -> analysis).

    Args:
        text: The text to analyze.

    Returns:
        In-depth analysis and recommendations.
    """
    logger.debug("[GraphTool] analyze_text: %r", text[:100])

    graph = Graph.from_dict(_ANALYZE_GRAPH_CONFIG)
    agent = _get_agent()

    await graph.vertices["src"].set_data("text", text)

    executor = Executor(graph=graph, agents=agent, timeout=1200.0)
    result = await executor.run()

    if not result.success:
        return f"Analysis failed: {result.errors}"

    for vid, info in result.vertex_results.items():
        v = graph.vertices.get(vid)
        if v and v.settings.get("type") == "sink":
            for key, val in info.get("data", {}).items():
                if isinstance(val, str) and val:
                    return val

    return "Analysis complete but no result obtained"


# =====================================================================
# Subgraph 3: Shell Command Execution Graph
# =====================================================================

_SHELL_GRAPH_CONFIG = {
    "metadata": {"name": "shell_subgraph"},
    "vertices": [
        {"id": "src", "settings": {"type": "source"},
         "initial_data": [{"channel": "text", "value": ""}]},
        {"id": "mid", "settings": {}},
        {"id": "sink", "settings": {"type": "sink"}},
    ],
    "edges": [
        {"id": "e_shell", "source": "src", "destination": "mid",
         "channel": "text",
         "script": "framework/serve/shell_edge.py:ShellCmdEdge",
         "settings": {"timeout": 10}},
        {"id": "e_summarize", "source": "mid", "destination": "sink",
         "channel": "text", "concurrency_type": "llm",
         "settings": {
             "prompt": "Briefly summarize the following command output.",
             "model": "gpt-4o-mini",
         }},
    ],
}


async def run_shell_command(command: str) -> str:
    """Shell tool — executes shell command subgraph and returns command summary.

    Args:
        command: Shell command to execute.

    Returns:
        Summary of command execution output.
    """
    logger.debug("[GraphTool] run_shell_command: %r", command)

    graph = Graph.from_dict(_SHELL_GRAPH_CONFIG)
    agent = _get_agent()

    await graph.vertices["src"].set_data("text", command)

    edge = graph.edges["e_shell"]
    edge.settings["command"] = command

    executor = Executor(graph=graph, agents=agent, timeout=1200.0)
    result = await executor.run()

    if not result.success:
        return f"Command execution failed: {result.errors}"

    for vid, info in result.vertex_results.items():
        v = graph.vertices.get(vid)
        if v and v.settings.get("type") == "sink":
            for key, val in info.get("data", {}).items():
                if isinstance(val, str) and val:
                    return val

    return "Command executed but no result obtained"


# =====================================================================
# General Tool: Execute Any Registered Graph
# =====================================================================

async def run_graph(graph_name: str, input_text: str) -> str:
    """General graph execution tool — executes a registered graph by name.

    Args:
        graph_name: Registered name of the graph.
        input_text: Input text passed to source vertex.

    Returns:
        Execution result.
    """
    logger.debug("[GraphTool] run_graph: %r with input %r", graph_name, input_text[:50])
    return f"Graph '{graph_name}' is not registered. Please use a specific tool function."
