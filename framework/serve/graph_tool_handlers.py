"""GraphTool Handlers — LLM 可调用的工具，内部触发 VEA Graph 执行。

每个 handler 接收 LLM 传来的参数，构建并执行一个 VEA Graph，返回结果。
Graph 配置通过 GraphRegistry 或硬编码在 handler 中。

设计思路：
  客户端 → Serve 层 → ToolCallEdge（主图）
                    ↓ LLM 决定调用工具
                    → graph_search_tool → 执行搜索子图 → 返回结果
                    → graph_summarize_tool → 执行总结子图 → 返回结果
                    → LLM 整合最终答案

用法（edge settings）：
{
  "tools": [{
    "type": "function",
    "function": {
      "name": "search_knowledge",
      "description": "搜索知识库",
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

# ── Agent 缓存（避免重复创建） ──
_agent_cache: Dict[str, HttpLLMAgent] = {}


def _get_agent(agent_key: str = "default") -> HttpLLMAgent:
    """获取或创建缓存的 LLM Agent。"""
    if agent_key not in _agent_cache:
        _agent_cache[agent_key] = HttpLLMAgent(
            api_key=os.environ.get("OPENAI_API_KEY", "sk-Ji81kJUtXPFAHftIfxb3RNhwTyh3IxTy"),
            base_url=os.environ.get("LLM_BASE_URL", "https://token.sensenova.cn/v1"),
        )
    return _agent_cache[agent_key]


# =====================================================================
# 子图 1: 知识库搜索图（单 edge LLM 查询）
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
             "prompt": "你是一个知识库搜索助手。根据用户查询，生成结构化的搜索结果摘要。用中文回答。",
             "model": "sensenova-6.8-flash-lite",
         }},
    ],
}


async def search_knowledge(query: str) -> str:
    """搜索知识库工具 — 执行搜索子图，返回结构化搜索结果。

    参数:
        query: 用户搜索关键词

    返回:
        结构化的搜索结果文本
    """
    logger.debug("[GraphTool] search_knowledge: %r", query)

    graph = Graph.from_dict(_SEARCH_GRAPH_CONFIG)
    agent = _get_agent()

    # 注入查询到 source vertex
    await graph.vertices["src"].set_data("text", query)

    executor = Executor(graph=graph, agents=agent, timeout=1200.0)
    result = await executor.run()

    if not result.success:
        return f"搜索失败: {result.errors}"

    # 从 sink vertex 提取结果
    for vid, info in result.vertex_results.items():
        v = graph.vertices.get(vid)
        if v and v.settings.get("type") == "sink":
            for key, val in info.get("data", {}).items():
                if isinstance(val, str) and val:
                    return val

    return f"未找到关于 {query} 的结果"


# =====================================================================
# 子图 2: 多步骤分析图（两 edge：预处理 → 分析）
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
        # Edge 1: 预处理（提取关键信息）
        {"id": "e_extract", "source": "src", "destination": "mid",
         "channel": "text", "concurrency_type": "llm",
         "settings": {
             "prompt": "从以下文本中提取关键信息要点，用 JSON 格式输出 {\"points\": [...]}。",
             "model": "sensenova-6.8-flash-lite",
         }},
        # Edge 2: 分析（基于提取的要点生成洞察）
        {"id": "e_analyze", "source": "mid", "destination": "sink",
         "channel": "text", "concurrency_type": "llm",
         "settings": {
             "prompt": "基于提取的关键信息，生成深度分析和建议。用中文回答。",
             "model": "sensenova-6.8-flash-lite",
         }},
    ],
}


async def analyze_text(text: str) -> str:
    """分析工具 — 执行多步骤分析子图（提取 → 分析）。

    参数:
        text: 要分析的文本

    返回:
        深度分析和建议
    """
    logger.debug("[GraphTool] analyze_text: %r", text[:100])

    graph = Graph.from_dict(_ANALYZE_GRAPH_CONFIG)
    agent = _get_agent()

    await graph.vertices["src"].set_data("text", text)

    executor = Executor(graph=graph, agents=agent, timeout=1200.0)
    result = await executor.run()

    if not result.success:
        return f"分析失败: {result.errors}"

    for vid, info in result.vertex_results.items():
        v = graph.vertices.get(vid)
        if v and v.settings.get("type") == "sink":
            for key, val in info.get("data", {}).items():
                if isinstance(val, str) and val:
                    return val

    return "分析完成但未获取结果"


# =====================================================================
# 子图 3: Shell 命令执行图
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
        # Edge 1: 执行 shell 命令
        {"id": "e_shell", "source": "src", "destination": "mid",
         "channel": "text",
         "script": "framework/serve/shell_edge.py:ShellCmdEdge",
         "settings": {"timeout": 10}},
        # Edge 2: 总结 shell 输出
        {"id": "e_summarize", "source": "mid", "destination": "sink",
         "channel": "text", "concurrency_type": "llm",
         "settings": {
             "prompt": "简要总结以下命令输出，用中文。",
             "model": "sensenova-6.8-flash-lite",
         }},
    ],
}


async def run_shell_command(command: str) -> str:
    """Shell 执行工具 — 执行 shell 命令子图，返回命令输出总结。

    参数:
        command: 要执行的 shell 命令

    返回:
        命令输出的总结
    """
    logger.debug("[GraphTool] run_shell_command: %r", command)

    graph = Graph.from_dict(_SHELL_GRAPH_CONFIG)
    agent = _get_agent()

    # 注入命令到 source vertex
    await graph.vertices["src"].set_data("text", command)

    # 修改 shell edge 的命令
    edge = graph.edges["e_shell"]
    edge.settings["command"] = command

    executor = Executor(graph=graph, agents=agent, timeout=1200.0)
    result = await executor.run()

    if not result.success:
        return f"命令执行失败: {result.errors}"

    for vid, info in result.vertex_results.items():
        v = graph.vertices.get(vid)
        if v and v.settings.get("type") == "sink":
            for key, val in info.get("data", {}).items():
                if isinstance(val, str) and val:
                    return val

    return "命令执行完成但未获取结果"


# =====================================================================
# 通用工具: 执行任意注册图
# =====================================================================

async def run_graph(graph_name: str, input_text: str) -> str:
    """通用图执行工具 — 按名称执行预注册的 Graph。

    参数:
        graph_name: 图注册名（需在外部注册）
        input_text: 输入文本

    返回:
        图执行结果

    注意: 需要外部通过 GraphRegistry 注册图配置。
    此函数作为示例，实际使用时需要传入 registry。
    """
    logger.debug("[GraphTool] run_graph: %r with input %r", graph_name, input_text[:50])
    return f"图 '{graph_name}' 未注册。请使用具体工具函数。"
