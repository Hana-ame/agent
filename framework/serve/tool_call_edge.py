"""ToolCall Edge — LLM tool calling loop.

The edge receives user messages, passes tool definitions to the LLM,
loops over tool_calls -> executes tools -> returns results until LLM stops.

Data flow:
  src vertex ──[ToolCallEdge]──▶ mid vertex ──▶ ... ──▶ sink vertex

Tools are defined in edge settings under "tools" (list of JSON schemas).
Tool handlers are mapped via "tool_handlers" (callable mapping or script path).

Usage example (graph config):
{
  "edges": [{
    "id": "e_tool",
    "source": "src",
    "destination": "mid",
    "channel": "text",
    "script": "tool_call_edge.py:ToolCallEdge",
    "settings": {
      "prompt": "You are a helpful assistant with access to tools.",
      "model": "gpt-4o-mini",
      "tools": [
        {
          "type": "function",
          "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
              "type": "object",
              "properties": {"city": {"type": "string"}},
              "required": ["city"]
            }
          }
        }
      ],
      "tool_handlers": {
        "get_weather": "tool_handlers.py:get_weather"
      },
      "max_iterations": 5
    }
  }]
}
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from framework.agents._http_base import _HTTPAgentBase
from framework.edge import Edge

logger = logging.getLogger("tool_call_edge")


class ToolCallEdge(Edge):
    """LLM tool-calling loop edge.

    1. Sends user message + tools definitions to the LLM
    2. Parses ``tool_calls`` from the response
    3. Executes each tool via registered handlers
    4. Sends tool results back to the LLM
    5. Loops until ``finish_reason == "stop"`` (or max iterations)
    """

    async def compute(self, data: Any, agent: Optional[Any], settings: Dict) -> Any:
        """Execute the tool-calling loop."""
        tools = settings.get("tools")
        if not tools:
            # No tools defined — fall through to default LLM behavior
            return await super().compute(data, agent, settings)

        max_iter = settings.get("max_iterations", 5)
        handlers = self._resolve_handlers(settings.get("tool_handlers", {}))

        # Build initial messages
        user_content = str(data) if not isinstance(data, str) else data
        messages: List[Dict[str, Any]] = []
        if self.prompt:
            messages.append({"role": "system", "content": str(self.prompt)})
        messages.append({"role": "user", "content": user_content})

        model = self.model or "gpt-4o-mini"

        for iteration in range(max_iter):
            logger.debug("[ToolCall:%s] Iteration %d/%d", self.id, iteration + 1, max_iter)

            # Call LLM with tools
            response = await self._call_llm_with_tools(
                agent, messages, model, tools, settings
            )

            # Check finish reason
            choice = response.get("choices", [{}])[0]
            finish_reason = choice.get("finish_reason", "stop")
            msg = choice.get("message", {})

            if finish_reason != "tool_calls":
                # Done — return final content
                return msg.get("content", "")

            # Process tool calls
            tool_calls = msg.get("tool_calls", [])
            messages.append(msg)  # Add assistant message with tool_calls

            for tc in tool_calls:
                fn_name = tc.get("function", {}).get("name", "")
                fn_args_str = tc.get("function", {}).get("arguments", "{}")
                tc_id = tc.get("id", "")

                try:
                    fn_args = json.loads(fn_args_str) if isinstance(fn_args_str, str) else fn_args_str
                except json.JSONDecodeError:
                    fn_args = {}

                # Execute tool
                result = await self._execute_tool(handlers, fn_name, fn_args)
                result_str = json.dumps(result, ensure_ascii=False) if not isinstance(result, str) else result

                logger.debug("[ToolCall:%s] Tool %r → %s", self.id, fn_name, result_str[:200])

                # Add tool result message
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": result_str,
                })

        # Max iterations reached
        return messages[-1].get("content", f"[max_iterations reached after {max_iter} rounds]")

    # ------------------------------------------------------------------
    # LLM call with tools
    # ------------------------------------------------------------------

    async def _call_llm_with_tools(
        self,
        agent: Any,
        messages: List[Dict],
        model: str,
        tools: List[Dict],
        settings: Dict,
    ) -> Dict:
        """Call the LLM with tools in the payload and return parsed response."""
        if agent is None:
            from framework.agents.http_llm_agent import HttpLLMAgent
            agent = HttpLLMAgent(api_key="public", base_url="https://api.openai.com/v1")

        # Build payload manually (agent.build_payload doesn't support tools)
        import httpx

        api_key = getattr(agent, "api_key", "public")
        base_url = getattr(agent, "base_url", "https://api.openai.com/v1")
        url = f"{base_url.rstrip('/')}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        payload = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
        }

        # Override from settings
        payload.update(settings.get("llm_kwargs", {}))

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            return resp.json()

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    async def _execute_tool(
        self,
        handlers: Dict[str, Callable],
        name: str,
        args: Dict,
    ) -> Any:
        """Execute a tool by name. Supports async and sync handlers."""
        handler = handlers.get(name)
        if handler is None:
            return f"Error: unknown tool '{name}'"
        try:
            result = handler(**args)
            if asyncio.iscoroutine(result):
                result = await result
            return result
        except Exception as exc:
            logger.error("[ToolCall:%s] Tool %r failed: %s", self.id, name, exc)
            return f"Error: {exc}"

    def _resolve_handlers(self, raw: Dict) -> Dict[str, Callable]:
        """Resolve tool handler references to actual callables.

        Supports:
        - Direct callable: {"my_tool": some_function}
        - Script path:     {"my_tool": "path/to/module.py:function_name"}
        """
        resolved: Dict[str, Callable] = {}
        for name, ref in raw.items():
            if callable(ref):
                resolved[name] = ref
            elif isinstance(ref, str):
                resolved[name] = self._load_from_script(ref)
            else:
                logger.warning("[ToolCall:%s] Unknown handler type for %r: %r",
                               self.id, name, type(ref))
        return resolved

    @staticmethod
    def _load_from_script(ref: str) -> Optional[Callable]:
        """Load a callable from 'module.py:function' or 'module:function'."""
        if ":" not in ref:
            return None
        path_part, func_name = ref.split(":", 1)
        path = Path(path_part)

        # Add parent to sys.path
        parent = str(path.parent.resolve())
        if parent not in sys.path:
            sys.path.insert(0, parent)

        try:
            if path.suffix == ".py":
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    path.stem, str(path.resolve())
                )
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
            else:
                mod = __import__(path_part, fromlist=[func_name])
            return getattr(mod, func_name)
        except Exception as exc:
            logger.error("[ToolCall] Failed to load %r: %s", ref, exc)
            return None
