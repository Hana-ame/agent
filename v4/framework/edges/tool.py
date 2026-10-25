"""ToolEdgeV4 and LLMToolEdgeV4 implementations for declarative and dynamic tool calling."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Union

from framework.edges.base import EdgeResultV4, EdgeV4, AgentProtocol
from framework.edges.llm import LLMEdgeV4
from framework.edges.registry import register_edge_type
from framework.vertex_v4 import VertexStateV4, VertexStoreV4

logger = logging.getLogger("vertex_edge_agent.edges.tool")


@register_edge_type("tool", "tool_call")
class ToolEdgeV4(EdgeV4):
    """Declarative static tool edge for Agent Harness integration.

    Directly produces an OpenAI-compatible tool call (e.g. bash(command="ls"))
    that the external agent harness executes in its environment/sandbox.
    """

    EMITS_TOOL_CALL = True

    def __init__(
        self,
        edge_id: str,
        input_vertex: str,
        output_vertex: str,
        tool_name: str = "bash",
        arguments: Optional[Union[Dict[str, Any], str]] = None,
        arguments_template: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
        concurrency_limit: Optional[int] = None,
        concurrency_group: Optional[str] = None,
        priority: int = 0,
        timeout: Optional[float] = None,
    ):
        edge_settings = dict(settings or {})
        edge_settings["tool_name"] = tool_name
        if arguments is not None:
            edge_settings["arguments"] = arguments
        if arguments_template is not None:
            edge_settings["arguments_template"] = arguments_template
        super().__init__(
            edge_id=edge_id,
            input_vertex=input_vertex,
            output_vertex=output_vertex,
            edge_type="tool",
            settings=edge_settings,
            concurrency_limit=concurrency_limit,
            concurrency_group=concurrency_group,
            priority=priority,
            timeout=timeout,
        )
        self.tool_name = tool_name
        self.arguments = arguments if arguments is not None else {}
        self.arguments_template = arguments_template

    @classmethod
    def from_config_dict(cls, data: Dict[str, Any], base_dir: Optional[str] = None) -> "ToolEdgeV4":
        """Build a tool edge from a config dict (``tool_name``/``arguments`` aliases)."""
        from framework.edges.base import _config_common_kwargs, _config_edge_id, _config_endpoint

        settings = dict(data.get("settings") or {})
        in_v = _config_endpoint(data, "input")
        out_v = _config_endpoint(data, "output")
        if not in_v or not out_v:
            raise ValueError("Both input and output vertices are required for a tool edge")
        tool_name = (
            data.get("tool_name") or data.get("tool")
            or settings.get("tool_name") or settings.get("tool") or "bash"
        )
        arguments = (
            data.get("arguments") or data.get("args")
            or settings.get("arguments") or settings.get("args") or {}
        )
        arguments_template = data.get("arguments_template") or settings.get("arguments_template")
        return cls(
            edge_id=_config_edge_id(data),
            input_vertex=str(in_v),
            output_vertex=str(out_v),
            tool_name=str(tool_name),
            arguments=arguments,
            arguments_template=arguments_template,
            **_config_common_kwargs(data),
        )

    def build_tool_call(self, in_v_content: str = "", call_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate OpenAI standard tool_call dictionary."""
        cid = call_id or f"call_{uuid.uuid4().hex[:12]}"
        if self.arguments_template:
            interpolated = self.arguments_template.replace("{input}", in_v_content)
            try:
                args = json.loads(interpolated)
            except Exception:
                args = {"command": interpolated}
        elif isinstance(self.arguments, dict):
            args = {}
            for k, v in self.arguments.items():
                if isinstance(v, str) and "{input}" in v:
                    args[k] = v.replace("{input}", in_v_content)
                else:
                    args[k] = v
        elif isinstance(self.arguments, str):
            args = {"command": self.arguments.replace("{input}", in_v_content)}
        else:
            args = {}

        return {
            "id": cid,
            "type": "function",
            "function": {
                "name": self.tool_name,
                "arguments": json.dumps(args) if isinstance(args, dict) else str(args),
            },
        }

    async def run(
        self,
        session_id: str,
        store: VertexStoreV4,
        agent: Optional[AgentProtocol] = None,
        auto_transition: bool = True,
        tool_output: Optional[str] = None,
        **kwargs: Any,
    ) -> EdgeResultV4:
        """Settle tool execution output into downstream vertex."""
        satisfied, reason, in_v, out_v = self.check_handshake(session_id, store)
        if not satisfied or in_v is None or out_v is None:
            return EdgeResultV4(
                edge_id=self.id,
                success=False,
                skipped=True,
                reason=reason,
            )

        output_str = str(tool_output) if tool_output is not None else in_v.content

        merge_strategy = self.settings.get("merge_strategy", "overwrite")
        store.apply_merge_strategy(
            session_id=session_id,
            name=self.output_vertex,
            incoming_content=output_str,
            strategy=merge_strategy,
        )

        if auto_transition:
            store.update_vertex_state(session_id, self.output_vertex, VertexStateV4.DATA_READY.value)
            store.increment_processed_count(session_id, self.output_vertex)

        return EdgeResultV4(
            edge_id=self.id,
            success=True,
            output=output_str,
            metadata={"tool_name": self.tool_name},
        )


@register_edge_type("llm_tool", "llm_tool_call")
class LLMToolEdgeV4(LLMEdgeV4):
    """Dynamic LLM-driven tool edge for Agent Harness integration.

    The LLM inspects upstream context and determines which tool call to emit
    to the external agent harness.
    """

    EMITS_TOOL_CALL = True

    def __init__(
        self,
        edge_id: str,
        input_vertex: str,
        output_vertex: str,
        model: str = "sensenova-6.8-flash-lite",
        prompt_template: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        agent: Optional[AgentProtocol] = None,
        settings: Optional[Dict[str, Any]] = None,
        concurrency_limit: Optional[int] = None,
        concurrency_group: Optional[str] = "llm",
        priority: int = 0,
        timeout: Optional[float] = None,
        agent_mode: Optional[str] = None,
    ):
        edge_settings = dict(settings or {})
        # Persist the tool catalog in settings so it survives to_dict()/DB round
        # trips; previously it lived only on the instance and was silently lost.
        edge_settings["tools"] = list(tools or [])
        super().__init__(
            edge_id=edge_id,
            input_vertex=input_vertex,
            output_vertex=output_vertex,
            model=model,
            prompt_template=prompt_template,
            settings=edge_settings,
            concurrency_limit=concurrency_limit,
            concurrency_group=concurrency_group,
            priority=priority,
            timeout=timeout,
            agent_mode=agent_mode,
        )
        self.type = "llm_tool"
        self.tools = list(edge_settings["tools"])
        if agent:
            self.agent = agent

    @classmethod
    def from_config_dict(cls, data: Dict[str, Any], base_dir: Optional[str] = None) -> "LLMToolEdgeV4":
        """Build an LLM tool edge from a config dict (round-trips ``tools``)."""
        from framework.edges.base import _config_common_kwargs, _config_edge_id, _config_endpoint

        settings = dict(data.get("settings") or {})
        in_v = _config_endpoint(data, "input")
        out_v = _config_endpoint(data, "output")
        if not in_v or not out_v:
            raise ValueError("Both input and output vertices are required for an llm_tool edge")
        model = data.get("model") or settings.get("model", "sensenova-6.8-flash-lite")
        prompt_template = (
            data.get("prompt_template") or data.get("prompt") or settings.get("prompt")
        )
        tools = data.get("tools") or settings.get("tools") or []
        return cls(
            edge_id=_config_edge_id(data),
            input_vertex=str(in_v),
            output_vertex=str(out_v),
            model=model,
            prompt_template=prompt_template,
            tools=tools,
            agent_mode=data.get("agent_mode") or settings.get("agent_mode"),
            **_config_common_kwargs(data),
        )

    def build_tool_call_from_llm_response(
        self,
        response: Dict[str, Any],
        call_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Extract tool_call structure from agent or LLM response dict."""
        choices = response.get("choices", [{}])
        msg = choices[0].get("message", {}) if choices else {}
        tcs = msg.get("tool_calls")
        if tcs and isinstance(tcs, list) and len(tcs) > 0:
            tc = dict(tcs[0])
            if call_id and "id" in tc:
                tc["id"] = call_id
            return tc
        return None
