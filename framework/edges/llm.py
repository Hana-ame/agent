"""LLMEdgeV4 implementation: LLM inference edge with prompt templating and validation."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from framework.edges.base import EdgeResultV4, EdgeV4, AgentProtocol, _resolve_script_callable
from framework.edges.registry import register_edge_type
from framework.vertex_v4 import (
    VertexAttributeV4,
    VertexRecordV4,
    VertexStateV4,
    VertexStoreV4,
)

logger = logging.getLogger("vertex_edge_agent.edges.llm")


@register_edge_type("llm", "llm_edge")
class LLMEdgeV4(EdgeV4):
    """Executes LLM inference between upstream input and downstream output.

    Supports four agent invocation modes via ``agent_mode``:
    - ``"chat"``: calls ``agent.chat(prompt, ...)`` (default)
    - ``"generate"``: calls ``agent.generate(prompt, ...)``
    - ``"process"``: calls ``agent.process(raw_input, prompt, ...)``
    - ``"callable"``: calls ``agent(prompt, ...)``

    If ``agent_mode`` is not set, it auto-detects from agent capabilities.
    """

    def __init__(
        self,
        edge_id: str,
        input_vertex: str,
        output_vertex: str,
        model: str = "sensenova-6.8-flash-lite",
        prompt_template: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
        concurrency_limit: Optional[int] = None,
        concurrency_group: Optional[str] = "llm",
        priority: int = 0,
        timeout: Optional[float] = None,
        agent_mode: Optional[str] = None,
        edge_type: Optional[str] = None,
    ):
        edge_settings = dict(settings or {})
        edge_settings["model"] = model
        if prompt_template:
            edge_settings["prompt"] = prompt_template
        if agent_mode:
            edge_settings["agent_mode"] = agent_mode
        super().__init__(
            edge_id=edge_id,
            input_vertex=input_vertex,
            output_vertex=output_vertex,
            edge_type=edge_type or "llm",
            settings=edge_settings,
            concurrency_limit=concurrency_limit,
            concurrency_group=concurrency_group or "llm",
            priority=priority,
            timeout=timeout,
        )

    async def run(
        self,
        session_id: str,
        store: VertexStoreV4,
        agent: Optional[AgentProtocol] = None,
        auto_transition: bool = True,
        **kwargs: Any,
    ) -> EdgeResultV4:
        """Execute LLM edge with prompt construction and response delivery."""
        satisfied, reason, in_v, out_v = self.check_handshake(session_id, store)
        if not satisfied or in_v is None or out_v is None:
            return EdgeResultV4(
                edge_id=self.id,
                success=False,
                skipped=True,
                reason=reason,
            )

        if agent is None:
            from framework.agents import HttpLLMAgent
            agent = HttpLLMAgent()

        prompt_template = self.settings.get("prompt", "{input}")
        rendered_prompt = prompt_template.replace("{input}", in_v.content)

        model = self.settings.get("model", "sensenova-6.8-flash-lite")
        temperature = float(self.settings.get("temperature", 0.7))

        try:
            # Stage intermediate prompt draft
            store.stage_output(
                session_id=session_id,
                edge_id=self.id,
                key="rendered_prompt",
                value=rendered_prompt,
                vertex_name=self.output_vertex,
            )

            # Invoke agent via subclass protocol
            response = await self._invoke_agent(agent, rendered_prompt, in_v, out_v)
            output_str = str(response)

            # If downstream vertex expects JSON, validate syntax
            if out_v.has_attribute(VertexAttributeV4.JSON):
                try:
                    # Strip markdown fence if present
                    clean_str = output_str.strip()
                    if clean_str.startswith("```json"):
                        clean_str = clean_str[7:]
                    elif clean_str.startswith("```"):
                        clean_str = clean_str[3:]
                    if clean_str.endswith("```"):
                        clean_str = clean_str[:-3]
                    json.loads(clean_str.strip())
                    output_str = clean_str.strip()
                except Exception as json_err:
                    raise ValueError(f"Malformed JSON output: {json_err}") from json_err

            # Apply merge strategy
            merge_strategy = self.settings.get("merge_strategy", "overwrite")
            reducer_script_path = self.settings.get("reducer_script")
            reducer_fn = None
            if reducer_script_path:
                try:
                    reducer_fn = _resolve_script_callable(reducer_script_path, ["reduce", "merge"])
                except Exception as e:
                    logger.warning("Failed to resolve reducer script %s: %s", reducer_script_path, e)
                    
            store.apply_merge_strategy(
                session_id=session_id,
                name=self.output_vertex,
                incoming_content=output_str,
                strategy=merge_strategy,
                reducer_fn=reducer_fn,
            )

            if auto_transition:
                store.update_vertex_state(session_id, self.output_vertex, VertexStateV4.DATA_READY.value)
                store.increment_processed_count(session_id, self.output_vertex)

            return EdgeResultV4(
                edge_id=self.id,
                success=True,
                output=output_str,
                metadata={"model": model},
            )

        except Exception as exc:
            err_msg = str(exc)
            logger.error("[LLMEdgeV4:%s] Inference failed: %s", self.id, err_msg)
            store.stage_output(
                session_id=session_id,
                edge_id=self.id,
                key="error_feedback",
                value=err_msg,
                vertex_name=self.output_vertex,
                metadata={"model": model, "exception": type(exc).__name__},
            )
            store.update_vertex_state(session_id, self.output_vertex, VertexStateV4.REJECT.value)

            return EdgeResultV4(
                edge_id=self.id,
                success=False,
                error=err_msg,
                reason="LLM inference error",
            )

    @property
    def model(self) -> str:
        """Target LLM model identifier."""
        return str(self.settings.get("model", "sensenova-6.8-flash-lite"))

    @property
    def prompt_template(self) -> str:
        """Configured prompt template containing '{input}' placeholder."""
        return str(self.settings.get("prompt", "{input}"))

    @property
    def temperature(self) -> float:
        """Sampling temperature."""
        return float(self.settings.get("temperature", 0.7))

    @classmethod
    def from_config_dict(cls, data: Dict[str, Any], base_dir: Optional[str] = None) -> "LLMEdgeV4":
        """Build an LLM edge from a config dict (``model``/``prompt``/``agent_mode``)."""
        from framework.edges.base import _config_common_kwargs, _config_edge_id, _config_endpoint

        settings = dict(data.get("settings") or {})
        in_v = _config_endpoint(data, "input")
        out_v = _config_endpoint(data, "output")
        if not in_v or not out_v:
            raise ValueError(f"Both input and output vertices are required for '{cls.__name__}'")
        model = data.get("model") or settings.get("model", "sensenova-6.8-flash-lite")
        prompt_template = (
            data.get("prompt_template") or data.get("prompt") or settings.get("prompt")
        )
        agent_mode = data.get("agent_mode") or settings.get("agent_mode")
        return cls(
            edge_id=_config_edge_id(data),
            input_vertex=str(in_v),
            output_vertex=str(out_v),
            model=model,
            prompt_template=prompt_template,
            agent_mode=agent_mode,
            edge_type=data.get("type") or data.get("edge_type"),
            **_config_common_kwargs(data),
        )

    @classmethod
    def from_base(cls, base_edge: LLMEdgeV4) -> LLMEdgeV4:
        """Create a specialized instance from an existing base LLMEdgeV4."""
        agent_mode = base_edge.settings.get("agent_mode")
        return cls(
            edge_id=base_edge.id,
            input_vertex=base_edge.input_vertex,
            output_vertex=base_edge.output_vertex,
            model=base_edge.model,
            prompt_template=base_edge.prompt_template,
            settings=dict(base_edge.settings),
            concurrency_limit=base_edge.concurrency_limit,
            concurrency_group=base_edge.concurrency_group,
            priority=base_edge.priority,
            timeout=base_edge.timeout,
            agent_mode=agent_mode,
        )

    async def _invoke_agent(
        self,
        agent: Any,
        rendered_prompt: str,
        in_v: VertexRecordV4,
        out_v: VertexRecordV4,
    ) -> Any:
        """Invoke agent with rendered prompt.

        Dispatches based on ``agent_mode`` setting or auto-detects from agent capabilities:
        - "chat": agent.chat(prompt, ...)
        - "process": agent.process(raw_input, prompt, ...)
        - "generate": agent.generate(prompt, ...)
        - "callable": agent(prompt, ...)

        Handles both sync and async agent methods.
        """
        mode = self.settings.get("agent_mode")

        # Auto-detect if not explicitly set
        if not mode:
            if hasattr(agent, "chat") and callable(agent.chat):
                mode = "chat"
            elif hasattr(agent, "process") and callable(agent.process):
                mode = "process"
            elif hasattr(agent, "generate") and callable(agent.generate):
                mode = "generate"
            elif callable(agent):
                mode = "callable"
            else:
                raise TypeError(
                    f"Agent {type(agent).__name__} has no callable chat/process/generate method"
                )

        if mode == "chat":
            if not hasattr(agent, "chat") or not callable(agent.chat):
                raise TypeError(f"Agent {type(agent).__name__} does not implement callable 'chat'")
            res = agent.chat(
                [{"role": "user", "content": rendered_prompt}],
                model=self.model,
                temperature=self.temperature,
            )
            return await res if asyncio.iscoroutine(res) else res
        if mode == "process":
            if not hasattr(agent, "process") or not callable(agent.process):
                raise TypeError(f"Agent {type(agent).__name__} does not implement callable 'process'")
            res = agent.process(
                in_v.content,
                rendered_prompt,
                model=self.model,
                temperature=self.temperature,
                settings=self.settings,
            )
            return await res if asyncio.iscoroutine(res) else res
        if mode == "generate":
            if not hasattr(agent, "generate") or not callable(agent.generate):
                raise TypeError(f"Agent {type(agent).__name__} does not implement callable 'generate'")
            res = agent.generate(
                rendered_prompt,
                model=self.model,
                temperature=self.temperature,
            )
            return await res if asyncio.iscoroutine(res) else res
        if mode == "callable":
            if not callable(agent):
                raise TypeError(f"Agent {type(agent).__name__} is not callable")
            res = agent(rendered_prompt)
            return await res if asyncio.iscoroutine(res) else res

        raise ValueError(f"Unknown agent_mode '{mode}', expected chat|process|generate|callable")
