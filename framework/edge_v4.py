"""V4 Standalone Edge Execution Engine.

Implements two-sided handshake forward edges and reflexive self-loop recovery edges.
Each edge executes against strictly two vertex references (input and output) within a session.
Supports both in-framework orchestration and standalone CLI/process execution.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Protocol, Tuple, Union, runtime_checkable

# Bootstrap repository root into sys.path to enable execution from any working directory
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from framework.vertex_v4 import (
    VertexAttributeV4,
    VertexRecordV4,
    VertexStateV4,
    VertexStoreV4,
)

logger = logging.getLogger("vertex_edge_agent.edge_v4")

@runtime_checkable
class AgentProtocol(Protocol):
    """Protocol for LLM agents compatible with LLMEdgeV4.
    
    Agents must implement at least one of: chat(), generate(), process(), or __call__().
    Dispatch priority: chat > generate > process > __call__.
    """
    def chat(self, prompt: str, **kwargs) -> str: ...


class MockAgentV4:
    """Mock agent supporting chat, generate, process, and callable protocols for offline testing."""

    def __init__(self, response_text: str = '{"mock": true, "status": "success", "content": "mock agent response"}'):
        self.response_text = response_text
        self.calls: List[Dict[str, Any]] = []

    async def chat(self, messages: Any, **kwargs: Any) -> str:
        self.calls.append({"protocol": "chat", "messages": messages, "kwargs": kwargs})
        return self.response_text

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        self.calls.append({"protocol": "generate", "prompt": prompt, "kwargs": kwargs})
        return self.response_text

    async def process(self, content: Any, prompt: str, **kwargs: Any) -> str:
        self.calls.append({"protocol": "process", "content": content, "prompt": prompt, "kwargs": kwargs})
        return self.response_text

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        self.calls.append({"protocol": "callable", "prompt": prompt, "kwargs": kwargs})
        return self.response_text




@dataclass
class EdgeResultV4:
    """Result of an EdgeV4 execution."""

    edge_id: str
    success: bool
    skipped: bool = False
    output: Any = None
    error: Optional[str] = None
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize result to dictionary."""
        return {
            "edge_id": self.edge_id,
            "success": self.success,
            "skipped": self.skipped,
            "output": self.output,
            "error": self.error,
            "reason": self.reason,
            "metadata": self.metadata,
        }


class EdgeV4:
    """Base class for V4 executable edges."""

    def __init__(
        self,
        edge_id: str,
        input_vertex: str,
        output_vertex: str,
        edge_type: str = "code",
        settings: Optional[Dict[str, Any]] = None,
        concurrency_limit: Optional[int] = None,
        concurrency_group: Optional[str] = None,
        priority: int = 0,
        timeout: Optional[float] = None,
    ):
        self.id = edge_id
        self.input_vertex = str(input_vertex)
        self.output_vertex = str(output_vertex)
        self.type = edge_type
        self.settings = dict(settings or {})
        if concurrency_limit is not None:
            self.settings["concurrency_limit"] = int(concurrency_limit)
        if concurrency_group is not None:
            self.settings["concurrency_group"] = str(concurrency_group)
        if priority != 0:
            self.settings["priority"] = int(priority)
        if timeout is not None:
            self.settings["timeout"] = float(timeout)

    @property
    def concurrency_limit(self) -> Optional[int]:
        """Optional limit on concurrent executions of this specific edge."""
        val = self.settings.get("concurrency_limit")
        return int(val) if val is not None else None

    @property
    def concurrency_group(self) -> str:
        """Concurrency group category (defaults to edge type if not set)."""
        return str(self.settings.get("concurrency_group") or self.type)

    @property
    def priority(self) -> int:
        """Execution priority (higher value schedules earlier within the same tier)."""
        return int(self.settings.get("priority", 0))

    @property
    def timeout(self) -> Optional[float]:
        """Per-edge timeout in seconds."""
        val = self.settings.get("timeout")
        return float(val) if val is not None else None

    @property
    def is_reflexive(self) -> bool:
        """Return True if this edge connects a vertex to itself."""
        return self.input_vertex == self.output_vertex

    def to_dict(self) -> Dict[str, Any]:
        """Serialize edge definition to dictionary representation."""
        res: Dict[str, Any] = {
            "id": self.id,
            "type": self.type,
            "input_vertex": self.input_vertex,
            "output_vertex": self.output_vertex,
            "priority": self.priority,
            "settings": dict(self.settings),
        }
        if hasattr(self, "script") and self.script is not None:
            res["script"] = self.script
        if hasattr(self, "trigger_state") and self.trigger_state is not None:
            res["trigger_state"] = self.trigger_state
        if hasattr(self, "target_state") and self.target_state is not None:
            res["target_state"] = self.target_state
        if hasattr(self, "max_retries") and self.max_retries is not None:
            res["max_retries"] = self.max_retries
        if self.concurrency_limit is not None:
            res["concurrency_limit"] = self.concurrency_limit
        if self.concurrency_group is not None:
            res["concurrency_group"] = self.concurrency_group
        if self.timeout is not None:
            res["timeout"] = self.timeout
        if hasattr(self, "model") and self.model is not None:
            res["model"] = self.model
        if hasattr(self, "prompt_template") and self.prompt_template is not None:
            res["prompt"] = self.prompt_template
        return res

    @classmethod
    def from_config_file(
        cls,
        path: Union[str, Path],
        base_dir: Optional[str] = None,
    ) -> "EdgeV4":
        """Instantiate an edge instance from a JSON configuration file."""
        return cls.from_config(path, base_dir=base_dir)

    @classmethod
    def from_config(
        cls,
        config: Union[str, Path, Dict[str, Any]],
        base_dir: Optional[str] = None,
    ) -> "EdgeV4":
        """Instantiate an EdgeV4 or appropriate subclass from a JSON file or dict.

        Supports driving all edge parameters from a single JSON configuration.
        """
        if isinstance(config, (str, Path)):
            cfg_path = Path(config)
            if not cfg_path.is_absolute() and base_dir:
                cfg_path = Path(base_dir) / cfg_path
            if not cfg_path.exists() and os.path.exists(os.path.join(_REPO_ROOT, str(config))):
                cfg_path = Path(_REPO_ROOT) / config
            if not cfg_path.exists():
                raise FileNotFoundError(f"Edge config JSON file not found: {config}")
            base_dir = str(cfg_path.parent)
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif isinstance(config, dict):
            data = dict(config)
        else:
            raise TypeError(f"Config must be a file path or dict, got {type(config).__name__}")

        edge_id = str(data.get("id") or data.get("edge_id") or "edge_1")
        e_type = str(data.get("type") or data.get("edge_type") or "code")
        in_v = data.get("input_vertex") or data.get("input") or data.get("source") or data.get("source_id")
        out_v = data.get("output_vertex") or data.get("output") or data.get("destination") or data.get("destination_id")
        settings = dict(data.get("settings") or {})
        script = data.get("script")
        priority = int(data.get("priority", settings.get("priority", 0)))
        concurrency_limit = data.get("concurrency_limit", settings.get("concurrency_limit"))
        concurrency_group = data.get("concurrency_group", settings.get("concurrency_group"))
        timeout = data.get("timeout", settings.get("timeout"))

        # Resolve script path against base_dir if relative
        if script and isinstance(script, str) and not os.path.isabs(script) and base_dir:
            if ":" in script:
                s_file, s_func = script.split(":", 1)
                cand = os.path.join(base_dir, s_file)
                if os.path.exists(cand):
                    script = f"{cand}:{s_func}"
            else:
                cand = os.path.join(base_dir, script)
                if os.path.exists(cand):
                    script = cand

        # Dynamically import subclasses to prevent circular imports
        from framework.chat_llm_edge_v4 import ChatLLMEdgeV4
        from framework.generate_llm_edge_v4 import GenerateLLMEdgeV4
        from framework.process_llm_edge_v4 import ProcessLLMEdgeV4
        from framework.callable_llm_edge_v4 import CallableLLMEdgeV4

        # Select target class
        if cls is not EdgeV4 and issubclass(cls, EdgeV4):
            target_cls = cls
        elif e_type == "reflexive" or (in_v and out_v and in_v == out_v):
            target_cls = ReflexiveEdgeV4
        elif e_type in ("tool", "tool_call"):
            target_cls = ToolEdgeV4
        elif e_type in ("llm_tool", "llm_tool_call"):
            target_cls = LLMToolEdgeV4
        elif e_type == "code":
            target_cls = CodeEdgeV4
        elif e_type == "llm_chat":
            target_cls = ChatLLMEdgeV4
        elif e_type == "llm_generate":
            target_cls = GenerateLLMEdgeV4
        elif e_type == "llm_process":
            target_cls = ProcessLLMEdgeV4
        elif e_type == "llm_callable":
            target_cls = CallableLLMEdgeV4
        elif e_type == "llm":
            target_cls = LLMEdgeV4
        else:
            target_cls = CodeEdgeV4

        if issubclass(target_cls, ReflexiveEdgeV4):
            target_node = in_v or out_v
            if not target_node:
                raise ValueError("Target vertex ('input' or 'output') is required for reflexive edge")
            trigger_state = data.get("trigger_state") or settings.get("trigger_state", VertexStateV4.REJECT.value)
            target_state = data.get("target_state") or settings.get("target_state", VertexStateV4.TODO_URGENT.value)
            raw_retries = data.get("max_retries") or settings.get("max_retries", 3)
            return ReflexiveEdgeV4(
                edge_id=edge_id,
                vertex_name=str(target_node),
                trigger_state=trigger_state,
                target_state=target_state,
                max_retries=int(raw_retries),
                script=script,
                settings=settings,
                concurrency_limit=concurrency_limit,
                concurrency_group=concurrency_group,
                priority=priority,
                timeout=timeout,
            )
        else:
            if not in_v or not out_v:
                raise ValueError(f"Both input and output vertices are required for {e_type} edge")
            if issubclass(target_cls, ToolEdgeV4):
                tool_name = data.get("tool_name") or data.get("tool") or settings.get("tool_name") or settings.get("tool") or "bash"
                arguments = data.get("arguments") or data.get("args") or settings.get("arguments") or settings.get("args") or {}
                arguments_template = data.get("arguments_template") or settings.get("arguments_template")
                return ToolEdgeV4(
                    edge_id=edge_id,
                    input_vertex=str(in_v),
                    output_vertex=str(out_v),
                    tool_name=str(tool_name),
                    arguments=arguments,
                    arguments_template=arguments_template,
                    settings=settings,
                    concurrency_limit=concurrency_limit,
                    concurrency_group=concurrency_group,
                    priority=priority,
                    timeout=timeout,
                )
            if issubclass(target_cls, LLMToolEdgeV4):
                model = data.get("model") or settings.get("model", "sensenova-6.8-flash-lite")
                prompt_template = data.get("prompt_template") or data.get("prompt") or settings.get("prompt")
                tools = data.get("tools") or settings.get("tools") or []
                return LLMToolEdgeV4(
                    edge_id=edge_id,
                    input_vertex=str(in_v),
                    output_vertex=str(out_v),
                    model=model,
                    prompt_template=prompt_template,
                    tools=tools,
                    settings=settings,
                    concurrency_limit=concurrency_limit,
                    concurrency_group=concurrency_group,
                    priority=priority,
                    timeout=timeout,
                )
            if issubclass(target_cls, LLMEdgeV4):
                model = data.get("model") or settings.get("model", "sensenova-6.8-flash-lite")
                prompt_template = data.get("prompt_template") or data.get("prompt") or settings.get("prompt")
                return target_cls(
                    edge_id=edge_id,
                    input_vertex=str(in_v),
                    output_vertex=str(out_v),
                    model=model,
                    prompt_template=prompt_template,
                    settings=settings,
                    concurrency_limit=concurrency_limit,
                    concurrency_group=concurrency_group,
                    priority=priority,
                    timeout=timeout,
                )
            else:
                return CodeEdgeV4(
                    edge_id=edge_id,
                    input_vertex=str(in_v),
                    output_vertex=str(out_v),
                    script=script,
                    settings=settings,
                    concurrency_limit=concurrency_limit,
                    concurrency_group=concurrency_group,
                    priority=priority,
                    timeout=timeout,
                )

    def check_handshake(
        self,
        session_id: str,
        store: VertexStoreV4,
    ) -> Tuple[bool, str, Optional[VertexRecordV4], Optional[VertexRecordV4]]:
        """Evaluate two-sided handshake contract against current vertex states.

        Returns:
            Tuple of (is_satisfied, reason, input_vertex_record, output_vertex_record)
        """
        in_v = store.get_vertex(session_id, self.input_vertex)
        if in_v is None:
            return False, f"Input vertex '{self.input_vertex}' not found in session '{session_id}'", None, None

        if self.is_reflexive:
            trigger = getattr(self, 'trigger_state', 'reject')
            if in_v.state == trigger:
                return True, f"Reflexive edge ready: {in_v.name} is {trigger}", in_v, in_v
            else:
                return False, f"Reflexive edge not ready: {in_v.name} is {in_v.state}, need {trigger}", in_v, in_v

        out_v = store.get_vertex(session_id, self.output_vertex)
        if out_v is None:
            return False, f"Output vertex '{self.output_vertex}' not found in session '{session_id}'", in_v, None

        # Two-sided handshake requirement:
        # Upstream must be 'data ready' AND Downstream must be in ('todo', 'todo urgent')
        upstream_ready = in_v.state == VertexStateV4.DATA_READY.value
        downstream_demanded = out_v.state in (
            VertexStateV4.TODO.value,
            VertexStateV4.TODO_URGENT.value,
        )

        if not upstream_ready:
            return (
                False,
                f"Upstream '{self.input_vertex}' state is '{in_v.state}', required '{VertexStateV4.DATA_READY.value}'",
                in_v,
                out_v,
            )

        if not downstream_demanded:
            return (
                False,
                f"Downstream '{self.output_vertex}' state is '{out_v.state}', required 'todo' or 'todo urgent'",
                in_v,
                out_v,
            )

        return True, "Handshake satisfied", in_v, out_v

    async def run(
        self,
        session_id: str,
        store: VertexStoreV4,
        agent: Optional[AgentProtocol] = None,
        auto_transition: bool = True,
        **kwargs: Any,
    ) -> EdgeResultV4:
        """Execute the edge against the storage store. Must be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement run()")

    async def run_standalone(
        self,
        session_id: str,
        store_or_db_path: Union[VertexStoreV4, str, Path],
        agent: Optional[AgentProtocol] = None,
    ) -> EdgeResultV4:
        """Execute this edge as an autonomous standalone unit."""
        should_close = False
        if isinstance(store_or_db_path, VertexStoreV4):
            store = store_or_db_path
        else:
            store = VertexStoreV4(str(store_or_db_path))
            should_close = True

        try:
            return await self.run(session_id=session_id, store=store, agent=agent)
        finally:
            if should_close:
                store.close()



def _resolve_script_callable(script_str: str, default_func_names: list[str]) -> Callable:
    """Resolve script to callable function."""
    parts = script_str.split(":")
    file_path = parts[0]
    func_name = parts[1] if len(parts) > 1 else None

    from framework.utils.script_loader import load_script
    mod = load_script(file_path)
    if func_name:
        fn = getattr(mod, func_name, None)
        if fn and callable(fn):
            return fn
        raise AttributeError(f"Callable '{func_name}' not found in '{file_path}'")

    # Look for common function names
    for candidate in default_func_names:
        fn = getattr(mod, candidate, None)
        if fn and callable(fn):
            return fn

    raise ValueError(f"No valid callable entrypoint found in '{file_path}'")

class CodeEdgeV4(EdgeV4):
    """Executes Python transformation logic or scripts between vertices."""

    def __init__(
        self,
        edge_id: str,
        input_vertex: str,
        output_vertex: str,
        script: Optional[Union[str, Callable]] = None,
        settings: Optional[Dict[str, Any]] = None,
        concurrency_limit: Optional[int] = None,
        concurrency_group: Optional[str] = None,
        priority: int = 0,
        timeout: Optional[float] = None,
    ):
        super().__init__(
            edge_id=edge_id,
            input_vertex=input_vertex,
            output_vertex=output_vertex,
            edge_type="code",
            settings=settings,
            concurrency_limit=concurrency_limit,
            concurrency_group=concurrency_group,
            priority=priority,
            timeout=timeout,
        )
        self.script = script
        self._callable: Optional[Callable] = None
        if callable(script):
            self._callable = script

    def _resolve_callable(self) -> Callable:
        """Resolve script to callable function."""
        if self._callable is not None:
            return self._callable

        if callable(self.script):
            self._callable = self.script
            return self._callable

        if isinstance(self.script, str):
            stripped = self.script.strip()
            if stripped.startswith("lambda ") or stripped.startswith("lambda:"):
                try:
                    self._callable = eval(stripped, {"__builtins__": __builtins__})
                    return self._callable
                except Exception as exc:
                    logger.warning("Failed to evaluate inline lambda '%s': %s", stripped, exc)
            self._callable = _resolve_script_callable(self.script, ["execute", "process", "run", "transform"])
            return self._callable

        # Default passthrough transformation
        self._callable = lambda content, settings, staging=None: content
        return self._callable

    async def run(
        self,
        session_id: str,
        store: VertexStoreV4,
        agent: Optional[AgentProtocol] = None,
        auto_transition: bool = True,
        **kwargs: Any,
    ) -> EdgeResultV4:
        """Execute code edge following the two-sided handshake contract."""
        satisfied, reason, in_v, out_v = self.check_handshake(session_id, store)
        if not satisfied or in_v is None or out_v is None:
            return EdgeResultV4(
                edge_id=self.id,
                success=False,
                skipped=True,
                reason=reason,
            )

        try:
            fn = self._resolve_callable()
            # Retrieve any latest staged records for context
            latest_staging = store.get_latest_staged_for_vertex(session_id, self.output_vertex)
            staging_dict = latest_staging.to_dict() if latest_staging else {}

            import inspect
            try:
                sig = inspect.signature(fn)
                pos_params = [
                    p for p in sig.parameters.values()
                    if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                ]
                has_var = any(p.kind == p.VAR_POSITIONAL for p in sig.parameters.values())
                if len(pos_params) == 1 and not has_var:
                    args = (in_v.content,)
                elif len(pos_params) == 2 and not has_var:
                    args = (in_v.content, self.settings)
                else:
                    args = (in_v.content, self.settings, staging_dict)
            except Exception:
                args = (in_v.content, self.settings, staging_dict)

            if asyncio.iscoroutinefunction(fn):
                res = await fn(*args)
            else:
                res = fn(*args)

            output_str = str(res) if res is not None else ""

            # Handshake fulfillment: Apply merge strategy
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

            # Handshake fulfillment: transition to 'data ready' and increment count
            if auto_transition:
                store.update_vertex_state(session_id, self.output_vertex, VertexStateV4.DATA_READY.value)
                store.increment_processed_count(session_id, self.output_vertex)

            return EdgeResultV4(
                edge_id=self.id,
                success=True,
                output=output_str,
                metadata={"length": len(output_str)},
            )

        except Exception as exc:
            err_msg = str(exc)
            logger.error("[CodeEdgeV4:%s] Execution failed: %s", self.id, err_msg)
            # Stage failure diagnostic attributed to this edge
            store.stage_output(
                session_id=session_id,
                edge_id=self.id,
                key="error_feedback",
                value=err_msg,
                vertex_name=self.output_vertex,
                metadata={"exception_type": type(exc).__name__},
            )
            # Transition downstream state to 'reject'
            store.update_vertex_state(session_id, self.output_vertex, VertexStateV4.REJECT.value)

            return EdgeResultV4(
                edge_id=self.id,
                success=False,
                error=err_msg,
                reason="Execution threw exception",
            )


class ToolEdgeV4(EdgeV4):
    """Declarative static tool edge for Agent Harness integration.

    Directly produces an OpenAI-compatible tool call (e.g. bash(command="ls"))
    that the external agent harness executes in its environment/sandbox.
    """

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


class LLMEdgeV4(EdgeV4):
    """Executes LLM inference between upstream input and downstream output."""

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
    ):
        edge_settings = dict(settings or {})
        edge_settings["model"] = model
        if prompt_template:
            edge_settings["prompt"] = prompt_template
        super().__init__(
            edge_id=edge_id,
            input_vertex=input_vertex,
            output_vertex=output_vertex,
            edge_type="llm",
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
    def from_base(cls, base_edge: LLMEdgeV4) -> LLMEdgeV4:
        """Create a specialized subclass instance from an existing base LLMEdgeV4."""
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
        )

    async def _invoke_agent(
        self,
        agent: Any,
        rendered_prompt: str,
        in_v: VertexRecordV4,
        out_v: VertexRecordV4,
    ) -> Any:
        """Invoke agent with rendered prompt.

        Subclasses in dedicated modules (framework.chat_llm_edge_v4, framework.generate_llm_edge_v4,
        framework.process_llm_edge_v4, framework.callable_llm_edge_v4) implement specialized protocols.
        Base LLMEdgeV4 provides fallback protocol delegation if instantiated directly.
        """
        from framework.chat_llm_edge_v4 import ChatLLMEdgeV4
        from framework.generate_llm_edge_v4 import GenerateLLMEdgeV4
        from framework.process_llm_edge_v4 import ProcessLLMEdgeV4
        from framework.callable_llm_edge_v4 import CallableLLMEdgeV4

        if hasattr(agent, "chat") and callable(agent.chat):
            sub = ChatLLMEdgeV4.from_base(self)
            return await sub._invoke_agent(agent, rendered_prompt, in_v, out_v)
        if hasattr(agent, "process") and callable(agent.process):
            sub = ProcessLLMEdgeV4.from_base(self)
            return await sub._invoke_agent(agent, rendered_prompt, in_v, out_v)
        if hasattr(agent, "generate") and callable(agent.generate):
            sub = GenerateLLMEdgeV4.from_base(self)
            return await sub._invoke_agent(agent, rendered_prompt, in_v, out_v)
        if callable(agent):
            sub = CallableLLMEdgeV4.from_base(self)
            return await sub._invoke_agent(agent, rendered_prompt, in_v, out_v)
        raise TypeError(
            f"Agent {type(agent).__name__} has no callable chat/process/generate method"
        )


class LLMToolEdgeV4(LLMEdgeV4):
    """Dynamic LLM-driven tool edge for Agent Harness integration.

    The LLM inspects upstream context and determines which tool call to emit
    to the external agent harness.
    """

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
        priority: int = 0,
        timeout: Optional[float] = None,
    ):
        super().__init__(
            edge_id=edge_id,
            input_vertex=input_vertex,
            output_vertex=output_vertex,
            model=model,
            prompt_template=prompt_template,
            settings=settings,
            priority=priority,
            timeout=timeout,
        )
        self.type = "llm_tool"
        self.tools = tools or []
        if agent:
            self.agent = agent

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


class ReflexiveEdgeV4(EdgeV4):
    """Reflexive recovery edge (input == output) designed for reset on reject."""

    def __init__(
        self,
        edge_id: str,
        vertex_name: str,
        trigger_state: str = VertexStateV4.REJECT.value,
        target_state: str = VertexStateV4.TODO_URGENT.value,
        max_retries: int = 3,
        script: Optional[Union[str, Callable]] = None,
        settings: Optional[Dict[str, Any]] = None,
        concurrency_limit: Optional[int] = None,
        concurrency_group: Optional[str] = "reflexive",
        priority: int = 0,
        timeout: Optional[float] = None,
    ):
        super().__init__(
            edge_id=edge_id,
            input_vertex=vertex_name,
            output_vertex=vertex_name,
            edge_type="reflexive",
            settings=settings,
            concurrency_limit=concurrency_limit,
            concurrency_group=concurrency_group or "reflexive",
            priority=priority,
            timeout=timeout,
        )
        self.trigger_state = trigger_state
        self.target_state = target_state
        self.max_retries = int(self.settings.get("max_retries", max_retries))
        self.script = script
        self._callable: Optional[Callable] = None
        if callable(script):
            self._callable = script

    def _resolve_callable(self) -> Optional[Callable]:
        if self._callable is not None:
            return self._callable
        if isinstance(self.script, str):
            self._callable = _resolve_script_callable(self.script, ["recover"])
            return self._callable
        return None

    async def run(
        self,
        session_id: str,
        store: VertexStoreV4,
        agent: Optional[AgentProtocol] = None,
        **kwargs: Any,
    ) -> EdgeResultV4:
        """Execute reflexive recovery upon target vertex entering trigger state."""
        target_v = store.get_vertex(session_id, self.output_vertex)
        if target_v is None:
            return EdgeResultV4(
                edge_id=self.id,
                success=False,
                skipped=True,
                reason=f"Target vertex '{self.output_vertex}' not found",
            )

        if target_v.state != self.trigger_state:
            return EdgeResultV4(
                edge_id=self.id,
                success=False,
                skipped=True,
                reason=f"Target state is '{target_v.state}', waiting for '{self.trigger_state}'",
            )

        try:
            # Retrieve diagnostics from staging table
            latest_error = store.get_latest_staged_for_vertex(
                session_id=session_id,
                vertex_name=self.output_vertex,
                key="error_feedback",
            )
            error_info = latest_error.value if latest_error else "No staged error diagnostic"

            # Circuit breaker: Check if maximum iterations reached
            if target_v.processed_count >= self.max_retries:
                logger.warning(
                    "[ReflexiveEdgeV4:%s] Max retries (%d) reached for vertex '%s'. Locking to 'forbidden'.",
                    self.id,
                    self.max_retries,
                    self.output_vertex,
                )
                store.update_vertex_state(session_id, self.output_vertex, VertexStateV4.FORBIDDEN.value)
                store.stage_output(
                    session_id=session_id,
                    edge_id=self.id,
                    key="retry_exhausted",
                    value=f"Exceeded max retries: {self.max_retries}",
                    vertex_name=self.output_vertex,
                )
                return EdgeResultV4(
                    edge_id=self.id,
                    success=False,
                    error="Max retries exceeded",
                    reason=f"Attempted {target_v.processed_count} retries out of {self.max_retries}",
                )

            # Apply optional recovery transformer
            new_content = target_v.content
            fn = self._resolve_callable()
            if fn:
                try:
                    if asyncio.iscoroutinefunction(fn):
                        new_content = await fn(target_v.content, error_info, self.settings)
                    else:
                        new_content = fn(target_v.content, error_info, self.settings)
                except Exception as rec_err:
                    logger.error("[ReflexiveEdgeV4:%s] Recovery script failed: %s", self.id, rec_err)
                    # Recovery script failed — count this toward circuit breaker
                    store.increment_processed_count(session_id, target_v.name)
                    store.stage_output(
                        session_id=session_id,
                        edge_id=self.id,
                        key="error_feedback",
                        value=f"Recovery script failed: {rec_err}",
                        vertex_name=self.output_vertex,
                        metadata={"exception_type": type(rec_err).__name__},
                    )
                    return EdgeResultV4(
                        edge_id=self.id,
                        success=False,
                        error=f"Recovery script failed: {rec_err}",
                        reason="Recovery transform threw exception",
                    )

            # Reset state back to target_state ('todo urgent' or 'todo') and increment processed_count
            store.update_vertex_content(
                session_id=session_id,
                name=self.output_vertex,
                content=str(new_content),
                state=self.target_state,
                increment_count=True,
            )

            store.stage_output(
                session_id=session_id,
                edge_id=self.id,
                key="reflexive_reset",
                value=f"Reset vertex to '{self.target_state}', attempt #{target_v.processed_count + 1}",
                vertex_name=self.output_vertex,
                metadata={"diagnostic": error_info},
            )

            return EdgeResultV4(
                edge_id=self.id,
                success=True,
                output=new_content,
                metadata={
                    "attempt": target_v.processed_count + 1,
                    "reset_to": self.target_state,
                },
            )
        except Exception as exc:
            logger.error("[ReflexiveEdgeV4:%s] Unhandled error: %s", self.id, exc)
            return EdgeResultV4(
                edge_id=self.id,
                success=False,
                error=str(exc),
                reason="Reflexive recovery encountered unhandled exception",
            )


# ------------------------------------------------------------------
# Standalone CLI Entrypoint
# ------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse standalone command line arguments."""
    parser = argparse.ArgumentParser(description="Standalone V4 Edge Runner")
    parser.add_argument("--config", "-c", default=None, help="Path to JSON file specifying all edge parameters")
    parser.add_argument("--dir", default=None, help="Base directory for resolving relative script and database paths")
    parser.add_argument("--db", default=None, help="SQLite database path")
    parser.add_argument("--session", default=None, help="Session identifier (required if not in config JSON)")
    parser.add_argument("--edge-id", default=None, help="Edge identifier")
    parser.add_argument(
        "--type",
        choices=["code", "llm", "llm_chat", "llm_generate", "llm_process", "llm_callable", "reflexive"],
        default=None,
        help="Edge type",
    )
    parser.add_argument("--input", default=None, help="Input vertex name")
    parser.add_argument("--output", default=None, help="Output vertex name")
    parser.add_argument("--script", default=None, help="Script path for code or recovery edge")
    parser.add_argument("--model", default=None, help="Model name for LLM edge")
    parser.add_argument("--settings", default=None, help="JSON settings string")
    parser.add_argument("--seed-input", default=None, help="Optional initial input string to seed into input vertex")
    parser.add_argument("--mock", action="store_true", default=None, help="Run with mock LLM agent for offline testing")
    return parser.parse_args()


def main() -> None:
    """Standalone runner entrypoint capable of executing from any working directory."""
    args = parse_args()

    config_json: Dict[str, Any] = {}
    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.is_absolute() and args.dir:
            cfg_path = Path(args.dir) / cfg_path
        if not cfg_path.exists() and os.path.exists(os.path.join(_REPO_ROOT, str(args.config))):
            cfg_path = Path(_REPO_ROOT) / args.config
        if not cfg_path.exists():
            sys.stderr.write(f"Error: Config file not found: {args.config}\n")
            sys.exit(1)
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                config_json = json.load(f)
        except Exception as e:
            sys.stderr.write(f"Error reading JSON config file '{args.config}': {e}\n")
            sys.exit(1)

    settings_dict = dict(config_json.get("settings") or {})
    if args.settings is not None:
        try:
            cli_settings = json.loads(args.settings)
            settings_dict.update(cli_settings)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON for --settings: {e}", file=sys.stderr)
            sys.exit(1)

    base_dir = args.dir or config_json.get("dir")
    if args.config and not base_dir:
        base_dir = str(Path(args.config).resolve().parent)
    if base_dir:
        abs_base_dir = os.path.abspath(base_dir)
        if abs_base_dir not in sys.path:
            sys.path.insert(0, abs_base_dir)

    session = args.session or config_json.get("session") or config_json.get("session_id")
    if not session:
        sys.stderr.write("Error: --session is required (either via CLI argument or 'session' in JSON config)\n")
        sys.exit(1)

    db_path = args.db or config_json.get("db", ":memory:")
    if db_path != ":memory:" and not os.path.isabs(db_path) and base_dir:
        db_path = os.path.join(base_dir, db_path)

    seed_input = args.seed_input if args.seed_input is not None else config_json.get("seed_input")
    is_mock = args.mock if args.mock is not None else bool(config_json.get("mock", False))

    edge_config = dict(config_json)
    if args.edge_id:
        edge_config["id"] = args.edge_id
    elif "id" not in edge_config and "edge_id" not in edge_config:
        edge_config["id"] = "standalone_edge"

    if args.type:
        edge_config["type"] = args.type
    elif "type" not in edge_config and "edge_type" not in edge_config:
        edge_config["type"] = "code"

    if args.input:
        edge_config["input_vertex"] = args.input
    if args.output:
        edge_config["output_vertex"] = args.output
    if args.script:
        edge_config["script"] = args.script
    if args.model:
        edge_config["model"] = args.model
    edge_config["settings"] = settings_dict

    try:
        edge = EdgeV4.from_config(edge_config, base_dir=base_dir)
    except Exception as exc:
        sys.stderr.write(f"Error initializing edge: {exc}\n")
        sys.exit(1)

    store = VertexStoreV4(db_path)
    try:
        if seed_input is not None and edge.input_vertex:
            store.save_vertex(
                session_id=session,
                name=edge.input_vertex,
                content=seed_input,
                state=VertexStateV4.DATA_READY.value,
            )
            if edge.output_vertex and store.get_vertex(session, edge.output_vertex) is None:
                store.save_vertex(
                    session_id=session,
                    name=edge.output_vertex,
                    content="",
                    state=VertexStateV4.TODO.value,
                )

        agent: Optional[AgentProtocol] = None
        if is_mock:
            mock_resp = settings_dict.get(
                "mock_response",
                '{"mock": true, "status": "success", "content": "mock agent response"}',
            )
            agent = MockAgentV4(response_text=str(mock_resp))

        result = asyncio.run(edge.run(session, store, agent=agent))
        sys.stdout.write(json.dumps(result.to_dict(), indent=2) + "\n")
        if not result.success and not result.skipped:
            sys.exit(1)
    finally:
        store.close()


if __name__ == "__main__":
    main()

# Re-export subclasses located in separate files for backwards-compatibility and unified access
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
    "ChatLLMEdgeV4",
    "GenerateLLMEdgeV4",
    "ProcessLLMEdgeV4",
    "CallableLLMEdgeV4",
    "ReflexiveEdgeV4",
]

