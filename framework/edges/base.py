"""Base definitions for V4 edges: protocols, results, and the EdgeV4 abstract base class."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union, runtime_checkable

_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from framework.vertex_v4 import (
    VertexAttributeV4,
    VertexRecordV4,
    VertexStateV4,
    VertexStoreV4,
)

logger = logging.getLogger("vertex_edge_agent.edges.base")


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
            if callable(self.script):
                res["script"] = getattr(self.script, "__qualname__", getattr(self.script, "__name__", str(self.script)))
            else:
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
        from framework.edges.code import CodeEdgeV4
        from framework.edges.tool import ToolEdgeV4, LLMToolEdgeV4
        from framework.edges.reflexive import ReflexiveEdgeV4
        from framework.edges.llm import LLMEdgeV4
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
                agent_mode = data.get("agent_mode") or settings.get("agent_mode")
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
                    agent_mode=agent_mode,
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
