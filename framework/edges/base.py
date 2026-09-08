"""Base definitions for V4 edges: protocols, results, and the EdgeV4 abstract base class."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple, Type, Union, runtime_checkable

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
    """Resolve script to callable function.

    Untrusted references are screened at the API boundary
    (``SessionGraphManagerV4.add_or_update_edge``); this runtime path trusts the
    configured script and only enforces ``VEA_SCRIPT_ROOTS`` when set.
    """
    from framework.utils.script_loader import load_script

    parts = script_str.split(":")
    file_path = parts[0]
    func_name = parts[1] if len(parts) > 1 else None

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


def _config_endpoint(data: Dict[str, Any], role: str) -> Optional[str]:
    """Extract an input/output vertex name from the accepted config aliases."""
    if role == "input":
        return data.get("input_vertex") or data.get("input") or data.get("source") or data.get("source_id")
    return data.get("output_vertex") or data.get("output") or data.get("destination") or data.get("destination_id")


def _config_edge_id(data: Dict[str, Any]) -> str:
    """Extract the edge id from the accepted config aliases."""
    return str(data.get("id") or data.get("edge_id") or "edge_1")


def _config_script(data: Dict[str, Any], base_dir: Optional[str]) -> Optional[Any]:
    """Resolve a relative ``script`` reference against ``base_dir``."""
    script = data.get("script")
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
    return script


def _config_common_kwargs(data: Dict[str, Any]) -> Dict[str, Any]:
    """Collect the constructor kwargs shared by every edge type."""
    settings = dict(data.get("settings") or {})
    return {
        "settings": settings,
        "concurrency_limit": data.get("concurrency_limit", settings.get("concurrency_limit")),
        "concurrency_group": data.get("concurrency_group", settings.get("concurrency_group")),
        "priority": int(data.get("priority", settings.get("priority", 0))),
        "timeout": data.get("timeout", settings.get("timeout")),
    }


def is_dynamic_edge_type(edge_type: Optional[str]) -> bool:
    """Return True when ``edge_type`` names a user script/class instead of a built-in.

    Examples: ``"my_edge.py:MyEdge"``, ``"edges/custom.py"``.
    """
    if not edge_type or not isinstance(edge_type, str):
        return False
    return ":" in edge_type or edge_type.strip().endswith(".py")


def _filter_kwargs(func: Callable, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Drop kwargs ``func`` does not accept (so custom ``__init__``s stay simple)."""
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins
        return dict(kwargs)
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in params}


def _resolve_script_spec(
    spec: str,
    base_dir: Optional[str],
    script_roots: Optional[Sequence[Union[str, Path]]] = None,
) -> Tuple[str, Optional[str]]:
    """Split ``"path.py:ClassName"`` and resolve ``path.py`` on disk.

    Relative paths are resolved against ``base_dir`` first (the directory of the
    config file), then any ``script_roots`` the caller configured (for example a
    server started with ``--script-root ./my_edges``).
    """
    path_part, _, cls_name = spec.partition(":")
    if path_part and not os.path.isabs(path_part):
        candidates: List[str] = []
        if base_dir:
            candidates.append(os.path.join(base_dir, path_part))
        candidates.extend(os.path.join(str(root), path_part) for root in (script_roots or ()))
        for candidate in candidates:
            if os.path.exists(candidate):
                path_part = candidate
                break
    return path_part, (cls_name or None)


def _load_dynamic_edge_class(
    spec: str,
    base_dir: Optional[str],
    script_roots: Optional[Sequence[Union[str, Path]]] = None,
) -> Type["EdgeV4"]:
    """Load a user-defined edge class from ``"my_edge.py:MyEdge"``.

    Confinement is enforced by :func:`framework.utils.script_loader.load_script`
    (``VEA_SCRIPT_ROOTS``, or ``script_roots`` when provided). The HTTP layer
    additionally screens the spec with :func:`validate_script_reference` before
    reaching this point.
    """
    from framework.utils.script_loader import load_class_from_script

    path_part, cls_name = _resolve_script_spec(spec, base_dir, script_roots)
    allowed = [Path(root).resolve() for root in script_roots] if script_roots else None
    loaded = load_class_from_script(path_part, EdgeV4, cls_name, allowed_roots=allowed)
    if loaded is EdgeV4:
        raise ValueError(
            f"'{spec}' does not define an EdgeV4 subclass"
            + (f" named '{cls_name}'" if cls_name else "")
        )
    return loaded


def _instantiate_edge_class(
    cls: Type["EdgeV4"],
    data: Dict[str, Any],
    base_dir: Optional[str] = None,
    type_spec: Optional[str] = None,
    dynamic: bool = False,
    load_spec: Optional[str] = None,
) -> "EdgeV4":
    """Build ``cls`` from a config dict, tolerating custom constructor signatures.

    If the class overrides :meth:`EdgeV4.from_config_dict` that override is used;
    otherwise the config is mapped onto the constructor, and any kwarg the
    constructor does not accept is dropped.
    """
    overrides = getattr(cls, "from_config_dict", None)
    if overrides is not None and getattr(overrides, "__func__", None) is not EdgeV4.from_config_dict.__func__:
        # The override owns its own requirements (a reflexive edge needs only one
        # endpoint, an LLM edge needs a model, ...).
        edge = overrides(data, base_dir=base_dir)
    else:
        in_v = _config_endpoint(data, "input")
        out_v = _config_endpoint(data, "output")
        if not in_v or not out_v:
            raise ValueError(f"Both input and output vertices are required for '{cls.__name__}'")
        settings = dict(data.get("settings") or {})
        kwargs: Dict[str, Any] = {
            "edge_id": _config_edge_id(data),
            "input_vertex": str(in_v),
            "output_vertex": str(out_v),
            "script": _config_script(data, base_dir),
            "settings": settings,
            "concurrency_limit": data.get("concurrency_limit", settings.get("concurrency_limit")),
            "concurrency_group": data.get("concurrency_group", settings.get("concurrency_group")),
            "priority": int(data.get("priority", settings.get("priority", 0))),
            "timeout": data.get("timeout", settings.get("timeout")),
            "edge_type": type_spec,
        }
        edge = cls(**_filter_kwargs(cls.__init__, kwargs))

    if type_spec:
        # Preserve the declared spec so to_dict()/DB round trips resolve the same class.
        edge.type = type_spec
        if dynamic:
            # A relative spec only resolves next to its manifest (or inside a
            # configured script root), so record the absolute form for
            # store/snapshot round trips. Prefer the module the class actually
            # came from: re-resolving the raw spec here resolved it against the
            # repository root and recorded a path that did not exist.
            path_part, cls_name = _resolve_script_spec(load_spec or type_spec, base_dir)
            module = sys.modules.get(getattr(type(edge), "__module__", ""))
            resolved = os.path.abspath(getattr(module, "__file__", None) or path_part)
            edge.settings.setdefault(
                "_edge_class_spec", f"{resolved}:{cls_name}" if cls_name else resolved
            )
    return edge


class EdgeV4:
    """Base class for V4 executable edges.

    Class-level capability flags let the executor and server layer branch on
    behaviour instead of on concrete class identity:

    * ``EMITS_TOOL_CALL`` — the edge yields OpenAI ``tool_calls`` for an external
      harness instead of computing a result (``ToolEdgeV4``, ``LLMToolEdgeV4``).
    * ``IS_RECOVERY`` — a self-loop recovery edge (``ReflexiveEdgeV4``).
    """

    #: Edge type strings this class is registered under (set by the registry).
    edge_type_names: Tuple[str, ...] = ()
    #: True when the edge emits OpenAI tool calls rather than computing output.
    EMITS_TOOL_CALL: bool = False
    #: True for reflexive recovery edges.
    IS_RECOVERY: bool = False

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
        script_roots: Optional[Sequence[Union[str, Path]]] = None,
    ) -> "EdgeV4":
        """Instantiate the right EdgeV4 subclass from a JSON file or dict.

        Dispatch is driven by :data:`framework.edges.registry.EDGE_REGISTRY`, so a
        new edge type only needs to register itself (see
        :func:`framework.edges.registry.register_edge_type`) and implement
        :meth:`from_config_dict`.

        ``script_roots`` restricts where a dynamic ``"my_edge.py:MyEdge"`` spec may
        be loaded from (and is searched when the spec is relative). Omit it to use
        the defaults: repository root plus ``VEA_SCRIPT_ROOTS``.
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

        from framework.edges import registry as edge_registry

        e_type = str(data.get("type") or data.get("edge_type") or "code")
        in_v = _config_endpoint(data, "input")
        out_v = _config_endpoint(data, "output")

        # Resolution order:
        #   1. the class this was called on (MyEdge.from_config(...))
        #   2. a registered built-in type name ("code", "llm", "tool", ...)
        #   3. a user script/class spec ("my_edge.py:MyEdge")
        explicit_cls = cls is not EdgeV4 and issubclass(cls, EdgeV4)
        target_cls: Optional[Type["EdgeV4"]] = cls if explicit_cls else edge_registry.get_edge_class(e_type)
        dynamic = False
        load_spec: Optional[str] = None
        if target_cls is None and is_dynamic_edge_type(e_type):
            # to_dict()/DB records the absolute spec under "_edge_class_spec" so a
            # relative "my_edge.py:MyEdge" still resolves from another directory.
            recorded = (data.get("settings") or {}).get("_edge_class_spec")
            load_spec = recorded or e_type
            target_cls = _load_dynamic_edge_class(load_spec, base_dir, script_roots)
            dynamic = True

        # A self-loop is a reflexive recovery edge by default — but never override
        # a class the caller named explicitly.
        if (
            not explicit_cls
            and not dynamic
            and in_v
            and out_v
            and in_v == out_v
            and not getattr(target_cls, "IS_RECOVERY", False)
        ):
            reflexive_cls = edge_registry.get_edge_class("reflexive")
            if reflexive_cls is not None:
                target_cls = reflexive_cls

        if target_cls is None:
            raise ValueError(
                f"Unsupported edge type '{e_type}'. Use a registered type "
                f"({', '.join(edge_registry.edge_type_choices())}) or a script spec "
                "such as 'my_edge.py:MyEdge'."
            )
        return _instantiate_edge_class(
            target_cls,
            data,
            base_dir,
            type_spec=e_type if dynamic else None,
            dynamic=dynamic,
            load_spec=load_spec,
        )

    @classmethod
    def from_config_dict(
        cls,
        data: Dict[str, Any],
        base_dir: Optional[str] = None,
    ) -> "EdgeV4":
        """Build this edge type from a flattened config dict.

        Subclasses override this to read their own fields; the default handles a
        plain edge with an optional script.
        """
        in_v = _config_endpoint(data, "input")
        out_v = _config_endpoint(data, "output")
        if not in_v or not out_v:
            raise ValueError(f"Both input and output vertices are required for '{cls.__name__}'")
        return cls(
            edge_id=_config_edge_id(data),
            input_vertex=str(in_v),
            output_vertex=str(out_v),
            script=_config_script(data, base_dir),
            **_config_common_kwargs(data),
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
