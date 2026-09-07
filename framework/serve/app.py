"""FastAPI application factory and graph registry for VEA serve layer.

The :class:`GraphRegistry` holds named graph configurations and their
associated agent settings.  :func:`create_app` wires them into a FastAPI
application with an OpenAI-compatible ``/v1/chat/completions`` endpoint.

Design notes
------------
* **Additive only** — no existing VEA source files are modified.
* **Graph config format** — same as ``framework.Graph.from_dict()``.
  Each entry in the registry can carry:

    * ``config``  – the graph dict (vertices, edges, metadata)
    * ``agent``   – agent kwargs (``base_url``, ``api_key``, ``model``, …)
    * ``fallback`` – list of model names to try on failure
    * ``stream``  – ``"events"`` (graph events as SSE) or
      ``"tokens"`` (direct ``stream_process()``)

* **Default graph** – if a request's ``model`` is not in the registry,
  the first registered graph is used.  If no graphs are registered, a
  minimal direct-LLM graph is created on the fly.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from ..agents.http_llm_agent import HttpLLMAgent
from ..agents.mock_agent import MockAgent
from ..graph import Graph
from ..executor import Executor
from .schemas import (
    ChatCompletionChunk,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChunkChoice,
    ChunkDelta,
    UsageInfo,
    _new_id,
)

logger = logging.getLogger("vertex_edge_agent.serve")

# ---------------------------------------------------------------------------
# Graph registry
# ---------------------------------------------------------------------------

class GraphRegistry:
    """Named registry of VEA graph configs + agent settings.

    Parameters
    ----------
    default_model : str
        Model name used when a request's ``model`` is not found.
    agent_defaults : dict
        Default agent kwargs (``base_url``, ``api_key``, …) shared across
        all graphs that don't specify their own.
    """

    def __init__(
        self,
        default_model: str = "default",
        agent_defaults: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.default_model = default_model
        self.agent_defaults = agent_defaults or {}
        self._graphs: Dict[str, Dict[str, Any]] = {}

    # -- registration ------------------------------------------------

    def add(
        self,
        name: str,
        config: Dict[str, Any],
        *,
        agent: Optional[Dict[str, Any]] = None,
        fallback: Optional[List[str]] = None,
        stream: str = "events",
        base_dir: Optional[str] = None,
        agent_cls: Any = None,
    ) -> "GraphRegistry":
        """Register a named graph.  Returns *self* for chaining.

        Parameters
        ----------
        agent_cls : optional
            Agent class to instantiate (defaults to :class:`HttpLLMAgent`).
            Pass ``MockAgent`` for testing.
        """
        entry: Dict[str, Any] = {
            "config": config,
            "agent": agent or {},
            "fallback": fallback or [],
            "stream": stream,
            "base_dir": base_dir,
            "agent_cls": agent_cls,
        }
        self._graphs[name] = entry
        logger.debug("[Registry] Added graph %r (%d vertices, %d edges)",
                     name, len(config.get("vertices", [])),
                     len(config.get("edges", [])))
        return self

    def add_from_json(self, name: str, json_path: str, **kwargs: Any) -> "GraphRegistry":
        """Register a graph from a JSON config file.

        Resolves ``script`` paths relative to the JSON file's directory.
        """
        path = Path(json_path)
        base_dir = str(path.parent)
        import json as _json
        with open(path, "r", encoding="utf-8") as fh:
            config = _json.load(fh)
        return self.add(name, config, base_dir=base_dir, **kwargs)

    # -- lookup ------------------------------------------------------

    def get(self, model: str) -> Optional[Dict[str, Any]]:
        """Return the registry entry for *model*, or ``None``."""
        return self._graphs.get(model)

    def resolve(self, model: str) -> Dict[str, Any]:
        """Return the registry entry for *model*, falling back to default."""
        entry = self._graphs.get(model)
        if entry is None:
            entry = self._graphs.get(self.default_model)
        if entry is None:
            raise KeyError(
                f"Model {model!r} not in registry and no default registered"
            )
        return entry

    @property
    def model_names(self) -> List[str]:
        """All registered model names."""
        return list(self._graphs.keys())

    def __len__(self) -> int:
        return len(self._graphs)

    def __bool__(self) -> bool:
        return len(self._graphs) > 0


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def _build_agent(
    registry: GraphRegistry,
    entry: Dict[str, Any],
    override_model: Optional[str] = None,
) -> HttpLLMAgent:
    """Construct an agent from a registry entry.

    If ``entry["agent_cls"]`` is set, uses that class; otherwise defaults
    to :class:`HttpLLMAgent`.
    """
    agent_cls = entry.get("agent_cls")
    if agent_cls is None:
        agent_cls = HttpLLMAgent

    if agent_cls is MockAgent:
        # MockAgent takes only response_fn, not kwargs
        return MockAgent()

    agent_cfg = {**registry.agent_defaults, **entry.get("agent", {})}
    model = override_model or agent_cfg.pop("model", None)
    return agent_cls(
        api_key=agent_cfg.pop("api_key", "public"),
        base_url=agent_cfg.pop("base_url", "https://api.openai.com/v1"),
        trust_env=agent_cfg.pop("trust_env", True),
        proxy=agent_cfg.pop("proxy", None),
        max_retries=agent_cfg.pop("max_retries", 3),
        timeout=float(agent_cfg.pop("timeout", 300.0)),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_messages(req: ChatCompletionRequest) -> List[Dict[str, str]]:
    """Convert request messages to the ``[{role, content}]`` list VEA expects."""
    return [
        {"role": m.role, "content": m.content or ""}
        for m in req.messages
    ]


def _make_usage(agent: Any) -> UsageInfo:
    """Build a :class:`UsageInfo` from an agent's token log."""
    if agent is None or not hasattr(agent, "get_usage_summary"):
        return UsageInfo()
    summary = agent.get_usage_summary()
    return UsageInfo(
        prompt_tokens=summary["prompt_tokens"],
        completion_tokens=summary["completion_tokens"],
        total_tokens=summary["total_tokens"],
    )


def _build_sse_chunk(
    request_id: str,
    model: str,
    content: Optional[str] = None,
    finish_reason: Optional[str] = None,
    created: Optional[int] = None,
) -> str:
    """Build a single OpenAI-format SSE ``data:`` line."""
    chunk = ChatCompletionChunk(
        id=request_id,
        created=created or int(time.time()),
        model=model,
        choices=[
            ChunkChoice(
                index=0,
                delta=ChunkDelta(content=content, role="assistant" if content is None and finish_reason is None else None),
                finish_reason=finish_reason,
            )
        ],
    )
    return chunk.to_sse()


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app(
    registry: Optional[GraphRegistry] = None,
    *,
    graphs: Optional[Dict[str, Any]] = None,
    default_model: str = "default",
    agent_defaults: Optional[Dict[str, Any]] = None,
    title: str = "VEA Serve API",
    version: str = "1.0.0",
) -> FastAPI:
    """Create a FastAPI application exposing OpenAI-compatible endpoints.

    Parameters
    ----------
    registry : GraphRegistry, optional
        Pre-built registry.  If provided, *graphs*, *default_model*, and
        *agent_defaults* are ignored.
    graphs : dict, optional
        Short-hand mapping ``{name: {"config": ..., "agent": ...}}``
        used when *registry* is not provided.
    default_model : str
        Fallback model name.
    agent_defaults : dict, optional
        Default agent kwargs shared across all graphs.
    title, version : str
        FastAPI metadata.

    Returns
    -------
    FastAPI
        A fully wired application with ``/v1/chat/completions`` and
        ``/v1/health`` endpoints.
    """
    # -- build registry -------------------------------------------------
    if registry is not None:
        reg = registry
    else:
        reg = GraphRegistry(
            default_model=default_model,
            agent_defaults=agent_defaults,
        )
        if graphs:
            for name, entry in graphs.items():
                reg.add(name, **entry)

    # -- lifespan --------------------------------------------------------
    _agents: Dict[str, HttpLLMAgent] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        # Build agents lazily on first request (per model name)
        yield
        # Close all agents on shutdown
        for agent in _agents.values():
            try:
                await agent.close()
            except Exception:
                pass
        _agents.clear()

    app = FastAPI(title=title, version=version, lifespan=lifespan)

    # -- /v1/health ------------------------------------------------------
    @app.get("/v1/health")
    async def health() -> JSONResponse:
        """Liveness probe."""
        return JSONResponse({
            "status": "ok",
            "models": reg.model_names,
            "graph_count": len(reg),
        })

    # -- /v1/chat/completions --------------------------------------------
    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest) -> Any:
        """OpenAI-compatible chat completions endpoint.

        Routes by ``req.model`` to the appropriate VEA graph.
        ``stream=True`` returns SSE chunks; otherwise returns a JSON
        :class:`ChatCompletionResponse`.
        """
        # --- resolve graph entry ----------------------------------------
        model_name = req.graph_id or req.model
        try:
            entry = reg.resolve(model_name)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

        messages = _extract_messages(req)
        if not messages:
            raise HTTPException(status_code=422, detail="messages must not be empty")
        request_id = _new_id()

        # --- streaming --------------------------------------------------
        if req.stream:
            return StreamingResponse(
                _stream_handler(reg, entry, messages, req, request_id),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # --- non-streaming ----------------------------------------------
        result = await _run_graph(
            reg, entry, messages, req, request_id
        )
        return result

    # -- helper: non-streaming execution ---------------------------------
    async def _run_graph(
        reg: GraphRegistry,
        entry: Dict[str, Any],
        messages: List[Dict[str, str]],
        req: ChatCompletionRequest,
        request_id: str,
    ) -> ChatCompletionResponse:
        """Execute the graph synchronously and return an OpenAI response."""
        fallback_models = list(entry.get("fallback") or [])
        attempts = [entry["model"] if "model" in entry else req.model] + fallback_models

        last_error: Optional[Exception] = None
        for attempt_model in attempts:
            try:
                agent = _build_agent(reg, entry, override_model=attempt_model)
                _agents[attempt_model] = agent
                content = await _execute(entry, messages, req, agent)
                usage = _make_usage(agent)
                return ChatCompletionResponse(
                    id=request_id,
                    model=req.model,
                    choices=[
                        ChatCompletionChoice(
                            index=0,
                            message=req.messages[-1].model_copy(update={
                                "role": "assistant",
                                "content": content,
                            }),
                            finish_reason="stop",
                        )
                    ],
                    usage=usage,
                )
            except Exception as exc:
                logger.warning("[Serve] Model %r failed: %s", attempt_model, exc)
                last_error = exc
                continue

        raise HTTPException(
            status_code=502,
            detail=f"All models failed. Last error: {last_error}",
        )

    # -- helper: streaming execution -------------------------------------
    async def _stream_handler(
        reg: GraphRegistry,
        entry: Dict[str, Any],
        messages: List[Dict[str, str]],
        req: ChatCompletionRequest,
        request_id: str,
    ) -> AsyncGenerator[str, None]:
        """Yield OpenAI-format SSE chunks for a streaming request."""
        fallback_models = list(entry.get("fallback") or [])
        attempts = [entry["model"] if "model" in entry else req.model] + fallback_models

        # Initial role chunk
        yield _build_sse_chunk(request_id, req.model, content=None, created=int(time.time()))

        for attempt_model in attempts:
            try:
                agent = _build_agent(reg, entry, override_model=attempt_model)
                _agents[attempt_model] = agent
                stream_mode = entry.get("stream", "events")

                if stream_mode == "tokens":
                    async for delta in _stream_tokens(reg, entry, messages, req, agent, request_id, req.model):
                        yield delta
                    break
                else:
                    async for chunk in _stream_events(entry, messages, req, agent, request_id, req.model):
                        yield chunk
                    break
            except Exception as exc:
                logger.warning("[Serve] Stream model %r failed: %s", attempt_model, exc)
                continue
        else:
            # All attempts failed — emit error chunk
            yield _build_sse_chunk(
                request_id, req.model,
                content="Error: all models failed",
                finish_reason="stop",
            )
            yield "data: [DONE]\n\n"
            return

        # Final DONE chunk
        yield _build_sse_chunk(request_id, req.model, finish_reason="stop")
        yield "data: [DONE]\n\n"

    # -- helper: token-level streaming -----------------------------------
    async def _stream_tokens(
        reg: GraphRegistry,
        entry: Dict[str, Any],
        messages: List[Dict[str, str]],
        req: ChatCompletionRequest,
        agent: Any,
        request_id: str,
        model: str,
    ) -> AsyncGenerator[str, None]:
        """Stream content deltas directly from the LLM via ``stream_process()``."""
        # The actual LLM model name comes from agent config (merged defaults + entry)
        agent_cfg = {**reg.agent_defaults, **entry.get("agent", {})}
        llm_model = agent_cfg.get("model", req.model)
        if not hasattr(agent, "stream_process"):
            # Fallback: non-streaming agent, emit the full result as one chunk
            prompt = ""
            data: Any = None
            if messages:
                if len(messages) == 1:
                    data = messages[0]["content"]
                else:
                    system_msg = next((m for m in messages if m["role"] == "system"), None)
                    user_msg = next((m for m in messages if m["role"] == "user"), None)
                    prompt = system_msg["content"] if system_msg else ""
                    data = user_msg["content"] if user_msg else messages[-1]["content"]
            result = await agent.process(data, prompt, llm_model, None)
            if isinstance(result, str):
                yield _build_sse_chunk(request_id, model, content=result)
            return

        prompt = ""
        data: Any = None
        if messages:
            if len(messages) == 1:
                data = messages[0]["content"]
            else:
                system_msg = next((m for m in messages if m["role"] == "system"), None)
                user_msg = next((m for m in messages if m["role"] == "user"), None)
                prompt = system_msg["content"] if system_msg else ""
                data = user_msg["content"] if user_msg else messages[-1]["content"]

        async for delta in agent.stream_process(data, prompt, llm_model, req.model_dump()):
            yield _build_sse_chunk(request_id, model, content=delta)

    # -- helper: event-level streaming -----------------------------------
    async def _stream_events(
        entry: Dict[str, Any],
        messages: List[Dict[str, str]],
        req: ChatCompletionRequest,
        agent: HttpLLMAgent,
        request_id: str,
        model: str,
    ) -> AsyncGenerator[str, None]:
        """Execute the graph and relay edge-completion events as SSE chunks."""
        graph = _build_graph(entry)
        # Inject the user's last message into the source vertex
        source = _find_source(graph)
        if source:
            user_content = messages[-1]["content"] if messages else ""
            await source.set_data("text", user_content)
        executor = Executor(graph=graph, agents=agent, timeout=300.0)

        full_content = ""
        async for event in executor.stream():
            if event.event_type == "edge_completed":
                payload = event.payload or {}
                result = payload.get("result")
                if isinstance(result, str) and result:
                    full_content = result
                    yield _build_sse_chunk(request_id, model, content=result)
                elif result:
                    yield _build_sse_chunk(request_id, model, content=str(result)[:500])
            elif event.event_type == "workflow_finished":
                if not full_content:
                    # Fallback: pull result from sink vertex via executor._result
                    for vid, info in executor._result.vertex_results.items():
                        v = graph.vertices.get(vid)
                        if v and v.settings.get("type") == "sink":
                            for key, val in info.get("data", {}).items():
                                if isinstance(val, str) and val:
                                    full_content = val
                                    break
                        if full_content:
                            break
                    if full_content:
                        yield _build_sse_chunk(request_id, model, content=full_content)
                break

    # -- helper: graph execution (non-streaming) -------------------------
    async def _execute(
        entry: Dict[str, Any],
        messages: List[Dict[str, str]],
        req: ChatCompletionRequest,
        agent: HttpLLMAgent,
    ) -> str:
        """Build the graph, inject messages, execute, and return the answer."""
        graph = _build_graph(entry)
        # Inject the user's last message into the source vertex's data store
        source = _find_source(graph)
        if source:
            user_content = messages[-1]["content"] if messages else ""
            await source.set_data("text", user_content)

        executor = Executor(graph=graph, agents=agent, timeout=300.0)
        result = await executor.run()

        if not result.success:
            raise RuntimeError(f"Graph execution failed: {result.errors}")

        # Extract answer from sink vertex via result
        return _extract_answer(graph, result)

    # -- helpers: graph construction -------------------------------------
    def _build_graph(entry: Dict[str, Any]) -> Graph:
        """Build a VEA Graph from a registry entry's config."""
        config = entry["config"]
        base_dir = entry.get("base_dir")
        # Deep-copy to avoid mutating the registry's stored config
        import copy
        config = copy.deepcopy(config)
        return Graph.from_dict(config, base_dir=base_dir)

    def _find_source(graph: Graph) -> Any:
        """Return the first source vertex (no incoming non-loop edges)."""
        for v in graph.vertices.values():
            non_loop_incoming = [
                eid for eid in v.incoming_edges
                if eid not in v.loop_incoming_edges
            ]
            if not non_loop_incoming:
                return v
        return None

    def _extract_answer(graph: Graph, result: Any) -> str:
        """Pull the answer string from sink vertices via ExecutionResult."""
        for vid, info in result.vertex_results.items():
            v = graph.vertices.get(vid)
            if v and v.settings.get("type") == "sink":
                data = info.get("data", {})
                for key, val in data.items():
                    if isinstance(val, str) and val:
                        return val
                    if isinstance(val, dict):
                        # Standard content keys, plus MockAgent's "output"
                        content = (
                            val.get("content")
                            or val.get("answer")
                            or val.get("output")
                            or ""
                        )
                        if content:
                            return content
        # Fallback: edge results
        for eid, val in result.edge_results.items():
            if isinstance(val, str) and val:
                return val
        return ""

    return app


# ---------------------------------------------------------------------------
# Module-level convenience: a default app for uvicorn
# ---------------------------------------------------------------------------

def _build_default_app() -> FastAPI:
    """Build a default app that reads config from environment variables."""
    import os

    # Read LLM config from env
    agent_defaults = {
        "base_url": os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
        "api_key": os.environ.get("OPENAI_API_KEY", "public"),
        "model": os.environ.get("LLM_MODEL", "default"),
        "trust_env": True,
    }

    # Build a minimal default graph if none is configured
    default_graph = {
        "metadata": {"name": "default_llm"},
        "vertices": [
            {"id": "source", "settings": {"type": "source"},
             "initial_data": [{"channel": "text", "value": ""}]},
            {"id": "sink", "settings": {"type": "sink"}},
        ],
        "edges": [
            {"id": "e_llm", "source": "source", "destination": "sink",
             "channel": "text", "concurrency_type": "llm",
             "settings": {"prompt": "", "model": "default"}},
        ],
    }

    reg = GraphRegistry(default_model="default", agent_defaults=agent_defaults)
    reg.add("default", default_graph)

    return create_app(registry=reg, agent_defaults=agent_defaults)


# Expose for ``uvicorn framework.serve.app:app``
app = _build_default_app()
