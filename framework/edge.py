"""Edge module — Routing connection + computation pipeline between vertices.

Background:
Previously, the architecture was split into Edge (routing), Pipeline (5-stage orchestration), and Agent (LLM backend).
Pipeline was a stateless pseudo-class created afresh on every execute call, simply copying fields from Edge.
Orchestration logic (guard, pre-process, compute, retry/timeout, post-process, schema validation, memory, telemetry)
has now been unified directly within Edge.

Subclasses can override hooks:
- condition(data, settings) -> bool: Guard stage evaluation.
- pre_process(data, settings) -> data: Pre-computation data transformation.
- post_process(result, settings) -> result: Post-computation data transformation.
- compute(data, agent, settings) -> result: Core single computation.
- execute(source_vertex, dest_vertex, agents, **kwargs): Full execution override.
"""

import asyncio
import logging
import time
from typing import Any, Dict, Optional, Union

from .agents import MockAgent, BaseAgent
from .utils.errors import AbortPipeline, GuardAbortError, HookError, ComputeError

logger = logging.getLogger("vertex_edge_agent.edge")


class Edge:
    """Directed edge connecting a source vertex to a destination vertex.

    Attributes (Routing layer):
        id, source_id, destination_id, channel

    Attributes (Scheduling layer):
        concurrency_type, max_iterations

    Attributes (Computation layer, parsed from settings):
        prompt, model, retry_policy, timeout, output_schema,
        memory_read, memory_write, _match, _threshold, _operator, _field

    Subclass hooks (override to customise computation):
        condition(data, settings) -> bool        — guard stage
        pre_process(data, settings) -> data      — pre-compute transform
        post_process(result, settings) -> result — post-compute transform
        compute(data, agent, settings) -> result — single computation (LLM or pure Python)
    """

    def __init__(
        self,
        edge_id: str,
        source_id: str,
        destination_id: str,
        channel: str = "default",
        settings: Optional[Dict] = None,
        concurrency_type: str = "default",
        max_iterations: int = 0,
    ):
        # ── Routing layer ──
        self.id = edge_id
        self.source_id = source_id
        self.destination_id = destination_id
        self.channel = channel

        # ── Scheduling layer ──
        self.concurrency_type = concurrency_type
        self.max_iterations = max_iterations

        # ── Execution state ──
        self.completed: bool = False
        self.aborted: bool = False
        self.abort_reason: Optional[str] = None
        self.result: Any = None
        self.error: Optional[str] = None

        # ── Computation layer: parsed from settings dict ──
        s = settings or {}
        self.settings = s
        self.prompt = s.get("prompt", "")
        self._base_prompt = self.prompt
        self.model = s.get("model", "default")
        # ``skip_compute`` — offline / pure-data edge: skip the LLM compute
        # stage entirely. pre_process output flows straight to post_process
        # (config: ``"settings": {"skip_compute": true}``). Generic edges
        # (fetch/parse/transform) use this instead of a mock agent.
        self.skip_compute = bool(s.get("skip_compute", False))
        # No per-edge agent from config. Script Edge subclasses may set their
        # own agent (e.g. ``self.agent = OpenCodeAgentRunner()`` in ``__init__``);
        # plain edges fall back to the executor agent, then MockAgent.
        self.agent = None
        self.retry_policy = s.get("retry_policy", {})
        self.timeout = float(s.get("timeout", 0))
        self.output_schema = s.get("output_schema")
        self.memory_read = s.get("memory_read", [])
        self.memory_write = s.get("memory_write", {})
        # Gate configuration for condition() default implementation
        self._match = s.get("match")
        self._threshold = s.get("threshold")
        self._operator = s.get("operator", "==")
        self._field = s.get("field")

        logger.debug(
            "[Edge:%s] Created %s -> %s | channel=%s model=%s",
            self.id, source_id, destination_id, self.channel, self.model,
        )

    # ==================================================================
    # Subclass hooks — override in subclasses to customise behaviour
    # ==================================================================


    def condition(self, data: Any, settings: Dict) -> bool:
        
        if "condition" in settings and callable(settings["condition"]):
            return bool(settings["condition"](data))
            
        if "guard" in settings:
            from .utils.guard import build_guard
            return build_guard(settings["guard"]).evaluate(data)

        if self._match is not None and isinstance(self._match, dict) and isinstance(data, dict):
            return all(data.get(k) == v for k, v in self._match.items())

        if self._threshold is not None:
            val = data.get(self._field) if isinstance(data, dict) and self._field else data
            if val is None:
                return False
            import operator
            ops = {
                "==": operator.eq, "!=": operator.ne, ">": operator.gt,
                "<": operator.lt, ">=": operator.ge, "<=": operator.le,
                "contains": operator.contains,
            }
            fn = ops.get(str(self._operator).lower(), operator.eq)
            try:
                return fn(val, self._threshold)
            except TypeError:
                return False
        return True

    async def pre_process(self, data: Any, settings: Dict) -> Any:
        """Pre-compute transform. Default: passthrough.

        Can be synchronous or asynchronous (coroutines are automatically awaited).
        """
        return data

    async def post_process(self, result: Any, settings: Dict) -> Any:
        """Post-compute transform. Default: passthrough.

        Can be synchronous or asynchronous (coroutines are automatically awaited).
        """
        return result

    async def compute(self, data: Any, agent: Optional[BaseAgent], settings: Dict) -> Any:
        """Single computation. Default: call agent.process if prompt/model present, else passthrough.

        Agent precedence (most specific first):
        1. ``self.agent`` — set by a script ``Edge`` subclass (e.g. ``OpenCodeEdge``)
        2. ``agent``      — the executor-level agent passed in
        3. ``MockAgent()`` — the deterministic fallback
        """
        if self.prompt or (self.model and self.model != "default"):
            active_agent = self.agent or agent or MockAgent()
            if self.timeout > 0:
                result = await asyncio.wait_for(
                    active_agent.process(
                        data=data, prompt=self.prompt,
                        model=self.model, settings=self.settings,
                    ),
                    timeout=self.timeout,
                )
            else:
                result = await active_agent.process(
                    data=data, prompt=self.prompt,
                    model=self.model, settings=self.settings,
                )
            return result
        # No prompt/model: passthrough
        return data

    # ==================================================================
    # Orchestration — internal methods, subclasses rarely override
    # ==================================================================

    async def _run_pre_process(self, data: Any) -> Any:
        result = self.pre_process(data, self.settings)
        if asyncio.iscoroutine(result):
            result = await result
        return result

    async def _run_post_process(self, result: Any) -> Any:
        out = self.post_process(result, self.settings)
        if asyncio.iscoroutine(out):
            out = await out
        return out

    async def _run_compute(self, data: Any, agents, memory=None, telemetry=None) -> Any:
        """Compute with retry/backoff + memory reads/writes + schema validation + telemetry."""
        # ── Memory reads (before compute) ──
        if memory and self.memory_read:
            if isinstance(data, dict):
                for m_key in self.memory_read:
                    data[m_key] = await memory.get(m_key)
            else:
                mem_context = {m_key: await memory.get(m_key) for m_key in self.memory_read}
                logger.debug("[Edge:%s] Injected memory reads: %s", self.id, mem_context)

        # ── Retry loop ──
        retry_policy = self.retry_policy or {}
        max_retries = int(retry_policy.get("max_retries", 0))
        backoff_factor = float(retry_policy.get("backoff_factor", 1.0))
        retry_on_exc = retry_policy.get("retry_on", ["Exception"])

        t_compute_start = time.monotonic()
        prompt_tokens_est = 0
        completion_tokens_est = 0

        # ── Prompt state isolation ──
        # ``_base_prompt`` is the frozen template captured in ``__init__``.  Each
        # retry rebuilds the active prompt from it (single [SYSTEM FEEDBACK]
        # block) instead of appending onto ``self.prompt``, so self-correction
        # feedback can never accumulate into an irreversible stacked stack.
        # ``self.prompt`` is restored afterwards, keeping the edge reusable.
        orig_prompt = self.prompt
        active_prompt = self._base_prompt
        try:
            attempt = 0
            while True:
                try:
                    self.prompt = active_prompt
                    if self.skip_compute:
                        # No LLM call: pre_process output is the compute result.
                        # Post-process/schema/telemetry still run unchanged.
                        result = data
                    else:
                        result = await self.compute(data, agents, self.settings)
                    result = await self._run_post_process(result)

                    # ── Schema validation ──
                    if self.output_schema:
                        from .utils.schema import SchemaRegistry
                        schema_model = SchemaRegistry.get(self.output_schema)
                        if schema_model:
                            try:
                                if hasattr(schema_model, "model_validate"):
                                    validated_obj = schema_model.model_validate(result)
                                else:
                                    validated_obj = schema_model.parse_obj(result)
                                result = validated_obj.model_dump()
                            except Exception as e:
                                raise e

                    # Token estimation (for telemetry)
                    if telemetry:
                        from .utils.telemetry import estimate_tokens
                        prompt_tokens_est += estimate_tokens(str(active_prompt) + str(data))
                        completion_tokens_est += estimate_tokens(str(result))

                    break

                except Exception as compute_or_hook_exc:
                    err_name = compute_or_hook_exc.__class__.__name__
                    matched = any(
                        r_exc == "Exception" or r_exc == err_name
                        for r_exc in retry_on_exc
                    )

                    if matched and attempt < max_retries:
                        attempt += 1
                        delay = backoff_factor * (2 ** (attempt - 1))
                        err_detail = str(compute_or_hook_exc)
                        logger.warning(
                            "[Edge:%s] Business retry attempt %d/%d after %s: %s. Backing off for %.2fs...",
                            self.id, attempt, max_retries, err_name, err_detail, delay,
                        )
                        # Rebuild from the frozen template — a fresh, single
                        # feedback block referencing the latest error only.
                        active_prompt = (
                            f"{self._base_prompt}\n\n"
                            f"[SYSTEM FEEDBACK: Your previous output produced a {err_name}]: {err_detail}"
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            "[Edge:%s] FAILED (no more retries): %s",
                            self.id, compute_or_hook_exc, exc_info=True,
                        )
                        raise compute_or_hook_exc
        finally:
            self.prompt = orig_prompt

        # ── Telemetry record ──
        if telemetry:
            latency_ms = (time.monotonic() - t_compute_start) * 1000.0
            telemetry.record_edge(
                edge_id=self.id,
                prompt_tokens=prompt_tokens_est,
                completion_tokens=completion_tokens_est,
                model=self.model,
                latency_ms=latency_ms,
            )

        # ── Memory writes (after compute) ──
        if memory and self.memory_write:
            for src_key, target_mem_key in self.memory_write.items():
                if isinstance(result, dict) and src_key in result:
                    await memory.set(target_mem_key, result[src_key])
                elif src_key == "$":
                    await memory.set(target_mem_key, result)

        return result

    # ==================================================================
    # Public proxy — evaluate_condition alias for backwards compatibility
    # ==================================================================

    def evaluate_condition(self, data: Any, settings: Optional[Dict] = None) -> bool:
        """Public proxy to condition(). Kept for backward-compat with tests/external callers."""
        return self.condition(data, settings or self.settings)

    # ==================================================================
    # Execution
    # ==================================================================

    async def execute(self, source_vertex, dest_vertex, agents, **kwargs) -> Any:
        """5-stage execution: fetch → guard → pre_process → compute(retry/timeout) → deliver."""
        from .vertex import EdgeSignal

        memory = kwargs.get("memory")
        telemetry = kwargs.get("telemetry")

        # 1. Fetch data from source vertex
        data = await source_vertex.fetch_data(channel=self.channel)

        # 2. Guard — condition check
        if not self.condition(data, self.settings):
            self.aborted = True
            self.abort_reason = f"Guard condition not satisfied on '{self.id}'"
            await dest_vertex.receive_signal(
                self.id, EdgeSignal.ABORTED, payload=self.abort_reason
            )
            return None

        # 3. Pre-process hook
        data = await self._run_pre_process(data)

        # 4. Compute (retry/backoff/timeout/schema/telemetry/memory)
        try:
            result = await self._run_compute(data, agents, memory, telemetry)
        except Exception as exc:
            self.error = str(exc)
            await dest_vertex.receive_signal(
                self.id, EdgeSignal.FAILED, payload=str(exc)
            )
            raise

        # 5. Deliver result
        self.completed = True
        self.result = result
        await dest_vertex.receive_signal(
            self.id, EdgeSignal.COMPLETED, payload=result, channel=self.channel
        )
        return result

    # ==================================================================
    # Reset
    # ==================================================================

    def reset(self) -> None:
        """Reset edge execution state for re-runs."""
        self.completed = False
        self.aborted = False
        self.abort_reason = None
        self.result = None
        self.error = None
        self.prompt = self._base_prompt

    def __repr__(self) -> str:
        status = (
            "✓" if self.completed
            else ("⊘" if self.aborted
            else ("✗" if self.error else "·"))
        )
        return (
            f"{self.__class__.__name__}"
            f"({self.id} {self.source_id}->{self.destination_id} [{status}])"
        )

def edge_transform(pre=None, post=None, guard=None):
    class FunctionalEdge(Edge):
        pass
    if guard: FunctionalEdge.condition = staticmethod(guard)
    if pre: FunctionalEdge.pre_process = staticmethod(pre)
    if post: FunctionalEdge.post_process = staticmethod(post)
    return FunctionalEdge

class MapEdge(Edge):
    async def execute(self, source_vertex, dest_vertex, agents, **kwargs):
        from .vertex import EdgeSignal
        from .utils.script_loader import load_class_from_script
        import os
        
        self.aborted = False
        self.error = None
        
        items = await source_vertex.fetch_data(self.channel)
        if not isinstance(items, list):
            items = [items]
            
        if not self.condition(items, self.settings):
            self.aborted = True
            await dest_vertex.receive_signal(self.id, EdgeSignal.ABORTED)
            return None
            
        pipeline_config = self.settings.get("pipeline", [])
        sem = asyncio.Semaphore(self.settings.get("max_concurrency", 5))
        
        async def process_one(item):
            async with sem:
                res = item
                for step_idx, step_conf in enumerate(pipeline_config):
                    # Build a transient edge for this step
                    etype = step_conf.get("type", "default")
                    script = step_conf.get("script")
                    edge_cls = Edge
                    if script:
                        try:
                            if ":" in script:
                                path_part, cls_name = script.split(":", 1)
                                edge_cls = load_class_from_script(path_part, Edge, cls_name)
                            else:
                                edge_cls = load_class_from_script(script, Edge, Edge)
                        except Exception as e:
                            logger.error("Failed to load map step script %s: %s", script, e)
                            
                    step_edge = edge_cls(
                        edge_id=f"{self.id}_step{step_idx}",
                        source_id=self.source_id,
                        destination_id=self.destination_id,
                        channel=self.channel,
                        settings=step_conf.get("settings", step_conf) # fallback to step_conf
                    )
                    
                    # 1. pre_process
                    res = await step_edge._run_pre_process(res)
                    # 2. compute
                    res = await step_edge._run_compute(res, agents, kwargs.get("memory"), kwargs.get("telemetry"))
                    
                return res

        results = await asyncio.gather(
            *[process_one(item) for item in items],
            return_exceptions=True
        )
        
        self.completed = True
        self.result = []
        for r in results:
            if isinstance(r, Exception):
                logger.error("MapEdge %s item failed: %s", self.id, r)
            else:
                self.result.append(r)
                await dest_vertex.receive_signal(self.id, EdgeSignal.COMPLETED, r, self.channel)
                
        return self.result
