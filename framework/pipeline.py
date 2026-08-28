"""Pipeline module — 5-stage computation pipeline for an Edge.

Separates execution responsibility from routing configuration (Edge),
following the Single Responsibility Principle.

Stages (run in order by :meth:`run`):
    0. Upstream abort check (Done by Edge now)
    1. Fetch data from source vertex (Done by Edge now)
    2. Guard  (``evaluate_condition``)
    3. Pre-process hook
    4. Compute  (LLM agent or transparent pass-through)
    5. Post-process hook
    6. Deliver to destination vertex (Done by Edge now)
"""

import asyncio
import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger("vertex_edge_agent.pipeline")

from .utils.errors import AbortPipeline, GuardAbortError, HookError, ComputeError

class Pipeline:
    def __init__(
        self,
        prompt: str,
        model: str,
        settings: Dict,
        pipeline_module=None,
        hook_provider=None,
        agent=None,
        log_id: str = "Unknown"
    ):
        self.prompt = prompt
        self.model = model
        self.settings = settings
        self.pipeline_module = pipeline_module
        self.hook_provider = hook_provider
        self.agent = agent
        self.log_id = log_id

    def evaluate_condition(self, data: Any) -> bool:
        if self.hook_provider:
            hook = getattr(self.hook_provider, "condition", None)
            if callable(hook):
                return bool(hook(data, self.settings))

        if self.pipeline_module:
            hook = getattr(self.pipeline_module, "condition", getattr(self.pipeline_module, "evaluate_condition", None))
            if callable(hook):
                return bool(hook(data, self.settings))

        if not self.settings:
            return True

        if "condition" in self.settings and callable(self.settings["condition"]):
            return bool(self.settings["condition"](data))

        if "match" in self.settings and isinstance(self.settings["match"], dict) and isinstance(data, dict):
            return all(data.get(k) == v for k, v in self.settings["match"].items())

        if "threshold" in self.settings:
            threshold = self.settings["threshold"]
            op = str(self.settings.get("operator", "==")).lower()
            val = data.get(self.settings["field"]) if isinstance(data, dict) and "field" in self.settings else data
            if val is None:
                return False
            import operator
            ops = {
                "==": operator.eq, "!=": operator.ne, ">": operator.gt,
                "<": operator.lt, ">=": operator.ge, "<=": operator.le, "contains": operator.contains
            }
            fn = ops.get(op, operator.eq)
            try:
                return fn(val, threshold)
            except TypeError:
                return False

        return True

    async def _run_pre_process(self, data: Any) -> Any:
        if self.hook_provider:
            hook = getattr(self.hook_provider, "pre_process", None)
            if callable(hook):
                data = await hook(data, self.settings) if asyncio.iscoroutinefunction(hook) else hook(data, self.settings)
        if self.pipeline_module:
            hook = getattr(self.pipeline_module, "pre_process", None)
            if callable(hook):
                data = await hook(data, self.settings) if asyncio.iscoroutinefunction(hook) else hook(data, self.settings)
        return data

    async def _run_post_process(self, result: Any) -> Any:
        if self.hook_provider:
            hook = getattr(self.hook_provider, "post_process", None)
            if callable(hook):
                result = await hook(result, self.settings) if asyncio.iscoroutinefunction(hook) else hook(result, self.settings)
        if self.pipeline_module:
            hook = getattr(self.pipeline_module, "post_process", None)
            if callable(hook):
                result = await hook(result, self.settings) if asyncio.iscoroutinefunction(hook) else hook(result, self.settings)
        return result

    async def run(self, data: Any, agents=None, memory=None, telemetry=None) -> Any:
        logger.debug("[Pipeline:%s] START compute", self.log_id)
        
        if not self.evaluate_condition(data):
            raise AbortPipeline(f"Guard condition not satisfied on '{self.log_id}'")
            
        data = await self._run_pre_process(data)
        
        if memory and isinstance(self.settings, dict):
            mem_reads = self.settings.get("memory_read", [])
            if mem_reads:
                if isinstance(data, dict):
                    for m_key in mem_reads:
                        data[m_key] = await memory.get(m_key)
                else:
                    mem_context = {m_key: await memory.get(m_key) for m_key in mem_reads}
                    logger.debug("[Pipeline:%s] Injected memory reads: %s", self.log_id, mem_context)
                    
        retry_policy = self.settings.get("retry_policy", {}) if isinstance(self.settings, dict) else {}
        max_retries = int(retry_policy.get("max_retries", 0))
        backoff_factor = float(retry_policy.get("backoff_factor", 1.0))
        retry_on_exc = retry_policy.get("retry_on", ["Exception"])
        
        current_prompt = self.prompt
        base_prompt = self.prompt
        
        t_compute_start = time.monotonic()
        prompt_tokens_est = 0
        completion_tokens_est = 0
        
        attempt = 0
        while True:
            try:
                if current_prompt or (self.model and self.model != "default"):
                    from .agents import MockAgent
                    active_agent = self.agent or agents or MockAgent()
                    edge_timeout = float(self.settings.get("timeout", 0)) if isinstance(self.settings, dict) else 0
                    if edge_timeout > 0:
                        result = await asyncio.wait_for(
                            active_agent.process(
                                data=data,
                                prompt=current_prompt,
                                model=self.model,
                                settings=self.settings,
                            ),
                            timeout=edge_timeout,
                        )
                    else:
                        result = await active_agent.process(
                            data=data,
                            prompt=current_prompt,
                            model=self.model,
                            settings=self.settings,
                        )
                    from .utils.telemetry import estimate_tokens
                    prompt_tokens_est += estimate_tokens(str(current_prompt) + str(data))
                    completion_tokens_est += estimate_tokens(str(result))
                else:
                    result = data

                result = await self._run_post_process(result)
                
                out_schema_name = self.settings.get("output_schema") if isinstance(self.settings, dict) else None
                if out_schema_name:
                    from .utils.schema import SchemaRegistry
                    schema_model = SchemaRegistry.get(out_schema_name)
                    if schema_model:
                        try:
                            if hasattr(schema_model, "model_validate"):
                                validated_obj = schema_model.model_validate(result)
                            else:
                                validated_obj = schema_model.parse_obj(result)
                            result = validated_obj.model_dump()
                        except Exception as e:
                            raise e
                        
                break
                
            except Exception as compute_or_hook_exc:
                err_name = compute_or_hook_exc.__class__.__name__
                matched = False
                for r_exc in retry_on_exc:
                    if r_exc == "Exception" or r_exc == err_name:
                        matched = True
                        break
                
                if matched and attempt < max_retries:
                    attempt += 1
                    delay = backoff_factor * (2 ** (attempt - 1))
                    err_detail = str(compute_or_hook_exc)
                    logger.warning(
                        "[Pipeline:%s] Business retry attempt %d/%d after %s: %s. Backing off for %.2fs...",
                        self.log_id, attempt, max_retries, err_name, err_detail, delay,
                    )
                    feedback = (
                        f"\n\n[SYSTEM FEEDBACK: Your previous output produced a {err_name}: {err_detail}.\n"
                        f"Please correct your response and provide a valid output format.]"
                    )
                    current_prompt = base_prompt + feedback
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "[Pipeline:%s] FAILED (no more retries): %s",
                        self.log_id, compute_or_hook_exc, exc_info=True,
                    )
                    raise compute_or_hook_exc

        if telemetry:
            latency_ms = (time.monotonic() - t_compute_start) * 1000.0
            telemetry.record_edge(
                edge_id=self.log_id,
                prompt_tokens=prompt_tokens_est,
                completion_tokens=completion_tokens_est,
                model=self.model,
                latency_ms=latency_ms,
            )

        if memory and isinstance(self.settings, dict):
            mem_writes = self.settings.get("memory_write", {})
            if mem_writes:
                for src_key, target_mem_key in mem_writes.items():
                    if isinstance(result, dict) and src_key in result:
                        await memory.set(target_mem_key, result[src_key])
                    elif src_key == "$":
                        await memory.set(target_mem_key, result)

        return result
