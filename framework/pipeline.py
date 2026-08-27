"""EdgePipeline module — 5-stage computation pipeline for an Edge.

Separates execution responsibility from routing configuration (Edge),
following the Single Responsibility Principle.

Stages (run in order by :meth:`run`):
    0. Upstream abort check
    1. Fetch data from source vertex
    2. Guard  (``evaluate_condition``)
    3. Pre-process hook
    4. Compute  (LLM agent or transparent pass-through)
    5. Post-process hook
    6. Deliver to destination vertex
"""

import asyncio
import logging
import time
from typing import Any, Dict, Optional

from .vertex import VertexState, EdgeSignal

logger = logging.getLogger("vertex_edge_agent.pipeline")


class EdgePipeline:
    """Encapsulates the 5-stage data transformation pipeline for an Edge.

    Args:
        prompt:         Prompt forwarded to the LLM agent.
        model:          Model identifier (e.g. ``"gemini-pro"``).
        settings:       Arbitrary settings dict (guards, hooks, LLM kwargs …).
        script_module:  Optional loaded external Python module with hook
                        functions (``evaluate_condition``, ``pre_process``,
                        ``post_process``).
        hook_provider:  Optional object (typically the owning :class:`Edge`
                        subclass instance) that may define overridable hook
                        methods: ``condition``, ``pre_process``,
                        ``post_process``.  Checked *before* the script module.

    Attributes:
        completed:     ``True`` if the pipeline completed successfully.
        aborted:       ``True`` if aborted by a guard or upstream abort.
        abort_reason:  Human-readable description of the abort cause.
        result:        Final value delivered to the destination vertex.
        error:         Error message string if the pipeline raised.
    """

    def __init__(
        self,
        prompt: str,
        model: str,
        settings: Dict,
        script_module=None,
        hook_provider=None,
    ):
        self.prompt = prompt
        self.model = model
        self.settings = settings

        self._script_module = script_module
        # hook_provider is checked for: condition, pre_process, post_process
        self._hook_provider = hook_provider

        # Execution state
        self.completed: bool = False
        self.aborted: bool = False
        self.abort_reason: Optional[str] = None
        self.result: Any = None
        self.error: Optional[str] = None

    # ------------------------------------------------------------------
    # Script module
    # ------------------------------------------------------------------
    def set_script_module(self, module) -> None:
        """Attach a loaded external script module."""
        self._script_module = module
        logger.debug("[Pipeline] Script module attached: %s", module)

    # ------------------------------------------------------------------
    # Stage 2 — Guard
    # ------------------------------------------------------------------
    def evaluate_condition(self, data: Any) -> bool:
        """Return ``True`` if the guard condition allows *data* through.

        Resolution order:
        1. ``hook_provider.condition(data, settings)`` — subclass override.
        2. Script module functions searched in order:
           ``evaluate_condition``, ``condition``, ``on_gate``, ``guard``.
        3. Declarative settings keys: ``condition`` callable, ``match`` dict,
           or ``threshold`` / ``operator`` / ``field`` combination.
        4. Default: ``True`` (no guard configured).
        """
        # 1. Hook provider (typically an Edge subclass defining `condition`)
        if self._hook_provider is not None:
            hook = getattr(self._hook_provider, "condition", None)
            if callable(hook):
                return bool(hook(data, self.settings))

        # 2. Script module hooks
        if self._script_module:
            for hook_name in ("evaluate_condition", "condition", "on_gate", "guard"):
                hook = getattr(self._script_module, hook_name, None)
                if callable(hook):
                    return bool(hook(data, self.settings))

        # 3. Declarative settings
        if not self.settings:
            return True

        if "condition" in self.settings and callable(self.settings["condition"]):
            return bool(self.settings["condition"](data))

        if (
            "match" in self.settings
            and isinstance(self.settings["match"], dict)
            and isinstance(data, dict)
        ):
            return all(data.get(k) == v for k, v in self.settings["match"].items())

        if "threshold" in self.settings:
            threshold = self.settings["threshold"]
            op = str(self.settings.get("operator", "==")).lower()
            val = data
            if isinstance(data, dict) and "field" in self.settings:
                val = data.get(self.settings["field"])

            try:
                if op in (">", "gt"):
                    return val > threshold
                elif op in (">=", "gte", "ge"):
                    return val >= threshold
                elif op in ("<", "lt"):
                    return val < threshold
                elif op in ("<=", "lte", "le"):
                    return val <= threshold
                elif op in ("==", "eq"):
                    return val == threshold
                elif op in ("!=", "ne"):
                    return val != threshold
                elif op == "in":
                    return val in threshold
                elif op == "contains":
                    return threshold in val
            except Exception as exc:
                logger.warning("[Pipeline] Threshold comparison failed: %s", exc)
                return False

        # Settings present but no guard condition matched → pass through
        return True

    # ------------------------------------------------------------------
    # Stage 3 — Pre-process helper
    # ------------------------------------------------------------------
    async def _run_pre_process(self, edge_id: str, data: Any) -> Any:
        """Dispatch the pre-process hook (hook_provider → script_module). Supports async."""
        import asyncio
        if self._hook_provider is not None:
            hook = getattr(self._hook_provider, "pre_process", None)
            if callable(hook):
                data = hook(data, self.settings)
                if asyncio.iscoroutine(data):
                    data = await data
                return data

        if self._script_module is not None:
            hook = getattr(self._script_module, "pre_process", None)
            if callable(hook):
                data = hook(data, self.settings)
                if asyncio.iscoroutine(data):
                    data = await data
                return data
        return data

        if self._script_module:
            hook = getattr(self._script_module, "pre_process", None)
            if callable(hook):
                data = hook(data, self.settings)
                logger.debug(
                    "[Pipeline:%s] After module.pre_process: %s",
                    edge_id, repr(data)[:200],
                )

        return data

    # ------------------------------------------------------------------
    # Stage 5 — Post-process helper
    # ------------------------------------------------------------------
    async def _run_post_process(self, edge_id: str, result: Any) -> Any:
        """Dispatch the post-process hook (hook_provider → script_module). Supports async."""
        import asyncio
        if self._hook_provider is not None:
            hook = getattr(self._hook_provider, "post_process", None)
            if callable(hook):
                result = hook(result, self.settings)
                if asyncio.iscoroutine(result):
                    result = await result
                return result

        if self._script_module is not None:
            hook = getattr(self._script_module, "post_process", None)
            if callable(hook):
                result = hook(result, self.settings)
                if asyncio.iscoroutine(result):
                    result = await result
                return result
        return result

        if self._script_module:
            hook = getattr(self._script_module, "post_process", None)
            if callable(hook):
                result = hook(result, self.settings)
                logger.debug(
                    "[Pipeline:%s] After module.post_process: %s",
                    edge_id, repr(result)[:200],
                )

        return result

    # ------------------------------------------------------------------
    # Main runner — all 6 stages in sequence
    # ------------------------------------------------------------------
    async def run(
        self,
        edge_id: str,
        source_id: str,
        destination_id: str,
        channel: str,
        source_vertex,
        dest_vertex,
        agents,
        **kwargs,
    ) -> Any:
        """Execute all pipeline stages sequentially and return the final result.

        Returns:
            The final result delivered to *dest_vertex*, or ``None`` if aborted.

        Raises:
            Any exception raised during compute or hook stages (after recording
            it in :attr:`error` and signalling FAILED to *dest_vertex*).
        """
        logger.info(
            "[Pipeline:%s] RUN  %s -[%s]-> %s",
            edge_id, source_id, channel, destination_id,
        )

        try:
            # 0 — Upstream abort propagation
            if (
                hasattr(source_vertex, "state")
                and source_vertex.state == VertexState.ABORTED
            ):
                self.aborted = True
                self.abort_reason = (
                    f"Upstream source vertex '{source_id}' is ABORTED"
                )
                logger.info(
                    "[Pipeline:%s] Source ABORTED -> notifying '%s'",
                    edge_id, destination_id,
                )
                await dest_vertex.receive_signal(
                    edge_id, EdgeSignal.ABORTED, payload=self.abort_reason
                )
                return None

            # 1 — Fetch data from source
            data = await source_vertex.fetch_data(channel=channel)
            logger.debug(
                "[Pipeline:%s] Source data: %s", edge_id, repr(data)[:200]
            )

            # 2 — Guard
            if not self.evaluate_condition(data):
                self.aborted = True
                self.abort_reason = (
                    f"Guard condition not satisfied on edge '{edge_id}'"
                )
                logger.info(
                    "[Pipeline:%s] Guard NOT satisfied -> ABORTING (dest: '%s')",
                    edge_id, destination_id,
                )
                await dest_vertex.receive_signal(
                    edge_id, EdgeSignal.ABORTED, payload=self.abort_reason
                )
                return None

            # 3 — Pre-process
            data = await self._run_pre_process(edge_id, data)
            # --- Memory reads (Pillar A) ---
            memory = kwargs.get("memory")
            if memory and isinstance(self.settings, dict):
                mem_reads = self.settings.get("memory_read", [])
                if mem_reads:
                    if isinstance(data, dict):
                        for m_key in mem_reads:
                            data[m_key] = await memory.get(m_key)
                    else:
                        mem_context = {m_key: await memory.get(m_key) for m_key in mem_reads}
                        logger.debug("[Pipeline:%s] Injected memory reads: %s", edge_id, mem_context)

            # Retry configuration (business logic / self-correction retry)
            retry_policy = self.settings.get("retry_policy", {}) if isinstance(self.settings, dict) else {}
            max_retries = int(retry_policy.get("max_retries", 0))
            backoff_factor = float(retry_policy.get("backoff_factor", 1.0))
            retry_on = retry_policy.get("retry_on", None)  # list of exception class names or None (all)

            attempt = 0
            current_prompt = self.prompt
            base_prompt = self.prompt

            t_compute_start = time.monotonic()
            prompt_tokens_est = 0
            completion_tokens_est = 0

            while True:
                try:
                    # 4 — Compute (LLM agent or transparent pass-through)
                    if current_prompt or (self.model and self.model != "default"):
                        result = await agents.process(
                            data=data,
                            prompt=current_prompt,
                            model=self.model,
                            settings=self.settings,
                        )
                        from .telemetry import estimate_tokens
                        prompt_tokens_est += estimate_tokens(str(current_prompt) + str(data))
                        completion_tokens_est += estimate_tokens(str(result))
                    else:
                        result = data  # Pass-through

                    # 5 — Post-process (can raise ValueError, KeyError, JSONDecodeError, etc.)
                    result = await self._run_post_process(edge_id, result)
                    
                    # ── Runtime Schema Validation (Pydantic) ────────────
                    out_schema_name = self.settings.get("output_schema") if isinstance(self.settings, dict) else None
                    if out_schema_name:
                        from .schema import SchemaRegistry
                        schema_model = SchemaRegistry.get(out_schema_name)
                        if schema_model:
                            # Attempt to validate and parse the dictionary into the Pydantic model
                            # If it fails, Pydantic throws a ValidationError which is caught by our retry logic!
                            if hasattr(schema_model, "model_validate"):
                                validated_obj = schema_model.model_validate(result) # Pydantic v2
                            else:
                                validated_obj = schema_model.parse_obj(result) # Pydantic v1
                            # Convert back to dict for standard downstream transmission
                            result = validated_obj.model_dump()
                            
                    break

                except Exception as compute_or_hook_exc:
                    attempt += 1
                    should_retry = False
                    if attempt <= max_retries:
                        if retry_on is None:
                            should_retry = True
                        else:
                            exc_name = compute_or_hook_exc.__class__.__name__
                            should_retry = exc_name in retry_on or any(
                                issubclass(compute_or_hook_exc.__class__, base)
                                for base in [Exception]
                                if getattr(base, "__name__", "") in retry_on
                            )

                    if should_retry:
                        delay = backoff_factor * (2 ** (attempt - 1))
                        err_type = compute_or_hook_exc.__class__.__name__
                        err_detail = str(compute_or_hook_exc)
                        logger.warning(
                            "[Pipeline:%s] Business retry attempt %d/%d after %s: %s. Backing off for %.2fs...",
                            edge_id, attempt, max_retries, err_type, err_detail, delay,
                        )
                        # Inject feedback into prompt for self-correction
                        feedback = (
                            f"\n\n[SYSTEM FEEDBACK: Your previous output produced a {err_type}: {err_detail}.\n"
                            f"Please correct your response and provide a valid output format.]"
                        )
                        current_prompt = base_prompt + feedback
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            "[Pipeline:%s] FAILED (no more retries): %s",
                            edge_id, compute_or_hook_exc, exc_info=True,
                        )
                        raise compute_or_hook_exc

            # --- Record Telemetry (Pillar B) ---
            telemetry = kwargs.get("telemetry")
            if telemetry:
                latency_ms = (time.monotonic() - t_compute_start) * 1000.0
                telemetry.record_edge(
                    edge_id=edge_id,
                    prompt_tokens=prompt_tokens_est,
                    completion_tokens=completion_tokens_est,
                    model=self.model,
                    latency_ms=latency_ms,
                )

            # --- Memory writes (Pillar A) ---
            if memory and isinstance(self.settings, dict):
                mem_writes = self.settings.get("memory_write", {})
                if isinstance(mem_writes, dict):
                    for src_field, target_mem_key in mem_writes.items():
                        if isinstance(result, dict) and src_field in result:
                            await memory.set(target_mem_key, result[src_field])
                        else:
                            await memory.set(target_mem_key, result)

            # 6 — Deliver
            await dest_vertex.receive_signal(
                edge_id, EdgeSignal.COMPLETED, payload=result, channel=channel
            )
            logger.info(
                "[Pipeline:%s] Delivered to '%s' via channel='%s'",
                edge_id, destination_id, channel,
            )

            self.completed = True
            self.result = result
            return result

        except Exception as exc:
            self.error = str(exc)
            logger.error(
                "[Pipeline:%s] FAILED: %s", edge_id, exc, exc_info=True
            )
            await dest_vertex.receive_signal(
                edge_id, EdgeSignal.FAILED, payload=str(exc)
            )
            raise

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset execution state so this pipeline can be re-run."""
        self.completed = False
        self.aborted = False
        self.abort_reason = None
        self.result = None
        self.error = None

    def __repr__(self) -> str:
        status = (
            "✓" if self.completed
            else ("⊘" if self.aborted
            else ("✗" if self.error else "·"))
        )
        return f"EdgePipeline(model={self.model!r} [{status}])"
