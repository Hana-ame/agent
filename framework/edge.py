"""Edge module — Routing connection + computation pipeline between vertices.

改造背景（Pipeline 降级为 Edge 内部方法）：
原架构 Edge(路由) → Pipeline(5 阶段编排) → Agent(LLM 调用) 三层。
Pipeline 是无状态、每次 execute 都 new、所有字段从 Edge 拷贝的"伪对象"——
本质是方法提取伪装成类。hook_provider=self 证明 hook 和 Edge 是一体的，
_get_pipeline_instance 无参实例化 Edge 子类会崩（Edge.__init__ 要构造参数），
证明 Pipeline 把 Edge 的 hook 拆成独立对象的设计是错的。

现在：Pipeline.run 的编排逻辑全部搬进 Edge。Edge 既是路由载体也是计算编排器。
hook 直接是 Edge 自身方法（condition/pre_process/post_process/compute），不绕中间层。
Agent 层保留不动（后端可插拔是真实需求）。

config 三层（见 graph.py）：
- 路由层（顶层显式）：id, source, destination, channel
- 调度层（顶层显式）：concurrency_type, max_iterations
- 计算层（settings dict → __init__ 一次性解析成 self 属性）：prompt/model/agent/
  match/threshold/operator/field/retry_policy/timeout/memory_read/memory_write/output_schema

三级 override 粒度（和 vertex 的 on_receive/on_ready 对称）：
- 轻：override condition / pre_process / post_process
- 中：override compute（换计算方式，retry/timeout 仍由 _run_compute 管）
- 重：override execute（完全自定义，连编排都不要）
"""

import asyncio
import logging
import time
from typing import Any, Dict, Optional, Union

from .agents import get_agent, MockAgent, BaseAgent
from .utils.errors import AbortPipeline, GuardAbortError, HookError, ComputeError

logger = logging.getLogger("vertex_edge_agent.edge")


class Edge:
    """Directed edge connecting a source vertex to a destination vertex.

    Attributes (路由层):
        id, source_id, destination_id, channel

    Attributes (调度层):
        concurrency_type, max_iterations

    Attributes (计算层, 从 settings 解析):
        prompt, model, agent, retry_policy, timeout, output_schema,
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
        # ── 路由层 ──
        self.id = edge_id
        self.source_id = source_id
        self.destination_id = destination_id
        self.channel = channel

        # ── 调度层 ──
        # concurrency_type 给 executor._fire_edge 选 semaphore 用，Edge 自身 execute 不读它。
        self.concurrency_type = concurrency_type
        self.max_iterations = max_iterations

        # ── 执行状态 ──
        self.completed: bool = False
        self.aborted: bool = False
        self.abort_reason: Optional[str] = None
        self.result: Any = None
        self.error: Optional[str] = None

        # ── 计算层：settings 一次性解析成属性，运行时不再 get ──
        # ⚠️ self.settings 保留原样 dict，传给 agent.process 和 hook 签名（子类可能塞任意 key，
        # 如 SelectPipeline 的 "index"）。提升的属性只是已知高频字段的便捷访问。
        s = settings or {}
        self.settings = s
        self.prompt = s.get("prompt", "")
        self.model = s.get("model", "default")
        # agent spec 可能是 str("mock"/"http"/"pi"/"path:Class") / dict / BaseAgent 实例 / None
        self.agent = get_agent(s.get("agent"))
        self.retry_policy = s.get("retry_policy", {})
        self.timeout = float(s.get("timeout", 0))
        self.output_schema = s.get("output_schema")
        self.memory_read = s.get("memory_read", [])
        self.memory_write = s.get("memory_write", {})
        # gate 配置：基类 condition 默认实现用（见 self.condition）
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
        """Guard stage. Default: settings-based match/threshold/operator/field logic.

        子类 override 此方法可做完全自定义的判断（如 SelectPipeline 检查 index < len）。
        基类默认实现从 __init__ 解析的 self._match/_threshold/_operator/_field 读取，
        兼容原 pipeline.evaluate_condition 的 settings-based gate 逻辑（test_gate_edge 依赖）。
        """
        # settings 里可能有 callable condition（运行时传入的动态判断）
        if "condition" in settings and callable(settings["condition"]):
            return bool(settings["condition"](data))

        # match: dict 字段全匹配
        if self._match is not None and isinstance(self._match, dict) and isinstance(data, dict):
            return all(data.get(k) == v for k, v in self._match.items())

        # threshold: 字段值与阈值做 operator 比较
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

        纯 Python edge（如 FetchThreadsPipeline）override 此方法做 httpx 抓取/文本变换。
        支持同步或异步：返回 coroutine 时自动 await。
        """
        return data

    async def post_process(self, result: Any, settings: Dict) -> Any:
        """Post-compute transform. Default: passthrough.

        纯 Python edge（如 FilterPipeline）override 此方法做 JSON 解析/正则提取。
        支持同步或异步：返回 coroutine 时自动 await。
        """
        return result

    async def compute(self, data: Any, agent: Optional[BaseAgent], settings: Dict) -> Any:
        """Single computation. Default: call agent.process if prompt/model present, else passthrough.

        ⚠️ 纯 Python edge 不 override 此方法——靠 pre_process/post_process 做全部变换，
        compute 默认透传 data（因为无 prompt/model）。这是 S1 的 FetchThreads/Select/Fetch 的模式。
        混合 edge（Filter/Summarize）也不 override compute——默认实现会调 LLM agent。
        子类 override compute 用于完全自定义计算（如不用 LLM 的纯 Python 变换，
        或换一种 agent 调用方式）。retry/timeout 由 _run_compute 统一包裹，子类不用管。
        """
        if self.prompt or (self.model and self.model != "default"):
            active_agent = agent or MockAgent()
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
        # 无 prompt/model：透传（纯 Python edge 靠 pre/post 处理）
        return data

    # ==================================================================
    # Orchestration — internal methods, subclasses rarely override
    # ==================================================================

    async def _run_pre_process(self, data: Any) -> Any:
        """Call self.pre_process, awaiting if it returns a coroutine."""
        result = self.pre_process(data, self.settings)
        if asyncio.iscoroutine(result):
            result = await result
        return result

    async def _run_post_process(self, result: Any) -> Any:
        """Call self.post_process, awaiting if it returns a coroutine."""
        out = self.post_process(result, self.settings)
        if asyncio.iscoroutine(out):
            out = await out
        return out

    async def _run_compute(self, data: Any, agents, memory=None, telemetry=None) -> Any:
        """Compute with retry/backoff + memory reads/writes + schema validation + telemetry.

        从原 pipeline.py:117-243 Pipeline.run 搬来。retry 包裹的是 compute + post_process +
        schema 验证（和原逻辑一致：post_process 和 schema 在 retry 循环内，pre_process 在外）。
        """
        # ── Memory reads (compute 前) ──
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

        attempt = 0
        while True:
            try:
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

                # token 估算（telemetry 用）
                if telemetry:
                    from .utils.telemetry import estimate_tokens
                    prompt_tokens_est += estimate_tokens(str(self.prompt) + str(data))
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
                    # 把错误反馈追加到 prompt（原 pipeline 逻辑），子类可在 compute 里读 self.prompt
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "[Edge:%s] FAILED (no more retries): %s",
                        self.id, compute_or_hook_exc, exc_info=True,
                    )
                    raise compute_or_hook_exc

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

        # ── Memory writes (compute 后) ──
        if memory and self.memory_write:
            for src_key, target_mem_key in self.memory_write.items():
                if isinstance(result, dict) and src_key in result:
                    await memory.set(target_mem_key, result[src_key])
                elif src_key == "$":
                    await memory.set(target_mem_key, result)

        return result

    # ==================================================================
    # Public proxy — condition 给外部测试/调用用（evaluate_condition 别名兼容）
    # ==================================================================

    def evaluate_condition(self, data: Any, settings: Optional[Dict] = None) -> bool:
        """Public proxy to condition(). Kept for backward-compat with tests/external callers."""
        return self.condition(data, settings or self.settings)

    # ==================================================================
    # Execution
    # ==================================================================

    async def execute(self, source_vertex, dest_vertex, agents, **kwargs) -> Any:
        """5-stage execution: fetch → guard → pre_process → compute(retry/timeout) → deliver.

        kwargs 接收 executor 传入的 memory= / telemetry=（向后兼容原 Pipeline.run 签名）。
        """
        from .vertex import EdgeSignal

        memory = kwargs.get("memory")
        telemetry = kwargs.get("telemetry")

        # 1. Fetch data from source vertex
        data = await source_vertex.fetch_data(channel=self.channel)

        # 2. Guard — condition 不满足则 abort（不发数据到 dest）
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
