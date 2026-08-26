"""Edge module - Connection between vertices in the graph.

边(Edge)模块 —— 图中连接顶点的边。

An Edge reads the MAIN data from its source vertex (via ``get()``), processes
it through a PI Agent, and writes the result to the destination vertex keyed
by THIS edge's own ID (``dest.set(result, edge_id=self.id)``) so that fan-in
vertices record each incoming edge in its own slot without overwriting.

一条边从源顶点读取「主数据」，经 PI Agent 处理后，把结果以**本边 ID** 为键
写入目标顶点(``dest.set(result, edge_id=self.id)``)：
因此 fan-in 顶点会按来源边 ID 分槽记录每条入边的数据，天然不覆盖。
外部脚本可在 PI Agent 处理前预处理数据(pre_process)，或在交付前后处理结果(post_process)。
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("vertex_edge_agent.edge")


class Edge:
    """Directed edge connecting a source vertex to a destination vertex.

    连接源顶点与目标顶点的有向边。

    Attributes:
        id:              Unique edge identifier.  边唯一标识(也是写目标时的数据 key)。
        source_id:       Source vertex ID.       源顶点 ID。
        destination_id:  Destination vertex ID.  目标顶点 ID。
        prompt:          Prompt sent to the PI Agent.  发送给 PI Agent 的提示词。
        model:           Model identifier for the PI Agent.  模型标识。
        settings:        Arbitrary settings dict passed to agent & scripts.
                         透传给 agent 与脚本的任意配置字典。
        script_path:     Optional path to an external Python script.
                         可选的外部 Python 脚本路径。
    """

    def __init__(
        self,
        edge_id: str,
        source_id: str,
        destination_id: str,
        prompt: str = "",
        model: str = "default",
        settings: Optional[Dict] = None,
        script_path: Optional[str] = None,
    ):
        self.id = edge_id
        self.source_id = source_id
        self.destination_id = destination_id
        self.prompt = prompt
        self.model = model
        self.settings = settings or {}
        self.script_path = script_path
        self._script_module = None  # 加载后的外部脚本模块

        # 执行状态
        self.completed: bool = False   # 是否执行完成
        self.result: Any = None        # 执行结果
        self.error: Optional[str] = None  # 出错时的错误信息

        logger.info(
            "[Edge:%s] Created %s -> %s | model=%s",
            self.id, source_id, destination_id, model,
        )

    def set_script_module(self, module):
        """Attach a loaded external script module.

        挂载已加载的外部脚本模块。
        """
        self._script_module = module
        logger.debug("[Edge:%s] Script module attached: %s", self.id, module)

    async def execute(self, source_vertex, dest_vertex, pi_agent) -> Any:
        """Execute the edge pipeline.

        执行整条边的处理流水线。

        Steps:  处理步骤：
            1. ``source_vertex.get()``  → main data (主数据)
               从源顶点读取主数据
            2. Script ``pre_process(data, settings)`` (optional)
               可选：脚本预处理
            3. ``pi_agent.process(data, prompt, model, settings)``
               PI Agent 处理
            4. Script ``post_process(result, settings)`` (optional)
               可选：脚本后处理
            5. ``dest_vertex.set(result, edge_id=self.id)``
               以本边 ID 为键把结果写入目标顶点

        Returns the final result written to the destination vertex.
        返回写入目标顶点的最终结果。
        """
        logger.info(
            "[Edge:%s] EXECUTE  %s -> %s",
            self.id, self.source_id, self.destination_id,
        )

        try:
            # 1 — 从源顶点读取主数据
            data = await source_vertex.get()
            logger.debug("[Edge:%s] Source data: %s", self.id, repr(data)[:200])
            if data is None:
                logger.warning(
                    "[Edge:%s] Source vertex '%s' returned None (no main data)",
                    self.id, self.source_id,
                )

            # 2 — 脚本预处理(pre_process)
            if self._script_module and hasattr(self._script_module, "pre_process"):
                data = self._script_module.pre_process(data, self.settings)
                logger.debug("[Edge:%s] After pre_process: %s", self.id, repr(data)[:200])

            # 3 — PI Agent 处理
            result = await pi_agent.process(
                data=data,
                prompt=self.prompt,
                model=self.model,
                settings=self.settings,
            )
            logger.debug("[Edge:%s] PI Agent result: %s", self.id, repr(result)[:200])

            # 4 — 脚本后处理(post_process)
            if self._script_module and hasattr(self._script_module, "post_process"):
                result = self._script_module.post_process(result, self.settings)
                logger.debug("[Edge:%s] After post_process: %s", self.id, repr(result)[:200])

            # 5 — 以本边 ID 为键写入目标顶点(fan-in 按边 ID 分槽，不覆盖)
            await dest_vertex.set(result, edge_id=self.id)
            logger.info(
                "[Edge:%s] Delivered to '%s' | key=edge:%s",
                self.id, self.destination_id, self.id,
            )

            # 记录执行成功状态
            self.completed = True
            self.result = result
            return result

        except Exception as exc:
            # 任一环节失败，记录错误信息并向上抛出
            self.error = str(exc)
            logger.error("[Edge:%s] FAILED: %s", self.id, exc, exc_info=True)
            raise

    def reset(self):
        """Reset edge state for re-execution.

        重置边的执行状态(用于重新执行)。
        """
        self.completed = False
        self.result = None
        self.error = None

    # ------------------------------------------------------------------
    # Abort propagation  中止信号透传
    # ------------------------------------------------------------------
    async def forward_abort(self, dest_vertex, reason: str):
        """向目标顶点写入一个 Abort 信号(以本边 ID 为键)，并标记本边为已中止。

        用于「数据不足 / 前置条件不满足」等场景：本边不再执行正常逻辑，
        而是构造一个携带 reason 的 AbortSignal 写入目标顶点，
        下游边检测到后会继续透传，直到最终顶点。

        Args:
            dest_vertex: 目标顶点(Edge.execute 里的 dest_vertex)。
            reason:      中止原因(人类可读)。

        Returns:
            写出的 AbortSignal 对象。
        """
        from .signal import AbortSignal

        signal = AbortSignal(reason, source=self.id)
        await dest_vertex.set(signal, edge_id=self.id)
        self.completed = True
        self.result = signal
        self.error = None
        logger.warning("[Edge:%s] FORWARD ABORT: %s", self.id, reason)
        return signal

    def __repr__(self):
        # 用符号直观表示边的状态：✓ 完成 / ✗ 出错 / · 未执行
        status = "✓" if self.completed else ("✗" if self.error else "·")
        return f"Edge({self.id} {self.source_id}->{self.destination_id} [{status}])"
