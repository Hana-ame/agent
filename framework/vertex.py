"""Vertex module - Node in the computation graph.

顶点(Vertex)模块 —— 计算图中的节点。

数据存储键为 (edge_id, tags)：按「来源边 ID」+「标签」分槽。
    每条边写入目标时带上自己的 ID，目标顶点按边 ID 分槽记录，fan-in 天然不覆盖；
    tags 表达该数据的语义标签(读/写时区分)。
    非边来源的数据用保留键 edge_id：
        KEY_INIT  ("__init__")   初始数据(源顶点预置)
        KEY_SELF  ("__self__")   顶点自产结果(on_ready / prepare_outputs 合并产物)

具备生命周期状态机，并支持外部 Python 脚本进行数据处理、校验与拒绝。
"""

import asyncio
import enum
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("vertex_edge_agent.vertex")

# 保留键：非边来源的数据
KEY_INIT = "__init__"   # 初始数据
KEY_SELF = "__self__"   # 顶点自产(on_ready 合并产物)


def _key(edge_id: Optional[str], tags: Optional[List[str]]) -> Tuple[str, Tuple[str, ...]]:
    """规范化存储键：key = (edge_id, tuple(sorted(tags)))。"""
    return (edge_id or KEY_SELF, tuple(sorted(tags or [])))


class VertexState(enum.Enum):
    """States for vertex lifecycle.  顶点的生命周期状态。"""
    IDLE = "idle"                # 等待输入数据
    READY = "ready"              # 所有输入已接收，可以开始处理
    PROCESSING = "processing"    # 正在触发输出边
    DONE = "done"                # 所有处理已完成
    ERROR = "error"              # 发生错误


class DataRejectedError(Exception):
    """Raised when a vertex rejects incoming data via its script.

    当顶点通过脚本拒绝接收到的数据时抛出。
    """
    pass


class Vertex:
    """A vertex (node) in the computation graph.

    计算图中的一个顶点(节点)。

    Data is keyed by (source edge ID, tags).  数据以「来源边 ID + 标签」为键存储。
    Has a state machine for lifecycle management.  拥有生命周期状态机。
    Supports external scripts for data handling/validation/rejection.
    支持外部脚本进行数据处理 / 校验 / 拒绝。

    Methods:
        get(edge_id, tags) -> data               读取数据(参数可省)
        set(data, edge_id, tags) -> bool         写入数据
        get_all_data() -> {(edge_id, tags): data}
        prepare_outputs()                        出边触发前运行 on_ready 钩子
    """

    def __init__(
        self,
        vertex_id: str,
        settings: Optional[Dict] = None,
        script_path: Optional[str] = None,
        initial_data: Optional[List[Dict]] = None,
    ):
        self.id = vertex_id
        self.settings = settings or {}
        self.script_path = script_path
        self._script_module = None  # 加载后的外部脚本模块

        # 数据存储：key = (edge_id, tags) -> data（非边来源数据用 KEY_INIT / KEY_SELF）
        self._data_store: Dict[Tuple[str, Tuple[str, ...]], Any] = {}
        self._lock = asyncio.Lock()  # 保护数据存储的异步锁

        # 状态管理
        self._state = VertexState.IDLE
        self._ready_event = asyncio.Event()  # 进入 READY 时置位，供等待方使用
        self._on_ready_cb = None             # 进入 READY 时的通知回调(事件驱动调度用)

        # 边(Edge)追踪
        self.incoming_edges: List[str] = []   # 入边 ID 列表
        self.outgoing_edges: List[str] = []   # 出边 ID 列表
        self.required_input_count: int = 0    # 期望接收的输入数量(等于入边数)
        self._received_input_count: int = 0   # 已实际接收的输入数量

        # 错误信息
        self.error_message: Optional[str] = None

        # 加载初始数据(仅源顶点通常会有；按 (__init__, tags) 分槽，保留各份的 tag)
        if initial_data:
            for item in initial_data:
                key = _key(KEY_INIT, item.get("tags"))
                self._data_store[key] = item.get("value")
                logger.debug(
                    "[Vertex:%s] Loaded initial data -> key=%s, value=%s",
                    self.id, key, repr(item.get("value"))[:120],
                )

        logger.debug(
            "[Vertex:%s] Created | settings=%s | script=%s | initial_keys=%s",
            self.id, self.settings, self.script_path, list(self._data_store.keys()),
        )

    # ------------------------------------------------------------------
    # State property  状态属性
    # ------------------------------------------------------------------
    @property
    def state(self) -> VertexState:
        return self._state

    @state.setter
    def state(self, new_state: VertexState):
        old = self._state
        self._state = new_state
        logger.debug("[Vertex:%s] %s -> %s", self.id, old.value, new_state.value)
        # 进入 READY 时置位事件，否则清除，供 wait_ready 使用
        if new_state == VertexState.READY:
            self._ready_event.set()
            # 事件驱动调度：通知注册的回调(executor 用它唤醒主循环)
            if self._on_ready_cb is not None:
                self._on_ready_cb(self)
        else:
            self._ready_event.clear()

    def set_ready_callback(self, cb):
        """注册「进入 READY 状态」的通知回调。

        Executor 用它做事件驱动调度：顶点 READY 时立即通知主循环，
        取代轮询扫描。同一时刻通常只注册一个回调(由执行器注入)。
        """
        self._on_ready_cb = cb

    # ------------------------------------------------------------------
    # Script  外部脚本
    # ------------------------------------------------------------------
    def set_script_module(self, module):
        """Attach a loaded external script module.

        挂载已加载的外部脚本模块。
        """
        self._script_module = module
        logger.debug("[Vertex:%s] Script module attached: %s", self.id, module)

    # ------------------------------------------------------------------
    # Data access  数据访问
    # ------------------------------------------------------------------
    async def get(
        self,
        edge_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Any:
        """读取数据。

        - edge_id + tags：读精确槽 (edge_id, tags)。
        - 仅 edge_id：读该来源边写入的槽(任一 tag)。
        - 仅 tags：读带这些 tag 的槽(任一边)。
        - 都不给：读「主数据」—— 优先自产(__self__)，否则唯一槽，否则初始(__init__)。
        """
        async with self._lock:
            if edge_id is not None and tags is not None:
                data = self._data_store.get(_key(edge_id, tags))
            elif edge_id is not None:
                data = self._first_locked(lambda k: k[0] == edge_id)
            elif tags is not None:
                want = tuple(sorted(tags))
                data = self._first_locked(lambda k: k[1] == want)
            else:
                data = self._main_data_locked()
        logger.debug(
            "[Vertex:%s] GET edge=%s tags=%s -> %s",
            self.id, edge_id, tags, repr(data)[:120],
        )
        return data

    def _first_locked(self, pred) -> Any:
        """按条件取第一个匹配槽的值。"""
        for k, v in self._data_store.items():
            if pred(k):
                return v
        return None

    def _main_data_locked(self) -> Any:
        """主数据：__self__ > 唯一槽 > 第一个 __init__ 槽。"""
        if _key(KEY_SELF, None) in self._data_store:
            return self._data_store[_key(KEY_SELF, None)]
        if len(self._data_store) == 1:
            return next(iter(self._data_store.values()))
        # 多个槽时：返回第一个初始数据槽
        for k, v in self._data_store.items():
            if k[0] == KEY_INIT:
                return v
        return self._data_store.get(_key(KEY_INIT, None))

    async def set(
        self,
        data: Any,
        edge_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> bool:
        """写入数据。

        edge_id 提供：按来源边 ID 分槽记录(不同边各占一槽，不覆盖)。
        edge_id 缺省：当作顶点自产，存到保留键 __self__。
        tags：写入时打的标签(与 edge_id 一起组成 key = (edge_id, tags))。

        Returns True on success.  Raises ``DataRejectedError`` if the
        vertex script's ``on_receive`` rejects the data.

        成功返回 True；若顶点脚本的 ``on_receive`` 拒绝该数据，
        则抛出 ``DataRejectedError``。
        """
        key = _key(edge_id, tags)
        logger.debug("[Vertex:%s] SET %s <- %s", self.id, key, repr(data)[:120])

        # --- 运行顶点脚本的 on_receive 钩子 ---
        # 钩子可对数据做转换，或抛异常表示拒绝
        if self._script_module and hasattr(self._script_module, "on_receive"):
            try:
                data = self._script_module.on_receive(
                    data, edge_id or "", tags or [], self.settings
                )
                logger.debug(
                    "[Vertex:%s] on_receive returned: %s", self.id, repr(data)[:120]
                )
            except Exception as exc:
                # 脚本抛异常 => 数据被拒绝（异常会作为 DataRejectedError 上抛）
                logger.debug(
                    "[Vertex:%s] on_receive REJECTED data: %s", self.id, exc
                )
                raise DataRejectedError(
                    f"Vertex '{self.id}' rejected data: {exc}"
                ) from exc

        async with self._lock:
            self._data_store[key] = data
            self._received_input_count += 1
            logger.debug(
                "[Vertex:%s] Input %d/%d received",
                self.id, self._received_input_count, self.required_input_count,
            )
            # 当所有要求的输入都到达时，顶点进入 READY 状态，等待执行器拾取
            if (
                self.required_input_count > 0
                and self._received_input_count >= self.required_input_count
            ):
                self.state = VertexState.READY
        return True

    async def get_all_data(self) -> Dict[Tuple[str, Tuple[str, ...]], Any]:
        """Return a copy of the entire data store ((edge_id, tags) -> data).

        返回整个数据存储的副本((来源边 ID, 标签) -> 数据)。
        """
        async with self._lock:
            return dict(self._data_store)

    async def prepare_outputs(self):
        """Run the script's ``on_ready`` hook to consolidate data.

        运行脚本的 ``on_ready`` 钩子，把多个输入整合为输出(存到 __self__)。

        The hook receives all stored data ((edge_id, tags) -> data) and the
        vertex settings, and should return a single consolidated value that
        will be stored under the reserved key __self__ (the "main data").

        钩子接收所有已存数据({(edge_id, tags): data})与顶点配置，应返回一个
        合并后的值，会被存到保留键 __self__(即主数据)。
        """
        if self._script_module and hasattr(self._script_module, "on_ready"):
            logger.debug("[Vertex:%s] Running on_ready hook", self.id)
            all_data = dict(self._data_store)
            try:
                outputs = self._script_module.on_ready(all_data, self.settings)
                if outputs is not None:
                    # 兼容旧脚本返回 {(key, (tags,)): value} 的 dict：按返回的
                    # (data_id, tags) 分槽存储，使下游按 get-tags 能读到；
                    # 同时额外存一份主数据(__self__)，保证无 tag 读取也能拿到。
                    async with self._lock:
                        value = None
                        if isinstance(outputs, dict) and outputs:
                            for out_key, val in outputs.items():
                                value = val
                                edge_id = out_key[0] if isinstance(out_key, tuple) else KEY_SELF
                                tags = out_key[1] if (isinstance(out_key, tuple)
                                                       and len(out_key) > 1) else None
                                self._data_store[_key(edge_id, tags)] = val
                        else:
                            value = outputs
                        if value is not None and _key(KEY_SELF, None) not in self._data_store:
                            self._data_store[_key(KEY_SELF, None)] = value
                            logger.debug(
                                "[Vertex:%s] on_ready set %s = %s",
                                self.id, _key(KEY_SELF, None), repr(value)[:120],
                            )
            except Exception as exc:
                logger.error(
                    "[Vertex:%s] on_ready hook failed: %s", self.id, exc, exc_info=True
                )
                raise

    # ------------------------------------------------------------------
    # Helpers  辅助方法
    # ------------------------------------------------------------------
    async def wait_ready(self, timeout: Optional[float] = None):
        """Block until the vertex reaches READY state.

        阻塞直到顶点进入 READY 状态(可指定超时)。
        """
        await asyncio.wait_for(self._ready_event.wait(), timeout=timeout)

    def is_source(self) -> bool:
        """True if this vertex has no incoming edges.  是否为源顶点(无入边)。"""
        return len(self.incoming_edges) == 0

    def is_sink(self) -> bool:
        """True if this vertex has no outgoing edges.  是否为汇顶点(无出边)。"""
        return len(self.outgoing_edges) == 0

    def reset(self):
        """Reset vertex to initial state (for re-runs).

        重置顶点到初始状态(用于重新执行)。
        """
        self._state = VertexState.IDLE
        self._ready_event.clear()
        self._received_input_count = 0
        self.error_message = None
        logger.debug("[Vertex:%s] Reset to IDLE", self.id)

    def __repr__(self):
        return (
            f"Vertex(id={self.id!r}, state={self._state.value}, "
            f"in={len(self.incoming_edges)}, out={len(self.outgoing_edges)})"
        )
