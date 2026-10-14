"""VEA V5 核心数据原语模块 (Primitives).

本模块定义了 VEA V5 最核心的四个基础原语：
1. Vertex (顶点)：纯粹的数据容器，不再需要手动管理 V4 中繁琐的 7 种状态机枚举。
   顶点只有两个直观状态：` pending ` (待计算) 和 ` ready ` (数据已就绪)。
2. Edge (计算边)：连接两个顶点的计算单元 (`input_vertex -> output_vertex`)，
   支持三种最直观的执行模式：
   - `"llm"`：大语言模型提示词推理（自动将上游顶点数据填入 `{input}`）
   - `"code"`：Python 函数或代码字符串转换（直接传入 callable 或 Python 函数源码，无需繁琐白名单配置）
   - `"passthrough"`：直通透传
3. VirtualEdge (虚拟边)：用于连接两个“名称不同但语义相似”的顶点（例如从分立句子中提取出的
   `v_营收数据` 和 `v_收入明细`）。虚拟边在执行时不消耗大模型 Token，而是自动将上游相似顶点的数据
   以透传或合并（JSON 合并 / 列表追加 / 文本拼接）的方式桥接到下游顶点，并记录相似度分数与连接理由。
4. Triplet (V-E-V 分立三元组)：对应自然语言中“从 X 经过 Y 到 Z”的一句分立描述，是搭建与合并 DAG 的最小积木。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union


class VertexStatus(str, Enum):
    """V5 极简顶点状态（彻底废弃 V4 繁琐的 7 态枚举）。"""

    PENDING = "pending"  # 尚未计算或因上游修改需要重新计算
    READY = "ready"      # 数据已就绪，可触发下游边执行
    FAILED = "failed"    # 上游边计算失败


class MergeMode(str, Enum):
    """当多条边（或虚拟边）汇入同一个顶点时的数据合并策略。"""

    OVERWRITE = "overwrite"      # 后写入的数据覆盖已有数据（默认）
    JSON_MERGE = "json_merge"    # 将多个 JSON 字典合并为一个字典: {**old, **new}
    LIST_APPEND = "list_append"  # 将多个输出聚合为列表: [item1, item2, ...]
    TEXT_JOIN = "text_join"      # 使用换行符将多段文本拼接在一起


@dataclass
class Vertex:
    """VEA V5 顶点（纯粹的数据与语义描述容器）。

    Attributes:
        name: 顶点唯一名称（如 `"x1"`、`"原始财报"`、`"营收数据"`）。
        content: 顶点当前存储的数据内容（支持任意类型：字符串、字典、列表、数字；内部可自动序列化）。
        description: 顶点的自然语言语义描述（用于计算不同分立顶点之间的相似度，以自动生成虚拟边）。
        status: 顶点状态（`pending` / `ready` / `failed`）。
        metadata: 附加元数据字典（如来源句子编号、合并历史等）。
    """

    name: str
    content: Any = None
    description: str = ""
    status: VertexStatus = VertexStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_ready(self) -> bool:
        """判断该顶点的数据是否已经就绪。"""
        return self.status == VertexStatus.READY

    @property
    def semantic_text(self) -> str:
        """返回用于相似度匹配的语义文本（结合名称与描述）。"""
        if self.description and self.description != self.name:
            return f"{self.name}: {self.description}"
        return self.name

    def set_ready(self, value: Any) -> None:
        """写入数据并将状态置为 READY。"""
        self.content = value
        self.status = VertexStatus.READY

    def reset(self) -> None:
        """重置顶点为待计算状态（用于闭环更新 DAG 后的增量重跑）。"""
        self.content = None
        self.status = VertexStatus.PENDING

    def content_as_str(self) -> str:
        """将顶点数据安全转换为字符串，供 LLM Prompt 模板渲染使用。"""
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        try:
            return json.dumps(self.content, ensure_ascii=False)
        except Exception:
            return str(self.content)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典。"""
        return {
            "name": self.name,
            "content": self.content,
            "description": self.description,
            "status": self.status.value,
            "metadata": dict(self.metadata),
        }


@dataclass
class Edge:
    """VEA V5 计算边（连接 `source` 顶点与 `target` 顶点的逻辑变换单元）。

    对应“从 X 经过 Y 到 Z”中的“Y”。

    Attributes:
        id: 边的唯一标识符。
        source: 输入顶点名称（`V_in`）。
        target: 输出顶点名称（`V_out`）。
        action: 边的动作描述或名称（如 `"提取营收"`、`"y1"`）。
        kind: 边的计算类型：
            - `"llm"`：使用 `prompt` 调用大模型（若未提供 `prompt`，默认使用 `action` 作为指令）。
            - `"code"`：执行 Python 函数 `fn` 或代码字符串 `code`。
            - `"passthrough"`：直接把 `source` 的数据传给 `target`。
        prompt: 当 `kind="llm"` 时的提示词模板（支持 `{input}` 占位符）。
        fn: 当 `kind="code"` 时可直接传入的 Python 可调用函数 `fn(input_data) -> output_data`。
        code: 当 `kind="code"` 时可传入的 Python 函数源码字符串（需定义 `def transform(data): ...`）。
        merge_mode: 当多条边汇入同一 `target` 顶点时的合并策略（默认 `TEXT_JOIN` 或 `OVERWRITE`）。
        metadata: 附加元数据。
    """

    id: str
    source: str
    target: str
    action: str = ""
    kind: str = "llm"  # "llm" | "code" | "passthrough" | "virtual"
    prompt: Optional[str] = None
    fn: Optional[Callable[[Any], Any]] = None
    code: Optional[str] = None
    merge_mode: MergeMode = MergeMode.OVERWRITE
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_virtual(self) -> bool:
        """是否为连接相似顶点的虚拟边。"""
        return False

    def render_prompt(self, input_text: str) -> str:
        """根据上游顶点输入文本渲染完整 Prompt。"""
        template = self.prompt or self.action or "请处理以下输入：\n{input}"
        if "{input}" in template:
            return template.replace("{input}", input_text)
        return f"{template}\n\n输入数据：\n{input_text}"

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典。"""
        return {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "action": self.action,
            "kind": self.kind,
            "prompt": self.prompt,
            "code": self.code,
            "merge_mode": self.merge_mode.value if isinstance(self.merge_mode, MergeMode) else str(self.merge_mode),
            "is_virtual": self.is_virtual,
            "metadata": dict(self.metadata),
        }


@dataclass
class VirtualEdge(Edge):
    """VEA V5 虚拟边（Virtual Edge）。

    当系统从多句分立描述（如“从 x1 经过 y1 到 z1_a”、“从 z1_b 经过 y2 到 z2”）中提取出多个分立的
    `V-E-V` 三元组时，如果发现产出顶点 `z1_a` 与某个输入顶点 `z1_b` 语义高度相似，
    合并器就会在 `z1_a` 和 `z1_b` 之间自动插入一条 `VirtualEdge`。

    虚拟边在运行时无需调用大模型，直接将 `source` 顶点的数据桥接/汇聚到 `target` 顶点中，
    从而将原本断开的分立句子粘合成一张完整可运行的 DAG。

    Attributes:
        similarity: 两个顶点之间的相似度得分（0.0 ~ 1.0）。
        reason: 判定两个顶点相似的原因说明（例如 `"名称/语义相似: '商品价格列表' ≈ '商品价格清单' (0.86)"`）。
    """

    similarity: float = 1.0
    reason: str = "语义相似自动连接"

    def __post_init__(self) -> None:
        self.kind = "virtual"
        if not self.action:
            self.action = f"≈虚拟连接({self.similarity:.2f})"
        # 多个上游相似顶点汇入同一个下游顶点时，默认采用智能合并（字典合并或文本拼接）
        if self.merge_mode == MergeMode.OVERWRITE:
            self.merge_mode = MergeMode.TEXT_JOIN

    @property
    def is_virtual(self) -> bool:
        return True

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base["similarity"] = round(self.similarity, 4)
        base["reason"] = self.reason
        return base


@dataclass
class Triplet:
    """分立的 `V - E - V` 三元组（对应一句话：“从 X 经过 Y 到 Z”）。

    Attributes:
        source_name: 起点名称（X）。
        edge_action: 中间经过的处理动作或计算描述（Y）。
        target_name: 终点名称（Z）。
        source_desc: 起点的补充描述（可选）。
        target_desc: 终点的补充描述（可选）。
        edge_kind: 边的计算类型（`"llm"` 或 `"code"`）。
        prompt: 可选的具体 LLM 提示词模板。
        code: 可选的具体 Python 转换代码。
        fn: 可选的 Python 函数。
        raw_sentence: 原始自然语言句子（用于溯源）。
    """

    source_name: str
    edge_action: str
    target_name: str
    source_desc: str = ""
    target_desc: str = ""
    edge_kind: str = "llm"
    prompt: Optional[str] = None
    code: Optional[str] = None
    fn: Optional[Callable[[Any], Any]] = None
    raw_sentence: str = ""

    def __repr__(self) -> str:
        return f"Triplet({self.source_name!r} --[{self.edge_action!r}]--> {self.target_name!r})"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_name": self.source_name,
            "edge_action": self.edge_action,
            "target_name": self.target_name,
            "source_desc": self.source_desc,
            "target_desc": self.target_desc,
            "edge_kind": self.edge_kind,
            "prompt": self.prompt,
            "code": self.code,
            "raw_sentence": self.raw_sentence,
        }
