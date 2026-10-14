"""VEA V5 (Vertex-Edge Agent Framework v5.0) — 极简、自演进的智能体有向无环图框架.

核心特性：
1. **一句话搭图 (`V - E - V`)**：`dag.add_step("x1", "y1", "z1")`，自动推导起终点，告别 V4 的 7 态状态机。
2. **第一个 VEA DAG（合并器 `SentenceToDAGMerger`）**：输入分立的自然语言句子，层层拆解为 `V-E-V` 三元组，
   自动计算顶点相似度并用虚拟边（`VirtualEdge`，“这两个挺像的”）连接成完整可运行的 DAG。
3. **验证驱动的闭环自演进 (`ClosedLoopRunner` & `DAGToolKit`)**：运行 DAG 产出结果 -> 验证器评判好坏 ->
   通过 6 个极简 Tool-Call 原语动态修改/增补 DAG -> 增量重跑（自动跳过未受影响的上游节点）直至验证通过。
"""

from vea_v5.closed_loop import ClosedLoopRunner, ClosedLoopSummary, LoopIterationRecord, VerifyResult
from vea_v5.dag import CycleError, DAG, default_text_similarity
from vea_v5.executor import EdgeTrace, ExecutionReport, Executor, merge_values
from vea_v5.llm import LLMProtocol, OpenAICompatibleLLM, SmartMockLLM
from vea_v5.merger import MergerResult, SentenceToDAGMerger, parse_vev_from_text
from vea_v5.primitives import Edge, MergeMode, Triplet, Vertex, VertexStatus, VirtualEdge
from vea_v5.toolkit import DAGToolKit, VEA_V5_TOOLS_SCHEMA

__version__ = "5.0.0"

__all__ = [
    "__version__",
    "Vertex",
    "VertexStatus",
    "Edge",
    "VirtualEdge",
    "Triplet",
    "MergeMode",
    "DAG",
    "CycleError",
    "default_text_similarity",
    "Executor",
    "ExecutionReport",
    "EdgeTrace",
    "merge_values",
    "LLMProtocol",
    "OpenAICompatibleLLM",
    "SmartMockLLM",
    "SentenceToDAGMerger",
    "MergerResult",
    "parse_vev_from_text",
    "DAGToolKit",
    "VEA_V5_TOOLS_SCHEMA",
    "ClosedLoopRunner",
    "ClosedLoopSummary",
    "LoopIterationRecord",
    "VerifyResult",
]
