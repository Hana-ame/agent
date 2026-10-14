"""VEA V5 面向 LLM Agent 的极简 Tool-Call 原语工具箱 (Toolkit).

为了让大模型（Agent）能够以最简单、最不容易出错的方式自由搭建和动态修改 DAG，
本模块将所有图操作浓缩为 6 个原子 Tool-Call 原语：
1. `add_vev_step`：一句话插入“从 `source` 经过 `action` 到 `target`”的计算步骤（最核心原语！）
2. `set_vertex_input`：设置某个入口顶点的初始输入数据
3. `link_similar_vertices`：在两个相似顶点之间连一条虚拟边 `VirtualEdge`（或者调用 `auto_link_similar` 自动全局连接）
4. `update_edge`：根据验证反馈修改某条边的提示词 `prompt` 或 Python 转换代码 `code`（自动触发下游增量重置）
5. `remove_edge`：删除一条边
6. `run_and_verify`：运行当前 DAG 并调用验证器检验结果好坏
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from vea_v5.dag import DAG
from vea_v5.executor import ExecutionReport, Executor
from vea_v5.llm import LLMProtocol, SmartMockLLM
from vea_v5.primitives import MergeMode


# 标准 OpenAI Function Calling 工具定义 Schema（可直接传给任意支持 tool_calls 的大模型）
VEA_V5_TOOLS_SCHEMA: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "add_vev_step",
            "description": "向 DAG 中添加一步 '从 source 经过 action 到 target' 的 V-E-V 计算步骤。若顶点不存在会自动创建。",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "起点顶点名称（例如 'x1' 或 '原始文本'）"},
                    "action": {"type": "string", "description": "边执行的动作描述或指令（例如 '提取关键词'）"},
                    "target": {"type": "string", "description": "终点顶点名称（例如 'z1' 或 '关键词列表'）"},
                    "edge_id": {"type": "string", "description": "可选的边 ID（如 'e_step1'）"},
                    "prompt": {"type": "string", "description": "可选的大模型 Prompt 模板（用 {input} 代表上游数据）"},
                    "code": {
                        "type": "string",
                        "description": "可选的 Python 转换函数源码（定义 `def transform(data): return ...`），提供后将作为代码边执行",
                    },
                    "merge_mode": {
                        "type": "string",
                        "enum": ["overwrite", "text_join", "json_merge", "list_append"],
                        "description": "多条边汇入同一终点时的数据合并方式",
                    },
                },
                "required": ["source", "action", "target"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_vertex_input",
            "description": "为指定顶点写入初始输入数据（自动将该顶点置为 READY 并重置其下游以备重算）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "顶点名称"},
                    "content": {"type": "string", "description": "要写入的数据内容"},
                    "description": {"type": "string", "description": "可选的顶点语义描述"},
                },
                "required": ["name", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "link_similar_vertices",
            "description": "在两个语义相似的顶点之间插入虚拟边（VirtualEdge），或自动扫描全图连接所有相似顶点。",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "上游产出顶点名称（若为空则开启全图自动相似匹配）"},
                    "target": {"type": "string", "description": "下游待输入顶点名称"},
                    "similarity": {"type": "number", "description": "相似度分数 (0~1)"},
                    "reason": {"type": "string", "description": "相似连接理由（如 '这两个顶点都表示商品价格数据'）"},
                    "threshold": {"type": "number", "description": "自动匹配模式下的相似度阈值（默认 0.45）"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_edge",
            "description": "当验证结果不理想时，修改某条边的 Prompt 或 Python 代码；系统会自动重置该边下游顶点以便增量重跑。",
            "parameters": {
                "type": "object",
                "properties": {
                    "edge_id": {"type": "string", "description": "要修改的边 ID"},
                    "action": {"type": "string", "description": "更新后的动作描述"},
                    "prompt": {"type": "string", "description": "更新后的 LLM Prompt 模板"},
                    "code": {"type": "string", "description": "更新后的 Python `def transform(data): ...` 代码"},
                    "merge_mode": {
                        "type": "string",
                        "enum": ["overwrite", "text_join", "json_merge", "list_append"],
                    },
                },
                "required": ["edge_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_edge",
            "description": "从 DAG 中删除一条边，并重置原下游顶点状态。",
            "parameters": {
                "type": "object",
                "properties": {"edge_id": {"type": "string"}},
                "required": ["edge_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_and_verify",
            "description": "执行当前 DAG（自动跳过未受影响的已就绪上游顶点），并用验证器检验结果好坏。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


class DAGToolKit:
    """面向 LLM Agent 的极简 DAG 原语操作台。"""

    def __init__(
        self,
        dag: Optional[DAG] = None,
        llm: Optional[LLMProtocol] = None,
        verifier_fn: Optional[Callable[[Any, DAG], Dict[str, Any]]] = None,
    ) -> None:
        self.dag = dag or DAG("Agent_Constructed_DAG")
        self.llm = llm or SmartMockLLM()
        self.verifier_fn = verifier_fn

    def add_vev_step(
        self,
        source: str,
        action: str,
        target: str,
        edge_id: Optional[str] = None,
        prompt: Optional[str] = None,
        code: Optional[str] = None,
        merge_mode: str = "overwrite",
    ) -> Dict[str, Any]:
        """Tool 1: 插入一步 `V - E - V`（从 source 经过 action 到 target）。"""
        try:
            edge = self.dag.add_step(
                source=source,
                action=action,
                target=target,
                edge_id=edge_id,
                prompt=prompt,
                code=code,
                merge_mode=MergeMode(merge_mode),
            )
            return {"status": "ok", "edge": edge.to_dict()}
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    def set_vertex_input(self, name: str, content: Any, description: str = "") -> Dict[str, Any]:
        """Tool 2: 设置顶点的输入内容。"""
        v = self.dag.add_vertex(name=name, content=content, description=description)
        return {"status": "ok", "vertex": v.to_dict()}

    def link_similar_vertices(
        self,
        source: Optional[str] = None,
        target: Optional[str] = None,
        similarity: float = 0.85,
        reason: str = "这两个顶点语义相似",
        threshold: float = 0.45,
    ) -> Dict[str, Any]:
        """Tool 3: 手动或自动在相似顶点之间架设虚拟边 `VirtualEdge`。"""
        try:
            if source and target:
                ve = self.dag.add_virtual_edge(
                    source=source,
                    target=target,
                    similarity=similarity,
                    reason=reason,
                )
                return {"status": "ok", "virtual_edges": [ve.to_dict()]}
            else:
                ves = self.dag.auto_link_similar_vertices(threshold=threshold)
                return {"status": "ok", "virtual_edges": [v.to_dict() for v in ves]}
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    def update_edge(
        self,
        edge_id: str,
        action: Optional[str] = None,
        prompt: Optional[str] = None,
        code: Optional[str] = None,
        fn: Optional[Callable[[Any], Any]] = None,
        merge_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Tool 4: 修改已有边的计算内容（Prompt / Python 代码 / 函数），并自动重置下游以备增量重跑。"""
        if edge_id not in self.dag.edges:
            return {"status": "error", "error": f"边 '{edge_id}' 不存在"}
        edge = self.dag.edges[edge_id]
        if action is not None:
            edge.action = action
        if prompt is not None:
            edge.prompt = prompt
            edge.kind = "llm"
        if code is not None:
            edge.code = code
            edge.fn = None
            edge.kind = "code"
        if fn is not None:
            edge.fn = fn
            edge.kind = "code"
        if merge_mode is not None:
            edge.merge_mode = MergeMode(merge_mode)

        invalidated = self.dag.invalidate_from(edge.target)
        return {
            "status": "updated",
            "edge": edge.to_dict(),
            "invalidated_vertices": invalidated,
        }

    def remove_edge(self, edge_id: str) -> Dict[str, Any]:
        """Tool 5: 删除一条边。"""
        ok = self.dag.remove_edge(edge_id)
        return {"status": "deleted" if ok else "not_found", "edge_id": edge_id}

    async def run_and_verify(self) -> Dict[str, Any]:
        """Tool 6: 运行 DAG 并调用验证器返回好坏评估结果。"""
        executor = Executor(self.dag, llm=self.llm)
        report: ExecutionReport = await executor.run()
        verification: Dict[str, Any] = {"passed": report.success}
        if self.verifier_fn is not None:
            verification = self.verifier_fn(report.final_output, self.dag)

        return {
            "execution_success": report.success,
            "executed_edges": report.executed_edges,
            "skipped_cached_vertices": report.skipped_vertices,
            "final_output": report.final_output,
            "all_vertices": report.all_vertices,
            "errors": report.errors,
            "verification": verification,
        }

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """统一执行来自 LLM 的 Tool Call。"""
        dispatch = {
            "add_vev_step": self.add_vev_step,
            "set_vertex_input": self.set_vertex_input,
            "link_similar_vertices": self.link_similar_vertices,
            "update_edge": self.update_edge,
            "remove_edge": self.remove_edge,
        }
        if tool_name == "run_and_verify":
            return await self.run_and_verify()
        if tool_name in dispatch:
            return dispatch[tool_name](**arguments)
        raise ValueError(f"未知的 Tool 原语: {tool_name}")
