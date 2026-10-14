"""示例 4：模拟大模型 Agent 纯靠 Tool-Call 原语一步步插点、插边、连接相似顶点并改图闭环.

本示例展示外部 Agent 如何使用 `DAGToolKit` 的 6 个极简 Tool Call 原语：
  - `set_vertex_input`
  - `add_vev_step`
  - `link_similar_vertices`
  - `update_edge`
  - `remove_edge`
  - `run_and_verify`

运行方式：
    python3 examples/04_llm_toolcall_builder.py
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vea_v5 import DAG, DAGToolKit, VEA_V5_TOOLS_SCHEMA


async def main() -> None:
    print(f"暴露给 LLM 的极简 Tool-Call 原语共 {len(VEA_V5_TOOLS_SCHEMA)} 个:")
    for schema in VEA_V5_TOOLS_SCHEMA:
        fn_info = schema["function"]
        print(f"  - {fn_info['name']}: {fn_info['description']}")

    # 定义验证器：要求输出列表必须是大于 10 的数字平方和 (即 12^2 + 20^2 = 144 + 400 = 544)
    def verifier(output: Any, dag: DAG) -> Dict[str, Any]:
        if output == 544:
            return {"passed": True, "score": 1.0, "feedback": "计算结果 544 完全正确！"}
        return {
            "passed": False,
            "score": 0.5,
            "feedback": f"期望输出大于10的数字平方和 544，但实际得到 {output}（请检查是否对过滤后的数字做了平方处理）",
        }

    toolkit = DAGToolKit(verifier_fn=verifier)

    print("\n--- 第一阶段：Agent 通过 Tool Call 原语一句一句搭建初始 DAG ---")
    tool_calls_round_1 = [
        ("set_vertex_input", {"name": "x1", "content": "12, 5, 8, 20"}),
        ("add_vev_step", {
            "source": "x1",
            "action": "解析数字并过滤大于10的项",
            "target": "过滤后的大数列表",
            "edge_id": "e_filter",
            "code": "def transform(s): return [int(x.strip()) for x in s.split(',') if int(x.strip()) > 10]",
        }),
        # 注意这里 Agent 写第二句话时，把起点名字写成了 "大数列表"（与上一句的 "过滤后的大数列表" 相似但不同名！）
        ("add_vev_step", {
            "source": "大数列表",
            "action": "直接求和(漏了平方)",
            "target": "z2",
            "edge_id": "e_sum",
            "code": "def transform(nums): return sum(nums)",
        }),
        # Agent 调用 link_similar_vertices 自动把相似的 "过滤后的大数列表" 和 "大数列表" 用虚拟边连起来！
        ("link_similar_vertices", {"threshold": 0.45}),
        ("run_and_verify", {}),
    ]

    for tool_name, tool_args in tool_calls_round_1:
        res = await toolkit.call_tool(tool_name, tool_args)
        if tool_name == "link_similar_vertices":
            print("  [虚拟边自动连接结果]:", res["virtual_edges"])
        elif tool_name == "run_and_verify":
            print("  [第一轮运行与验证结果]:", res)

    print("\n--- 第二阶段：Agent 根据验证失败反馈，调用 update_edge 原语修正 e_sum 并增量重跑 ---")
    tool_calls_round_2 = [
        ("update_edge", {
            "edge_id": "e_sum",
            "action": "计算平方和",
            "code": "def transform(nums): return sum(x * x for x in nums)",
        }),
        ("run_and_verify", {}),
    ]

    for tool_name, tool_args in tool_calls_round_2:
        res = await toolkit.call_tool(tool_name, tool_args)
        if tool_name == "run_and_verify":
            print("  [第二轮增量重跑结果 (注意 e_filter 被自动跳过无需重跑!)]:", res)

    print("\n最终 DAG 拓扑快照:")
    print(toolkit.dag.visualize_ascii())


if __name__ == "__main__":
    asyncio.run(main())
