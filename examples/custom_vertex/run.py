#!/usr/bin/env python3
"""运行自定义顶点示例。

演示如何用 Python 子类自定义 Vertex，并手动组装图运行：
    input(清洗) --e1--> processor(清洗+统计) --e2--> output(汇总报告)

由于 Graph.from_json 只能构造内置 Vertex，自定义顶点需要手动组装：
    1. 创建自定义顶点实例
    2. 手动登记边的关联关系(outgoing/incoming/required_input_count)
    3. 调用 graph.validate() 校验 DAG

Usage:
    python examples/custom_vertex/run.py
"""

import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from framework import Graph, Edge, Executor, MockPIAgent
from custom_vertex import SanitizeVertex, ReportVertex


def add_edge(graph, edge_id, source, dest, **kwargs):
    """手动添加一条边并登记关联关系(模拟 Graph.from_dict 内部逻辑)。"""
    edge = Edge(edge_id, source.id, dest.id, **kwargs)
    graph.edges[edge.id] = edge
    # 源顶点登记出边
    source.outgoing_edges.append(edge.id)
    # 目标顶点登记入边，并更新所需输入数量
    dest.incoming_edges.append(edge.id)
    dest.required_input_count = len(dest.incoming_edges)
    return edge


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-7s  %(message)s",
    )

    graph = Graph()

    # 1) 创建自定义顶点(可用 initial_data 给源顶点预置数据)
    src = SanitizeVertex(
        "input",
        settings={"type": "source"},
        initial_data=[
            {"data_id": "text", "tags": ["en"], "value": "  Hello   world!  "},
        ],
    )
    processor = SanitizeVertex("processor")
    output = ReportVertex("output")

    # 2) 注册顶点
    graph.vertices = {"input": src, "processor": processor, "output": output}

    # 3) 手动添加边
    add_edge(graph, "e1", src, processor, data_id="text", tags=["en"], prompt="Process:", model="gemini-flash")
    add_edge(graph, "e2", processor, output, data_id="stats", tags=["summary"], prompt="Summarize:", model="gemini-pro")

    # 4) 校验 DAG(引用完整性 + 无环)
    graph.validate()

    # 5) 执行
    result = await Executor(graph, MockPIAgent(), max_concurrency=4, timeout=30).run()

    print("\n" + result.summary())

    # 展示自定义顶点记录的词数统计(自定义属性)
    print("\n── 自定义属性 word_counts ──")
    for vid in ["input", "processor"]:
        print(f"  [{vid}] {graph.vertices[vid].word_counts}")

    return result


if __name__ == "__main__":
    asyncio.run(main())
