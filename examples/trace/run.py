#!/usr/bin/env python3
"""Trace 示例：运行中逐步输出中间过程（事件驱动调度演示）。

图(6 顶点 6 边)：全部普通 Edge，fan-in 靠「按来源边 ID 记录」不覆盖。
    A(source, text:en_in)
      --e1(摘要)--> B
    B --e2(译法)--> C | B --e3(译西)--> D                     ← fan-out
    C --e4--> E | D --e5--> E                                ← fan-in
    E(按边 ID 记录 e4/e5 两份 → on_ready 合并 text:digest)
      --e6(报告)--> F

关键：Edge 写目标时自动带上自己的 ID(edge_id)，Vertex 按来源边 ID 分槽
      记录(_edge_data[edge_id]=data)，fan-in 顶点在输入到齐后用自己的
      prepare_outputs/get_all_edge_data() 按边 ID 合并 —— 无需任何读写 tag 解耦。

运行时会输出三类中间过程：
    1) DEBUG 调度日志(Executor/Vertex/Edge)
    2) TraceAgent：每条边的 输入 → 当前图状态快照 → 输出
    3) 结束后每个顶点的完整中间数据，并保存校对 json

用法:
    python3 examples/trace/run.py             # Mock，逐步(sleep 0.4s)
    python3 examples/trace/run.py --fast      # 秒跑
    python3 examples/trace/run.py --real      # 真实 opencode agent
    python3 examples/trace/run.py --out x.json # 自定义校对文件路径
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from framework import Graph, Vertex, Edge, Executor, MockPIAgent, OpenCodeAgent
from framework.vertex import KEY_SELF


# ----------------------------------------------------------------------
# 1) TraceAgent —— 在每条边调用 agent 时打印「输入 → 当前图状态 → 输出」
# ----------------------------------------------------------------------
class TraceAgent(MockPIAgent):
    def __init__(self, graph, sleep: float = 0.4, real: bool = False, **kw):
        super().__init__()
        self.graph = graph
        self.sleep = sleep
        self.real = real
        self._real_agent = OpenCodeAgent(**kw) if real else None
        self._n = 0

    def _snapshot(self) -> str:
        """当前全图顶点状态快照。"""
        return "  ".join(
            f"{vid}={v.state.value[:4]}" for vid, v in self.graph.vertices.items()
        )

    async def process(self, data, prompt, model, settings=None) -> Any:
        self._n += 1
        print("\n" + "─" * 62)
        print(f"  ▶ agent#{self._n}   model={model}")
        print(f"    ├ prompt: {str(prompt)[:70]}")
        print(f"    ├ 输入数据: {repr(data)[:220]}")
        print(f"    ├ 图状态:   {self._snapshot()}")
        if self.real:
            result = await self._real_agent.process(data, prompt, model, settings)
        else:
            # Mock：根据 model 做简单变换，让结果可读
            if model == "fr":
                result = f"[法] {data}"
            elif model == "es":
                result = f"[西] {data}"
            elif model == "digest":
                result = f"[报告] {data}"
            else:
                result = f"[{model}] {data}"
            if self.sleep:
                await asyncio.sleep(self.sleep)  # 放慢节奏便于观察
        print(f"    └ 输出数据: {repr(result)[:220]}")
        return result


# ----------------------------------------------------------------------
# 2) MergeVertex —— 自定义顶点：fan-in 到齐后按来源边 ID 合并成 digest
# ----------------------------------------------------------------------
class MergeVertex(Vertex):
    async def _store(self, data):
        """直接写入自产槽 __self__，不触发状态机。"""
        async with self._lock:
            self._data_store[KEY_SELF] = data
        return data

    async def prepare_outputs(self):
        # 按来源边 ID 拿到每次写入(e4→法文, e5→西文)，合并成自产结果
        edge_data = await self.get_all_data()
        lines = [f"[{eid}] {val}" for eid, val in edge_data.items()]
        combined = "\n".join(lines)
        await self._store(combined)   # 合并结果 → __self__(主数据)
        print(f"  ── [{self.id}] 按边 ID 合并 {list(edge_data.keys())} → __self__:\n{combined}")
        await super().prepare_outputs()


# ----------------------------------------------------------------------
# 3) 组装图（全部普通 Edge，fan-in 靠按边 ID 记录）
# ----------------------------------------------------------------------
def add_edge(graph, edge, source, dest):
    graph.edges[edge.id] = edge
    source.outgoing_edges.append(edge.id)
    dest.incoming_edges.append(edge.id)
    dest.required_input_count = len(dest.incoming_edges)


def build_graph() -> Graph:
    graph = Graph()
    A = Vertex("A", initial_data=[{
        "value": "The vertex-edge framework lets you compose AI pipelines as a graph.",
    }])
    B, C, D = Vertex("B"), Vertex("C"), Vertex("D")
    E = MergeVertex("E")
    F = Vertex("F")
    graph.vertices = {"A": A, "B": B, "C": C, "D": D, "E": E, "F": F}

    # 全部普通 Edge：读源主数据，写目标按本边 ID 分槽；fan-in 靠边 ID 记录不覆盖
    add_edge(graph, Edge("e1", "A", "B",
                         prompt="Summarize the text:", model="sum"), A, B)
    add_edge(graph, Edge("e2", "B", "C",
                         prompt="Translate to French:", model="fr"), B, C)
    add_edge(graph, Edge("e3", "B", "D",
                         prompt="Translate to Spanish:", model="es"), B, D)
    add_edge(graph, Edge("e4", "C", "E", model="m"), C, E)
    add_edge(graph, Edge("e5", "D", "E", model="m"), D, E)
    add_edge(graph, Edge("e6", "E", "F",
                         prompt="Produce final report:", model="digest"), E, F)
    return graph


# ----------------------------------------------------------------------
# 4) 结果持久化：输入 json + 全部 vertex 结果，便于校对
# ----------------------------------------------------------------------
def serialize_input(graph) -> dict:
    """把图的结构(输入侧)序列化为 json 可读的 dict。"""
    return {
        "vertices": [
            {
                "id": v.id,
                "settings": v.settings,
                "script": v.script_path,
                "initial_data": [
                    {"data_id": k[0], "tags": list(k[1]), "value": val}
                    for k, val in v._data_store.items()
                ],
            }
            for v in graph.vertices.values()
        ],
        "edges": [
            {
                "id": e.id,
                "source": e.source_id,
                "destination": e.destination_id,
                "prompt": e.prompt,
                "model": e.model,
                "settings": e.settings,
                "script": e.script_path,
            }
            for e in graph.edges.values()
        ],
    }


def save_output(result, out_path: str, agent_desc: str, input_graph: dict) -> str:
    """把 输入 json(运行前快照) + 所有 vertex/edge 结果 保存为校对文件。"""
    payload = {
        "metadata": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "success": result.success,
            "execution_time_s": round(result.execution_time, 3),
            "agent": agent_desc,
            "graph": f"V={len(input_graph['vertices'])}, E={len(input_graph['edges'])}",
            "vertex_count": len(input_graph["vertices"]),
            "edge_count": len(input_graph["edges"]),
        },
        "input_graph": input_graph,
        "results": {
            "vertex_results": result.vertex_results,
            "edge_results": result.edge_results,
            "errors": result.errors,
        },
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
    return out_path


# ----------------------------------------------------------------------
# 5) 运行
# ----------------------------------------------------------------------
async def main() -> None:
    ap = argparse.ArgumentParser(description="Trace 示例：事件驱动中间过程")
    ap.add_argument("--out", default=None,
                    help="校对文件输出路径(默认 examples/trace/trace_result.json)")
    ap.add_argument("--fast", action="store_true", help="不 sleep，秒跑")
    ap.add_argument("--real", action="store_true", help="真实 opencode agent")
    ap.add_argument("--proxy", default="1", help="opencode 代理 1..6")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if not args.fast else logging.INFO,
        format="  %(levelname)-7s %(name)s | %(message)s",
    )

    graph = build_graph()
    print("=" * 62)
    print("  图: A →B→{C,D} →E→ F   (fan-out + fan-in, 按边 ID 记录)")
    print("  顶点:", list(graph.vertices))
    print("=" * 62)

    agent = TraceAgent(graph, sleep=(0 if args.fast else 0.4),
                       real=args.real, proxy=args.proxy)
    # 运行前：快照输入图(初始数据)，供校对
    input_snapshot = serialize_input(graph)
    result = await Executor(graph, agent, max_concurrency=3, timeout=60).run()

    print("\n" + "=" * 62)
    print("  结束：每个顶点的中间数据")
    print("=" * 62)
    for vid, info in result.vertex_results.items():
        print(f"\n  [{vid}] state={info['state']}")
        for key, val in info.get("data", {}).items():
            print(f"    {key}: {repr(val)[:200]}")

    print("\n" + result.summary())

    # 保存校对文件：输入 json(运行前快照) + 所有顶点/边结果
    out_path = args.out or os.path.join(os.path.dirname(__file__), "trace_result.json")
    saved = save_output(result, out_path, agent_desc=f"{args}", input_graph=input_snapshot)
    print(f"\n💾 校对文件已保存: {saved}")


if __name__ == "__main__":
    asyncio.run(main())
