#!/usr/bin/env python3
"""Trace 示例：JSON 驱动 + 存档到新 json。

图定义在 config.json(读取，不修改)；运行结果存档到新的 trace_result.json。

log 分三块：
    1) 建图后：一口气打印整张图(顶点 + 每条边 ID/source/dest/tags/model)
    2) 每跑一条边：打印 ID、model、prompt+data(多行)、proxy(有则值/无则空)
    3) 运行成功：打印每个顶点的数据 + 存档到新 json

图(4 顶点 4 边)：多条边直接指向同一个 merge 顶点(fan-in)，按边 ID+tag 分槽不覆盖。

用法:
    python3 examples/trace/run.py             # 默认：真实 opencode agent(hy3-free)
    python3 examples/trace/run.py --mock      # Mock agent，快速验证，不联网
    python3 examples/trace/run.py --proxy 3   # 换代理出口
    python3 examples/trace/run.py --out x.json
"""

import argparse
import asyncio
import logging
import os
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from framework import (
    Graph, Executor, MockPIAgent, OpenCodeAgent,
    snapshot_initial_data, build_archive, save_archive,
)


# ----------------------------------------------------------------------
# 1) 建图后：一口气打印整张图(顶点 + 每条边 ID/source/dest/tags/model)
# ----------------------------------------------------------------------
def print_graph(graph: Graph) -> None:
    print("=" * 62)
    print("图结构")
    print("  顶点: " + ", ".join(graph.vertices))
    print("  边:")
    for e in graph.edges.values():
        tags = []
        if e.read_tags:
            tags.append(f"read={e.read_tags}")
        if e.set_tags:
            tags.append(f"set={e.set_tags}")
        tag_str = " ".join(tags) if tags else "-"
        pt = " [passthrough]" if e.passthrough else ""
        print(f"    [{e.id}] {e.source_id} -> {e.destination_id}"
              f"  tags=({tag_str})  model={e.model}{pt}")
    print("=" * 62)


# ----------------------------------------------------------------------
# 2) 每跑一条边：打印 ID、model、prompt+data(多行)、proxy
# ----------------------------------------------------------------------
class TraceAgent(MockPIAgent):
    def __init__(self, graph, real: bool = False, proxy_desc: str = "", **kw):
        super().__init__()
        self.graph = graph
        self.real = real
        self.proxy_desc = proxy_desc
        self._real_agent = OpenCodeAgent(**kw) if real else None
        self._n = 0

    async def process(self, data, prompt, model, settings=None) -> Any:
        self._n += 1
        edge_id = (settings or {}).get("edge_id", "?")
        print(f"\n▶ [{edge_id}]  agent#{self._n}")
        print(f"  model = {model}")
        print(f"  proxy = {self.proxy_desc}")
        print("  prompt:")
        for ln in str(prompt).split("\n"):
            print(f"    {ln}")
        print("  data:")
        for ln in str(data).split("\n"):
            print(f"    {ln}")

        if self.real:
            result = await self._real_agent.process(data, prompt, model, settings)
        else:
            result = await super().process(data, prompt, model, settings)

        print(f"[{edge_id}] → output:")
        for ln in str(result).split("\n"):
            print(f"      {ln}")
        return result


# ----------------------------------------------------------------------
# 3) 成功：打印每个顶点数据
# ----------------------------------------------------------------------
def print_result(result) -> None:
    print("\n" + "=" * 62)
    print("运行成功：每个顶点的数据")
    print("=" * 62)
    for vid, info in result.vertex_results.items():
        print(f"\n  [{vid}] state={info['state']}")
        for key, val in info.get("data", {}).items():
            if isinstance(val, str) and "\n" in val:
                print(f"    {key}:")
                for ln in val.split("\n"):
                    print(f"      {ln}")
            else:
                print(f"    {key}: {repr(val)[:200]}")


# ----------------------------------------------------------------------
# 存档(落盘)由框架提供：build_archive / save_archive / snapshot_initial_data
# 本示例只负责：运行前快照初始输入 → 跑图 → 存档到新 json，不碰原始 config.json
# ----------------------------------------------------------------------
async def main() -> None:
    ap = argparse.ArgumentParser(description="Trace 示例：JSON 驱动")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.json"),
                    help="图配置 json(读取,不改动)")
    ap.add_argument("--mock", action="store_true",
                    help="用 Mock agent(快速,不联网)；默认=真实 opencode")
    ap.add_argument("--proxy", default="1", help="opencode 代理 1..6")
    ap.add_argument("--timeout", type=float, default=None,
                    help="整体执行超时(秒；默认 mock 60 / real 900)")
    ap.add_argument("--out", default=None,
                    help="存档文件输出路径(默认 examples/trace/trace_result.json)")
    args = ap.parse_args()

    logging.disable(logging.CRITICAL)  # 屏蔽框架日志，trace 只用自定义 print 输出

    # ① 从 JSON 读取图(不改动原始 json)
    graph = Graph.from_json(args.config)
    print_graph(graph)

    # ② agent 选择：默认 real，--mock 切 mock；记录 proxy 描述
    if args.mock:
        agent = TraceAgent(graph, real=False, proxy_desc="")
        agent_desc = "mock"
        timeout = args.timeout or 60
    else:
        proxy_list = OpenCodeAgent(proxy=args.proxy).proxies
        proxy_desc = proxy_list[0] if proxy_list else ""
        agent = TraceAgent(graph, real=True, proxy_desc=proxy_desc,
                           proxy=args.proxy, timeout=480)
        agent_desc = f"opencode {proxy_desc}"
        timeout = args.timeout or 900

    # 运行前：快照每个顶点的初始数据(输入侧)—— 框架提供
    init_snapshot = snapshot_initial_data(graph)
    result = await Executor(graph, agent, max_concurrency=3, timeout=timeout).run()

    # ③ 成功：打印每个顶点数据
    print_result(result)

    # 存档：新建 json(输入+结果合并)，不碰原始 config.json —— 框架提供
    out_path = args.out or os.path.join(os.path.dirname(__file__), "trace_result.json")
    archive = build_archive(graph, result, init_snapshot, agent_desc)
    saved = save_archive(archive, out_path)
    print(f"\n💾 存档已保存(新文件): {saved}")


if __name__ == "__main__":
    asyncio.run(main())
