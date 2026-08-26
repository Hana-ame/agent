#!/usr/bin/env python3
"""运行 stage1st 资讯收集任务（代码扩展 Edge 版，agent 筛选 + 按边 ID 分发）。

流程：
    trigger(源,tick)
      └─[IndexEdge]      抓整个版块 index(全部帖) → index 顶点 (posts, [])
    index
      └─[SelectAIEdge]   agent 边：筛选 AI 相关 → 输出 JSON → picked 顶点 (picked, [])  ← 接收筛选结果
    picked
      ├─[ThreadFetchEdge#1] 按边ID分发帖子[0] → 爬最后页往回 → docs (post_1)   ← fan-out
      ├─[ThreadFetchEdge#2] 按边ID分发帖子[1]                 → docs (post_2)
      └─[ThreadFetchEdge#3] 按边ID分发帖子[2]                 → docs (post_3)
    docs(捕捉文本文档)
      ├─[ThreadAgentEdge#1] agent: 读文档 → AI总结 → summary (post_1)           ← fan-out
      ├─[ThreadAgentEdge#2]                              → summary (post_2)
      └─[ThreadAgentEdge#3]                              → summary (post_3)
    summary(fan-in → READY)
      └─[DigestEdge]    agent: 读全部 → AI汇总播报 → final

跑法:
    # Mock（最快，验证图结构）
    python3 examples/stage1st/run.py --mock

    # 真实（stage1st 真实爬 + opencode hy3-free + 代理）
    ../chatto-bot/.venv/bin/python examples/stage1st/run.py

    # 换 AI 后端 / 代理 / 分发数
    ../chatto-bot/.venv/bin/python examples/stage1st/run.py --agent pi --model DeepSeek-V4-Flash-0731-Event
    ../chatto-bot/.venv/bin/python examples/stage1st/run.py --proxy 3
    ../chatto-bot/.venv/bin/python examples/stage1st/run.py --threads 2
"""

import argparse
import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from framework import Graph, Vertex, Executor, MockPIAgent, OpenCodeAgent, PICLIPIAgent
from framework.signal import is_abort, abort_reason
from stage1st import (
    IndexEdge, SelectAIEdge, ThreadFetchEdge, ThreadAgentEdge, DigestEdge, FORUM_URL,
)


def add_edge(graph, edge, source, dest):
    """手动登记一条边的关联关系(模拟 Graph.from_dict 内部逻辑)。"""
    graph.edges[edge.id] = edge
    source.outgoing_edges.append(edge.id)
    dest.incoming_edges.append(edge.id)
    dest.required_input_count = len(dest.incoming_edges)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="stage1st 资讯收集(vertex-edge 版)")
    p.add_argument("--mock", action="store_true", help="用 mock 抓取 + Mock agent(不联网)")
    p.add_argument("--agent", default="opencode", choices=["opencode", "pi", "mock"],
                   help="AI 后端(默认 opencode)")
    p.add_argument("--model", default=None, help="模型名(默认 opencode-zen/hy3-free)")
    p.add_argument("--proxy", default="1", help="opencode 代理 1..6 / URL / off")
    p.add_argument("--threads", type=int, default=8, help="fan-out 期望边数(不足则 Abort)")
    p.add_argument("--concurrency", type=int, default=4,
                   help="最大并发边数(避免同时打爆 agent) ")
    p.add_argument("--timeout", type=float, default=None, help="整体超时(秒)")
    return p


async def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    graph = Graph()
    n = args.threads
    model = args.model or "opencode-zen/hy3-free"

    # 顶点：trigger → index → picked → docs → summary → final
    trigger = Vertex("trigger", initial_data=[{"data_id": "tick", "value": "go"}])
    index_v = Vertex("index")
    picked = Vertex("picked")
    docs = Vertex("docs")
    summary = Vertex("summary")
    final = Vertex("final")
    graph.vertices = {
        "trigger": trigger, "index": index_v, "picked": picked, "docs": docs,
        "summary": summary, "final": final,
    }

    # 1. 抓整个 index → index 顶点
    add_edge(graph, IndexEdge("index", "trigger", "index", forum_url=FORUM_URL),
             trigger, index_v)

    # 2. agent 筛选 AI 相关 → picked 顶点(接收筛选 JSON)；不足 n 个会 Abort
    add_edge(graph, SelectAIEdge(
        "select", "index", "picked", max_threads=n,
        prompt="从下面的 stage1st 帖子清单中，只挑选与 AI(人工智能/大模型/Agent/机器学习)相关的帖子。"
               "输出一个 JSON 数组，每个元素包含 tid、title、url，需要挑出恰好 %d 个；"
               "如果确实不够 %d 个，也要尽量多挑。不要输出任何其他文字。" % (n, n),
        model=model,
    ), index_v, picked)

    # 3. fan-out：按边 ID 分发第 1..n 个 → 纯脚本爬取 → docs 顶点(捕捉文本文档)
    for i in range(n):
        add_edge(graph, ThreadFetchEdge(
            f"fetch{i+1}", "picked", "docs", post_index=i,
            data_id=f"post_{i+1}", tags=[], recent_hours=24,
        ), picked, docs)

    # 4. fan-out：每帖一条 agent 边，读文档 → AI 总结 → summary
    for i in range(n):
        add_edge(graph, ThreadAgentEdge(
            f"sum{i+1}", "docs", "summary",
            data_id=f"post_{i+1}", tags=[],
            prompt="单独总结这个 stage1st 帖子的 AI 相关讨论要点，中文，≤150字",
            model=model,
        ), docs, summary)

    # 5. fan-in 汇总 → final
    add_edge(graph, DigestEdge(
        "digest", "summary", "final",
        prompt="把下面多个帖子的总结汇总成一段猫娘语气的 AI 资讯播报，条目清晰，中文",
        model=model,
    ), summary, final)

    graph.validate()
    print(f"▶ 图: {graph}")
    print(f"▶ 顶点: {list(graph.vertices)}")

    # agent 选择
    if args.mock or args.agent == "mock":
        agent = MockPIAgent()
        print("▶ Agent: Mock + mock 抓取")
    elif args.agent == "opencode":
        agent = OpenCodeAgent(model=model, proxy=args.proxy, timeout=480)
        print(f"▶ Agent: opencode({model}) proxy={agent.proxies}")
    else:
        agent = PICLIPIAgent(model=model, timeout=300, thinking="low")
        print("▶ Agent: pi({model})")

    timeout = args.timeout or (900 if not args.mock else 30)
    result = await Executor(graph, agent, max_concurrency=args.concurrency, timeout=timeout).run()

    print("\n" + result.summary())

    # 展示 picked 顶点的筛选结果 + 一份文本文档 + 最终播报
    print("\n" + "=" * 60)
    print("  SELECT AI 相关帖 (picked 顶点)")
    print("=" * 60)
    picked_data = result.vertex_results.get("picked", {}).get("data", {})
    for key, val in picked_data.items():
        print(f"── [{key}] ──")
        if isinstance(val, list):
            for i, t in enumerate(val, 1):
                print(f"  {i}. {t.get('title', '')}")
        else:
            print(f"  {val}")

    print("\n" + "=" * 60)
    print("  示例文本文档 (docs 顶点 post_1)")
    print("=" * 60)
    docs_data = result.vertex_results.get("docs", {}).get("data", {})
    for key, val in list(docs_data.items())[:1]:
        if is_abort(val):
            print(f"\n── [{key}] ──\n⛔ [ABORTED] {abort_reason(val)}")
        else:
            print(f"\n── [{key}] ──\n{val[:600]}")

    print("\n" + "=" * 60)
    print("  FINAL OUTPUT")
    print("=" * 60)
    data = result.vertex_results.get("final", {}).get("data", {})
    if not data:
        print("\n  (final 顶点无数据)")
    for key, val in data.items():
        if is_abort(val):
            print(f"\n── [{key}] ──\n⛔ ABORTED\n   reason: {abort_reason(val)}")
        else:
            print(f"\n── [{key}] ──\n{val}")
    return result


if __name__ == "__main__":
    asyncio.run(main())
