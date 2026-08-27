#!/usr/bin/env python3
"""通用运行器：跑任意 config.json。

通用运行器：把任意 config.json 交给框架执行，支持多种 agent 后端。
(重新设计的日志)
    ① 图初始化完成 → 一口气打印整张图：顶点 + 边(ID/source/dest/tags/model)
    ② 每跑一条边   → 打印 ID、model、proxy(有则值/无则空)、prompt+data(逐行)
    ③ 运行成功     → 打印每个顶点的最终输出

用法:
    # 用 Mock agent（快速、无需联网、确定性输出）
    python3 scripts/run_graph.py examples/simple/config.json --agent mock

    # 用真实 opencode agent（opencode CLI 调用真实 LLM，默认 opencode-zen/hy3-free）
    python3 scripts/run_graph.py config.template.json --agent opencode

    # 用真实 pi agent（pi CLI 调用真实 LLM，较慢）
    python3 scripts/run_graph.py examples/complex/config.json --agent pi

    # 控制并发与超时 / 开启框架 DEBUG 日志查看调度细节
    python3 scripts/run_graph.py my_config.json --concurrency 4 --timeout 120 --verbose
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from framework import (
    Graph, Executor, PIAgent, MockPIAgent, PICLIPIAgent, OpenCodeAgent, OpenAIAgent,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run any vertex-edge agent config.json")
    p.add_argument("config", help="path to config.json")
    p.add_argument("--agent", default="opencode", choices=["mock", "pi", "opencode", "openai"],
                   help="agent backend (default: opencode → opencode-zen/hy3-free)")
    p.add_argument("--provider", default=None, help="pi provider (pi backend)")
    p.add_argument("--model", default=None,
                   help="model name (opencode default: opencode-zen/hy3-free; pi: plain id; opencode: provider/model)")
    p.add_argument("--proxy", default="1",
                   help="opencode proxy: 1..6 (default 1) or full URL; 'off' to disable; comma-list for failover")
    p.add_argument("--timeout", type=float, default=None,
                   help="overall execution timeout in seconds")
    p.add_argument("--concurrency", type=int, default=10,
                   help="max concurrent edges")
    p.add_argument("--verbose", action="store_true",
                   help="framework DEBUG logging (默认关闭；旧日志已降为 DEBUG 级别)")
    return p


# ----------------------------------------------------------------------
# ① 图初始化完成：一口气打印整张图（顶点 + 边：ID/source/dest/tags/model）
# ----------------------------------------------------------------------
def print_graph(graph: Graph) -> None:
    name = graph.metadata.get("name") or ""
    desc = graph.metadata.get("description") or ""
    print("=" * 64)
    print("图初始化完成" + (f"：{name}" if name else ""))
    if desc:
        print(f"  {desc}")

    print(f"  顶点 ({len(graph.vertices)}):")
    for v in graph.vertices.values():
        extra = ""
        if v.script_path:
            extra += f"  script={os.path.basename(v.script_path)}"
        print(f"    [{v.id}]  in={len(v.incoming_edges)}  out={len(v.outgoing_edges)}{extra}")

    print(f"  边 ({len(graph.edges)}):")
    for e in graph.edges.values():
        tag_parts = []
        if e.read_tags:
            tag_parts.append(f"get={','.join(e.read_tags)}")
        if e.set_tags:
            tag_parts.append(f"set={','.join(e.set_tags)}")
        tag_str = "  ".join(tag_parts) if tag_parts else "-"
        pt = "  [passthrough]" if e.passthrough else ""
        print(f"    [{e.id}]  {e.source_id} -> {e.destination_id}"
              f"  tags=({tag_str})  model={e.model}{pt}")
    print("=" * 64)


# ----------------------------------------------------------------------
# ② 每跑一条边：打印 ID、model、proxy、prompt+data(逐行)；成功后打印输出
# ----------------------------------------------------------------------
def _fmt(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)
    return str(value)


class LoggingAgent(PIAgent):
    """装饰一个真实 agent，在其每次调用前后打印边级日志。

    proxy_desc：代理描述；无代理时传 ""（需求：proxy=有则值 / 无则空）。
    """

    def __init__(self, inner: PIAgent, proxy_desc: str = ""):
        self._inner = inner
        self._proxy = proxy_desc
        self._n = 0

    async def process(self, data, prompt, model, settings=None) -> Any:
        self._n += 1
        edge_id = (settings or {}).get("edge_id", "?")
        print(f"\n▶ [{edge_id}]  agent#{self._n}")
        print(f"  model = {model}")
        print(f"  proxy = {self._proxy}")
        print("  prompt:")
        for ln in _fmt(prompt).splitlines():
            print(f"    {ln}")
        print("  data:")
        for ln in _fmt(data).splitlines():
            print(f"    {ln}")

        result = await self._inner.process(data, prompt, model, settings)

        print(f"[{edge_id}] → output:")
        for ln in _fmt(result).splitlines():
            print(f"    {ln}")
        return result


# ----------------------------------------------------------------------
# ③ 运行成功：打印每个顶点的最终输出
# ----------------------------------------------------------------------
def print_result(result) -> None:
    print("\n" + "=" * 64)
    if result.success:
        print("运行成功 ✓  每个顶点的最终输出")
    else:
        print("运行失败 ✗  每个顶点的数据（排查）")
    print("=" * 64)
    for vid, info in result.vertex_results.items():
        print(f"\n  [{vid}]  state={info['state']}")
        data = info.get("data", {})
        if not data:
            print("    (no data)")
        for key, val in data.items():
            print(f"    {key}:")
            for ln in _fmt(val).splitlines():
                print(f"      {ln}")
    if result.errors:
        print("\n  ERRORS:")
        for err in result.errors:
            print(f"    • {err}")


# ----------------------------------------------------------------------
async def main():
    args = build_parser().parse_args()

    # 旧框架日志全部为 DEBUG；默认 INFO 不显示，--verbose 才输出
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # 加载图
    graph = Graph.from_json(args.config)
    print_graph(graph)  # ① 初始化完成 → 一口气打印整张图

    # 构造 agent（按后端选择；默认 opencode + opencode-zen/hy3-free）
    if args.agent == "mock":
        inner = MockPIAgent()
        print("▶ Agent: Mock (deterministic, no network)")
        agent = LoggingAgent(inner, proxy_desc="")
    elif args.agent == "pi":
        # pi 后端：--model 默认是 opencode 格式，需显式传 pi 模型名或用 config 边级 model
        inner = PICLIPIAgent(provider=args.provider, model=args.model,
                             timeout=180, thinking="low")
        print("▶ Agent: REAL pi agent (pi CLI)")
        agent = LoggingAgent(inner, proxy_desc="")
    elif args.agent == "opencode":
        model = args.model or "opencode-zen/hy3-free"  # 默认免费模型
        inner = OpenCodeAgent(model=model, timeout=240, proxy=args.proxy)
        proxy_desc = inner.proxies[0] if inner.proxies else ""
        print(f"▶ Agent: REAL opencode agent (opencode CLI, model={model}, proxies={inner.proxies})")
        agent = LoggingAgent(inner, proxy_desc=proxy_desc)
    elif args.agent == "openai":
        model = args.model or "gpt-4o-mini"  # OpenAI 默认模型
        inner = OpenAIAgent(model=model, timeout=240, proxy=os.environ.get("OPENAI_PROXY"))
        pdesc = inner.proxy or os.environ.get("HTTPS_PROXY", "") or "(system)"
        print(f"▶ Agent: REAL OpenAI agent (REST API, model={model}, proxy={pdesc})")
        agent = LoggingAgent(inner, proxy_desc=pdesc)
    else:
        raise SystemExit(f"unknown agent backend: {args.agent}")

    # 执行
    timeout = args.timeout or (600 if args.agent != "mock" else 30)
    executor = Executor(
        graph, pi_agent=agent,
        max_concurrency=args.concurrency,
        timeout=timeout,
    )
    result = await executor.run()

    print_result(result)  # ③ 成功/失败：打印每个顶点输出

    return result


if __name__ == "__main__":
    asyncio.run(main())
