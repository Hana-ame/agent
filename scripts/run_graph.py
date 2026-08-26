#!/usr/bin/env python3
"""通用运行器：跑任意 config.json。

通用运行器：把任意 config.json 交给框架执行，支持多种 agent 后端。

用法:
    # 用 Mock agent（快速、无需联网、确定性输出）
    python3 scripts/run_graph.py examples/simple/config.json

    # 用真实 pi agent（pi CLI 调用真实 LLM，较慢）
    python3 scripts/run_graph.py examples/complex/config.json --agent pi

    # 用真实 opencode agent（opencode CLI 调用真实 LLM，默认 opencode-zen/hy3-free）
    python3 scripts/run_graph.py config.template.json --agent opencode

    # 控制并发与超时
    python3 scripts/run_graph.py my_config.json --concurrency 4 --timeout 120

    # 开启 DEBUG 日志查看调度细节
    python3 scripts/run_graph.py my_config.json --verbose
"""

import argparse
import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from framework import Graph, Executor, MockPIAgent, PICLIPIAgent, OpenCodeAgent


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run any vertex-edge agent config.json")
    p.add_argument("config", help="path to config.json")
    p.add_argument("--agent", default="opencode", choices=["mock", "pi", "opencode"],
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
    p.add_argument("--verbose", action="store_true", help="DEBUG logging")
    return p


async def main():
    args = build_parser().parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # 加载图
    graph = Graph.from_json(args.config)
    print(f"▶ Loaded {args.config}: {graph}")

    # 构造 agent（按后端选择；默认 opencode + opencode-zen/hy3-free）
    if args.agent == "mock":
        agent = MockPIAgent()
        print("▶ Agent: Mock (deterministic, no network)")
    elif args.agent == "pi":
        # pi 后端：--model 默认是 opencode 格式，需显式传 pi 模型名或用 config 边级 model
        agent = PICLIPIAgent(provider=args.provider, model=args.model,
                             timeout=180, thinking="low")
        print("▶ Agent: REAL pi agent (pi CLI)")
    elif args.agent == "opencode":
        model = args.model or "opencode-zen/hy3-free"  # 默认免费模型
        agent = OpenCodeAgent(model=model, timeout=240, proxy=args.proxy)
        print(f"▶ Agent: REAL opencode agent (opencode CLI, model={model}, proxies={agent.proxies})")
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

    print("\n" + result.summary())

    # 兜底：若失败，打印各顶点最终数据便于排查
    if not result.success:
        print("\n── vertex data dump ──")
        for vid, info in result.vertex_results.items():
            print(f"  [{vid}] state={info['state']} keys={list(info.get('data', {}).keys())}")
        for err in result.errors:
            print(f"  ERROR: {err}")

    return result


if __name__ == "__main__":
    asyncio.run(main())
