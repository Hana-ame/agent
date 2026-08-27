#!/usr/bin/env python3
"""Run the complex DAG pipeline with a REAL coding agent.

用真实 coding agent 运行复杂 DAG 流水线。
目的：多源科技情报分析 —— 把一篇新闻 + 分析关注点送入图，
经过 fan-out/fan-in 汇聚后生成一份结构化的周报摘要。

默认后端：opencode CLI，免费模型 opencode-zen/hy3-free。

Demonstrates:  演示内容：
    • Multiple source vertices            多个源顶点
    • Fan-out (input_a → transform AND merge)  扇出
    • Fan-in  (transform + input_a → merge)    扇入
    • Real LLM via opencode CLI          通过 opencode 命令行调用真实模型
    • External vertex/edge scripts        外部顶点/边脚本
    • Concurrent edge execution           并发的边执行

Usage:
    python examples/complex/run.py
"""

import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from framework import Graph, Executor, OpenCodeAgent, PICLIPIAgent


# 选择后端：默认 opencode；想用 pi 时把 USE_PI 设为 True
USE_PI = False


def setup_logging():
    """配置日志输出级别与格式。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


async def main():
    setup_logging()
    logger = logging.getLogger("complex_example")

    # 加载配置文件中的图定义
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    logger.debug("Loading graph from %s", config_path)

    graph = Graph.from_json(config_path)

    # 真实 agent：默认 opencode（免费模型 opencode-zen/hy3-free）
    if USE_PI:
        agent = PICLIPIAgent(provider="scnet", model="DeepSeek-V4-Flash-0731-Event", timeout=180, thinking="low")
        backend = "pi"
    else:
        agent = OpenCodeAgent(timeout=240)   # 默认模型 opencode-zen/hy3-free
        backend = "opencode"
    executor = Executor(graph, pi_agent=agent, max_concurrency=8, timeout=600)

    logger.debug("▶ Running with real %s agent (this calls the LLM 5 times)...", backend)
    result = await executor.run()

    print("\n" + result.summary())

    # 打印最终周报全文(放在 output 顶点 final:report)
    print("\n" + "=" * 60)
    print("  FINAL INTELLIGENCE DIGEST (raw output)")
    print("=" * 60)
    out_data = result.vertex_results.get("output", {}).get("data", {})
    for key, val in out_data.items():
        print(f"\n── [{key}] ──\n{val}")

    return result


if __name__ == "__main__":
    asyncio.run(main())
