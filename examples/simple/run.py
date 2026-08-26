#!/usr/bin/env python3
"""Run the simple pipeline example.

运行简单流水线示例(3 个顶点的线性管道)。
Usage:
    python examples/simple/run.py
"""

import asyncio
import logging
import os
import sys

# 允许从项目根目录运行
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from framework import Graph, Executor, MockPIAgent


def setup_logging():
    """配置日志输出级别与格式。"""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


async def main():
    setup_logging()
    logger = logging.getLogger("simple_example")

    # 加载配置文件中的图定义
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    logger.info("Loading graph from %s", config_path)

    graph = Graph.from_json(config_path)

    # 使用 MockPIAgent(安装 pi_agent 后可换成 ExternalPIAgent 接入真实模型)
    agent = MockPIAgent()
    executor = Executor(graph, pi_agent=agent, max_concurrency=4, timeout=30)

    result = await executor.run()

    # 打印执行结果摘要
    print("\n" + result.summary())
    return result


if __name__ == "__main__":
    asyncio.run(main())
