#!/usr/bin/env python3
"""S1 AI Report (MapEdge) — local run.

Runs ``config.json`` — a MapEdge clone of s1_ai_report following the
hn_ai_report pattern: fetch thread list -> LLM filter -> ProcessThreadsMap
(concurrent fetch-replies + summarize pipeline) -> report.md.

The LLM agent (HttpLLMAgent -> free hy3-free on opencode.ai/zen) is passed at
the executor level, exactly like ``hn_ai_report/demo.py``. Set HTTPS_PROXY to
the load-balanced Clash pool for the egress if needed.

    python examples/s1_ai_report_map/demo.py
"""

import asyncio
import json as _json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.basicConfig(level=logging.INFO)

from framework.graph import Graph
from framework.executor.base import Executor
from framework.agents import HttpLLMAgent


def _read_endpoint_from_config(config_path: str) -> str:
    """Return the *explicit* LLM endpoint declared in config.json settings.

    The framework does NOT guess/fallback-fill an endpoint — the config must
    carry the complete URL (including ``/chat/completions``) in the settings of
    an ``llm`` edge or pipeline step. Raises if none is declared.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = _json.load(f)

    def _first(settings: dict):
        s = settings or {}
        if s.get("base_url"):
            return s["base_url"]
        for step in (s.get("pipeline") or []):
            ss = step.get("settings") or {}
            if ss.get("base_url"):
                return ss["base_url"]
        return None

    for e in cfg.get("edges", []):
        found = _first(e.get("settings") or {})
        if found:
            return found
    raise SystemExit(
        "config.json: no explicit base_url endpoint in settings "
        "(require full URL e.g. https://opencode.ai/zen/v1/chat/completions)"
    )


async def main():
    print("Loading graph from config.json...")
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    g = Graph.from_json(config_path)

    # endpoint must be explicit in config.json settings — no default fallback.
    base_url = _read_endpoint_from_config(config_path)
    agent = HttpLLMAgent(
        api_key=os.environ.get("LLM_API_KEY", "public"),
        base_url=base_url,
    )

    print("Executing graph with HttpLLMAgent...")
    try:
        executor = Executor(
            g,
            agents=agent,
            max_concurrency=10,
            concurrency_config={"llm": 1, "fetch": 10, "default": 10},
        )
        await executor.run()
    finally:
        usage = agent.get_usage_summary()
        print("\n=== REAL TOKEN USAGE (from upstream responses) ===")
        print(usage)
        await agent.close()

    report_path = os.path.join(os.path.dirname(__file__), "report.md")
    if os.path.exists(report_path):
        print("\n=== SUCCESS: GENERATED REPORT ===")
        with open(report_path, "r", encoding="utf-8") as f:
            print(f.read())
    else:
        print("\nERROR: Report file was not generated.")


if __name__ == "__main__":
    asyncio.run(main())
