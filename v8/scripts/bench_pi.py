#!/usr/bin/env python3
"""基准：pi 子进程启动 vs pi 端到端 vs v7 直连端到端。

用于解释「v8 每个节点一次 pi」的固定开销，以及并行到底能省多少。

```bash
python scripts/bench_pi.py --runs 2
```
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

V8 = Path(__file__).resolve().parent.parent
ROOT = V8.parent
for p in (str(V8), str(V8 / "scripts"), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import dag8  # noqa: E402
from dag import DAG  # noqa: E402
from dag.llm import OpenAICompatibleClient  # noqa: E402
from dag8.pi import find_pi  # noqa: E402

GATEWAY = "https://laptop-8000.moonchan.xyz/v1"
MODEL = "deepseek-flash"
PROMPT = "只回复两个字：就绪"

ONE_EDGE = {
    "name": "bench_v7",
    "models": {"laptop": {"api": GATEWAY, "key": "not-needed", "model": MODEL}},
    "contents": [{"tags": ["task"], "contents": PROMPT, "gen": 0}],
    "edges": [{
        "id": "one", "layer": 0, "input_tags": ["task"], "output_tags": ["out"],
        "processor": {"model": "laptop", "prompt": PROMPT},
    }],
}


def bench_spawn(runs: int) -> list[float]:
    exe = find_pi()
    if not exe:
        return []
    lat = []
    for _ in range(runs):
        t0 = time.perf_counter()
        subprocess.run([exe, "--version"], capture_output=True, timeout=30)
        lat.append(time.perf_counter() - t0)
    return lat


def bench_pi(runs: int, timeout: float) -> list[tuple[bool, float]]:
    cfg = dag8.PiConfig(provider="laptop", model=MODEL, timeout=timeout, retries=1)
    agent = dag8.PiAgent(cfg)
    try:                       # 预热：排除首次冷启动
        agent.run(PROMPT, timeout=timeout)
    except Exception:          # noqa: BLE001
        pass
    out = []
    for _ in range(runs):
        t0 = time.perf_counter()
        ok = True
        try:
            agent.run(PROMPT, timeout=timeout)
        except Exception:      # noqa: BLE001
            ok = False
        out.append((ok, time.perf_counter() - t0))
    return out


def bench_v7(runs: int, tmp: Path) -> list[tuple[bool, float]]:
    path = tmp / "v7.json"
    path.write_text(json.dumps(ONE_EDGE, ensure_ascii=False), encoding="utf-8")

    def once() -> tuple[bool, float]:
        d = DAG(name="bench", models=str(path), llm_client=OpenAICompatibleClient(timeout=60))
        d.load(str(path))
        t0 = time.perf_counter()
        try:
            d.run(until_tag="out", max_steps=3)
            return True, time.perf_counter() - t0
        except Exception:  # noqa: BLE001
            return False, time.perf_counter() - t0

    try:
        once()
    except Exception:  # noqa: BLE001
        pass
    return [once() for _ in range(runs)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=60)
    args = ap.parse_args()

    outdir = V8 / "data" / "bench"
    outdir.mkdir(parents=True, exist_ok=True)

    spawn = bench_spawn(args.runs)
    pi = bench_pi(args.runs, args.timeout)
    v7 = bench_v7(args.runs, outdir)

    def med(xs):
        return statistics.median(xs) if xs else 0.0

    def ok_lat(rows):
        return [x for ok, x in rows if ok]

    def fmt(rows):
        return [f"{x:.1f}{'' if ok else '✗'}" for ok, x in rows]

    lines = [
        "# pi / v7 基准",
        "",
        f"- runs={args.runs}（预热 1 次后统计，✗ = 失败）",
        f"- pi 启动（--version）：p50 {med(spawn):.2f}s  {[round(x, 1) for x in spawn]}",
        f"- pi 端到端（一次 trivial 调用）：p50 {med(ok_lat(pi)):.2f}s  {fmt(pi)}",
        f"- v7 直连端到端（一次 trivial 调用）：p50 {med(ok_lat(v7)):.2f}s  {fmt(v7)}",
        f"- pi 相对 v7 的固定开销约：{med(ok_lat(pi)) - med(ok_lat(v7)):+.2f}s",
    ]
    report = "\n".join(lines) + "\n"
    (outdir / "report.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
