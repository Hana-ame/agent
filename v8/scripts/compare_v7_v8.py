#!/usr/bin/env python3
"""对照实验：同一张 fan-out 图，v7 串行 vs v8 波次并行（convert 后）。

图：task → def / benefit（同层，可并行）→ final（合并）。

```bash
python scripts/compare_v7_v8.py --runs 2 --workers 2 --rpm 3
```
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

V8 = Path(__file__).resolve().parent.parent
ROOT = V8.parent
for p in (str(V8), str(V8 / "scripts"), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import dag8  # noqa: E402
import search_schedule as ss  # noqa: E402
from dag import DAG  # noqa: E402
from dag.llm import OpenAICompatibleClient  # noqa: E402
from dag8.convert import convert_v7, convert_v7_to_graphs  # noqa: E402
from dag8.ratelimit import shared_limiter  # noqa: E402

GATEWAY = "https://laptop-8000.moonchan.xyz/v1"
MODEL = "deepseek-flash"
TASK = "用三句话解释「标签驱动的 DAG」"

def make_v7(width: int = 2) -> dict:
    width = max(1, width)
    branches = []
    for i in range(width):
        branches.append({
            "id": f"b{i}", "layer": 0, "input_tags": ["task"],
            "output_tags": [f"t{i}"],
            "processor": {"model": "laptop",
                          "prompt": f"用一句话说明「标签驱动的 DAG」的第 {i+1} 个好处"},
        })
    branches.append({
        "id": "final", "layer": 1,
        "input_tags": [f"t{i}" for i in range(width)],
        "output_tags": ["final"],
        "processor": {"model": "laptop",
                      "prompt": "把下面几句话合并成一段 80 字以内的解释，只输出正文：\n{contents}"},
    })
    return {
        "name": f"fanout_v7_w{width}",
        "models": {"laptop": {"api": GATEWAY, "key": "not-needed", "model": MODEL}},
        "contents": [{"tags": ["task"], "contents": TASK, "gen": 0}],
        "edges": branches,
    }


V7 = {
    "name": "fanout_v7",
    "models": {"laptop": {"api": GATEWAY, "key": "not-needed", "model": MODEL}},
    "contents": [{"tags": ["task"], "contents": TASK, "gen": 0}],
    "edges": [
        {"id": "def", "layer": 0, "input_tags": ["task"], "output_tags": ["def"],
         "processor": {"model": "laptop", "prompt": "用一句话定义「标签驱动的 DAG」"}},
        {"id": "benefit", "layer": 0, "input_tags": ["task"], "output_tags": ["benefit"],
         "processor": {"model": "laptop", "prompt": "用一句话说明标签驱动 DAG 的好处"}},
        {"id": "final", "layer": 1, "input_tags": ["def", "benefit"], "output_tags": ["final"],
         "processor": {"model": "laptop",
                       "prompt": "把下面两句话合并成一段 80 字以内的解释，只输出正文：\n{contents}"}},
    ],
}


def _v7_client(rpm: int):
    lim = shared_limiter(rpm)

    class _Client(OpenAICompatibleClient):
        def _blocking_once(self, *a, **k):
            lim.acquire()
            return super()._blocking_once(*a, **k)

    return _Client(timeout=120, max_retries=4)


def run_v7(path: Path, runs: int, rpm: int):
    lat, outs, failed = [], [], 0
    for _ in range(runs):
        out = ""
        t0 = time.perf_counter()
        for attempt in range(2):
            try:
                d = DAG(name="v7_fanout", models=str(path), llm_client=_v7_client(rpm))
                d.load(str(path))
                d.run(until_tag="final", max_steps=10)
                out = d.get_result("final") or ""
                break
            except Exception:  # noqa: BLE001 - 429 等，重试一次
                if attempt == 1:
                    failed += 1
                else:
                    time.sleep(10)
        lat.append(time.perf_counter() - t0)
        outs.append(out)
    return lat, outs, failed


def run_v8(cfg_path: Path, runs: int, workers: int, rpm: int):
    lat, outs = [], []
    for i in range(runs):
        t0 = time.perf_counter()
        res = dag8.run(str(cfg_path), db_path=str(cfg_path.parent / f"v8_{i}.sqlite3"),
                       max_workers=workers, rpm=rpm, max_seconds=300)
        lat.append(time.perf_counter() - t0)
        outs.append(res.result or "")
    return lat, outs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--rpm", type=int, default=3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--width", type=int, default=2)
    ap.add_argument("--mode", choices=["pi", "graph"], default="pi",
                    help="pi = v8 pi 节点；graph = 每条边一张单边 v7 子图")
    args = ap.parse_args()

    global V7
    V7 = make_v7(args.width)
    outdir = Path(args.out) if args.out else V8 / "data" / "compare-v7-v8"
    outdir.mkdir(parents=True, exist_ok=True)
    v7_path = outdir / "v7.json"
    v7_path.write_text(json.dumps(V7, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.mode == "graph":
        cfg, subs = convert_v7_to_graphs(V7, subdir="graphs", provider="laptop",
                                         name="fanout_v8_graphs")
        (outdir / "graphs").mkdir(exist_ok=True)
        for name, sub in subs.items():
            (outdir / "graphs" / name).write_text(
                json.dumps(sub, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        cfg = convert_v7(V7, provider="laptop", name="fanout_v8")
        cfg["pi"]["model"] = MODEL
    cfg_path = outdir / "v8.json"
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    judge = ss.make_judge(gap=2)
    v7_lat, v7_out, v8_lat, v8_out = [], [], [], []
    pairs = []
    v7_failed = 0
    for _ in range(args.runs):
        # 交错执行：同一时间窗内先 v7 再 v8，避免网关漂移只打一边
        a, ao, af = run_v7(v7_path, 1, args.rpm)
        b, bo = run_v8(cfg_path, 1, args.workers, args.rpm)
        v7_lat += a
        v7_out += ao
        v8_lat += b
        v8_out += bo
        v7_failed += af
        if af == 0 and b and b[0] > 0:
            pairs.append(a[0] / b[0])

    q7 = [judge(o)[0] or 0 for o in v7_out if o]
    q8 = [judge(o)[0] or 0 for o in v8_out if o]

    def med(xs):
        return statistics.median(xs) if xs else 0.0

    report = [
        "# v7 串行 vs v8 波次并行",
        "",
        f"- runs={args.runs} workers={args.workers} rpm={args.rpm} mode={args.mode}",
        f"- v7: 时延 p50 {med(v7_lat):.1f}s，质量 {med(q7):.1f}，失败 {v7_failed}/{args.runs}",
        f"- v8: 时延 p50 {med(v8_lat):.1f}s，质量 {med(q8):.1f}",
        f"- 配对加速比（每次 v7/v8）：{[round(x, 2) for x in pairs]}，"
        f"p50 {med(pairs):.2f}×",
        f"- 比值口径（p50 之比）：{(med(v7_lat) / med(v8_lat)) if med(v8_lat) else 0:.2f}×",
        "",
        "## v8 输出",
        "",
        v8_out[-1] if v8_out else "（空）",
    ]
    (outdir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
