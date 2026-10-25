#!/usr/bin/env python3
"""对 configs/use 下三个 DAG 做参数搜索：模型 × workers × 重复。

质量用确定性 rubric，时延用墙钟；结果写 data/use_sweep.json。
"""
from __future__ import annotations

import copy
import json
import statistics
import sys
import time
from pathlib import Path

V8 = Path(__file__).resolve().parent.parent
for p in (str(V8), str(V8.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

import dag8  # noqa: E402

USE = V8 / "configs" / "use"
DB = V8 / "data" / "use_sweep.sqlite3"
MODELS = ["deepseek-flash", "deepseek-v4-flash"]
RUNS = 2


def score(name: str, text: str) -> int:
    t = text or ""
    lines = [x for x in t.splitlines() if x.strip()]
    if name == "research_brief":
        s = 0
        s += 1 if len(lines) >= 3 else 0
        s += 1 if ("有向无环图" in t or "DAG" in t) else 0
        s += 1 if "拓扑排序" in t else 0
        s += 1 if all(len(x) <= 90 for x in lines) else 0
        return s
    if name == "sqlite_report":
        s = 0
        s += 1 if "华东" in t else 0
        s += 1 if ("西南" in t or "最低" in t) else 0
        s += 1 if len(t) <= 60 else 0
        return s
    s = 0
    s += 1 if "字符" in t else 0
    s += 1 if ("行" in t or "词" in t) else 0
    s += 1 if len(t) <= 80 else 0
    return s


def main() -> int:
    rows = []
    for name in ("research_brief", "sqlite_report", "script_judge"):
        base = json.loads((USE / f"{name}.json").read_text(encoding="utf-8"))
        workers_list = [1, 2] if name == "research_brief" else [1]
        for model in MODELS:
            for workers in workers_list:
                for run in range(RUNS):
                    cfg = copy.deepcopy(base)
                    cfg["pi"] = {"provider": "laptop", "model": model,
                                 "tools": False, "timeout": 180, "retries": 2}
                    tmp = USE / f".sweep_{name}.json"
                    tmp.write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
                    t0 = time.perf_counter()
                    try:
                        res = dag8.run(tmp, db_path=DB, max_workers=workers,
                                       rpm=3, max_seconds=240)
                        wall = time.perf_counter() - t0
                        rows.append({
                            "dag": name, "model": model, "workers": workers,
                            "run": run + 1, "status": res.status,
                            "wall": round(wall, 2),
                            "quality": score(name, res.result or ""),
                            "result": (res.result or "")[:200],
                        })
                    except Exception as exc:  # noqa: BLE001
                        rows.append({
                            "dag": name, "model": model, "workers": workers,
                            "run": run + 1, "status": "exception",
                            "wall": round(time.perf_counter() - t0, 2),
                            "quality": 0, "result": f"{type(exc).__name__}: {exc}",
                        })
                    print(f"{name:16s} {model:20s} w{workers} r{run+1} "
                          f"{rows[-1]['status']:6s} {rows[-1]['wall']:6.1f}s "
                          f"q={rows[-1]['quality']}", flush=True)
    tmp = USE / ".sweep_research_brief.json"
    if tmp.exists():
        tmp.unlink()

    (V8 / "data" / "use_sweep.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n== summary ==")
    for name in ("research_brief", "sqlite_report", "script_judge"):
        best = None
        for model in MODELS:
            subset = [r for r in rows if r["dag"] == name and r["model"] == model]
            if not subset:
                continue
            q = statistics.mean(r["quality"] for r in subset)
            p50 = statistics.median(r["wall"] for r in subset)
            ok = sum(1 for r in subset if r["status"] == "ok")
            print(f"{name:16s} {model:20s} q={q:.2f} p50={p50:.1f}s "
                  f"ok={ok}/{len(subset)}")
            if best is None or (q, -p50) > (best[0], -best[1]):
                best = (q, p50, model)
        if best:
            print(f"  -> best {name}: {best[2]} q={best[0]:.2f} p50={best[1]:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
