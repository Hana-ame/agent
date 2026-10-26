#!/usr/bin/env python3
"""驱动 DSH session 报告：

1. extract.py 抽取历史 session 结论
2. dag8.run(dag.json) 并行 6 个 v7 单 session 图 → merge → persist
3. persist.py 写 Markdown、结构化 JSON、v9 知识库
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V8 = ROOT / "v8"
for p in (str(ROOT), str(V8)):
    if p not in sys.path:
        sys.path.insert(0, p)

import dag8  # noqa: E402
from v7.llm import OpenAICompatibleClient  # noqa: E402

HERE = Path(__file__).resolve().parent
CFG = HERE / "dag.json"
EXTRACT = HERE / "extract.py"
N = 6
PLACEHOLDER = {
    "session_id": "__empty__", "title": "（空位）", "description": "",
    "turns": 0, "tools": 0, "status": "",
    "request": "", "conclusion": "（没有更多历史 session）",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=N)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--rpm", type=int, default=1)
    ap.add_argument("--max-seconds", type=float, default=1800)
    args = ap.parse_args()
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    input_path = reports / "dsh_sessions_input.json"

    subprocess.run(
        [sys.executable, str(EXTRACT), "--limit", str(args.limit), "--out", str(input_path)],
        check=True, cwd=str(ROOT),
    )
    sessions = json.loads(input_path.read_text(encoding="utf-8"))["sessions"]
    while len(sessions) < args.limit:
        sessions.append(dict(PLACEHOLDER, session_id=f"__empty_{len(sessions)}__"))

    seeds = {f"s{i + 1}": json.dumps(s, ensure_ascii=False)
             for i, s in enumerate(sessions[:args.limit])}
    res = dag8.run(
        str(CFG),
        seed_content=seeds["s1"],
        extra_seeds={k: v for k, v in seeds.items() if k != "s1"},
        db_path=str(ROOT / "v8" / "data" / "dsh_report.sqlite3"),
        max_workers=args.workers, rpm=args.rpm, use_cache=False,
        max_seconds=args.max_seconds, keep_going=True,
        graph_llm_client=OpenAICompatibleClient(timeout=300, max_retries=5),
    )

    report_path = reports / "dsh_sessions_report.md"
    report_text = ""
    try:
        import sqlite3
        con = sqlite3.connect(str(ROOT / "v8" / "data" / "dsh_report.sqlite3"))
        row = con.execute(
            "SELECT content FROM contents WHERE run_id=? AND tags LIKE '%report%' "
            "ORDER BY id DESC LIMIT 1", (res.run_id,)).fetchone()
        con.close()
        if row and row[0].strip():
            report_text = row[0]
    except Exception:  # noqa: BLE001
        report_text = ""
    if report_text:
        report_path.write_text(report_text, encoding="utf-8")

    run_info = {
        "status": res.status,
        "run_id": getattr(res, "run_id", ""),
        "workers_used": getattr(res, "workers_used", []),
        "edges": [
            {"edge": e.edge_id, "status": e.status,
             "elapsed": round(e.elapsed or 0, 2),
             "error": (e.error or "")[:200]}
            for e in res.edges
        ],
        "report": str(report_path),
    }
    (reports / "dsh_report_run.json").write_text(
        json.dumps(run_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(run_info, ensure_ascii=False, indent=2))
    return 0 if res.status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())