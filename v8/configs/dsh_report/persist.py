#!/usr/bin/env python3
"""v8 script 节点：把报告与结构化结论落盘，同时写 v9 知识库。

输入：DAG8_INPUTS 里的 report、o1..o6。
输出 artifact：
- reports/dsh_sessions_report.md
- reports/dsh_sessions_structured.json
- v9/data/dsh_report.sqlite3
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from v9.knowledge import Knowledge  # noqa: E402

N = 6


def load_item(raw: str) -> dict:
    try:
        value = json.loads(raw)
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def main() -> int:
    data = json.loads(os.environ["DAG8_INPUTS"])
    report = str(data.get("report") or "")
    items = []
    for i in range(1, N + 1):
        item = load_item(data.get(f"o{i}", ""))
        if item.get("session_id"):
            items.append(item)

    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    if not report.strip():
        report = "# DSH 历史会话报告\n\n（空：merge 没有产出正文）\n"
    (reports / "dsh_sessions_report.md").write_text(report, encoding="utf-8")
    (reports / "dsh_sessions_structured.json").write_text(
        json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    db = ROOT / "v9" / "data" / "dsh_report.sqlite3"
    db.parent.mkdir(parents=True, exist_ok=True)
    k = Knowledge(str(db))
    claims = 0
    try:
        for item in items:
            sid = str(item["session_id"])
            topic = str(item.get("topic") or sid)
            conclusion = str(item.get("conclusion") or "").strip()
            source = f"dsh-session://{sid}"
            k.add_entity(topic, source_url=source)
            if conclusion:
                k.add_claim(topic, "session_conclusion", sid, conclusion,
                            confidence=1.0, source_url=source)
                claims += 1
        stats = k.stats()
    finally:
        k.close()

    print(json.dumps({
        "status": "ok",
        "sessions": len(items),
        "claims": claims,
        "db": str(db),
        "report": str(reports / "dsh_sessions_report.md"),
        "stats": stats,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())