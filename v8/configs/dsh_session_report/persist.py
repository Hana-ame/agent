#!/usr/bin/env python3
"""v8 script 节点：把单 session 复盘报告落盘，并写 v9 知识库。"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from v9.knowledge import Knowledge  # noqa: E402


def main() -> int:
    data = json.loads(os.environ["DAG8_INPUTS"])
    report = str(data.get("report") or "").strip()
    try:
        session = json.loads(data.get("session") or "{}")
    except ValueError:
        session = {}
    sid = str(session.get("session_id") or "")
    title = str(session.get("title") or sid or "DSH session")
    if not report:
        raise SystemExit("report 为空")

    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    slug = sid.replace("session-", "")[:12] or "unknown"
    path = reports / f"session_{slug}_report.md"
    path.write_text(report, encoding="utf-8")
    (reports / "dsh_session_report_latest.md").write_text(report, encoding="utf-8")
    (reports / f"session_{slug}.json").write_text(
        json.dumps(session, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    db = ROOT / "v9" / "data" / "dsh_session_report.sqlite3"
    db.parent.mkdir(parents=True, exist_ok=True)
    k = Knowledge(str(db))
    claims = 0
    try:
        source = f"dsh-session://{sid}"
        k.add_entity(title, source_url=source)
        conclusion = str(session.get("conclusion") or "").strip()
        if conclusion:
            k.add_claim(title, "session_conclusion", sid, conclusion,
                        confidence=1.0, source_url=source)
            claims = 1
        stats = k.stats()
    finally:
        k.close()

    print(json.dumps({
        "status": "ok",
        "session": sid,
        "report": str(path),
        "latest": str(reports / "dsh_session_report_latest.md"),
        "claims": claims,
        "db": str(db),
        "stats": stats,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())