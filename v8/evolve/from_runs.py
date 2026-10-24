"""从历史 dag8 run 直接生成人工评价包（不重跑模型）。"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from evolve.leakage import audit_optional
from evolve.review import build_review
from evolve.tasks import Task, check_output, generic_checks


def load_runs(db_path: str | Path, last: int = 5) -> List[Dict[str, Any]]:
    db = Path(db_path)
    if not db.is_file():
        raise FileNotFoundError(f"sqlite 不存在：{db}")
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT run_id, dag_name, config_path, started_at, finished_at, status, "
            "result_tag, result_content_id FROM runs WHERE status='ok' "
            "ORDER BY started_at DESC LIMIT ?",
            (max(1, int(last)),),
        ).fetchall()
        out = []
        for r in rows:
            content = None
            if r["result_content_id"] is not None:
                c = conn.execute(
                    "SELECT content FROM contents WHERE id=?", (r["result_content_id"],)
                ).fetchone()
                content = c["content"] if c else None
            if not content and r["result_tag"]:
                c = conn.execute(
                    "SELECT content FROM contents WHERE run_id=? AND tags LIKE ? "
                    "ORDER BY id DESC LIMIT 1",
                    (r["run_id"], f'%"{r["result_tag"]}"%'),
                ).fetchone()
                content = c["content"] if c else None
            if not content:
                continue
            elapsed = (r["finished_at"] or r["started_at"]) - r["started_at"]
            out.append({
                "run_id": r["run_id"],
                "dag_name": r["dag_name"],
                "config_path": r["config_path"],
                "started_at": r["started_at"],
                "elapsed": round(float(elapsed), 1),
                "output": content,
                "result_tag": r["result_tag"],
            })
        return out
    finally:
        conn.close()


def build_from_runs(db_path: str | Path, out_dir: str | Path, last: int = 5,
                    title: str = "历史 run 回顾",
                    source_path: Optional[str] = None,
                    answer_path: Optional[str] = None) -> Path:
    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)
    runs = load_runs(db_path, last=last)
    if not runs:
        raise RuntimeError("没有可评价的成功 run")

    task = Task(
        id="from-runs",
        title=title,
        config="",
        seed="",
        description="直接从 dag8.sqlite3 里取最近的成功 run，做人工横向对比。",
        criteria=[
            "覆盖：是否覆盖源材料的核心论点、案例与结论",
            "准确：人名/典故/术语/因果链是否与源材料一致",
            "结构：标题、表格、阶段、支柱、勘误表是否齐全",
            "可读：是否像给人读的笔记，而不是提纲或逐字稿",
        ],
        checks=generic_checks(),
    )

    records = []
    seen_outputs = set()
    for r in runs:
        digest = hashlib.sha256(r["output"].encode("utf-8")).hexdigest()
        if digest in seen_outputs:
            continue
        seen_outputs.add(digest)
        cdir = base / "candidates" / r["run_id"]
        cdir.mkdir(parents=True, exist_ok=True)
        (cdir / "output.md").write_text(r["output"], encoding="utf-8")
        cfg_text = ""
        if r.get("config_path"):
            cp = Path(r["config_path"])
            if cp.is_file():
                cfg_text = cp.read_text(encoding="utf-8", errors="replace")
        leakage = audit_optional(
            source_path, answer_path,
            {"prompt": cfg_text, "output": r["output"]},
        )
        auto = check_output(r["output"], task.checks, leakage=leakage)
        rec = {
            "candidate_id": r["run_id"],
            "name": f"{r['dag_name']} / {r['run_id']}",
            "status": "ok",
            "elapsed": r["elapsed"],
            "output_chars": len(r["output"].strip()),
            "output_path": str(cdir / "output.md"),
            "auto": auto,
            "leakage": leakage,
            "summary": {
                "name": r["dag_name"],
                "workers": None,
                "edges": None,
                "kinds": {},
                "prompt_chars": None,
                "models": [],
                "lineage": {"parent": None, "op": "existing-run",
                            "detail": r["config_path"]},
            },
            "error": None,
            "run_id": r["run_id"],
            "started_at": time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(r["started_at"])
            ),
        }
        (cdir / "record.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        records.append(rec)

    (base / "records.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return build_review(base, task, records)
