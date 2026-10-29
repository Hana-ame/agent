"""把多个 evolve 评价包合并成一个（多次试验一起评估）。"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

from evolve.review import build_review
from evolve.tasks import Task


def merge_reviews(run_dirs: List[str | Path], out_dir: str | Path,
                  title: str = "多次试验合并评估") -> Path:
    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    task_data: Dict[str, Any] = {}
    for raw in run_dirs:
        rd = Path(raw)
        review_path = rd / "review.json"
        if not review_path.is_file():
            raise FileNotFoundError(f"没有 review.json：{review_path}")
        review = json.loads(review_path.read_text(encoding="utf-8"))
        if not task_data:
            task_data = review.get("task") or {}
        for c in review.get("candidates", []):
            cid = f"{rd.name}:{c['candidate_id']}"
            dest = base / "candidates" / cid.replace("/", "_").replace(":", "_")
            dest.mkdir(parents=True, exist_ok=True)
            src = Path(c.get("output_path", ""))
            out = dest / "output.md"
            if src.is_file():
                shutil.copy2(src, out)
            else:
                out.write_text(c.get("output", ""), encoding="utf-8")
            rec = dict(c)
            rec["candidate_id"] = cid
            rec["name"] = f"{rd.name} / {c.get('name','')}"
            rec["output_path"] = str(out)
            records.append(rec)

    task = Task(
        id=str(task_data.get("id") or "merge"),
        title=title,
        config=str(task_data.get("config") or ""),
        seed=str(task_data.get("seed") or ""),
        description=f"合并 {len(run_dirs)} 个评价包。",
        criteria=[str(x) for x in (task_data.get("criteria") or [])],
        checks={},
        source_path=str(task_data.get("source_path") or ""),
    )
    return build_review(base, task, records)
