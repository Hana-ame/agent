"""执行一个 Genome，产出可人工评价的输出 + 自动检查。"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import dag8

from evolve.genome import Genome
from evolve.leakage import audit_optional
from evolve.tasks import Task, check_output


def run_candidate(
    genome: Genome,
    task: Task,
    out_dir: str | Path,
    *,
    agent: Optional[Any] = None,
    use_cache: bool = True,
    source_path: Optional[str] = None,
    answer_path: Optional[str] = None,
) -> Dict[str, Any]:
    """跑一个候选，落 output.md / record.json，返回 record。"""
    base = Path(out_dir) / "candidates" / genome.id
    base.mkdir(parents=True, exist_ok=True)
    record: Dict[str, Any] = {
        "candidate_id": genome.id,
        "name": genome.name,
        "status": "error",
        "elapsed": 0.0,
        "output_chars": 0,
        "output_path": str(base / "output.md"),
        "auto": {"score": 0.0, "passed": 0, "total": 0, "checks": []},
        "summary": genome.summary(),
        "error": None,
    }
    t0 = time.perf_counter()
    try:
        cfg = genome.materialize(base)
        res = dag8.run(
            cfg,
            seed_content=task.seed or None,
            db_path=base / "run.sqlite3",
            max_workers=genome.workers,
            use_cache=use_cache,
            agent=agent,
        )
        record["status"] = res.status
        record["run_id"] = res.run_id
        record["elapsed"] = round(time.perf_counter() - t0, 1)
        output = res.result or ""
        record["output_chars"] = len(output.strip())
        (base / "output.md").write_text(output + ("\n" if output else ""), encoding="utf-8")
        leakage = audit_optional(
            source_path or task.source_path,
            answer_path,
            {"prompt": json.dumps(genome.config, ensure_ascii=False), "output": output},
        )
        record["leakage"] = leakage
        record["auto"] = check_output(output, task.checks, leakage=leakage)
        if res.status != "ok":
            record["error"] = "; ".join(
                f"{e.edge_id}: {e.error}" for e in res.edges if e.status == "error"
            )[:2000] or res.status
    except Exception as exc:  # noqa: BLE001 - 失败也要进评价包，标 error
        record["elapsed"] = round(time.perf_counter() - t0, 1)
        record["error"] = f"{type(exc).__name__}: {exc}"
        (base / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        (base / "output.md").write_text("", encoding="utf-8")

    (base / "record.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return record
