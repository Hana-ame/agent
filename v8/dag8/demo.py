"""完全离线的 demo：用 command 节点展示 tag 路由、波次并行与契约。

不调用任何模型，适合 onboarding、网关故障时回归、CI 冒烟。

    python -m dag8 demo
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

__all__ = ["build_config", "run_demo"]


def build_config() -> Dict[str, Any]:
    return {
        "name": "offline_demo",
        "seed_tag": "task",
        "seed_content": "标签驱动 DAG",
        "result_tag": "out",
        "edges": [
            {"id": "left", "kind": "command", "layer": 0,
             "input_tags": ["task"], "output_tags": ["a"],
             "command": "printf A"},
            {"id": "right", "kind": "command", "layer": 0,
             "input_tags": ["task"], "output_tags": ["b"],
             "command": "printf B"},
            {"id": "merge", "kind": "command", "layer": 1,
             "input_tags": ["a", "b"], "output_tags": ["out"],
             "command": "printf '%s|%s' \"{tag:a}\" \"{tag:b}\"",
             "expect": {"min_chars": 3}},
        ],
    }


def run_demo(max_workers: int = 2) -> Tuple[Any, Path]:
    """跑 demo，返回 (RunResult, 临时目录)。调用方负责清理目录。"""
    from dag8.scheduler import run as schedule_run

    tmp = Path(tempfile.mkdtemp(prefix="dag8_demo_"))
    cfg_path = tmp / "demo.json"
    cfg_path.write_text(json.dumps(build_config(), ensure_ascii=False), encoding="utf-8")
    res = schedule_run(cfg_path, db_path=tmp / "demo.sqlite3",
                       max_workers=max(1, int(max_workers)))
    return res, tmp
