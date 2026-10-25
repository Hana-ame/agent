"""v9 推理周期 = 一张 v8 DAG。

两路 http 并行抓证据 → v7 图生成 → script 校验入库。
v9 只选实体对、设置 seed，不再自己编排调用。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
V8 = ROOT / "v8"
for p in (str(ROOT), str(V8)):
    if p not in sys.path:
        sys.path.insert(0, p)

import dag8  # noqa: E402

from v9.expand import wiki_url  # noqa: E402
from v9.knowledge import Knowledge  # noqa: E402

CFG = Path(__file__).resolve().parent / "configs" / "cycle.json"


class CycleEngine:
    def __init__(self, knowledge: Knowledge, *,
                 db_path: str | Path | None = None) -> None:
        self.k = knowledge
        self.db = Path(db_path or (ROOT / "v9" / "data" / "cycle.sqlite3"))

    def step(self) -> Dict[str, Any]:
        pairs = self.k.unexplored_pairs(1)
        if not pairs:
            return {"status": "no_pairs"}
        a, b = pairs[0]
        os.environ["V9_DB"] = self.k.path
        try:
            res = dag8.run(
                str(CFG), seed_content=a,
                extra_seeds={"b": b, "url_a": wiki_url(a), "url_b": wiki_url(b)},
                db_path=str(self.db), max_workers=2, rpm=3,
                use_cache=False, max_seconds=300,
            )
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "pair": [a, b],
                    "error": f"{type(exc).__name__}: {exc}"}
        if res.status != "ok" or not res.result:
            return {"status": "error", "pair": [a, b],
                    "error": res.summary()}
        data = json.loads(res.result)
        return {"pair": [a, b], **data}

    def run(self, cycles: int = 3) -> List[Dict[str, Any]]:
        out = []
        for _ in range(cycles):
            res = self.step()
            out.append(res)
            if res["status"] == "no_pairs":
                break
        return out
