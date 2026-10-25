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
CFG_DERIVE = Path(__file__).resolve().parent / "configs" / "derive.json"


def find_chain(k: Knowledge):
    """找 A→B + B→C，且 A→C 未连、未试过。"""
    claims = k.claims()
    linked = {tuple(sorted((c["subject"], c["object"]))) for c in claims}
    for c1 in claims:
        for c2 in claims:
            if c1["object"] != c2["subject"]:
                continue
            a, b, c = c1["subject"], c1["object"], c2["object"]
            if a == c or tuple(sorted((a, c))) in linked:
                continue
            if k.is_explored(a, c):
                continue
            return a, b, c, c1, c2
    return None


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

    def derive_step(self) -> Dict[str, Any]:
        chain = find_chain(self.k)
        if chain is None:
            return {"status": "no_chain"}
        a, b, c, c1, c2 = chain
        os.environ["V9_DB"] = self.k.path
        res = dag8.run(
            str(CFG_DERIVE), seed_content=json.dumps(c1, ensure_ascii=False),
            extra_seeds={
                "c2_json": json.dumps(c2, ensure_ascii=False),
                "c1_id": str(c1["id"]), "c2_id": str(c2["id"]),
                "a": a, "c": c,
            },
            db_path=str(self.db), max_workers=1, rpm=3,
            use_cache=False, max_seconds=300,
        )
        if res.status != "ok" or not res.result:
            return {"status": "error", "pair": [a, c], "error": res.summary()}
        return {"pair": [a, c], **json.loads(res.result)}

    def derive(self, cycles: int = 2) -> List[Dict[str, Any]]:
        out = []
        for _ in range(cycles):
            res = self.derive_step()
            out.append(res)
            if res["status"] in ("no_chain", "error"):
                break
        return out

    def run(self, cycles: int = 3) -> List[Dict[str, Any]]:
        out = []
        for _ in range(cycles):
            res = self.step()
            out.append(res)
            if res["status"] == "no_pairs":
                break
        return out
