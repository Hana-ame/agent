#!/usr/bin/env python3
"""v8 script 节点：把多跳结果写入知识库，标 derived_from。"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from v9.knowledge import Knowledge  # noqa: E402


def main() -> int:
    data = json.loads(os.environ["DAG8_INPUTS"])
    claim = json.loads(data["claim_json"])
    a, c = data["a"], data["c"]
    c1, c2 = int(data["c1_id"]), int(data["c2_id"])
    k = Knowledge(os.environ.get("V9_DB", str(ROOT / "v9" / "data" / "knowledge.sqlite3")))
    try:
        relation = str(claim.get("relation") or "none").strip()
        statement = str(claim.get("statement") or "").strip()
        if relation == "none" or not statement:
            k.mark_explored(a, c, "none")
            print(json.dumps({"status": "no_relation"}, ensure_ascii=False))
            return 0
        if len(statement) > 200:
            k.mark_explored(a, c, "too_long")
            print(json.dumps({"status": "rejected", "reason": "too_long"},
                             ensure_ascii=False))
            return 0
        cid = k.add_claim(a, relation, c, statement,
                          confidence=float(claim.get("confidence", 0.5)),
                          derived_from=[c1, c2])
        print(json.dumps({"status": "claim", "claim_id": cid}, ensure_ascii=False))
        return 0
    finally:
        k.close()


if __name__ == "__main__":
    sys.exit(main())
