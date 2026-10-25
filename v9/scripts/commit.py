#!/usr/bin/env python3
"""v8 script 节点：校验并写入 v9 知识库。输入来自 DAG8_INPUTS。"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from v9.infer import normalize_source  # noqa: E402
from v9.knowledge import Knowledge  # noqa: E402


def main() -> int:
    data = json.loads(os.environ["DAG8_INPUTS"])
    claim = json.loads(data["claim_json"])
    a, b = data["a"], data["b"]
    url_a, url_b = data["url_a"], data["url_b"]
    k = Knowledge(os.environ.get("V9_DB", str(ROOT / "v9" / "data" / "knowledge.sqlite3")))
    try:
        for url, text in ((url_a, data.get("ev_a", "")), (url_b, data.get("ev_b", ""))):
            if not k.get_source(url):
                k.put_source(url, text)
        relation = str(claim.get("relation") or "none").strip()
        statement = str(claim.get("statement") or "").strip()
        if relation == "none" or not statement:
            k.mark_explored(a, b, "none")
            print(json.dumps({"status": "no_relation"}, ensure_ascii=False))
            return 0
        src = normalize_source(str(claim.get("evidence_url") or ""), (url_a, url_b))
        if src is None:
            k.mark_explored(a, b, "bad_source")
            print(json.dumps({"status": "rejected", "reason": "bad_source"},
                             ensure_ascii=False))
            return 0
        if len(statement) > 200:
            k.mark_explored(a, b, "too_long")
            print(json.dumps({"status": "rejected", "reason": "too_long"},
                             ensure_ascii=False))
            return 0
        cid = k.add_claim(a, relation, b, statement,
                          confidence=float(claim.get("confidence", 0.5)),
                          source_url=src)
        print(json.dumps({"status": "claim", "claim_id": cid}, ensure_ascii=False))
        return 0
    finally:
        k.close()


if __name__ == "__main__":
    sys.exit(main())
