#!/usr/bin/env python3
"""v8 script 节点：把 N 条预设的 JSON 结果汇总成命中清单。输入在 DAG8_INPUTS。"""
from __future__ import annotations

import json
import os
import sys


def main() -> int:
    data = json.loads(os.environ["DAG8_INPUTS"])
    rows = []
    for key, val in data.items():
        if not key.startswith("check_"):
            continue
        preset_id = key[len("check_"):]
        try:
            parsed = json.loads(val) if isinstance(val, str) else val
        except ValueError:
            parsed = {"satisfied": False, "score": 0,
                      "reason": f"解析失败：{str(val)[:80]}"}
        parsed = {k: parsed.get(k) for k in ("satisfied", "score", "reason")}
        rows.append({"preset": preset_id, **parsed})
    rows.sort(key=lambda r: -(r.get("score") or 0))
    hits = [r for r in rows if r.get("satisfied")]
    print(json.dumps({
        "checked": len(rows),
        "satisfied": len(hits),
        "hits": hits,
        "all": rows,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
