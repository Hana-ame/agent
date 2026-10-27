#!/usr/bin/env python3
"""读取待评判的结论文档。"""
from __future__ import annotations
import json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

def main() -> int:
    data = json.loads(os.environ.get("DAG8_INPUTS") or "{}")
    raw = str(data.get("doc_path") or "").strip()
    if not raw:
        raise SystemExit("缺少 doc_path")
    p = Path(raw)
    if not p.is_absolute():
        p = ROOT / p
    p = p.resolve()
    if not p.exists():
        raise SystemExit(f"文档不存在: {p}")
    text = p.read_text(encoding="utf-8", errors="replace")
    print(json.dumps({"path": str(p), "chars": len(text), "text": text}, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
